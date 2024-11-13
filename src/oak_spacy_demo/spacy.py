"""Optimized (Sci)Spacy implementation for annotation."""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from curies.api import NoCURIEDelimiterError
from oaklib import get_adapter
from scispacy.abbreviation import AbbreviationDetector  # noqa
from scispacy.linking import EntityLinker  # noqa
from spacy.language import Language
from spacy.tokens import Doc

from .constants import _get_uri_converter, annotated_columns

logger = logging.getLogger(__name__)


def get_ontology_cache_filename(resource: str) -> str:
    """Get the ontology cache filename based on the resource file."""
    resource_path = Path(resource)
    return resource_path.stem + "_cache.json"


@dataclass
class AnnotationConfig:
    """Configuration for annotation process."""

    MODELS = {
        "sci_sm": "en_core_sci_sm",
        "sci_md": "en_core_sci_md",
        "sci_lg": "en_core_sci_lg",
        "sci_scibert": "en_core_sci_scibert",
        "craft_md": "en_ner_craft_md",
        "jnlpba_md": "en_ner_jnlpba_md",
        "bc5cdr_md": "en_ner_bc5cdr_md",
        "bionlp13cg_md": "en_ner_bionlp13cg_md",
    }


@dataclass
class AnnotationResult:
    """Container for annotation results."""

    label: Optional[str]
    uri: Optional[str]
    text: Optional[str]
    source_text: str
    exact_match: bool
    start: Optional[int]
    end: Optional[int]


class OntologyCache:
    """Handle ontology caching operations."""

    def __init__(self, cache_path: Path):
        """Initialize cache path."""
        self.cache_path = cache_path

    def load(self) -> Dict[str, str]:
        """Load ontology from cache if exists."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def save(self, ontology: Dict[str, str]) -> None:
        """Save ontology to cache."""
        with open(self.cache_path, "w") as f:
            json.dump(ontology, f, indent=4)


def build_ontology(oi) -> Dict[str, str]:
    """Build ontology dictionary efficiently."""
    # Get base ontology
    ontology = {oi.label(curie): curie for curie in oi.entities() if oi.label(curie) is not None}

    # Add aliases efficiently
    aliases = {term: mondo_id for mondo_id in ontology.values() for term in (oi.entity_aliases(mondo_id) or [])}

    return {**ontology, **aliases}


def setup_nlp_pipeline(model_name: str, patterns: List[Dict], linker: str) -> Language:
    """Entity ruler setup for spaCy pipeline."""
    nlp = spacy.load(AnnotationConfig.MODELS.get(model_name, "sci_sm"))
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    nlp.add_pipe("abbreviation_detector")
    #! Linker not needed as of now.
    # nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": linker})
    ruler.add_patterns(patterns)
    return nlp


def process_entities(doc: Doc, source_text: str) -> Tuple[List[AnnotationResult], bool]:
    """Process entities from spaCy doc."""
    results = []
    exact_match_found = False
    converter = _get_uri_converter()

    for ent in doc.ents:
        is_exact = ent.text == source_text
        try:
            uri = converter.expand(ent.label_)
        except NoCURIEDelimiterError as e:
            uri = ent.label_
            logger.warning(f"Error expanding URI for {ent.label_}: {e}")
        result = AnnotationResult(
            label=ent.label_,
            uri=uri,
            text=ent.text,
            source_text=source_text,
            exact_match=is_exact,
            start=ent.start_char,
            end=ent.end_char,
        )
        results.append(result)
        if is_exact:
            exact_match_found = True

    # Handle case with no entities
    if not results:
        results.append(
            AnnotationResult(
                label=None, uri=None, text=None, source_text=source_text, exact_match=False, start=None, end=None
            )
        )

    return results, exact_match_found


def write_results(results: List[AnnotationResult], exact_match: bool, writers: Tuple[csv.writer, csv.writer]) -> None:
    """Write results to appropriate files."""
    writer_exact, writer_partial = writers

    for result in results:
        row = [result.label, result.uri, result.text, result.source_text, result.exact_match, result.start, result.end]

        if result.exact_match:
            writer_exact.writerow(row)
        else:
            writer_partial.writerow(row)


def annotate_via_spacy(
    dataframe: pd.DataFrame,
    column: str,
    resource: str,
    outfile: Path,
    cache_dir: Optional[Path] = None,
    model: str = "sci_sm",
    linker: str = "umls",
    batch_size: int = 1000,
) -> None:
    """
    Annotate dataframe column text using optimized spacy implementation.

    :param dataframe: Input DataFrame
    :param column: Column to be annotated
    :param resource: Ontology resource file path
    :param outfile: Output file path
    :param cache_dir: Directory for cache files
    :param model: SciSpacy model to use
    :param batch_size: Number of texts to process in each batch

    """
    # Setup paths
    cache_dir = cache_dir or Path.cwd()
    cache_file = cache_dir / get_ontology_cache_filename(resource)
    outfile_unmatched = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")

    # Setup resource
    resource_path = Path(resource)
    resource = (
        str(resource_path).replace(resource_path.suffix, ".db")
        if Path(str(resource_path).replace(resource_path.suffix, ".db")).exists()
        else str(resource_path)
    )

    # Initialize ontology
    ontology_cache = OntologyCache(cache_file)
    ontology = ontology_cache.load()

    if not ontology:
        oi = get_adapter(f"sqlite:{resource}")
        ontology = build_ontology(oi)
        ontology_cache.save(ontology)

    # Setup spaCy pipeline
    patterns = [{"label": curie, "pattern": label} for label, curie in ontology.items()]
    nlp = setup_nlp_pipeline(model_name=model, patterns=patterns, linker=linker)

    # Process in batches
    with open(outfile, "w", newline="") as f1, open(outfile_unmatched, "w", newline="") as f2:
        writer_exact = csv.writer(f1, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_partial = csv.writer(f2, delimiter="\t", quoting=csv.QUOTE_NONE)

        # Write headers
        for writer in (writer_exact, writer_partial):
            writer.writerow(annotated_columns)

        # Process in batches
        texts = dataframe[column].unique()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            docs = nlp.pipe(batch_texts)

            for doc, text in zip(docs, batch_texts, strict=False):
                results, exact_match = process_entities(doc, text)
                write_results(results, exact_match, (writer_exact, writer_partial))

    # Remove duplicates efficiently
    for file in (outfile, outfile_unmatched):
        df = pd.read_csv(file, sep="\t")
        df.drop_duplicates().to_csv(file, index=False, sep="\t")


if __name__ == "__main__":
    pass
