"""Optimized (Sci)Spacy implementation for annotation."""

import csv
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from curies.api import NoCURIEDelimiterError
from oaklib import get_adapter
from spacy.language import Language
from spacy.tokens import Doc

from .constants import _get_uri_converter, annotated_columns

logger = logging.getLogger(__name__)


# Existing dataclasses and helper classes remain the same
@dataclass
class AnnotationConfig:
    """Configuration for annotation."""

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
    """Container for annotation results to improve code readability."""

    label: Optional[str]
    uri: Optional[str]
    text: Optional[str]
    source_text: str
    exact_match: bool
    start: Optional[int]
    end: Optional[int]


class OntologyCache:
    """Cache for ontology."""

    def __init__(self, cache_path: Path):
        """Initialize cache with cache path."""
        self.cache_path = cache_path

    def load(self) -> Dict[str, str]:
        """Load ontology from cache."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def save(self, ontology: Dict[str, str]) -> None:
        """Save ontology to cache."""
        with open(self.cache_path, "w") as f:
            json.dump(ontology, f, indent=4)


def setup_nlp_pipeline(model_name: str, patterns: List[Dict], linker: str) -> Language:
    """Entity ruler setup for spaCy pipeline."""
    nlp = spacy.load(AnnotationConfig.MODELS.get(model_name, "bc5cdr_md"))
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    return nlp


def process_batch(texts: List[str], nlp: Language) -> List[Tuple[List[AnnotationResult], bool]]:
    """Process a batch of texts and return results."""
    results = []
    for doc, text in zip(nlp.pipe(texts), texts, strict=False):
        batch_results = process_entities(doc, text)
        results.append(batch_results)
    return results


def process_entities(doc: Doc, source_text: str) -> Tuple[List[AnnotationResult], bool]:
    """Process entities from spaCy doc."""
    results = []
    # exact_match_found = False
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
        # if is_exact:
        #     exact_match_found = True

    if not results:
        results.append(
            AnnotationResult(
                label=None, uri=None, text=None, source_text=source_text, exact_match=False, start=None, end=None
            )
        )

    return results


def write_results_batch(results_batch: List[Tuple[List[AnnotationResult], bool]], queue: mp.Queue) -> None:
    """Write batch results to queue for processing."""
    for results in results_batch:
        for result in results:
            row = [
                result.label,
                result.uri,
                result.text,
                result.source_text,
                result.exact_match,
                result.start,
                result.end,
            ]
            queue.put((result.exact_match, row))


def writer_process(queue: mp.Queue, outfile: Path, outfile_unmatched: Path, done_event: mp.Event) -> None:
    """Process that handles writing results to files."""
    with open(outfile, "w", newline="") as f1, open(outfile_unmatched, "w", newline="") as f2:

        writer_exact = csv.writer(f1, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_partial = csv.writer(f2, delimiter="\t", quoting=csv.QUOTE_NONE)

        # Write headers
        for writer in (writer_exact, writer_partial):
            writer.writerow(annotated_columns)

        while not (done_event.is_set() and queue.empty()):
            try:
                is_exact, row = queue.get(timeout=1)
                if is_exact:
                    writer_exact.writerow(row)
                else:
                    writer_partial.writerow(row)
            except:
                continue


def build_ontology(oi) -> Dict[str, str]:
    """Build ontology dictionary efficiently."""
    ontology = {oi.label(curie): curie for curie in oi.entities() if oi.label(curie) is not None}

    aliases = {term: mondo_id for mondo_id in ontology.values() for term in (oi.entity_aliases(mondo_id) or [])}

    return {**ontology, **aliases}


def get_ontology_cache_filename(resource: str) -> str:
    """Get the ontology cache filename based on the resource file."""
    resource_path = Path(resource)
    return resource_path.stem + "_cache.json"


def annotate_via_spacy(
    dataframe: pd.DataFrame,
    column: str,
    resource: str,
    outfile: Path,
    cache_dir: Optional[Path] = None,
    model: str = "bc5cdr_md",
    linker: str = "umls",
    batch_size: int = 1000,
    n_processes: int = None,
) -> None:
    """
    Multiprocessing-enabled annotation of dataframe column text using spacy.

    Args:
        dataframe: Input DataFrame
        column: Column to be annotated
        resource: Ontology resource file path
        outfile: Output file path
        cache_dir: Directory for cache files
        model: SciSpacy model to use
        linker: SciSpaCy linker to use for ontology expansion
        batch_size: Number of texts to process in each batch
        n_processes: Number of processes to use (defaults to CPU count - 1)

    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

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

    # Setup patterns and create shared nlp pipeline
    patterns = [{"label": curie, "pattern": label} for label, curie in ontology.items()]
    nlp = setup_nlp_pipeline(model_name=model, patterns=patterns, linker=linker)

    # Setup multiprocessing components
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    done_event = mp.Event()

    # Start writer process
    writer_proc = mp.Process(target=writer_process, args=(queue, outfile, outfile_unmatched, done_event))
    writer_proc.start()

    # Process texts in parallel
    texts = dataframe[column].unique()
    text_batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    with mp.Pool(processes=n_processes) as pool:
        process_batch_partial = partial(process_batch, nlp=nlp)

        # Process batches and write results
        for results_batch in pool.imap(process_batch_partial, text_batches):
            write_results_batch(results_batch, queue)

    # Signal completion and wait for writer to finish
    done_event.set()
    writer_proc.join()

    # Remove duplicates efficiently
    for file in (outfile, outfile_unmatched):
        df = pd.read_csv(file, sep="\t")
        df.drop_duplicates().to_csv(file, index=False, sep="\t")


if __name__ == "__main__":
    pass
