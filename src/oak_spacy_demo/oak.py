"""Main python file with optimized text annotation functionality."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from oaklib import get_adapter
from oaklib.datamodels.text_annotator import TextAnnotation, TextAnnotationConfiguration

from .constants import _get_uri_converter, annotated_columns


@dataclass
class AnnotationResult:
    """Container for annotation results to improve code readability."""

    matched: Dict[str, List[TextAnnotation]]
    unmatched: Dict[str, List[TextAnnotation]]


def _overlap(a: str, b: str) -> int:
    """Get number of characters in 2 strings that overlap using set intersection."""
    return len(set(a) & set(b))


def _annotate_terms(
    terms: pd.Series, adapter: object, config: TextAnnotationConfiguration
) -> Dict[str, List[TextAnnotation]]:
    """Batch annotate terms using provided configuration."""
    return {term: list(adapter.annotate_text(term.replace("_", " "), config)) for term in terms.unique()}


def _handle_unmatched_terms(
    unmatched_terms: Dict[str, List], adapter: object, config: TextAnnotationConfiguration
) -> Dict[str, List[TextAnnotation]]:
    """Process unmatched terms with relaxed matching criteria."""
    results = {}
    config.matches_whole_text = False

    for term in unmatched_terms:
        annotations = [x for x in adapter.annotate_text(term.replace("_", " "), config) if len(x.object_label) > 2]
        if annotations:
            # Find annotation with maximum overlap
            max_overlap_annotation = max(annotations, key=lambda obj: _overlap(obj.object_label, term))
            max_overlap_annotation.subject_label = (
                term if not max_overlap_annotation.subject_label else max_overlap_annotation.subject_label
            )
            results[term] = [max_overlap_annotation]
    return results


def _write_annotations(annotations: Dict[str, List[TextAnnotation]], columns: List[str], output_file: Path) -> None:
    """Write annotations to TSV file and remove duplicates."""
    converter = _get_uri_converter()
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer.writerow(columns)
        for responses in annotations.values():
            for response in responses:
                row = [getattr(response, col, None) for col in columns]
                # replace None with empty string in row[1]
                row[1] = row[1] if row[1] else converter.expand(row[0])
                writer.writerow(row)

    # Remove duplicates efficiently using pandas
    df = pd.read_csv(output_file, sep="\t")
    df.drop_duplicates().to_csv(output_file, index=False, sep="\t")


def annotate_via_oak(
    dataframe: pd.DataFrame,
    column: str,
    resource: str,
    outfile: Path,
) -> None:
    """
    Annotate dataframe column text using oaklib + llm with improved efficiency.

    Each row in the column is treated as a complete term.

    :param dataframe: Input DataFrame
    :param column: Column to be annotated
    :param resource: Ontology resource file path
    :param outfile: Output file path

    """
    # Setup resource path
    resource_path = Path(resource)
    db_path = resource.replace(resource_path.suffix, ".db")
    resource = db_path if Path(db_path).exists() else resource

    # Initialize adapter and configuration
    adapter = get_adapter(f"sqlite:{resource}")
    config = TextAnnotationConfiguration(
        include_aliases=True,
        matches_whole_text=True,
    )

    # First pass annotation with exact matching
    annotated_terms = _annotate_terms(dataframe[column], adapter, config)

    # Separate exact matches and unmatched terms
    exact_matches = {k: v for k, v in annotated_terms.items() if v}
    unmatched_terms = {k: v for k, v in annotated_terms.items() if not v}

    # Process unmatched terms separately
    partial_matches = {}
    if unmatched_terms:
        partial_matches = _handle_unmatched_terms(unmatched_terms, adapter, config)

    # Write results to separate files
    unmatched_outfile = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")
    _write_annotations(exact_matches, annotated_columns, outfile)
    _write_annotations(partial_matches, annotated_columns, unmatched_outfile)


if __name__ == "__main__":
    pass
