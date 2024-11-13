"""Main python file with clean multiprocessing-based text annotation functionality."""

import csv
import itertools
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from oaklib import get_adapter
from oaklib.datamodels.text_annotator import TextAnnotation, TextAnnotationConfiguration

from oak_spacy_demo.constants import _get_uri_converter, annotated_columns


@dataclass
class AnnotationResult:
    """Container for annotation results to improve code readability."""

    matched: Dict[str, List[TextAnnotation]]
    unmatched: Dict[str, List[TextAnnotation]]


class AnnotationWorker:
    """Worker class for handling annotations in each process."""

    def __init__(self, resource: str):
        """Initialize worker with its own adapter instance."""
        self.adapter = get_adapter(f"sqlite:{resource}")
        self.config = TextAnnotationConfiguration(
            include_aliases=True,
            matches_whole_text=True,
        )

    @staticmethod
    def _overlap(a: str, b: str) -> int:
        """Get number of characters in 2 strings that overlap using set intersection."""
        return len(set(a) & set(b))

    def process_batch(self, terms: List[str]) -> Dict[str, List[TextAnnotation]]:
        """Process a batch of terms using the worker's adapter."""
        return {term: list(self.adapter.annotate_text(term.replace("_", " "), self.config)) for term in terms}

    def process_unmatched_batch(self, terms: List[str]) -> Dict[str, List[TextAnnotation]]:
        """Process a batch of unmatched terms using relaxed matching."""
        self.config.matches_whole_text = False
        results = {}

        for term in terms:
            annotations = [
                x for x in self.adapter.annotate_text(term.replace("_", " "), self.config) if len(x.object_label) > 2
            ]
            if annotations:
                max_overlap_annotation = max(annotations, key=lambda obj: self._overlap(obj.object_label, term))
                max_overlap_annotation.subject_label = (
                    term if not max_overlap_annotation.subject_label else max_overlap_annotation.subject_label
                )
                results[term] = [max_overlap_annotation]
        return results


class MultiprocessingAnnotator:
    """Main class for handling multiprocessing annotation logic."""

    def __init__(self, resource: str, n_processes: int = None, batch_size: int = 100):
        """Initialize the annotator with given parameters."""
        self.resource = resource
        self.n_processes = n_processes or mp.cpu_count()
        self.batch_size = batch_size

    @staticmethod
    def _chunk_data(data: List, chunk_size: int) -> List[List]:
        """Split data into chunks of specified size."""
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _initialize_pool(self) -> mp.Pool:
        """Initialize a process pool with the worker class."""
        return mp.Pool(processes=self.n_processes, initializer=self._init_worker, initargs=(self.resource,))

    @staticmethod
    def _init_worker(resource: str):
        """Initialize worker process with AnnotationWorker instance."""
        global worker
        worker = AnnotationWorker(resource)

    @staticmethod
    def _process_batch_wrapper(terms: List[str]) -> Dict[str, List[TextAnnotation]]:
        """Process matched batches."""
        global worker
        return worker.process_batch(terms)

    @staticmethod
    def _process_unmatched_batch_wrapper(terms: List[str]) -> Dict[str, List[TextAnnotation]]:
        """Process unmatched batches."""
        global worker
        return worker.process_unmatched_batch(terms)

    def annotate_terms(self, terms: pd.Series) -> Dict[str, List[TextAnnotation]]:
        """Batch annotate terms using multiprocessing."""
        unique_terms = list(terms.unique())
        batches = self._chunk_data(unique_terms, self.batch_size)

        with self._initialize_pool() as pool:
            results = pool.map(self._process_batch_wrapper, batches)

        return dict(itertools.chain.from_iterable(d.items() for d in results))

    def handle_unmatched_terms(self, unmatched_terms: Dict[str, List]) -> Dict[str, List[TextAnnotation]]:
        """Process unmatched terms with relaxed matching using multiprocessing."""
        terms_list = list(unmatched_terms.keys())
        batches = self._chunk_data(terms_list, self.batch_size)

        with self._initialize_pool() as pool:
            results = pool.map(self._process_unmatched_batch_wrapper, batches)

        return dict(itertools.chain.from_iterable(d.items() for d in results))


def _write_annotations_efficient(
    annotations: Dict[str, List[TextAnnotation]], columns: List[str], output_file: Path, converter
) -> None:
    """Write annotations to TSV file efficiently using pandas."""
    rows = []
    for responses in annotations.values():
        for response in responses:
            row = [getattr(response, col, None) for col in columns]
            row[1] = row[1] if row[1] else converter.expand(row[0])
            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.drop_duplicates().to_csv(output_file, index=False, sep="\t", quoting=csv.QUOTE_NONE)


def annotate_via_oak(
    dataframe: pd.DataFrame, column: str, resource: str, outfile: Path, n_processes: int = None, batch_size: int = 100
) -> None:
    """
    Annotate dataframe column text using oaklib with multiprocessing.

    :param dataframe: Input DataFrame
    :param column: Column to be annotated
    :param resource: Ontology resource file path
    :param outfile: Output file path
    :param n_processes: Number of processes to use (defaults to CPU count)
    :param batch_size: Number of terms to process in each batch
    """
    # Setup resource path
    resource_path = Path(resource)
    db_path = resource.replace(resource_path.suffix, ".db")
    resource = db_path if Path(db_path).exists() else resource

    # Initialize the multiprocessing annotator
    annotator = MultiprocessingAnnotator(resource=resource, n_processes=n_processes, batch_size=batch_size)

    # First pass annotation with exact matching
    annotated_terms = annotator.annotate_terms(dataframe[column])

    # Separate exact matches and unmatched terms
    exact_matches = {k: v for k, v in annotated_terms.items() if v}
    unmatched_terms = {k: v for k, v in annotated_terms.items() if not v}

    # Process unmatched terms
    partial_matches = {}
    if unmatched_terms:
        partial_matches = annotator.handle_unmatched_terms(unmatched_terms)

    # Write results efficiently
    converter = _get_uri_converter()
    unmatched_outfile = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")

    _write_annotations_efficient(exact_matches, annotated_columns, outfile, converter)
    _write_annotations_efficient(partial_matches, annotated_columns, unmatched_outfile, converter)


if __name__ == "__main__":
    pass
