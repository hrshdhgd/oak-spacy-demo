"""Test annotation function."""

import unittest
from pathlib import Path

import pandas as pd

from oak_spacy_demo.oak import annotate_via_oak
from oak_spacy_demo.spacy import annotate_via_spacy

OUTPUT_DIR = Path(__file__).parents[0] / "output"
DATA_DIR = Path(__file__).parents[1] / "data"
CACHE_DIR = Path(__file__).parents[0] / "cache"


class TestDemo(unittest.TestCase):
    """Test package."""

    def setUp(self) -> None:
        """Set up."""
        self.oak_matched_output = OUTPUT_DIR / "test_oak.tsv"
        self.oak_unmatched_output = OUTPUT_DIR / "test_oak_unmatched.tsv"
        self.spacy_matched_output = OUTPUT_DIR / "test_spacy.tsv"
        self.spacy_unmatched_output = OUTPUT_DIR / "test_spacy_unmatched.tsv"
        self.resource = str(DATA_DIR / "mondo_subset.owl")
        self.cache_dir = CACHE_DIR

    def test_oak_annotate(self):
        """Test oak annotate."""
        df = pd.DataFrame({"text": ["schizophrenia 3", "This is another test sentence."]})
        annotate_via_oak(dataframe=df, column="text", resource=self.resource, outfile=self.oak_matched_output)
        self.assertTrue(self.oak_matched_output.exists())
        self.assertTrue(self.oak_unmatched_output.exists())

    def test_spacy_annotate(self):
        """Test spacy annotate."""
        df = pd.DataFrame({"text": ["schizophrenia 3", "This is another test sentence."]})
        annotate_via_spacy(
            dataframe=df,
            column="text",
            resource=self.resource,
            outfile=self.oak_matched_output,
            cache_dir=self.cache_dir,
        )
        self.assertTrue(self.oak_matched_output.exists())
        self.assertTrue(self.oak_unmatched_output.exists())

    def tearDown(self) -> None:
        """Clean up."""
        if self.oak_matched_output.exists():
            self.oak_matched_output.unlink()
        if self.oak_unmatched_output.exists():
            self.oak_unmatched_output.unlink()
