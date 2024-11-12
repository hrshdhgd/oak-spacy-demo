"""Test annotation function."""
import unittest
from pathlib import Path

import pandas as pd

from oak_spacy_demo.main import annotate

OUTPUT_DIR = Path(__file__).parents[0] / "output"
DATA_DIR = Path(__file__).parents[1] / "data"
class TestDemo(unittest.TestCase):

    """Test package."""

    def setUp(self) -> None:
        """Set up."""
        self.matched_output = OUTPUT_DIR/"test.tsv"
        self.unmatched_output = OUTPUT_DIR/"test_unmatched.tsv"
        self.resource = DATA_DIR/"mondo_subset.owl"

    def test_oak_annotate(self):
        """Test oak annotate."""
        df = pd.DataFrame({"text": ["schizophrenia 3", "This is another test sentence."]})
        annotate(dataframe=df, column="text", resource=self.resource, outfile=self.matched_output)
        self.assertTrue(self.matched_output.exists())
        self.assertTrue(self.unmatched_output.exists())

    def tearDown(self) -> None:
        """Clean up."""
        if self.matched_output.exists():
            self.matched_output.unlink()
        if self.unmatched_output.exists():
            self.unmatched_output.unlink()
