"""Command line interface for oak-spacy-demo."""
import logging
from pathlib import Path

import click
import pandas as pd

from oak_spacy_demo import __version__
from oak_spacy_demo.oak import annotate_via_oak, annotate_via_spacy

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """
    CLI for oak-spacy-demo.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)


@main.command()
@click.option("--tool", default="oak", type=click.Choice(["oak", "spacy"]))
@click.option("-i", "--input-file", type=click.Path(exists=True), required=False)
@click.option("-d", "--dataframe", type=pd.DataFrame, required=False)
@click.option("-c", "--column", type=str)
@click.option("-o", "--output", type=str)
@click.option("-r", "--resource", type=str)
def annotate(tool: str, input_file: str, dataframe:pd.DataFrame, column:str, resource:str, output: str):
    if input_file:
        if Path(input_file).suffix in [".tsv", ".csv"]:
            df = pd.read_csv(input_file, sep="\t")
        else:
            raise ValueError("Input file should be a .tsv or .csv file.")
    elif dataframe:
        df = dataframe
    else:
        raise ValueError("Either input file or dataframe should be provided.")

    if output:
        output = Path(output)
    if tool == "oak":
        annotate_via_oak(dataframe=df, column=column, resource=resource, outfile=output)
    elif tool == "spacy":
        annotate_via_spacy(dataframe=df, column=column, resource=resource, outfile=output)
    else:
        raise ValueError("Tool should be either 'oak' or 'spacy'.")

if __name__ == "__main__":
    main()
