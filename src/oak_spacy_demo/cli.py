"""Command line interface for oak-spacy-demo."""

import logging
import os
from pathlib import Path

import click
import pandas as pd

from oak_spacy_demo import __version__
from oak_spacy_demo.constants import SCI_SPACY_LINKERS
from oak_spacy_demo.oak import annotate_via_oak
from oak_spacy_demo.spacy import annotate_via_spacy

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
@click.option("--cache-dir", type=click.Path(exists=True), required=False)
@click.option("-l", "--linker", type=click.Choice(SCI_SPACY_LINKERS), default="umls", required=False)
@click.option("-b", "--batch-size", type=int, default=1000)
def annotate(
    tool: str,
    input_file: str,
    dataframe: pd.DataFrame,
    column: str,
    resource: str,
    cache_dir: str,
    output: str,
    linker: str,
    batch_size: int,
):
    if input_file:
        if Path(input_file).suffix in [".tsv", ".csv"]:
            separator = "\t" if Path(input_file).suffix == ".tsv" else ","
            df = pd.read_csv(input_file, sep=separator)
        else:
            raise ValueError("Input file should be a .tsv or .csv file.")
    elif dataframe:
        df = dataframe
    else:
        raise ValueError("Either input file or dataframe should be provided.")

    if output:
        output = Path(output)
    else:
        output = Path(f"{column}.tsv")

    n_processes = max(1, os.cpu_count() - 1)
    if tool == "oak":
        annotate_via_oak(dataframe=df, column=column, resource=resource, outfile=output, n_processes=n_processes)
    elif tool == "spacy":
        annotate_via_spacy(
            dataframe=df,
            column=column,
            resource=resource,
            outfile=output,
            cache_dir=Path(cache_dir),
            linker=linker,
            n_processes=n_processes,
            batch_size=batch_size,
        )
    else:
        raise ValueError("Tool should be either 'oak' or 'spacy'.")


if __name__ == "__main__":
    main()
