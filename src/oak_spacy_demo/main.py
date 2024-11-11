"""Main python file."""


import csv
from pathlib import Path

import pandas as pd
from oaklib import get_adapter
from oaklib.datamodels.text_annotator import TextAnnotationConfiguration

MONDO_PREFIX = "MONDO"
CHEBI_PREFIX = "CHEBI"

OBJECT_ID_COLUMN = "object_id"
OBJECT_LABEL_COLUMN = "object_label"
SUBJECT_LABEL_COLUMN = "subject_label"
MATCHES_WHOLE_TEXT_COLUMN = "matches_whole_text"
START_COLUMN = "start"
END_COLUMN = "end"
ACTION_COLUMN = "action"
SUPPLEMENT = "supplement"
REPLACEMENT = "replacement"
CUSTOM_TERM_COLUMN = "custom_term"


DATA_DIR = Path(__file__).parents[2] / "data"

MONDO_SOURCE = DATA_DIR / "mondo.owl"
CHEBI_SOURCE = "data"

PREFIX_SOURCE_MAP = {
    MONDO_PREFIX: MONDO_SOURCE,
    CHEBI_PREFIX: CHEBI_SOURCE,
}

def _overlap(a, b):
    """Get number of characters in 2 strings that overlap."""
    return len(set(a) & set(b))

def annotate(
    dataframe:pd.DataFrame,
    column: str,
    prefix: str,
    # exclusion_list: List,
    outfile: Path,
    ):
    """
    Annotate dataframe column text using oaklib + llm.

    :param dataframe: Input DataFrame
    :param prefix: Ontology to be used.
    :param exclusion_list: Tokens that can be ignored.
    :param outfile: Output file path.
    """
    ontology = prefix
    outfile_for_unmatched = outfile.with_name(outfile.stem + "_unmatched" + outfile.suffix)
    oi = get_adapter(f"sqlite:{PREFIX_SOURCE_MAP[ontology]}")
    matches_whole_text = True
    annotated_columns = [
        OBJECT_ID_COLUMN,
        OBJECT_LABEL_COLUMN,
        SUBJECT_LABEL_COLUMN,
        MATCHES_WHOLE_TEXT_COLUMN,
        START_COLUMN,
        END_COLUMN,
        CUSTOM_TERM_COLUMN,
    ]

    configuration = TextAnnotationConfiguration(
        include_aliases=True,
        matches_whole_text=matches_whole_text,
    )

    unique_terms_set = {
        item.strip()
        for sublist in dataframe[column].drop_duplicates().to_list()
        for item in sublist.split(", ")
    }

    unique_terms_annotated = {
        term: list(oi.annotate_text(term.replace("_", " "), configuration))
        for term in unique_terms_set
    }
    terms_not_annotated = {k: v for k, v in unique_terms_annotated.items() if v == []}
    # The annotations upto this point is matches_whole_text = True.
    # There are still some terms that aren't annotated.
    # For those we flip matches_whole_text = False and then rerun.
    if len(terms_not_annotated) > 0:
        configuration.matches_whole_text = False
        unique_terms_not_annotated_set = set(terms_not_annotated.keys())
        unique_terms_annotated_not_whole_match = {
            term: [
                x
                for x in oi.annotate_text(term.replace("_", " "), configuration)
                if len(x.object_label) > 2
            ]
            for term in unique_terms_not_annotated_set
        }

        # Initialize an empty dictionary
        max_overlap_dict = {}

        # Iterate over items in the original dictionary
        for k, v in unique_terms_annotated_not_whole_match.items():
            # Find the max value using the overlap function and assign it to the new dictionary
            if v != []:
                max_overlap_dict[k] = [max(v, key=lambda obj: _overlap(obj.object_label, k))]
        # Now new_dict is equivalent to unique_terms_annotated_not_whole_match in the original code
        unique_terms_annotated_not_whole_match = max_overlap_dict


    with (
        open(str(outfile), "w", newline="") as file_1,
        open(str(outfile_for_unmatched), "w", newline="") as file_2,
    ):
        writer_1 = csv.writer(file_1, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_2 = csv.writer(file_2, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_1.writerow(annotated_columns)
        writer_2.writerow(annotated_columns)

        for row in dataframe.iterrows():
            row = row[1]
            term = row[column]
            responses = unique_terms_annotated.get(term, None)
            if responses:
                writer = writer_1
            else:
                responses = unique_terms_annotated_not_whole_match.get(term, None)
                writer = writer_2

            if responses:
                for response in responses:
                    response_dict = response.__dict__
                    # response_dict[TAX_ID_COLUMN] = row[1].iloc[0]
                    response_dict[CUSTOM_TERM_COLUMN] = term

                    # Ensure the order of columns matches the header
                    row_to_write = [response_dict.get(col) for col in annotated_columns]
                    writer.writerow(row_to_write)




if __name__ == "__main__":
    pass
