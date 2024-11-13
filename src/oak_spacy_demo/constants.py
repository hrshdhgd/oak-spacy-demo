"""All constants used in the project are defined here."""

import curies

OBJECT_ID_COLUMN = "object_id"
OBJECT_LABEL_COLUMN = "object_label"
SUBJECT_LABEL_COLUMN = "subject_label"
MATCHES_WHOLE_TEXT_COLUMN = "matches_whole_text"
START_COLUMN = "start"
END_COLUMN = "end"
CUSTOM_TERM_COLUMN = "custom_term"

annotated_columns = [
    OBJECT_ID_COLUMN,
    OBJECT_LABEL_COLUMN,
    SUBJECT_LABEL_COLUMN,
    MATCHES_WHOLE_TEXT_COLUMN,
    START_COLUMN,
    END_COLUMN,
]

SCI_SPACY_LINKERS = ["umls", "mesh", "go", "hpo", "rxnorm"]


def _get_uri_converter():
    """Get URI converter."""
    return curies.load_prefix_map(
        {
            "MONDO": "http://purl.obolibrary.org/obo/MONDO_",
        }
    )
