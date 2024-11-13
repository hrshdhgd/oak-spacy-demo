"""(Sci)Spacy for annotation."""

import csv
import json
from pathlib import Path

import pandas as pd
import spacy
from oaklib import get_adapter

from .constants import annotated_columns

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

SCI_SPACY_LINKERS = ["umls", "mesh", "go", "hpo", "rxnorm"]
ONTOLOGY_CACHE_FILENAME = Path("ontology_cache.json")


def annotate_via_spacy(
    dataframe: pd.DataFrame,
    column: str,
    resource: str,
    outfile: Path,
    cache_dir: Path,
):
    """
    Annotate dataframe column text using spacy.

    :param dataframe: Input DataFrame
    :param column: Column to be annotated.
    :param resource: ontology resource file path.
    :param outfile: Output file path.
    """
    if cache_dir:
        ontology_cache_file = cache_dir / ONTOLOGY_CACHE_FILENAME
    else:
        ontology_cache_file = ONTOLOGY_CACHE_FILENAME
    outfile_for_unmatched = outfile.with_name(outfile.stem + "_unmatched" + outfile.suffix)
    resource_path = Path(resource)
    resource_path_db = resource.replace(resource_path.suffix, ".db")
    if Path(resource_path_db).exists():
        resource = resource_path_db
    nlp = spacy.load(MODELS.get("sci_sm"))
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    oi = get_adapter(f"sqlite:{resource}")
    if ontology_cache_file.exists():
        with open(ontology_cache_file, "r") as f:
            ontology = json.load(f)
    else:
        ontology = {oi.label(curie): curie for curie in oi.entities() if oi.label(curie) is not None}
        alaises = {
            curie: oi.entity_aliases(curie)
            for curie in oi.entities()
            if oi.label(curie) is not None and oi.entity_aliases(curie) is not None
        }
        aliases_flipped = {term: mondo_id for mondo_id, terms in alaises.items() for term in terms}
        ontology.update(aliases_flipped)
        with open(ontology_cache_file, "w") as f:
            json.dump(ontology, f, indent=4)

    patterns = [{"label": curie, "pattern": label} for label, curie in ontology.items()]
    ruler.add_patterns(patterns)

    column_data = dataframe[column].tolist()

    with (
        open(str(outfile), "w", newline="") as file_1,
        open(str(outfile_for_unmatched), "w", newline="") as file_2,
    ):
        writer_1 = csv.writer(file_1, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_2 = csv.writer(file_2, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer_1.writerow(annotated_columns)
        writer_2.writerow(annotated_columns)

        for txt in column_data:
            doc = nlp(txt)
            matched = False  # Track if any entity matches the text

            for ent in doc.ents:
                row_data = [
                    ent.label_,
                    ent.text,
                    txt,
                    ent.text == txt,
                    ent.start_char,
                    ent.end_char,
                ]
                if ent.text == txt:
                    writer_1.writerow(row_data)
                    matched = True
                else:
                    writer_2.writerow(row_data)

            # If no entity matched or no entities were found, write to writer_2
            if not doc.ents or not matched:
                writer_2.writerow([None, None, txt, False, None, None])

    pd.read_csv(outfile, sep="\t").drop_duplicates().to_csv(outfile, index=False, sep="\t")
    pd.read_csv(outfile_for_unmatched, sep="\t").drop_duplicates().to_csv(outfile_for_unmatched, index=False, sep="\t")
