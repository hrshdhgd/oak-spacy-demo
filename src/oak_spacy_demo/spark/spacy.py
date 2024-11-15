"""Annotate text in a Spark DataFrame using spaCy."""

import json
from pathlib import Path

import spacy
from oaklib import get_adapter
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, StringType, StructField, StructType


def get_ontology_cache_filename(resource: str) -> str:
    """Get the ontology cache filename based on the resource file."""
    resource_path = Path(resource)
    return resource_path.stem + "_cache.json"


class OntologyCache:
    """Cache class for ontology dictionaries."""

    def __init__(self, cache_path: Path):
        """Initialize the OntologyCache class with a cache file path."""
        self.cache_path = cache_path

    def load(self) -> dict:
        """Load the ontology cache from the cache file."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def save(self, ontology: dict) -> None:
        """Save the ontology cache to the cache file."""
        with open(self.cache_path, "w") as f:
            json.dump(ontology, f, indent=4)


def build_ontology(oi) -> dict:
    """Build an ontology dictionary via an OAK adapter."""
    ontology = {oi.label(curie): curie for curie in oi.entities() if oi.label(curie) is not None}
    aliases = {term: mondo_id for mondo_id in ontology.values() for term in (oi.entity_aliases(mondo_id) or [])}
    return {**ontology, **aliases}


def setup_nlp_pipeline(model_name: str, patterns: list) -> spacy.language.Language:
    """Instantiate the spaCy pipeline with an entity ruler."""
    nlp = spacy.load(model_name)
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    return nlp


def process_text(nlp, text: str) -> list:
    """Process text using a spaCy pipeline."""
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        result = {
            "label": ent.label_,
            "text": ent.text,
            "source_text": text,
            "exact_match": ent.text == text,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        results.append(result)
    if not results:
        results.append(
            {"label": None, "text": None, "source_text": text, "exact_match": False, "start": None, "end": None}
        )
    return results


def annotate_via_spacy_spark(
    spark_df: DataFrame,
    column: str,
    resource: str,
    outfile: Path,
    cache_dir: Path = None,
    model: str = "en_core_web_sm",
    num_partitions: int = None,
):
    """Annotate text in a Spark DataFrame using spaCy."""
    cache_dir = cache_dir or Path.cwd()
    cache_file = cache_dir / get_ontology_cache_filename(resource)
    outfile_unmatched = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")

    resource_path = Path(resource)
    resource = (
        str(resource_path).replace(resource_path.suffix, ".db")
        if Path(str(resource_path).replace(resource_path.suffix, ".db")).exists()
        else str(resource_path)
    )

    ontology_cache = OntologyCache(cache_file)
    ontology = ontology_cache.load()

    if not ontology:
        oi = get_adapter(f"sqlite:{resource}")
        ontology = build_ontology(oi)
        ontology_cache.save(ontology)

    patterns = [{"label": curie, "pattern": label} for label, curie in ontology.items()]
    nlp = setup_nlp_pipeline(model, patterns)

    annotation_schema = ArrayType(
        StructType(
            [
                StructField("label", StringType(), True),
                StructField("text", StringType(), True),
                StructField("source_text", StringType(), True),
                StructField("exact_match", BooleanType(), True),
                StructField("start", IntegerType(), True),
                StructField("end", IntegerType(), True),
            ]
        )
    )

    process_text_udf = udf(lambda text: process_text(nlp, text), annotation_schema)

    if num_partitions:
        spark_df = spark_df.repartition(num_partitions)

    results_df = spark_df.select(explode(process_text_udf(col(column))).alias("annotation")).select(
        col("annotation.label"),
        col("annotation.text"),
        col("annotation.source_text"),
        col("annotation.exact_match"),
        col("annotation.start"),
        col("annotation.end"),
    )

    exact_matches = results_df.filter(col("exact_match") == True)
    partial_matches = results_df.filter(col("exact_match") == False)

    exact_matches.dropDuplicates().write.csv(str(outfile), sep="\t", header=True, mode="overwrite")
    partial_matches.dropDuplicates().write.csv(str(outfile_unmatched), sep="\t", header=True, mode="overwrite")


if __name__ == "__main__":
    # spark = (
    #     SparkSession.builder.appName("TextAnnotation")
    #     .config("spark.driver.memory", "4g")
    #     .config("spark.executor.memory", "4g")
    #     .getOrCreate()
    # )
    pass
