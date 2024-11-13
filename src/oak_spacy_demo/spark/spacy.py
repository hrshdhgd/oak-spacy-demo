"""PySpark wrapper for spaCy NLP pipeline."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import spacy
from oaklib import get_adapter
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, StringType, StructField, StructType


@dataclass
class AnnotationConfig:
    """Configuration for annotation process."""

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
    ONTOLOGY_CACHE_FILENAME = "ontology_cache.json"

class OntologyCache:
    """Handle ontology caching operations."""

    def __init__(self, cache_path: Path):
        """Initialize cache path."""
        self.cache_path = cache_path

    def load(self) -> Dict[str, str]:
        """Load ontology dictionary from cache if exists."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def save(self, ontology: Dict[str, str]) -> None:
        """Save ontology dictionary to cache."""
        with open(self.cache_path, "w") as f:
            json.dump(ontology, f, indent=4)

def build_ontology(oi) -> Dict[str, str]:
    """Build ontology dictionary efficiently."""
    ontology = {
        oi.label(curie): curie
        for curie in oi.entities()
        if oi.label(curie) is not None
    }

    aliases = {
        term: mondo_id
        for mondo_id in ontology.values()
        for term in (oi.entity_aliases(mondo_id) or [])
    }

    return {**ontology, **aliases}

def setup_nlp_pipeline(model_name: str, patterns: List[Dict]) -> spacy.language.Language:
    """Set up spaCy pipeline with entity ruler."""
    nlp = spacy.load(AnnotationConfig.MODELS.get(model_name, "sci_sm"))
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(patterns)
    return nlp

def process_text(nlp, text: str) -> List[Dict]:
    """Process single text and return list of annotations."""
    doc = nlp(text)
    results = []

    for ent in doc.ents:
        result = {
            "label": ent.label_,
            "text": ent.text,
            "source_text": text,
            "exact_match": ent.text == text,
            "start": ent.start_char,
            "end": ent.end_char
        }
        results.append(result)

    if not results:
        results.append({
            "label": None,
            "text": None,
            "source_text": text,
            "exact_match": False,
            "start": None,
            "end": None
        })

    return results

def annotate_via_spacy_spark(
    spark_df: DataFrame,
    column: str,
    resource: str,
    outfile: Path,
    cache_dir: Optional[Path] = None,
    model: str = "sci_sm",
    num_partitions: int = None
) -> None:
    """
    Annotate dataframe column text using PySpark for distributed processing.

    :param spark_df: Input PySpark DataFrame
    :param column: Column to be annotated
    :param resource: Ontology resource file path
    :param outfile: Output file path
    :param cache_dir: Directory for cache files
    :param model: SciSpacy model to use
    :param num_partitions: Number of partitions for parallel processing

    """
    # Setup paths and cache
    cache_dir = cache_dir or Path.cwd()
    cache_file = cache_dir / AnnotationConfig.ONTOLOGY_CACHE_FILENAME
    outfile_unmatched = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")

    # Setup resource path
    resource_path = Path(resource)
    resource = str(resource_path).replace(resource_path.suffix, ".db") \
               if Path(str(resource_path).replace(resource_path.suffix, ".db")).exists() \
               else str(resource_path)

    # Initialize ontology
    ontology_cache = OntologyCache(cache_file)
    ontology = ontology_cache.load()

    if not ontology:
        oi = get_adapter(f"sqlite:{resource}")
        ontology = build_ontology(oi)
        ontology_cache.save(ontology)

    # Setup spaCy pipeline
    patterns = [{"label": curie, "pattern": label} for label, curie in ontology.items()]
    nlp = setup_nlp_pipeline(model, patterns)

    # Define schema for the annotation results
    annotation_schema = ArrayType(StructType([
        StructField("label", StringType(), True),
        StructField("text", StringType(), True),
        StructField("source_text", StringType(), True),
        StructField("exact_match", BooleanType(), True),
        StructField("start", IntegerType(), True),
        StructField("end", IntegerType(), True)
    ]))

    # Create UDF for processing
    process_text_udf = udf(lambda text: process_text(nlp, text), annotation_schema)

    # Process the data
    if num_partitions:
        spark_df = spark_df.repartition(num_partitions)

    # Apply the UDF and explode the results
    results_df = spark_df.select(
        explode(process_text_udf(col(column))).alias("annotation")
    ).select(
        col("annotation.label"),
        col("annotation.text"),
        col("annotation.source_text"),
        col("annotation.exact_match"),
        col("annotation.start"),
        col("annotation.end")
    )

    # Split and save results
    exact_matches = results_df.filter(col("exact_match") == True)
    partial_matches = results_df.filter(col("exact_match") == False)

    # Save results with deduplication
    exact_matches.dropDuplicates().write.csv(
        str(outfile),
        sep="\t",
        header=True,
        mode="overwrite"
    )

    partial_matches.dropDuplicates().write.csv(
        str(outfile_unmatched),
        sep="\t",
        header=True,
        mode="overwrite"
    )

if __name__ == "__main__":
    # Example usage
    spark = SparkSession.builder \
        .appName("TextAnnotation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
