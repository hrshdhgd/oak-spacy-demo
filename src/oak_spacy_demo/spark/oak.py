"""Main python file with PySpark-based text annotation functionality."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from oaklib import get_adapter
from oaklib.datamodels.text_annotator import TextAnnotation, TextAnnotationConfiguration
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


@dataclass
class AnnotationResult:
    """Container for annotation results to improve code readability."""

    object_id: str
    object_label: str
    match_type: str
    match_string: str
    start: int
    end: int

def create_spark_session(app_name: str = "OAK Annotator") -> SparkSession:
    """Create and configure Spark session."""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.default.parallelism", "100")
            .getOrCreate())

def _overlap(a: str, b: str) -> int:
    """Get number of characters in 2 strings that overlap using set intersection."""
    return len(set(a) & set(b))

def _convert_annotation_to_dict(annotation: TextAnnotation) -> Dict:
    """Convert TextAnnotation object to dictionary for Spark processing."""
    return {
        "object_id": annotation.object_id,
        "object_label": annotation.object_label,
        "match_type": annotation.match_type,
        "match_string": annotation.match_string,
        "start": annotation.start,
        "end": annotation.end
    }

def _annotate_single_term(
    term: str,
    adapter: object,
    config: TextAnnotationConfiguration,
    exact_match: bool = True
) -> List[Dict]:
    """Annotate a single term and return results as list of dicts."""
    if exact_match:
        annotations = list(adapter.annotate_text(term.replace("_", " "), config))
        return [_convert_annotation_to_dict(ann) for ann in annotations]
    else:
        annotations = [
            x for x in adapter.annotate_text(term.replace("_", " "), config)
            if len(x.object_label) > 2
        ]
        if annotations:
            max_overlap_ann = max(
                annotations,
                key=lambda obj: _overlap(obj.object_label, term)
            )
            return [_convert_annotation_to_dict(max_overlap_ann)]
        return []

def annotate_via_oak_spark(
    input_df: pd.DataFrame,
    column: str,
    resource: str,
    outfile: Path,
) -> None:
    """
    Annotate dataframe column text using oaklib + llm with PySpark for distributed processing.

    Args:
        input_df: Input pandas DataFrame
        column: Column to be annotated
        resource: Ontology resource file path
        outfile: Output file path

    """
    # Initialize Spark
    spark = create_spark_session()

    # Convert input DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(input_df)

    # Setup resource path and adapter
    resource_path = Path(resource)
    db_path = resource.replace(resource_path.suffix, ".db")
    resource = db_path if Path(db_path).exists() else resource
    adapter = get_adapter(f"sqlite:{resource}")

    # Create configurations
    exact_config = TextAnnotationConfiguration(
        include_aliases=True,
        matches_whole_text=True,
    )
    partial_config = TextAnnotationConfiguration(
        include_aliases=True,
        matches_whole_text=False,
    )

    # Define schema for annotation results
    annotation_schema = ArrayType(StructType([
        StructField("object_id", StringType(), True),
        StructField("object_label", StringType(), True),
        StructField("match_type", StringType(), True),
        StructField("match_string", StringType(), True),
        StructField("start", StringType(), True),
        StructField("end", StringType(), True)
    ]))

    # Create UDF for exact matching
    @udf(annotation_schema)
    def annotate_exact(term):
        if term:
            return _annotate_single_term(term, adapter, exact_config)
        return []

    # Create UDF for partial matching
    @udf(annotation_schema)
    def annotate_partial(term):
        if term:
            return _annotate_single_term(term, adapter, partial_config, exact_match=False)
        return []

    # Process exact matches
    exact_matches = (spark_df
        .select(col(column))
        .distinct()
        .withColumn("annotations", annotate_exact(col(column)))
        .filter(col("annotations").isNotNull() & (col("annotations") != array()))
        .select(col(column), explode("annotations").alias("annotation"))
    )

    # Process unmatched terms for partial matching
    unmatched_terms = (spark_df
        .select(col(column))
        .distinct()
        .withColumn("annotations", annotate_exact(col(column)))
        .filter(col("annotations").isNull() | (col("annotations") == array()))
        .withColumn("partial_annotations", annotate_partial(col(column)))
        .filter(col("partial_annotations").isNotNull() & (col("partial_annotations") != array()))
        .select(col(column), explode("partial_annotations").alias("annotation"))
    )

    # Convert results back to pandas and save
    exact_matches.toPandas().to_csv(outfile, sep="\t", index=False)
    unmatched_outfile = outfile.with_name(f"{outfile.stem}_unmatched{outfile.suffix}")
    unmatched_terms.toPandas().to_csv(unmatched_outfile, sep="\t", index=False)

    # Clean up
    spark.stop()

if __name__ == "__main__":
    pass
