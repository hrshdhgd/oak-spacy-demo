from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, split, explode, array_contains
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
from pathlib import Path

# Initialize Spark session
spark = SparkSession.builder.appName("OntologyAnnotation").getOrCreate()

# Define utility functions
def _overlap(a, b):
    """Get number of characters in 2 strings that overlap using set intersection."""
    return len(set(a) & set(b))

@udf(returnType=ArrayType(StringType()))
def annotate_text(text, adapter):
    """UDF to annotate text using the adapter"""
    return [str(a.object_label) for a in adapter.annotate_text(text.replace("_", " "))]

@udf(returnType=ArrayType(StringType()))
def annotate_text_relaxed(text, adapter):
    """UDF to annotate text with relaxed matching"""
    annotations = [a for a in adapter.annotate_text(text.replace("_", " ")) if len(a.object_label) > 2]
    if annotations:
        max_overlap_annotation = max(annotations, key=lambda obj: _overlap(obj.object_label, text))
        max_overlap_annotation.subject_label = text if not max_overlap_annotation.subject_label else max_overlap_annotation.subject_label
        return [str(max_overlap_annotation.object_label)]
    else:
        return []

def annotate_via_spark(df, column, resource, outfile, n_partitions=None):
    """Annotate dataframe column text using PySpark"""
    # Setup resource path
    resource_path = Path(resource)
    db_path = resource.replace(resource_path.suffix, ".db")
    resource = db_path if Path(db_path).exists() else resource

    # Load adapter
    adapter = spark.sparkContext._jvm.org.ihmc.ontology.bridge.oaklib.get_adapter(f"sqlite:{resource}")

    # Annotate dataframe column
    df = df.repartition(n_partitions or spark.sparkContext.defaultParallelism)
    annotated = df.withColumn("exact_matches", annotate_text(col(column), adapter))
    unmatched = annotated.filter(~array_contains(col("exact_matches"), col("value")))
    unmatched = unmatched.withColumn("partial_matches", annotate_text_relaxed(col(column), adapter))

    # Write results
    converter = spark.sparkContext._jvm.org.ihmc.ontology.bridge.oaklib.get_uri_converter()
    annotated.select(col(column).alias("value"), explode("exact_matches").alias("object_label"), converter.expand(col("object_label")).alias("subject_label")).write.option("header", True).option("sep", "\t").option("quoting", 0).csv(str(outfile))
    unmatched.select(col(column).alias("value"), explode("partial_matches").alias("object_label"), converter.expand(col("object_label")).alias("subject_label")).write.option("header", True).option("sep", "\t").option("quoting", 0).csv(str(outfile.parent / f"{outfile.stem}_unmatched{outfile.suffix}"))

if __name__ == "__main__":
    # Example usage
    # df = spark.createDataFrame(pd.DataFrame({"text": ["example 1", "example 2", "example 3"]}))
    # annotate_via_spark(df, "text", "path/to/resource.db", Path("output.tsv"), n_partitions=4)
    pass