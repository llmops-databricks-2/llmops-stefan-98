# Databricks notebook source
# /// script
# dependencies = [
#   "..",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Notubiz chunk statistics
# MAGIC
# MAGIC Inspect chunk sizes and check the configured character limit.

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from notubiz_curator.chunking_processor import MAX_CHARS
from notubiz_curator.config import get_env, load_config

config = load_config(env=get_env())
catalog, schema = config.catalog, config.schema
spark = SparkSession.builder.getOrCreate()
chunks_df = spark.table(f"{catalog}.{schema}.notubiz_chunks")

logger.info(f"Total chunks available: {chunks_df.count()}")
chunks_df.select("document_id", "chunk_index", "text").show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze chunk statistics
# MAGIC
# MAGIC Character length is an estimate, not an exact token count.

# COMMAND ----------

chunk_stats = chunks_df.select(
    F.avg(F.length(col("text"))).alias("avg_length"),
    F.min(F.length(col("text"))).alias("min_length"),
    F.max(F.length(col("text"))).alias("max_length"),
    F.sum(F.when(F.length(col("text")) > MAX_CHARS, 1).otherwise(0)).alias("oversized_chunks"),
    F.count("*").alias("total_chunks"),
).collect()[0]

logger.info(f"Chunk Statistics: {chunk_stats['total_chunks']} chunks")
if chunk_stats["total_chunks"] == 0:
    raise ValueError("The chunks table is empty")
logger.info(
    f"Characters: avg={chunk_stats['avg_length']:.0f}, min={chunk_stats['min_length']}, max={chunk_stats['max_length']}"
)
if chunk_stats["oversized_chunks"]:
    raise ValueError(f"{chunk_stats['oversized_chunks']} chunks exceed {MAX_CHARS} characters")
