# Databricks notebook source
# MAGIC %md
# MAGIC # Initialize the Genie source table
# MAGIC
# MAGIC Create the schema and an empty Delta table if missing. Existing data is
# MAGIC preserved. Ingestion and Lakebase synchronization run in separate jobs.

# COMMAND ----------

"""Create the Notubiz schema and empty Delta table before Genie space deployment."""

from loguru import logger
from pyspark.sql import SparkSession

from notubiz_curator.config import get_env, load_config

dbutils.widgets.text("env", get_env())  # noqa: F821
env = dbutils.widgets.get("env").strip()  # noqa: F821
spark = SparkSession.builder.getOrCreate()
cfg = load_config("../../project_config.yml", env=env)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {cfg.catalog}.{cfg.schema}.notubiz_documents (
        document_id STRING NOT NULL,
        organization STRING,
        document_title STRING,
        meeting_id STRING,
        meeting_date STRING,
        date_modified STRING,
        source_url STRING,
        pdf_source_url STRING,
        content_type STRING,
        document_text STRING,
        ingested_at TIMESTAMP NOT NULL
    ) USING DELTA
""")
logger.info(f"Genie source table ready: {cfg.catalog}.{cfg.schema}.notubiz_documents")
