"""Delta storage helpers for normalized Notubiz documents."""

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from notubiz_curator.config import ProjectConfig
from notubiz_curator.ingestion import NotubizDocument

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

DEFAULT_DOCUMENTS_TABLE = "notubiz_documents"
DOCUMENT_COLUMNS = (
    "document_id",
    "organization",
    "document_title",
    "meeting_id",
    "meeting_date",
    "date_modified",
    "source_url",
    "pdf_source_url",
    "content_type",
    "document_text",
)


def get_spark(databricks_profile: str | None = None) -> "SparkSession":
    """Return the runtime Spark session or a local Databricks Connect session."""
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()

    if not databricks_profile:
        msg = (
            "A Databricks Connect profile is required outside a Databricks runtime. "
            "Pass --databricks-profile or set DATABRICKS_CONFIG_PROFILE."
        )
        raise ValueError(msg)

    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    return builder.profile(databricks_profile).serverless().getOrCreate()


def write_documents(
    documents: list[NotubizDocument],
    config: ProjectConfig,
    table_name: str = DEFAULT_DOCUMENTS_TABLE,
    spark: "SparkSession | None" = None,
    databricks_profile: str | None = None,
) -> str:
    """Merge normalized document records into a Delta table in the configured schema."""
    active_spark = spark or get_spark(databricks_profile)
    table = f"{config.full_schema_name}.{table_name}"
    document_df = _create_document_dataframe(documents, active_spark)
    _merge_documents(document_df, table, active_spark)
    return table


def overwrite_documents(
    documents: list[NotubizDocument],
    config: ProjectConfig,
    table_name: str = DEFAULT_DOCUMENTS_TABLE,
    spark: "SparkSession | None" = None,
    databricks_profile: str | None = None,
) -> str:
    """Replace a document table during the explicit one-time bootstrap."""
    active_spark = spark or get_spark(databricks_profile)
    document_df = _create_document_dataframe(documents, active_spark)
    table = f"{config.full_schema_name}.{table_name}"
    document_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    return table


def _create_document_dataframe(documents: list[NotubizDocument], spark: "SparkSession") -> object:
    """Create a typed Spark DataFrame from normalized document records."""
    if not documents:
        msg = "At least one document is required to write a Delta table"
        raise ValueError(msg)

    from pyspark.sql.types import StringType, StructField, StructType, TimestampType

    schema = StructType(
        [
            StructField("document_id", StringType(), False),
            StructField("organization", StringType(), True),
            StructField("document_title", StringType(), True),
            StructField("meeting_id", StringType(), True),
            StructField("meeting_date", StringType(), True),
            StructField("date_modified", StringType(), True),
            StructField("source_url", StringType(), True),
            StructField("pdf_source_url", StringType(), True),
            StructField("content_type", StringType(), True),
            StructField("document_text", StringType(), True),
            StructField("ingested_at", TimestampType(), False),
        ]
    )
    ingested_at = datetime.now(UTC)
    rows = [(*document.as_record().values(), ingested_at) for document in documents]
    return spark.createDataFrame(rows, schema=schema)


def _merge_documents(document_df: object, table: str, spark: "SparkSession") -> None:
    """Insert new documents and update only records whose source values changed."""
    if not spark.catalog.tableExists(table):
        document_df.write.format("delta").saveAsTable(table)
        return

    source_view = "notubiz_documents_source"
    document_df.createOrReplaceTempView(source_view)
    unchanged_condition = " AND ".join(
        f"target.{column} <=> source.{column}" for column in DOCUMENT_COLUMNS if column != "document_id"
    )
    update_assignments = ", ".join(
        f"{column} = source.{column}" for column in DOCUMENT_COLUMNS if column != "document_id"
    )
    insert_columns = ", ".join((*DOCUMENT_COLUMNS, "ingested_at"))
    insert_values = ", ".join(f"source.{column}" for column in (*DOCUMENT_COLUMNS, "ingested_at"))
    spark.sql(
        f"""
        MERGE INTO {table} AS target
        USING {source_view} AS source
        ON target.document_id = source.document_id
        WHEN MATCHED AND NOT ({unchanged_condition}) THEN UPDATE SET
          {update_assignments},
          ingested_at = source.ingested_at
        WHEN NOT MATCHED THEN INSERT ({insert_columns})
        VALUES ({insert_values})
        """
    )
