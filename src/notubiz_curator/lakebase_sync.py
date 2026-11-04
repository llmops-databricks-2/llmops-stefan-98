"""Create and trigger the Notubiz Lakebase Synced Table."""

import time
from typing import TYPE_CHECKING

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.database import (
    SyncedDatabaseTable,
    SyncedTableSchedulingPolicy,
    SyncedTableSpecPgSpecificType,
    SyncedTableSpecTypeOverride,
)
from databricks.sdk.service.pipelines import UpdateInfoState
from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger
from psycopg import sql

from notubiz_curator.chunking_processor import DEFAULT_CHUNKS_TABLE
from notubiz_curator.config import ProjectConfig
from notubiz_curator.vector_search import (
    EMBEDDING_DIMENSIONS,
    POSTGRES_DATABASE,
    SYNCED_TABLE_NAME,
    LakebasePgVectorManager,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _validate_source(spark: "SparkSession", source: str) -> None:
    from pyspark.sql import functions as F

    df = spark.table(source)
    required = {
        "chunk_id",
        "document_id",
        "document_title",
        "meeting_id",
        "meeting_date",
        "source_url",
        "chunk_index",
        "section",
        "text",
        "embedding",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")
    counts = df.agg(
        F.count("*").alias("total"),
        F.countDistinct("chunk_id").alias("distinct_ids"),
        F.count(F.when(F.col("chunk_id").isNull() | (F.col("chunk_id") == ""), 1)).alias("invalid_ids"),
        F.count(F.when(F.col("embedding").isNull() | (F.size("embedding") != EMBEDDING_DIMENSIONS), 1)).alias(
            "invalid_embeddings"
        ),
    ).first()
    if counts is None or not counts["total"]:
        raise ValueError(f"{source} contains no chunks")
    if counts["invalid_ids"] or counts["total"] != counts["distinct_ids"]:
        raise ValueError(f"{source} contains missing or duplicate chunk_id values")
    if counts["invalid_embeddings"]:
        raise ValueError(f"{source} contains embeddings without {EMBEDDING_DIMENSIONS} dimensions")
    quoted = ".".join(f"`{part.replace('`', '``')}`" for part in source.split("."))
    spark.sql(f"ALTER TABLE {quoted} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    logger.info("Validated {} chunks and enabled Change Data Feed on {}", counts["total"], source)


def _validate_existing(
    table: SyncedDatabaseTable, source: str, expected_branch_id: str, properties: dict[str, str]
) -> None:
    spec = table.spec
    expected_override = SyncedTableSpecTypeOverride(
        column_name="embedding",
        pg_type=SyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
        size=EMBEDDING_DIMENSIONS,
    )
    if (
        spec is None
        or spec.source_table_full_name != source
        or table.effective_logical_database_name != POSTGRES_DATABASE
        or properties.get("database_branch_id") != expected_branch_id
        or spec.primary_key_columns != ["chunk_id"]
        or spec.scheduling_policy != SyncedTableSchedulingPolicy.TRIGGERED
        or expected_override not in (spec.type_overrides or [])
    ):
        raise ValueError(
            "Existing Synced Table does not match the expected source, destination, key, policy or vector type"
        )


def sync_chunks(
    spark: "SparkSession",
    config: ProjectConfig,
    workspace: WorkspaceClient,
    timeout_seconds: int = 3600,
) -> str:
    """Create or trigger a Synced Table, wait for the update, and ensure its index."""
    if not config.lakebase_enabled:
        raise ValueError("Lakebase is not enabled for this environment")
    if not config.lakebase_instance or not config.usage_policy_id:
        raise ValueError("lakebase_instance and usage_policy_id are required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    source = f"{config.full_schema_name}.{DEFAULT_CHUNKS_TABLE}"
    synced = f"{config.full_schema_name}.{SYNCED_TABLE_NAME}"
    name = f"synced_tables/{synced}"
    branch = f"projects/{config.lakebase_instance}/branches/production"
    _validate_source(spark, source)
    manager = LakebasePgVectorManager(config=config, workspace=workspace)

    try:
        existing = workspace.postgres.get_synced_table(name=name)
    except NotFound:
        existing = None

    update_id = None
    pipeline_id = None
    if existing is None:
        manager.ensure_extension()
        workspace.postgres.create_synced_table(
            synced_table=SyncedTable(
                spec=SyncedTableSyncedTableSpec(
                    source_table_full_name=source,
                    branch=branch,
                    postgres_database=POSTGRES_DATABASE,
                    primary_key_columns=["chunk_id"],
                    scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                    create_database_objects_if_missing=True,
                    new_pipeline_spec=NewPipelineSpec(
                        budget_policy_id=config.usage_policy_id,
                        storage_catalog=config.catalog,
                        storage_schema=config.schema,
                    ),
                    type_overrides=[
                        SyncedTableSyncedTableSpecTypeOverride(
                            column_name="embedding",
                            pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                            size=EMBEDDING_DIMENSIONS,
                        )
                    ],
                )
            ),
            synced_table_id=synced,
        ).wait()
        logger.info("Created Synced Table {}; initial synchronization started", synced)
    else:
        if not existing.status or not existing.status.pipeline_id:
            raise RuntimeError(f"No pipeline is available for {synced}")
        if existing.status.project != f"projects/{config.lakebase_instance}":
            raise ValueError(f"Existing Synced Table {synced} belongs to a different Lakebase project")
        branch_id = workspace.postgres.get_branch(name=branch).uid
        properties = workspace.tables.get(full_name=synced).properties
        if not branch_id or not properties:
            raise RuntimeError(f"Branch metadata is unavailable for {synced}")
        _validate_existing(workspace.database.get_synced_database_table(name=synced), source, branch_id, properties)
        manager.ensure_extension()
        pipeline_id = existing.status.pipeline_id
        update_id = workspace.pipelines.start_update(pipeline_id=pipeline_id).update_id
        if not update_id:
            raise RuntimeError(f"No update ID returned for pipeline {pipeline_id}")
        logger.info("Started sync update {} for {}", update_id, synced)

    deadline = time.monotonic() + timeout_seconds
    while True:
        table = workspace.postgres.get_synced_table(name=name)
        status = table.status
        if not status:
            raise RuntimeError(f"No status returned for {synced}")
        if status.detailed_state in {
            SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED,
            SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED,
        }:
            raise RuntimeError(f"Synced Table failed: {status.detailed_state} | {status.message}")
        if update_id:
            if not pipeline_id:
                raise RuntimeError(f"No pipeline is available for update {update_id}")
            update = workspace.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_id).update
            if not update:
                raise RuntimeError(f"No status returned for sync update {update_id}")
            if update.state == UpdateInfoState.COMPLETED:
                break
            if update.state in {UpdateInfoState.FAILED, UpdateInfoState.CANCELED}:
                raise RuntimeError(f"Sync update {update_id} ended in state {update.state}")
        elif status.detailed_state in {
            SyncedTableState.SYNCED_TABLE_ONLINE,
            SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
        }:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for Synced Table {synced}")
        logger.info("Waiting for sync: {}", status.detailed_state)
        time.sleep(30)

    logger.info("Synced Table {} is up to date", synced)
    manager.ensure_indexes()
    with manager.connection() as conn:
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING lakebase_ann (embedding vector_cosine_ops)").format(
                sql.Identifier(f"{SYNCED_TABLE_NAME}_embedding_ann"),
                sql.Identifier(config.schema, SYNCED_TABLE_NAME),
            )
        )
    logger.info("Lakebase ANN index ready on {}", synced)
    return synced
