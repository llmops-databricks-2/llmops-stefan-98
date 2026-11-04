"""Search Notubiz chunks in a Lakebase Synced Table with pgvector."""

import math
import os
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger
from psycopg import sql
from psycopg.conninfo import make_conninfo

from notubiz_curator.config import ProjectConfig

SYNCED_TABLE_NAME = "chunks_synced"
EMBEDDING_DIMENSIONS = 1024
POSTGRES_DATABASE = "databricks_postgres"


class LakebasePgVectorManager:
    """Create indexes and query the read-only Lakebase copy of Notubiz chunks."""

    synced_table_name = SYNCED_TABLE_NAME

    def __init__(self, config: ProjectConfig, workspace: WorkspaceClient | None = None) -> None:
        if not config.lakebase_enabled:
            raise ValueError("Lakebase is not enabled for this environment")
        if not config.lakebase_instance:
            raise ValueError("lakebase_instance is required")
        self.config = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.lakebase_project_id = config.lakebase_instance
        self.embedding_model = config.embedding_endpoint
        self.table_name = sql.Identifier(self.schema, self.synced_table_name)
        self.text_expr = sql.SQL("to_tsvector('dutch', coalesce(document_title, '') || ' ' || coalesce(text, ''))")
        self._connection_string: str | None = None
        self.workspace = workspace

    def _get_connection_string(self) -> str:
        """Create a short-lived Lakebase connection string."""
        if self._connection_string:
            return self._connection_string
        workspace = self.workspace or WorkspaceClient()
        branch = f"projects/{self.lakebase_project_id}/branches/production"
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT") or f"{branch}/endpoints/primary"
        prefix = f"{branch}/endpoints/"
        if (
            not endpoint_name.startswith(prefix)
            or "/" in endpoint_name[len(prefix) :]
            or not endpoint_name[len(prefix) :]
        ):
            raise ValueError("Lakebase endpoint must belong to the configured project and production branch")
        endpoint = workspace.postgres.get_endpoint(name=endpoint_name)
        if not endpoint.name or not endpoint.status or not endpoint.status.hosts or not endpoint.status.hosts.host:
            raise RuntimeError(f"No Lakebase host is available for {endpoint_name}")
        hosts = endpoint.status.hosts
        host = hosts.host
        database = POSTGRES_DATABASE
        if os.environ.get("LAKEBASE_ENDPOINT"):
            requested_host = os.environ.get("PGHOST")
            if requested_host and requested_host != host:
                raise ValueError("PGHOST must be the configured Lakebase endpoint's unpooled host")
            database = os.environ.get("PGDATABASE", POSTGRES_DATABASE)
            if database != POSTGRES_DATABASE:
                raise ValueError(f"PGDATABASE must be {POSTGRES_DATABASE}")
        credential = workspace.postgres.generate_database_credential(endpoint=endpoint.name)
        username = os.environ.get("DATABRICKS_CLIENT_ID") or workspace.current_user.me().user_name
        if not username or not credential.token:
            raise RuntimeError("Lakebase username or database credential is missing")
        self._connection_string = make_conninfo(
            host=host,
            dbname=database,
            user=username,
            password=credential.token,
            sslmode="require",
        )
        return self._connection_string

    def connection(self) -> psycopg.Connection:
        """Open a Lakebase connection using a fresh workspace credential."""
        self._connection_string = None
        try:
            return psycopg.connect(self._get_connection_string())
        finally:
            self._connection_string = None

    def ensure_extension(self) -> None:
        """Install the course's Lakebase vector extension before table creation."""
        with self.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
        logger.info("Lakebase vector extension ready")

    def ensure_indexes(self) -> None:
        """Create the full-text index used by hybrid search."""
        with self.connection() as conn:
            conn.execute(
                sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {}
                    ON {} USING gin (({}))
                """).format(sql.Identifier(f"{self.synced_table_name}_search_gin"), self.table_name, self.text_expr)
            )
        logger.info("Full-text index ready on {}.{}", self.schema, self.synced_table_name)

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using the model configured for the source chunks."""
        workspace = self.workspace or WorkspaceClient()
        client = workspace.serving_endpoints.get_open_ai_client()
        response = client.embeddings.create(model=self.embedding_model, input=query)
        if not response.data or not response.data[0].embedding:
            raise ValueError("The embedding endpoint returned no vector")
        vector = list(response.data[0].embedding)
        if len(vector) != EMBEDDING_DIMENSIONS or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Expected a finite {EMBEDDING_DIMENSIONS}-dimensional embedding")
        return vector

    @staticmethod
    def _vector_literal(vector: list[float] | tuple[float, ...]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def _fetch(self, statement: sql.Composed, params: list[Any]) -> list[dict[str, Any]]:
        with self.connection() as conn, conn.cursor() as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
            if cursor.description is None:
                raise RuntimeError("Search query did not return columns")
            columns = [column.name for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def search(self, query: str, num_results: int = 5, meeting_id: str | None = None) -> list[dict[str, Any]]:
        """Rank chunks by cosine similarity, optionally restricting the meeting."""
        if num_results < 1:
            raise ValueError("num_results must be positive")
        vector = self._vector_literal(self.embed_query(query))
        where = sql.SQL("WHERE embedding IS NOT NULL")
        params: list[Any] = [vector]
        if meeting_id is not None:
            where += sql.SQL(" AND meeting_id = %s")
            params.append(meeting_id)
        params.extend([vector, num_results])
        statement = sql.SQL("""
            SELECT chunk_id, document_id, document_title, meeting_id, meeting_date, source_url,
                   chunk_index, section, text, 1 - (embedding <=> %s::vector) AS score
            FROM {}
            {}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """).format(self.table_name, where)
        return self._fetch(statement, params)

    def hybrid_search(
        self,
        query: str,
        num_results: int = 5,
        meeting_id: str | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Combine cosine similarity and Dutch PostgreSQL full-text ranking."""
        if num_results < 1:
            raise ValueError("num_results must be positive")
        if not 0 <= vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        query_vector = self._vector_literal(self.embed_query(query))
        text_weight = 1.0 - vector_weight
        where = sql.SQL("WHERE embedding IS NOT NULL")
        params: list[Any] = [query_vector, query]
        if meeting_id is not None:
            where += sql.SQL(" AND meeting_id = %s")
            params.append(meeting_id)
        params.extend([vector_weight, text_weight, num_results])
        statement = sql.SQL("""
            WITH ranked AS (
                SELECT chunk_id, document_id, document_title, meeting_id, meeting_date, source_url,
                       chunk_index, section, text, 1 - (embedding <=> %s::vector) AS vector_score,
                       ts_rank_cd({}, plainto_tsquery('dutch', %s)) AS text_score
                FROM {}
                {}
            )
            SELECT *, (%s * vector_score + %s * text_score) AS score
            FROM ranked
            ORDER BY score DESC
            LIMIT %s
        """).format(self.text_expr, self.table_name, where)
        return self._fetch(statement, params)
