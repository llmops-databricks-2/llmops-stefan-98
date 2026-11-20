"""Session message persistence in Lakebase PostgreSQL."""

import json
import os
from typing import Any

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.conninfo import make_conninfo

from notubiz_curator.vector_search import POSTGRES_DATABASE


class LakebaseMemory:
    """Persist complete conversation turns in Lakebase."""

    def __init__(self, project_id: str, workspace: WorkspaceClient | None = None) -> None:
        self.project_id = project_id
        self.workspace = workspace

    def _get_connection_string(self) -> str:
        client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
        client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
        sp_host = os.environ.get("LAKEBASE_SP_HOST")
        supplied = (client_id, client_secret, sp_host)
        if any(supplied) and not all(supplied):
            raise ValueError("Set all of LAKEBASE_SP_CLIENT_ID, LAKEBASE_SP_CLIENT_SECRET and LAKEBASE_SP_HOST")
        if all(supplied):
            workspace = WorkspaceClient(host=sp_host, client_id=client_id, client_secret=client_secret)
            username = client_id
        else:
            workspace = self.workspace or WorkspaceClient()
            username = workspace.config.client_id or workspace.current_user.me().user_name

        branch = f"projects/{self.project_id}/branches/production"
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT") or f"{branch}/endpoints/primary"
        prefix = f"{branch}/endpoints/"
        if (
            not endpoint_name.startswith(prefix)
            or not endpoint_name[len(prefix) :]
            or "/" in endpoint_name[len(prefix) :]
        ):
            raise ValueError("Lakebase endpoint must belong to the configured project and production branch")
        endpoint = workspace.postgres.get_endpoint(name=endpoint_name)
        if not endpoint.name or not endpoint.status or not endpoint.status.hosts or not endpoint.status.hosts.host:
            raise RuntimeError(f"No Lakebase host is available for {endpoint_name}")
        host = endpoint.status.hosts.host
        if os.environ.get("LAKEBASE_ENDPOINT"):
            if os.environ.get("PGHOST") and os.environ["PGHOST"] != host:
                raise ValueError("PGHOST must be the configured Lakebase endpoint's unpooled host")
            if os.environ.get("PGDATABASE", POSTGRES_DATABASE) != POSTGRES_DATABASE:
                raise ValueError(f"PGDATABASE must be {POSTGRES_DATABASE}")
        credential = workspace.postgres.generate_database_credential(endpoint=endpoint.name)
        if not username or not credential.token:
            raise RuntimeError("Lakebase username or database credential is missing")
        return make_conninfo(
            host=host, dbname=POSTGRES_DATABASE, user=username, password=credential.token, sslmode="require"
        )

    def _connection(self) -> psycopg.Connection:
        """Create a connection with a fresh Lakebase credential."""
        return psycopg.connect(self._get_connection_string())

    def ensure_schema(self) -> None:
        """Create the session table and index if they do not exist."""
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id)")

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in insertion order."""
        with self._connection() as conn:
            result = conn.execute(
                "SELECT message_data FROM session_messages WHERE session_id = %s ORDER BY id ASC", (session_id,)
            ).fetchall()
            return [row[0] for row in result]

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append all messages from a completed turn in one transaction."""
        with self._connection() as conn:
            for message in messages:
                conn.execute(
                    "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
                    (session_id, json.dumps(message, ensure_ascii=False)),
                )
