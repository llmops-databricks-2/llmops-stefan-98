"""Set agent credentials before importing MLflow or initializing agent clients."""

import os

from databricks.sdk import WorkspaceClient

from notubiz_curator.config import ProjectConfig


def setup_auth(cfg: ProjectConfig) -> None:
    """Read SPN credentials from UC secrets and configure the agent environment."""
    workspace = WorkspaceClient()

    client_id = workspace.secrets_uc.get_secret(
        full_name=f"{cfg.catalog}.{cfg.schema}.databricks_client_id",
        include_value=True,
    ).effective_value
    client_secret = workspace.secrets_uc.get_secret(
        full_name=f"{cfg.catalog}.{cfg.schema}.databricks_client_secret",
        include_value=True,
    ).effective_value
    if not client_id or not client_secret or not workspace.config.host:
        raise RuntimeError("Databricks App SPN credentials or host are missing")

    os.environ["DATABRICKS_CLIENT_ID"] = client_id
    os.environ["DATABRICKS_CLIENT_SECRET"] = client_secret
    os.environ["DATABRICKS_HOST"] = workspace.config.host
