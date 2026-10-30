"""Create or trigger the Notubiz Lakebase Synced Table."""

import argparse
import os

from databricks.sdk import WorkspaceClient

from notubiz_curator.config import load_config
from notubiz_curator.lakebase_sync import sync_chunks
from notubiz_curator.storage import get_spark


def main() -> None:
    """Run the triggered Lakebase synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("dev", "acc", "prd"), default="dev")
    parser.add_argument("--config-path", default="project_config.yml")
    parser.add_argument("--databricks-profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    args = parser.parse_args()
    config = load_config(args.config_path, env=args.environment)
    if not config.lakebase_enabled:
        raise ValueError("Lakebase is not enabled for this environment")
    spark = get_spark(args.databricks_profile)
    workspace = WorkspaceClient(profile=args.databricks_profile) if args.databricks_profile else WorkspaceClient()
    sync_chunks(spark, config, workspace)


if __name__ == "__main__":
    main()
