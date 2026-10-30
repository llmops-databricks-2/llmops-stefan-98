"""Backfill all current Notubiz documents into a Delta table once."""

import argparse
import logging
import os

from notubiz_curator.config import load_config
from notubiz_curator.ingestion import fetch_all_documents
from notubiz_curator.storage import DEFAULT_DOCUMENTS_TABLE, overwrite_documents

logger = logging.getLogger(__name__)


def main() -> None:
    """Fetch the full snapshot and explicitly replace the Delta table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("dev", "acc", "prd"), default="dev")
    parser.add_argument("--config-path", default="project_config.yml")
    parser.add_argument("--table-name", default=DEFAULT_DOCUMENTS_TABLE)
    parser.add_argument("--databricks-profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    arguments = parser.parse_args()

    documents = fetch_all_documents()
    config = load_config(arguments.config_path, env=arguments.environment)
    table = overwrite_documents(
        documents,
        config,
        arguments.table_name,
        databricks_profile=arguments.databricks_profile,
    )
    logger.info("Bootstrapped %d Notubiz documents into %s.", len(documents), table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
