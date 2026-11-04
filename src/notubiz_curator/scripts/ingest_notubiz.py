"""Retrieve the most recent Notubiz documents."""

import argparse
import logging
import os

from notubiz_curator.config import load_config
from notubiz_curator.ingestion import fetch_documents
from notubiz_curator.storage import DEFAULT_DOCUMENTS_TABLE, write_documents

logger = logging.getLogger(__name__)


def main() -> None:
    """Retrieve documents and write them to a Delta table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-count", type=int, default=50)
    parser.add_argument("--environment", choices=("dev", "acc", "prd"), default="dev")
    parser.add_argument("--config-path", default="project_config.yml")
    parser.add_argument("--table-name", default=DEFAULT_DOCUMENTS_TABLE)
    parser.add_argument("--databricks-profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    arguments = parser.parse_args()

    documents = fetch_documents(arguments.document_count)
    config = load_config(arguments.config_path, env=arguments.environment)
    table = write_documents(
        documents,
        config,
        arguments.table_name,
        databricks_profile=arguments.databricks_profile,
    )
    logger.info("Wrote %d Notubiz documents to %s.", len(documents), table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
