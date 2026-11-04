"""Create a Notubiz chunks table from the existing document table."""

import argparse
import logging
import os

from notubiz_curator.chunking_processor import DEFAULT_CHUNKS_TABLE, MAX_CHARS, ChunkingProcessor
from notubiz_curator.config import load_config
from notubiz_curator.storage import DEFAULT_DOCUMENTS_TABLE, get_spark


def main() -> None:
    """Run the one-time chunking and embedding pipeline."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("dev", "acc", "prd"), default="dev")
    parser.add_argument("--config-path", default="project_config.yml")
    parser.add_argument("--source-table", default=DEFAULT_DOCUMENTS_TABLE)
    parser.add_argument("--target-table", default=DEFAULT_CHUNKS_TABLE)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--databricks-profile", default=os.getenv("DATABRICKS_CONFIG_PROFILE"))
    args = parser.parse_args()
    config = load_config(args.config_path, env=args.environment)
    spark = get_spark(args.databricks_profile)
    processor = ChunkingProcessor(
        spark,
        config,
        source_table=args.source_table,
        target_table=args.target_table,
        max_chars=args.max_chars,
    )
    processor.process_chunks()


if __name__ == "__main__":
    main()
