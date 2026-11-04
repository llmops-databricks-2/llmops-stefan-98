"""Clean, chunk and embed Notubiz documents."""

import html
import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from notubiz_curator.config import ProjectConfig
from notubiz_curator.storage import DEFAULT_DOCUMENTS_TABLE

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)
DEFAULT_CHUNKS_TABLE = "notubiz_chunks"
MAX_CHARS = 1200
OVERLAP_CHARS = 100


@dataclass(frozen=True)
class TextChunk:
    """A Markdown chunk with its section heading."""

    text: str
    section: str


class ChunkingProcessor:
    """Process existing Notubiz Markdown into chunks and embeddings."""

    def __init__(
        self,
        spark: "SparkSession",
        config: ProjectConfig,
        source_table: str = DEFAULT_DOCUMENTS_TABLE,
        target_table: str = DEFAULT_CHUNKS_TABLE,
        max_chars: int = MAX_CHARS,
    ) -> None:
        if source_table == target_table:
            raise ValueError("Source and target tables must differ")
        if max_chars < 32 or max_chars > MAX_CHARS:
            raise ValueError(f"max_chars must be between 32 and {MAX_CHARS}")
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.source_table = self._qualified_table(source_table)
        self.target_table = self._qualified_table(target_table)
        self.target_name = target_table
        self.max_chars = max_chars

    @staticmethod
    def _identifier(name: str) -> str:
        if not name or "\x00" in name:
            raise ValueError("Table identifiers must be nonempty and cannot contain NUL")
        return f"`{name.replace('`', '``')}`"

    def _qualified_table(self, name: str) -> str:
        return ".".join(self._identifier(part) for part in (self.catalog, self.schema, name))

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """Clean Markdown artifacts while retaining document structure."""
        text = html.unescape(text.replace("\r\n", "\n").replace("\r", "\n"))
        text = re.sub(r"!\[([^\]\n]*)\]\([^)\n]+\)", r"\1", text)
        text = re.sub(r"\[([^\]\n]+)\]\([^)\n]+\)", r"\1", text)
        text = re.sub(r"\\([\\`*_{}\[\]()#+.!>~-])", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _extract_chunks(markdown: str, title: str | None, max_chars: int = MAX_CHARS) -> list[TextChunk]:
        """Split Markdown by heading and character count."""
        if max_chars < 32 or max_chars > MAX_CHARS:
            raise ValueError(f"max_chars must be between 32 and {MAX_CHARS}")
        cleaned = ChunkingProcessor._clean_chunk(markdown)
        if not cleaned:
            return []

        headers = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
        )
        chunks: list[TextChunk] = []
        for section in headers.split_text(cleaned):
            heading = " > ".join(section.metadata[key] for key in ("h1", "h2", "h3", "h4") if key in section.metadata)
            prefix = "\n".join(
                part
                for part in (f"Document: {title}" if title else "", f"Section: {heading}" if heading else "")
                if part
            )[: min(200, max_chars // 4)]
            budget = max_chars - len(prefix) - (2 if prefix else 0)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=min(OVERLAP_CHARS, budget // 4),
                separators=["\n\n", "\n", " ", ""],
            )
            for body in splitter.split_text(section.page_content):
                text = f"{prefix}\n\n{body}" if prefix else body
                if len(text) > max_chars:
                    raise ValueError(f"Chunk has {len(text)} characters, exceeding {max_chars}")
                chunks.append(TextChunk(text=text, section=heading))
        return chunks

    def process_chunks(self) -> str:
        """Create a chunk table after all documents and embeddings succeed.

        An existing target is never replaced. A staging table remains if the run fails.
        """
        from pyspark.sql import functions as F
        from pyspark.sql.types import ArrayType, StringType, StructField, StructType

        if self.spark.catalog.tableExists(self.target_table):
            raise ValueError(f"Target table {self.target_table} already exists; it will not be overwritten")
        if not self.spark.catalog.tableExists(self.source_table):
            raise ValueError(f"Source table {self.source_table} does not exist")
        version = self.spark.sql(f"DESCRIBE HISTORY {self.source_table} LIMIT 1").select("version").first()
        if version is None:
            raise ValueError(f"No Delta history found for source table {self.source_table}")
        source_version = int(version["version"])
        df = self.spark.sql(f"SELECT * FROM {self.source_table} VERSION AS OF {source_version}")
        counts = df.agg(
            F.count("*").alias("total"),
            F.countDistinct("document_id").alias("distinct_ids"),
            F.count(F.when(F.col("document_id").isNull() | (F.col("document_id") == ""), 1)).alias("missing_ids"),
        ).first()
        if counts is None or counts["missing_ids"] or counts["total"] != counts["distinct_ids"]:
            raise ValueError("Source documents contain missing or duplicate document_id values")

        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("text", StringType(), False),
                    StructField("section", StringType(), False),
                ]
            ),
            containsNull=False,
        )
        max_chars = self.max_chars

        def extract_chunks(markdown: str | None, title: str | None) -> list[tuple[str, str]]:
            if not markdown:
                return []
            return [
                (chunk.text, chunk.section) for chunk in ChunkingProcessor._extract_chunks(markdown, title, max_chars)
            ]

        extract_chunks_udf = F.udf(extract_chunks, chunk_schema)
        chunks_df = (
            df.withColumn("chunks", extract_chunks_udf(F.col("document_text"), F.col("document_title")))
            .select(
                "document_id",
                "meeting_id",
                "meeting_date",
                "document_title",
                "source_url",
                F.posexplode("chunks").alias("chunk_index", "chunk"),
            )
            .select(
                F.sha2(F.concat(F.col("document_id"), F.lit("\x00"), F.col("chunk_index").cast("string")), 256).alias(
                    "chunk_id"
                ),
                "document_id",
                "meeting_id",
                "meeting_date",
                "document_title",
                "source_url",
                "chunk_index",
                F.col("chunk.section").alias("section"),
                F.col("chunk.text").alias("text"),
            )
        )

        run_id = uuid.uuid4().hex
        staging = self._qualified_table(f"{self.target_name}_staging_{run_id}")
        logger.info("Creating chunks from %d documents.", counts["total"])
        chunks_df.write.format("delta").saveAsTable(staging)
        chunks_df = self.spark.table(staging)
        total = chunks_df.count()
        logger.info("Created %d chunks.", total)
        if total == 0:
            raise ValueError("No usable document chunks were produced; target table was not created")

        endpoint = self.cfg.embedding_endpoint.replace("'", "''")
        embedded_df = chunks_df.withColumn("embedding", F.expr(f"CAST(ai_query('{endpoint}', text) AS ARRAY<FLOAT>)"))
        view = f"notubiz_chunks_{run_id}"
        embedded_df.createOrReplaceTempView(view)
        logger.info("Creating embeddings for %d chunks.", total)
        try:
            self.spark.sql(
                f"CREATE TABLE {self.target_table} TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true') "
                f"AS SELECT * FROM `{view}`"
            )
        finally:
            self.spark.catalog.dropTempView(view)
        logger.info("Created embeddings for %d chunks.", total)
        self.spark.sql(f"DROP TABLE {staging}")
        logger.info("Created %s with %d embedded chunks.", self.target_table, total)
        return f"{self.cfg.full_schema_name}.{self.target_name}"
