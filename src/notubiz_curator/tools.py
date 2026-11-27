"""Local document search tool for the Notubiz agent."""

import json
from datetime import date

import mlflow
from loguru import logger
from mlflow.entities import SpanType

from notubiz_curator.mcp import ToolInfo
from notubiz_curator.vector_search import LakebasePgVectorManager


def create_search_documents_tool(manager: LakebasePgVectorManager) -> ToolInfo:
    """Wrap the existing Lakebase hybrid search in a Chat Completions tool."""

    @mlflow.trace(name="search_documents", span_type=SpanType.TOOL)
    def search_documents(
        query: str,
        num_results: int = 5,
        meeting_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(num_results, bool) or not isinstance(num_results, int) or not 1 <= num_results <= 20:
            raise ValueError("num_results must be an integer between 1 and 20")
        if meeting_id is not None and (not isinstance(meeting_id, str) or not meeting_id.strip()):
            raise ValueError("meeting_id must be a non-empty string")

        parsed_date_from = _parse_date_filter(date_from, "date_from")
        parsed_date_to = _parse_date_filter(date_to, "date_to")
        if parsed_date_from is not None and parsed_date_to is not None and parsed_date_from > parsed_date_to:
            raise ValueError("date_from must be on or before date_to")

        logger.info(
            "Hybrid search started: requested_results={}, meeting_filter={}, date_from={}, date_to={}",
            num_results,
            meeting_id is not None,
            date_from,
            date_to,
        )
        results = manager.hybrid_search(
            query=query,
            num_results=num_results,
            meeting_id=meeting_id,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
        )
        logger.info("Hybrid search returned {} chunks", len(results))
        return json.dumps(
            [
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "document_title": row["document_title"],
                    "meeting_id": row["meeting_id"],
                    "meeting_date": row["meeting_date"],
                    "source_url": row["source_url"],
                    "chunk_index": row["chunk_index"],
                    "section": row["section"],
                    "excerpt": (row["text"] or "")[:500],
                    "score": row["score"],
                }
                for row in results
            ],
            ensure_ascii=False,
            default=str,
        )

    return ToolInfo(
        name="search_documents",
        spec={
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Hybrid search in Notubiz document excerpts. Use for questions about document contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "num_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "Use 10 for broad questions asking for multiple or all items.",
                        },
                        "meeting_id": {"type": "string", "description": "Optional meeting ID to filter results."},
                        "date_from": {
                            "type": "string",
                            "format": "date",
                            "description": "Inclusive meeting date lower bound in YYYY-MM-DD format.",
                        },
                        "date_to": {
                            "type": "string",
                            "format": "date",
                            "description": "Inclusive meeting date upper bound in YYYY-MM-DD format.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        exec_fn=search_documents,
    )


def _parse_date_filter(value: str | None, parameter: str) -> date | None:
    """Validate an optional ISO date filter."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{parameter} must be a non-empty date in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{parameter} must be a valid date in YYYY-MM-DD format") from exc
