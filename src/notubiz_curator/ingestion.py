"""OpenBesluitvorming client helpers for Notubiz document retrieval."""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API_BASE_URL = "https://openbesluitvorming.nl/api"
WETTERSKIP_FRYSLAN_SOURCE = "wetterskip_fryslan"
DEFAULT_TIMEOUT_SECONDS = 30
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotubizDocument:
    """One normalized public governance document suitable for later Delta storage."""

    document_id: str
    organization: str | None
    document_title: str | None
    meeting_id: str | None
    meeting_date: str | None
    date_modified: str | None
    source_url: str | None
    pdf_source_url: str | None
    content_type: str | None
    document_text: str | None

    def as_record(self) -> dict[str, str | None]:
        """Return a serializable record for a later Spark DataFrame or Delta write."""
        return asdict(self)


def _request_bytes(url: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    """Request bytes and honor an API rate-limit response once."""
    return _request(url, timeout_seconds)[0]


def _request(url: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> tuple[bytes, dict[str, str]]:
    """Request a response body and headers, honoring one rate-limit response."""
    request = Request(url, headers={"User-Agent": "notubiz-governance-assistant/1.0"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            if error.code != 429 or attempt:
                raise
            retry_after = int(error.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
        except URLError:
            raise
    msg = f"Request unexpectedly exhausted retry loop: {url}"
    raise RuntimeError(msg)


def _request_json(url: str) -> dict[str, Any]:
    """Request and decode one JSON response."""
    return json.loads(_request_bytes(url).decode("utf-8"))


def _request_snapshot_page(cursor: str | None = None) -> tuple[list[dict[str, Any]], str | None, bool]:
    """Return one OpenBesluitvorming bulk snapshot page."""
    parameters = {"source": WETTERSKIP_FRYSLAN_SOURCE, "limit": 1000}
    if cursor:
        parameters["cursor"] = cursor
    content, headers = _request(f"{API_BASE_URL}/export/snapshot?{urlencode(parameters)}")
    records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    has_more = normalized_headers.get("x-has-more", "false").lower() == "true"
    return records, normalized_headers.get("x-next-cursor"), has_more


def search_document_candidates(
    document_count: int,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> list[dict[str, Any]]:
    """Return the most recent Notubiz document search results."""
    if not 1 <= document_count <= 100:
        msg = "document_count must be between 1 and 100"
        raise ValueError(msg)

    parameters = urlencode(
        {
            "organization": WETTERSKIP_FRYSLAN_SOURCE,
            "entityType": "Document",
            "sort": "date_desc",
            "limit": 100,
        }
    )
    payload = request_json(f"{API_BASE_URL}/search?{parameters}")
    candidates = payload.get("results", [])
    if len(candidates) < document_count:
        msg = (
            f"OpenBesluitvorming returned {len(candidates)} Wetterskip Fryslan documents; "
            f"{document_count} are required."
        )
        raise RuntimeError(msg)

    return candidates[:document_count]


def get_document_detail(
    entity_id: str,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> dict[str, Any]:
    """Return a document record including extracted text and its public source URL."""
    return request_json(f"{API_BASE_URL}/entities/{quote(entity_id, safe='')}")


def normalize_document(detail: dict[str, Any]) -> NotubizDocument:
    """Map an API document detail response to a storage-neutral document record."""
    document_id = detail.get("entityId")
    if not isinstance(document_id, str):
        msg = "Document detail does not contain a string entityId"
        raise ValueError(msg)

    pdf_url = detail.get("pdfUrl")
    download_url = detail.get("downloadUrl")
    return NotubizDocument(
        document_id=document_id,
        organization=detail.get("organization"),
        document_title=detail.get("title"),
        meeting_id=detail.get("meetingId"),
        meeting_date=detail.get("sortDate"),
        date_modified=detail.get("dateModified"),
        source_url=pdf_url or download_url,
        pdf_source_url=pdf_url,
        content_type=detail.get("contentType"),
        document_text=detail.get("markdownText"),
    )


def fetch_documents(
    document_count: int,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> list[NotubizDocument]:
    """Retrieve and normalize the most recent documents without persisting data."""
    candidates = search_document_candidates(document_count, request_json=request_json)
    return [normalize_document(get_document_detail(candidate["entityId"], request_json)) for candidate in candidates]


def fetch_all_documents(
    request_snapshot_page: Callable[
        [str | None], tuple[list[dict[str, Any]], str | None, bool]
    ] = _request_snapshot_page,
    request_json: Callable[[str], dict[str, Any]] = _request_json,
) -> list[NotubizDocument]:
    """Retrieve every current document from the bulk snapshot feed."""
    cursor = None
    entity_ids: list[str] = []
    while True:
        records, cursor, has_more = request_snapshot_page(cursor)
        entity_ids.extend(
            record["entity_id"]
            for record in records
            if record.get("op") == "upsert"
            and record.get("entity_type") == "Document"
            and isinstance(record.get("entity_id"), str)
        )
        if not has_more:
            break
        if not cursor:
            msg = "Snapshot response indicated more records but did not provide X-Next-Cursor"
            raise RuntimeError(msg)

    documents = []
    for entity_id in entity_ids:
        try:
            documents.append(normalize_document(get_document_detail(entity_id, request_json)))
        except HTTPError as error:
            if error.code != 404:
                raise
            logger.warning("Skipping snapshot document %s because its detail endpoint returned 404.", entity_id)
    return documents
