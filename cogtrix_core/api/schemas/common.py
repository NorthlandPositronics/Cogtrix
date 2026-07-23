"""Shared envelope, error, and pagination models.

Every REST response is wrapped in ``APIResponse[T]``.  Errors always use the
same envelope with ``data: null`` and a structured ``APIError`` object.
Pagination uses cursor-based semantics — never offset.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


def _serialize_datetime(dt: datetime, _info: Any = None) -> str:
    """Serialize datetime to ISO 8601 with Z suffix for UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    s = dt.isoformat()
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


# ---------------------------------------------------------------------------
# Timestamps & IDs
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(UTC)


def ensure_utc(v: datetime) -> datetime:
    """Attach UTC tzinfo to naive datetimes returned by SQLite.

    SQLite stores datetimes as plain strings and SQLAlchemy returns them
    without tzinfo even when the column uses ``DateTime(timezone=True)``.
    Pydantic then serialises them without the ``Z`` suffix, causing
    JavaScript's ``new Date()`` to interpret them as local time instead of
    UTC.  This validator is applied to all ``*Out`` schemas that carry
    DB-sourced datetime fields.
    """
    if v is not None and v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


def _new_request_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Error object
# ---------------------------------------------------------------------------


class APIError(BaseModel):
    """Structured error payload included in every error response envelope.

    ``code`` is a machine-readable constant (SCREAMING_SNAKE_CASE).
    ``message`` is a human-readable description safe to display in the UI.
    ``details`` carries optional extra context (validation field errors, etc.).
    """

    code: str = Field(
        ...,
        description="Machine-readable error code (e.g. SESSION_NOT_FOUND).",
        examples=["SESSION_NOT_FOUND"],
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
        examples=["The requested session does not exist."],
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra context (validation errors, upstream error, etc.).",
    )


# ---------------------------------------------------------------------------
# Response meta
# ---------------------------------------------------------------------------


class ResponseMeta(BaseModel):
    """Metadata attached to every API response.

    ``request_id`` is a UUID v4 that the frontend can log for support traces.
    ``timestamp`` is the server-side UTC moment the response was generated.
    """

    request_id: str = Field(
        default_factory=_new_request_id,
        description="UUID v4 identifying this specific request.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    timestamp: datetime = Field(
        default_factory=_now_utc,
        description="UTC timestamp of the response in ISO 8601 format.",
        examples=["2026-03-04T12:34:56.789Z"],
    )


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class APIResponse(BaseModel, Generic[T]):
    """Standard response envelope for all REST endpoints.

    Success: ``data`` is populated, ``error`` is null.
    Failure: ``data`` is null, ``error`` is populated.
    ``meta`` is always present.

    The generic type parameter ``T`` constrains ``data`` in typed contexts.
    """

    model_config = ConfigDict(
        json_encoders={datetime: _serialize_datetime},
    )

    data: T | None = Field(
        default=None,
        description="Response payload on success; null on error.",
    )
    error: APIError | None = Field(
        default=None,
        description="Error object on failure; null on success.",
    )
    meta: ResponseMeta = Field(
        default_factory=ResponseMeta,
        description="Request metadata (request_id, timestamp).",
    )


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class CursorPage(BaseModel, Generic[T]):
    """Cursor-based paginated list result.

    Use ``next_cursor`` as the ``cursor`` query parameter in the subsequent
    request.  When ``has_more`` is False, no further pages exist.
    """

    items: list[T] = Field(
        ...,
        description="Items on the current page.",
    )
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor to pass as ?cursor= for the next page; null when on last page.",
    )
    has_more: bool = Field(
        ...,
        description="True when additional pages are available beyond this one.",
    )
    total: int | None = Field(
        default=None,
        description="Total item count across all pages (optional; may be null when expensive).",
    )


# ---------------------------------------------------------------------------
# Canonical error codes
# ---------------------------------------------------------------------------

ERROR_CODES = {
    # Auth
    "UNAUTHORIZED": "Missing or invalid bearer token.",
    "TOKEN_EXPIRED": "The JWT has expired; refresh the token and retry.",
    "FORBIDDEN": "Authenticated user lacks permission for this action.",
    # Resource
    "NOT_FOUND": "The requested resource does not exist.",
    "SESSION_NOT_FOUND": "The requested session does not exist.",
    "MESSAGE_NOT_FOUND": "The requested message does not exist.",
    "TOOL_NOT_FOUND": "The requested tool does not exist in the registry.",
    "DOCUMENT_NOT_FOUND": "The requested RAG document does not exist.",
    "MCP_SERVER_NOT_FOUND": "The requested MCP server is not configured.",
    "FACT_NOT_FOUND": "The requested knowledge fact does not exist.",
    # Validation
    "VALIDATION_ERROR": "Request body or query parameter validation failed.",
    "INVALID_CURSOR": "The pagination cursor is malformed or expired.",
    # Conflict
    "SESSION_ALREADY_EXISTS": "A session with this ID already exists.",
    "SESSION_NAME_DUPLICATE": "A session with this name already exists for this user.",
    "TOOL_ALREADY_ACTIVE": "The tool is already in the active set.",
    "TOOL_ALREADY_DISABLED": "The tool is already disabled.",
    # Business logic
    "TOOL_EXPANSION_FAILED": "The requested tool could not be loaded.",
    "MODEL_UNAVAILABLE": "The requested model is not available from the provider.",
    "PROVIDER_UNREACHABLE": "Cannot reach the configured LLM provider.",
    "INGEST_FAILED": "Document ingestion failed.",
    "WIZARD_STEP_ERROR": "Setup wizard step could not be completed.",
    "CONFIG_INVALID": "The supplied configuration is invalid.",
    "MEMORY_CLEAR_FAILED": "Memory could not be cleared.",
    "ASSISTANT_ALREADY_RUNNING": "The assistant service is already running.",
    "ASSISTANT_NOT_RUNNING": "The assistant service is not running.",
    "MCP_RESTART_FAILED": "MCP server restart failed.",
    "SCHEDULED_MSG_NOT_FOUND": "The scheduled message does not exist or is not cancellable.",
    "DEFERRED_MSG_NOT_FOUND": "The deferred message record does not exist.",
    # System
    "INTERNAL_ERROR": "An unexpected server error occurred.",
    "NOT_IMPLEMENTED": "This endpoint is not yet implemented.",
}
