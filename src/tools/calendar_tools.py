"""Google Calendar tools — list, create, and search calendar events.

Tools:
    calendar_list_events  — list upcoming events from a calendar
    calendar_create_event — create a new calendar event
    calendar_search_events — full-text search across events

Configuration (services.google_calendar in .cogtrix.yaml):
    credentials_file:     OAuth2 client secrets file path
    token_file:           cached OAuth2 token path (auto-created after first auth)
    service_account_file: service account JSON path (headless auth, takes priority)
    calendar_id:          default calendar ID (default: "primary")

Authentication:
    - If service_account_file is set: use service account credentials (no browser).
    - Otherwise: use OAuth2 with token caching.  Loads token_file if it exists,
      refreshes if expired, and prompts via browser if no valid token is found.

TOOL_SETUP(config) is called automatically by ToolRegistry after this
module is loaded.  Do not call configure_calendar_tools() from configure.py.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Config

try:
    from googleapiclient.discovery import build as _build  # pyright: ignore[reportMissingImports]  # fmt: skip  # noqa: I001
    from googleapiclient.errors import HttpError  # pyright: ignore[reportMissingImports]

    _HAS_GOOGLE = True
except ImportError:  # pragma: no cover
    _HAS_GOOGLE = False
    HttpError = Exception  # type: ignore[assignment,misc]

log = logging.getLogger("cogtrix.tools.calendar")

_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ── Module-level state (set by configure_calendar_tools) ──────────────────────

_client: Any = None  # googleapiclient Resource, or None when unconfigured
_calendar_id: str = "primary"
_gcal_config: dict[str, Any] = {}

# ── Error message constants ───────────────────────────────────────────────────

_NOT_CONFIGURED_MSG = (
    "Google Calendar is not configured. Add services.google_calendar to .cogtrix.yaml"
)
_INSTALL_HINT = "Google Calendar dependencies not installed. " "Run: uv add 'cogtrix[google]'"


# ── Authentication helpers ────────────────────────────────────────────────────


def _expand(path: str) -> str:
    """Expand ~ and resolve symlinks in a path string."""
    return str(Path(os.path.expanduser(path)).resolve())


def _build_credentials(gcal: dict[str, Any]) -> Any:
    """Build Google credentials from config dict.

    Tries service account first, then OAuth2 with token caching.
    Returns a credentials object, or None if configuration is incomplete.
    """
    sa_file = gcal.get("service_account_file", "")
    creds_file = gcal.get("credentials_file", "")
    token_file = gcal.get("token_file", "")

    if sa_file:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_file(
            _expand(sa_file), scopes=_SCOPES
        )

    if not creds_file:
        return None

    creds = None
    if token_file:
        token_path = Path(_expand(token_file))
        if token_path.exists():
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow  # pyright: ignore[reportMissingImports]  # fmt: skip  # noqa: I001

            flow = InstalledAppFlow.from_client_secrets_file(_expand(creds_file), _SCOPES)
            creds = flow.run_local_server(port=0)

        if token_file:
            token_path = Path(_expand(token_file))
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


# ── Configuration ─────────────────────────────────────────────────────────────


def configure_calendar_tools(gcal: dict[str, Any]) -> None:
    """Set runtime configuration from the services.google_calendar dict."""
    global _client, _calendar_id, _gcal_config
    _gcal_config = {**gcal}
    _calendar_id = gcal.get("calendar_id", "primary") or "primary"

    if not _HAS_GOOGLE:
        return
    if not (gcal.get("credentials_file") or gcal.get("service_account_file")):
        return

    try:
        creds = _build_credentials(gcal)
        if creds is not None:
            _client = _build("calendar", "v3", credentials=creds)
    except Exception as exc:
        log.warning("Failed to initialise Google Calendar client: %s", exc)
        _client = None


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    svc = getattr(config, "services", {}) or {}
    gcal = svc.get("google_calendar", {}) or {}
    configure_calendar_tools(gcal)


def is_configured() -> bool:
    """Return True when Google libraries are installed and credentials are configured."""
    if not _HAS_GOOGLE:
        return False
    return bool(_gcal_config.get("credentials_file") or _gcal_config.get("service_account_file"))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _format_events(items: list[dict[str, Any]]) -> str:
    """Format a list of event dicts as a human-readable string."""
    lines: list[str] = []
    for event in items:
        start_obj = event.get("start", {})
        start = start_obj.get("dateTime") or start_obj.get("date") or "?"
        summary = event.get("summary", "(no title)")
        event_id = (event.get("id") or "")[:8]
        lines.append(f"{start} | {summary} | {event_id}")
    return "\n".join(lines) if lines else "No events found."


# ── Input schemas ─────────────────────────────────────────────────────────────


class CalendarListEventsInput(BaseModel):
    calendar_id: str = Field(
        default="",
        description=(
            "Calendar ID to query. Defaults to the configured calendar (usually 'primary')."
        ),
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of events to return (1–250).",
    )
    time_min: str = Field(
        default="",
        description=(
            "Lower bound (inclusive) for event start time. "
            "ISO 8601 string, e.g. '2026-03-29T00:00:00Z'. Empty = now."
        ),
    )
    time_max: str = Field(
        default="",
        description=(
            "Upper bound (exclusive) for event start time. " "ISO 8601 string. Empty = unbounded."
        ),
    )


class CalendarCreateEventInput(BaseModel):
    summary: str = Field(..., description="Title / summary of the event.")
    start: str = Field(..., description="Start datetime in ISO 8601, e.g. '2026-04-01T10:00:00Z'.")
    end: str = Field(..., description="End datetime in ISO 8601, e.g. '2026-04-01T11:00:00Z'.")
    description: str = Field(default="", description="Optional event description / notes.")
    calendar_id: str = Field(
        default="",
        description="Calendar to add the event to. Defaults to the configured calendar.",
    )
    attendees: str = Field(
        default="",
        description="Comma-separated email addresses of attendees (optional).",
    )


class CalendarSearchEventsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Free-text search query across event titles, descriptions, " "and attendee names."
        ),
    )
    calendar_id: str = Field(
        default="",
        description="Calendar to search. Defaults to the configured calendar.",
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (1–250).",
    )


# ── Tool functions ────────────────────────────────────────────────────────────


def calendar_list_events(
    calendar_id: str = "",
    max_results: int = 10,
    time_min: str = "",
    time_max: str = "",
) -> str:
    """List upcoming events from a Google Calendar."""
    if not _HAS_GOOGLE:
        return _INSTALL_HINT
    if _client is None:
        return _NOT_CONFIGURED_MSG

    cal_id = calendar_id or _calendar_id
    kwargs: dict[str, Any] = {
        "calendarId": cal_id,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if time_min:
        kwargs["timeMin"] = time_min
    else:
        kwargs["timeMin"] = datetime.now(UTC).isoformat()
    if time_max:
        kwargs["timeMax"] = time_max

    try:
        result = _client.events().list(**kwargs).execute()
        items = result.get("items", [])
        return _format_events(items)
    except HttpError as exc:
        return f"Google Calendar API error: {exc}"


def calendar_create_event(
    summary: str,
    start: str,
    end: str,
    description: str = "",
    calendar_id: str = "",
    attendees: str = "",
) -> str:
    """Create a new event on a Google Calendar."""
    if not _HAS_GOOGLE:
        return _INSTALL_HINT
    if _client is None:
        return _NOT_CONFIGURED_MSG

    cal_id = calendar_id or _calendar_id
    body: dict[str, Any] = {
        "summary": summary,
        "start": {"dateTime": start, "timeZone": "UTC"},
        "end": {"dateTime": end, "timeZone": "UTC"},
    }
    if description:
        body["description"] = description
    if attendees:
        body["attendees"] = [{"email": e.strip()} for e in attendees.split(",") if e.strip()]

    try:
        event = _client.events().insert(calendarId=cal_id, body=body).execute()
        event_id = (event.get("id") or "")[:8]
        return f"Event created: {summary} [{event_id}]"
    except HttpError as exc:
        return f"Google Calendar API error: {exc}"


def calendar_search_events(
    query: str,
    calendar_id: str = "",
    max_results: int = 10,
) -> str:
    """Search for events in a Google Calendar using full-text search."""
    if not _HAS_GOOGLE:
        return _INSTALL_HINT
    if _client is None:
        return _NOT_CONFIGURED_MSG

    cal_id = calendar_id or _calendar_id
    try:
        result = (
            _client.events()
            .list(
                calendarId=cal_id,
                q=query,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        items = result.get("items", [])
        return _format_events(items)
    except HttpError as exc:
        return f"Google Calendar API error: {exc}"


# ── Tool registry entries ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "calendar_list_events",
        "description": (
            "List upcoming events from Google Calendar. "
            "Optionally filter by time range or specify a calendar ID. "
            "Returns each event as: start | summary | id."
        ),
        "input_schema": CalendarListEventsInput,
        "requires_confirmation": False,
        "function": calendar_list_events,
    },
    {
        "name": "calendar_create_event",
        "description": (
            "Create a new event on Google Calendar. "
            "Provide a title, start/end in ISO 8601, and optional description and attendees."
        ),
        "input_schema": CalendarCreateEventInput,
        "requires_confirmation": True,
        "function": calendar_create_event,
    },
    {
        "name": "calendar_search_events",
        "description": (
            "Full-text search across Google Calendar events. "
            "Searches event titles, descriptions, and attendee names."
        ),
        "input_schema": CalendarSearchEventsInput,
        "requires_confirmation": False,
        "function": calendar_search_events,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "TOOL_SETUP",
    "configure_calendar_tools",
    "is_configured",
    "calendar_list_events",
    "calendar_create_event",
    "calendar_search_events",
    "CalendarListEventsInput",
    "CalendarCreateEventInput",
    "CalendarSearchEventsInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
