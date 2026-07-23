"""Tests for cogtrix_core/tools/calendar_tools.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cogtrix_core.tools.calendar_tools as _mod
from cogtrix_core.tools.calendar_tools import (
    TOOL_CONFIGS,
    CalendarCreateEventInput,
    CalendarListEventsInput,
    CalendarSearchEventsInput,
    configure_calendar_tools,
    is_configured,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_fake_client(items: list[dict] | None = None) -> MagicMock:
    """Return a mock googleapiclient Resource with a stub events() chain."""
    if items is None:
        items = []
    result = {"items": items}
    execute = MagicMock(return_value=result)
    list_call = MagicMock(return_value=MagicMock(execute=execute))
    insert_call = MagicMock(
        return_value=MagicMock(execute=MagicMock(return_value={"id": "abc12345def"}))
    )
    events = MagicMock()
    events.return_value = MagicMock(list=list_call, insert=insert_call)
    client = MagicMock()
    client.events = events
    return client


def _fake_config(
    credentials_file: str = "",
    service_account_file: str = "",
    calendar_id: str = "primary",
) -> SimpleNamespace:
    """Minimal Config-like object with services.google_calendar section."""
    gcal: dict = {}
    if credentials_file:
        gcal["credentials_file"] = credentials_file
    if service_account_file:
        gcal["service_account_file"] = service_account_file
    if calendar_id != "primary":
        gcal["calendar_id"] = calendar_id
    return SimpleNamespace(services={"google_calendar": gcal})


# ── is_configured ─────────────────────────────────────────────────────────────


def test_is_configured_false_when_no_credentials() -> None:
    configure_calendar_tools({})
    assert is_configured() is False


def test_is_configured_true_when_credentials_file_set() -> None:
    with patch.object(_mod, "_HAS_GOOGLE", True):
        configure_calendar_tools({"credentials_file": "/path/to/creds.json"})
    # Restore clean state so other tests are not affected
    configure_calendar_tools({})
    # Check using the flag state that was set during the patch
    with patch.object(_mod, "_HAS_GOOGLE", True):
        configure_calendar_tools({"credentials_file": "/path/to/creds.json"})
        assert is_configured() is True
    configure_calendar_tools({})


def test_is_configured_true_when_service_account_file_set() -> None:
    with patch.object(_mod, "_HAS_GOOGLE", True):
        configure_calendar_tools({"service_account_file": "/path/to/sa.json"})
        assert is_configured() is True
    configure_calendar_tools({})


def test_is_configured_false_when_google_not_installed() -> None:
    with patch.object(_mod, "_HAS_GOOGLE", False):
        configure_calendar_tools({"credentials_file": "/path/to/creds.json"})
        assert is_configured() is False
    configure_calendar_tools({})


# ── TOOL_SETUP ────────────────────────────────────────────────────────────────


def test_tool_setup_wires_state_from_config() -> None:
    config = _fake_config(calendar_id="work@group.calendar.google.com")
    # configure_calendar_tools won't build a real client (no creds), but it
    # should still store the calendar_id
    with patch.object(_mod, "_HAS_GOOGLE", True):
        _mod.TOOL_SETUP(config)  # type: ignore[arg-type]
    assert _mod._calendar_id == "work@group.calendar.google.com"
    configure_calendar_tools({})


def test_tool_setup_uses_primary_when_calendar_id_missing() -> None:
    config = _fake_config()
    _mod.TOOL_SETUP(config)  # type: ignore[arg-type]
    assert _mod._calendar_id == "primary"


# ── TOOL_CONFIGS ──────────────────────────────────────────────────────────────


def test_tool_configs_has_three_entries() -> None:
    assert len(TOOL_CONFIGS) == 3


def test_tool_configs_names() -> None:
    names = {c["name"] for c in TOOL_CONFIGS}
    assert names == {"calendar_list_events", "calendar_create_event", "calendar_search_events"}


def test_calendar_create_event_requires_confirmation() -> None:
    entry = next(c for c in TOOL_CONFIGS if c["name"] == "calendar_create_event")
    assert entry["requires_confirmation"] is True


def test_calendar_list_events_no_confirmation() -> None:
    entry = next(c for c in TOOL_CONFIGS if c["name"] == "calendar_list_events")
    assert entry["requires_confirmation"] is False


# ── client None guard ─────────────────────────────────────────────────────────


def test_list_events_returns_not_configured_when_client_is_none() -> None:
    with patch.object(_mod, "_client", None), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_list_events()
    assert "not configured" in result.lower() or "google calendar" in result.lower()


def test_create_event_returns_not_configured_when_client_is_none() -> None:
    with patch.object(_mod, "_client", None), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_create_event(
            "Meeting", "2026-04-01T10:00:00Z", "2026-04-01T11:00:00Z"
        )
    assert "not configured" in result.lower() or "google calendar" in result.lower()


def test_search_events_returns_not_configured_when_client_is_none() -> None:
    with patch.object(_mod, "_client", None), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_search_events("budget")
    assert "not configured" in result.lower() or "google calendar" in result.lower()


def test_list_events_returns_install_hint_when_google_not_installed() -> None:
    with patch.object(_mod, "_HAS_GOOGLE", False):
        result = _mod.calendar_list_events()
    assert "install" in result.lower() or "not installed" in result.lower()


# ── calendar_list_events ──────────────────────────────────────────────────────


def test_list_events_returns_formatted_event_list() -> None:
    items = [
        {"start": {"dateTime": "2026-04-01T10:00:00Z"}, "summary": "Team sync", "id": "ev001abc"},
        {"start": {"date": "2026-04-02"}, "summary": "All-day event", "id": "ev002abc"},
    ]
    client = _make_fake_client(items)
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_list_events()
    assert "Team sync" in result
    assert "All-day event" in result
    assert "2026-04-01T10:00:00Z" in result


def test_list_events_returns_no_events_found_for_empty_response() -> None:
    client = _make_fake_client([])
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_list_events()
    assert result == "No events found."


def test_list_events_uses_config_calendar_id_as_default() -> None:
    client = _make_fake_client([])
    with (
        patch.object(_mod, "_client", client),
        patch.object(_mod, "_HAS_GOOGLE", True),
        patch.object(_mod, "_calendar_id", "team@group.calendar.google.com"),
    ):
        _mod.calendar_list_events()
    call_kwargs = client.events.return_value.list.call_args[1]
    assert call_kwargs["calendarId"] == "team@group.calendar.google.com"


def test_list_events_explicit_calendar_id_overrides_default() -> None:
    client = _make_fake_client([])
    with (
        patch.object(_mod, "_client", client),
        patch.object(_mod, "_HAS_GOOGLE", True),
        patch.object(_mod, "_calendar_id", "default@example.com"),
    ):
        _mod.calendar_list_events(calendar_id="other@example.com")
    call_kwargs = client.events.return_value.list.call_args[1]
    assert call_kwargs["calendarId"] == "other@example.com"


def test_list_events_http_error_returns_error_string() -> None:
    """HttpError is sanitized — no raw exception or API key leaks to the LLM."""
    import json

    try:
        from googleapiclient.errors import HttpError  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("googleapiclient not installed")

    client = MagicMock()
    # Build a real HttpError with a 403 response
    resp = SimpleNamespace(status=403, reason="Forbidden")
    body = json.dumps({"error": {"message": "Rate limit exceeded", "code": 403}}).encode()
    fake_error = HttpError(resp, body)
    client.events.return_value.list.return_value.execute.side_effect = fake_error
    with (
        patch.object(_mod, "_client", client),
        patch.object(_mod, "_HAS_GOOGLE", True),
        patch.object(_mod, "HttpError", HttpError),
    ):
        result = _mod.calendar_list_events()
    # Result must contain "error" (new format) or status code — both satisfy the assertion
    assert "error" in result.lower() or "403" in result
    # Result must follow the safe pattern: status code followed by (category)
    assert "403 (" in result


# ── calendar_create_event ─────────────────────────────────────────────────────


def test_create_event_returns_confirmation_string() -> None:
    client = _make_fake_client()
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_create_event(
            summary="Sprint review",
            start="2026-04-05T14:00:00Z",
            end="2026-04-05T15:00:00Z",
        )
    assert "Sprint review" in result
    assert "created" in result.lower()


def test_create_event_body_has_correct_fields() -> None:
    client = _make_fake_client()
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        _mod.calendar_create_event(
            summary="Budget meeting",
            start="2026-04-01T09:00:00Z",
            end="2026-04-01T10:00:00Z",
            description="Q2 planning",
        )
    insert_call = client.events.return_value.insert
    body = insert_call.call_args[1]["body"]
    assert body["summary"] == "Budget meeting"
    assert body["start"]["dateTime"] == "2026-04-01T09:00:00Z"
    assert body["end"]["dateTime"] == "2026-04-01T10:00:00Z"
    assert body["description"] == "Q2 planning"


def test_create_event_with_attendees_passes_attendees_list() -> None:
    client = _make_fake_client()
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        _mod.calendar_create_event(
            summary="Planning",
            start="2026-04-01T10:00:00Z",
            end="2026-04-01T11:00:00Z",
            attendees="alice@example.com, bob@example.com",
        )
    insert_call = client.events.return_value.insert
    body = insert_call.call_args[1]["body"]
    assert {"email": "alice@example.com"} in body["attendees"]
    assert {"email": "bob@example.com"} in body["attendees"]


def test_create_event_http_error_returns_error_string() -> None:
    """HttpError is sanitized — no raw exception or API key leaks to the LLM."""
    import json

    try:
        from googleapiclient.errors import HttpError  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("googleapiclient not installed")

    client = MagicMock()
    # Build a real HttpError with a 409 response
    resp = SimpleNamespace(status=409, reason="Conflict")
    body = json.dumps({"error": {"message": "Conflict", "code": 409}}).encode()
    fake_error = HttpError(resp, body)
    client.events.return_value.insert.return_value.execute.side_effect = fake_error
    with (
        patch.object(_mod, "_client", client),
        patch.object(_mod, "_HAS_GOOGLE", True),
        patch.object(_mod, "HttpError", HttpError),
    ):
        result = _mod.calendar_create_event("X", "2026-04-01T10:00:00Z", "2026-04-01T11:00:00Z")
    # Result must contain "error" or status code — both satisfy the assertion
    assert "error" in result.lower() or "409" in result
    # Result must follow the safe pattern: status code followed by (category)
    assert "409 (" in result


# ── calendar_search_events ────────────────────────────────────────────────────


def test_search_events_calls_list_with_q_parameter() -> None:
    client = _make_fake_client([])
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        _mod.calendar_search_events(query="budget review")
    call_kwargs = client.events.return_value.list.call_args[1]
    assert call_kwargs["q"] == "budget review"


def test_search_events_returns_formatted_results() -> None:
    items = [
        {
            "start": {"dateTime": "2026-04-10T15:00:00Z"},
            "summary": "Budget review",
            "id": "srch001x",
        }
    ]
    client = _make_fake_client(items)
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_search_events(query="budget")
    assert "Budget review" in result


def test_search_events_returns_no_events_found_for_empty() -> None:
    client = _make_fake_client([])
    with patch.object(_mod, "_client", client), patch.object(_mod, "_HAS_GOOGLE", True):
        result = _mod.calendar_search_events(query="nonexistent")
    assert result == "No events found."


# ── input schemas ─────────────────────────────────────────────────────────────


def test_calendar_list_events_input_defaults() -> None:
    m = CalendarListEventsInput()
    assert m.calendar_id == ""
    assert m.max_results == 10
    assert m.time_min == ""
    assert m.time_max == ""


def test_calendar_create_event_input_required_fields() -> None:
    m = CalendarCreateEventInput(
        summary="X", start="2026-04-01T10:00:00Z", end="2026-04-01T11:00:00Z"
    )
    assert m.summary == "X"
    assert m.description == ""
    assert m.attendees == ""


def test_calendar_search_events_input_defaults() -> None:
    m = CalendarSearchEventsInput(query="test")
    assert m.max_results == 10
    assert m.calendar_id == ""
