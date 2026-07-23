"""Structured audit log for tool calls, user actions, config changes, and auth events.

All audit writes are thread-safe and append-only.  The log is stored as NDJSON
(newline-delimited JSON) so it can be tailed, grepped, and streamed efficiently.

Public API
----------
configure_audit(path, enabled)   — call once at startup
record_tool_call(...)            — called by ToolCallLogger
record_user_action(...)          — called by API route handlers
record_config_change(...)        — called by config route handlers
record_auth(...)                 — called by auth route handlers
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("cogtrix.audit")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

CATEGORY_TOOL_CALL = "tool_call"
CATEGORY_USER_ACTION = "user_action"
CATEGORY_CONFIG_CHANGE = "config_change"
CATEGORY_AUTH = "auth"
CATEGORY_SYSTEM = "system"


@dataclass
class AuditEvent:
    """A single structured audit record."""

    event_id: str
    timestamp: str  # ISO 8601 UTC
    category: str
    action: str
    actor: str
    status: str  # "ok" | "error" | "denied"
    detail: dict[str, Any]
    duration_ms: int | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Thread-safe append-only NDJSON audit logger with query and tail support."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, event: AuditEvent) -> None:
        """Append one audit event to the log file."""
        if not self._enabled:
            return
        line = json.dumps(asdict(event), ensure_ascii=False)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                log.warning("audit: failed to write event %s", event.event_id)

    def _make_event(
        self,
        category: str,
        action: str,
        actor: str,
        status: str,
        detail: dict[str, Any] | None,
        duration_ms: int | None,
    ) -> AuditEvent:
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=_now_iso(),
            category=category,
            action=action,
            actor=actor,
            status=status,
            detail=detail or {},
            duration_ms=duration_ms,
        )

    def log(
        self,
        category: str,
        action: str,
        actor: str,
        status: str = "ok",
        detail: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.record(self._make_event(category, action, actor, status, detail, duration_ms))

    # ------------------------------------------------------------------
    # Read / query
    # ------------------------------------------------------------------

    def tail(self, n: int = 100) -> list[AuditEvent]:
        """Return the last *n* events without reading the whole file."""
        if not self._path.exists():
            return []
        try:
            lines = _tail_lines(self._path, n)
        except OSError:
            return []
        events: list[AuditEvent] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                events.append(_dict_to_event(d))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return events

    def query(
        self,
        *,
        category: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return up to *limit* matching events (most-recent first)."""
        if not self._path.exists():
            return []
        results: list[AuditEvent] = []
        try:
            with self._path.open(encoding="utf-8") as fh:
                raw_lines = fh.readlines()
        except OSError:
            return []

        for line in reversed(raw_lines):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                evt = _dict_to_event(d)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            if category is not None and evt.category != category:
                continue
            if actor is not None and evt.actor != actor:
                continue
            if action is not None and evt.action != action:
                continue
            if since is not None:
                try:
                    evt_dt = datetime.fromisoformat(evt.timestamp.replace("Z", "+00:00"))
                    if evt_dt < since:
                        continue
                except ValueError:
                    pass

            results.append(evt)
            if len(results) >= limit:
                break

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tail_lines(path: Path, n: int) -> list[str]:
    """Efficiently read the last *n* lines of a file."""
    if n <= 0:
        return []
    chunk = 8192
    raw_lines: list[bytes] = []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        buf = b""
        pos = size
        while pos > 0 and len(raw_lines) < n + 1:
            read_size = min(chunk, pos)
            pos -= read_size
            fh.seek(pos)
            buf = fh.read(read_size) + buf
            raw_lines = buf.split(b"\n")
        # Last element may be a partial line at file end; trim trailing empty
        if raw_lines and raw_lines[-1] == b"":
            raw_lines = raw_lines[:-1]
    return [ln.decode("utf-8", errors="replace") for ln in raw_lines[-n:]]


def _dict_to_event(d: dict[str, Any]) -> AuditEvent:
    return AuditEvent(
        event_id=d.get("event_id", ""),
        timestamp=d.get("timestamp", ""),
        category=d.get("category", ""),
        action=d.get("action", ""),
        actor=d.get("actor", ""),
        status=d.get("status", "ok"),
        detail=d.get("detail") or {},
        duration_ms=d.get("duration_ms"),
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_audit: AuditLogger | None = None
_audit_lock = threading.Lock()


def configure_audit(path: Path | str, *, enabled: bool = True) -> None:
    """Initialise the module-level audit logger.  Call once at startup."""
    global _audit
    with _audit_lock:
        _audit = AuditLogger(Path(path), enabled=enabled)
    log.debug("audit: configured path=%s enabled=%s", path, enabled)


def _get_audit() -> AuditLogger | None:
    return _audit


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def record_tool_call(
    tool_name: str,
    *,
    actor: str = "cli",
    status: str = "ok",
    detail: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> None:
    a = _get_audit()
    if a is None:
        return
    a.log(
        CATEGORY_TOOL_CALL,
        tool_name,
        actor=actor,
        status=status,
        detail=detail,
        duration_ms=duration_ms,
    )


def record_user_action(
    action: str,
    *,
    actor: str = "unknown",
    status: str = "ok",
    detail: dict[str, Any] | None = None,
) -> None:
    a = _get_audit()
    if a is None:
        return
    a.log(CATEGORY_USER_ACTION, action, actor=actor, status=status, detail=detail)


def record_config_change(
    action: str,
    *,
    actor: str = "unknown",
    status: str = "ok",
    detail: dict[str, Any] | None = None,
) -> None:
    a = _get_audit()
    if a is None:
        return
    a.log(CATEGORY_CONFIG_CHANGE, action, actor=actor, status=status, detail=detail)


def record_auth(
    action: str,
    *,
    actor: str = "unknown",
    status: str = "ok",
    detail: dict[str, Any] | None = None,
) -> None:
    a = _get_audit()
    if a is None:
        return
    a.log(CATEGORY_AUTH, action, actor=actor, status=status, detail=detail)


def record_system(
    action: str,
    *,
    actor: str = "system",
    status: str = "ok",
    detail: dict[str, Any] | None = None,
) -> None:
    a = _get_audit()
    if a is None:
        return
    a.log(CATEGORY_SYSTEM, action, actor=actor, status=status, detail=detail)
