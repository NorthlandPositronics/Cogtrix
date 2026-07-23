"""Tests for cogtrix_core/audit.py — structured audit log module."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cogtrix_core.audit as audit_module
from cogtrix_core.audit import (
    AuditLogger,
    configure_audit,
    record_auth,
    record_config_change,
    record_system,
    record_tool_call,
    record_user_action,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.log"


@pytest.fixture()
def logger(log_path: Path) -> AuditLogger:
    return AuditLogger(log_path, enabled=True)


@pytest.fixture(autouse=True)
def reset_global_audit():
    """Ensure module-level singleton is reset after each test."""
    original = audit_module._audit
    yield
    audit_module._audit = original


# ---------------------------------------------------------------------------
# 1. AuditLogger basic write
# ---------------------------------------------------------------------------


def test_logger_writes_ndjson_line(logger: AuditLogger, log_path: Path) -> None:
    logger.log("tool_call", "write_file", actor="cli")
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["category"] == "tool_call"
    assert record["action"] == "write_file"
    assert record["actor"] == "cli"
    assert record["status"] == "ok"


def test_logger_writes_multiple_events(logger: AuditLogger, log_path: Path) -> None:
    for i in range(5):
        logger.log("user_action", f"action_{i}", actor="user1")
    lines = log_path.read_text().splitlines()
    assert len(lines) == 5


def test_logger_records_duration_ms(logger: AuditLogger, log_path: Path) -> None:
    logger.log("tool_call", "shell", actor="cli", duration_ms=123)
    record = json.loads(log_path.read_text().strip())
    assert record["duration_ms"] == 123


def test_logger_records_detail(logger: AuditLogger, log_path: Path) -> None:
    logger.log("auth", "login", actor="alice", detail={"reason": "bad_password"}, status="error")
    record = json.loads(log_path.read_text().strip())
    assert record["status"] == "error"
    assert record["detail"]["reason"] == "bad_password"


# ---------------------------------------------------------------------------
# 2. AuditEvent dataclass
# ---------------------------------------------------------------------------


def test_audit_event_has_uuid(logger: AuditLogger, log_path: Path) -> None:
    logger.log("system", "startup", actor="system")
    record = json.loads(log_path.read_text().strip())
    event_id = record["event_id"]
    # UUID4 format: 8-4-4-4-12 hex characters
    assert len(event_id) == 36
    assert event_id.count("-") == 4


def test_audit_event_timestamp_is_utc_iso(logger: AuditLogger, log_path: Path) -> None:
    before = datetime.now(UTC)
    logger.log("auth", "logout", actor="u1")
    after = datetime.now(UTC)
    record = json.loads(log_path.read_text().strip())
    ts = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
    assert before <= ts <= after


# ---------------------------------------------------------------------------
# 3. Disabled logger
# ---------------------------------------------------------------------------


def test_disabled_logger_writes_nothing(log_path: Path) -> None:
    disabled = AuditLogger(log_path, enabled=False)
    disabled.log("tool_call", "read_file", actor="cli")
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# 4. tail()
# ---------------------------------------------------------------------------


def test_tail_returns_last_n_events(logger: AuditLogger) -> None:
    for i in range(20):
        logger.log("tool_call", f"tool_{i}", actor="cli")
    results = logger.tail(5)
    assert len(results) == 5
    # Most-recent N lines — last 5 written are tool_15..tool_19
    actions = [e.action for e in results]
    assert "tool_19" in actions


def test_tail_returns_empty_for_missing_file(log_path: Path) -> None:
    fresh = AuditLogger(log_path, enabled=True)
    # No events written, file not yet created
    assert fresh.tail(10) == []


def test_tail_zero_returns_empty(logger: AuditLogger) -> None:
    logger.log("tool_call", "x", actor="cli")
    assert logger.tail(0) == []


# ---------------------------------------------------------------------------
# 5. query()
# ---------------------------------------------------------------------------


def test_query_filter_by_category(logger: AuditLogger) -> None:
    logger.log("tool_call", "write_file", actor="cli")
    logger.log("auth", "login", actor="u1")
    logger.log("tool_call", "read_file", actor="cli")

    results = logger.query(category="auth")
    assert len(results) == 1
    assert results[0].action == "login"


def test_query_filter_by_actor(logger: AuditLogger) -> None:
    logger.log("tool_call", "x", actor="alice")
    logger.log("tool_call", "y", actor="bob")
    logger.log("tool_call", "z", actor="alice")

    results = logger.query(actor="alice")
    assert len(results) == 2
    assert all(e.actor == "alice" for e in results)


def test_query_filter_by_action(logger: AuditLogger) -> None:
    logger.log("tool_call", "write_file", actor="cli")
    logger.log("tool_call", "read_file", actor="cli")
    logger.log("tool_call", "write_file", actor="cli")

    results = logger.query(action="write_file")
    assert len(results) == 2


def test_query_filter_by_since(logger: AuditLogger) -> None:
    logger.log("tool_call", "old_tool", actor="cli")
    cutoff = datetime.now(UTC)
    time.sleep(0.01)
    logger.log("tool_call", "new_tool", actor="cli")

    results = logger.query(since=cutoff)
    assert len(results) == 1
    assert results[0].action == "new_tool"


def test_query_limit(logger: AuditLogger) -> None:
    for i in range(50):
        logger.log("user_action", f"a{i}", actor="u")
    results = logger.query(limit=10)
    assert len(results) == 10


def test_query_empty_file(log_path: Path) -> None:
    fresh = AuditLogger(log_path, enabled=True)
    assert fresh.query() == []


# ---------------------------------------------------------------------------
# 6. Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_produce_valid_ndjson(logger: AuditLogger, log_path: Path) -> None:
    """All lines written by concurrent threads must be valid JSON."""
    errors: list[Exception] = []

    def write_events() -> None:
        for _ in range(20):
            logger.log("tool_call", "concurrent_tool", actor="cli")

    threads = [threading.Thread(target=write_events) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 100
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(exc)
    assert not errors, f"Invalid JSON lines: {errors}"


# ---------------------------------------------------------------------------
# 7. configure_audit + convenience functions
# ---------------------------------------------------------------------------


def test_configure_audit_sets_singleton(tmp_path: Path) -> None:
    p = tmp_path / "audit.log"
    configure_audit(p, enabled=True)
    assert audit_module._audit is not None
    assert audit_module._audit._path == p


def test_convenience_record_tool_call(tmp_path: Path) -> None:
    p = tmp_path / "audit.log"
    configure_audit(p, enabled=True)
    record_tool_call("my_tool", actor="test-session", duration_ms=50)
    record = json.loads(p.read_text().strip())
    assert record["category"] == "tool_call"
    assert record["action"] == "my_tool"
    assert record["duration_ms"] == 50


def test_convenience_record_auth(tmp_path: Path) -> None:
    p = tmp_path / "audit.log"
    configure_audit(p, enabled=True)
    record_auth("login", actor="user42", status="error", detail={"reason": "bad_password"})
    record = json.loads(p.read_text().strip())
    assert record["category"] == "auth"
    assert record["status"] == "error"


def test_convenience_record_config_change(tmp_path: Path) -> None:
    p = tmp_path / "audit.log"
    configure_audit(p, enabled=True)
    record_config_change("reload_config", actor="admin1")
    record = json.loads(p.read_text().strip())
    assert record["category"] == "config_change"
    assert record["action"] == "reload_config"


def test_convenience_record_user_action(tmp_path: Path) -> None:
    p = tmp_path / "audit.log"
    configure_audit(p, enabled=True)
    record_user_action("session_create", actor="user7", detail={"session_id": "abc"})
    record = json.loads(p.read_text().strip())
    assert record["category"] == "user_action"
    assert record["detail"]["session_id"] == "abc"


def test_convenience_noop_when_not_configured(tmp_path: Path) -> None:
    """Convenience functions must not raise when audit is not configured."""
    audit_module._audit = None
    record_tool_call("x")
    record_auth("login")
    record_config_change("update_config")
    record_user_action("session_create")
    record_system("startup")
    # No exception = pass


# ---------------------------------------------------------------------------
# 8. Config fields
# ---------------------------------------------------------------------------


def test_config_has_audit_fields() -> None:
    from cogtrix_core.config import Config

    cfg = Config()
    assert cfg.audit_log_enabled is True
    assert cfg.audit_log_path == "data/audit/audit.log"


def test_config_audit_yaml_parsing(tmp_path: Path) -> None:
    from cogtrix_core.config import _apply_config_file  # type: ignore[attr-defined]

    cfg_file = tmp_path / "cogtrix.yaml"
    cfg_file.write_text(
        "audit_log:\n  enabled: false\n  path: /tmp/test-audit.log\n",
        encoding="utf-8",
    )

    from cogtrix_core.config import Config

    cfg = Config()
    _apply_config_file(cfg, cfg_file)
    assert cfg.audit_log_enabled is False
    assert cfg.audit_log_path == "/tmp/test-audit.log"
