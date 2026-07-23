"""Tests for src/tools/email_tools.py — mock imaplib and smtplib."""

from __future__ import annotations

import imaplib
from unittest.mock import MagicMock

import pytest

import src.tools.email_tools as _mod
from src.tools.email_tools import (
    configure_email,
    is_configured,
    read_email,
    search_email,
    send_email,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

RAW_MESSAGE = (
    b"From: sender@example.com\r\n"
    b"To: you@example.com\r\n"
    b"Subject: Test subject\r\n"
    b"Date: Mon, 01 Jan 2024 10:00:00 +0000\r\n"
    b"\r\n"
    b"Hello, this is the body."
)

_CREDS = {
    "imap_host": "imap.example.com",
    "smtp_host": "smtp.example.com",
    "username": "user@example.com",
    "password": "secret",
}


def _imap_fetch_response(uid: bytes, raw: bytes) -> list:
    """Build an imaplib fetch response: [(b'UID (RFC822 {n}', raw), b')']."""
    header = f"{uid.decode()} (RFC822 {{{len(raw)}}}".encode()
    return [(header, raw), b")"]


def _make_imap_cls(uids: list[bytes] | None = None, raw: bytes = RAW_MESSAGE) -> MagicMock:
    """Return a mock IMAP4_SSL *class* (callable) with pre-programmed responses."""
    conn = MagicMock()
    uid_list = uids if uids is not None else [b"1", b"2", b"3"]
    conn.select.return_value = ("OK", [b"3"])
    conn.search.return_value = ("OK", [b" ".join(uid_list) if uid_list else b""])
    if uid_list:
        conn.fetch.return_value = ("OK", _imap_fetch_response(uid_list[-1], raw))
    else:
        conn.fetch.return_value = ("OK", [])
    # Make the class callable and return conn
    cls = MagicMock(return_value=conn)
    cls._conn = conn  # expose for assertions
    return cls


def _make_smtp_conn() -> MagicMock:
    """Return a mock SMTP connection usable as a context manager."""
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ── is_configured ─────────────────────────────────────────────────────────────


def test_is_configured_false_when_empty() -> None:
    _mod._config = {}
    assert is_configured() is False


def test_is_configured_true_when_all_required_fields_set() -> None:
    _mod._config = {}
    configure_email(_CREDS)
    assert is_configured() is True


def test_is_configured_false_when_password_missing() -> None:
    _mod._config = {}
    configure_email({k: v for k, v in _CREDS.items() if k != "password"})
    assert is_configured() is False


# ── configure_email ───────────────────────────────────────────────────────────


def test_configure_email_uses_default_ports_at_runtime() -> None:
    _mod._config = {}
    configure_email(_CREDS)
    # Defaults are applied at use-time via .get(key, default), not stored in _config
    assert _mod._config.get("imap_port", 993) == 993
    assert _mod._config.get("smtp_port", 587) == 587
    assert _mod._config.get("use_tls", True) is True


def test_configure_email_stores_explicit_overrides() -> None:
    _mod._config = {}
    configure_email({**_CREDS, "imap_port": 143, "smtp_port": 465, "use_tls": False})
    assert _mod._config["imap_port"] == 143
    assert _mod._config["smtp_port"] == 465
    assert _mod._config["use_tls"] is False


def test_configure_email_merges_incrementally() -> None:
    _mod._config = {}
    configure_email({"imap_host": "imap.example.com"})
    configure_email({"smtp_host": "smtp.example.com"})
    assert _mod._config["imap_host"] == "imap.example.com"
    assert _mod._config["smtp_host"] == "smtp.example.com"


# ── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset email config to known credentials before each test."""
    _mod._config = {}
    configure_email(_CREDS)
    yield
    _mod._config = {}


# ── read_email ────────────────────────────────────────────────────────────────


def test_read_email_returns_messages(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = read_email(limit=1, folder="INBOX")

    assert "Test subject" in result
    assert "sender@example.com" in result


def test_read_email_no_messages(monkeypatch) -> None:
    imap_cls = _make_imap_cls(uids=[])
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = read_email(limit=5, folder="INBOX")
    assert "No messages" in result


def test_read_email_select_failure(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    imap_cls._conn.select.return_value = ("NO", [b"no such folder"])
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = read_email(limit=5, folder="NoExist")
    assert "Error" in result


def test_read_email_not_configured() -> None:
    _mod._config = {}
    result = read_email(limit=5, folder="INBOX")
    assert "Error" in result or "not configured" in result.lower()


def test_read_email_uses_plain_imap_when_tls_disabled(monkeypatch) -> None:
    _mod._config = {}
    configure_email({**_CREDS, "use_tls": False})
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4", imap_cls)

    result = read_email(limit=1, folder="INBOX")
    imap_cls.assert_called_once()
    assert "Test subject" in result


def test_read_email_clamps_limit(monkeypatch) -> None:
    """limit > 50 should be silently clamped to 50."""
    uids = [str(i).encode() for i in range(1, 100)]
    imap_cls = _make_imap_cls(uids=uids)
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    read_email(limit=200, folder="INBOX")
    fetch_arg = imap_cls._conn.fetch.call_args[0][0]
    assert len(fetch_arg.split(",")) <= 50


# ── send_email ────────────────────────────────────────────────────────────────


def test_send_email_happy_path(monkeypatch) -> None:
    smtp_conn = _make_smtp_conn()
    smtp_cls = MagicMock(return_value=smtp_conn)
    monkeypatch.setattr("smtplib.SMTP", smtp_cls)

    result = send_email(to="recipient@example.com", subject="Hello", body="World", cc="")

    smtp_conn.starttls.assert_called_once()
    smtp_conn.login.assert_called_once_with("user@example.com", "secret")
    assert smtp_conn.sendmail.called
    assert "sent" in result.lower() or "success" in result.lower()


def test_send_email_with_cc(monkeypatch) -> None:
    smtp_conn = _make_smtp_conn()
    monkeypatch.setattr("smtplib.SMTP", MagicMock(return_value=smtp_conn))

    send_email(to="a@example.com", subject="S", body="B", cc="c@example.com,d@example.com")

    assert smtp_conn.sendmail.called
    recipients = smtp_conn.sendmail.call_args[0][1]
    assert "c@example.com" in recipients
    assert "d@example.com" in recipients


def test_send_email_not_configured() -> None:
    _mod._config = {}
    result = send_email(to="x@x.com", subject="s", body="b", cc="")
    assert "Error" in result or "not configured" in result.lower()


def test_send_email_smtp_error(monkeypatch) -> None:
    import smtplib

    smtp_conn = _make_smtp_conn()
    smtp_conn.starttls.side_effect = smtplib.SMTPException("TLS failed")
    monkeypatch.setattr("smtplib.SMTP", MagicMock(return_value=smtp_conn))

    result = send_email(to="x@x.com", subject="s", body="b", cc="")
    assert "Error" in result or "error" in result.lower()


def test_send_email_no_tls(monkeypatch) -> None:
    _mod._config = {}
    configure_email({**_CREDS, "use_tls": False})
    smtp_conn = _make_smtp_conn()
    monkeypatch.setattr("smtplib.SMTP", MagicMock(return_value=smtp_conn))

    result = send_email(to="x@x.com", subject="s", body="b", cc="")
    smtp_conn.starttls.assert_not_called()
    assert "sent" in result.lower() or "success" in result.lower()


# ── search_email ──────────────────────────────────────────────────────────────


def test_search_email_by_subject(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(subject="Test", sender="", since="", folder="INBOX", limit=5)

    search_args = imap_cls._conn.search.call_args[0]
    assert any("SUBJECT" in str(a) for a in search_args)
    assert "Test subject" in result


def test_search_email_by_sender(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    search_email(subject="", sender="alice@example.com", since="", folder="INBOX", limit=5)

    search_args = imap_cls._conn.search.call_args[0]
    assert any("FROM" in str(a) for a in search_args)


def test_search_email_since_date(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    search_email(subject="", sender="", since="01-Jan-2024", folder="INBOX", limit=5)

    search_args = imap_cls._conn.search.call_args[0]
    assert any("SINCE" in str(a) for a in search_args)


def test_search_email_no_criteria_uses_all(monkeypatch) -> None:
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    search_email(subject="", sender="", since="", folder="INBOX", limit=5)

    search_args = imap_cls._conn.search.call_args[0]
    assert any("ALL" in str(a) for a in search_args)


def test_search_email_no_results(monkeypatch) -> None:
    imap_cls = _make_imap_cls(uids=[])
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(subject="nonexistent", sender="", since="", folder="INBOX", limit=5)
    assert "No messages" in result or "0" in result


def test_search_email_not_configured() -> None:
    _mod._config = {}
    result = search_email(subject="x", sender="", since="", folder="INBOX", limit=5)
    assert "Error" in result or "not configured" in result.lower()


def test_search_email_crlf_injection_prevention(monkeypatch) -> None:
    """Issue #870: IMAP protocol injection via CRLF in subject/sender."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    # Malicious subject with CRLF injection payload
    search_email(
        subject='test"\r\nDELETE INBOX',
        sender="evil\r\nSTATUS INBOX (MESSAGES)",
        since="\x00\x1f",
        folder="INBOX",
        limit=5,
    )

    search_args = imap_cls._conn.search.call_args[0]
    criteria_str = " ".join(str(a) for a in search_args)

    # CRLF and control characters must be stripped from all fields
    assert "\r" not in criteria_str
    assert "\n" not in criteria_str
    assert "\x00" not in criteria_str
    assert "\x1f" not in criteria_str
    # The SEARCH command must remain a single logical line (no IMAP command injection)
    # After sanitization, the text is just search content, not new IMAP commands
    assert "SUBJECT" in criteria_str
    assert "FROM" in criteria_str
    # Since was all control chars so it's stripped away to empty
    assert "SINCE" not in criteria_str


# ── TOOL_CONFIGS integrity ────────────────────────────────────────────────────


# ── folder sanitization (issue #883) ──────────────────────────────────────────


def test_read_email_sanitizes_folder_crlf(monkeypatch) -> None:
    """Issue #883: Ensure folder CRLF is stripped before conn.select()."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    read_email(limit=1, folder="INBOX\r\nDELETE INBOX")

    # CRLF must be stripped (prevents IMAP command splitting)
    select_arg = imap_cls._conn.select.call_args[0][0]
    assert "\r" not in select_arg
    assert "\n" not in select_arg


def test_search_email_sanitizes_folder_crlf(monkeypatch) -> None:
    """Issue #883: Ensure folder CRLF is stripped before conn.select() in search_email."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    search_email(
        subject="test", sender="", since="", folder="Sent\r\nSTATUS INBOX (MESSAGES)", limit=5
    )

    select_arg = imap_cls._conn.select.call_args[0][0]
    assert "\r" not in select_arg
    assert "\n" not in select_arg


# ── since date validation (issue #883) ────────────────────────────────────────


def test_search_email_since_valid_date_passes(monkeypatch) -> None:
    """Valid DD-Mon-YYYY dates should be accepted."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(subject="", sender="", since="01-Jan-2024", folder="INBOX", limit=5)
    assert "Error" not in result


def test_search_email_since_invalid_with_extra_terms(monkeypatch) -> None:
    """Invalid since with injection payload should be rejected with error."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(
        subject="",
        sender="",
        since='01-Jan-2024 ALL OR FROM "evil@evil.com"',
        folder="INBOX",
        limit=5,
    )
    assert "Error" in result or "DD-Mon-YYYY" in result


def test_search_email_since_invalid_garbage(monkeypatch) -> None:
    """Non-date since values should be rejected."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(subject="", sender="", since="not-a-date", folder="INBOX", limit=5)
    assert "Error" in result or "DD-Mon-YYYY" in result


def test_search_email_since_empty_is_allowed(monkeypatch) -> None:
    """Empty since should default to ALL search (no date filter)."""
    imap_cls = _make_imap_cls()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", imap_cls)

    result = search_email(subject="", sender="", since="", folder="INBOX", limit=5)
    assert "Error" not in result
    search_args = imap_cls._conn.search.call_args[0]
    assert any("ALL" in str(a) for a in search_args)


def test_tool_configs_structure() -> None:
    from src.tools.email_tools import TOOL_CONFIG, TOOL_CONFIGS

    assert len(TOOL_CONFIGS) == 3
    names = {c["name"] for c in TOOL_CONFIGS}
    assert names == {"read_email", "send_email", "search_email"}

    send_cfg = next(c for c in TOOL_CONFIGS if c["name"] == "send_email")
    assert send_cfg["requires_confirmation"] is True

    for name in ("read_email", "search_email"):
        cfg = next(c for c in TOOL_CONFIGS if c["name"] == name)
        assert cfg["requires_confirmation"] is False

    assert TOOL_CONFIG["name"] == "read_email"
