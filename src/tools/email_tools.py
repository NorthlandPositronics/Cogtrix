"""IMAP/SMTP email tools — read, send, and search email.

Three tools are exposed:
    read_email   — Fetch recent messages from an IMAP folder.
    send_email   — Send a message via SMTP.
    search_email — Search messages by subject, sender, or date range.

Configuration (config file, ``services.email`` section):
    imap_host   — IMAP server hostname (required)
    imap_port   — IMAP port (default: 993)
    smtp_host   — SMTP server hostname (required)
    smtp_port   — SMTP port (default: 587)
    username    — Login username / From address (required)
    password    — Login password (required)
    use_tls     — Use TLS for IMAP and STARTTLS for SMTP (default: true)

All three tools are removed from the agent by ``filter_unconfigured_tools``
when the required credentials are absent.
"""

from __future__ import annotations

import email as _email_stdlib
import email.header as _email_header
import imaplib
import logging
import re
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment, misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

log = logging.getLogger("cogtrix.tools.email")

# ── Module-level configuration ────────────────────────────────────────────────

_config: dict[str, Any] = {}


def configure_email(config: dict[str, Any]) -> None:
    """Set runtime configuration.  Called from ``configure.py`` at startup.

    Expected keys:
        imap_host, imap_port, smtp_host, smtp_port,
        username, password, use_tls
    """
    global _config
    # Atomic reference swap — safe for concurrent readers without a lock
    _config = {**_config, **config}


def is_configured() -> bool:
    """Return True when all required credentials are present."""
    return bool(
        _config.get("imap_host")
        and _config.get("smtp_host")
        and _config.get("username")
        and _config.get("password")
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_imap_connection() -> imaplib.IMAP4 | imaplib.IMAP4_SSL:
    host = _config.get("imap_host", "")
    port = int(_config.get("imap_port", 993))
    username = _config.get("username", "")
    password = _config.get("password", "")
    use_tls = _config.get("use_tls", True)

    if not host or not username or not password:
        raise RuntimeError(
            "Email not configured. Set services.email.imap_host, "
            "username, and password in .cogtrix.yaml."
        )

    if use_tls:
        ctx = ssl.create_default_context()
        conn = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
    else:
        conn = imaplib.IMAP4(host, port)

    conn.login(username, password)
    return conn


def _decode_header_value(raw: str | bytes | None) -> str:
    """Decode an RFC-2047-encoded header value to a plain Python string."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = _email_header.decode_header(raw)
    decoded_parts: list[str] = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(part)
    return "".join(decoded_parts)


def _extract_body(msg: Any) -> str:
    """Extract plain-text body from an email.message.Message object."""
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ct == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(payload.decode(charset, errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body_parts.append(payload.decode(charset, errors="replace"))
    return "\n".join(body_parts)


def _format_message(uid: bytes, raw: list[Any], include_body: bool = True) -> str:
    """Format a raw IMAP fetch result into a human-readable string."""
    raw_bytes = b""
    for part in raw:
        if isinstance(part, tuple) and len(part) >= 2:
            raw_bytes = part[1] if isinstance(part[1], bytes) else b""
            break
    if not raw_bytes:
        return f"[{uid.decode()}] (empty)"

    msg = _email_stdlib.message_from_bytes(raw_bytes)
    subject = _decode_header_value(msg.get("Subject"))
    from_ = _decode_header_value(msg.get("From"))
    to_ = _decode_header_value(msg.get("To"))
    date_ = _decode_header_value(msg.get("Date"))

    lines = [
        f"UID: {uid.decode()}",
        f"From: {from_}",
        f"To:   {to_}",
        f"Date: {date_}",
        f"Subject: {subject}",
    ]
    if include_body:
        body = _extract_body(msg)
        if body:
            # Truncate very long bodies to keep output manageable
            if len(body) > 2000:
                body = body[:2000] + "\n... (truncated)"
            lines.append(f"\n{body}")
    return "\n".join(lines)


# ── Input schemas ─────────────────────────────────────────────────────────────


class ReadEmailInput(BaseModel):
    limit: int = Field(
        default=10,
        description="Maximum number of messages to return (1-50).",
    )
    folder: str = Field(
        default="INBOX",
        description="IMAP folder to read from (e.g. 'INBOX', 'Sent', 'Drafts').",
    )


class SendEmailInput(BaseModel):
    to: str = Field(description="Recipient email address (or comma-separated list).")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")
    cc: str = Field(
        default="",
        description="Optional CC recipients (comma-separated).",
    )


class SearchEmailInput(BaseModel):
    subject: str = Field(
        default="",
        description="Search for messages whose Subject contains this string.",
    )
    sender: str = Field(
        default="",
        description="Search for messages from this sender address or name.",
    )
    since: str = Field(
        default="",
        description=(
            "Earliest date for results in DD-Mon-YYYY format "
            "(e.g. '01-Jan-2024').  Leave blank to search all dates."
        ),
    )
    folder: str = Field(
        default="INBOX",
        description="IMAP folder to search in.",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of messages to return (1-50).",
    )


# ── Tool functions ────────────────────────────────────────────────────────────


def read_email(limit: int = 10, folder: str = "INBOX") -> str:
    """Fetch recent messages from an IMAP email folder.

    Connects to the configured IMAP server, selects *folder* (default
    INBOX), and returns the *limit* most recent messages with their
    headers and plain-text bodies.

    Args:
        limit:  Number of messages to fetch (1-50; default 10).
        folder: IMAP folder name (default 'INBOX').

    Returns:
        Formatted string listing each message, or an error string.
    """
    limit = max(1, min(limit, 50))
    folder = _sanitize_imap_value(folder)

    try:
        conn = _get_imap_connection()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error connecting to IMAP: {exc}"

    try:
        status, data = conn.select(folder, readonly=True)
        if status != "OK":
            return f"Error selecting folder {folder!r}: {data}"

        status, data = conn.search(None, "ALL")
        if status != "OK":
            return "Error: IMAP search failed."

        uid_list: list[bytes] = data[0].split() if data[0] else []
        if not uid_list:
            return f"No messages in {folder!r}."

        # Take the most recent `limit` messages (last N UIDs)
        selected = uid_list[-limit:]
        selected_ids = b",".join(selected).decode()

        status, fetch_data = conn.fetch(selected_ids, "(RFC822)")
        if status != "OK":
            return "Error fetching messages."

        output: list[str] = [
            f"Showing {len(selected)} of {len(uid_list)} messages in {folder!r}.\n"
        ]
        # fetch_data alternates (header_bytes_tuple, closing_paren_bytes)
        # We collect the tuple elements
        idx = 0
        for uid in reversed(selected):
            raw_parts: list[Any] = []
            while idx < len(fetch_data):
                item = fetch_data[idx]
                idx += 1
                if isinstance(item, tuple):
                    raw_parts.append(item)
                    break
            output.append(_format_message(uid, raw_parts))
            output.append("─" * 60)

        return "\n".join(output)
    except Exception as exc:
        return f"Error reading email: {exc}"
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def send_email(to: str, subject: str, body: str, cc: str = "") -> str:
    """Send an email via SMTP.

    Uses the configured SMTP server with STARTTLS (or plain, if
    use_tls is false).  The From address is taken from the configured
    username.

    Args:
        to:      Recipient(s) — comma-separated email addresses.
        subject: Subject line.
        body:    Plain-text body.
        cc:      Optional CC recipients — comma-separated.

    Returns:
        Confirmation string, or an error message.
    """
    host = _config.get("smtp_host", "")
    port = int(_config.get("smtp_port", 587))
    username = _config.get("username", "")
    password = _config.get("password", "")
    use_tls = _config.get("use_tls", True)

    if not host or not username or not password:
        return (
            "Error: Email not configured. Set services.email.smtp_host, "
            "username, and password in .cogtrix.yaml."
        )

    if not to.strip():
        return "Error: 'to' field is required."
    if not subject.strip():
        return "Error: 'subject' field is required."
    if not body.strip():
        return "Error: 'body' field is required."

    msg = MIMEMultipart()
    msg["From"] = username
    msg["To"] = to
    msg["Subject"] = subject
    if cc.strip():
        msg["Cc"] = cc

    msg.attach(MIMEText(body, "plain", "utf-8"))

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    if cc.strip():
        recipients += [addr.strip() for addr in cc.split(",") if addr.strip()]

    try:
        if use_tls:
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(username, password)
                server.sendmail(username, recipients, msg.as_string())
        else:
            with smtplib.SMTP(host, port) as server:
                server.login(username, password)
                server.sendmail(username, recipients, msg.as_string())

        to_summary = to
        if cc.strip():
            to_summary += f" (cc: {cc})"
        return f"Email sent successfully to {to_summary}."
    except smtplib.SMTPAuthenticationError:
        return "Error: SMTP authentication failed. Check username and password."
    except smtplib.SMTPException as exc:
        return f"Error sending email: {exc}"
    except Exception as exc:
        return f"Error sending email: {exc}"


def _sanitize_imap_value(value: str) -> str:
    """Strip CRLF and ASCII control characters to prevent IMAP protocol injection.

    IMAP is line-oriented and uses CRLF as a command delimiter.
    Raw newlines in search criteria could split a single SEARCH command
    into multiple IMAP commands, allowing protocol injection.
    """
    return re.sub(r"[\r\n\x00-\x1f]", "", value)


def search_email(
    subject: str = "",
    sender: str = "",
    since: str = "",
    folder: str = "INBOX",
    limit: int = 10,
) -> str:
    """Search email messages by subject, sender, and/or date range via IMAP.

    Builds an IMAP SEARCH criteria from the provided filters and returns
    matching messages most-recent-first.  At least one filter should be
    specified; if all are empty, returns the most recent messages (same
    as read_email).

    Args:
        subject: Substring to match against the Subject header.
        sender:  Substring to match against the From header.
        since:   Earliest date in 'DD-Mon-YYYY' format (e.g. '01-Jan-2024').
        folder:  IMAP folder to search (default 'INBOX').
        limit:   Maximum number of results to return (1-50).

    Returns:
        Formatted list of matching messages, or an error message.
    """
    limit = max(1, min(limit, 50))
    folder = _sanitize_imap_value(folder)

    try:
        conn = _get_imap_connection()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error connecting to IMAP: {exc}"

    try:
        status, data = conn.select(folder, readonly=True)
        if status != "OK":
            return f"Error selecting folder {folder!r}: {data}"

        # Build IMAP SEARCH criteria (sanitize inputs to prevent injection)
        criteria_parts: list[str] = []
        clean_subject = _sanitize_imap_value(subject).strip()
        if clean_subject:
            # IMAP SEARCH SUBJECT uses quoted strings
            escaped = clean_subject.replace('"', '\\"')
            criteria_parts.append(f'SUBJECT "{escaped}"')
        clean_sender = _sanitize_imap_value(sender).strip()
        if clean_sender:
            escaped = clean_sender.replace('"', '\\"')
            criteria_parts.append(f'FROM "{escaped}"')
        clean_since = _sanitize_imap_value(since).strip()
        if clean_since:
            if not re.match(r"^\d{2}-[A-Za-z]{3}-\d{4}$", clean_since):
                return (
                    f"Error: 'since' must be in DD-Mon-YYYY format "
                    f"(e.g. '01-Jan-2024'), got {clean_since!r}"
                )
            criteria_parts.append(f"SINCE {clean_since}")

        criteria = " ".join(criteria_parts) if criteria_parts else "ALL"

        status, data = conn.search(None, criteria)
        if status != "OK":
            return f"Error: IMAP SEARCH failed (criteria: {criteria!r})."

        uid_list: list[bytes] = data[0].split() if data[0] else []
        if not uid_list:
            return f"No messages matched in {folder!r} (criteria: {criteria!r})."

        selected = uid_list[-limit:]
        selected_ids = b",".join(selected).decode()

        status, fetch_data = conn.fetch(selected_ids, "(RFC822)")
        if status != "OK":
            return "Error fetching messages."

        output: list[str] = [
            f"Found {len(uid_list)} match(es); showing {len(selected)} most recent "
            f"(folder={folder!r}, criteria={criteria!r}).\n"
        ]
        idx = 0
        for uid in reversed(selected):
            raw_parts: list[Any] = []
            while idx < len(fetch_data):
                item = fetch_data[idx]
                idx += 1
                if isinstance(item, tuple):
                    raw_parts.append(item)
                    break
            output.append(_format_message(uid, raw_parts))
            output.append("─" * 60)

        return "\n".join(output)
    except Exception as exc:
        return f"Error searching email: {exc}"
    finally:
        try:
            conn.logout()
        except Exception:
            pass


# ── Tool registration ─────────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "read_email",
        "description": (
            "Fetch recent email messages from an IMAP folder.\n"
            "\n"
            "Connects to the configured IMAP server and returns up to `limit` "
            "messages from `folder` (default: INBOX), most-recent first.  "
            "Each result includes the sender, recipient, date, subject, and "
            "plain-text body (truncated at 2 000 chars).\n"
            "\n"
            "Requires: services.email configured in .cogtrix.yaml."
        ),
        "input_schema": ReadEmailInput,
        "function": read_email,
        "requires_confirmation": False,
    },
    {
        "name": "send_email",
        "description": (
            "Send an email via SMTP.\n"
            "\n"
            "Composes a plain-text message and delivers it through the "
            "configured SMTP server using STARTTLS.  The From address is "
            "the configured username.\n"
            "\n"
            "Requires: services.email configured in .cogtrix.yaml."
        ),
        "input_schema": SendEmailInput,
        "function": send_email,
        "requires_confirmation": True,
    },
    {
        "name": "search_email",
        "description": (
            "Search email messages by subject, sender, and/or date via IMAP.\n"
            "\n"
            "Builds an IMAP SEARCH query from the provided filters and "
            "returns matching messages most-recent first.  All filters are "
            "optional; combining multiple filters narrows the results.\n"
            "\n"
            "Parameters:\n"
            "  subject — substring to match in the Subject header\n"
            "  sender  — substring to match in the From header\n"
            "  since   — earliest date in 'DD-Mon-YYYY' format\n"
            "  folder  — IMAP folder to search (default: INBOX)\n"
            "  limit   — max results to return (default: 10, max: 50)\n"
            "\n"
            "Requires: services.email configured in .cogtrix.yaml."
        ),
        "input_schema": SearchEmailInput,
        "function": search_email,
        "requires_confirmation": False,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "read_email",
    "send_email",
    "search_email",
    "configure_email",
    "is_configured",
    "ReadEmailInput",
    "SendEmailInput",
    "SearchEmailInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
