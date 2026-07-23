"""
Telegram messaging tool — send and receive messages via a Telegram Bot.

Create a bot via @BotFather on Telegram, obtain a token, and configure
Cogtrix to use it.

Configuration
=============
Environment variables (override config file)::

    COGTRIX_TELEGRAM_TOKEN     Bot token from @BotFather  (required)
    COGTRIX_TELEGRAM_SEND      "true"/"false"              (default: true)
    COGTRIX_TELEGRAM_RECEIVE   "true"/"false"              (default: true)
    COGTRIX_TELEGRAM_FILTER    "none"|"allow"|"ignore"|"blacklist" (legacy: "whitelist" maps to "allow")
    COGTRIX_TELEGRAM_CONTACTS  Comma-separated chat IDs or usernames

Config file (``services.telegram`` section)::

    services:
      telegram:
        bot_token: "123456:ABC-DEF..."
        allow_send: true
        allow_receive: true
        require_confirmation: true
        filter_mode: "allow"
        contacts:
          - "123456789"
          - "@alice_username"
        phonebook:
          alice: "123456789"
          team:  "-1001234567890"
        rate_limit: 30
        max_message_length: 4096
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.tools._telegram_client import REQUESTS_AVAILABLE, TelegramBotClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in _TRUE_STRINGS:
        return True
    if val in _FALSE_STRINGS:
        return False
    return default


@dataclass
class TelegramConfig:
    """Merged config from environment variables and config file."""

    bot_token: str | None = None

    allow_send: bool = True
    allow_receive: bool = True
    require_confirmation: bool = True

    filter_mode: str = "none"  # "none" | "allow" | "ignore" | "blacklist"
    contacts: list[str] = field(default_factory=list)
    phonebook: dict[str, str] = field(default_factory=dict)

    rate_limit: int = 30  # messages per hour, 0 = unlimited
    max_message_length: int = 4096


def _load_config() -> TelegramConfig:
    """Build config by merging config file values with environment overrides."""
    cfg = TelegramConfig()

    # 1) Load from config file (best-effort)
    try:
        from src.config import load_config

        app_cfg = load_config()
        tg = app_cfg.services.get("telegram", {})
        if tg:
            cfg.bot_token = tg.get("bot_token", cfg.bot_token)
            cfg.allow_send = tg.get("allow_send", cfg.allow_send)
            cfg.allow_receive = tg.get("allow_receive", cfg.allow_receive)
            cfg.require_confirmation = tg.get("require_confirmation", cfg.require_confirmation)
            cfg.filter_mode = tg.get("filter_mode", cfg.filter_mode)
            cfg.contacts = tg.get("contacts", cfg.contacts)
            cfg.phonebook = tg.get("phonebook", cfg.phonebook)
            cfg.rate_limit = tg.get("rate_limit", cfg.rate_limit)
            cfg.max_message_length = tg.get("max_message_length", cfg.max_message_length)
    except Exception:
        pass

    # 2) Environment overrides (highest priority)
    if token := os.getenv("COGTRIX_TELEGRAM_TOKEN"):
        cfg.bot_token = token.strip()
    cfg.allow_send = _env_bool("COGTRIX_TELEGRAM_SEND", cfg.allow_send)
    cfg.allow_receive = _env_bool("COGTRIX_TELEGRAM_RECEIVE", cfg.allow_receive)
    fm = os.environ.get("COGTRIX_TELEGRAM_FILTER", "").lower().strip()
    if fm:
        _LEGACY_MODES = {"whitelist": "allow"}
        fm = _LEGACY_MODES.get(fm, fm)
        if fm in ("none", "allow", "ignore", "blacklist"):
            cfg.filter_mode = fm
    if contacts_str := os.getenv("COGTRIX_TELEGRAM_CONTACTS"):
        cfg.contacts = [c.strip() for c in contacts_str.split(",") if c.strip()]

    return cfg


# Singleton — loaded once at import time
_cfg = _load_config()

# ---------------------------------------------------------------------------
# Rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------

_send_timestamps: deque[float] = deque()


def _rate_limit_ok() -> bool:
    """Return True if we haven't hit the hourly send limit."""
    if _cfg.rate_limit <= 0:
        return True
    now = time.monotonic()
    window = 3600.0  # 1 hour
    while _send_timestamps and (now - _send_timestamps[0]) > window:
        _send_timestamps.popleft()
    return len(_send_timestamps) < _cfg.rate_limit


def _record_send() -> None:
    _send_timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Contact filtering
# ---------------------------------------------------------------------------


def _resolve_contact(name_or_id: str) -> str:
    """Resolve a phonebook nickname or normalize a chat ID / username.

    Returns the chat ID (numeric string) or ``@username``.
    """
    key = name_or_id.strip()

    # Resolve phonebook nickname (case-insensitive)
    for k, v in _cfg.phonebook.items():
        if k.lower() == key.lower():
            return v

    return key


def _check_contact(name_or_id: str) -> tuple[bool, str]:
    """Enforce allow/ignore/blacklist rules for outbound messages.

    Returns:
        (allowed, reason) — reason is non-empty only when blocked.
    """
    mode = _cfg.filter_mode
    if mode == "none":
        return True, ""

    resolved = _resolve_contact(name_or_id)
    normalized_contacts = {_resolve_contact(c) for c in _cfg.contacts}

    if mode in ("allow", "whitelist"):
        if resolved not in normalized_contacts:
            msg = (
                f"Contact {resolved} is not in the allow list"
                if mode == "allow"
                else f"Contact {resolved} is not in the allowed whitelist."
            )
            return False, msg
        return True, ""

    if mode == "ignore":
        if resolved in normalized_contacts:
            return False, f"Contact {resolved} is in the ignore list"
        return True, ""

    if mode == "blacklist":
        if resolved in normalized_contacts:
            return False, f"Contact {resolved} is blacklisted"
        return True, ""

    return True, ""


def _check_receive_contact(chat_id: int | str) -> bool:
    """Check whether an inbound message passes the contact filter."""
    mode = _cfg.filter_mode
    if mode == "none":
        return True
    cid = str(chat_id)
    normalized_contacts = {_resolve_contact(c) for c in _cfg.contacts}
    if mode in ("allow", "whitelist"):
        return cid in normalized_contacts
    if mode in ("ignore", "blacklist"):
        return cid not in normalized_contacts
    return True


# ---------------------------------------------------------------------------
# Bot client singleton
# ---------------------------------------------------------------------------


def _get_client() -> TelegramBotClient:
    return TelegramBotClient(token=_cfg.bot_token or "")


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class TelegramSendInput(BaseModel):
    """Input schema for sending a Telegram text message."""

    to: str = Field(
        description=(
            "Recipient — a Telegram chat ID (numeric), @username, "
            "or a phonebook nickname (e.g. 'alice')."
        )
    )
    message: str = Field(description="The text message to send.")


class TelegramSendPhotoInput(BaseModel):
    """Input schema for sending a Telegram photo."""

    to: str = Field(description="Recipient — a Telegram chat ID, @username, or phonebook nickname.")
    photo_url: str = Field(description="Public URL of the photo to send.")
    caption: str | None = Field(
        default=None,
        description="Optional caption for the photo.",
    )


class TelegramCheckInput(BaseModel):
    """Input schema for checking Telegram messages."""

    limit: int = Field(
        default=10,
        description="Number of recent messages to retrieve (max 50).",
    )


class TelegramContactsInput(BaseModel):
    """Input schema for listing phonebook contacts."""

    pass  # No parameters needed


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def telegram_send(to: str, message: str) -> str:
    """
    Send a text message via Telegram.

    Args:
        to: Recipient chat ID, @username, or phonebook nickname.
        message: Text message body.

    Returns:
        Confirmation string or error message.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install with: pip install requests"

    if not _cfg.bot_token:
        return "Error: Telegram bot token not configured."

    # Contact filter
    allowed, reason = _check_contact(to)
    if not allowed:
        return f"Blocked: {reason}"

    # Rate limit
    if not _rate_limit_ok():
        return (
            f"Rate limit reached ({_cfg.rate_limit} messages/hour). "
            "Please wait before sending again."
        )

    # Truncate
    if len(message) > _cfg.max_message_length:
        message = message[: _cfg.max_message_length]

    chat_id = _resolve_contact(to)
    client = _get_client()
    result = client.send_message(chat_id, message)

    if result.ok:
        _record_send()
        display_to = to if to != chat_id else chat_id
        return f"Telegram message sent to {display_to} (id: {result.message_id})"
    return f"Failed to send: {result.error}"


def telegram_send_photo(
    to: str,
    photo_url: str,
    caption: str | None = None,
) -> str:
    """
    Send a photo via Telegram.

    Args:
        to: Recipient chat ID, @username, or phonebook nickname.
        photo_url: Public URL of the photo.
        caption: Optional caption text.

    Returns:
        Confirmation string or error message.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install with: pip install requests"

    if not _cfg.bot_token:
        return "Error: Telegram bot token not configured."

    allowed, reason = _check_contact(to)
    if not allowed:
        return f"Blocked: {reason}"

    if not _rate_limit_ok():
        return (
            f"Rate limit reached ({_cfg.rate_limit} messages/hour). "
            "Please wait before sending again."
        )

    chat_id = _resolve_contact(to)
    client = _get_client()
    result = client.send_photo(chat_id, photo_url, caption=caption)

    if result.ok:
        _record_send()
        display_to = to if to != chat_id else chat_id
        return f"Photo sent to {display_to} via Telegram (id: {result.message_id})"
    return f"Failed to send photo: {result.error}"


def telegram_check(limit: int = 10) -> str:
    """
    Retrieve recent Telegram messages sent to the bot.

    Args:
        limit: Max number of messages (capped at 50).

    Returns:
        Formatted list of recent messages.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Install with: pip install requests"

    if not _cfg.bot_token:
        return "Error: Telegram bot token not configured."

    limit = min(max(1, limit), 50)
    client = _get_client()
    messages = client.get_updates(limit=limit)

    if not messages:
        return "No recent Telegram messages."

    # Apply contact filter
    if _cfg.filter_mode != "none":
        messages = [m for m in messages if _check_receive_contact(m.chat_id)]

    if not messages:
        return "No messages matching your contact filter."

    output: list[str] = [f"Recent Telegram messages ({len(messages)}):\n"]
    for msg in messages:
        sender = msg.from_username or msg.from_first_name or str(msg.from_id or "?")
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(msg.date)) if msg.date else "?"
        chat_label = msg.chat_title or str(msg.chat_id)
        body = msg.text[:300] + "..." if len(msg.text) > 300 else msg.text
        media_tag = ""
        if msg.has_photo:
            media_tag = " [photo]"
        elif msg.has_document:
            media_tag = " [document]"
        output.append(f"  [{ts}] {sender} in {chat_label}: {body}{media_tag}")

    return "\n".join(output)


def telegram_contacts() -> str:
    """
    List the configured phonebook contacts for Telegram.

    Returns:
        Formatted phonebook listing.
    """
    if not _cfg.phonebook:
        return (
            "No Telegram phonebook contacts configured.\n"
            "Add them in .cogtrix.json under services.telegram.phonebook."
        )

    lines = ["Telegram phonebook contacts:\n"]
    for nick, chat_id in sorted(_cfg.phonebook.items()):
        lines.append(f"  {nick}: {chat_id}")

    if _cfg.filter_mode != "none":
        lines.append(f"\nFilter mode: {_cfg.filter_mode}")
        if _cfg.contacts:
            lines.append(f"Filter list: {', '.join(_cfg.contacts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# is_configured — gating for the registry
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """Return True if Telegram integration is usable.

    Checks:
        1. ``requests`` library is installed.
        2. A bot token is configured.
        3. At least one capability (send or receive) is enabled.
    """
    if not REQUESTS_AVAILABLE:
        return False
    if not _cfg.bot_token:
        return False
    if not _cfg.allow_send and not _cfg.allow_receive:
        return False
    return True


# ---------------------------------------------------------------------------
# TOOL_CONFIGS — dynamically built based on capability flags
# ---------------------------------------------------------------------------


def _build_tool_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    if _cfg.allow_send:
        configs.append(
            {
                "name": "telegram_send",
                "description": (
                    "Send a text message via Telegram to a chat ID, @username, "
                    "or phonebook contact. Requires a configured bot token.\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks you to send a Telegram message\n"
                    "- You need to notify someone via Telegram\n"
                ),
                "input_schema": TelegramSendInput,
                "requires_confirmation": _cfg.require_confirmation,
                "function": telegram_send,
            }
        )
        configs.append(
            {
                "name": "telegram_send_photo",
                "description": (
                    "Send a photo via Telegram given a public URL. "
                    "Optionally include a caption.\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks you to send an image/photo via Telegram\n"
                ),
                "input_schema": TelegramSendPhotoInput,
                "requires_confirmation": _cfg.require_confirmation,
                "function": telegram_send_photo,
            }
        )

    if _cfg.allow_receive:
        configs.append(
            {
                "name": "telegram_check",
                "description": (
                    "Retrieve recent Telegram messages sent to the bot.\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks to check Telegram messages\n"
                    "- You need to read incoming Telegram messages\n"
                ),
                "input_schema": TelegramCheckInput,
                "requires_confirmation": False,
                "function": telegram_check,
            }
        )

    # Contacts tool is always available when any capability is enabled
    if configs:
        configs.append(
            {
                "name": "telegram_contacts",
                "description": (
                    "List the phonebook contacts configured for Telegram. "
                    "Shows nicknames, chat IDs, and active filter rules."
                ),
                "input_schema": TelegramContactsInput,
                "requires_confirmation": False,
                "function": telegram_contacts,
            }
        )

    return configs


TOOL_CONFIGS = _build_tool_configs()
TOOL_CONFIG = TOOL_CONFIGS[0] if TOOL_CONFIGS else {}


__all__ = [
    "telegram_send",
    "telegram_send_photo",
    "telegram_check",
    "telegram_contacts",
    "is_configured",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
    "TelegramSendInput",
    "TelegramSendPhotoInput",
    "TelegramCheckInput",
    "TelegramContactsInput",
]
