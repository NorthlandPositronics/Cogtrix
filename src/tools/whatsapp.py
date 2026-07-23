"""
WhatsApp messaging tool — send and receive messages via Waha.

Waha is a self-hosted Docker container that wraps WhatsApp Web behind a
REST API.  Run it alongside Cogtrix::

    docker run -p 3000:3000 devlikeapro/waha

Then scan the QR code in the Waha dashboard (http://localhost:3000/)
and configure Cogtrix to use it.

Configuration
=============
Environment variables (override config file)::

    COGTRIX_WHATSAPP_URL       Waha server URL   (default: http://localhost:3000)
    COGTRIX_WHATSAPP_API_KEY   Waha X-Api-Key    (default: empty)
    COGTRIX_WHATSAPP_SESSION   Waha session name  (default: "default")
    COGTRIX_WHATSAPP_SEND      "true"/"false"     (default: true)
    COGTRIX_WHATSAPP_RECEIVE   "true"/"false"     (default: true)
    COGTRIX_WHATSAPP_FILTER    "none"|"allow"|"ignore"|"blacklist" (legacy: "whitelist" maps to "allow")
    COGTRIX_WHATSAPP_CONTACTS  Comma-separated E.164 numbers

Config file (``services.whatsapp`` section)::

    services:
      whatsapp:
        waha_url: "http://localhost:3000"
        api_key: "yoursecretkey"
        session: "default"
        allow_send: true
        allow_receive: true
        require_confirmation: true
        filter_mode: "allow"           # "none" | "allow" | "ignore" | "blacklist"
        contacts:
          - "+14155551234"
          - "+442071234567"
        phonebook:
          alice: "+14155551234"
          bob:   "+442071234567"
        rate_limit: 30                # messages per hour (0 = unlimited)
        max_message_length: 4096

TOOL_SETUP(config) is called automatically by ToolRegistry after this module
is imported.  It rebuilds the module singleton ``_cfg`` from the passed app
``Config`` (which ``_apply_env_vars`` populates before env vars are unset) so
the API key survives the ``COGTRIX_WHATSAPP_API_KEY`` unset (#2223 phase 2).
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from src.logging_config import get_logger
from src.tools._whatsapp_client import REQUESTS_AVAILABLE, WahaClient
from src.tools.delegate import register_tool_categories

if TYPE_CHECKING:
    from src.config import Config

log = get_logger()

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
class WhatsAppConfig:
    """Merged config from environment variables and config file."""

    waha_url: str = "http://localhost:3000"
    api_key: str | None = None
    session: str = "default"

    allow_send: bool = True
    allow_receive: bool = True
    require_confirmation: bool = True

    filter_mode: str = "none"  # "none" | "allow" | "ignore" | "blacklist"
    contacts: list[str] = field(default_factory=list)
    phonebook: dict[str, str] = field(default_factory=dict)

    rate_limit: int = 30  # messages per hour, 0 = unlimited
    max_message_length: int = 4096


def _load_config() -> WhatsAppConfig:
    """Build config by merging config file values with environment overrides."""
    cfg = WhatsAppConfig()

    # 1) Load from config file (best-effort). #2101: reuse the process-wide
    #    resolved config so the environment is read once; the api_key survives the
    #    #2223/#2102 unset via the secret cache.
    try:
        from src.config import get_cached_config

        app_cfg = get_cached_config()
        wa = app_cfg.services.get("whatsapp", {})
        if wa:
            cfg.waha_url = wa.get("waha_url", cfg.waha_url)
            cfg.api_key = wa.get("api_key", cfg.api_key)
            cfg.session = wa.get("session", cfg.session)
            cfg.allow_send = wa.get("allow_send", cfg.allow_send)
            cfg.allow_receive = wa.get("allow_receive", cfg.allow_receive)
            cfg.require_confirmation = wa.get("require_confirmation", cfg.require_confirmation)
            cfg.filter_mode = wa.get("filter_mode", cfg.filter_mode)
            cfg.contacts = wa.get("contacts", cfg.contacts)
            cfg.phonebook = wa.get("phonebook", cfg.phonebook)
            cfg.rate_limit = wa.get("rate_limit", cfg.rate_limit)
            cfg.max_message_length = wa.get("max_message_length", cfg.max_message_length)
    except Exception as exc:
        log.warning("Failed to parse WhatsApp config: %s", exc)

    # 2) Environment overrides (highest priority)
    if url := os.getenv("COGTRIX_WHATSAPP_URL"):
        cfg.waha_url = url.strip()
    if key := os.getenv("COGTRIX_WHATSAPP_API_KEY"):
        cfg.api_key = key.strip()
    if session := os.getenv("COGTRIX_WHATSAPP_SESSION"):
        cfg.session = session.strip()
    cfg.allow_send = _env_bool("COGTRIX_WHATSAPP_SEND", cfg.allow_send)
    cfg.allow_receive = _env_bool("COGTRIX_WHATSAPP_RECEIVE", cfg.allow_receive)
    fm = os.environ.get("COGTRIX_WHATSAPP_FILTER", "").lower().strip()
    if fm:
        _LEGACY_MODES = {"whitelist": "allow"}
        fm = _LEGACY_MODES.get(fm, fm)
        if fm in ("none", "allow", "ignore", "blacklist"):
            cfg.filter_mode = fm
    if contacts_str := os.getenv("COGTRIX_WHATSAPP_CONTACTS"):
        cfg.contacts = [c.strip() for c in contacts_str.split(",") if c.strip()]

    return cfg


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module.

    Rebuilds the module singleton from the passed app ``Config`` so the API
    key survives the ``COGTRIX_WHATSAPP_API_KEY`` env unset (#2223 phase 2).
    Re-running is idempotent — it simply replaces ``_cfg`` in-place.
    """
    global _cfg
    wa = (getattr(config, "services", None) or {}).get("whatsapp", {})
    cfg = WhatsAppConfig()
    if wa:
        cfg.waha_url = wa.get("waha_url", cfg.waha_url)
        cfg.api_key = wa.get("api_key", cfg.api_key)
        cfg.session = wa.get("session", cfg.session)
        cfg.allow_send = wa.get("allow_send", cfg.allow_send)
        cfg.allow_receive = wa.get("allow_receive", cfg.allow_receive)
        cfg.require_confirmation = wa.get("require_confirmation", cfg.require_confirmation)
        cfg.filter_mode = wa.get("filter_mode", cfg.filter_mode)
        cfg.contacts = wa.get("contacts", cfg.contacts)
        cfg.phonebook = wa.get("phonebook", cfg.phonebook)
        cfg.rate_limit = wa.get("rate_limit", cfg.rate_limit)
        cfg.max_message_length = wa.get("max_message_length", cfg.max_message_length)
    # Env overrides remain as fallback; after phase-2 unset they no-op
    if url := os.getenv("COGTRIX_WHATSAPP_URL"):
        cfg.waha_url = url.strip()
    if key := os.getenv("COGTRIX_WHATSAPP_API_KEY"):
        cfg.api_key = key.strip()
    if session := os.getenv("COGTRIX_WHATSAPP_SESSION"):
        cfg.session = session.strip()
    cfg.allow_send = _env_bool("COGTRIX_WHATSAPP_SEND", cfg.allow_send)
    cfg.allow_receive = _env_bool("COGTRIX_WHATSAPP_RECEIVE", cfg.allow_receive)
    fm = os.environ.get("COGTRIX_WHATSAPP_FILTER", "").lower().strip()
    if fm:
        _LEGACY_MODES = {"whitelist": "allow"}
        fm = _LEGACY_MODES.get(fm, fm)
        if fm in ("none", "allow", "ignore", "blacklist"):
            cfg.filter_mode = fm
    if contacts_str := os.getenv("COGTRIX_WHATSAPP_CONTACTS"):
        cfg.contacts = [c.strip() for c in contacts_str.split(",") if c.strip()]
    _cfg = cfg


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

_E164_RE = re.compile(r"^\+\d{7,15}$")


def _normalize_number(number_or_nick: str) -> str:
    """Resolve a phonebook nickname or normalize a phone number.

    Returns the number in E.164 format (``+12345678901``) or the
    Waha chatId format (``12345678901@c.us``).
    """
    key = number_or_nick.strip()

    # Resolve phonebook nickname
    if key.lower() in {k.lower(): k for k in _cfg.phonebook}:
        for k, v in _cfg.phonebook.items():
            if k.lower() == key.lower():
                key = v
                break

    # Strip Waha suffixes if present (for filtering comparison)
    key = key.replace("@c.us", "").replace("@s.whatsapp.net", "")
    # Ensure leading +
    if key.isdigit():
        key = f"+{key}"
    return key


def _to_chat_id(number: str) -> str:
    """Convert an E.164 number to Waha chatId format."""
    n = _normalize_number(number)
    digits = n.lstrip("+")
    if "@" in digits:
        return digits
    return f"{digits}@c.us"


def _check_contact(number_or_nick: str) -> tuple[bool, str]:
    """Enforce allow/ignore/blacklist rules for outbound messages.

    Returns:
        (allowed, reason) — reason is non-empty only when blocked.
    """
    mode = _cfg.filter_mode
    if mode == "none":
        return True, ""

    number = _normalize_number(number_or_nick)

    normalized_contacts = {_normalize_number(_resolve_contact(c)) for c in _cfg.contacts}

    if mode in ("allow", "whitelist"):
        if number not in normalized_contacts:
            msg = (
                f"Contact {number} is not in the allow list"
                if mode == "allow"
                else f"Contact {number} is not in the allowed whitelist."
            )
            return False, msg
        return True, ""

    if mode == "ignore":
        if number in normalized_contacts:
            return False, f"Contact {number} is in the ignore list"
        return True, ""

    if mode == "blacklist":
        if number in normalized_contacts:
            return False, f"Contact {number} is blacklisted"
        return True, ""

    return True, ""


def _resolve_contact(name_or_number: str) -> str:
    """Resolve a phonebook name to its number, or return the value unchanged."""
    return _cfg.phonebook.get(name_or_number, name_or_number)


def _check_receive_contact(from_field: str) -> bool:
    """Check whether an inbound message passes the contact filter."""
    mode = _cfg.filter_mode
    if mode == "none":
        return True
    number = from_field.replace("@c.us", "").replace("@s.whatsapp.net", "")
    if number.isdigit():
        number = f"+{number}"
    normalized_contacts = {_normalize_number(_resolve_contact(c)) for c in _cfg.contacts}
    if mode in ("allow", "whitelist"):
        return number in normalized_contacts
    if mode in ("ignore", "blacklist"):
        return number not in normalized_contacts
    return True


# ---------------------------------------------------------------------------
# Waha client singleton
# ---------------------------------------------------------------------------


def _get_client() -> WahaClient:
    return WahaClient(
        base_url=_cfg.waha_url,
        api_key=_cfg.api_key,
        session=_cfg.session,
    )


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class WhatsAppSendInput(BaseModel):
    """Input schema for sending a WhatsApp text message."""

    to: str = Field(
        description=(
            "Recipient — a phone number in international format "
            "(e.g. '+14155551234') or a phonebook nickname (e.g. 'alice')."
        )
    )
    message: str = Field(description="The text message to send.")


class WhatsAppSendImageInput(BaseModel):
    """Input schema for sending a WhatsApp image."""

    to: str = Field(
        description=("Recipient — a phone number in international format or a phonebook nickname.")
    )
    image_url: str = Field(description="Public URL of the image to send (JPEG preferred).")
    caption: str | None = Field(
        default=None,
        description="Optional caption for the image.",
    )


class WhatsAppCheckInput(BaseModel):
    """Input schema for checking WhatsApp messages."""

    contact: str | None = Field(
        default=None,
        description=(
            "Filter messages from/to this contact (phone number or "
            "phonebook nickname). Leave empty to fetch from all chats."
        ),
    )
    limit: int = Field(
        default=10,
        description="Number of recent messages to retrieve (max 50).",
    )


class WhatsAppContactsInput(BaseModel):
    """Input schema for listing phonebook contacts."""

    pass  # No parameters needed


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def whatsapp_send(to: str, message: str) -> str:
    """
    Send a text message via WhatsApp.

    Args:
        to: Recipient phone number (E.164) or phonebook nickname.
        message: Text message body.

    Returns:
        Confirmation string or error message.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

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

    chat_id = _to_chat_id(to)
    client = _get_client()
    result = client.send_text(chat_id, message)

    if result.ok:
        _record_send()
        display_to = to if not _E164_RE.match(_normalize_number(to)) else _normalize_number(to)
        return f"Message sent to {display_to} (id: {result.message_id})"
    return f"Failed to send: {result.error}"


def whatsapp_send_image(
    to: str,
    image_url: str,
    caption: str | None = None,
) -> str:
    """
    Send an image via WhatsApp.

    Args:
        to: Recipient phone number (E.164) or phonebook nickname.
        image_url: Public URL of the image.
        caption: Optional caption text.

    Returns:
        Confirmation string or error message.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

    allowed, reason = _check_contact(to)
    if not allowed:
        return f"Blocked: {reason}"

    if not _rate_limit_ok():
        return (
            f"Rate limit reached ({_cfg.rate_limit} messages/hour). "
            "Please wait before sending again."
        )

    chat_id = _to_chat_id(to)
    client = _get_client()
    result = client.send_image(chat_id, image_url, caption=caption)

    if result.ok:
        _record_send()
        display_to = to if not _E164_RE.match(_normalize_number(to)) else _normalize_number(to)
        return f"Image sent to {display_to} (id: {result.message_id})"
    return f"Failed to send image: {result.error}"


def whatsapp_check(
    contact: str | None = None,
    limit: int = 10,
) -> str:
    """
    Retrieve recent WhatsApp messages.

    Args:
        contact: Optional filter — phone number or phonebook nickname.
        limit: Max number of messages (capped at 50).

    Returns:
        Formatted list of recent messages.
    """
    if not REQUESTS_AVAILABLE:
        return "Error: requests library not available. Run: uv add requests"

    limit = min(max(1, limit), 50)
    chat_id = _to_chat_id(contact) if contact else None

    client = _get_client()

    if chat_id:
        messages = client.get_messages(chat_id=chat_id, limit=limit)
    else:
        # Waha requires chatId for /api/messages — use chats overview instead
        chats = client.get_chats_overview(limit=limit)
        messages = [c.last_message for c in chats if c.last_message is not None]

    if not messages:
        target = f" from {contact}" if contact else ""
        return f"No recent messages{target}."

    # Apply inbound contact filter (skip outgoing messages — their
    # from_number is the user's own number, not the contact)
    if _cfg.filter_mode != "none":
        messages = [m for m in messages if m.from_me or _check_receive_contact(m.from_number)]

    if not messages:
        return "No messages matching your contact filter."

    output: list[str] = [f"Recent messages ({len(messages)}):\n"]
    for msg in messages:
        direction = "You" if msg.from_me else msg.from_number.replace("@c.us", "")
        ts = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(msg.timestamp)) if msg.timestamp else "?"
        )
        body = msg.body[:300] + "..." if len(msg.body) > 300 else msg.body
        media_tag = " [media]" if msg.has_media else ""
        output.append(f"  [{ts}] {direction}: {body}{media_tag}")

    return "\n".join(output)


def whatsapp_contacts() -> str:
    """
    List the configured phonebook contacts.

    Returns:
        Formatted phonebook listing.
    """
    if not _cfg.phonebook:
        return (
            "No phonebook contacts configured.\n"
            "Add them in .cogtrix.json under services.whatsapp.phonebook."
        )

    lines = ["Phonebook contacts:\n"]
    for nick, number in sorted(_cfg.phonebook.items()):
        lines.append(f"  {nick}: {number}")

    if _cfg.filter_mode != "none":
        lines.append(f"\nFilter mode: {_cfg.filter_mode}")
        if _cfg.contacts:
            lines.append(f"Filter list: {', '.join(_cfg.contacts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# is_configured — gating for the registry
# ---------------------------------------------------------------------------


def is_configured() -> bool:
    """Return True if WhatsApp integration is usable.

    Checks:
        1. ``requests`` library is installed.
        2. A Waha URL is configured (always true — has a default).
        3. At least one capability (send or receive) is enabled.
    """
    if not REQUESTS_AVAILABLE:
        return False
    # If both capabilities are disabled, don't load the tool
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
                "name": "whatsapp_send",
                "description": (
                    "Send a text message via WhatsApp to a phone number or "
                    "phonebook contact. Requires a running Waha container.\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks you to send a WhatsApp message\n"
                    "- You need to notify someone via WhatsApp\n"
                ),
                "input_schema": WhatsAppSendInput,
                "requires_confirmation": _cfg.require_confirmation,
                "function": whatsapp_send,
                "category": "messaging",
            }
        )
        configs.append(
            {
                "name": "whatsapp_send_image",
                "description": (
                    "Send an image via WhatsApp given a public URL. "
                    "Optionally include a caption.\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks you to send an image/photo via WhatsApp\n"
                ),
                "input_schema": WhatsAppSendImageInput,
                "requires_confirmation": _cfg.require_confirmation,
                "function": whatsapp_send_image,
                "category": "messaging",
            }
        )

    if _cfg.allow_receive:
        configs.append(
            {
                "name": "whatsapp_check",
                "description": (
                    "Retrieve recent WhatsApp messages. Optionally filter "
                    "by a specific contact (phone number or phonebook nick).\n"
                    "\n"
                    "USE THIS TOOL WHEN:\n"
                    "- The user asks to check WhatsApp messages\n"
                    "- You need to read incoming messages\n"
                ),
                "input_schema": WhatsAppCheckInput,
                "requires_confirmation": False,
                "function": whatsapp_check,
                "category": "privacy",
            }
        )

    # Contacts tool is always available when any capability is enabled
    configs.append(
        {
            "name": "whatsapp_contacts",
            "description": (
                "List the phonebook contacts configured for WhatsApp. "
                "Shows nicknames, phone numbers, and active filter rules."
            ),
            "input_schema": WhatsAppContactsInput,
            "requires_confirmation": False,
            "function": whatsapp_contacts,
            "category": "readonly",
        }
    )

    _categories: dict[str, str] = {}
    for cfg in configs:
        name = cfg["name"]
        cat = cfg.get("category", "readonly")
        _categories[name] = cat
    register_tool_categories(_categories)

    return configs


TOOL_CONFIGS = _build_tool_configs()
TOOL_CONFIG = TOOL_CONFIGS[0] if TOOL_CONFIGS else {}


__all__ = [
    "whatsapp_send",
    "whatsapp_send_image",
    "whatsapp_check",
    "whatsapp_contacts",
    "is_configured",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
    "TOOL_SETUP",
    "WhatsAppSendInput",
    "WhatsAppSendImageInput",
    "WhatsAppCheckInput",
    "WhatsAppContactsInput",
]
