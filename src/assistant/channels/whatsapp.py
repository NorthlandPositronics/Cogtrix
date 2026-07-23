"""
WhatsApp channel for Cogtrix assistant mode.

Wraps WahaClient to provide polling and sending via the Waha self-hosted API.
Contact filtering mirrors the logic in src/tools/whatsapp.py.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.assistant.channel import Channel, IncomingMessage
from src.tools._whatsapp_client import REQUESTS_AVAILABLE, WahaClient

log = logging.getLogger("cogtrix")


def _normalize_number(number: str) -> str:
    """Normalize a phone number or Waha chatId to E.164 form.

    Handles ``@c.us``, ``@s.whatsapp.net``, and ``@lid`` (Linked ID) formats.
    LID-format identifiers are returned as-is since they can't be mapped to
    phone numbers.
    """
    key = number.strip()
    if "@lid" in key:
        return key
    key = key.replace("@c.us", "").replace("@s.whatsapp.net", "")
    if key.isdigit():
        key = f"+{key}"
    return key


def _extract_phone_from_name(name: str | None) -> str | None:
    """Try to extract an E.164 phone number from a chat display name.

    Waha often sets ``chat.name`` to a formatted phone number for contacts
    that aren't saved in the phone book (e.g. ``+971 50 406 9790``).
    Returns the normalized number or ``None`` if the name doesn't look like one.
    """
    if not name:
        return None
    digits = re.sub(r"[^0-9+]", "", name)
    if not digits:
        return None
    if digits.startswith("+") and len(digits) >= 8:
        return digits
    if digits.isdigit() and len(digits) >= 7:
        return f"+{digits}"
    return None


def _check_receive_contact(
    from_field: str,
    filter_mode: str,
    contacts: list[str],
    chat_id: str = "",
    chat_name: str | None = None,
    phonebook: dict[str, str] | None = None,
) -> bool:
    """Return True if the inbound sender passes the contact filter.

    Builds a set of candidate identifiers from the message metadata and checks
    them against the contacts list.  For ``@lid`` senders whose phone number
    is not available, the phonebook is consulted: if any phonebook *value*
    (a phone number in the contacts list) is mapped from a *key* that
    case-insensitively matches the ``chat_name``, the sender is considered
    to match that contact entry.
    """
    if filter_mode == "none":
        return True

    candidates = {_normalize_number(from_field)}
    if chat_id and chat_id != from_field:
        candidates.add(_normalize_number(chat_id))
    phone_from_name = _extract_phone_from_name(chat_name)
    if phone_from_name:
        candidates.add(phone_from_name)

    # Phonebook reverse-lookup: match chat display name to a phonebook key
    # so that "@lid" contacts whose name matches a phonebook entry (e.g.
    # phonebook "me" → "+971503308667" and chat name "D A") can be resolved.
    # This doesn't help directly — but if the user adds an entry like
    # "d a" → "+971503308667", it will match.
    #
    # More usefully: allow the contacts list to contain raw LID identifiers
    # (e.g. "178774490505455@lid") alongside phone numbers.
    if phonebook and chat_name:
        name_lower = chat_name.strip().lower()
        for pb_key, pb_value in phonebook.items():
            if pb_key.strip().lower() == name_lower:
                candidates.add(_normalize_number(pb_value))

    resolved_contacts: list[str] = []
    for c in contacts:
        if phonebook and c in phonebook:
            resolved_contacts.append(phonebook[c])
        else:
            resolved_contacts.append(c)
    normalized_contacts = {_normalize_number(c) for c in resolved_contacts}

    if filter_mode == "whitelist":
        return bool(candidates & normalized_contacts)
    if filter_mode == "blacklist":
        return not bool(candidates & normalized_contacts)
    return True


class WhatsAppChannel(Channel):
    """WhatsApp channel backed by a Waha instance."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._client = WahaClient(
            base_url=config.get("waha_url", "http://localhost:3000"),
            api_key=config.get("api_key"),
            session=config.get("session", "default"),
        )
        self._filter_mode: str = config.get("filter_mode", "none")
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._lid_cache: dict[str, str | None] = {}
        self._last_seen_timestamp: int = 0
        self._seen_ids: dict[str, float] = {}
        self._SEEN_TTL: float = 600.0
        self._session_check_interval: float = 60.0
        self._last_session_check: float = 0.0

    @property
    def name(self) -> str:
        return "whatsapp"

    def _resolve_lid(self, lid: str) -> str | None:
        """Resolve a ``@lid`` identifier to a phone number, with caching."""
        if lid in self._lid_cache:
            return self._lid_cache[lid]
        pn = self._client.resolve_lid(lid)
        if pn:
            self._lid_cache[lid] = pn
            log.debug("Resolved LID %s → %s", lid, pn)
        return pn

    def _ensure_session(self) -> None:
        """Check if the Waha session is active and start it if not.

        Throttled to run at most once per ``_session_check_interval`` seconds
        to avoid hammering the Waha API on every poll cycle.
        """
        now = time.monotonic()
        if now - self._last_session_check < self._session_check_interval:
            return
        self._last_session_check = now
        try:
            info = self._client.get_session()
            if info.status == "WORKING":
                return
            log.warning("Waha session '%s' is %s — attempting restart", info.name, info.status)
            if self._client.start_session():
                log.info("Waha session '%s' start request sent", self._client.session)
            else:
                log.error("Failed to start Waha session '%s'", self._client.session)
        except Exception as exc:
            log.debug("Waha session check failed: %s", exc)

    def poll(self) -> list[IncomingMessage]:
        self._ensure_session()
        chats = self._client.get_chats_overview(limit=50)
        result: list[IncomingMessage] = []
        batch_max_ts = self._last_seen_timestamp

        for chat in chats:
            msg = chat.last_message
            if msg is None:
                continue
            if msg.from_me:
                continue
            if msg.timestamp <= self._last_seen_timestamp:
                continue

            # Advance high-water mark for ALL messages so filtered chats
            # don't reappear on the next poll cycle.
            if msg.timestamp > batch_max_ts:
                batch_max_ts = msg.timestamp

            # Resolve @lid identifiers to phone numbers for whitelist matching
            resolved_from = msg.from_number
            resolved_phone: str | None = None
            if "@lid" in msg.from_number:
                pn = self._resolve_lid(msg.from_number)
                if pn:
                    resolved_from = pn
                    resolved_phone = pn
            resolved_chat_id = chat.id
            if "@lid" in chat.id and chat.id != msg.from_number:
                pn = self._resolve_lid(chat.id)
                if pn:
                    resolved_chat_id = pn

            if not _check_receive_contact(
                resolved_from,
                self._filter_mode,
                self._contacts,
                chat_id=resolved_chat_id,
                chat_name=chat.name,
                phonebook=self._phonebook,
            ):
                log.debug(
                    "Filtered out message from chat_id=%s name=%r (from=%s)",
                    chat.id,
                    chat.name,
                    msg.from_number,
                )
                continue
            if not msg.body.strip():
                continue

            now = time.monotonic()
            if len(self._seen_ids) > 100:
                cutoff = now - self._SEEN_TTL
                self._seen_ids = {k: v for k, v in self._seen_ids.items() if v > cutoff}
            if msg.id in self._seen_ids:
                continue
            self._seen_ids[msg.id] = now

            sender = msg.from_number
            for suffix in ("@c.us", "@s.whatsapp.net", "@lid"):
                sender = sender.replace(suffix, "")

            result.append(
                IncomingMessage(
                    channel=self.name,
                    chat_id=chat.id,
                    message_id=msg.id,
                    sender_id=sender,
                    sender_name=chat.name,
                    text=msg.body,
                    timestamp=float(msg.timestamp),
                    resolved_phone=resolved_phone,
                )
            )

        self._last_seen_timestamp = batch_max_ts
        return result

    def send(self, chat_id: str, text: str) -> bool:
        result = self._client.send_text(chat_id, text)
        if not result.ok:
            log.error("WhatsApp send failed to %s: %s", chat_id, result.error)
        return result.ok

    def is_ready(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        allow_receive = self._config.get("allow_receive", True)
        if not allow_receive:
            return False
        try:
            return self._client.is_ready()
        except Exception:
            return False
