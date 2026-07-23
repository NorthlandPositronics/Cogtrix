"""
WhatsApp channel for Cogtrix assistant mode.

Wraps WahaClient to provide polling and sending via the Waha self-hosted API.
Contact filtering mirrors the logic in src/tools/whatsapp.py.
"""

from __future__ import annotations

import logging
from typing import Any

from src.assistant.channel import Channel, IncomingMessage
from src.tools._whatsapp_client import REQUESTS_AVAILABLE, WahaClient

log = logging.getLogger("cogtrix")


def _normalize_number(number: str) -> str:
    """Normalize a phone number or Waha chatId to E.164 form."""
    key = number.strip().replace("@c.us", "").replace("@s.whatsapp.net", "")
    if key.isdigit():
        key = f"+{key}"
    return key


def _check_receive_contact(from_field: str, filter_mode: str, contacts: list[str]) -> bool:
    """Return True if the inbound sender passes the contact filter."""
    if filter_mode == "none":
        return True
    number = _normalize_number(from_field)
    normalized = {_normalize_number(c) for c in contacts}
    if filter_mode == "whitelist":
        return number in normalized
    if filter_mode == "blacklist":
        return number not in normalized
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
        self._last_seen_timestamp: int = 0
        self._seen_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "whatsapp"

    def poll(self) -> list[IncomingMessage]:
        raw_messages = self._client.get_messages(limit=20)
        result: list[IncomingMessage] = []

        for msg in raw_messages:
            if msg.from_me:
                continue
            if msg.id in self._seen_ids:
                continue
            if msg.timestamp < self._last_seen_timestamp:
                continue
            if not _check_receive_contact(msg.from_number, self._filter_mode, self._contacts):
                continue
            if not msg.body.strip():
                continue

            self._seen_ids.add(msg.id)
            if msg.timestamp > self._last_seen_timestamp:
                self._last_seen_timestamp = msg.timestamp

            sender = msg.from_number.replace("@c.us", "").replace("@s.whatsapp.net", "")
            chat_id = msg.from_number if msg.from_number else sender

            result.append(
                IncomingMessage(
                    channel=self.name,
                    chat_id=chat_id,
                    message_id=msg.id,
                    sender_id=sender,
                    sender_name=None,
                    text=msg.body,
                    timestamp=float(msg.timestamp),
                )
            )

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
