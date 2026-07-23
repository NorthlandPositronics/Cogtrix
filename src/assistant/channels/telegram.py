"""
Telegram channel for Cogtrix assistant mode.

Wraps TelegramBotClient to provide long-polling and sending via the Bot API.
Contact filtering mirrors the logic in src/tools/telegram.py.
"""

from __future__ import annotations

import logging
from typing import Any

from src.assistant.channel import Channel, IncomingMessage
from src.tools._telegram_client import REQUESTS_AVAILABLE, TelegramBotClient

log = logging.getLogger("cogtrix")


def _resolve_contact(name_or_id: str, phonebook: dict[str, str]) -> str:
    """Resolve a phonebook nickname or normalize a chat ID / username."""
    key = name_or_id.strip()
    for k, v in phonebook.items():
        if k.lower() == key.lower():
            return v
    return key


def _check_receive_contact(
    chat_id: int | str, filter_mode: str, contacts: list[str], phonebook: dict[str, str]
) -> bool:
    """Return True if the inbound chat passes the contact filter."""
    if filter_mode == "none":
        return True
    cid = str(chat_id)
    normalized = {_resolve_contact(c, phonebook) for c in contacts}
    if filter_mode == "whitelist":
        return cid in normalized
    if filter_mode == "blacklist":
        return cid not in normalized
    return True


class TelegramChannel(Channel):
    """Telegram channel backed by a Bot API token."""

    def __init__(self, config: dict[str, Any], long_poll_timeout: int = 30) -> None:
        self._config = config
        self._bot_token: str | None = config.get("bot_token")
        self._filter_mode: str = config.get("filter_mode", "none")
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._long_poll_timeout = long_poll_timeout
        self._last_update_id: int = 0
        self._client = TelegramBotClient(token=self._bot_token or "") if self._bot_token else None

    @property
    def name(self) -> str:
        return "telegram"

    def poll(self) -> list[IncomingMessage]:
        if self._client is None:
            return []

        offset = self._last_update_id + 1 if self._last_update_id else None
        raw_messages = self._client.get_updates(
            offset=offset,
            timeout=self._long_poll_timeout,
        )

        result: list[IncomingMessage] = []
        for msg in raw_messages:
            if not _check_receive_contact(
                msg.chat_id, self._filter_mode, self._contacts, self._phonebook
            ):
                continue
            if not msg.text.strip():
                continue

            if msg.update_id > self._last_update_id:
                self._last_update_id = msg.update_id

            sender_name = msg.from_first_name or msg.from_username
            result.append(
                IncomingMessage(
                    channel=self.name,
                    chat_id=str(msg.chat_id),
                    message_id=str(msg.message_id),
                    sender_id=str(msg.from_id or msg.chat_id),
                    sender_name=sender_name,
                    text=msg.text,
                    timestamp=float(msg.date),
                )
            )

        return result

    def send(self, chat_id: str, text: str) -> bool:
        if self._client is None:
            return False
        result = self._client.send_message(chat_id, text)
        if not result.ok:
            log.error("Telegram send failed to %s: %s", chat_id, result.error)
        return result.ok

    def is_ready(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        if not self._bot_token:
            return False
        return True
