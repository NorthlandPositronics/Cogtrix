"""
Telegram channel for Cogtrix assistant mode.

Wraps TelegramBotClient to provide long-polling and sending via the Bot API.
Contact filtering mirrors the logic in src/tools/telegram.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.assistant.channel import Channel, IncomingMessage, SendResult, parse_duration
from src.tools._telegram_client import REQUESTS_AVAILABLE, TelegramBotClient

log = logging.getLogger("cogtrix")


_VALID_FILTER_MODES = frozenset({"none", "allow", "ignore", "blacklist"})


def _normalize_filter_mode(mode: str) -> str:
    """Map legacy filter_mode values to current names."""
    mode = mode.strip().lower()
    if mode == "whitelist":
        return "allow"
    if mode not in _VALID_FILTER_MODES:
        log.warning("Unrecognized filter_mode %r — defaulting to 'none'", mode)
        return "none"
    return mode


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
    if filter_mode == "allow":
        return cid in normalized
    if filter_mode in ("ignore", "blacklist"):
        return cid not in normalized
    return True


class TelegramChannel(Channel):
    """Telegram channel backed by a Bot API token."""

    def __init__(self, config: dict[str, Any], long_poll_timeout: int = 30) -> None:
        self._config = config
        self._bot_token: str | None = config.get("bot_token")
        self._filter_mode: str = _normalize_filter_mode(config.get("filter_mode", "none"))
        self._contacts: list[str] = config.get("contacts", [])
        self._phonebook: dict[str, str] = config.get("phonebook", {})
        self._long_poll_timeout = long_poll_timeout
        self._ignore_older_than: float | None = parse_duration(config.get("ignore_older_than"))
        self._last_update_id: int = 0
        if self._filter_mode != "none":
            log.info(
                "Telegram filter_mode=%s with %d contacts",
                self._filter_mode,
                len(self._contacts),
            )
        self._client = (
            TelegramBotClient(
                token=self._bot_token or "",
                timeout=long_poll_timeout + 10,
            )
            if self._bot_token
            else None
        )

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
        batch_max_id = self._last_update_id
        for msg in raw_messages:
            if msg.update_id > batch_max_id:
                batch_max_id = msg.update_id

            if self._ignore_older_than is not None:
                age = time.time() - msg.date
                if age > self._ignore_older_than:
                    log.debug(
                        "Skipping message %d from chat %s — too old (%.0fs > %.0fs)",
                        msg.message_id,
                        msg.chat_id,
                        age,
                        self._ignore_older_than,
                    )
                    continue

            if not _check_receive_contact(
                msg.chat_id, self._filter_mode, self._contacts, self._phonebook
            ):
                if self._filter_mode == "blacklist" and self._client is not None:
                    try:
                        self._client.delete_message(msg.chat_id, msg.message_id)
                    except Exception:
                        pass
                continue
            if not msg.text.strip():
                continue

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

        self._last_update_id = batch_max_id
        return result

    def send(self, chat_id: str, text: str) -> SendResult:
        if self._client is None:
            return SendResult(ok=False, error="Telegram client not initialized")
        result = self._client.send_message(chat_id, text)
        if not result.ok:
            log.error("Telegram send failed to %s: %s", chat_id, result.error)
        return SendResult(
            ok=result.ok,
            message_id=str(result.message_id) if result.message_id else None,
            error=result.error,
        )

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        if self._client is None:
            return SendResult(ok=False, error="Telegram client not initialized")
        try:
            msg_id_int = int(message_id)
        except (ValueError, TypeError):
            return SendResult(ok=False, error=f"Invalid message_id: {message_id}")
        result = self._client.edit_message_text(chat_id, msg_id_int, text)
        return SendResult(
            ok=result.ok,
            message_id=message_id if result.ok else None,
            error=result.error,
        )

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        if self._client is None:
            return False
        try:
            return self._client.delete_message(int(chat_id), int(message_id))
        except (ValueError, TypeError):
            return False

    def is_ready(self) -> bool:
        if not REQUESTS_AVAILABLE:
            return False
        if not self._bot_token:
            return False
        return True
