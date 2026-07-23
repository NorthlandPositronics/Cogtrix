"""
Channel abstraction for the Cogtrix assistant mode.

Defines the common interface that all messaging backends must implement,
and the IncomingMessage dataclass used throughout the pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dhms])", re.IGNORECASE)


def parse_duration(value: str | int | float | None) -> float | None:
    """Parse a human-readable duration string into seconds.

    Accepts formats like ``"24h"``, ``"1.5h"``, ``"30m"``, ``"7d"``,
    ``"1d12h"``, ``"90s"``, or a plain number (treated as seconds).
    Returns ``None`` if *value* is ``None``, empty, zero, or negative.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    value = str(value).strip()
    if not value:
        return None
    if value.startswith("-"):
        return None
    matches = _DURATION_RE.findall(value)
    if not matches:
        try:
            secs = float(value)
            return secs if secs > 0 else None
        except ValueError:
            return None
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    total = sum(float(n) * units[u.lower()] for n, u in matches)
    return float(total) if total > 0 else None


@dataclass
class IncomingMessage:
    """A normalized inbound message from any messaging channel."""

    channel: str
    chat_id: str
    message_id: str
    sender_id: str
    sender_name: str | None
    text: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved_phone: str | None = None

    @property
    def session_key(self) -> str:
        return f"{self.channel}::{self.chat_id}"


@dataclass
class SendResult:
    """Result of a send or edit operation."""

    ok: bool
    message_id: str | None = None
    error: str | None = None


class Channel(ABC):
    """Abstract base for a messaging channel backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel identifier, e.g. 'whatsapp' or 'telegram'."""

    @abstractmethod
    def poll(self) -> list[IncomingMessage]:
        """Return new inbound messages since the last call."""

    @abstractmethod
    def send(self, chat_id: str, text: str) -> SendResult:
        """Send *text* to *chat_id*. Returns SendResult with message_id on success."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the channel is configured and reachable."""

    def edit_message(self, chat_id: str, message_id: str, text: str) -> SendResult:
        """Edit a previously sent message. No-op by default (not all channels support it)."""
        return SendResult(ok=False, error="edit not supported")

    def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message. Returns True on success."""
        return False

    def archive_chat(self, chat_id: str) -> bool:
        """Archive a chat. Returns True on success."""
        return False

    def send_typing(self, chat_id: str) -> None:  # noqa: B027
        """Send a typing indicator (optional, no-op by default)."""
