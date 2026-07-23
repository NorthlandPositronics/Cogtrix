"""
Channel abstraction for the Cogtrix assistant mode.

Defines the common interface that all messaging backends must implement,
and the IncomingMessage dataclass used throughout the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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

    @property
    def session_key(self) -> str:
        return f"{self.channel}::{self.chat_id}"


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
    def send(self, chat_id: str, text: str) -> bool:
        """Send *text* to *chat_id*. Returns True on success."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the channel is configured and reachable."""

    def send_typing(self, chat_id: str) -> None:  # noqa: B027
        """Send a typing indicator (optional, no-op by default)."""
