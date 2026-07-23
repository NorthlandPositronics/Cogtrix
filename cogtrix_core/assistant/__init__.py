"""
Assistant mode for Cogtrix — headless messaging service over WhatsApp and Telegram.

Maintains per-chat conversation sessions with isolated memory, polls messaging
channels for new messages, and responds using the Cogtrix agent pipeline.
"""

from cogtrix_core.assistant.channel import Channel, IncomingMessage
from cogtrix_core.assistant.service import AssistantService

__all__ = [
    "AssistantService",
    "Channel",
    "IncomingMessage",
]
