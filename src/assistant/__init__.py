"""
Assistant mode for Cogtrix — headless messaging service over WhatsApp and Telegram.

Maintains per-chat conversation sessions with isolated memory, polls messaging
channels for new messages, and responds using the Cogtrix agent pipeline.
"""

from src.assistant.channel import Channel, IncomingMessage
from src.assistant.service import AssistantService

__all__ = [
    "AssistantService",
    "Channel",
    "IncomingMessage",
]
