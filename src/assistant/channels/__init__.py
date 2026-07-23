"""
Built-in messaging channel implementations for Cogtrix assistant mode.
"""

from src.assistant.channels.discord import DiscordChannel
from src.assistant.channels.slack import SlackChannel
from src.assistant.channels.telegram import TelegramChannel
from src.assistant.channels.whatsapp import WhatsAppChannel

__all__ = ["DiscordChannel", "SlackChannel", "TelegramChannel", "WhatsAppChannel"]
