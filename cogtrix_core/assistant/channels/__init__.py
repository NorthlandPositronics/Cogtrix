"""
Built-in messaging channel implementations for Cogtrix assistant mode.
"""

from cogtrix_core.assistant.channels.discord import DiscordChannel
from cogtrix_core.assistant.channels.slack import SlackChannel
from cogtrix_core.assistant.channels.telegram import TelegramChannel
from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel

__all__ = ["DiscordChannel", "SlackChannel", "TelegramChannel", "WhatsAppChannel"]
