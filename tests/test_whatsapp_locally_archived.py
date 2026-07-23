"""Tests for BUG-113: locally-archived WhatsApp chats must be skipped in poll.

When the blacklist filter archives a chat via ``archive_chat()``, the chat ID
is added to ``_locally_archived``.  On subsequent polls, chats in this set are
skipped regardless of the WAHA ``archived`` field (which WhatsApp resets to
``False`` when a new message arrives).
"""

from __future__ import annotations

import collections
import time
from unittest.mock import MagicMock, patch

from cogtrix_core.assistant.channels.whatsapp import WhatsAppChannel

try:
    from cogtrix_core.tools._whatsapp_client import ChatOverview, Message
except ImportError:
    import pytest

    pytest.skip("WhatsApp client not available", allow_module_level=True)


def _make_channel(filter_mode: str = "blacklist", contacts: list[str] | None = None):
    """Create a WhatsAppChannel with minimal attributes for unit testing."""
    with patch("cogtrix_core.tools._whatsapp_client.WahaClient.__init__", return_value=None):
        ch = WhatsAppChannel.__new__(WhatsAppChannel)
        ch._chat_watermarks = {}
        ch._watermark_timestamps = {}
        ch._overview_snapshot = {}
        ch._snapshot_timestamps = {}
        ch._lid_cache = collections.OrderedDict()
        ch._lid_cache_lock = MagicMock()
        ch._LID_CACHE_MAX = 1024
        ch._LID_NEGATIVE_TTL = 300.0
        ch._filter_mode = filter_mode
        ch._contacts = contacts or ["blocked_user"]
        ch._phonebook = {}
        ch._overview_limit = 50
        ch._SNAPSHOT_TTL = 3600.0
        ch._WATERMARK_TTL = 604800.0
        ch._seen_ids = {}
        ch._SEEN_TTL = 600.0
        ch._message_fetch_limit = 50
        ch._FETCH_ERROR_BASE = 30.0
        ch._FETCH_ERROR_MAX = 300.0
        ch._chat_errors = {}
        ch._ignore_archived = True
        ch._ignore_older_than = None
        ch._locally_archived = set()
        ch._archived_snapshot = set()
        ch._client = MagicMock()
        ch._session_check_interval = 60.0
        ch._last_session_check = 0.0
    return ch


def _make_overview(
    chat_id: str, msg_id: str = "m1", archived: bool = False, name: str | None = None
) -> ChatOverview:
    """Create a ChatOverview with a last_message."""
    return ChatOverview(
        id=chat_id,
        name=name or chat_id,
        last_message=Message(
            id=msg_id,
            timestamp=int(time.time()),
            from_number=chat_id,
            to=None,
            body="hello",
            from_me=False,
            has_media=False,
            media_url=None,
        ),
        archived=archived,
    )


class TestLocallyArchivedBlacklist:
    """Blacklisted chats are added to _locally_archived."""

    def test_blacklist_filter_adds_to_locally_archived(self) -> None:
        """When a message is blacklisted, the chat ID is added to _locally_archived."""
        ch = _make_channel(filter_mode="blacklist", contacts=["blocked_user"])
        chat = _make_overview("blocked_user@c.us")
        msg = chat.last_message

        ch._process_message(msg, chat, time.monotonic())

        assert "blocked_user@c.us" in ch._locally_archived
        ch._client.delete_message.assert_called_once()
        ch._client.archive_chat.assert_called_once_with("blocked_user@c.us")


class TestLocallyArchivedPolling:
    """Chats in _locally_archived are skipped during poll Phase 1."""

    def test_locally_archived_chat_skipped_even_when_not_waha_archived(self) -> None:
        """A chat in _locally_archived is skipped even when WAHA returns archived=False."""
        ch = _make_channel(filter_mode="blacklist", contacts=["blocked_user"])
        ch._locally_archived.add("blocked_user@c.us")

        overview = _make_overview("blocked_user@c.us", archived=False)
        ch._client.get_chats_overview.return_value = [overview]
        ch._client.get_chat_messages.return_value = []

        result = ch.poll()

        assert result == []
        # Should NOT have fetched messages for this chat
        ch._client.get_chat_messages.assert_not_called()

    def test_locally_archived_re_archives_when_waha_unarchived(self) -> None:
        """When WhatsApp auto-unarchives a locally-archived chat, we re-archive it."""
        ch = _make_channel()
        ch._locally_archived.add("chat@c.us")

        overview = _make_overview("chat@c.us", archived=False)
        ch._client.get_chats_overview.return_value = [overview]

        ch.poll()

        ch._client.archive_chat.assert_called_once_with("chat@c.us")

    def test_locally_archived_no_rearchive_when_already_archived(self) -> None:
        """When the chat is already archived in WAHA, don't call archive_chat again."""
        ch = _make_channel()
        ch._locally_archived.add("chat@c.us")

        overview = _make_overview("chat@c.us", archived=True)
        ch._client.get_chats_overview.return_value = [overview]

        ch.poll()

        ch._client.archive_chat.assert_not_called()

    def test_non_archived_chat_passes_through(self) -> None:
        """A chat NOT in _locally_archived is still processed normally."""
        ch = _make_channel(filter_mode="none")

        overview = _make_overview("normal_chat@c.us", archived=False)
        ch._client.get_chats_overview.return_value = [overview]
        ch._client.get_chat_messages.return_value = [overview.last_message]

        result = ch.poll()

        # Message should be processed (filter_mode=none means everyone passes)
        assert len(result) == 1


class TestUnarchiveLocally:
    """unarchive_locally() removes chat from the suppression set."""

    def test_unarchive_removes_from_set(self) -> None:
        ch = _make_channel()
        ch._locally_archived.add("chat@c.us")

        ch.unarchive_locally("chat@c.us")

        assert "chat@c.us" not in ch._locally_archived

    def test_unarchive_nonexistent_is_safe(self) -> None:
        ch = _make_channel()

        result = ch.unarchive_locally("nonexistent@c.us")
        # chat_id was not in the set — returns False (nothing was removed);
        # the call is always safe (no exception raised).
        assert result is False

    def test_chat_processed_after_unarchive(self) -> None:
        """After unarchive_locally, the chat is processed on the next poll."""
        ch = _make_channel(filter_mode="none")
        ch._locally_archived.add("chat@c.us")

        ch.unarchive_locally("chat@c.us")

        overview = _make_overview("chat@c.us", archived=False)
        ch._client.get_chats_overview.return_value = [overview]
        ch._client.get_chat_messages.return_value = [overview.last_message]

        result = ch.poll()
        assert len(result) == 1
