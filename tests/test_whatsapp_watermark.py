"""Tests for BUG-055: WhatsApp watermark eviction replay fix.

When _evict_stale_snapshots removes a chat's watermark entry, the next call to
_fetch_new_messages must use ``time.time() - _REACTIVATION_LOOKBACK`` as the
filter timestamp. This limits message replay to a short window (default 300 s)
rather than replaying everything since process start.

BUG-107: _startup_ts was removed as dead code; cold-start protection is handled
solely via _REACTIVATION_LOOKBACK.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_channel(config: dict | None = None):
    """Create a WhatsAppChannel with minimal real config."""
    try:
        from src.assistant.channels.whatsapp import WhatsAppChannel
    except ImportError:
        import pytest

        pytest.skip("WhatsApp channel not available")

    with patch("src.tools._whatsapp_client.WahaClient.__init__", return_value=None):
        ch = WhatsAppChannel.__new__(WhatsAppChannel)
        # Manually initialize the minimal attributes needed
        import collections

        ch._chat_watermarks = {}
        ch._watermark_timestamps = {}
        ch._snapshot_timestamps = {}
        ch._lid_cache = collections.OrderedDict()
        ch._LID_CACHE_MAX = 1024
        ch._LID_NEGATIVE_TTL = 300.0
        ch._filter_mode = "none"
        ch._contacts = []
        ch._phonebook = {}
        ch._overview_limit = 50
        ch._SNAPSHOT_TTL = 60.0 * 60
        ch._WATERMARK_TTL = 7 * 24 * 3600.0
        ch._FETCH_ERROR_BASE = 30.0
        ch._chat_errors = {}
        ch._client = MagicMock()

    return ch


class TestWatermarkEvictionFallback:
    """BUG-055: After watermark eviction, filter_ts must be based on _REACTIVATION_LOOKBACK."""

    def test_fallback_uses_reactivation_lookback(self):
        """When watermark is absent, filter_ts must be time.time() - _REACTIVATION_LOOKBACK."""
        from src.assistant.channels.whatsapp import _REACTIVATION_LOOKBACK

        ch = _make_channel()
        # Simulate an evicted watermark: no entry for this chat
        ch._chat_watermarks = {}

        fake_now = 9999.0
        captured: list[float] = []
        chat = MagicMock()
        chat.id = "chat_evicted"

        def _fake_get_messages(**kwargs):
            captured.append(kwargs.get("filter_timestamp_gte", -1.0))
            return []

        ch._client.get_chat_messages = _fake_get_messages
        with patch("src.assistant.channels.whatsapp.time.time", return_value=fake_now):
            ch._fetch_new_messages(chat)

        assert captured, "get_chat_messages was never called"
        used_ts = captured[0]

        # Must be approximately time.time() - _REACTIVATION_LOOKBACK
        expected = fake_now - _REACTIVATION_LOOKBACK
        assert (
            abs(used_ts - expected) < 1.0
        ), f"filter_ts {used_ts} not close to expected {expected}"

    def test_existing_watermark_is_used_unchanged(self):
        """When a valid watermark exists, it must be used as filter_ts unchanged."""
        ch = _make_channel()
        watermark_value = 5000.0
        chat = MagicMock()
        chat.id = "chat_with_watermark"
        ch._chat_watermarks = {"chat_with_watermark": watermark_value}

        captured: list[float] = []

        def _fake_get_messages(**kwargs):
            captured.append(kwargs.get("filter_timestamp_gte", -1.0))
            return []

        ch._client.get_chat_messages = _fake_get_messages
        with patch("src.assistant.channels.whatsapp.time.time", return_value=9999.0):
            ch._fetch_new_messages(chat)

        assert captured, "get_chat_messages was never called"
        assert captured[0] == watermark_value

    def test_reactivation_lookback_constant_is_300(self):
        """_REACTIVATION_LOOKBACK must be 300.0 seconds."""
        from src.assistant.channels.whatsapp import _REACTIVATION_LOOKBACK

        assert _REACTIVATION_LOOKBACK == 300.0
