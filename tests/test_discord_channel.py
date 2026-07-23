"""
Tests for the Discord channel implementation.

All tests mock the REST client so no live network or discord.py installation
is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    msg_id: str = "1000000000000000000",
    content: str = "hello",
    author_id: str = "42",
    author_name: str = "Alice",
    *,
    is_bot: bool = False,
    global_name: str | None = None,
) -> dict[str, Any]:
    """Build a minimal Discord message dict."""
    author: dict[str, Any] = {"id": author_id, "username": author_name}
    if global_name:
        author["global_name"] = global_name
    if is_bot:
        author["bot"] = True
    return {"id": msg_id, "content": content, "author": author}


# A snowflake ID that decodes to a timestamp roughly 10 years after the Discord
# epoch (comfortably in the "not too old" range for tests that need a recent ts).
_RECENT_SNOWFLAKE = "1001000000000000000"  # ~2022-07-24


def _make_discord_channel(config: dict[str, Any] | None = None) -> Any:
    """Return a DiscordChannel with _HAS_DISCORD and _REQUESTS_AVAILABLE forced
    to True and the REST client replaced by a MagicMock."""
    with (
        patch("src.assistant.channels.discord._HAS_DISCORD", True),
        patch("src.assistant.channels.discord._REQUESTS_AVAILABLE", True),
    ):
        from src.assistant.channels.discord import DiscordChannel

        ch = DiscordChannel(config or {"bot_token": "tok"})
        ch._client = MagicMock()
        return ch


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestNormalizeFilterMode:
    def test_valid_modes_returned_unchanged(self) -> None:
        from src.assistant.channels.discord import _normalize_filter_mode

        for mode in ("none", "allow", "ignore", "blacklist"):
            assert _normalize_filter_mode(mode) == mode

    def test_whitelist_mapped_to_allow(self) -> None:
        from src.assistant.channels.discord import _normalize_filter_mode

        assert _normalize_filter_mode("whitelist") == "allow"

    def test_unknown_mode_defaults_to_none(self) -> None:
        from src.assistant.channels.discord import _normalize_filter_mode

        assert _normalize_filter_mode("rubbish") == "none"

    def test_strips_whitespace_and_lowercases(self) -> None:
        from src.assistant.channels.discord import _normalize_filter_mode

        assert _normalize_filter_mode("  ALLOW  ") == "allow"


class TestResolveContact:
    def test_phonebook_hit_case_insensitive(self) -> None:
        from src.assistant.channels.discord import _resolve_contact

        pb = {"Alice": "111", "Bob": "222"}
        assert _resolve_contact("alice", pb) == "111"

    def test_raw_id_returned_when_no_match(self) -> None:
        from src.assistant.channels.discord import _resolve_contact

        assert _resolve_contact("999", {}) == "999"


class TestCheckReceiveContact:
    def test_filter_none_always_passes(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("any", "none", [], {}) is True

    def test_allow_mode_passes_listed_id(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("42", "allow", ["42"], {}) is True

    def test_allow_mode_blocks_unlisted_id(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("99", "allow", ["42"], {}) is False

    def test_ignore_mode_blocks_listed_id(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("42", "ignore", ["42"], {}) is False

    def test_ignore_mode_passes_unlisted_id(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("99", "ignore", ["42"], {}) is True

    def test_blacklist_mode_blocks_listed(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        assert _check_receive_contact("42", "blacklist", ["42"], {}) is False

    def test_phonebook_resolution_in_filter(self) -> None:
        from src.assistant.channels.discord import _check_receive_contact

        # "alice" in contacts resolves to "111" via phonebook → should pass allow
        assert _check_receive_contact("111", "allow", ["alice"], {"alice": "111"}) is True


class TestSnowflakeToTimestamp:
    def test_epoch_zero_snowflake_gives_discord_epoch(self) -> None:
        from src.assistant.channels.discord import _snowflake_to_timestamp

        # Snowflake with all-zero timestamp bits → should equal Discord epoch (seconds)
        # Discord epoch = 1420070400000 ms = 1420070400.0 s
        ts = _snowflake_to_timestamp("0")
        assert ts == pytest.approx(1420070400.0, rel=1e-6)

    def test_recent_snowflake_gives_recent_timestamp(self) -> None:
        from src.assistant.channels.discord import _snowflake_to_timestamp

        ts = _snowflake_to_timestamp(_RECENT_SNOWFLAKE)
        # Should be well after 2020-01-01 (1577836800)
        assert ts > 1577836800.0


# ---------------------------------------------------------------------------
# DiscordChannel construction
# ---------------------------------------------------------------------------


class TestDiscordChannelInit:
    def test_raises_import_error_when_discord_not_installed(self) -> None:
        with patch("src.assistant.channels.discord._HAS_DISCORD", False):
            from src.assistant.channels.discord import DiscordChannel

            with pytest.raises(ImportError, match="discord.py not installed"):
                DiscordChannel({"bot_token": "tok"})

    def test_is_ready_true_with_token(self) -> None:
        ch = _make_discord_channel({"bot_token": "tok"})
        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            assert ch.is_ready() is True

    def test_is_ready_false_without_token(self) -> None:
        ch = _make_discord_channel({"bot_token": ""})
        assert ch.is_ready() is False

    def test_is_configured_classmethod(self) -> None:
        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            from src.assistant.channels.discord import DiscordChannel

            assert DiscordChannel.is_configured({"bot_token": "tok"}) is True
            assert DiscordChannel.is_configured({}) is False

    def test_name_property(self) -> None:
        ch = _make_discord_channel()
        assert ch.name == "discord"

    def test_allowed_guilds_cast_to_str(self) -> None:
        ch = _make_discord_channel({"bot_token": "tok", "allowed_guilds": [123, 456]})
        assert ch._allowed_guilds == ["123", "456"]

    def test_ignore_older_than_parsed(self) -> None:
        ch = _make_discord_channel({"bot_token": "tok", "ignore_older_than": "24h"})
        assert ch._ignore_older_than == pytest.approx(86400.0)


# ---------------------------------------------------------------------------
# poll() — cold-start seeding
# ---------------------------------------------------------------------------


class TestDiscordChannelColdStart:
    def test_first_poll_returns_empty_and_seeds(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = [{"id": "C1", "type": 0}]
        ch._client.get_messages.return_value = [_make_message("500")]

        result = ch.poll()

        assert result == []
        assert ch._seeded is True
        assert ch._last_seen.get("C1") == "500"

    def test_first_poll_empty_channel_no_watermark(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = [{"id": "C1", "type": 0}]
        ch._client.get_messages.return_value = []

        ch.poll()
        assert "C1" not in ch._last_seen

    def test_cold_start_message_fetch_error_still_seeds(self) -> None:
        """When channel discovery succeeds but message fetch fails, seeding still happens."""
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = [{"id": "C1", "type": 0}]
        ch._client.get_messages.side_effect = RuntimeError("network error")

        result = ch.poll()

        # Channel was discovered, so seeding succeeds despite message fetch error
        assert result == []
        assert ch._seeded is True
        # Watermark was not set because get_messages failed
        assert "C1" not in ch._last_seen

    def test_cold_start_empty_discovery_no_seed(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = []

        result = ch.poll()
        assert result == []
        # _seeded remains False because no channels were discovered
        assert ch._seeded is False

    def test_second_poll_fetches_messages(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = [{"id": "C1", "type": 0}]
        # Cold-start returns the seed message; second poll returns new message
        ch._client.get_messages.side_effect = [
            [_make_message("100")],  # seed call
            [_make_message(_RECENT_SNOWFLAKE, content="hi", author_id="7")],
        ]

        ch.poll()  # cold-start

        result = ch.poll()
        assert len(result) == 1
        assert result[0].text == "hi"
        assert result[0].sender_id == "7"


# ---------------------------------------------------------------------------
# poll() — message filtering
# ---------------------------------------------------------------------------


class TestDiscordChannelPollFiltering:
    def _seeded_channel(self, config: dict[str, Any] | None = None) -> Any:
        ch = _make_discord_channel(config or {"bot_token": "tok"})
        ch._seeded = True
        ch._channel_guilds = {"C1": "G1"}
        return ch

    def test_bot_messages_skipped(self) -> None:
        ch = self._seeded_channel()
        ch._client.get_messages.return_value = [_make_message(_RECENT_SNOWFLAKE, is_bot=True)]
        assert ch.poll() == []

    def test_empty_content_skipped(self) -> None:
        ch = self._seeded_channel()
        ch._client.get_messages.return_value = [_make_message(_RECENT_SNOWFLAKE, content="")]
        assert ch.poll() == []

    def test_watermark_advanced_to_newest_id(self) -> None:
        ch = self._seeded_channel()
        # Two messages returned newest-first (Discord order)
        ch._client.get_messages.return_value = [
            _make_message("200", content="newer"),
            _make_message("100", content="older"),
        ]
        ch.poll()
        # After poll, watermark = the newest processed id
        assert ch._last_seen["C1"] == "200"

    def test_fetch_error_does_not_crash_poll(self) -> None:
        ch = self._seeded_channel()
        ch._client.get_messages.side_effect = RuntimeError("503")
        assert ch.poll() == []

    def test_allow_filter_blocks_non_listed(self) -> None:
        ch = self._seeded_channel({"bot_token": "tok", "filter_mode": "allow", "contacts": ["42"]})
        ch._client.get_messages.return_value = [_make_message(_RECENT_SNOWFLAKE, author_id="99")]
        assert ch.poll() == []

    def test_allow_filter_passes_listed(self) -> None:
        ch = self._seeded_channel({"bot_token": "tok", "filter_mode": "allow", "contacts": ["42"]})
        ch._client.get_messages.return_value = [_make_message(_RECENT_SNOWFLAKE, author_id="42")]
        result = ch.poll()
        assert len(result) == 1

    def test_ignore_older_than_skips_old_messages(self) -> None:
        ch = self._seeded_channel({"bot_token": "tok", "ignore_older_than": "1s"})
        # Use a snowflake from 2015 (Discord epoch) — definitely older than 1s
        ch._client.get_messages.return_value = [_make_message("0", content="old")]
        assert ch.poll() == []

    def test_global_name_preferred_over_username(self) -> None:
        ch = self._seeded_channel()
        ch._client.get_messages.return_value = [
            _make_message(
                _RECENT_SNOWFLAKE,
                author_name="rawname",
                global_name="DisplayName",
            )
        ]
        result = ch.poll()
        assert result[0].sender_name == "DisplayName"

    def test_no_client_poll_returns_empty(self) -> None:
        ch = _make_discord_channel()
        ch._client = None
        assert ch.poll() == []


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------


class TestDiscordChannelSend:
    def test_send_success(self) -> None:
        ch = _make_discord_channel()
        ch._client.send_message.return_value = {"id": "MSG1"}
        result = ch.send("C1", "hello")
        assert result.ok is True
        assert result.message_id == "MSG1"

    def test_send_splits_long_message(self) -> None:
        ch = _make_discord_channel()
        ch._client.send_message.return_value = {"id": "X"}
        long_text = "A" * 4001
        ch.send("C1", long_text)
        # Should have sent 3 chunks (4001 / 2000 = ceiling 3)
        assert ch._client.send_message.call_count == 3

    def test_send_error_returns_failure(self) -> None:
        ch = _make_discord_channel()
        ch._client.send_message.side_effect = RuntimeError("403 Forbidden")
        result = ch.send("C1", "hi")
        assert result.ok is False
        assert "403" in (result.error or "")

    def test_send_no_client_returns_failure(self) -> None:
        ch = _make_discord_channel()
        ch._client = None
        result = ch.send("C1", "hi")
        assert result.ok is False


# ---------------------------------------------------------------------------
# edit_message() / delete_message()
# ---------------------------------------------------------------------------


class TestDiscordChannelEditDelete:
    def test_edit_success(self) -> None:
        ch = _make_discord_channel()
        ch._client.edit_message.return_value = {"id": "MSG1"}
        result = ch.edit_message("C1", "MSG1", "updated")
        assert result.ok is True
        assert result.message_id == "MSG1"

    def test_edit_error_returns_failure(self) -> None:
        ch = _make_discord_channel()
        ch._client.edit_message.side_effect = RuntimeError("not found")
        result = ch.edit_message("C1", "MSG1", "updated")
        assert result.ok is False

    def test_edit_no_client_returns_failure(self) -> None:
        ch = _make_discord_channel()
        ch._client = None
        result = ch.edit_message("C1", "MSG1", "x")
        assert result.ok is False

    def test_delete_success(self) -> None:
        ch = _make_discord_channel()
        ch._client.delete_message.return_value = True
        assert ch.delete_message("C1", "MSG1") is True

    def test_delete_error_returns_false(self) -> None:
        ch = _make_discord_channel()
        ch._client.delete_message.side_effect = RuntimeError("oops")
        assert ch.delete_message("C1", "MSG1") is False

    def test_delete_no_client_returns_false(self) -> None:
        ch = _make_discord_channel()
        ch._client = None
        assert ch.delete_message("C1", "MSG1") is False


# ---------------------------------------------------------------------------
# _discover_text_channels()
# ---------------------------------------------------------------------------


class TestDiscordDiscoverChannels:
    def test_filters_non_text_channel_types(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}]
        ch._client.get_guild_channels.return_value = [
            {"id": "C1", "type": 0},  # GUILD_TEXT — included
            {"id": "C2", "type": 2},  # GUILD_VOICE — excluded
            {"id": "C3", "type": 5},  # GUILD_NEWS — included
        ]
        result = ch._discover_text_channels()
        assert set(result.keys()) == {"C1", "C3"}

    def test_respects_allowed_guilds_filter(self) -> None:
        ch = _make_discord_channel({"bot_token": "tok", "allowed_guilds": ["G2"]})
        ch._client.get_guilds.return_value = [{"id": "G1"}, {"id": "G2"}]
        ch._client.get_guild_channels.return_value = [{"id": "C1", "type": 0}]
        result = ch._discover_text_channels()
        # Only G2's channel should appear
        assert set(result.keys()) == {"C1"}
        ch._client.get_guild_channels.assert_called_once_with("G2")

    def test_guild_fetch_error_returns_empty(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.side_effect = RuntimeError("401")
        assert ch._discover_text_channels() == {}

    def test_channel_fetch_error_skips_guild(self) -> None:
        ch = _make_discord_channel()
        ch._client.get_guilds.return_value = [{"id": "G1"}, {"id": "G2"}]
        ch._client.get_guild_channels.side_effect = [
            RuntimeError("forbidden"),
            [{"id": "C2", "type": 0}],
        ]
        result = ch._discover_text_channels()
        assert set(result.keys()) == {"C2"}

    def test_no_client_returns_empty(self) -> None:
        ch = _make_discord_channel()
        ch._client = None
        assert ch._discover_text_channels() == {}
