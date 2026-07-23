"""
Tests for src/assistant/channels/slack.py

slack_sdk is not installed in the test environment, so all tests patch
src.assistant.channels.slack._HAS_SLACK = True and inject a MagicMock
WebClient, exactly as the Discord channel tests do for requests.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(config: dict[str, Any] | None = None) -> Any:
    """Return a SlackChannel with a mocked WebClient and _HAS_SLACK=True."""
    from src.assistant.channels.slack import SlackChannel

    cfg = {"bot_token": "xoxb-test-token", **(config or {})}
    with (
        patch("src.assistant.channels.slack._HAS_SLACK", True),
        patch("src.assistant.channels.slack.WebClient") as mock_cls,
    ):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        ch = SlackChannel(cfg)
        ch._client = mock_client  # re-attach so tests can inspect calls
    return ch


def _ts(offset: float = 0.0) -> str:
    """Return a Slack-style timestamp string near now."""
    return f"{time.time() + offset:.6f}"


def _user_msg(text: str, user: str = "U123", ts: str | None = None) -> dict[str, Any]:
    return {"ts": ts or _ts(), "user": user, "text": text}


def _bot_msg(text: str, ts: str | None = None) -> dict[str, Any]:
    return {"ts": ts or _ts(), "bot_id": "B999", "text": text}


def _seed_channels(ch: Any, channel_ids: list[str]) -> None:
    """Simulate a completed cold-start so poll() enters the fetch loop."""
    ch._joined_channels = list(channel_ids)
    ch._seeded = True
    for cid in channel_ids:
        ch._last_ts[cid] = _ts(-60)  # watermark 60 s ago


# ---------------------------------------------------------------------------
# 1. Basic properties
# ---------------------------------------------------------------------------


class TestSlackChannelName:
    def test_name_returns_slack(self):
        ch = _make_channel()
        assert ch.name == "slack"


# ---------------------------------------------------------------------------
# 2. is_configured() classmethod
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_returns_false_when_token_empty(self):
        with patch("src.assistant.channels.slack._HAS_SLACK", True):
            from src.assistant.channels.slack import SlackChannel

            assert SlackChannel.is_configured({}) is False
            assert SlackChannel.is_configured({"bot_token": ""}) is False

    def test_returns_true_when_token_set(self):
        with patch("src.assistant.channels.slack._HAS_SLACK", True):
            from src.assistant.channels.slack import SlackChannel

            assert SlackChannel.is_configured({"bot_token": "xoxb-abc"}) is True


# ---------------------------------------------------------------------------
# 3. Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_raises_import_error_when_sdk_missing(self):
        with patch("src.assistant.channels.slack._HAS_SLACK", False):
            from src.assistant.channels.slack import SlackChannel

            with pytest.raises(ImportError, match="slack-sdk not installed"):
                SlackChannel({"bot_token": "xoxb-test"})


# ---------------------------------------------------------------------------
# 4. poll() — cold-start / seeding
# ---------------------------------------------------------------------------


class TestPollColdStart:
    def test_first_poll_returns_empty_and_seeds_watermarks(self):
        ch = _make_channel()
        ch._joined_channels = ["C001", "C002"]
        ch._seeded = False  # reset to trigger cold-start

        # Simulate conversations_list (used in _discover_joined_channels)
        ch._client.conversations_list.return_value = {
            "channels": [
                {"id": "C001", "is_member": True},
                {"id": "C002", "is_member": True},
            ],
            "response_metadata": {"next_cursor": ""},
        }
        # Seed: each channel has one recent message
        recent_ts = _ts(-5)
        ch._client.conversations_history.return_value = {
            "messages": [{"ts": recent_ts, "user": "U1", "text": "hi"}]
        }

        result = ch.poll()

        assert result == []
        assert ch._seeded is True
        # Watermarks must be set to the latest message ts in each channel
        assert ch._last_ts["C001"] == recent_ts
        assert ch._last_ts["C002"] == recent_ts


# ---------------------------------------------------------------------------
# 5. poll() — yields IncomingMessage with correct fields
# ---------------------------------------------------------------------------


class TestPollYieldsMessages:
    def test_poll_yields_incoming_message(self):
        ch = _make_channel()
        ts_val = _ts(-2)
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hello world", user="U42", ts=ts_val)]
        }
        # Mock user resolution
        ch._client.users_info.return_value = {
            "user": {"profile": {"display_name": "Alice", "real_name": "Alice A"}}
        }

        msgs = ch.poll()

        assert len(msgs) == 1
        m = msgs[0]
        assert m.channel == "slack"
        assert m.chat_id == "C001"
        assert m.message_id == ts_val
        assert m.sender_id == "U42"
        assert m.sender_name == "Alice"
        assert m.text == "hello world"
        assert abs(m.timestamp - float(ts_val)) < 0.001

    def test_poll_reverses_newest_first_order(self):
        """Slack returns newest-first; poll must deliver oldest-first."""
        ch = _make_channel()
        ts1, ts2 = _ts(-10), _ts(-5)
        _seed_channels(ch, ["C001"])

        # newest-first (as Slack returns)
        ch._client.conversations_history.return_value = {
            "messages": [
                _user_msg("second", ts=ts2),
                _user_msg("first", ts=ts1),
            ]
        }
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": "Bob"}}}

        msgs = ch.poll()

        assert len(msgs) == 2
        assert msgs[0].text == "first"
        assert msgs[1].text == "second"


# ---------------------------------------------------------------------------
# 6. poll() — skips bot messages
# ---------------------------------------------------------------------------


class TestPollSkipsBotMessages:
    def test_skips_message_with_bot_id(self):
        ch = _make_channel()
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {"messages": [_bot_msg("I am a bot")]}

        msgs = ch.poll()

        assert msgs == []

    def test_skips_bot_message_subtype(self):
        ch = _make_channel()
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [{"ts": _ts(-1), "subtype": "bot_message", "text": "automated"}]
        }

        msgs = ch.poll()

        assert msgs == []


# ---------------------------------------------------------------------------
# 7. poll() — ignore_older_than
# ---------------------------------------------------------------------------


class TestPollIgnoreOlderThan:
    def test_skips_message_older_than_threshold(self):
        ch = _make_channel({"ignore_older_than": "1m"})  # 60 seconds
        _seed_channels(ch, ["C001"])

        old_ts = _ts(-120)  # 2 minutes ago — should be skipped
        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("old news", ts=old_ts)]
        }

        msgs = ch.poll()

        assert msgs == []

    def test_passes_message_within_threshold(self):
        ch = _make_channel({"ignore_older_than": "1h"})
        _seed_channels(ch, ["C001"])

        recent_ts = _ts(-30)  # 30 seconds ago — within 1 hour
        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("fresh", ts=recent_ts)]
        }
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": ""}}}

        msgs = ch.poll()

        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# 8. poll() — filter_mode "allow"
# ---------------------------------------------------------------------------


class TestPollFilterAllow:
    def test_allow_mode_passes_listed_user(self):
        ch = _make_channel({"filter_mode": "allow", "contacts": ["U_ALLOWED"]})
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hi", user="U_ALLOWED")]
        }
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": ""}}}

        msgs = ch.poll()

        assert len(msgs) == 1

    def test_allow_mode_drops_unlisted_user(self):
        ch = _make_channel({"filter_mode": "allow", "contacts": ["U_ALLOWED"]})
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hi", user="U_STRANGER")]
        }

        msgs = ch.poll()

        assert msgs == []


# ---------------------------------------------------------------------------
# 9. poll() — filter_mode "ignore"
# ---------------------------------------------------------------------------


class TestPollFilterIgnore:
    def test_ignore_mode_drops_listed_user(self):
        ch = _make_channel({"filter_mode": "ignore", "contacts": ["U_IGNORED"]})
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hi", user="U_IGNORED")]
        }

        msgs = ch.poll()

        assert msgs == []

    def test_ignore_mode_passes_unlisted_user(self):
        ch = _make_channel({"filter_mode": "ignore", "contacts": ["U_IGNORED"]})
        _seed_channels(ch, ["C001"])

        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hi", user="U_OTHER")]
        }
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": ""}}}

        msgs = ch.poll()

        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# 10. poll() — allowed_channels filter
# ---------------------------------------------------------------------------


class TestPollAllowedChannels:
    def test_skips_channel_not_in_allowed_list(self):
        ch = _make_channel({"allowed_channels": ["C_ALLOWED"]})
        _seed_channels(ch, ["C_ALLOWED", "C_OTHER"])

        # Only C_ALLOWED should be fetched
        ch._client.conversations_history.return_value = {"messages": [_user_msg("hey")]}
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": ""}}}

        ch.poll()

        # Only one channel was polled (C_ALLOWED)
        assert ch._client.conversations_history.call_count == 1
        call_kwargs = ch._client.conversations_history.call_args[1]
        assert call_kwargs["channel"] == "C_ALLOWED"


# ---------------------------------------------------------------------------
# 11. poll() — phonebook resolves display name to user_id
# ---------------------------------------------------------------------------


class TestPollPhonebook:
    def test_phonebook_resolves_display_name_to_user_id(self):
        ch = _make_channel(
            {
                "filter_mode": "allow",
                "contacts": ["Alice"],
                "phonebook": {"Alice": "U_ALICE"},
            }
        )
        _seed_channels(ch, ["C001"])

        # Message from U_ALICE should pass (resolved via phonebook)
        ch._client.conversations_history.return_value = {
            "messages": [_user_msg("hello", user="U_ALICE")]
        }
        ch._client.users_info.return_value = {"user": {"profile": {"display_name": ""}}}

        msgs = ch.poll()

        assert len(msgs) == 1
        assert msgs[0].sender_id == "U_ALICE"


# ---------------------------------------------------------------------------
# 12. send()
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_calls_chat_post_message(self):
        ch = _make_channel()
        send_ts = _ts()
        ch._client.chat_postMessage.return_value = {"ok": True, "ts": send_ts}

        result = ch.send("C001", "Hello Slack")

        ch._client.chat_postMessage.assert_called_once_with(channel="C001", text="Hello Slack")
        assert result.ok is True
        assert result.message_id == send_ts

    def test_send_returns_failure_on_exception(self):
        ch = _make_channel()
        ch._client.chat_postMessage.side_effect = Exception("api error")

        result = ch.send("C001", "oops")

        assert result.ok is False
        assert result.error is not None

    def test_send_returns_failure_when_client_none(self):
        ch = _make_channel({"bot_token": ""})
        ch._client = None

        result = ch.send("C001", "hi")

        assert result.ok is False


# ---------------------------------------------------------------------------
# 13. edit_message()
# ---------------------------------------------------------------------------


class TestEditMessage:
    def test_edit_calls_chat_update_and_returns_ok(self):
        ch = _make_channel()
        edit_ts = _ts(-10)
        ch._client.chat_update.return_value = {"ok": True, "ts": edit_ts}

        result = ch.edit_message("C001", edit_ts, "updated text")

        ch._client.chat_update.assert_called_once_with(
            channel="C001", ts=edit_ts, text="updated text"
        )
        assert result.ok is True
        assert result.message_id == edit_ts

    def test_edit_returns_failure_on_exception(self):
        ch = _make_channel()
        ch._client.chat_update.side_effect = Exception("not found")

        result = ch.edit_message("C001", _ts(-5), "new text")

        assert result.ok is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# 14. delete_message()
# ---------------------------------------------------------------------------


class TestDeleteMessage:
    def test_delete_calls_chat_delete_and_returns_true(self):
        ch = _make_channel()
        ch._client.chat_delete.return_value = {"ok": True}

        ok = ch.delete_message("C001", _ts(-30))

        ch._client.chat_delete.assert_called_once()
        assert ok is True

    def test_delete_returns_false_on_exception(self):
        ch = _make_channel()
        ch._client.chat_delete.side_effect = Exception("channel_not_found")

        ok = ch.delete_message("C001", _ts(-30))

        assert ok is False

    def test_delete_returns_false_when_client_none(self):
        ch = _make_channel({"bot_token": ""})
        ch._client = None

        assert ch.delete_message("C001", "1234.5678") is False
