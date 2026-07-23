"""Regression tests for HTTP 429 rate-limit handling in Discord and Slack channels.

Fixes Issue #935 — Discord and Slack REST clients lack HTTP 429 Retry-After
parsing. Without this, rate-limited channels recover slowly or not at all when
the API requires a specific backoff window.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# FakeSlackApiError — simulated Slack 429 exceptions
# ---------------------------------------------------------------------------
# _FakeSlackApiErrorWithRetry inherits from Exception (not the real
# SlackApiError) because the real class has `response` as a read-only property
# that conflicts with our `response` property and its __init__ signature.
# In tests, we patch src.assistant.channels.slack.SlackApiError to point to
# these fake classes, so `isinstance(exc, SlackApiError)` returns True.


class _FakeSlackApiErrorWithRetry(Exception):
    """Simulates SlackApiError with a Retry-After header on 429.

    Inherits from the real SlackApiError so that
    ``isinstance(exc, SlackApiError)`` is True in poll()/send() handlers.
    The parent __init__ sets self.response (triggering the setter, storing in
    _response), so we override self.response afterward with a mock that has
    a .headers attribute — the format _parse_slack_retry_after expects.
    """

    def __init__(self, retry_after_seconds: float) -> None:
        # _custom_response must be a mock with .headers so that
        # _parse_slack_retry_after(exc).response.headers["Retry-After"]
        # returns the delay string.  Do this BEFORE calling super().__init__
        # so that when parent __init__ sets self.response, it stores the mock
        # in _response (via the property setter), and our override takes effect.
        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": str(retry_after_seconds)}
        self.__dict__["_custom_response"] = mock_response
        super().__init__("ratelimited", mock_response)
        # Override self.response so exc.response returns our mock (not the
        # parent's _response which is the headers dict, lacking .headers).
        self.__dict__["response"] = mock_response

    @property
    def response(self):  # type: ignore
        # Check our custom response first (stored in instance __dict__).
        if "_custom_response" in self.__dict__:
            return self.__dict__["_custom_response"]
        # Fallback to parent class's response property for non-429 paths.
        return super().response


class _FakeSlackApiErrorNoRetry(Exception):
    """Simulates SlackApiError without a Retry-After header (e.g. 404)."""

    def __init__(self) -> None:
        mock_response = MagicMock()
        mock_response.headers = {}
        self.__dict__["_custom_response"] = mock_response
        super().__init__("not_found", mock_response)
        self.__dict__["response"] = mock_response

    @property
    def response(self):  # type: ignore
        if "_custom_response" in self.__dict__:
            return self.__dict__["_custom_response"]
        return super().response


# ---------------------------------------------------------------------------
# Discord — RateLimitError exception
# ---------------------------------------------------------------------------


class TestDiscordRateLimitError:
    def test_exception_has_retry_after_and_route_attrs(self):
        from src.assistant.channels.discord import RateLimitError

        exc = RateLimitError(retry_after=10.5, route="/channels/123/messages")
        assert exc.retry_after == 10.5
        assert exc.route == "/channels/123/messages"
        assert "10.5" in str(exc)
        assert "/channels/123/messages" in str(exc)

    def test_retry_after_floored_at_1_second(self):
        from src.assistant.channels.discord import RateLimitError

        exc = RateLimitError(retry_after=0.3)
        assert exc.retry_after == 1.0

    def test_empty_route_handled_gracefully(self):
        from src.assistant.channels.discord import RateLimitError

        exc = RateLimitError(retry_after=5.0)
        assert exc.route == ""
        str(exc)  # must not raise


# ---------------------------------------------------------------------------
# Discord REST client — header parsing
# ---------------------------------------------------------------------------


class TestDiscordRestClientRateLimit:
    def test_raises_rate_limit_error_on_429(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "3"}

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._get("/channels/123/messages")
            assert exc_info.value.retry_after == 3.0
            assert exc_info.value.route == "/channels/123/messages"

    def test_parses_retry_after_header_seconds(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "7"}

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._get("/guilds/456/channels")
            assert exc_info.value.retry_after == 7.0

    def test_parses_x_rate_limit_reset_after_header_milliseconds(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"X-RateLimit-Reset-After": "2500"}  # 2500ms = 2.5s

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._get("/users/@me/guilds")
            assert exc_info.value.retry_after == 2.5

    def test_falls_back_to_5s_when_no_retry_header(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._get("/channels/789/messages")
            assert exc_info.value.retry_after == 5.0

    def test_passthrough_on_non_429_http_error(self):
        from src.assistant.channels.discord import _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {}
        mock_resp.raise_for_status.side_effect = RuntimeError("server error")

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            with pytest.raises(RuntimeError, match="server error"):
                _DiscordRestClient("tok")._get("/channels/123/messages")

    def test_post_raises_rate_limit_error_on_429(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "4"}

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._post("/channels/123/messages", {})
            assert exc_info.value.retry_after == 4.0

    def test_patch_raises_rate_limit_error_on_429(self):
        from src.assistant.channels.discord import RateLimitError, _DiscordRestClient

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "2"}

        with patch("src.assistant.channels.discord._requests") as mock_requests:
            mock_requests.patch.return_value = mock_resp
            with pytest.raises(RateLimitError) as exc_info:
                _DiscordRestClient("tok")._patch("/channels/123/messages/456", {})
            assert exc_info.value.retry_after == 2.0


# ---------------------------------------------------------------------------
# Discord channel — poll() retry on 429
# ---------------------------------------------------------------------------


class TestDiscordChannelPollRateLimit:
    def test_poll_retries_on_rate_limit_error_and_returns_messages(self):
        from src.assistant.channels.discord import DiscordChannel, RateLimitError

        mock_client = MagicMock()
        mock_client.get_messages.side_effect = [
            RateLimitError(retry_after=0.01, route="/channels/C1/messages"),
            [{"id": "999", "content": "hello", "author": {"id": "U1", "username": "user1"}}],
        ]

        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            with patch("src.assistant.channels.discord._REQUESTS_AVAILABLE", True):
                channel = DiscordChannel({"bot_token": "fake"})
        channel._client = mock_client
        channel._last_seen = {"C1": "998"}
        channel._channel_guilds = {"C1": "G1"}
        channel._seeded = True

        with patch("time.sleep"):
            result = channel.poll()

        assert len(result) == 1
        assert result[0].text == "hello"
        assert mock_client.get_messages.call_count == 2

    def test_poll_falls_back_to_warning_on_retry_failure(self):
        from src.assistant.channels.discord import DiscordChannel, RateLimitError

        mock_client = MagicMock()
        mock_client.get_messages.side_effect = [
            RateLimitError(retry_after=0.01),
            RuntimeError("still broken"),
        ]

        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            with patch("src.assistant.channels.discord._REQUESTS_AVAILABLE", True):
                channel = DiscordChannel({"bot_token": "fake"})
        channel._client = mock_client
        channel._last_seen = {"C1": "998"}
        channel._channel_guilds = {"C1": "G1"}
        channel._seeded = True

        with patch("time.sleep"):
            result = channel.poll()

        assert result == []
        assert mock_client.get_messages.call_count == 2


# ---------------------------------------------------------------------------
# Discord channel — send() retry on 429
# ---------------------------------------------------------------------------


class TestDiscordChannelSendRateLimit:
    def test_send_retries_on_rate_limit_error_and_returns_ok(self):
        from src.assistant.channels.discord import DiscordChannel, RateLimitError

        mock_client = MagicMock()
        mock_client.send_message.side_effect = [
            RateLimitError(retry_after=0.01),
            {"id": "1000"},
        ]

        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            with patch("src.assistant.channels.discord._REQUESTS_AVAILABLE", True):
                channel = DiscordChannel({"bot_token": "fake"})
        channel._client = mock_client

        with patch("time.sleep"):
            result = channel.send("C1", "test message")

        assert result.ok is True
        assert result.message_id == "1000"
        assert mock_client.send_message.call_count == 2

    def test_send_returns_error_on_retry_failure(self):
        from src.assistant.channels.discord import DiscordChannel, RateLimitError

        mock_client = MagicMock()
        mock_client.send_message.side_effect = [
            RateLimitError(retry_after=0.01),
            RuntimeError("still broken"),
        ]

        with patch("src.assistant.channels.discord._HAS_DISCORD", True):
            with patch("src.assistant.channels.discord._REQUESTS_AVAILABLE", True):
                channel = DiscordChannel({"bot_token": "fake"})
        channel._client = mock_client

        with patch("time.sleep"):
            result = channel.send("C1", "test message")

        assert result.ok is False
        assert result.error is not None
        assert "still broken" in result.error
        assert mock_client.send_message.call_count == 2


# ---------------------------------------------------------------------------
# Slack — helper
# ---------------------------------------------------------------------------


class TestSlackRetryAfterHelper:
    def test_returns_delay_from_retry_after_header(self):
        from src.assistant.channels.slack import _parse_slack_retry_after

        mock_resp = MagicMock()
        mock_resp.headers = {"Retry-After": "8.5"}

        mock_exc = MagicMock()
        mock_exc.response = mock_resp

        result = _parse_slack_retry_after(mock_exc)
        assert result == 8.5

    def test_returns_none_when_no_response_attr(self):
        from src.assistant.channels.slack import _parse_slack_retry_after

        exc = RuntimeError("not a SlackApiError")
        assert _parse_slack_retry_after(exc) is None

    def test_returns_none_when_response_has_no_headers(self):
        from src.assistant.channels.slack import _parse_slack_retry_after

        mock_exc = MagicMock()
        mock_exc.response = MagicMock()
        mock_exc.response.headers = {}

        assert _parse_slack_retry_after(mock_exc) is None

    def test_returns_none_on_invalid_header_value(self):
        from src.assistant.channels.slack import _parse_slack_retry_after

        mock_resp = MagicMock()
        mock_resp.headers = {"Retry-After": "not-a-number"}

        mock_exc = MagicMock()
        mock_exc.response = mock_resp

        assert _parse_slack_retry_after(mock_exc) is None


# ---------------------------------------------------------------------------
# Slack channel — poll() retry on 429
# ---------------------------------------------------------------------------


def _make_slack_channel_with_mock_client(mock_client: MagicMock) -> Any:
    """Create a SlackChannel with mocked _client.

    The slack module may already be imported (cached in sys.modules) with
    _HAS_SLACK = False and WebClient = None because http_retry wasn't in
    sys.modules at load time.  We handle this by patching all the relevant
    module-level names to working values before calling SlackChannel.__init__.
    """
    # Gracefully skip if slack_sdk is not installed (optional dependency).
    pytest.importorskip("slack_sdk")

    from slack_sdk import WebClient as RealWebClient
    from slack_sdk.http_retry import RetryHandler as RealRetryHandler
    from slack_sdk.http_retry.builtin_handlers import (
        RateLimitErrorRetryHandler as RealRateLimitErrorRetryHandler,
    )

    import src.assistant.channels.slack as slack_mod

    # Patch all module-level names that were set incorrectly on first import.
    # _HAS_SLACK controls the ImportError guard; WebClient/RetryHandler are
    # used in __init__ assertions and WebClient construction.
    slack_mod._HAS_SLACK = True
    slack_mod.WebClient = RealWebClient
    slack_mod.SlackApiError = _FakeSlackApiErrorWithRetry
    slack_mod.RetryHandler = RealRetryHandler
    slack_mod.RateLimitErrorRetryHandler = RealRateLimitErrorRetryHandler

    from src.assistant.channels.slack import SlackChannel

    channel = SlackChannel({"bot_token": "fake"})
    channel._client = mock_client
    return channel


class TestSlackChannelPollRateLimit:
    def test_poll_retries_on_429_and_returns_messages(self):
        mock_client = MagicMock()
        call_count = [0]

        def conversations_history_side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                raise _FakeSlackApiErrorWithRetry(0.01)
            return {"messages": [{"ts": "999000", "user": "U1", "text": "hello"}]}

        mock_client.conversations_history.side_effect = conversations_history_side_effect

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._last_ts = {"C1": "998000"}
        channel._joined_channels = ["C1"]
        channel._allowed_channels = []
        channel._seeded = True

        with patch("time.sleep"):
            result = channel.poll()

        assert len(result) == 1
        assert result[0].text == "hello"
        assert mock_client.conversations_history.call_count == 2

    def test_poll_falls_back_to_warning_on_non_429_slack_api_error(self):
        mock_client = MagicMock()
        mock_client.conversations_history.side_effect = _FakeSlackApiErrorNoRetry()

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._last_ts = {"C1": "998000"}
        channel._joined_channels = ["C1"]
        channel._allowed_channels = []
        channel._seeded = True

        result = channel.poll()
        assert result == []
        assert mock_client.conversations_history.call_count == 1

    def test_poll_falls_back_on_retry_failure_after_429(self):
        mock_client = MagicMock()
        call_count = [0]

        def conversations_history_side_effect(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                raise _FakeSlackApiErrorWithRetry(0.01)
            raise RuntimeError("still broken")

        mock_client.conversations_history.side_effect = conversations_history_side_effect

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._last_ts = {"C1": "998000"}
        channel._joined_channels = ["C1"]
        channel._allowed_channels = []
        channel._seeded = True

        with patch("time.sleep"):
            result = channel.poll()

        assert result == []
        assert mock_client.conversations_history.call_count == 2


# ---------------------------------------------------------------------------
# Slack channel — send() retry on 429
# ---------------------------------------------------------------------------


class TestSlackChannelSendRateLimit:
    def test_send_retries_on_429_and_returns_ok(self):
        mock_client = MagicMock()
        mock_client.conversations_history.return_value = {"messages": []}
        # Use a plain dict for the success response — Slack SDK returns a dict
        # with a "ts" key, and dict.get is directly usable (not callable).
        mock_success_resp = {"ts": "1000000"}
        mock_client.chat_postMessage.side_effect = [
            _FakeSlackApiErrorWithRetry(0.01),
            mock_success_resp,
        ]

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._dedup_cooldown_s = 0.0

        with patch("time.sleep"):
            result = channel.send("C1", "test message")

        assert result.ok is True
        assert result.message_id == "1000000"
        assert mock_client.chat_postMessage.call_count == 2

    def test_send_returns_error_on_non_429_slack_api_error(self):
        mock_client = MagicMock()
        mock_client.conversations_history.return_value = {"messages": []}
        mock_client.chat_postMessage.side_effect = _FakeSlackApiErrorNoRetry()

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._dedup_cooldown_s = 0.0

        result = channel.send("C1", "test message")

        assert result.ok is False
        assert mock_client.chat_postMessage.call_count == 1

    def test_send_returns_error_on_retry_failure_after_429(self):
        mock_client = MagicMock()
        mock_client.conversations_history.return_value = {"messages": []}
        call_count = [0]

        def chat_post_message_side_effect(**kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                raise _FakeSlackApiErrorWithRetry(0.01)
            raise RuntimeError("still broken")

        mock_client.chat_postMessage.side_effect = chat_post_message_side_effect

        channel = _make_slack_channel_with_mock_client(mock_client)
        channel._dedup_cooldown_s = 0.0

        with patch("time.sleep"):
            result = channel.send("C1", "test message")

        assert result.ok is False
        assert result.error is not None
        assert "still broken" in result.error
        assert mock_client.chat_postMessage.call_count == 2
