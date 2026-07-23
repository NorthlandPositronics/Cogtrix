"""Tests for the Telegram messaging tool."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Contact filtering
# ---------------------------------------------------------------------------


class TestContactFiltering:
    """Tests for whitelist / blacklist enforcement."""

    def test_whitelist_allows_listed_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("123456789")
        assert allowed is True
        assert reason == ""

    def test_whitelist_blocks_unlisted_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("999999999")
        assert allowed is False
        assert "not in the allowed whitelist" in reason

    def test_blacklist_blocks_listed_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "blacklist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("123456789")
        assert allowed is False
        assert "blacklist" in reason

    def test_blacklist_allows_unlisted_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "blacklist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("999999999")
        assert allowed is True

    def test_filter_none_allows_all(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "none"
        _cfg.contacts = ["123456789"]

        allowed, _ = _check_contact("999999999")
        assert allowed is True

    def test_phonebook_nickname_resolved(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {"alice": "123456789"}

        allowed, _ = _check_contact("alice")
        assert allowed is True

    def test_phonebook_case_insensitive(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {"Alice": "123456789"}

        allowed, _ = _check_contact("alice")
        assert allowed is True

    def test_allow_allows_listed_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "allow"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("123456789")
        assert allowed is True
        assert reason == ""

    def test_allow_blocks_unlisted_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "allow"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("999999999")
        assert allowed is False
        assert "allow list" in reason

    def test_ignore_blocks_listed_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "ignore"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("123456789")
        assert allowed is False
        assert "ignore" in reason

    def test_ignore_allows_unlisted_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "ignore"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, _ = _check_contact("999999999")
        assert allowed is True

    def test_blacklist_blocks_listed_contact_via_check_contact(self):
        from cogtrix_core.tools.telegram import _cfg, _check_contact

        _cfg.filter_mode = "blacklist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("123456789")
        assert allowed is False
        assert "blacklist" in reason


# ---------------------------------------------------------------------------
# Contact resolution
# ---------------------------------------------------------------------------


class TestContactResolution:
    """Tests for phonebook nickname resolution."""

    def test_resolve_plain_id(self):
        from cogtrix_core.tools.telegram import _cfg, _resolve_contact

        _cfg.phonebook = {}
        assert _resolve_contact("123456789") == "123456789"

    def test_resolve_username(self):
        from cogtrix_core.tools.telegram import _cfg, _resolve_contact

        _cfg.phonebook = {}
        assert _resolve_contact("@alice") == "@alice"

    def test_resolve_nickname_to_id(self):
        from cogtrix_core.tools.telegram import _cfg, _resolve_contact

        _cfg.phonebook = {"bob": "987654321"}
        assert _resolve_contact("bob") == "987654321"

    def test_resolve_nickname_case_insensitive(self):
        from cogtrix_core.tools.telegram import _cfg, _resolve_contact

        _cfg.phonebook = {"Alice": "123456789"}
        assert _resolve_contact("alice") == "123456789"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for the in-memory rate limiter."""

    def test_rate_limit_allows_under_threshold(self):
        from cogtrix_core.tools.telegram import _cfg, _rate_limit_ok, _send_timestamps

        _send_timestamps.clear()
        _cfg.rate_limit = 5
        assert _rate_limit_ok() is True

    def test_rate_limit_blocks_at_threshold(self):
        from cogtrix_core.tools.telegram import (
            _cfg,
            _rate_limit_ok,
            _record_send,
            _send_timestamps,
        )

        _send_timestamps.clear()
        _cfg.rate_limit = 3
        for _ in range(3):
            _record_send()
        assert _rate_limit_ok() is False

    def test_rate_limit_unlimited(self):
        from cogtrix_core.tools.telegram import _cfg, _rate_limit_ok, _send_timestamps

        _send_timestamps.clear()
        _cfg.rate_limit = 0
        for _ in range(1000):
            _send_timestamps.append(0)
        assert _rate_limit_ok() is True


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    """Tests for the is_configured gating function."""

    def test_configured_when_token_present(self):
        from cogtrix_core.tools.telegram import _cfg

        _cfg.bot_token = "123456:ABC"
        _cfg.allow_send = True
        _cfg.allow_receive = True

        with patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True):
            from cogtrix_core.tools.telegram import is_configured

            assert is_configured() is True

    def test_not_configured_without_token(self):
        from cogtrix_core.tools.telegram import _cfg

        _cfg.bot_token = None
        _cfg.allow_send = True
        _cfg.allow_receive = True

        with patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True):
            from cogtrix_core.tools.telegram import is_configured

            assert is_configured() is False

    def test_not_configured_both_disabled(self):
        from cogtrix_core.tools.telegram import _cfg

        _cfg.bot_token = "123456:ABC"
        _cfg.allow_send = False
        _cfg.allow_receive = False

        with patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True):
            from cogtrix_core.tools.telegram import is_configured

            assert is_configured() is False

    def test_not_configured_no_requests(self):
        from cogtrix_core.tools.telegram import _cfg

        _cfg.bot_token = "123456:ABC"
        with patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", False):
            from cogtrix_core.tools.telegram import is_configured

            assert is_configured() is False


# ---------------------------------------------------------------------------
# Receive filter
# ---------------------------------------------------------------------------


class TestReceiveFilter:
    """Tests for inbound message filtering."""

    def test_receive_whitelist_allows(self):
        from cogtrix_core.tools.telegram import _cfg, _check_receive_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        assert _check_receive_contact(123456789) is True

    def test_receive_whitelist_blocks(self):
        from cogtrix_core.tools.telegram import _cfg, _check_receive_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]
        _cfg.phonebook = {}

        assert _check_receive_contact(999999999) is False

    def test_receive_none_allows_all(self):
        from cogtrix_core.tools.telegram import _cfg, _check_receive_contact

        _cfg.filter_mode = "none"
        assert _check_receive_contact(999999999) is True


# ---------------------------------------------------------------------------
# Tool functions (with mocked Telegram client)
# ---------------------------------------------------------------------------


class TestTelegramSend:
    """Tests for the telegram_send tool function."""

    def setup_method(self):
        from cogtrix_core.tools.telegram import _cfg, _send_timestamps

        _cfg.bot_token = "123456:ABC"
        _cfg.filter_mode = "none"
        _cfg.rate_limit = 0
        _cfg.max_message_length = 4096
        _cfg.phonebook = {}
        _send_timestamps.clear()

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_send_success(self, mock_get_client):
        from cogtrix_core.tools._telegram_client import SendResult
        from cogtrix_core.tools.telegram import telegram_send

        mock_client = MagicMock()
        mock_client.send_message.return_value = SendResult(ok=True, message_id=42)
        mock_get_client.return_value = mock_client

        result = telegram_send("123456789", "Hello!")
        assert "sent" in result.lower()
        assert "42" in result
        mock_client.send_message.assert_called_once_with("123456789", "Hello!")

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_send_failure(self, mock_get_client):
        from cogtrix_core.tools._telegram_client import SendResult
        from cogtrix_core.tools.telegram import telegram_send

        mock_client = MagicMock()
        mock_client.send_message.return_value = SendResult(ok=False, error="Chat not found")
        mock_get_client.return_value = mock_client

        result = telegram_send("123456789", "Hello!")
        assert "failed" in result.lower()

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    def test_send_blocked_by_whitelist(self):
        from cogtrix_core.tools.telegram import _cfg, telegram_send

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["000000000"]

        result = telegram_send("123456789", "Hello!")
        assert "blocked" in result.lower()

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    def test_send_rate_limited(self):
        from cogtrix_core.tools.telegram import _cfg, _record_send, telegram_send

        _cfg.rate_limit = 2
        _record_send()
        _record_send()

        result = telegram_send("123456789", "Hello!")
        assert "rate limit" in result.lower()

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_send_truncates_long_message(self, mock_get_client):
        from cogtrix_core.tools._telegram_client import SendResult
        from cogtrix_core.tools.telegram import _cfg, telegram_send

        _cfg.max_message_length = 10

        mock_client = MagicMock()
        mock_client.send_message.return_value = SendResult(ok=True, message_id=1)
        mock_get_client.return_value = mock_client

        telegram_send("123456789", "A" * 100)
        call_args = mock_client.send_message.call_args
        assert len(call_args[0][1]) == 10

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    def test_send_without_token(self):
        from cogtrix_core.tools.telegram import _cfg, telegram_send

        _cfg.bot_token = None
        result = telegram_send("123456789", "Hello!")
        assert "not configured" in result.lower()


class TestTelegramCheck:
    """Tests for the telegram_check tool function."""

    def setup_method(self):
        from cogtrix_core.tools.telegram import _cfg

        _cfg.bot_token = "123456:ABC"
        _cfg.filter_mode = "none"
        _cfg.phonebook = {}

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_check_returns_messages(self, mock_get_client):
        from cogtrix_core.tools._telegram_client import TelegramMessage
        from cogtrix_core.tools.telegram import telegram_check

        mock_client = MagicMock()
        mock_client.get_updates.return_value = [
            TelegramMessage(
                message_id=1,
                date=1700000000,
                chat_id=123456789,
                chat_title="Alice",
                from_username="alice",
                text="Hi there!",
            ),
        ]
        mock_get_client.return_value = mock_client

        result = telegram_check()
        assert "Hi there!" in result
        assert "alice" in result

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_check_no_messages(self, mock_get_client):
        from cogtrix_core.tools.telegram import telegram_check

        mock_client = MagicMock()
        mock_client.get_updates.return_value = []
        mock_get_client.return_value = mock_client

        result = telegram_check()
        assert "no recent" in result.lower()

    @patch("cogtrix_core.tools.telegram.REQUESTS_AVAILABLE", True)
    @patch("cogtrix_core.tools.telegram._get_client")
    def test_check_filters_by_contact(self, mock_get_client):
        from cogtrix_core.tools._telegram_client import TelegramMessage
        from cogtrix_core.tools.telegram import _cfg, telegram_check

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]

        mock_client = MagicMock()
        mock_client.get_updates.return_value = [
            TelegramMessage(
                message_id=1,
                date=1700000000,
                chat_id=123456789,
                from_username="alice",
                text="Allowed",
            ),
            TelegramMessage(
                message_id=2,
                date=1700000001,
                chat_id=999999999,
                from_username="unknown",
                text="Filtered out",
            ),
        ]
        mock_get_client.return_value = mock_client

        result = telegram_check()
        assert "Allowed" in result
        assert "Filtered out" not in result


class TestTelegramContacts:
    """Tests for the telegram_contacts tool function."""

    def test_contacts_empty_phonebook(self):
        from cogtrix_core.tools.telegram import _cfg, telegram_contacts

        _cfg.phonebook = {}
        result = telegram_contacts()
        assert "no telegram phonebook" in result.lower()

    def test_contacts_with_entries(self):
        from cogtrix_core.tools.telegram import _cfg, telegram_contacts

        _cfg.phonebook = {
            "alice": "123456789",
            "team": "-1001234567890",
        }
        _cfg.filter_mode = "none"

        result = telegram_contacts()
        assert "alice" in result
        assert "123456789" in result
        assert "team" in result

    def test_contacts_shows_filter_mode(self):
        from cogtrix_core.tools.telegram import _cfg, telegram_contacts

        _cfg.phonebook = {"alice": "123456789"}
        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["123456789"]

        result = telegram_contacts()
        assert "whitelist" in result.lower()


# ---------------------------------------------------------------------------
# Telegram Bot client unit tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestTelegramBotClient:
    """Tests for the TelegramBotClient HTTP wrapper."""

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_send_message_success(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 42},
        }
        mock_requests.post.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        result = client.send_message(123456789, "Hello")
        assert result.ok is True
        assert result.message_id == 42

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_send_message_error(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": False,
            "description": "Bad Request: chat not found",
        }
        mock_requests.post.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        result = client.send_message(123456789, "Hello")
        assert result.ok is False
        assert "chat not found" in (result.error or "")

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_send_message_connection_error(self, mock_requests):
        import requests as real_requests

        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_requests.post.side_effect = real_requests.exceptions.ConnectionError()
        mock_requests.exceptions = real_requests.exceptions

        client = TelegramBotClient(token="123456:ABC")
        result = client.send_message(123456789, "Hello")
        assert result.ok is False
        assert "connect" in (result.error or "").lower()

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_get_updates(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": [
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 10,
                        "date": 1700000000,
                        "chat": {"id": 123456789, "first_name": "Alice"},
                        "from": {"id": 111, "username": "alice"},
                        "text": "Hello bot",
                    },
                }
            ],
        }
        mock_requests.get.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        messages = client.get_updates(limit=10)
        assert len(messages) == 1
        assert messages[0].text == "Hello bot"
        assert messages[0].from_username == "alice"
        assert messages[0].chat_id == 123456789

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_get_me_success(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": {
                "id": 123456,
                "is_bot": True,
                "first_name": "Test Bot",
                "username": "test_bot",
            },
        }
        mock_requests.get.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        info = client.get_me()
        assert info.username == "test_bot"
        assert info.first_name == "Test Bot"

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_is_ready_success(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": {
                "id": 123456,
                "is_bot": True,
                "first_name": "Test Bot",
                "username": "test_bot",
            },
        }
        mock_requests.get.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        assert client.is_ready() is True

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_is_ready_failure(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_requests.get.side_effect = Exception("Network error")

        client = TelegramBotClient(token="invalid")
        assert client.is_ready() is False

    @patch("cogtrix_core.tools._telegram_client.requests")
    def test_send_photo_success(self, mock_requests):
        from cogtrix_core.tools._telegram_client import TelegramBotClient

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "result": {"message_id": 99},
        }
        mock_requests.post.return_value = mock_resp

        client = TelegramBotClient(token="123456:ABC")
        result = client.send_photo(123456789, "https://example.com/img.jpg", caption="Test")
        assert result.ok is True
        assert result.message_id == 99
