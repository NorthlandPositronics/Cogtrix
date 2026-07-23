"""Tests for the WhatsApp messaging tool."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Contact filtering
# ---------------------------------------------------------------------------


class TestContactFiltering:
    """Tests for whitelist / blacklist enforcement."""

    def test_whitelist_allows_listed_contact(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("+14155551234")
        assert allowed is True
        assert reason == ""

    def test_whitelist_blocks_unlisted_contact(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("+19999999999")
        assert allowed is False
        assert "not in the allowed whitelist" in reason

    def test_blacklist_blocks_listed_contact(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "blacklist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("+14155551234")
        assert allowed is False
        assert "blacklist" in reason

    def test_blacklist_allows_unlisted_contact(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "blacklist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        allowed, reason = _check_contact("+19999999999")
        assert allowed is True

    def test_filter_none_allows_all(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "none"
        _cfg.contacts = ["+14155551234"]

        allowed, _ = _check_contact("+19999999999")
        assert allowed is True

    def test_phonebook_nickname_resolved(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {"alice": "+14155551234"}

        allowed, _ = _check_contact("alice")
        assert allowed is True

    def test_phonebook_case_insensitive(self):
        from src.tools.whatsapp import _cfg, _check_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {"Alice": "+14155551234"}

        allowed, _ = _check_contact("alice")
        assert allowed is True


# ---------------------------------------------------------------------------
# Number normalization & chatId conversion
# ---------------------------------------------------------------------------


class TestNumberNormalization:
    """Tests for phone number handling."""

    def test_normalize_e164(self):
        from src.tools.whatsapp import _normalize_number

        assert _normalize_number("+14155551234") == "+14155551234"

    def test_normalize_bare_digits(self):
        from src.tools.whatsapp import _cfg, _normalize_number

        _cfg.phonebook = {}
        result = _normalize_number("14155551234")
        assert result == "+14155551234"

    def test_normalize_with_chat_id_suffix(self):
        from src.tools.whatsapp import _cfg, _normalize_number

        _cfg.phonebook = {}
        result = _normalize_number("14155551234@c.us")
        assert result == "+14155551234"

    def test_normalize_with_whatsapp_net_suffix(self):
        from src.tools.whatsapp import _cfg, _normalize_number

        _cfg.phonebook = {}
        result = _normalize_number("14155551234@s.whatsapp.net")
        assert result == "+14155551234"

    def test_to_chat_id(self):
        from src.tools.whatsapp import _cfg, _to_chat_id

        _cfg.phonebook = {}
        assert _to_chat_id("+14155551234") == "14155551234@c.us"

    def test_to_chat_id_from_nickname(self):
        from src.tools.whatsapp import _cfg, _to_chat_id

        _cfg.phonebook = {"bob": "+442071234567"}
        assert _to_chat_id("bob") == "442071234567@c.us"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for the in-memory rate limiter."""

    def test_rate_limit_allows_under_threshold(self):
        from src.tools.whatsapp import _cfg, _rate_limit_ok, _send_timestamps

        _send_timestamps.clear()
        _cfg.rate_limit = 5
        assert _rate_limit_ok() is True

    def test_rate_limit_blocks_at_threshold(self):

        from src.tools.whatsapp import (
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
        from src.tools.whatsapp import _cfg, _rate_limit_ok, _send_timestamps

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

    def test_configured_when_requests_available(self):
        from src.tools.whatsapp import _cfg

        _cfg.allow_send = True
        _cfg.allow_receive = True

        with patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True):
            from src.tools.whatsapp import is_configured

            assert is_configured() is True

    def test_not_configured_both_disabled(self):
        from src.tools.whatsapp import _cfg

        _cfg.allow_send = False
        _cfg.allow_receive = False

        with patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True):
            from src.tools.whatsapp import is_configured

            assert is_configured() is False

    def test_not_configured_no_requests(self):
        with patch("src.tools.whatsapp.REQUESTS_AVAILABLE", False):
            from src.tools.whatsapp import is_configured

            assert is_configured() is False


# ---------------------------------------------------------------------------
# Receive filter
# ---------------------------------------------------------------------------


class TestReceiveFilter:
    """Tests for inbound message filtering."""

    def test_receive_whitelist_allows(self):
        from src.tools.whatsapp import _cfg, _check_receive_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        assert _check_receive_contact("14155551234@c.us") is True

    def test_receive_whitelist_blocks(self):
        from src.tools.whatsapp import _cfg, _check_receive_contact

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]
        _cfg.phonebook = {}

        assert _check_receive_contact("99999999999@c.us") is False

    def test_receive_none_allows_all(self):
        from src.tools.whatsapp import _cfg, _check_receive_contact

        _cfg.filter_mode = "none"
        assert _check_receive_contact("anything@c.us") is True


# ---------------------------------------------------------------------------
# Tool functions (with mocked Waha client)
# ---------------------------------------------------------------------------


class TestWhatsAppSend:
    """Tests for the whatsapp_send tool function."""

    def setup_method(self):
        from src.tools.whatsapp import _cfg, _send_timestamps

        _cfg.filter_mode = "none"
        _cfg.rate_limit = 0
        _cfg.max_message_length = 4096
        _cfg.phonebook = {}
        _send_timestamps.clear()

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_send_success(self, mock_get_client):
        from src.tools._whatsapp_client import SendResult
        from src.tools.whatsapp import whatsapp_send

        mock_client = MagicMock()
        mock_client.send_text.return_value = SendResult(ok=True, message_id="abc123")
        mock_get_client.return_value = mock_client

        result = whatsapp_send("+14155551234", "Hello!")
        assert "sent" in result.lower()
        assert "abc123" in result
        mock_client.send_text.assert_called_once_with("14155551234@c.us", "Hello!")

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_send_failure(self, mock_get_client):
        from src.tools._whatsapp_client import SendResult
        from src.tools.whatsapp import whatsapp_send

        mock_client = MagicMock()
        mock_client.send_text.return_value = SendResult(ok=False, error="Connection refused")
        mock_get_client.return_value = mock_client

        result = whatsapp_send("+14155551234", "Hello!")
        assert "failed" in result.lower()

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    def test_send_blocked_by_whitelist(self):
        from src.tools.whatsapp import _cfg, whatsapp_send

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+10000000000"]

        result = whatsapp_send("+14155551234", "Hello!")
        assert "blocked" in result.lower()

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    def test_send_rate_limited(self):
        from src.tools.whatsapp import _cfg, _record_send, whatsapp_send

        _cfg.rate_limit = 2
        _record_send()
        _record_send()

        result = whatsapp_send("+14155551234", "Hello!")
        assert "rate limit" in result.lower()

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_send_truncates_long_message(self, mock_get_client):
        from src.tools._whatsapp_client import SendResult
        from src.tools.whatsapp import _cfg, whatsapp_send

        _cfg.max_message_length = 10

        mock_client = MagicMock()
        mock_client.send_text.return_value = SendResult(ok=True, message_id="x")
        mock_get_client.return_value = mock_client

        whatsapp_send("+14155551234", "A" * 100)
        call_args = mock_client.send_text.call_args
        assert len(call_args[0][1]) == 10


class TestWhatsAppCheck:
    """Tests for the whatsapp_check tool function."""

    def setup_method(self):
        from src.tools.whatsapp import _cfg

        _cfg.filter_mode = "none"
        _cfg.phonebook = {}

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_check_returns_messages(self, mock_get_client):
        from src.tools._whatsapp_client import Message
        from src.tools.whatsapp import whatsapp_check

        mock_client = MagicMock()
        mock_client.get_messages.return_value = [
            Message(
                id="msg1",
                timestamp=1700000000,
                from_number="14155551234@c.us",
                body="Hi there!",
            ),
        ]
        mock_get_client.return_value = mock_client

        result = whatsapp_check()
        assert "Hi there!" in result
        assert "14155551234" in result

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_check_no_messages(self, mock_get_client):
        from src.tools.whatsapp import whatsapp_check

        mock_client = MagicMock()
        mock_client.get_messages.return_value = []
        mock_get_client.return_value = mock_client

        result = whatsapp_check()
        assert "no recent messages" in result.lower()

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_check_filters_by_contact(self, mock_get_client):
        from src.tools._whatsapp_client import Message
        from src.tools.whatsapp import _cfg, whatsapp_check

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]

        mock_client = MagicMock()
        mock_client.get_messages.return_value = [
            Message(
                id="msg1",
                timestamp=1700000000,
                from_number="14155551234@c.us",
                body="Allowed",
            ),
            Message(
                id="msg2",
                timestamp=1700000001,
                from_number="99999999999@c.us",
                body="Filtered out",
            ),
        ]
        mock_get_client.return_value = mock_client

        result = whatsapp_check()
        assert "Allowed" in result
        assert "Filtered out" not in result

    @patch("src.tools.whatsapp.REQUESTS_AVAILABLE", True)
    @patch("src.tools.whatsapp._get_client")
    def test_check_preserves_outgoing_messages_with_whitelist(self, mock_get_client):
        """Outgoing messages (from_me=True) must not be dropped by the contact filter."""
        from src.tools._whatsapp_client import Message
        from src.tools.whatsapp import _cfg, whatsapp_check

        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]

        mock_client = MagicMock()
        mock_client.get_messages.return_value = [
            Message(
                id="msg_out",
                timestamp=1700000000,
                from_number="my_own_number@c.us",
                body="Outgoing reply",
                from_me=True,
            ),
            Message(
                id="msg_in",
                timestamp=1700000001,
                from_number="14155551234@c.us",
                body="Incoming allowed",
            ),
        ]
        mock_get_client.return_value = mock_client

        result = whatsapp_check()
        assert "Outgoing reply" in result
        assert "Incoming allowed" in result


class TestWhatsAppContacts:
    """Tests for the whatsapp_contacts tool function."""

    def test_contacts_empty_phonebook(self):
        from src.tools.whatsapp import _cfg, whatsapp_contacts

        _cfg.phonebook = {}
        result = whatsapp_contacts()
        assert "no phonebook" in result.lower()

    def test_contacts_with_entries(self):
        from src.tools.whatsapp import _cfg, whatsapp_contacts

        _cfg.phonebook = {
            "alice": "+14155551234",
            "bob": "+442071234567",
        }
        _cfg.filter_mode = "none"

        result = whatsapp_contacts()
        assert "alice" in result
        assert "+14155551234" in result
        assert "bob" in result

    def test_contacts_shows_filter_mode(self):
        from src.tools.whatsapp import _cfg, whatsapp_contacts

        _cfg.phonebook = {"alice": "+14155551234"}
        _cfg.filter_mode = "whitelist"
        _cfg.contacts = ["+14155551234"]

        result = whatsapp_contacts()
        assert "whitelist" in result.lower()


# ---------------------------------------------------------------------------
# TOOL_CONFIGS structure
# ---------------------------------------------------------------------------


class TestToolConfigs:
    """Tests for the dynamically built TOOL_CONFIGS."""

    def test_tool_configs_have_required_fields(self):
        from src.tools.whatsapp import TOOL_CONFIGS

        for cfg in TOOL_CONFIGS:
            assert "name" in cfg
            assert "description" in cfg
            assert "input_schema" in cfg
            assert "requires_confirmation" in cfg
            assert "function" in cfg

    def test_send_tool_requires_confirmation_by_default(self):
        from src.tools.whatsapp import TOOL_CONFIGS

        send_tools = [c for c in TOOL_CONFIGS if "send" in c["name"]]
        for tool in send_tools:
            assert tool["requires_confirmation"] is True

    def test_check_tool_no_confirmation(self):
        from src.tools.whatsapp import TOOL_CONFIGS

        check_tools = [
            c for c in TOOL_CONFIGS if c["name"] in ("whatsapp_check", "whatsapp_contacts")
        ]
        for tool in check_tools:
            assert tool["requires_confirmation"] is False


# ---------------------------------------------------------------------------
# Waha client unit tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestWahaClient:
    """Tests for the WahaClient HTTP wrapper."""

    @patch("src.tools._whatsapp_client.requests")
    def test_send_text_success(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "msg_abc"}
        mock_requests.post.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000")
        result = client.send_text("14155551234@c.us", "Hello")
        assert result.ok is True
        assert result.message_id == "msg_abc"

    @patch("src.tools._whatsapp_client.requests")
    def test_send_text_http_error(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_requests.post.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000")
        result = client.send_text("14155551234@c.us", "Hello")
        assert result.ok is False
        assert "500" in (result.error or "")

    @patch("src.tools._whatsapp_client.requests")
    def test_send_text_connection_error(self, mock_requests):
        import requests as real_requests

        from src.tools._whatsapp_client import WahaClient

        mock_requests.post.side_effect = real_requests.exceptions.ConnectionError()
        mock_requests.exceptions = real_requests.exceptions

        client = WahaClient(base_url="http://localhost:3000")
        result = client.send_text("14155551234@c.us", "Hello")
        assert result.ok is False
        assert "connect" in (result.error or "").lower()

    @patch("src.tools._whatsapp_client.requests")
    def test_get_messages(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "id": "msg1",
                "timestamp": 1700000000,
                "from": "14155551234@c.us",
                "body": "Hello",
                "fromMe": False,
                "hasMedia": False,
            }
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000")
        messages = client.get_messages(limit=10)
        assert len(messages) == 1
        assert messages[0].body == "Hello"
        assert messages[0].from_number == "14155551234@c.us"

    @patch("src.tools._whatsapp_client.requests")
    def test_is_ready_working(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "default",
            "status": "WORKING",
            "me": {"id": "123@c.us"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000")
        assert client.is_ready() is True

    @patch("src.tools._whatsapp_client.requests")
    def test_is_ready_not_working(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "name": "default",
            "status": "SCAN_QR_CODE",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000")
        assert client.is_ready() is False

    @patch("src.tools._whatsapp_client.requests")
    def test_headers_include_api_key(self, mock_requests):
        from src.tools._whatsapp_client import WahaClient

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "msg_x"}
        mock_requests.post.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000", api_key="secret123")
        client.send_text("123@c.us", "Hi")

        call_kwargs = mock_requests.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("X-Api-Key") == "secret123"
