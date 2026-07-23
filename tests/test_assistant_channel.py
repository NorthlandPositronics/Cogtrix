"""Unit tests for cogtrix_core/assistant/channel.py — IncomingMessage and Channel ABC."""

from __future__ import annotations

import time
from abc import ABC

import pytest

from cogtrix_core.assistant.channel import Channel, IncomingMessage, parse_duration

# ---------------------------------------------------------------------------
# TestIncomingMessage
# ---------------------------------------------------------------------------


class TestIncomingMessage:
    """Tests for IncomingMessage dataclass."""

    def _make_msg(self, channel: str = "whatsapp", chat_id: str = "123") -> IncomingMessage:
        return IncomingMessage(
            channel=channel,
            chat_id=chat_id,
            message_id="msg-1",
            sender_id="sender-1",
            sender_name="Alice",
            text="Hello!",
            timestamp=time.time(),
        )

    def test_session_key_format(self):
        """session_key returns '{channel}::{chat_id}'."""
        msg = self._make_msg(channel="whatsapp", chat_id="14155551234")
        assert msg.session_key == "whatsapp::14155551234"

    def test_session_key_telegram(self):
        """session_key is correctly formed for Telegram channel."""
        msg = self._make_msg(channel="telegram", chat_id="-1001234567890")
        assert msg.session_key == "telegram::-1001234567890"

    def test_session_key_uses_separator(self):
        """session_key always uses '::' as the separator."""
        msg = self._make_msg(channel="mychan", chat_id="abc")
        assert "::" in msg.session_key
        assert msg.session_key.startswith("mychan::")
        assert msg.session_key.endswith("::abc")

    def test_metadata_custom(self):
        """metadata accepts arbitrary key-value pairs."""
        msg = IncomingMessage(
            channel="whatsapp",
            chat_id="1",
            message_id="m",
            sender_id="s",
            sender_name=None,
            text="hi",
            timestamp=0.0,
            metadata={"raw": "payload"},
        )
        assert msg.metadata["raw"] == "payload"

    def test_sender_name_can_be_none(self):
        """sender_name is optional and accepts None."""
        msg = IncomingMessage(
            channel="telegram",
            chat_id="42",
            message_id="m1",
            sender_id="u1",
            sender_name=None,
            text="test",
            timestamp=1234.5,
        )
        assert msg.sender_name is None


# ---------------------------------------------------------------------------
# TestChannelABC
# ---------------------------------------------------------------------------


class TestChannelABC:
    """Tests for Channel abstract base class."""

    def test_channel_cannot_be_instantiated_directly(self):
        """Channel ABC raises TypeError if instantiated without implementing abstracts."""
        with pytest.raises(TypeError):
            Channel()  # type: ignore[abstract]

    def test_channel_is_abstract(self):
        """Channel is an ABC."""
        assert issubclass(Channel, ABC)

    def test_concrete_subclass_works(self):
        """A subclass that implements all abstract methods can be instantiated."""

        class ConcreteChannel(Channel):
            @property
            def name(self) -> str:
                return "test"

            def poll(self) -> list[IncomingMessage]:
                return []

            def send(self, chat_id: str, text: str) -> bool:
                return True

            def is_ready(self) -> bool:
                return True

        ch = ConcreteChannel()
        assert ch.name == "test"
        assert ch.poll() == []
        assert ch.send("chat1", "hello") is True
        assert ch.is_ready() is True

    def test_subclass_missing_one_abstract_raises(self):
        """Subclass that omits any abstract method cannot be instantiated."""
        with pytest.raises(TypeError):

            class Incomplete(Channel):
                @property
                def name(self) -> str:
                    return "incomplete"

                def poll(self) -> list[IncomingMessage]:
                    return []

                def send(self, chat_id: str, text: str) -> bool:
                    return True

                # is_ready intentionally omitted

            Incomplete()  # type: ignore[abstract]

    def test_send_typing_default_noop(self):
        """send_typing is a default no-op that does not require override."""

        class MinimalChannel(Channel):
            @property
            def name(self) -> str:
                return "minimal"

            def poll(self) -> list[IncomingMessage]:
                return []

            def send(self, chat_id: str, text: str) -> bool:
                return True

            def is_ready(self) -> bool:
                return True

        ch = MinimalChannel()
        # Should not raise
        result = ch.send_typing("chat1")
        assert result is None


# ---------------------------------------------------------------------------
# TestParseDuration
# ---------------------------------------------------------------------------


class TestParseDuration:
    def test_hours(self):
        assert parse_duration("24h") == 86400.0

    def test_minutes(self):
        assert parse_duration("30m") == 1800.0

    def test_days(self):
        assert parse_duration("7d") == 604800.0

    def test_seconds(self):
        assert parse_duration("90s") == 90.0

    def test_compound(self):
        assert parse_duration("1d12h") == 129600.0

    def test_compound_hm(self):
        assert parse_duration("1h30m") == 5400.0

    def test_plain_number_string(self):
        assert parse_duration("3600") == 3600.0

    def test_int_value(self):
        assert parse_duration(3600) == 3600.0

    def test_float_value(self):
        assert parse_duration(60.5) == 60.5

    def test_none_returns_none(self):
        assert parse_duration(None) is None

    def test_empty_string_returns_none(self):
        assert parse_duration("") is None

    def test_zero_returns_none(self):
        assert parse_duration(0) is None

    def test_zero_string_returns_none(self):
        assert parse_duration("0") is None

    def test_case_insensitive(self):
        assert parse_duration("2H") == 7200.0

    def test_whitespace_in_compound(self):
        assert parse_duration("1d 12h") == 129600.0

    def test_invalid_string_returns_none(self):
        assert parse_duration("abc") is None

    def test_fractional_hours(self):
        assert parse_duration("1.5h") == 5400.0

    def test_fractional_days(self):
        assert parse_duration("0.5d") == 43200.0

    def test_negative_duration_returns_none(self):
        assert parse_duration("-5h") is None

    def test_negative_plain_number_returns_none(self):
        assert parse_duration("-60") is None
