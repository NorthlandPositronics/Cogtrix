"""#2279 — caption-less WhatsApp media must not crash the poll loop.

WAHA returns ``body=None`` for a media message with no caption. The unguarded
``msg.body.strip()`` raised ``AttributeError`` out of ``_process_message`` →
``poll()`` → the whole poll cycle aborted, and the same message was re-fetched and
re-crashed every cycle until it aged out — silently taking the assistant offline
for ALL chats. Two fixes: None-guard the body check, and isolate per-message
failures in ``poll()`` so one bad message can't kill the cycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.tools._whatsapp_client import ChatOverview, Message, WahaClient


def _make_channel(config: dict | None = None):
    from src.assistant.channels.whatsapp import WhatsAppChannel

    cfg: dict = {"waha_url": "http://localhost:3000", "session": "default"}
    if config:
        cfg.update(config)
    with patch.object(WahaClient, "__init__", lambda self, **kw: None):
        ch = WhatsAppChannel(cfg)
        ch._client = MagicMock()
    ch._ensure_session = lambda: None  # type: ignore[method-assign]
    return ch


def _media_msg_no_caption(id: str = "m-media", ts: int = 1000) -> Message:
    # WAHA hands us body=None for a caption-less media message. The dataclass
    # annotation says str, but the wire reality is None — that's the bug.
    return Message(
        id=id,
        timestamp=ts,
        from_number="123@c.us",
        body=None,  # type: ignore[arg-type]
        from_me=False,
        has_media=True,
        media_url="http://localhost:3000/api/files/default/x.jpeg",
    )


def _text_msg(id: str, ts: int, body: str) -> Message:
    return Message(id=id, timestamp=ts, from_number="123@c.us", body=body, from_me=False)


def _overview(last: Message) -> ChatOverview:
    return ChatOverview(id="123@c.us", name="Alice", last_message=last, archived=False)


class TestCaptionlessMediaDoesNotCrash:
    def test_process_message_none_body_failed_download_returns_none(self) -> None:
        """body=None + download_media returns None → filtered (None), not a raise."""
        ch = _make_channel({"analyze_media": True})
        ch._client.download_media.return_value = None  # fetch failed → images stays empty
        msg = _media_msg_no_caption()
        chat = _overview(msg)

        # Must NOT raise AttributeError on None.strip()
        result = ch._process_message(msg, chat, now=123.0)
        assert result is None

    def test_process_message_none_body_with_image_is_emitted(self) -> None:
        """body=None but media downloads OK → emitted with the image, text=''."""
        ch = _make_channel({"analyze_media": True})
        ch._client.download_media.return_value = (b"\xff\xd8\xff", "image/jpeg")
        msg = _media_msg_no_caption()
        chat = _overview(msg)

        result = ch._process_message(msg, chat, now=123.0)
        assert isinstance(result, IncomingMessage)
        assert result.text == ""  # never None downstream
        assert result.images and result.images[0].startswith("data:image/jpeg;base64,")

    def test_poll_one_crashing_message_does_not_drop_the_valid_one(self) -> None:
        """A message that raises inside _process_message must not abort the cycle —
        the valid text message in the same batch is still returned."""
        ch = _make_channel({"analyze_media": True})
        bad = _media_msg_no_caption(id="bad", ts=1001)
        good = _text_msg("good", 1002, "Hello there")
        ch._client.get_chats_overview.return_value = [_overview(good)]
        ch._client.get_chat_messages.return_value = [bad, good]

        # Force _process_message to raise on the bad message only (simulates any
        # per-message defect), passthrough for the good one.
        real = ch._process_message

        def _flaky(msg, chat, now):
            if msg.id == "bad":
                raise AttributeError("'NoneType' object has no attribute 'strip'")
            return real(msg, chat, now)

        ch._process_message = _flaky  # type: ignore[method-assign]

        result = ch.poll()  # must not raise
        ids = {m.message_id for m in result}
        assert "good" in ids
        assert "bad" not in ids
        # The bad message is marked seen so it isn't re-fetched and re-crashed.
        assert "bad" in ch._seen_ids

    def test_poll_real_captionless_media_end_to_end(self) -> None:
        """End-to-end: a real caption-less media message flows through poll()
        without the None-body fix tripping — and download failure → filtered."""
        ch = _make_channel({"analyze_media": True})
        ch._client.download_media.return_value = None
        media = _media_msg_no_caption(id="real-media", ts=2000)
        ch._client.get_chats_overview.return_value = [_overview(media)]
        ch._client.get_chat_messages.return_value = [media]

        result = ch.poll()  # must not raise
        assert result == []  # captionless + failed download → nothing to reply to
