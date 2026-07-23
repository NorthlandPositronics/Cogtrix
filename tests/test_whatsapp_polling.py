"""Tests for the two-phase WhatsApp polling architecture.

Covers WahaClient.get_chat_messages() and WhatsAppChannel.poll().
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.assistant.channels.whatsapp import WhatsAppChannel
from src.tools._whatsapp_client import ChatOverview, Message, WahaClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    id: str = "msg-1",
    timestamp: int = 1000,
    from_number: str = "123@c.us",
    body: str = "Hello",
    from_me: bool = False,
) -> Message:
    return Message(
        id=id,
        timestamp=timestamp,
        from_number=from_number,
        body=body,
        from_me=from_me,
    )


def _make_overview(
    chat_id: str = "123@c.us",
    name: str = "Alice",
    last_message: Message | None = None,
    archived: bool = False,
) -> ChatOverview:
    return ChatOverview(id=chat_id, name=name, last_message=last_message, archived=archived)


def _make_channel(config: dict | None = None) -> WhatsAppChannel:
    cfg: dict = {"waha_url": "http://localhost:3000", "session": "default"}
    if config:
        cfg.update(config)
    with patch.object(WahaClient, "__init__", lambda self, **kw: None):
        ch = WhatsAppChannel(cfg)
        ch._client = MagicMock(spec=WahaClient)
    ch._ensure_session = lambda: None  # type: ignore[method-assign]
    return ch


# ---------------------------------------------------------------------------
# TestWhatsAppPolling
# ---------------------------------------------------------------------------


class TestWhatsAppPolling:
    def test_poll_discovers_new_user_message(self) -> None:
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Hello")
        overview = _make_overview(chat_id="123@c.us", name="Alice", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert isinstance(result[0], IncomingMessage)
        assert result[0].text == "Hello"
        assert result[0].chat_id == "123@c.us"
        assert result[0].message_id == "msg-1"
        assert result[0].channel == "whatsapp"

    def test_poll_skips_unchanged_overview(self) -> None:
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000)
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._overview_snapshot["123@c.us"] = "msg-1"
        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_poll_catches_intermediate_messages(self) -> None:
        ch = _make_channel()
        msg1 = _make_message(id="msg-1", timestamp=1001, body="First")
        msg2 = _make_message(id="msg-2", timestamp=1002, body="Second")
        msg3 = _make_message(id="msg-3", timestamp=1003, body="Third")
        last = _make_message(id="msg-3", timestamp=1003, body="Third")
        overview = _make_overview(chat_id="123@c.us", last_message=last)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg1, msg2, msg3]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 3
        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps)

    def test_poll_fetches_when_agent_replied(self) -> None:
        ch = _make_channel()
        agent_reply = _make_message(
            id="reply-1", timestamp=1001, from_number="me@c.us", from_me=True
        )
        user_msg = _make_message(id="user-1", timestamp=999, from_number="123@c.us", body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=agent_reply)

        ch._chat_watermarks["123@c.us"] = 0
        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [user_msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].message_id == "user-1"

    def test_poll_agent_reply_no_new_user_messages(self) -> None:
        ch = _make_channel()
        agent_reply = _make_message(
            id="reply-1", timestamp=1001, from_number="me@c.us", from_me=True
        )
        overview = _make_overview(chat_id="123@c.us", last_message=agent_reply)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = []  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_poll_dedup_across_cycles(self) -> None:
        ch = _make_channel()
        msg1 = _make_message(id="msg-1", timestamp=1000, body="First")
        msg2 = _make_message(id="msg-2", timestamp=1001, body="Second")

        # First cycle: overview shows msg-1, fetch returns [msg-1]
        overview_v1 = _make_overview(chat_id="123@c.us", last_message=msg1)
        ch._client.get_chats_overview.return_value = [overview_v1]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg1]  # type: ignore[attr-defined]
        first = ch.poll()
        assert len(first) == 1

        # Second cycle: overview shows msg-2, fetch returns [msg-1, msg-2]
        overview_v2 = _make_overview(chat_id="123@c.us", last_message=msg2)
        ch._client.get_chats_overview.return_value = [overview_v2]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg1, msg2]  # type: ignore[attr-defined]
        second = ch.poll()

        assert len(second) == 1
        assert second[0].message_id == "msg-2"

    def test_poll_per_chat_watermark_independence(self) -> None:
        ch = _make_channel()
        ch._chat_watermarks["chat-A@c.us"] = 1000
        ch._chat_watermarks["chat-B@c.us"] = 2000

        msg_a1 = _make_message(id="a1", timestamp=1001, from_number="chat-A@c.us", body="A1")
        msg_a2 = _make_message(id="a2", timestamp=1002, from_number="chat-A@c.us", body="A2")
        msg_b1 = _make_message(id="b1", timestamp=2001, from_number="chat-B@c.us", body="B1")

        last_a = _make_message(id="a2", timestamp=1002)
        last_b = _make_message(id="b1", timestamp=2001)
        overview_a = _make_overview(chat_id="chat-A@c.us", name="A", last_message=last_a)
        overview_b = _make_overview(chat_id="chat-B@c.us", name="B", last_message=last_b)

        ch._client.get_chats_overview.return_value = [overview_a, overview_b]  # type: ignore[attr-defined]

        def side_effect(chat_id: str, **kwargs: object) -> list[Message]:
            if chat_id == "chat-A@c.us":
                return [msg_a1, msg_a2]
            return [msg_b1]

        ch._client.get_chat_messages.side_effect = side_effect  # type: ignore[attr-defined]

        result = ch.poll()

        ids = {m.message_id for m in result}
        assert "a1" in ids
        assert "a2" in ids
        assert "b1" in ids

    def test_poll_fetch_failure_isolates_chats(self) -> None:
        ch = _make_channel()
        msg_b = _make_message(id="b1", timestamp=2000, from_number="chat-B@c.us", body="Hi B")

        last_a = _make_message(id="a1", timestamp=1000)
        last_b = _make_message(id="b1", timestamp=2000)
        overview_a = _make_overview(chat_id="chat-A@c.us", name="A", last_message=last_a)
        overview_b = _make_overview(chat_id="chat-B@c.us", name="B", last_message=last_b)

        ch._client.get_chats_overview.return_value = [overview_a, overview_b]  # type: ignore[attr-defined]

        def side_effect(chat_id: str, **kwargs: object) -> list[Message]:
            if chat_id == "chat-A@c.us":
                raise RuntimeError("network error")
            return [msg_b]

        ch._client.get_chat_messages.side_effect = side_effect  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].message_id == "b1"
        # chat-A snapshot should NOT be updated (retry on next cycle)
        assert "chat-A@c.us" not in ch._overview_snapshot

    def test_poll_empty_body_filtered(self) -> None:
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000, body="   ")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_poll_contact_filter_applied(self) -> None:
        ch = _make_channel(config={"filter_mode": "whitelist", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Blocked")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_poll_watermark_advances_for_filtered(self) -> None:
        """When a message passes fetch but fails the per-message contact filter,
        the watermark still advances (prevents re-processing).
        Uses a group chat so the pre-filter cannot skip the HTTP fetch."""
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-1", timestamp=5000, from_number="123@c.us", body="Not allowed")
        overview = _make_overview(chat_id="group-1@g.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        ch.poll()

        assert ch._chat_watermarks.get("group-1@g.us") == 5000

    def test_poll_snapshot_eviction(self) -> None:
        ch = _make_channel()
        ch._SNAPSHOT_TTL = 0.0
        past_ts = time.monotonic() - 1.0
        ch._overview_snapshot["123@c.us"] = "msg-1"
        ch._snapshot_timestamps["123@c.us"] = past_ts

        msg = _make_message(id="msg-1", timestamp=1000, body="Still there")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        ch.poll()

        ch._client.get_chat_messages.assert_called_once()  # type: ignore[attr-defined]

    def test_poll_lid_resolution(self) -> None:
        ch = _make_channel()
        lid_sender = "178774490505455@lid"
        msg = _make_message(id="msg-1", timestamp=1000, from_number=lid_sender, body="Hey")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]
        ch._resolve_lid = MagicMock(return_value="+123456")  # type: ignore[method-assign]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].resolved_phone == "+123456"

    def test_poll_overview_none_last_message(self) -> None:
        ch = _make_channel()
        overview = _make_overview(chat_id="123@c.us", last_message=None)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_poll_messages_sorted_by_timestamp(self) -> None:
        ch = _make_channel()
        msg3 = _make_message(id="msg-3", timestamp=1003, body="C")
        msg1 = _make_message(id="msg-1", timestamp=1001, body="A")
        msg2 = _make_message(id="msg-2", timestamp=1002, body="B")
        last = _make_message(id="msg-3", timestamp=1003)
        overview = _make_overview(chat_id="123@c.us", last_message=last)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg3, msg1, msg2]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 3
        assert [r.message_id for r in result] == ["msg-1", "msg-2", "msg-3"]

    def test_poll_blacklist_deletes_and_archives(self) -> None:
        ch = _make_channel(config={"filter_mode": "blacklist", "contacts": ["123@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Spam")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.delete_message.assert_called_once_with("123@c.us", "msg-1")  # type: ignore[attr-defined]
        ch._client.archive_chat.assert_called_once_with("123@c.us")  # type: ignore[attr-defined]

    def test_poll_ignore_mode_skips_without_delete(self) -> None:
        ch = _make_channel(config={"filter_mode": "ignore", "contacts": ["123@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Ignored")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.delete_message.assert_not_called()  # type: ignore[attr-defined]
        ch._client.archive_chat.assert_not_called()  # type: ignore[attr-defined]

    def test_poll_allow_mode_only_allows_listed(self) -> None:
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Not allowed")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_poll_legacy_whitelist_normalized_to_allow(self) -> None:
        ch = _make_channel(config={"filter_mode": "whitelist", "contacts": ["123@c.us"]})
        assert ch._filter_mode == "allow"


# ---------------------------------------------------------------------------
# TestGetChatMessagesClient
# ---------------------------------------------------------------------------


class TestGetChatMessagesClient:
    @patch("src.tools._whatsapp_client.requests")
    def test_get_chat_messages_client_method(self, mock_requests: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {
                "id": "msg-42",
                "timestamp": 1700000000,
                "from": "123@c.us",
                "body": "Test body",
                "fromMe": False,
                "hasMedia": False,
            }
        ]
        mock_requests.get.return_value = mock_resp

        client = WahaClient(base_url="http://localhost:3000", session="default")
        messages = client.get_chat_messages(
            "123@c.us",
            limit=10,
            filter_from_me=False,
            filter_timestamp_gte=1000,
        )

        assert len(messages) == 1
        assert messages[0].id == "msg-42"
        assert messages[0].from_number == "123@c.us"
        assert messages[0].body == "Test body"
        assert messages[0].from_me is False

        call_args = mock_requests.get.call_args
        called_url: str = call_args[0][0] if call_args[0] else call_args.kwargs["url"]
        assert "/api/default/chats/123@c.us/messages" in called_url

        params: dict = call_args.kwargs.get("params") or call_args[1].get("params", {})
        assert params["limit"] == 10
        assert params["downloadMedia"] is False
        assert params["filter.fromMe"] is False
        assert params["filter.timestamp.gte"] == 1000


# ---------------------------------------------------------------------------
# ignore_archived tests
# ---------------------------------------------------------------------------


class TestIgnoreArchived:
    def test_archived_chat_skipped_by_default(self) -> None:
        """ignore_archived defaults to True — archived chats are skipped."""
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000)
        overview = _make_overview(chat_id="123@c.us", last_message=msg, archived=True)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_archived_chat_processed_when_flag_disabled(self) -> None:
        """ignore_archived=False means archived chats are still polled."""
        ch = _make_channel({"ignore_archived": False})
        msg = _make_message(id="msg-1", timestamp=1000, body="Hi from archive")
        overview = _make_overview(chat_id="123@c.us", last_message=msg, archived=True)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "Hi from archive"

    def test_non_archived_chat_not_affected(self) -> None:
        """Non-archived chats are always polled regardless of flag."""
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000, body="Normal")
        overview = _make_overview(chat_id="123@c.us", last_message=msg, archived=False)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "Normal"

    def test_mixed_archived_and_active_chats(self) -> None:
        """Only archived chats are skipped; active chats are polled normally."""
        ch = _make_channel()
        msg_a = _make_message(id="msg-a", timestamp=1000, from_number="111@c.us", body="Archived")
        msg_b = _make_message(id="msg-b", timestamp=1001, from_number="222@c.us", body="Active")
        archived = _make_overview(
            chat_id="111@c.us", name="Arch", last_message=msg_a, archived=True
        )
        active = _make_overview(chat_id="222@c.us", name="Act", last_message=msg_b, archived=False)

        ch._client.get_chats_overview.return_value = [archived, active]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg_b]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].chat_id == "222@c.us"


class TestChatOverviewArchived:
    def test_archived_parsed_from_archive_field(self) -> None:
        """ChatOverview.archived is parsed from top-level 'archive' field."""
        from unittest.mock import patch as _patch

        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.json.return_value = [
            {
                "id": "123@c.us",
                "name": "Alice",
                "archive": True,
                "lastMessage": None,
            }
        ]

        client = WahaClient(base_url="http://localhost:3000")
        with _patch("src.tools._whatsapp_client.requests") as mock_req:
            mock_req.get.return_value = raw_response
            chats = client.get_chats_overview()

        assert len(chats) == 1
        assert chats[0].archived is True

    def test_archived_parsed_from_chat_field(self) -> None:
        """ChatOverview.archived falls back to _chat.archive."""
        from unittest.mock import patch as _patch

        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.json.return_value = [
            {
                "id": "456@c.us",
                "name": "Bob",
                "_chat": {"archive": True},
                "lastMessage": None,
            }
        ]

        client = WahaClient(base_url="http://localhost:3000")
        with _patch("src.tools._whatsapp_client.requests") as mock_req:
            mock_req.get.return_value = raw_response
            chats = client.get_chats_overview()

        assert len(chats) == 1
        assert chats[0].archived is True

    def test_not_archived_by_default(self) -> None:
        """ChatOverview.archived defaults to False when field is missing."""
        from unittest.mock import patch as _patch

        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.json.return_value = [{"id": "789@c.us", "name": "Carol", "lastMessage": None}]

        client = WahaClient(base_url="http://localhost:3000")
        with _patch("src.tools._whatsapp_client.requests") as mock_req:
            mock_req.get.return_value = raw_response
            chats = client.get_chats_overview()

        assert len(chats) == 1
        assert chats[0].archived is False

    def test_explicit_archive_false_not_overridden(self) -> None:
        """archive: False at top level must NOT be overridden by _chat.archive: True."""
        from unittest.mock import patch as _patch

        raw_response = MagicMock()
        raw_response.status_code = 200
        raw_response.json.return_value = [
            {
                "id": "999@c.us",
                "name": "Dave",
                "archive": False,
                "_chat": {"archive": True},
                "lastMessage": None,
            }
        ]

        client = WahaClient(base_url="http://localhost:3000")
        with _patch("src.tools._whatsapp_client.requests") as mock_req:
            mock_req.get.return_value = raw_response
            chats = client.get_chats_overview()

        assert len(chats) == 1
        assert chats[0].archived is False


# ---------------------------------------------------------------------------
# ignore_older_than tests
# ---------------------------------------------------------------------------


class TestIgnoreOlderThan:
    def test_old_message_skipped(self) -> None:
        """Messages older than ignore_older_than are skipped."""
        ch = _make_channel({"ignore_older_than": "1h"})
        old_ts = int(time.time()) - 7200  # 2 hours ago
        msg = _make_message(id="old-1", timestamp=old_ts, body="Old msg")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []

    def test_recent_message_processed(self) -> None:
        """Messages within the window are processed normally."""
        ch = _make_channel({"ignore_older_than": "1h"})
        recent_ts = int(time.time()) - 60  # 1 minute ago
        msg = _make_message(id="new-1", timestamp=recent_ts, body="Recent")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "Recent"

    def test_disabled_by_default(self) -> None:
        """Without ignore_older_than, all messages are processed."""
        ch = _make_channel()
        old_ts = int(time.time()) - 86400 * 7  # 7 days ago
        msg = _make_message(id="ancient-1", timestamp=old_ts, body="Ancient")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "Ancient"

    def test_mixed_old_and_new_messages(self) -> None:
        """Only old messages are skipped; recent ones are processed."""
        ch = _make_channel({"ignore_older_than": "30m"})
        now = int(time.time())
        old_msg = _make_message(id="old-1", timestamp=now - 3600, body="Old")
        new_msg = _make_message(id="new-1", timestamp=now - 60, body="New")
        overview = _make_overview(chat_id="123@c.us", last_message=new_msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [old_msg, new_msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].text == "New"

    def test_duration_in_days(self) -> None:
        """Duration strings with 'd' suffix work correctly."""
        ch = _make_channel({"ignore_older_than": "7d"})
        within_ts = int(time.time()) - 86400 * 3  # 3 days ago
        msg = _make_message(id="msg-1", timestamp=within_ts, body="Within week")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1


# ---------------------------------------------------------------------------
# BUG-055: Watermark TTL is separate from snapshot TTL
# ---------------------------------------------------------------------------


class TestWatermarkTTL:
    def test_watermark_survives_snapshot_eviction(self) -> None:
        """Watermarks must NOT be evicted when snapshots expire (1h TTL)."""
        ch = _make_channel()
        ch._SNAPSHOT_TTL = 0.0
        past_ts = time.monotonic() - 1.0
        ch._overview_snapshot["123@c.us"] = "msg-1"
        ch._snapshot_timestamps["123@c.us"] = past_ts
        ch._chat_watermarks["123@c.us"] = 9999
        ch._watermark_timestamps["123@c.us"] = past_ts

        msg = _make_message(id="msg-2", timestamp=10000, body="New")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        ch.poll()

        assert (
            "123@c.us" not in ch._overview_snapshot or ch._overview_snapshot["123@c.us"] == "msg-2"
        )
        assert ch._chat_watermarks.get("123@c.us") == 10000

    def test_watermark_evicted_after_7_days(self) -> None:
        """Watermarks are evicted only after the 7-day WATERMARK_TTL."""
        ch = _make_channel()
        ch._WATERMARK_TTL = 0.0
        past_ts = time.monotonic() - 1.0
        ch._chat_watermarks["123@c.us"] = 5000
        ch._watermark_timestamps["123@c.us"] = past_ts

        ch._evict_stale_snapshots(time.monotonic())

        assert "123@c.us" not in ch._chat_watermarks
        assert "123@c.us" not in ch._watermark_timestamps


# ---------------------------------------------------------------------------
# BUG-058: filter_mode is case-insensitive
# ---------------------------------------------------------------------------


class TestFilterModeCaseInsensitive:
    def test_filter_mode_case_insensitive(self) -> None:
        """filter_mode 'Allow' and 'ALLOW' are treated the same as 'allow'."""
        ch_upper = _make_channel(config={"filter_mode": "Allow", "contacts": ["123@c.us"]})
        assert ch_upper._filter_mode == "allow"

        ch_caps = _make_channel(config={"filter_mode": "ALLOW", "contacts": ["123@c.us"]})
        assert ch_caps._filter_mode == "allow"

    def test_filter_mode_whitelist_case_insensitive(self) -> None:
        """filter_mode 'Whitelist' is normalized to 'allow'."""
        ch = _make_channel(config={"filter_mode": "Whitelist", "contacts": ["123@c.us"]})
        assert ch._filter_mode == "allow"

    def test_filter_mode_ignore_case_insensitive(self) -> None:
        """filter_mode 'IGNORE' is normalized to 'ignore'."""
        ch = _make_channel(config={"filter_mode": "IGNORE", "contacts": ["123@c.us"]})
        assert ch._filter_mode == "ignore"


# ---------------------------------------------------------------------------
# Optimization 1: Pre-filter before HTTP fetch
# ---------------------------------------------------------------------------


class TestPreFilter:
    def test_allow_mode_skips_unmatched_chat(self) -> None:
        """In allow mode, chats not matching any contact skip the HTTP fetch."""
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_allow_mode_fetches_matching_chat(self) -> None:
        """In allow mode, chats matching a contact are fetched normally."""
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["123@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        ch._client.get_chat_messages.assert_called_once()  # type: ignore[attr-defined]

    def test_ignore_mode_skips_ignored_chat(self) -> None:
        """In ignore mode, chats matching a contact skip the HTTP fetch."""
        ch = _make_channel(config={"filter_mode": "ignore", "contacts": ["123@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_blacklist_mode_still_fetches(self) -> None:
        """In blacklist mode, chats are always fetched (need delete+archive)."""
        ch = _make_channel(config={"filter_mode": "blacklist", "contacts": ["123@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="123@c.us", body="Spam")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert result == []
        ch._client.get_chat_messages.assert_called_once()  # type: ignore[attr-defined]
        ch._client.delete_message.assert_called_once()  # type: ignore[attr-defined]

    def test_group_chat_never_skipped(self) -> None:
        """Group chats (@g.us) are never pre-filtered because sender differs from chat.id."""
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-1", timestamp=1000, from_number="456@c.us", body="In group")
        overview = _make_overview(chat_id="111111-222222@g.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        ch._client.get_chat_messages.assert_called_once()  # type: ignore[attr-defined]
        assert len(result) == 1

    def test_snapshot_advances_on_skip(self) -> None:
        """When a chat is pre-filtered, the snapshot is updated to prevent re-detection."""
        ch = _make_channel(config={"filter_mode": "allow", "contacts": ["456@c.us"]})
        msg = _make_message(id="msg-99", timestamp=1000, from_number="123@c.us")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]

        ch.poll()

        assert ch._overview_snapshot.get("123@c.us") == "msg-99"

    def test_allow_mode_with_phonebook_match(self) -> None:
        """Pre-filter recognizes phonebook-resolved contacts."""
        ch = _make_channel(
            config={
                "filter_mode": "allow",
                "contacts": ["alice"],
                "phonebook": {"alice": "+971501234567"},
            }
        )
        msg = _make_message(id="msg-1", timestamp=1000)
        overview = _make_overview(chat_id="971501234567@c.us", name="Alice", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        ch.poll()

        ch._client.get_chat_messages.assert_called_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Optimization 2: LRU/TTL for _lid_cache
# ---------------------------------------------------------------------------


class TestLidCache:
    def test_positive_result_cached(self) -> None:
        """A resolved LID triggers only one HTTP call on repeated access."""
        ch = _make_channel()
        ch._client.resolve_lid.return_value = "+123456"  # type: ignore[attr-defined]

        result1 = ch._resolve_lid("111@lid")
        result2 = ch._resolve_lid("111@lid")

        assert result1 == "+123456"
        assert result2 == "+123456"
        assert ch._client.resolve_lid.call_count == 1  # type: ignore[attr-defined]

    def test_negative_result_cached_within_ttl(self) -> None:
        """A None result is cached and avoids re-resolution within TTL."""
        ch = _make_channel()
        ch._LID_NEGATIVE_TTL = 300.0
        ch._client.resolve_lid.return_value = None  # type: ignore[attr-defined]

        ch._resolve_lid("222@lid")
        ch._resolve_lid("222@lid")

        assert ch._client.resolve_lid.call_count == 1  # type: ignore[attr-defined]

    def test_negative_result_re_resolves_after_ttl(self) -> None:
        """Expired negative entries trigger re-resolution."""
        ch = _make_channel()
        ch._LID_NEGATIVE_TTL = 0.001  # instant expiry
        ch._client.resolve_lid.return_value = None  # type: ignore[attr-defined]

        ch._resolve_lid("333@lid")
        time.sleep(0.01)
        ch._resolve_lid("333@lid")

        assert ch._client.resolve_lid.call_count == 2  # type: ignore[attr-defined]

    def test_lru_eviction(self) -> None:
        """Cache evicts oldest entries when exceeding _LID_CACHE_MAX."""
        ch = _make_channel()
        ch._LID_CACHE_MAX = 3
        ch._client.resolve_lid.return_value = "+000"  # type: ignore[attr-defined]

        for i in range(5):
            ch._resolve_lid(f"{i}@lid")

        assert len(ch._lid_cache) == 3
        assert "0@lid" not in ch._lid_cache
        assert "1@lid" not in ch._lid_cache
        assert "4@lid" in ch._lid_cache

    def test_positive_never_expires(self) -> None:
        """Positive entries have infinite TTL and don't expire."""
        ch = _make_channel()
        ch._LID_NEGATIVE_TTL = 0.001
        ch._client.resolve_lid.return_value = "+999"  # type: ignore[attr-defined]

        ch._resolve_lid("444@lid")
        time.sleep(0.01)
        result = ch._resolve_lid("444@lid")

        assert result == "+999"
        assert ch._client.resolve_lid.call_count == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Optimization 3: Batch LID resolution
# ---------------------------------------------------------------------------


class TestBatchLidResolution:
    def test_prefetch_populates_cache(self) -> None:
        """_prefetch_lids resolves uncached LIDs before processing."""
        ch = _make_channel()
        ch._client.resolve_lid.return_value = "+111"  # type: ignore[attr-defined]

        messages = [
            _make_message(id="m1", from_number="aaa@lid"),
            _make_message(id="m2", from_number="bbb@lid"),
        ]
        ch._prefetch_lids(messages)

        assert "aaa@lid" in ch._lid_cache
        assert "bbb@lid" in ch._lid_cache

    def test_prefetch_skips_cached(self) -> None:
        """Already-cached LIDs are not re-resolved during prefetch."""
        ch = _make_channel()
        ch._lid_cache["aaa@lid"] = ("+111", float("inf"))
        ch._client.resolve_lid.return_value = "+222"  # type: ignore[attr-defined]

        messages = [
            _make_message(id="m1", from_number="aaa@lid"),
            _make_message(id="m2", from_number="bbb@lid"),
        ]
        ch._prefetch_lids(messages)

        # Only bbb@lid should trigger HTTP
        ch._client.resolve_lid.assert_called_once_with("bbb@lid")  # type: ignore[attr-defined]

    def test_prefetch_deduplicates(self) -> None:
        """Multiple messages from the same LID trigger only one resolution."""
        ch = _make_channel()
        ch._client.resolve_lid.return_value = "+333"  # type: ignore[attr-defined]

        messages = [
            _make_message(id="m1", from_number="same@lid"),
            _make_message(id="m2", from_number="same@lid"),
            _make_message(id="m3", from_number="same@lid"),
        ]
        ch._prefetch_lids(messages)

        assert ch._client.resolve_lid.call_count == 1  # type: ignore[attr-defined]

    def test_prefetch_no_lids_is_noop(self) -> None:
        """Messages without @lid don't trigger any resolution."""
        ch = _make_channel()

        messages = [
            _make_message(id="m1", from_number="123@c.us"),
            _make_message(id="m2", from_number="456@c.us"),
        ]
        ch._prefetch_lids(messages)

        ch._client.resolve_lid.assert_not_called()  # type: ignore[attr-defined]

    def test_poll_resolves_lid_once_per_unique(self) -> None:
        """End-to-end: multiple messages from same LID in one poll cycle
        result in exactly one resolve_lid call."""
        ch = _make_channel()
        lid = "999@lid"
        msg1 = _make_message(id="m1", timestamp=1000, from_number=lid, body="A")
        msg2 = _make_message(id="m2", timestamp=1001, from_number=lid, body="B")
        overview = _make_overview(chat_id="123@c.us", last_message=msg2)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg1, msg2]  # type: ignore[attr-defined]
        ch._client.resolve_lid.return_value = "+555"  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 2
        assert ch._client.resolve_lid.call_count == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fetch error backoff
# ---------------------------------------------------------------------------


class TestFetchErrorBackoff:
    def test_failed_chat_skipped_during_cooldown(self) -> None:
        """A chat that fails to fetch is skipped on the next poll cycle."""
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000, body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.side_effect = RuntimeError("500 Server Error")  # type: ignore[attr-defined]

        ch.poll()  # first attempt — fails, records error

        assert "123@c.us" in ch._chat_errors
        count, retry_after = ch._chat_errors["123@c.us"]
        assert count == 1

        # Second poll — should skip the chat (still in cooldown)
        ch._client.get_chat_messages.reset_mock()  # type: ignore[attr-defined]
        ch.poll()

        ch._client.get_chat_messages.assert_not_called()  # type: ignore[attr-defined]

    def test_error_cleared_on_success(self) -> None:
        """Successful fetch clears the error state for a chat."""
        ch = _make_channel()
        msg = _make_message(id="msg-1", timestamp=1000, body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        # Pre-populate an error entry with expired cooldown
        ch._chat_errors["123@c.us"] = (2, 0.0)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.return_value = [msg]  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert "123@c.us" not in ch._chat_errors

    def test_backoff_escalates(self) -> None:
        """Error count and backoff duration escalate on consecutive failures."""
        ch = _make_channel()
        ch._FETCH_ERROR_BASE = 10.0
        msg = _make_message(id="msg-1", timestamp=1000, body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.side_effect = RuntimeError("500")  # type: ignore[attr-defined]

        # First failure
        ch.poll()
        count1, _ = ch._chat_errors["123@c.us"]
        assert count1 == 1

        # Force cooldown expiry and retry
        ch._chat_errors["123@c.us"] = (1, 0.0)
        ch.poll()
        count2, _ = ch._chat_errors["123@c.us"]
        assert count2 == 2

    def test_backoff_capped_at_max(self) -> None:
        """Backoff duration never exceeds _FETCH_ERROR_MAX."""
        ch = _make_channel()
        ch._FETCH_ERROR_BASE = 30.0
        ch._FETCH_ERROR_MAX = 60.0

        # Simulate many failures
        ch._chat_errors["123@c.us"] = (10, 0.0)

        msg = _make_message(id="msg-1", timestamp=1000, body="Hi")
        overview = _make_overview(chat_id="123@c.us", last_message=msg)

        ch._client.get_chats_overview.return_value = [overview]  # type: ignore[attr-defined]
        ch._client.get_chat_messages.side_effect = RuntimeError("500")  # type: ignore[attr-defined]

        ch.poll()

        _, retry_after = ch._chat_errors["123@c.us"]
        now = time.monotonic()
        backoff = retry_after - now
        assert backoff <= 60.0 + 1.0  # allow 1s tolerance

    def test_other_chats_unaffected(self) -> None:
        """Error backoff for one chat doesn't affect other chats."""
        ch = _make_channel()
        msg_a = _make_message(id="a1", timestamp=1000, from_number="aaa@c.us", body="A")
        msg_b = _make_message(id="b1", timestamp=1000, from_number="bbb@c.us", body="B")
        overview_a = _make_overview(chat_id="aaa@c.us", name="A", last_message=msg_a)
        overview_b = _make_overview(chat_id="bbb@c.us", name="B", last_message=msg_b)

        ch._client.get_chats_overview.return_value = [overview_a, overview_b]  # type: ignore[attr-defined]

        def side_effect(chat_id: str, **kwargs: object) -> list[Message]:
            if chat_id == "aaa@c.us":
                raise RuntimeError("500")
            return [msg_b]

        ch._client.get_chat_messages.side_effect = side_effect  # type: ignore[attr-defined]

        result = ch.poll()

        assert len(result) == 1
        assert result[0].chat_id == "bbb@c.us"
        assert "aaa@c.us" in ch._chat_errors
        assert "bbb@c.us" not in ch._chat_errors
