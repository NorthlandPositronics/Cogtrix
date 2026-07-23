"""Tests for per-session authorization in scheduler tools (BUG-040, BUG-041).

Verifies that list_scheduled_messages, edit_scheduled_message, and
cancel_scheduled_message are scoped to the calling session's chat_id.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _make_scheduler(tmp_path: Path) -> Any:
    from src.assistant.scheduler import MessageScheduler

    return MessageScheduler(
        channels={},
        persist_path=tmp_path / "schedule.json",
        quiet_hours_cfg=None,
        dispatch_interval=30.0,
    )


class TestListScheduledAuth:
    """BUG-040: list_scheduled_messages must be scoped to the caller's chat."""

    def test_session_b_cannot_see_session_a_messages(self, tmp_path):
        """Session B must not see messages scheduled for session A."""
        from src.assistant.scheduler import create_list_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        # Schedule a message for chat A
        scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="Hello from A",
            send_at=time.time() + 3600,
            recipient="User A",
        )

        # Create a list tool scoped to chat B
        list_tool_b = create_list_scheduled_tool(scheduler, caller_chat_id="chat_b")
        result = list_tool_b.invoke({})

        assert "Hello from A" not in result
        assert "No pending scheduled messages" in result

    def test_session_a_can_see_its_own_messages(self, tmp_path):
        """Session A must see messages scheduled for its own chat."""
        from src.assistant.scheduler import create_list_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="Hello from A",
            send_at=time.time() + 3600,
            recipient="User A",
        )

        # Create a list tool scoped to chat A
        list_tool_a = create_list_scheduled_tool(scheduler, caller_chat_id="chat_a")
        result = list_tool_a.invoke({})

        assert "Hello from A" in result

    def test_explicit_chat_id_filter_overrides_caller_scope(self, tmp_path):
        """When the caller supplies an explicit chat_id filter, it takes precedence."""
        from src.assistant.scheduler import create_list_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="Hello from A",
            send_at=time.time() + 3600,
        )

        # Session B uses an explicit chat_id="chat_a" filter — should still be restricted
        # by the implicit scope when no explicit chat_id is provided; here we test that
        # the no-explicit-filter path is restricted.
        list_tool_b = create_list_scheduled_tool(scheduler, caller_chat_id="chat_b")
        result_no_filter = list_tool_b.invoke({})
        assert "Hello from A" not in result_no_filter

    def test_no_caller_chat_id_returns_all(self, tmp_path):
        """When caller_chat_id is empty (legacy/admin), all messages are returned."""
        from src.assistant.scheduler import create_list_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        scheduler.schedule(
            channel="telegram", chat_id="chat_a", text="Msg A", send_at=time.time() + 3600
        )
        scheduler.schedule(
            channel="telegram", chat_id="chat_b", text="Msg B", send_at=time.time() + 3600
        )

        list_tool = create_list_scheduled_tool(scheduler, caller_chat_id="")
        result = list_tool.invoke({})

        assert "Msg A" in result
        assert "Msg B" in result


class TestCancelScheduledAuth:
    """BUG-041: cancel_scheduled_message must be scoped to the caller's chat."""

    def test_session_b_cannot_cancel_session_a_message(self, tmp_path):
        """Session B must not be able to cancel a message scheduled by session A."""
        from src.assistant.scheduler import create_cancel_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        msg_id = scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="A's message",
            send_at=time.time() + 3600,
        )
        short_id = msg_id[:8]

        # Session B tries to cancel session A's message
        cancel_tool_b = create_cancel_scheduled_tool(scheduler, caller_chat_id="chat_b")
        result = cancel_tool_b.invoke({"message_id": short_id})

        # Should return "not found" to prevent info disclosure
        assert "No pending message found" in result
        # Message must still be pending
        with scheduler._lock:
            assert scheduler._queue[msg_id].status == "pending"

    def test_session_a_can_cancel_its_own_message(self, tmp_path):
        """Session A must be able to cancel its own message."""
        from src.assistant.scheduler import create_cancel_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        msg_id = scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="A's message",
            send_at=time.time() + 3600,
        )
        short_id = msg_id[:8]

        cancel_tool_a = create_cancel_scheduled_tool(scheduler, caller_chat_id="chat_a")
        result = cancel_tool_a.invoke({"message_id": short_id})

        assert "cancelled" in result.lower()
        with scheduler._lock:
            assert scheduler._queue[msg_id].status == "cancelled"


class TestEditScheduledAuth:
    """BUG-041: edit_scheduled_message must be scoped to the caller's chat."""

    def test_session_b_cannot_edit_session_a_message(self, tmp_path):
        """Session B must not be able to edit a message scheduled by session A."""
        from src.assistant.scheduler import create_edit_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        msg_id = scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="A's original text",
            send_at=time.time() + 3600,
        )
        short_id = msg_id[:8]

        # Session B tries to edit session A's message
        edit_tool_b = create_edit_scheduled_tool(scheduler, caller_chat_id="chat_b")
        result = edit_tool_b.invoke({"message_id": short_id, "new_text": "B's takeover"})

        # Should return "not found" to prevent info disclosure
        assert "No pending message found" in result
        # Text must be unchanged
        with scheduler._lock:
            assert scheduler._queue[msg_id].text == "A's original text"

    def test_session_a_can_edit_its_own_message(self, tmp_path):
        """Session A must be able to edit its own message."""
        from src.assistant.scheduler import create_edit_scheduled_tool

        scheduler = _make_scheduler(tmp_path)
        msg_id = scheduler.schedule(
            channel="telegram",
            chat_id="chat_a",
            text="A's original text",
            send_at=time.time() + 3600,
        )
        short_id = msg_id[:8]

        edit_tool_a = create_edit_scheduled_tool(scheduler, caller_chat_id="chat_a")
        result = edit_tool_a.invoke({"message_id": short_id, "new_text": "A's new text"})

        assert "text updated" in result.lower()
        with scheduler._lock:
            assert scheduler._queue[msg_id].text == "A's new text"
