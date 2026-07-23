"""Unit tests for src/assistant/scheduler.py and scheduler integration in handler."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.assistant.channel import SendResult
from src.assistant.scheduler import (
    MessageScheduler,
    QueueReplyState,
    QuietHoursPolicy,
    ScheduledMessage,
    ScheduleReplyState,
    _is_in_quiet_window,
    _next_quiet_end,
    create_queue_reply_tool,
    create_schedule_reply_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler(tmp_path: Path, quiet_cfg: dict | None = None) -> MessageScheduler:
    """Return a MessageScheduler backed by tmp_path with no channels."""
    return MessageScheduler(
        channels={},
        persist_path=tmp_path / "schedule.json",
        quiet_hours_cfg=quiet_cfg,
    )


def _make_channel_scheduler(
    tmp_path: Path, channel: MagicMock, name: str = "telegram"
) -> MessageScheduler:
    return MessageScheduler(
        channels={name: channel},
        persist_path=tmp_path / "schedule.json",
    )


# ---------------------------------------------------------------------------
# TestScheduledMessage
# ---------------------------------------------------------------------------


class TestScheduledMessage:
    """ScheduledMessage dataclass creation and round-trip serialization."""

    def test_defaults(self):
        now = time.time()
        msg = ScheduledMessage(
            id="abc",
            channel="telegram",
            chat_id="42",
            text="hello",
            send_at=now + 60,
            created_at=now,
        )
        assert msg.status == "pending"
        assert msg.attempts == 0
        assert msg.max_attempts == 3

    def test_to_dict_round_trip(self):
        now = time.time()
        msg = ScheduledMessage(
            id="xyz",
            channel="whatsapp",
            chat_id="99",
            text="hi",
            send_at=now + 120,
            created_at=now,
            status="sent",
            attempts=1,
        )
        d = msg.to_dict()
        restored = ScheduledMessage.from_dict(d)
        assert restored.id == msg.id
        assert restored.channel == msg.channel
        assert restored.chat_id == msg.chat_id
        assert restored.text == msg.text
        assert restored.send_at == msg.send_at
        assert restored.created_at == msg.created_at
        assert restored.status == msg.status
        assert restored.attempts == msg.attempts
        assert restored.max_attempts == msg.max_attempts


# ---------------------------------------------------------------------------
# TestScheduleReplyState
# ---------------------------------------------------------------------------


class TestScheduleReplyState:
    """ScheduleReplyState initial state."""

    def test_initial_values(self):
        state = ScheduleReplyState()
        assert state.was_called is False
        assert state.scheduled_text == ""
        assert state.delay_minutes == 0


# ---------------------------------------------------------------------------
# TestCreateScheduleReplyTool
# ---------------------------------------------------------------------------


class TestCreateScheduleReplyTool:
    """Tool factory and closure behavior."""

    def test_calling_tool_sets_state(self):
        state = ScheduleReplyState()
        tool = create_schedule_reply_tool(state)
        result = tool.invoke({"text": "See you soon!", "delay_minutes": 60})
        assert state.was_called is True
        assert state.scheduled_text == "See you soon!"
        assert state.delay_minutes == 60
        assert "60" in result

    def test_tool_closure_isolated_per_state(self):
        state_a = ScheduleReplyState()
        state_b = ScheduleReplyState()
        tool_a = create_schedule_reply_tool(state_a)
        _tool_b = create_schedule_reply_tool(state_b)
        tool_a.invoke({"text": "reply A", "delay_minutes": 30})
        assert state_a.was_called is True
        assert state_b.was_called is False

    def test_tool_does_not_queue_directly(self):
        """Tool only sets state — does not interact with any scheduler."""
        state = ScheduleReplyState()
        tool = create_schedule_reply_tool(state)
        # No scheduler involved; simply ensure no exception and state is set.
        tool.invoke({"text": "Later", "delay_minutes": 120})
        assert state.was_called is True


# ---------------------------------------------------------------------------
# TestMessageSchedulerSchedule
# ---------------------------------------------------------------------------


class TestMessageSchedulerSchedule:
    """MessageScheduler.schedule() and cancel_pending()."""

    def test_schedule_returns_string_id(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "hello", time.time() + 600)
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_schedule_adds_to_queue(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "hello", time.time() + 600)
        assert mid in sched._queue
        msg = sched._queue[mid]
        assert msg.channel == "telegram"
        assert msg.chat_id == "42"
        assert msg.text == "hello"
        assert msg.status == "pending"

    def test_cancel_pending_returns_count(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.schedule("telegram", "42", "a", time.time() + 600)
        sched.schedule("telegram", "42", "b", time.time() + 1200)
        cancelled = sched.cancel_pending("telegram", "42")
        assert cancelled == 2

    def test_cancel_pending_marks_status(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "hello", time.time() + 600)
        sched.cancel_pending("telegram", "42")
        assert sched._queue[mid].status == "cancelled"

    def test_cancel_pending_only_affects_target_chat(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid_a = sched.schedule("telegram", "42", "for 42", time.time() + 600)
        mid_b = sched.schedule("telegram", "99", "for 99", time.time() + 600)
        sched.cancel_pending("telegram", "42")
        assert sched._queue[mid_a].status == "cancelled"
        assert sched._queue[mid_b].status == "pending"

    def test_cancel_pending_zero_when_none(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        assert sched.cancel_pending("telegram", "42") == 0

    def test_cancel_does_not_affect_sent(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "hello", time.time() + 600)
        sched._queue[mid].status = "sent"
        cancelled = sched.cancel_pending("telegram", "42")
        assert cancelled == 0
        assert sched._queue[mid].status == "sent"

    def test_cancel_pending_cancels_sending(self, tmp_path):
        """cancel_pending() also cancels messages in 'sending' state."""
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "in flight", time.time() + 600)
        sched._queue[mid].status = "sending"
        cancelled = sched.cancel_pending("telegram", "42")
        assert cancelled == 1
        assert sched._queue[mid].status == "cancelled"

    def test_send_message_respects_external_cancel(self, tmp_path):
        """_send_message does not mark 'sent' if message was cancelled during send."""
        channel = MagicMock()

        sched = _make_channel_scheduler(tmp_path, channel)
        mid = sched.schedule("telegram", "42", "race me", time.time() - 10)

        def _cancel_during_send(_chat_id, _text):
            # Simulate cancel_pending() racing with channel.send()
            sched._queue[mid].status = "cancelled"
            return SendResult(ok=True)  # send physically succeeded

        channel.send.side_effect = _cancel_during_send
        sched._dispatch_due()
        # Even though channel.send() returned True, status should remain "cancelled"
        assert sched._queue[mid].status == "cancelled"


# ---------------------------------------------------------------------------
# TestMessageSchedulerPersistence
# ---------------------------------------------------------------------------


class TestMessageSchedulerPersistence:
    """Save/load round-trip."""

    def test_save_creates_file(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.schedule("telegram", "42", "hi", time.time() + 600)
        sched.save()
        persist_file = tmp_path / "schedule.json"
        assert persist_file.exists()

    def test_load_restores_queue(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "hi", time.time() + 600)
        sched.save()

        sched2 = _make_scheduler(tmp_path)
        assert mid in sched2._queue
        assert sched2._queue[mid].text == "hi"

    def test_save_does_not_raise_on_write_failure(self, tmp_path):
        """_atomic_write swallows IOErrors — save() never propagates exceptions."""
        sched = _make_scheduler(tmp_path)
        # Patch mkstemp so every write attempt fails.
        with patch("src.utils.atomic_write.tempfile.mkstemp", side_effect=OSError("disk full")):
            # Must not raise even though the underlying write fails.
            sched._atomic_write({"test": "data"})

    def test_load_skips_malformed_entry(self, tmp_path):
        persist_file = tmp_path / "schedule.json"
        persist_file.write_text(
            json.dumps({"bad-id": {"id": "bad-id", "broken": True}}), encoding="utf-8"
        )
        sched = _make_scheduler(tmp_path)
        assert "bad-id" not in sched._queue


# ---------------------------------------------------------------------------
# TestStaleExpiration
# ---------------------------------------------------------------------------


class TestStaleExpiration:
    """Messages overdue by > 2 h are expired on load."""

    def test_stale_message_expired_on_load(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        # Place a message with send_at far in the past (3 hours ago).
        past = time.time() - 3 * 3600
        mid = sched.schedule("telegram", "42", "old msg", past)
        sched.save()

        sched2 = _make_scheduler(tmp_path)
        assert sched2._queue[mid].status == "expired"

    def test_recent_pending_not_expired(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        future = time.time() + 600
        mid = sched.schedule("telegram", "42", "fresh", future)
        sched.save()

        sched2 = _make_scheduler(tmp_path)
        assert sched2._queue[mid].status == "pending"


# ---------------------------------------------------------------------------
# TestQuietHours
# ---------------------------------------------------------------------------


class TestQuietHours:
    """Quiet hours enforcement helpers."""

    def test_is_in_quiet_window_wraps_midnight(self):
        # 23:00–08:00 window; hour=02 should be in window.
        policy = QuietHoursPolicy(start_hour=23, end_hour=8, timezone="UTC")
        # 2025-01-01 02:00 UTC = 1735696800
        import datetime

        dt = datetime.datetime(2025, 1, 1, 2, 0, 0, tzinfo=datetime.UTC)
        assert _is_in_quiet_window(policy, dt.timestamp()) is True

    def test_is_in_quiet_window_outside(self):
        policy = QuietHoursPolicy(start_hour=23, end_hour=8, timezone="UTC")
        import datetime

        dt = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
        assert _is_in_quiet_window(policy, dt.timestamp()) is False

    def test_next_quiet_end_returns_future_timestamp(self):
        policy = QuietHoursPolicy(start_hour=23, end_hour=8, timezone="UTC")
        import datetime

        # During quiet window at 02:00; end should be 08:00 same day.
        dt = datetime.datetime(2025, 1, 1, 2, 0, 0, tzinfo=datetime.UTC)
        end = _next_quiet_end(policy, dt.timestamp())
        end_dt = datetime.datetime.fromtimestamp(end, tz=datetime.UTC)
        assert end_dt.hour == 8
        assert end_dt.date() == datetime.date(2025, 1, 1)

    def test_scheduler_defers_during_quiet_hours(self, tmp_path):
        """_dispatch_due defers messages that fall in quiet window."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)

        import datetime

        # Force current time to 02:00 UTC so quiet window is active.
        frozen_now = datetime.datetime(2025, 1, 1, 2, 0, 0, tzinfo=datetime.UTC)
        frozen_ts = frozen_now.timestamp()

        quiet_cfg = {"_default": {"quiet_hours": [23, 8], "timezone": "UTC"}}
        sched = MessageScheduler(
            channels={"telegram": channel},
            persist_path=tmp_path / "schedule.json",
            quiet_hours_cfg=quiet_cfg,
        )
        # Schedule a message due now.
        mid = sched.schedule("telegram", "42", "deferred msg", frozen_ts - 10)

        with patch("src.assistant.scheduler.time") as mock_time:
            mock_time.time.return_value = frozen_ts
            sched._dispatch_due()

        # Message should not have been sent; status still pending.
        channel.send.assert_not_called()
        assert sched._queue[mid].status == "pending"
        # send_at should have been pushed forward to 08:00.
        assert sched._queue[mid].send_at > frozen_ts


# ---------------------------------------------------------------------------
# TestDispatchSend
# ---------------------------------------------------------------------------


class TestDispatchSend:
    """MessageScheduler._dispatch_due() send path."""

    def test_sends_due_message(self, tmp_path):
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        sched = _make_channel_scheduler(tmp_path, channel)

        mid = sched.schedule("telegram", "42", "hi", time.time() - 10)
        sched._dispatch_due()

        channel.send.assert_called_once_with("42", "hi")
        assert sched._queue[mid].status == "sent"

    def test_failed_send_retries(self, tmp_path):
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=False)
        sched = _make_channel_scheduler(tmp_path, channel)

        mid = sched.schedule("telegram", "42", "retry me", time.time() - 10)
        sched._dispatch_due()

        assert sched._queue[mid].status == "pending"
        assert sched._queue[mid].attempts == 1
        # send_at advanced by backoff.
        assert sched._queue[mid].send_at > time.time()

    def test_exhausted_attempts_marked_failed(self, tmp_path):
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=False)
        sched = _make_channel_scheduler(tmp_path, channel)

        mid = sched.schedule("telegram", "42", "fail me", time.time() - 10)
        sched._queue[mid].attempts = 2  # already at max-1
        sched._queue[mid].max_attempts = 3
        sched._dispatch_due()

        assert sched._queue[mid].status == "failed"

    def test_unknown_channel_marks_failed(self, tmp_path):
        sched = _make_scheduler(tmp_path)  # no channels
        mid = sched.schedule("telegram", "42", "no channel", time.time() - 10)
        sched._dispatch_due()
        assert sched._queue[mid].status == "failed"

    def test_future_message_not_dispatched(self, tmp_path):
        channel = MagicMock()
        sched = _make_channel_scheduler(tmp_path, channel)

        sched.schedule("telegram", "42", "future", time.time() + 9999)
        sched._dispatch_due()

        channel.send.assert_not_called()

    def test_cancelled_message_not_dispatched(self, tmp_path):
        channel = MagicMock()
        sched = _make_channel_scheduler(tmp_path, channel)

        mid = sched.schedule("telegram", "42", "cancel me", time.time() - 10)
        sched._queue[mid].status = "cancelled"
        sched._dispatch_due()

        channel.send.assert_not_called()


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """MessageScheduler._cleanup_old() removes old terminal messages."""

    def test_old_sent_removed(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "done", time.time() - 10)
        sched._queue[mid].status = "sent"
        sched._queue[mid].created_at = time.time() - 25 * 3600  # > 24h old
        sched._cleanup_old()
        assert mid not in sched._queue

    def test_recent_sent_kept(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "recent", time.time() - 10)
        sched._queue[mid].status = "sent"
        sched._cleanup_old()
        assert mid in sched._queue

    def test_pending_never_cleaned(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.schedule("telegram", "42", "pending", time.time() - 10)
        sched._queue[mid].created_at = time.time() - 48 * 3600
        sched._cleanup_old()
        assert mid in sched._queue


# ---------------------------------------------------------------------------
# TestHandlerIntegration
# ---------------------------------------------------------------------------


class TestHandlerIntegration:
    """Integration between MessageHandler and MessageScheduler."""

    def _make_handler(
        self,
        tmp_path: Path,
        scheduler: MessageScheduler | None = None,
        agent_runner: Callable[..., Any] | None = None,
    ):
        from src.assistant.handler import MessageHandler
        from src.memory.context import MemoryContext

        if agent_runner is None:
            agent_runner = MagicMock(return_value="plain reply")

        session = MagicMock()
        session.session_key = "telegram::42"
        session.lock = MagicMock()
        session.lock.__enter__ = MagicMock(return_value=None)
        session.lock.__exit__ = MagicMock(return_value=False)
        session.guardrail_violations = 0
        session.last_sent_message_id = None
        session.memory_manager.prepare_context.return_value = MemoryContext(
            messages=[], context_prefix=None
        )

        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=agent_runner,
            scheduler=scheduler,
        )
        return handler, session

    def _make_msg(self):
        from src.assistant.channel import IncomingMessage

        return IncomingMessage(
            channel="telegram",
            chat_id="42",
            message_id="m1",
            sender_id="u1",
            sender_name="Alice",
            text="Hello",
            timestamp=time.time(),
        )

    def test_immediate_delivery_when_no_scheduler(self, tmp_path):
        """Without a scheduler, reply is sent immediately."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)

        handler, _ = self._make_handler(tmp_path, scheduler=None)
        handler.handle(self._make_msg(), channel)

        channel.send.assert_called_once()

    def test_immediate_delivery_when_agent_does_not_call_schedule_reply(self, tmp_path):
        """Scheduler present but agent does not call schedule_reply → immediate send."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)

        sched = _make_scheduler(tmp_path)
        runner = MagicMock(return_value="immediate answer")
        handler, _ = self._make_handler(tmp_path, scheduler=sched, agent_runner=runner)
        handler.handle(self._make_msg(), channel)

        channel.send.assert_called_once_with("42", "immediate answer")

    def test_scheduled_delivery_when_agent_calls_schedule_reply(self, tmp_path):
        """When agent calls schedule_reply, response is queued and NOT sent immediately."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)

        sched = _make_scheduler(tmp_path)

        def _runner_that_schedules(**kwargs):
            # Find the schedule_reply tool in active_tools_list and invoke it.
            tools = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                if getattr(t, "name", None) == "schedule_reply":
                    t.invoke({"text": "Scheduled reply!", "delay_minutes": 60})
                    break
            return "Scheduled reply!"

        handler, _ = self._make_handler(
            tmp_path, scheduler=sched, agent_runner=_runner_that_schedules
        )
        handler.handle(self._make_msg(), channel)

        # Immediate send should NOT have been called.
        channel.send.assert_not_called()
        # One message in the queue.
        assert len(sched._queue) == 1
        queued = next(iter(sched._queue.values()))
        assert queued.text == "Scheduled reply!"
        assert queued.status == "pending"

    def test_no_auto_cancel_on_new_message(self, tmp_path):
        """Incoming message no longer auto-cancels pending replies; agent manages the queue."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)

        sched = _make_scheduler(tmp_path)
        # Pre-populate a pending message.
        mid = sched.schedule("telegram", "42", "old reply", time.time() + 3600)

        handler, _ = self._make_handler(tmp_path, scheduler=sched)
        handler.handle(self._make_msg(), channel)

        # Message should remain pending — the agent decides whether to cancel.
        assert sched._queue[mid].status == "pending"

    def test_memory_updated_with_scheduled_text(self, tmp_path):
        """Memory is updated with the scheduled text (not an empty string)."""
        channel = MagicMock()
        sched = _make_scheduler(tmp_path)

        def _runner_schedules(**kwargs):
            tools = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                if getattr(t, "name", None) == "schedule_reply":
                    t.invoke({"text": "Delayed hello", "delay_minutes": 30})
                    break
            return "Delayed hello"

        handler, session = self._make_handler(
            tmp_path, scheduler=sched, agent_runner=_runner_schedules
        )
        handler.handle(self._make_msg(), channel)

        session.memory_manager.update.assert_called_once_with("Hello", "Delayed hello")

    def test_schedule_reply_tool_injected_into_active_tools(self, tmp_path):
        """schedule_reply tool is present in active_tools_list when scheduler is set."""
        sched = _make_scheduler(tmp_path)

        captured_tools: list = []

        def _capture_runner(**kwargs):
            captured_tools.extend(
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            return "ok"

        channel = MagicMock()
        handler, _ = self._make_handler(tmp_path, scheduler=sched, agent_runner=_capture_runner)
        handler.handle(self._make_msg(), channel)

        tool_names = [getattr(t, "name", None) for t in captured_tools]
        assert "schedule_reply" in tool_names

    def test_no_schedule_reply_tool_without_scheduler(self, tmp_path):
        """schedule_reply is NOT injected when scheduler is None."""
        captured_tools: list = []

        def _capture_runner(**kwargs):
            captured_tools.extend(
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            return "ok"

        channel = MagicMock()
        handler, _ = self._make_handler(tmp_path, scheduler=None, agent_runner=_capture_runner)
        handler.handle(self._make_msg(), channel)

        tool_names = [getattr(t, "name", None) for t in captured_tools]
        assert "schedule_reply" not in tool_names

    def test_queued_delivery_when_agent_calls_queue_reply(self, tmp_path):
        """When agent calls queue_reply, messages are queued and NOT sent immediately."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        sched = _make_scheduler(tmp_path)

        def _runner_that_queues(**kwargs):
            tools = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                if getattr(t, "name", None) == "queue_reply":
                    t.invoke({"text": "Message 1"})
                    t.invoke({"text": "Message 2", "gap_minutes": 5})
                    break
            return "Queued two messages"

        handler, _ = self._make_handler(tmp_path, scheduler=sched, agent_runner=_runner_that_queues)
        handler.handle(self._make_msg(), channel)

        channel.send.assert_not_called()
        pending = [m for m in sched._queue.values() if m.status == "pending"]
        assert len(pending) == 2
        texts = sorted([m.text for m in pending])
        assert texts == ["Message 1", "Message 2"]

    def test_queue_reply_tool_injected_when_scheduler_present(self, tmp_path):
        """queue_reply tool is present in active_tools when scheduler is set."""
        sched = _make_scheduler(tmp_path)
        captured_tools: list = []

        def _capture_runner(**kwargs):
            captured_tools.extend(
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            return "ok"

        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        handler, _ = self._make_handler(tmp_path, scheduler=sched, agent_runner=_capture_runner)
        handler.handle(self._make_msg(), channel)

        tool_names = [getattr(t, "name", None) for t in captured_tools]
        assert "queue_reply" in tool_names

    def test_schedule_and_queue_both_processed_in_same_turn(self, tmp_path):
        """Both schedule_reply and queue_reply calls are processed in the same turn."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        sched = _make_scheduler(tmp_path)

        def _runner_schedules_and_queues(**kwargs):
            tools = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            tool_map = {getattr(t, "name", None): t for t in tools}
            if "schedule_reply" in tool_map:
                tool_map["schedule_reply"].invoke({"text": "Scheduled text", "delay_minutes": 90})
            if "queue_reply" in tool_map:
                tool_map["queue_reply"].invoke({"text": "Queued text 1"})
                tool_map["queue_reply"].invoke({"text": "Queued text 2", "gap_minutes": 10})
            return "both scheduled and queued"

        handler, session = self._make_handler(
            tmp_path, scheduler=sched, agent_runner=_runner_schedules_and_queues
        )
        handler.handle(self._make_msg(), channel)

        # Immediate send should NOT have been called.
        channel.send.assert_not_called()

        # One scheduled message + two queued messages = 3 total pending.
        pending = [m for m in sched._queue.values() if m.status == "pending"]
        assert len(pending) == 3

        texts = sorted(m.text for m in pending)
        assert texts == ["Queued text 1", "Queued text 2", "Scheduled text"]

        # Memory text should include both the scheduled text and queued texts.
        call_args = session.memory_manager.update.call_args
        memory_text = call_args[0][1]
        assert "Scheduled text" in memory_text
        assert "Queued text 1" in memory_text
        assert "Queued text 2" in memory_text

    def test_edit_and_queue_both_processed_with_memory(self, tmp_path):
        """Both edit_last_reply and queue_reply are processed; memory includes both."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        channel.edit_message.return_value = SendResult(ok=True, message_id="m1")
        sched = _make_scheduler(tmp_path)

        def _runner_edits_and_queues(**kwargs):
            tools = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            tool_map = {getattr(t, "name", None): t for t in tools}
            if "edit_last_reply" in tool_map:
                tool_map["edit_last_reply"].invoke({"new_text": "Edited reply"})
            if "queue_reply" in tool_map:
                tool_map["queue_reply"].invoke({"text": "Queued after edit"})
            return "edited and queued"

        handler, session = self._make_handler(
            tmp_path, scheduler=sched, agent_runner=_runner_edits_and_queues
        )
        # Set a prior message ID so edit_last_reply is injected.
        session.last_sent_message_id = "m1"

        handler.handle(self._make_msg(), channel)

        # Edit should have been applied.
        channel.edit_message.assert_called_once_with("42", "m1", "Edited reply")

        # Queue item should have been scheduled.
        pending = [m for m in sched._queue.values() if m.status == "pending"]
        assert len(pending) == 1
        assert pending[0].text == "Queued after edit"

        # Immediate send should NOT have been called.
        channel.send.assert_not_called()

        # Memory should reference the queued text (queue_state wins over edit-only path).
        call_args = session.memory_manager.update.call_args
        memory_text = call_args[0][1]
        assert "Queued after edit" in memory_text

    def test_queue_reply_cap_at_ten_items(self, tmp_path):
        """Calling queue_reply more than 10 times returns an error and stops appending."""
        state = QueueReplyState()
        tool = create_queue_reply_tool(state)

        # First 10 calls should succeed.
        for i in range(10):
            result = tool.invoke({"text": f"Message {i + 1}"})
            assert "queued" in result.lower(), f"Expected success on call {i + 1}, got: {result}"

        assert len(state.items) == 10

        # 11th call should return an error and not append.
        error_result = tool.invoke({"text": "Overflow message"})
        assert len(state.items) == 10
        assert "limit" in error_result.lower() or "10" in error_result

        # 12th call should also be rejected.
        error_result_2 = tool.invoke({"text": "Also overflow"})
        assert len(state.items) == 10
        assert "limit" in error_result_2.lower() or "10" in error_result_2


# ---------------------------------------------------------------------------
# TestDynamicDispatch
# ---------------------------------------------------------------------------


class TestDynamicDispatch:
    """_next_wake_interval adapts sleep duration to pending queue state."""

    def test_next_wake_interval_adapts_to_queue(self, tmp_path):
        """_next_wake_interval returns time until next pending message, clamped."""
        sched = _make_scheduler(tmp_path)
        # No pending messages — returns default interval
        assert sched._next_wake_interval() == sched._dispatch_interval
        # Schedule a message due in 5 seconds
        sched.schedule("telegram", "42", "soon", time.time() + 5)
        interval = sched._next_wake_interval()
        assert 1.0 <= interval <= 6.0  # should be ~5s, clamped to min 1s


# ---------------------------------------------------------------------------
# TestQueueReplyState
# ---------------------------------------------------------------------------


class TestQueueReplyState:
    """QueueReplyState dataclass initial state and item construction."""

    def test_initial_values(self):
        state = QueueReplyState()
        assert state.items == []

    def test_item_creation(self):
        item = QueueReplyState.Item(text="hello", gap_minutes=5)
        assert item.text == "hello"
        assert item.gap_minutes == 5


# ---------------------------------------------------------------------------
# TestCreateQueueReplyTool
# ---------------------------------------------------------------------------


class TestCreateQueueReplyTool:
    """create_queue_reply_tool factory and closure behavior."""

    def test_single_call_appends_item(self):
        state = QueueReplyState()
        tool = create_queue_reply_tool(state)
        result = tool.invoke({"text": "First message"})
        assert len(state.items) == 1
        assert state.items[0].text == "First message"
        assert state.items[0].gap_minutes == 0
        assert "#1" in result

    def test_multiple_calls_append_in_order(self):
        state = QueueReplyState()
        tool = create_queue_reply_tool(state)
        tool.invoke({"text": "msg1"})
        tool.invoke({"text": "msg2", "gap_minutes": 30})
        tool.invoke({"text": "msg3", "gap_minutes": 60})
        assert len(state.items) == 3
        assert state.items[0].text == "msg1"
        assert state.items[1].text == "msg2"
        assert state.items[1].gap_minutes == 30
        assert state.items[2].text == "msg3"
        assert state.items[2].gap_minutes == 60

    def test_default_gap_is_zero(self):
        state = QueueReplyState()
        tool = create_queue_reply_tool(state)
        tool.invoke({"text": "no gap"})
        assert state.items[0].gap_minutes == 0

    def test_closure_isolation(self):
        state_a = QueueReplyState()
        state_b = QueueReplyState()
        tool_a = create_queue_reply_tool(state_a)
        _tool_b = create_queue_reply_tool(state_b)
        tool_a.invoke({"text": "only a"})
        assert len(state_a.items) == 1
        assert len(state_b.items) == 0


# ---------------------------------------------------------------------------
# TestQueueAfterTail
# ---------------------------------------------------------------------------


class TestQueueAfterTail:
    """MessageScheduler.queue_after_tail() ordering and persistence."""

    def test_empty_queue_uses_now(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        before = time.time()
        mid = sched.queue_after_tail("telegram", "42", "first")
        after = time.time()
        msg = sched._queue[mid]
        assert before <= msg.send_at <= after

    def test_appends_after_existing_tail(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        future = time.time() + 3600
        sched.schedule("telegram", "42", "existing", future)
        mid = sched.queue_after_tail("telegram", "42", "appended")
        msg = sched._queue[mid]
        assert msg.send_at >= future

    def test_gap_seconds_applied(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        future = time.time() + 3600
        sched.schedule("telegram", "42", "existing", future)
        mid = sched.queue_after_tail("telegram", "42", "gapped", gap_seconds=300)
        msg = sched._queue[mid]
        assert msg.send_at >= future + 300

    def test_sequential_calls_maintain_order(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid1 = sched.queue_after_tail("telegram", "42", "first")
        mid2 = sched.queue_after_tail("telegram", "42", "second", gap_seconds=60)
        mid3 = sched.queue_after_tail("telegram", "42", "third", gap_seconds=60)
        assert sched._queue[mid1].send_at < sched._queue[mid2].send_at
        assert sched._queue[mid2].send_at < sched._queue[mid3].send_at
        assert sched._queue[mid3].send_at - sched._queue[mid2].send_at >= 60

    def test_ignores_other_chats(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        far_future = time.time() + 99999
        sched.schedule("telegram", "99", "other chat", far_future)
        mid = sched.queue_after_tail("telegram", "42", "my chat")
        msg = sched._queue[mid]
        assert msg.send_at < far_future

    def test_ignores_non_pending_messages(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        far_future = time.time() + 99999
        old_mid = sched.schedule("telegram", "42", "cancelled", far_future)
        sched._queue[old_mid].status = "cancelled"
        mid = sched.queue_after_tail("telegram", "42", "new")
        msg = sched._queue[mid]
        assert msg.send_at < far_future

    def test_recipient_stored(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.queue_after_tail("telegram", "42", "hello", recipient="Alice")
        assert sched._queue[mid].recipient == "Alice"

    def test_persists_to_disk(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        mid = sched.queue_after_tail("telegram", "42", "persist me")
        sched2 = _make_scheduler(tmp_path)
        assert mid in sched2._queue
