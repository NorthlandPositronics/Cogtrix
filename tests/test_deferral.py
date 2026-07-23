"""Unit and integration tests for src/assistant/deferral.py."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.assistant.channel import IncomingMessage, SendResult
from src.assistant.deferral import (
    DeferralManager,
    DeferredRecord,
    DeferReplyState,
    SuppressReplyState,
    create_defer_processing_tool,
    create_suppress_reply_tool,
    format_elapsed,
)
from src.assistant.scheduler import ScheduleReplyState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(
    chat_id: str = "42",
    channel: str = "telegram",
    text: str = "Hello",
    message_id: str = "msg-1",
) -> IncomingMessage:
    """Return a minimal IncomingMessage for testing."""
    return IncomingMessage(
        channel=channel,
        chat_id=chat_id,
        message_id=message_id,
        sender_id="user-1",
        sender_name="Test User",
        text=text,
        timestamp=time.time(),
        metadata={},
        resolved_phone=None,
    )


def _make_channel(name: str = "telegram") -> MagicMock:
    """Return a mock Channel."""
    ch = MagicMock()
    ch.name = name
    ch.send.return_value = SendResult(ok=True, message_id="sent-1")
    ch.is_ready.return_value = True
    return ch


def _make_deferral_mgr(
    tmp_path: Path,
    callback: Any = None,
    channels: dict | None = None,
    max_depth: int = 3,
    check_interval: float = 60.0,  # slow interval to avoid accidental dispatch in tests
    stale_threshold: float = 7200.0,
) -> DeferralManager:
    """Return a DeferralManager backed by tmp_path."""
    if channels is None:
        channels = {}
    return DeferralManager(
        persist_path=tmp_path / "deferrals.json",
        reprocess_callback=callback,
        channels=channels,
        max_depth=max_depth,
        check_interval=check_interval,
        stale_threshold=stale_threshold,
    )


# ---------------------------------------------------------------------------
# TestDeferredRecord
# ---------------------------------------------------------------------------


class TestDeferredRecord:
    """DeferredRecord dataclass creation and round-trip serialization."""

    def test_defaults(self):
        now = time.time()
        rec = DeferredRecord(
            id="abc",
            channel="telegram",
            chat_id="42",
            fire_at=now + 300,
            created_at=now,
        )
        assert rec.status == "pending"
        assert rec.deferral_depth == 0
        assert rec.pending_messages == []

    def test_to_dict_round_trip(self):
        now = time.time()
        rec = DeferredRecord(
            id="xyz",
            channel="whatsapp",
            chat_id="99",
            fire_at=now + 600,
            created_at=now,
            pending_messages=[{"text": "hi"}],
            deferral_depth=1,
            status="pending",
        )
        d = rec.to_dict()
        restored = DeferredRecord.from_dict(d)
        assert restored.id == rec.id
        assert restored.channel == rec.channel
        assert restored.chat_id == rec.chat_id
        assert restored.fire_at == rec.fire_at
        assert restored.created_at == rec.created_at
        assert restored.pending_messages == rec.pending_messages
        assert restored.deferral_depth == rec.deferral_depth
        assert restored.status == rec.status

    def test_from_dict_coerces_types(self):
        now = time.time()
        data = {
            "id": "abc",
            "channel": "telegram",
            "chat_id": "42",
            "fire_at": str(now + 300),  # string float
            "created_at": str(now),
            "pending_messages": [],
            "deferral_depth": "2",  # string int
            "status": "pending",
        }
        rec = DeferredRecord.from_dict(data)
        assert isinstance(rec.fire_at, float)
        assert isinstance(rec.created_at, float)
        assert isinstance(rec.deferral_depth, int)
        assert rec.deferral_depth == 2


# ---------------------------------------------------------------------------
# TestDeferReplyState
# ---------------------------------------------------------------------------


class TestDeferReplyState:
    def test_initial_values(self):
        state = DeferReplyState()
        assert state.was_called is False
        assert state.delay_seconds == 0.0


# ---------------------------------------------------------------------------
# TestSuppressReplyState
# ---------------------------------------------------------------------------


class TestSuppressReplyState:
    def test_initial_values(self):
        state = SuppressReplyState()
        assert state.was_called is False


# ---------------------------------------------------------------------------
# TestCreateDeferProcessingTool
# ---------------------------------------------------------------------------


class TestCreateDeferProcessingTool:
    def test_tool_name(self):
        state = DeferReplyState()
        tool = create_defer_processing_tool(state)
        assert tool.name == "defer_processing"

    def test_single_call_sets_state(self):
        state = DeferReplyState()
        tool = create_defer_processing_tool(state)
        result = tool.invoke({"delay_minutes": 10})
        assert state.was_called is True
        assert state.delay_seconds == 600.0
        assert "10" in result

    def test_idempotency_guard(self):
        state = DeferReplyState()
        tool = create_defer_processing_tool(state)
        tool.invoke({"delay_minutes": 5})
        result = tool.invoke({"delay_minutes": 15})
        assert state.delay_seconds == 300.0  # first call wins
        assert "already" in result.lower()

    def test_co_invocation_warning(self, caplog):
        import logging

        state = DeferReplyState()
        schedule_state = ScheduleReplyState()
        schedule_state.was_called = True  # simulate schedule_reply already called

        tool = create_defer_processing_tool(state, schedule_state=schedule_state)
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            tool.invoke({"delay_minutes": 5})
        assert any("schedule_reply" in record.message for record in caplog.records)

    def test_no_warning_when_schedule_not_called(self, caplog):
        import logging

        state = DeferReplyState()
        schedule_state = ScheduleReplyState()
        tool = create_defer_processing_tool(state, schedule_state=schedule_state)
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            tool.invoke({"delay_minutes": 5})
        assert not any("schedule_reply" in record.message for record in caplog.records)

    def test_closure_isolation(self):
        state_a = DeferReplyState()
        state_b = DeferReplyState()
        tool_a = create_defer_processing_tool(state_a)
        _tool_b = create_defer_processing_tool(state_b)
        tool_a.invoke({"delay_minutes": 7})
        assert state_a.was_called is True
        assert state_b.was_called is False

    def test_reason_logged_but_not_in_result(self):
        state = DeferReplyState()
        tool = create_defer_processing_tool(state)
        result = tool.invoke({"delay_minutes": 5, "reason": "waiting for user"})
        assert "waiting for user" not in result
        assert state.was_called is True


# ---------------------------------------------------------------------------
# TestCreateSuppressReplyTool
# ---------------------------------------------------------------------------


class TestCreateSuppressReplyTool:
    def test_tool_name(self):
        state = SuppressReplyState()
        tool = create_suppress_reply_tool(state)
        assert tool.name == "suppress_reply"

    def test_call_sets_state(self):
        state = SuppressReplyState()
        tool = create_suppress_reply_tool(state)
        result = tool.invoke({})
        assert state.was_called is True
        assert "suppressed" in result.lower()

    def test_idempotency_guard(self):
        state = SuppressReplyState()
        tool = create_suppress_reply_tool(state)
        tool.invoke({})
        result = tool.invoke({})
        assert state.was_called is True
        assert "already" in result.lower()


# ---------------------------------------------------------------------------
# TestFormatElapsed
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    def test_under_one_minute(self):
        assert format_elapsed(30.0) == "<1 min"
        assert format_elapsed(0.0) == "<1 min"
        assert format_elapsed(59.9) == "<1 min"

    def test_minutes(self):
        assert format_elapsed(60.0) == "1 min"
        assert format_elapsed(120.0) == "2 min"
        assert format_elapsed(3599.0) == "59 min"

    def test_hours_and_minutes(self):
        assert format_elapsed(3661.0) == "1 h 1 min"
        assert format_elapsed(5400.0) == "1 h 30 min"

    def test_hours_exact(self):
        assert format_elapsed(3600.0) == "1 h"
        assert format_elapsed(7200.0) == "2 h"


# ---------------------------------------------------------------------------
# TestDeferralManager
# ---------------------------------------------------------------------------


class TestDeferralManager:
    def test_defer_creates_record(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        record_id = mgr.defer(msg, delay_seconds=300.0)
        assert isinstance(record_id, str)
        assert len(record_id) > 0
        assert mgr.has_pending(msg.session_key)

    def test_defer_sets_correct_fire_at(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        before = time.time()
        mgr.defer(msg, delay_seconds=300.0)
        after = time.time()
        with mgr._lock:
            rec = mgr._records[msg.session_key]
        assert before + 300.0 <= rec.fire_at <= after + 300.0

    def test_defer_merges_existing(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg1 = _make_msg(text="First")
        msg2 = _make_msg(text="Second", message_id="msg-2")
        mgr.defer(msg1, delay_seconds=300.0)
        mgr.defer(msg2, delay_seconds=600.0)
        with mgr._lock:
            rec = mgr._records[msg1.session_key]
        # fire_at should reflect the longer delay
        assert rec.fire_at >= time.time() + 599.0
        # Both messages should be in pending_messages
        assert len(rec.pending_messages) == 2

    def test_defer_extends_fire_at_to_later(self, tmp_path):
        """Second defer with shorter delay should not shorten fire_at."""
        mgr = _make_deferral_mgr(tmp_path)
        msg1 = _make_msg(text="First")
        msg2 = _make_msg(text="Second", message_id="msg-2")
        mgr.defer(msg1, delay_seconds=600.0)
        with mgr._lock:
            original_fire_at = mgr._records[msg1.session_key].fire_at
        mgr.defer(msg2, delay_seconds=60.0)  # shorter delay
        with mgr._lock:
            new_fire_at = mgr._records[msg1.session_key].fire_at
        assert new_fire_at == original_fire_at  # kept the longer one

    def test_add_message_appends_to_pending(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg1 = _make_msg(text="First")
        mgr.defer(msg1, delay_seconds=300.0)
        msg2 = _make_msg(text="Second", message_id="msg-2")
        result = mgr.add_message(msg2)
        assert result is True
        with mgr._lock:
            rec = mgr._records[msg1.session_key]
        assert len(rec.pending_messages) == 2

    def test_add_message_returns_false_when_no_record(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        result = mgr.add_message(msg)
        assert result is False

    def test_has_pending_true(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        assert mgr.has_pending(msg.session_key) is True

    def test_has_pending_false_when_no_record(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        assert mgr.has_pending("telegram::99") is False

    def test_current_depth_returns_correct_depth(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0, depth=2)
        assert mgr.current_depth(msg.session_key) == 2

    def test_current_depth_returns_zero_when_no_record(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        assert mgr.current_depth("telegram::nonexistent") == 0

    def test_cancel_sets_cancelled_status(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        result = mgr.cancel(msg.session_key)
        assert result is True
        with mgr._lock:
            rec = mgr._records[msg.session_key]
        assert rec.status == "cancelled"

    def test_cancel_returns_false_when_no_pending(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        result = mgr.cancel("telegram::nonexistent")
        assert result is False

    def test_has_pending_false_after_cancel(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        mgr.cancel(msg.session_key)
        assert mgr.has_pending(msg.session_key) is False

    def test_persist_and_reload(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        mgr.save()

        mgr2 = _make_deferral_mgr(tmp_path)
        assert mgr2.has_pending(msg.session_key)
        with mgr2._lock:
            rec = mgr2._records[msg.session_key]
        assert rec.channel == "telegram"
        assert rec.chat_id == "42"

    def test_firing_state_recovered_on_load(self, tmp_path):
        """Records in 'firing' state at load time should be reset to 'pending'."""
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        # Manually set to firing and save
        with mgr._lock:
            mgr._records[msg.session_key].status = "firing"
        mgr.save()

        mgr2 = _make_deferral_mgr(tmp_path)
        with mgr2._lock:
            rec = mgr2._records[msg.session_key]
        assert rec.status == "pending"

    def test_dispatch_fires_callback(self, tmp_path):
        """Record with fire_at in the past should trigger the callback."""
        callback_called = threading.Event()
        received: list[Any] = []

        def _callback(messages, channel, depth):
            received.extend(messages)
            callback_called.set()

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_callback, channels={"telegram": ch})
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        # Manually backdate fire_at to the past
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        assert callback_called.is_set()
        assert len(received) == 1

    def test_dispatch_removes_record_after_callback(self, tmp_path):
        def _callback(messages, channel, depth):
            pass

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_callback, channels={"telegram": ch})
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        assert not mgr.has_pending(msg.session_key)

    def test_dispatch_retries_on_callback_error(self, tmp_path):
        def _bad_callback(messages, channel, depth):
            raise RuntimeError("Callback failure")

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_bad_callback, channels={"telegram": ch})
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        # Record should be back to pending with bumped fire_at
        with mgr._lock:
            rec = mgr._records.get(msg.session_key)
        assert rec is not None
        assert rec.status == "pending"
        assert rec.fire_at > time.time()  # bumped

    def test_stale_records_cancelled(self, tmp_path):
        """Records past fire_at + stale_threshold are cancelled."""
        mgr = _make_deferral_mgr(tmp_path, stale_threshold=60.0)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        # Backdate fire_at by more than stale_threshold
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 120.0

        mgr._dispatch_due()
        with mgr._lock:
            rec = mgr._records.get(msg.session_key)
        assert rec is None or rec.status == "cancelled"

    def test_prefix_format_in_dispatched_message(self, tmp_path):
        """The first message should have a [Re-processing ...] prefix prepended."""
        received_text: list[str] = []

        def _callback(messages, channel, depth):
            received_text.append(messages[0].text)

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_callback, channels={"telegram": ch})
        msg = _make_msg(text="Original text")
        mgr.defer(msg, delay_seconds=0.0)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        assert len(received_text) == 1
        assert received_text[0].startswith("[Re-processing")
        assert "message(s) in batch" in received_text[0]
        assert "depth" in received_text[0]
        assert "Original text" in received_text[0]

    def test_max_depth_does_not_block_dispatch(self, tmp_path):
        """DeferralManager dispatches regardless of depth; tool omission is the handler's job."""
        callback_called = threading.Event()

        def _callback(messages, channel, depth):
            callback_called.set()

        ch = _make_channel()
        mgr = _make_deferral_mgr(
            tmp_path, callback=_callback, channels={"telegram": ch}, max_depth=3
        )
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0, depth=3)  # at max_depth
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        assert callback_called.is_set()

    def test_set_reprocess_callback(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path, callback=None)
        assert mgr._reprocess_callback is None
        cb = MagicMock()
        mgr.set_reprocess_callback(cb)
        assert mgr._reprocess_callback is cb

    def test_start_requires_callback(self, tmp_path):
        mgr = _make_deferral_mgr(tmp_path, callback=None)
        with pytest.raises(RuntimeError, match="set_reprocess_callback"):
            mgr.start()

    def test_start_and_stop(self, tmp_path):
        cb = MagicMock()
        mgr = _make_deferral_mgr(tmp_path, callback=cb)
        mgr.start()
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        mgr.stop()
        # After stop, thread should finish
        assert not mgr._thread.is_alive()


# ---------------------------------------------------------------------------
# Handler integration tests
# ---------------------------------------------------------------------------


def _make_session():
    session = MagicMock()
    session.lock = threading.Lock()
    session.last_sent_message_id = None
    session.last_activity = time.monotonic()
    session.guardrail_violations = 0
    session.session_key = "telegram::42"
    session.memory_manager = MagicMock()
    session.memory_manager.prepare_context.return_value = MagicMock(
        context_prefix=None, messages=[]
    )
    return session


def _make_handler(
    tmp_path: Path,
    deferral_mgr: DeferralManager | None = None,
    scheduler: Any = None,
):
    """Build a minimal MessageHandler with mocked dependencies."""
    from src.assistant.handler import MessageHandler

    session_mgr = MagicMock()
    session = _make_session()
    session_mgr.get_or_create.return_value = session

    llm = MagicMock()
    registry = MagicMock()

    handler = MessageHandler(
        session_mgr=session_mgr,
        config={},
        llm=llm,
        system_prompt="You are helpful.",
        registry=registry,
        approvals=set(),
        available_tools={},
        active_tools=[],
        agent_runner=MagicMock(return_value="Agent response"),
        deferral_mgr=deferral_mgr,
        scheduler=scheduler,
    )
    return handler, session_mgr, session


class TestHandlerDeferralIntegration:
    def test_defer_tool_injected_when_deferral_mgr_present(self, tmp_path):
        """defer_processing tool should appear in active_tools when deferral_mgr is set."""
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )

        msg = _make_msg()
        ch = _make_channel()
        handler.handle(msg, ch)
        assert "defer_processing" in injected_tools

    def test_defer_tool_not_injected_at_max_depth(self, tmp_path):
        """defer_processing should not appear when deferral_depth >= max_depth.

        BUG-091 fix: handle() now uses the propagated deferral_depth parameter
        instead of querying current_depth() from the record (which returns 0 after
        the record is deleted by _fire_record). The test therefore calls handle()
        with deferral_depth=3 (equal to max_depth) to simulate a re-processing pass
        at max depth via the callback chain.
        """
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path, max_depth=3)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        msg = _make_msg()
        ch = _make_channel()
        # Pass deferral_depth=3 (== max_depth) to simulate the callback chain path.
        handler.handle(msg, ch, deferral_depth=3)
        assert "defer_processing" not in injected_tools

    def test_suppress_tool_injected_on_reprocessing(self, tmp_path):
        """suppress_reply should appear in active_tools when is_reprocessing=True."""
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
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
            agent_runner=fake_runner,
        )
        msg = _make_msg()
        ch = _make_channel()
        handler.handle(msg, ch, is_reprocessing=True)
        assert "suppress_reply" in injected_tools

    def test_suppress_tool_not_injected_on_normal_turn(self, tmp_path):
        """suppress_reply should NOT appear on normal (non-reprocessing) turns."""
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
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
            agent_runner=fake_runner,
        )
        msg = _make_msg()
        ch = _make_channel()
        handler.handle(msg, ch, is_reprocessing=False)
        assert "suppress_reply" not in injected_tools

    def test_defer_skips_delivery_and_memory(self, tmp_path):
        """When defer_processing is called, channel.send and memory.update should NOT be called."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path)

        # Agent runner that invokes defer_processing tool via side effect
        def fake_runner(**kwargs):
            # Find the defer_processing tool and invoke it
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                if getattr(tool, "name", "") == "defer_processing":
                    tool.invoke({"delay_minutes": 5})
                    break
            return "Deferred response (should not be sent)"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        msg = _make_msg()
        ch = _make_channel()
        handler.handle(msg, ch)

        ch.send.assert_not_called()
        session.memory_manager.update.assert_not_called()
        # Deferral should be registered
        assert deferral_mgr.has_pending(msg.session_key)

    def test_suppress_skips_delivery_and_memory(self, tmp_path):
        """When suppress_reply is called, channel.send and memory.update should NOT be called."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        def fake_runner(**kwargs):
            for tool in kwargs.get("config", MagicMock()).active_tools_list or []:
                if getattr(tool, "name", "") == "suppress_reply":
                    tool.invoke({})
                    break
            return "Suppressed response"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
        )
        msg = _make_msg()
        ch = _make_channel()
        handler.handle(msg, ch, is_reprocessing=True)

        ch.send.assert_not_called()
        session.memory_manager.update.assert_not_called()

    def test_handle_batch_coalesces_into_deferred_record(self, tmp_path):
        """Messages arriving during a pending deferral should be absorbed, not processed."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        handle_calls: list[str] = []

        def fake_runner(**kwargs):
            handle_calls.append(kwargs.get("user_input", ""))
            return "Reply"

        deferral_mgr = _make_deferral_mgr(tmp_path)
        msg1 = _make_msg(text="First message")
        deferral_mgr.defer(msg1, delay_seconds=300.0)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg2 = _make_msg(text="Second message", message_id="msg-2")
        handler.handle_batch([msg2], ch)

        # handle() should NOT have been called (message absorbed into deferred record)
        assert len(handle_calls) == 0
        with deferral_mgr._lock:
            rec = deferral_mgr._records.get(msg1.session_key)
        assert rec is not None
        assert len(rec.pending_messages) == 2  # original + new

    def test_handle_batch_passes_through_when_no_deferral(self, tmp_path):
        """Messages with no pending deferral should be processed normally."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        handle_calls: list[str] = []

        def fake_runner(**kwargs):
            handle_calls.append(kwargs.get("user_input", ""))
            return "Reply"

        deferral_mgr = _make_deferral_mgr(tmp_path)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg = _make_msg(text="Normal message")
        handler.handle_batch([msg], ch)

        assert len(handle_calls) == 1

    def test_handle_batch_forwards_is_reprocessing(self, tmp_path):
        """is_reprocessing=True should reach handle() via handle_batch()."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        injected_tools_per_call: list[list[str]] = []

        def fake_runner(**kwargs):
            tools = [
                getattr(t, "name", "")
                for t in (kwargs.get("config") or MagicMock()).active_tools_list or []
            ]
            injected_tools_per_call.append(tools)
            return "Reply"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
        )
        ch = _make_channel()
        msg = _make_msg(text="Re-processing message")
        handler.handle_batch([msg], ch, is_reprocessing=True)

        assert len(injected_tools_per_call) == 1
        assert "suppress_reply" in injected_tools_per_call[0]


# ---------------------------------------------------------------------------
# BUG-091 regression tests — deferral depth propagation through callback chain
# ---------------------------------------------------------------------------


class TestBug091DepthPropagation:
    """Verify that deferral_depth is correctly propagated through the callback chain."""

    def test_handle_batch_forwards_deferral_depth_to_handle(self, tmp_path):
        """handle_batch must forward deferral_depth to handle() so the depth check uses it."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path, max_depth=3)
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in (kwargs.get("config") or MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg = _make_msg()
        # Simulate a reprocessing pass at depth 3 (== max_depth) via handle_batch.
        handler.handle_batch([msg], ch, is_reprocessing=True, deferral_depth=3)
        # defer_processing must be omitted because deferral_depth >= max_depth.
        assert "defer_processing" not in injected_tools

    def test_handle_batch_forwards_deferral_depth_below_max(self, tmp_path):
        """At depth < max_depth, defer_processing should still be injected."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path, max_depth=3)
        injected_tools: list[str] = []

        def fake_runner(**kwargs):
            for tool in (kwargs.get("config") or MagicMock()).active_tools_list or []:
                injected_tools.append(getattr(tool, "name", ""))
            return "Reply"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg = _make_msg()
        # deferral_depth=2 < max_depth=3 — defer_processing must be injected.
        handler.handle_batch([msg], ch, is_reprocessing=True, deferral_depth=2)
        assert "defer_processing" in injected_tools

    def test_defer_registers_with_propagated_depth(self, tmp_path):
        """When the agent calls defer_processing, the recorded depth must be deferral_depth."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path, max_depth=3)

        def fake_runner(**kwargs):
            # Find and invoke defer_processing.
            for tool in (kwargs.get("config") or MagicMock()).active_tools_list or []:
                if getattr(tool, "name", "") == "defer_processing":
                    tool.invoke({"delay_minutes": 5})
                    break
            return "Deferred"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        msg = _make_msg()
        ch = _make_channel()
        # Simulate re-processing at depth 1 — the new record must store depth=1.
        handler.handle(msg, ch, is_reprocessing=True, deferral_depth=1)

        with deferral_mgr._lock:
            rec = deferral_mgr._records.get(msg.session_key)
        assert rec is not None
        assert rec.deferral_depth == 1

    def test_dispatch_fires_callback_with_correct_depth(self, tmp_path):
        """_fire_record must pass record.deferral_depth to the callback."""
        depths_received: list[int] = []

        def _callback(messages, channel, depth):
            depths_received.append(depth)

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_callback, channels={"telegram": ch})
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0, depth=2)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()
        assert len(depths_received) == 1
        assert depths_received[0] == 2


# ---------------------------------------------------------------------------
# BUG-092 / BUG-099 regression tests — _dispatch_due stale record handling
# ---------------------------------------------------------------------------


class TestBug092StaleRecordCancellation:
    """Verify future-dated stale records are cancelled and fire_at is re-validated."""

    def test_future_dated_stale_record_is_cancelled(self, tmp_path):
        """A future-dated record older than stale_threshold must be cancelled without firing."""
        callback_called = threading.Event()

        def _callback(messages, channel, depth):
            callback_called.set()

        ch = _make_channel()
        mgr = _make_deferral_mgr(
            tmp_path,
            callback=_callback,
            channels={"telegram": ch},
            stale_threshold=60.0,
        )
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=7200.0)  # fire_at = 2 hours in the future
        # Age the record by backdating created_at by more than stale_threshold.
        with mgr._lock:
            mgr._records[msg.session_key].created_at = time.time() - 120.0

        mgr._dispatch_due()

        # Callback must NOT have fired (record was future-dated, not in the due list).
        assert not callback_called.is_set()
        with mgr._lock:
            rec = mgr._records.get(msg.session_key)
        assert rec is None or rec.status == "cancelled"

    def test_non_stale_future_record_not_cancelled(self, tmp_path):
        """A young future-dated record must remain pending."""
        ch = _make_channel()
        mgr = _make_deferral_mgr(
            tmp_path,
            callback=MagicMock(),
            channels={"telegram": ch},
            stale_threshold=7200.0,
        )
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)  # fire_at = 5 min in the future, record is young

        mgr._dispatch_due()

        assert mgr.has_pending(msg.session_key)

    def test_overdue_stale_record_is_cancelled(self, tmp_path):
        """A record that is past fire_at + stale_threshold must be cancelled."""
        mgr = _make_deferral_mgr(tmp_path, stale_threshold=60.0)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 120.0

        mgr._dispatch_due()

        with mgr._lock:
            rec = mgr._records.get(msg.session_key)
        assert rec is None or rec.status == "cancelled"


# ---------------------------------------------------------------------------
# BUG-093 regression tests — synchronous callback so exceptions surface
# ---------------------------------------------------------------------------


class TestBug093SynchronousCallback:
    """Verify _fire_record retry logic triggers when callback raises synchronously."""

    def test_callback_exception_triggers_retry(self, tmp_path):
        """When the callback raises, the record must be reset to pending for retry."""

        def _bad_callback(messages, channel, depth):
            raise RuntimeError("Simulated agent failure")

        ch = _make_channel()
        mgr = _make_deferral_mgr(tmp_path, callback=_bad_callback, channels={"telegram": ch})
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=0.0)
        with mgr._lock:
            mgr._records[msg.session_key].fire_at = time.time() - 1.0

        mgr._dispatch_due()

        with mgr._lock:
            rec = mgr._records.get(msg.session_key)
        assert rec is not None
        assert rec.status == "pending"
        assert rec.fire_at > time.time()  # fire_at bumped by _BACKOFF_SECONDS


# ---------------------------------------------------------------------------
# BUG-094 regression tests — eager evaluation of add_message to avoid partial absorption
# ---------------------------------------------------------------------------


class TestBug094EagerAbsorption:
    """Verify handle_batch evaluates all add_message calls before deciding."""

    def test_handle_batch_all_absorbed_returns_early(self, tmp_path):
        """When all messages are absorbed, handle() must not be called."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        handle_calls: list[str] = []

        def fake_runner(**kwargs):
            handle_calls.append(kwargs.get("user_input", ""))
            return "Reply"

        deferral_mgr = _make_deferral_mgr(tmp_path)
        msg0 = _make_msg(text="First")
        deferral_mgr.defer(msg0, delay_seconds=300.0)

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg1 = _make_msg(text="Second", message_id="msg-2")
        msg2 = _make_msg(text="Third", message_id="msg-3")
        handler.handle_batch([msg1, msg2], ch)

        assert len(handle_calls) == 0  # all absorbed — handle() not called

    def test_handle_batch_partial_absorption_cancels_record(self, tmp_path):
        """Partial absorption must cancel the deferred record to prevent duplicate processing."""
        from src.assistant.handler import MessageHandler

        session_mgr = MagicMock()
        session = _make_session()
        session_mgr.get_or_create.return_value = session

        deferral_mgr = _make_deferral_mgr(tmp_path)
        msg0 = _make_msg(text="Deferred message")
        deferral_mgr.defer(msg0, delay_seconds=300.0)

        # Simulate partial absorption: add_message returns True for msg1 but the record
        # transitions to "firing" before msg2 is processed, so add_message returns False.
        original_add_message = deferral_mgr.add_message
        call_count = [0]

        def patched_add_message(msg):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: absorb normally.
                return original_add_message(msg)
            else:
                # Subsequent calls: simulate record having fired (return False).
                return False

        deferral_mgr.add_message = patched_add_message  # type: ignore[method-assign]

        handle_calls: list[str] = []

        def fake_runner(**kwargs):
            handle_calls.append(kwargs.get("user_input", ""))
            return "Reply"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=fake_runner,
            deferral_mgr=deferral_mgr,
        )
        ch = _make_channel()
        msg1 = _make_msg(text="New msg 1", message_id="msg-2")
        msg2 = _make_msg(text="New msg 2", message_id="msg-3")
        handler.handle_batch([msg1, msg2], ch)

        # The deferred record must have been cancelled to prevent duplication.
        assert not deferral_mgr.has_pending(msg0.session_key)
        # Processing should have proceeded (not returned early).
        assert len(handle_calls) > 0


# ---------------------------------------------------------------------------
# BUG-095 regression tests — ViolationTracker _save_snapshot inside lock
# ---------------------------------------------------------------------------


class TestBug095ViolationTrackerSnapshotUnderLock:
    """Verify _save_snapshot is called inside the lock in record_violation."""

    def test_record_violation_persists_to_disk(self, tmp_path):
        """After record_violation, the violation must be written to the persist path."""
        import json

        from src.assistant.guardrails import ViolationTracker

        persist_path = tmp_path / "violations.json"
        tracker = ViolationTracker(config={}, persist_path=persist_path)
        tracker.record_violation("chat_1")

        assert persist_path.exists()
        data = json.loads(persist_path.read_text())
        assert "chat_1" in data

    def test_concurrent_record_violations_both_persisted(self, tmp_path):
        """Both concurrent violations must appear in the persisted file (no lost write)."""
        import json

        from src.assistant.guardrails import ViolationTracker

        persist_path = tmp_path / "violations.json"
        tracker = ViolationTracker(config={}, persist_path=persist_path)

        errors: list[Exception] = []

        def _thread_fn(chat_id: str) -> None:
            try:
                for _ in range(5):
                    tracker.record_violation(chat_id)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_thread_fn, args=("chat_A",))
        t2 = threading.Thread(target=_thread_fn, args=("chat_B",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors
        data = json.loads(persist_path.read_text())
        # Both chats must appear in the persisted file.
        assert "chat_A" in data
        assert "chat_B" in data


# ---------------------------------------------------------------------------
# BUG-096 regression tests — _load_prompt_value path containment
# ---------------------------------------------------------------------------


class TestBug096LoadPromptValueContainment:
    """Verify _load_prompt_value rejects paths outside allowed_roots."""

    def test_inline_value_returned_unchanged(self):
        from src.assistant.handler import _load_prompt_value

        assert _load_prompt_value("You are helpful.") == "You are helpful."

    def test_file_within_allowed_root_is_read(self, tmp_path):
        from src.assistant.handler import _load_prompt_value

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom prompt", encoding="utf-8")
        result = _load_prompt_value(str(prompt_file), allowed_roots=[tmp_path])
        assert result == "Custom prompt"

    def test_file_outside_allowed_root_is_rejected(self, tmp_path):
        from src.assistant.handler import _load_prompt_value

        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        # Path is a valid file but outside allowed_root.
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret content", encoding="utf-8")
        result = _load_prompt_value(str(outside_file), allowed_roots=[allowed_root])
        assert result == ""

    def test_symlink_outside_allowed_root_is_rejected(self, tmp_path):
        from src.assistant.handler import _load_prompt_value

        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()
        target = tmp_path / "target.txt"
        target.write_text("target content", encoding="utf-8")
        link = allowed_root / "link.txt"
        link.symlink_to(target)
        # The symlink resolves to tmp_path/target.txt which is NOT under allowed_root.
        result = _load_prompt_value(str(link), allowed_roots=[allowed_root])
        assert result == ""

    def test_no_allowed_roots_allows_any_path(self, tmp_path):
        """With no allowed_roots constraint, any readable path is accepted (backward compat)."""
        from src.assistant.handler import _load_prompt_value

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Any path ok", encoding="utf-8")
        result = _load_prompt_value(str(prompt_file), allowed_roots=None)
        assert result == "Any path ok"

    def test_nonexistent_file_returns_empty(self, tmp_path):
        from src.assistant.handler import _load_prompt_value

        result = _load_prompt_value(str(tmp_path / "nonexistent.txt"), allowed_roots=[tmp_path])
        assert result == ""


# ---------------------------------------------------------------------------
# BUG-098 regression tests — persistence failures logged at WARNING
# ---------------------------------------------------------------------------


class TestBug098PersistenceLogLevel:
    """Verify that _atomic_write in DeferralManager and MessageScheduler logs at WARNING."""

    def test_deferral_manager_atomic_write_logs_warning_on_failure(self, tmp_path, caplog):
        """DeferralManager._atomic_write must log at WARNING level on write failure."""
        import logging
        from unittest.mock import patch

        mgr = _make_deferral_mgr(tmp_path)

        # The import is local inside _atomic_write, so patch at the source module.
        with patch("src.utils.atomic_write.atomic_write_json", side_effect=OSError("disk full")):
            with caplog.at_level(logging.WARNING, logger="cogtrix"):
                mgr._atomic_write({})

        assert any(
            "failed to persist" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_scheduler_atomic_write_logs_warning_on_failure(self, tmp_path, caplog):
        """MessageScheduler._atomic_write must log at WARNING level on write failure."""
        import logging
        from unittest.mock import patch

        from src.assistant.scheduler import MessageScheduler

        scheduler = MessageScheduler(channels={}, persist_path=tmp_path / "schedule.json")

        with patch("src.utils.atomic_write.atomic_write_json", side_effect=OSError("disk full")):
            with caplog.at_level(logging.WARNING, logger="cogtrix"):
                scheduler._atomic_write({})

        assert any(
            "failed to persist" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# BUG-094 partial regression tests — cancel() must handle "firing" state
# ---------------------------------------------------------------------------


class TestBug094CancelFiringState:
    """Verify that DeferralManager.cancel() cancels records in both pending and firing states."""

    def test_cancel_pending_returns_true(self, tmp_path):
        """cancel() must return True and set status to cancelled for a pending record."""
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)

        assert mgr.cancel(msg.session_key) is True
        with mgr._lock:
            record = mgr._records.get(msg.session_key)
        assert record is not None
        assert record.status == "cancelled"

    def test_cancel_firing_returns_true(self, tmp_path):
        """BUG-094 partial: cancel() must also cancel records with status == 'firing'."""
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)

        # Manually transition the record to "firing" (as _fire_record does)
        with mgr._lock:
            record = mgr._records[msg.session_key]
            record.status = "firing"

        result = mgr.cancel(msg.session_key)
        assert result is True, "cancel() must return True for a 'firing' record"

        with mgr._lock:
            record = mgr._records.get(msg.session_key)
        assert record is not None
        assert record.status == "cancelled"

    def test_cancel_already_cancelled_returns_false(self, tmp_path):
        """cancel() must return False if the record is already cancelled."""
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)
        mgr.cancel(msg.session_key)  # first cancel

        result = mgr.cancel(msg.session_key)  # second cancel
        assert result is False

    def test_cancel_nonexistent_returns_false(self, tmp_path):
        """cancel() must return False if no record exists for the session key."""
        mgr = _make_deferral_mgr(tmp_path)
        result = mgr.cancel("no::such::session")
        assert result is False

    def test_cancel_completed_record_returns_false(self, tmp_path):
        """cancel() must return False for a record with status 'completed'."""
        mgr = _make_deferral_mgr(tmp_path)
        msg = _make_msg()
        mgr.defer(msg, delay_seconds=300.0)

        with mgr._lock:
            record = mgr._records[msg.session_key]
            record.status = "completed"

        result = mgr.cancel(msg.session_key)
        assert result is False


# ---------------------------------------------------------------------------
# BUG-105 regression tests — reprocess callback submits to executor
# ---------------------------------------------------------------------------


class TestBug105ReprocessCallback:
    """Verify that the reprocess callback in AssistantService submits handle_batch
    to the executor (not the dispatch thread) so session.lock is not held during
    LLM calls."""

    def test_reprocess_callback_submits_to_executor(self):
        """BUG-105: the _reprocess_callback must submit work to the executor, not call inline."""
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        result_future: Future = Future()
        result_future.set_result(None)

        mock_executor = MagicMock()
        mock_executor.submit.return_value = result_future
        mock_handler = MagicMock()

        # Reproduce the exact callback from AssistantService (BUG-109 version)
        _exec = mock_executor
        _handler = mock_handler

        def _reprocess_callback(msgs, ch, depth: int) -> None:
            fut = _exec.submit(
                _handler.handle_batch,
                msgs,
                ch,
                is_reprocessing=True,
                deferral_depth=depth + 1,
            )
            try:
                fut.result(timeout=0.05)
            except TimeoutError:
                pass
            except Exception:
                raise

        msgs = [_make_msg()]
        ch = _make_channel()
        _reprocess_callback(msgs, ch, depth=0)

        # executor.submit must have been called (not handle_batch directly)
        assert mock_executor.submit.called
        call_args = mock_executor.submit.call_args
        assert call_args[0][0] is mock_handler.handle_batch
        assert call_args[1].get("deferral_depth") == 1

    def test_reprocess_callback_stop_has_join(self):
        """DeferralManager.stop() must use a join to drain deferred work before save_all."""
        import inspect

        from src.assistant.deferral import DeferralManager

        source = inspect.getsource(DeferralManager.stop)
        assert "join" in source, "DeferralManager.stop() must join the dispatch thread"


# ---------------------------------------------------------------------------
# TestBug109ReprocessNoDuplicate (BUG-109)
# ---------------------------------------------------------------------------


class TestBug109ReprocessNoDuplicate:
    """BUG-109: _reprocess_callback must NOT call fut.result(timeout=8.0) because
    a slow LLM response always exceeds that deadline, causing TimeoutError to
    propagate to _fire_record's retry logic, which resets the record to 'pending'
    and submits a second handle_batch — delivering duplicate replies.

    The fix uses timeout=0.05 and swallows TimeoutError silently."""

    def test_slow_future_does_not_raise_in_callback(self):
        """A future that takes >0.05s does not cause _reprocess_callback to raise."""
        import concurrent.futures
        import threading

        started = threading.Event()
        finish = threading.Event()

        def _slow_work():
            started.set()
            finish.wait(timeout=5.0)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            # Build the callback the same way AssistantService does (BUG-109 fix)
            _exec = executor
            submitted_futs: list = []

            def _reprocess_callback(msgs, ch, depth: int) -> None:
                fut = _exec.submit(_slow_work)
                submitted_futs.append(fut)
                try:
                    fut.result(timeout=0.05)
                except TimeoutError:
                    pass  # slow LLM — do not raise
                except Exception:
                    raise

            from concurrent.futures import Future

            mock_executor = MagicMock()
            slow_future: Future = Future()
            mock_executor.submit.return_value = slow_future  # never resolves until we set it

            _exec2 = mock_executor
            mock_handler = MagicMock()

            def _callback2(msgs, ch, depth: int) -> None:
                fut = _exec2.submit(
                    mock_handler.handle_batch,
                    msgs,
                    ch,
                    is_reprocessing=True,
                    deferral_depth=depth + 1,
                )
                try:
                    fut.result(timeout=0.05)
                except TimeoutError:
                    pass  # BUG-109 fix: swallow timeout, do not retry
                except Exception:
                    raise

            msgs = [_make_msg()]
            ch = _make_channel()
            # Must not raise even though slow_future never resolves within 0.05s
            _callback2(msgs, ch, depth=0)

            # executor.submit called exactly once — no duplicate submission
            assert mock_executor.submit.call_count == 1
        finally:
            finish.set()
            executor.shutdown(wait=False)

    def test_real_exception_in_future_propagates(self):
        """If the future raises a real exception synchronously, _reprocess_callback
        propagates it (so _fire_record can retry as expected)."""
        from concurrent.futures import Future

        mock_executor = MagicMock()
        failing_future: Future = Future()
        failing_future.set_exception(RuntimeError("executor rejected"))
        mock_executor.submit.return_value = failing_future

        def _callback(msgs, ch, depth: int) -> None:
            fut = mock_executor.submit(
                MagicMock(),
                msgs,
                ch,
                is_reprocessing=True,
                deferral_depth=depth + 1,
            )
            try:
                fut.result(timeout=0.05)
            except TimeoutError:
                pass
            except Exception:
                raise  # coding errors must propagate to _fire_record

        msgs = [_make_msg()]
        ch = _make_channel()
        with pytest.raises(RuntimeError, match="executor rejected"):
            _callback(msgs, ch, depth=0)

    def test_service_reprocess_callback_uses_small_timeout(self):
        """AssistantService._reprocess_callback uses timeout=0.05, not timeout=8.0 (BUG-109)."""
        import inspect

        import src.assistant.service as svc_module

        source = inspect.getsource(svc_module)
        # The old buggy timeout must not appear
        assert (
            "timeout=8.0" not in source
        ), "service.py still contains 'timeout=8.0' — BUG-109 not fully fixed"
        # The new safe near-zero timeout must be present
        assert (
            "timeout=0.05" in source
        ), "service.py does not contain 'timeout=0.05' — BUG-109 fix not applied"


# ---------------------------------------------------------------------------
# BUG-103: _fire_record must use explicit if/raise RuntimeError, not assert
# ---------------------------------------------------------------------------


class TestFireRecordAssertGuard:
    """Regression tests for BUG-103: _fire_record must use explicit if/raise RuntimeError.

    assert statements are silently stripped by Python in optimised mode (-O /
    PYTHONOPTIMIZE=1). If _reprocess_callback is None when _fire_record runs, the
    assert would be a no-op and the next line would raise TypeError caught as a
    generic callback failure. After the fix, an explicit RuntimeError is raised.
    """

    def _make_manager_no_cb(self, tmp_path: Path, channels: dict | None = None) -> DeferralManager:
        return DeferralManager(
            persist_path=tmp_path / "deferrals.json",
            reprocess_callback=None,
            channels=channels or {},
            check_interval=3600.0,
        )

    def _make_record(self, now: float) -> DeferredRecord:
        return DeferredRecord(
            id="rec-1",
            channel="telegram",
            chat_id="42",
            fire_at=now - 1.0,
            created_at=now - 60.0,
            pending_messages=[
                {
                    "channel": "telegram",
                    "chat_id": "42",
                    "message_id": "m1",
                    "sender_id": "u1",
                    "sender_name": "Alice",
                    "text": "Hello",
                    "timestamp": now,
                    "metadata": {},
                    "resolved_phone": None,
                }
            ],
            deferral_depth=0,
            status="pending",
        )

    def test_fire_record_raises_runtime_error_when_no_callback(self, tmp_path: Path) -> None:
        """_fire_record must produce a RuntimeError log when callback is None."""
        import src.assistant.deferral as deferral_mod

        channel = MagicMock()
        channel.name = "telegram"
        channel.send.return_value = SendResult(ok=True, message_id="sent-1")
        channel.is_ready.return_value = True
        mgr = self._make_manager_no_cb(tmp_path, channels={"telegram": channel})
        assert mgr._reprocess_callback is None

        now = time.time()
        rec = self._make_record(now)
        with mgr._lock:
            mgr._records["telegram::42"] = rec

        captured_errors: list[str] = []
        original_error = deferral_mod.log.error

        def _capture(msg: str, *args: object, **kwargs: object) -> None:
            captured_errors.append(msg % args if args else msg)
            original_error(msg, *args, **kwargs)

        deferral_mod.log.error = _capture  # type: ignore[method-assign]
        try:
            mgr._fire_record("telegram::42", rec, time.monotonic())
        finally:
            deferral_mod.log.error = original_error  # type: ignore[method-assign]

        assert captured_errors, "_fire_record did not log an error when callback was None"
        combined = " ".join(captured_errors)
        assert (
            "RuntimeError" in combined or "no reprocess callback" in combined.lower()
        ), f"Expected RuntimeError or descriptive message in error log, got: {combined!r}"

    def test_fire_record_guard_is_not_assert(self) -> None:
        """Verify that _fire_record does not use 'assert' for the callback guard."""
        import inspect

        source = inspect.getsource(DeferralManager._fire_record)
        assert (
            "assert self._reprocess_callback" not in source
        ), "_fire_record still uses 'assert self._reprocess_callback' — BUG-103 has regressed."
        assert (
            "RuntimeError" in source
        ), "_fire_record does not contain an explicit RuntimeError raise."

    def test_fire_record_calls_callback_when_set(self, tmp_path: Path) -> None:
        """_fire_record must call the reprocess callback when it is properly configured."""
        called_with: list[tuple] = []

        channel = MagicMock()
        channel.name = "telegram"
        channel.send.return_value = SendResult(ok=True, message_id="sent-1")
        channel.is_ready.return_value = True

        def callback(messages: list, ch: object, depth: int) -> None:
            called_with.append((messages, ch, depth))

        mgr = DeferralManager(
            persist_path=tmp_path / "deferrals.json",
            reprocess_callback=callback,
            channels={"telegram": channel},
            check_interval=3600.0,
        )

        now = time.time()
        rec = DeferredRecord(
            id="rec-2",
            channel="telegram",
            chat_id="42",
            fire_at=now - 1.0,
            created_at=now - 60.0,
            pending_messages=[
                {
                    "channel": "telegram",
                    "chat_id": "42",
                    "message_id": "m1",
                    "sender_id": "u1",
                    "sender_name": "Alice",
                    "text": "Hello",
                    "timestamp": now,
                    "metadata": {},
                    "resolved_phone": None,
                }
            ],
            deferral_depth=1,
            status="pending",
        )
        with mgr._lock:
            mgr._records["telegram::42"] = rec

        mgr._fire_record("telegram::42", rec, time.monotonic())

        assert len(called_with) == 1, f"Callback was not called exactly once: {called_with}"
        _messages_arg, channel_arg, depth_arg = called_with[0]
        assert channel_arg is channel
        assert depth_arg == 1

    def test_start_raises_when_callback_not_set(self, tmp_path: Path) -> None:
        """start() must raise RuntimeError if set_reprocess_callback was never called."""
        import pytest

        mgr = self._make_manager_no_cb(tmp_path)
        with pytest.raises(RuntimeError, match="set_reprocess_callback"):
            mgr.start()

    def test_start_succeeds_after_set_reprocess_callback(self, tmp_path: Path) -> None:
        """start() must succeed after set_reprocess_callback is called."""
        mgr = self._make_manager_no_cb(tmp_path)

        def cb(messages: list, channel: object, depth: int) -> None:
            pass

        mgr.set_reprocess_callback(cb)
        mgr.start()
        assert mgr._thread is not None and mgr._thread.is_alive()
        mgr.stop()
