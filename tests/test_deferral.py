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
        """defer_processing should not appear when current_depth >= max_depth."""
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
        msg = _make_msg()
        # Create a pending record at max_depth
        deferral_mgr.defer(msg, delay_seconds=300.0, depth=2)
        # Now simulate a re-processing pass that would be depth 3 — but the mgr
        # reports current_depth=2; tool omission happens when depth < max_depth is False.
        # Set deferral_depth to max_depth directly.
        with deferral_mgr._lock:
            deferral_mgr._records[msg.session_key].deferral_depth = 3

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
        handler.handle(msg, ch)
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
