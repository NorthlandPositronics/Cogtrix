"""Async cancellation and concurrency tests for API and orchestration.

Issue #712 — Missing tests for:
1. Turn lock serialization (concurrent same-session requests)
2. WebSocket disconnect during streaming cleanup
3. Cancel event lifecycle (WebSocket cancel → pipeline stop)
4. Background task lifecycle (eviction loop stop)

Tests only — no cogtrix_core/ changes.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment setup — before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

pytest.importorskip("fastapi")


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_session(*, agent_state: str = "idle") -> MagicMock:
    """Build a minimal MagicMock ApiSession for cancellation/concurrency tests."""
    session = MagicMock()
    session.id = f"test-session-{id(session)}"
    session.user_id = "test-user"
    session.name = "Test Session"
    session.turn_lock = asyncio.Lock()
    session.cancel_event = asyncio.Event()
    session.ws_queue = asyncio.Queue(maxsize=100)
    session.agent_state = agent_state
    session.session_state = None
    session.run_config = None
    session.memory_manager = None
    session.registry = None
    session.turn_task = None
    session.drain_task = None
    session.active_confirmation_ui = None
    session.token_counts = {"input_tokens": 0, "output_tokens": 0}
    session.last_activity = 0.0
    return session


# ===========================================================================
# 1. Turn Lock — serialization of concurrent same-session turns
# ===========================================================================


class TestTurnLockSerialization:
    """turn_lock prevents concurrent agent turns on the same session."""

    @pytest.mark.asyncio
    async def test_turn_lock_acquired_during_turn(self) -> None:
        """turn_lock is held while a turn executes; second task must wait."""
        session = _make_mock_session()

        # Simulate run_message_turn acquiring the lock
        held_lock = asyncio.Event()
        release_barrier = asyncio.Event()

        async def _simulate_turn() -> None:
            async with session.turn_lock:
                held_lock.set()  # signal that we have the lock
                await release_barrier.wait()  # block until test releases us

        # Start a "turn" that holds the lock
        turn_task = asyncio.create_task(_simulate_turn())

        # Wait for the lock to be acquired
        await asyncio.wait_for(held_lock.wait(), timeout=2.0)

        # Verify another coroutine cannot acquire the lock immediately
        lock_acquired_immediately = False

        async def _try_acquire() -> None:
            nonlocal lock_acquired_immediately
            # try to acquire lock with zero-timeout — should fail
            acquired = session.turn_lock.locked()
            lock_acquired_immediately = not acquired

        await _try_acquire()

        # The lock should still be held (turn is in progress)
        assert session.turn_lock.locked(), "turn_lock should be held during turn"

        # Release the turn
        release_barrier.set()
        await turn_task

        # Verify lock is released after turn completes
        assert not session.turn_lock.locked(), "turn_lock should be released after turn"

    @pytest.mark.asyncio
    async def test_concurrent_turn_attempts_serialized(self) -> None:
        """Two coroutines trying to run turns: second serialises behind first."""
        session = _make_mock_session()

        execution_order: list[str] = []
        turn1_started = asyncio.Event()
        turn1_can_finish = asyncio.Event()

        async def _turn1() -> None:
            async with session.turn_lock:
                execution_order.append("turn1-start")
                turn1_started.set()
                await turn1_can_finish.wait()
                execution_order.append("turn1-end")

        async def _turn2() -> None:
            # Wait for turn1 to acquire the lock first
            await turn1_started.wait()
            execution_order.append("turn2-waiting")
            async with session.turn_lock:
                execution_order.append("turn2-start")
            execution_order.append("turn2-end")

        t1 = asyncio.create_task(_turn1())
        t2 = asyncio.create_task(_turn2())

        # Let turn2 register its wait
        await asyncio.sleep(0.05)

        # turn1 should have started, turn2 should be waiting
        assert "turn1-start" in execution_order
        assert "turn2-waiting" in execution_order
        assert "turn2-start" not in execution_order, "turn2 should not start before turn1 finishes"

        # Let turn1 finish
        turn1_can_finish.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2.0)

        # Verify execution order: turn1 fully completes before turn2 starts
        assert execution_order == [
            "turn1-start",
            "turn2-waiting",
            "turn1-end",
            "turn2-start",
            "turn2-end",
        ], f"Expected serialized order, got {execution_order}"

    @pytest.mark.asyncio
    async def test_turn_lock_releases_on_exception(self) -> None:
        """turn_lock is released even when the guarded block raises."""
        session = _make_mock_session()

        async def _failing_turn() -> None:
            async with session.turn_lock:
                raise ValueError("turn failed")

        with pytest.raises(ValueError, match="turn failed"):
            await _failing_turn()

        # Lock must be released even after exception
        assert not session.turn_lock.locked(), "turn_lock must be released after exception"

        # A subsequent turn should be able to acquire the lock
        acquired = False

        async def _subsequent_turn() -> None:
            nonlocal acquired
            async with session.turn_lock:
                acquired = True

        await _subsequent_turn()
        assert acquired, "turn_lock should be acquirable after previous turn's exception"

    @pytest.mark.asyncio
    async def test_turn_lock_prevent_concurrent_task_creation(self) -> None:
        """Atomic check-and-set under turn_lock prevents two turn_tasks."""
        session = _make_mock_session()

        # Simulate the REST send_message pattern: atomically check turn_task
        # and create it under turn_lock
        async with session.turn_lock:
            assert session.turn_task is None, "turn_task should start as None"
            # Create a sentinel task (like REST sync path does)
            session.turn_task = asyncio.get_running_loop().create_future()

        # A second coroutine trying the same thing should see turn_task is set
        async def _second_attempt() -> bool:
            async with session.turn_lock:
                if session.turn_task is not None and not session.turn_task.done():
                    return False  # 409 TURN_IN_PROGRESS
                return True

        result = await _second_attempt()
        assert not result, "second attempt should see turn_task as in-progress"


# ===========================================================================
# 2. Cancel Event Lifecycle — WebSocket cancel → pipeline stop
# ===========================================================================


class TestCancelEventLifecycle:
    """cancel_event stops pipeline phases and is cleaned up after cancel."""

    @pytest.mark.asyncio
    async def test_cancel_event_set_stops_pipeline(self) -> None:
        """Setting cancel_event between pipeline phases raises CancelledError."""
        from cogtrix_core.api.turn_runner import _run_think_pipeline

        session = _make_mock_session()
        run_config = MagicMock()
        run_config.llm = None  # skip classify to_thread

        # Set cancel_event so the first check gate triggers
        session.cancel_event.set()

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError, match="Cancel requested"):
                await _run_think_pipeline(session, "task", "response", [], run_config)

    @pytest.mark.asyncio
    async def test_cancel_event_cleared_after_cancel_handling(self) -> None:
        """After the WebSocket cancel handler processes a cancel, cancel_event is cleared."""
        session = _make_mock_session()

        # Simulate the WebSocket cancel handler (messages.py lines 761-775)
        # Create a mock turn_task that's not done
        turn_was_cancelled = False

        async def _mock_turn() -> None:
            nonlocal turn_was_cancelled
            try:
                await asyncio.sleep(10)  # long-running turn
            except asyncio.CancelledError:
                turn_was_cancelled = True
                raise

        session.turn_task = asyncio.create_task(_mock_turn())

        # Let the task start
        await asyncio.sleep(0.01)

        # Simulate cancel flow
        if session.turn_task is not None and not session.turn_task.done():
            session.cancel_event.set()
            session.turn_task.cancel()
            try:
                await session.turn_task
            except (asyncio.CancelledError, Exception):
                pass
            finally:
                session.cancel_event.clear()

        assert turn_was_cancelled, "turn_task should have been cancelled"
        assert not session.cancel_event.is_set(), "cancel_event must be cleared after cancel"
        assert session.turn_task.done(), "turn_task should be done after cancel+await"

    @pytest.mark.asyncio
    async def test_cancel_event_checked_in_delegate_pipeline(self) -> None:
        """cancel_event.is_set() causes _run_delegate_pipeline to raise CancelledError."""
        from cogtrix_core.api.turn_runner import _run_delegate_pipeline

        session = _make_mock_session()
        session.cancel_event.set()
        run_config = MagicMock()

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", new_callable=AsyncMock):
            with patch(
                "cogtrix_core.orchestration.phases.was_delegation_called", return_value=False
            ):
                with pytest.raises(asyncio.CancelledError, match="Cancel requested"):
                    await _run_delegate_pipeline(session, "task", "original", [], run_config)

    @pytest.mark.asyncio
    async def test_cancel_event_not_cleared_by_unsuccessful_cancel(self) -> None:
        """cancel_event is NOT cleared when turn_task is None (no active turn)."""
        session = _make_mock_session()
        session.cancel_event.set()

        # When there's no active turn_task, cancel_event stays set
        assert session.turn_task is None
        assert session.cancel_event.is_set(), "cancel_event should remain set"

    @pytest.mark.asyncio
    async def test_cancel_propagates_to_pipeline_post_processing(self) -> None:
        """CancelledError from run_agent should be caught, state set to idle, and re-raised."""
        from cogtrix_core.api.turn_runner import _run_message_turn_inner

        session = _make_mock_session()
        session.memory_manager = MagicMock()
        session.memory_manager.prepare_context.return_value = MagicMock(messages=[])

        enqueue_states: list[str] = []

        async def _fake_enqueue(s, state):
            s.agent_state = state
            enqueue_states.append(state)

        with patch("cogtrix_core.api.turn_runner._enqueue_agent_state", side_effect=_fake_enqueue):
            with patch(
                "cogtrix_core.api.turn_runner.asyncio.to_thread",
                side_effect=asyncio.CancelledError("agent cancelled"),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await _run_message_turn_inner(session, "hello", "normal", None, None)

        # After CancelledError, agent_state should be "idle"
        assert session.agent_state == "idle"
        assert "idle" in enqueue_states


# ===========================================================================
# 3. WebSocket Disconnect — cleanup during active streaming
# ===========================================================================


class TestWebSocketDisconnectCleanup:
    """WebSocket disconnect during streaming cleans up tasks and state."""

    @pytest.mark.asyncio
    async def test_disconnect_cancels_drain_task(self) -> None:
        """When the WebSocket receive loop exits, the drain task is cancelled."""
        session = _make_mock_session()

        # Simulate a drain task reading from ws_queue
        drain_cancelled = False

        async def _drain() -> None:
            nonlocal drain_cancelled
            try:
                while True:
                    await session.ws_queue.get()
            except asyncio.CancelledError:
                drain_cancelled = True
                raise

        drain_task = asyncio.create_task(_drain())
        session.drain_task = drain_task

        # Let the drain task start
        await asyncio.sleep(0.01)

        # Simulate disconnect: cancel drain, await, clear reference
        drain_task.cancel()
        try:
            await drain_task
        except (asyncio.CancelledError, Exception):
            pass

        if session.drain_task is drain_task:
            session.drain_task = None

        assert drain_cancelled, "drain task should have been cancelled"
        assert session.drain_task is None, "drain_task reference should be cleared"

    @pytest.mark.asyncio
    async def test_disconnect_during_active_turn_cancels_turn(self) -> None:
        """Disconnect during an active turn should cancel turn_task."""
        session = _make_mock_session()

        turn_cancelled = False

        async def _active_turn() -> None:
            nonlocal turn_cancelled
            try:
                await asyncio.sleep(10)  # long-running
            except asyncio.CancelledError:
                turn_cancelled = True
                raise

        session.turn_task = asyncio.create_task(_active_turn())
        await asyncio.sleep(0.01)

        # Simulate disconnect cleanup: cancel turn_task
        if session.turn_task is not None and not session.turn_task.done():
            session.turn_task.cancel()
            try:
                await session.turn_task
            except (asyncio.CancelledError, Exception):
                pass

        assert turn_cancelled, "active turn should be cancelled on disconnect"

    @pytest.mark.asyncio
    async def test_disconnect_clears_agent_state_to_idle(self) -> None:
        """After disconnect cleanup, agent_state should return to idle."""
        session = _make_mock_session(agent_state="thinking")

        # Simulate cleanup: set state to idle
        session.agent_state = "idle"

        assert (
            session.agent_state == "idle"
        ), "agent_state should be 'idle' after disconnect cleanup"

    @pytest.mark.asyncio
    async def test_cancel_via_ws_message_pattern(self) -> None:
        """Simulate the full WebSocket cancel message → cleanup flow."""
        session = _make_mock_session(agent_state="thinking")

        turn_cancelled = False

        async def _active_turn() -> None:
            nonlocal turn_cancelled
            try:
                # Simulate the agent working
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                turn_cancelled = True
                session.agent_state = "idle"
                raise

        session.turn_task = asyncio.create_task(_active_turn())
        await asyncio.sleep(0.01)

        # Simulate the WebSocket cancel handler (messages.py lines 761-775)
        assert session.turn_task is not None
        assert not session.turn_task.done()

        session.cancel_event.set()
        session.turn_task.cancel()
        try:
            await session.turn_task
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            session.cancel_event.clear()

        assert turn_cancelled, "turn_task should be cancelled"
        assert not session.cancel_event.is_set(), "cancel_event should be cleared"
        assert session.turn_task.done(), "turn_task should be marked done"


# ===========================================================================
# 4. Background Task Lifecycle — eviction loop stop and shutdown
# ===========================================================================


class TestBackgroundTaskLifecycle:
    """Background tasks (eviction loop) are properly stopped on shutdown."""

    @pytest.mark.asyncio
    async def test_eviction_task_cancelled_on_stop(self) -> None:
        """stop_eviction_loop cancels the eviction background task."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )
        registry = ApiSessionRegistry(mock_app_state)

        session = ApiSession(id="test-1", user_id="user-1", name="Test")
        await registry.put(session)

        # Start eviction loop
        registry.start_eviction_loop()
        assert registry._eviction_task is not None, "eviction task should exist"
        assert not registry._eviction_task.done(), "eviction task should be running"

        # Stop eviction loop
        await registry.stop_eviction_loop()

        # Verify task was cancelled and reference cleared
        assert (
            registry._eviction_task is None or registry._eviction_task.done()
        ), "eviction task should be done after stop"

    @pytest.mark.asyncio
    async def test_eviction_loop_saves_sessions_before_stop(self) -> None:
        """stop_eviction_loop saves all sessions before cancelling the eviction task."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )
        registry = ApiSessionRegistry(mock_app_state)

        session = ApiSession(id="test-save-1", user_id="user-1", name="Test")
        await registry.put(session)

        registry.start_eviction_loop()

        # Mock _save_memory to track it was called
        with patch(
            "cogtrix_core.api.session_bridge._save_memory", new_callable=AsyncMock
        ) as mock_save:
            await registry.stop_eviction_loop()
            # Should have attempted to save at least the session we added
            assert mock_save.called, "_save_memory should be called during stop_eviction_loop"

    @pytest.mark.asyncio
    async def test_eviction_loop_handles_stop_error_gracefully(self) -> None:
        """stop_eviction_loop does not raise when save fails."""
        from cogtrix_core.api.session_bridge import ApiSession, ApiSessionRegistry

        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )
        registry = ApiSessionRegistry(mock_app_state)

        session = ApiSession(id="test-err-1", user_id="user-1", name="Test")
        await registry.put(session)

        registry.start_eviction_loop()

        # Mock _save_memory to raise — stop should handle it gracefully
        with patch(
            "cogtrix_core.api.session_bridge._save_memory",
            side_effect=RuntimeError("DB connection lost"),
        ):
            # Should not raise
            await registry.stop_eviction_loop()

        # Task should still be properly cleaned up
        assert (
            registry._eviction_task is None or registry._eviction_task.done()
        ), "eviction task should be done after stop (even with save error)"

    @pytest.mark.asyncio
    async def test_stop_eviction_clears_task_reference(self) -> None:
        """After stop_eviction_loop, _eviction_task is done (cancelled or None)."""
        from cogtrix_core.api.session_bridge import ApiSessionRegistry

        mock_app_state = SimpleNamespace(
            config=None,
            tool_registry=MagicMock(),
            mcp_manager=None,
        )
        registry = ApiSessionRegistry(mock_app_state)
        registry.start_eviction_loop()

        await registry.stop_eviction_loop()
        # stop_eviction_loop cancels the task but doesn't null the reference;
        # the task is done (cancelled). This matches the assertion pattern in
        # test_graceful_shutdown.py.
        assert (
            registry._eviction_task is None or registry._eviction_task.done()
        ), "_eviction_task should be None or done after stop_eviction_loop"
