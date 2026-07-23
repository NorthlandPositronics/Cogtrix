"""Tests for ARCH-040-01 + BUG-088: save_all() must not deadlock with handle().

The bug: save_all() used to hold the registry lock while acquiring per-session
locks. handle() holds a per-session lock while calling get_or_create(), which
acquires the registry lock. This creates a lock-order inversion deadlock.

The fix: save_all() snapshots the session dict under the registry lock, releases
it, then acquires per-session locks individually (matching evict_idle() pattern).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.assistant.session import ChatSession, ChatSessionManager
from src.memory.context import MemoryContext


def _make_msg(chat_id: str) -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id=chat_id,
        message_id="m1",
        sender_id="u1",
        sender_name="Test",
        text="hello",
        timestamp=time.time(),
    )


def _make_manager() -> ChatSessionManager:
    config = MagicMock()
    config.services = {"assistant": {}}
    return ChatSessionManager(
        config=config,
        llm=MagicMock(),
        system_prompt="sys",
        registry=MagicMock(),
        max_sessions=50,
        idle_timeout=3600.0,
    )


def _make_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.prepare_context.return_value = MemoryContext(messages=[], context_prefix=None)
    return mm


def _patched_create(mgr: ChatSessionManager, mm: MagicMock):
    def _fake(msg: IncomingMessage) -> ChatSession:
        return ChatSession(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            memory_manager=mm,
        )

    return patch.object(mgr, "_create_session", side_effect=_fake)


class TestSaveAllNoDeadlock:
    """ARCH-040-01: save_all() must not deadlock when called concurrently with get_or_create()."""

    def test_save_all_completes_without_deadlock(self):
        """save_all() must return within 5 seconds even when other threads call get_or_create()."""
        mm = _make_mock_mm()
        mgr = _make_manager()

        with _patched_create(mgr, mm):
            # Pre-create a session so save_all() has something to save
            mgr.get_or_create(_make_msg("chat1"))

        completed = threading.Event()
        errors: list[Exception] = []

        def _run_save_all() -> None:
            try:
                mgr.save_all()
                completed.set()
            except Exception as exc:
                errors.append(exc)
                completed.set()

        # Run save_all() in a background thread
        t = threading.Thread(target=_run_save_all, daemon=True)
        t.start()

        # save_all() must complete within 5 seconds (no deadlock)
        finished = completed.wait(timeout=5.0)
        assert finished, "save_all() deadlocked — did not complete within 5 seconds"
        assert not errors, f"save_all() raised: {errors}"

    def test_save_all_concurrent_with_get_or_create(self):
        """save_all() and get_or_create() called concurrently must not deadlock."""
        mm = _make_mock_mm()
        mgr = _make_manager()

        with _patched_create(mgr, mm):
            mgr.get_or_create(_make_msg("chat1"))

        barrier = threading.Barrier(2, timeout=5.0)
        completed: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _run_save_all() -> None:
            try:
                barrier.wait()
                mgr.save_all()
                with lock:
                    completed.append("save_all")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def _run_get_or_create() -> None:
            try:
                barrier.wait()
                with _patched_create(mgr, mm):
                    mgr.get_or_create(_make_msg("chat2"))
                with lock:
                    completed.append("get_or_create")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        t1 = threading.Thread(target=_run_save_all, daemon=True)
        t2 = threading.Thread(target=_run_get_or_create, daemon=True)
        t1.start()
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive(), "save_all() thread is still alive — possible deadlock"
        assert not t2.is_alive(), "get_or_create() thread is still alive — possible deadlock"
        assert not errors, f"Thread raised: {errors}"
        assert set(completed) == {"save_all", "get_or_create"}

    def test_save_all_calls_memory_save_on_each_session(self):
        """save_all() must call memory_manager.save() for every active session."""
        mm1 = _make_mock_mm()
        mm2 = _make_mock_mm()
        mms = [mm1, mm2]
        idx = [0]

        mgr = _make_manager()

        def _fake(msg: IncomingMessage) -> ChatSession:
            m = mms[idx[0]]
            idx[0] += 1
            return ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=m,
            )

        with patch.object(mgr, "_create_session", side_effect=_fake):
            mgr.get_or_create(_make_msg("a"))
            mgr.get_or_create(_make_msg("b"))

        mgr.save_all()

        mm1.save.assert_called_once()
        mm2.save.assert_called_once()

    def test_save_all_skips_locked_sessions_without_deadlock(self):
        """When a session lock is held by another thread, save_all() skips it gracefully."""
        mm = _make_mock_mm()
        mgr = _make_manager()

        with _patched_create(mgr, mm):
            session = mgr.get_or_create(_make_msg("busy_chat"))

        # Hold the session lock to simulate an active handler
        ready = threading.Event()
        release = threading.Event()

        def _hold_lock() -> None:
            with session.lock:
                ready.set()
                release.wait(timeout=3.0)

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        ready.wait(timeout=2.0)

        # save_all() must not block indefinitely
        completed = threading.Event()

        def _save() -> None:
            mgr.save_all()
            completed.set()

        saver = threading.Thread(target=_save, daemon=True)
        saver.start()

        finished = completed.wait(timeout=15.0)  # 10s timeout + headroom
        release.set()
        holder.join(timeout=2.0)

        assert finished, "save_all() blocked waiting for a locked session"
