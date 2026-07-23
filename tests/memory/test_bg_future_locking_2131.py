"""Regression tests for #2131 C1/C2 — `_bg_future` / `_summary_dirty` must be
accessed under `_hybrid_lock` in `save()` and `shutdown()`.

The module lock contract (manager.py docstring) declares `_bg_future` and the
summary fields `_hybrid_lock`-protected. Previously `save()` read them unlocked
(so a background summarizer completing between turns could interleave into a
state where neither flush branch ran → lost rolling summary) and `shutdown()`
read+nulled `_bg_future` unlocked (racing a concurrent `_schedule_slow_path`).

These tests pin the lock-acquisition invariant (a counting RLock) plus the
behavioural branching, deterministically.
"""

from __future__ import annotations

import threading

from src.memory import modes  # noqa: F401 — triggers mode registration
from src.memory.modes.conversation import ConversationMemoryManager


class _MockStore:
    def __init__(self) -> None:
        self.data: dict = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


class _CountingRLock:
    """RLock wrapper that counts acquisitions (via both context-manager and
    explicit acquire) so a test can assert a critical section ran under it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.acquires = 0

    def __enter__(self):
        self.acquires += 1
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False

    def acquire(self, *a, **k):
        self.acquires += 1
        return self._lock.acquire(*a, **k)

    def release(self):
        return self._lock.release()


class _FakeFuture:
    def __init__(self, *, running: bool, done: bool) -> None:
        self._running = running
        self._done = done
        self.cancelled = False

    def running(self) -> bool:
        return self._running

    def done(self) -> bool:
        return self._done

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def _make_manager() -> ConversationMemoryManager:
    mgr = ConversationMemoryManager(_MockStore(), "test-2131")
    # Stub all persistence I/O so save()/shutdown() do no disk work.
    mgr._save_hybrid_meta = lambda *a, **k: None  # type: ignore[method-assign]
    mgr._save_mode_meta = lambda *a, **k: None  # type: ignore[method-assign]
    mgr._save_tier_cache = lambda *a, **k: None  # type: ignore[method-assign]
    mgr.save_messages_only = lambda *a, **k: None  # type: ignore[method-assign]
    mgr._vector_store = None
    return mgr


# ── C1: save() ───────────────────────────────────────────────────────


def test_save_reads_bg_future_under_lock() -> None:
    """Even on the fut-inactive path (where the old code took NO lock), save()
    must acquire _hybrid_lock to snapshot _bg_future/_summary_dirty."""
    mgr = _make_manager()
    lock = _CountingRLock()
    mgr._hybrid_lock = lock  # type: ignore[assignment]
    mgr._bg_future = None
    before = lock.acquires
    mgr.save()
    assert lock.acquires > before, "save() must snapshot _bg_future under _hybrid_lock"


def test_save_flushes_and_clears_dirty_when_bg_active_and_dirty() -> None:
    mgr = _make_manager()
    calls: list[str] = []
    mgr._save_hybrid_meta = lambda *a, **k: calls.append("hybrid")  # type: ignore[method-assign]
    mgr._bg_future = _FakeFuture(running=True, done=False)
    mgr._summary_dirty = True
    mgr.save()
    assert calls == ["hybrid"], "dirty summary must be flushed while bg is active"
    assert mgr._summary_dirty is False


def test_save_skips_flush_when_bg_active_and_not_dirty() -> None:
    mgr = _make_manager()
    calls: list[str] = []
    mgr._save_hybrid_meta = lambda *a, **k: calls.append("hybrid")  # type: ignore[method-assign]
    mgr._bg_future = _FakeFuture(running=True, done=False)
    mgr._summary_dirty = False
    mgr.save()
    assert calls == [], "no flush when a bg job is active and the summary is clean"


def test_save_flushes_when_bg_done() -> None:
    mgr = _make_manager()
    calls: list[str] = []
    mgr._save_hybrid_meta = lambda *a, **k: calls.append("hybrid")  # type: ignore[method-assign]
    mgr._bg_future = _FakeFuture(running=False, done=True)
    mgr._summary_dirty = False
    mgr.save()
    assert calls == ["hybrid"], "completed bg job → flush hybrid meta"


# ── C2: shutdown() ───────────────────────────────────────────────────


def test_shutdown_reads_and_nulls_bg_future_under_lock() -> None:
    mgr = _make_manager()
    lock = _CountingRLock()
    mgr._hybrid_lock = lock  # type: ignore[assignment]
    mgr._bg_future = None
    before = lock.acquires
    mgr.shutdown()
    assert lock.acquires > before, "shutdown() must read+null _bg_future under _hybrid_lock"
    assert mgr._bg_future is None


def test_shutdown_cancels_running_future_and_nulls_it() -> None:
    mgr = _make_manager()
    fut = _FakeFuture(running=True, done=False)
    mgr._bg_future = fut
    mgr.shutdown()
    assert fut.cancelled, "a running background job must be cancelled at shutdown"
    assert mgr._bg_future is None
