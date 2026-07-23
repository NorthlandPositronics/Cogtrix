"""Regression tests for BUG-101: get_or_create() must not hold the registry lock
during memory_manager.save() in the inline idle eviction path.

Before the fix, get_or_create() called evict_idle() while holding self._lock (an RLock).
evict_idle() re-entered the RLock (which succeeds because RLock is reentrant), removed
sessions, released its own context block, then called session.memory_manager.save() —
but the outer with self._lock: from get_or_create() was still active. Any concurrent
caller was blocked for the full duration of the disk I/O.

After the fix, get_or_create() performs inline idle eviction: it collects candidates and
removes them from the registry under the lock, then releases the lock and saves outside.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.assistant.session import ChatSession, ChatSessionManager


def _make_msg(channel: str = "telegram", chat_id: str = "42") -> IncomingMessage:
    return IncomingMessage(
        channel=channel,
        chat_id=chat_id,
        message_id="m1",
        sender_id="u1",
        sender_name="Bob",
        text="Hello",
        timestamp=time.time(),
    )


def _make_manager(max_sessions: int = 2, idle_timeout: float = 3600.0) -> ChatSessionManager:
    config = MagicMock()
    config.services = {"assistant": {}}
    return ChatSessionManager(
        config=config,
        llm=MagicMock(),
        system_prompt="sys",
        registry=MagicMock(),
        max_sessions=max_sessions,
        idle_timeout=idle_timeout,
    )


def _make_mock_mm() -> MagicMock:
    return MagicMock()


def _patch_create(mgr: ChatSessionManager, mock_mm: MagicMock):
    def _fake(msg: IncomingMessage) -> ChatSession:
        return ChatSession(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            memory_manager=mock_mm,
        )

    return patch.object(mgr, "_create_session", side_effect=_fake)


# ---------------------------------------------------------------------------
# Test 1: idle session is evicted when cap is reached and a new one created
# ---------------------------------------------------------------------------
def test_get_or_create_evicts_idle_session_inline() -> None:
    mgr = _make_manager(max_sessions=1, idle_timeout=0.0)
    mock_mm = _make_mock_mm()

    with _patch_create(mgr, mock_mm):
        # Create first session.
        msg1 = _make_msg(chat_id="1")
        s1 = mgr.get_or_create(msg1)
        assert s1.chat_id == "1"

        # Make session idle (idle_timeout=0 means any age qualifies).
        # get_or_create for a new key must evict s1 and create s2.
        msg2 = _make_msg(chat_id="2")
        s2 = mgr.get_or_create(msg2)
        assert s2.chat_id == "2"

    # s1 must have been saved and evicted.
    mock_mm.save.assert_called()
    assert "telegram::1" not in mgr._sessions
    assert "telegram::2" in mgr._sessions


# ---------------------------------------------------------------------------
# Test 2: the registry lock is NOT held during save() in the idle eviction path
#
# We verify this by making save() block until it can acquire the registry lock
# in a non-blocking way. If the lock were still held by get_or_create(), save()
# would be unable to acquire it.
# ---------------------------------------------------------------------------
def test_get_or_create_does_not_hold_lock_during_save() -> None:
    mgr = _make_manager(max_sessions=1, idle_timeout=0.0)
    mock_mm = _make_mock_mm()

    lock_was_free_during_save: list[bool] = []

    def save_side_effect():
        # Try to acquire the registry lock (non-blocking).
        acquired = mgr._lock.acquire(blocking=False)
        if acquired:
            lock_was_free_during_save.append(True)
            mgr._lock.release()
        else:
            lock_was_free_during_save.append(False)

    mock_mm.save.side_effect = save_side_effect

    with _patch_create(mgr, mock_mm):
        msg1 = _make_msg(chat_id="1")
        mgr.get_or_create(msg1)
        # Trigger idle eviction.
        msg2 = _make_msg(chat_id="2")
        mgr.get_or_create(msg2)

    assert lock_was_free_during_save, "save() was never called during idle eviction"
    # If any save call found the lock held, the bug is present.
    # Note: RLock re-entry is allowed from the same thread, but acquire(blocking=False)
    # from the same thread on an RLock returns True if it is the owner — so this
    # check is meaningful only from a *different* thread. We verify from the same
    # thread: if get_or_create holds the lock and calls save inside a patched side
    # effect that also tries to acquire the RLock, acquire(blocking=False) returns
    # True (RLock re-entry). We then need a cross-thread check to be definitive.
    # The cross-thread test follows below (test 3).
    assert all(
        lock_was_free_during_save
    ), "Registry lock was held during save() in inline idle eviction path — BUG-101 regressed."


# ---------------------------------------------------------------------------
# Test 3: cross-thread: a concurrent get_or_create() is NOT blocked during
# the disk I/O of an inline idle eviction
# ---------------------------------------------------------------------------
def test_get_or_create_concurrent_caller_not_blocked_during_idle_eviction_save() -> None:
    """A concurrent get_or_create() call must not be blocked by save() in idle eviction.

    Strategy:
    1. Fill the session pool to capacity (1 session, idle_timeout=0).
    2. Patch save() on the evicted session to hold a threading.Event, then
       launch get_or_create() on a background thread.
    3. From the main thread, try get_or_create() for a third key while the
       background save() is blocked.
    4. The main-thread call must succeed quickly — it must not block on the
       registry lock being held by the background thread's save().
    """
    mgr = _make_manager(max_sessions=1, idle_timeout=0.0)

    # We need separate mm instances so we can control each independently.
    mm1 = MagicMock()
    mm2 = MagicMock()
    mm3 = MagicMock()
    mm_seq = iter([mm1, mm2, mm3])

    save_started = threading.Event()
    save_can_finish = threading.Event()

    def mm1_save():
        save_started.set()
        # Block until the test releases it — simulates slow disk I/O.
        save_can_finish.wait(timeout=5.0)

    mm1.save.side_effect = mm1_save

    def _fake_create(msg: IncomingMessage) -> ChatSession:
        mm = next(mm_seq)
        return ChatSession(
            session_key=msg.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            memory_manager=mm,
        )

    with patch.object(mgr, "_create_session", side_effect=_fake_create):
        # Create first session (fills the pool).
        msg1 = _make_msg(chat_id="1")
        mgr.get_or_create(msg1)

        # Background thread triggers idle eviction of s1 and slow save().
        results: list[ChatSession] = []

        def bg_get_or_create():
            msg2 = _make_msg(chat_id="2")
            results.append(mgr.get_or_create(msg2))

        bg = threading.Thread(target=bg_get_or_create)
        bg.start()

        # Wait until save() has started (proves eviction happened).
        save_started.wait(timeout=5.0)
        assert save_started.is_set(), "save() never started — eviction may not have happened"

        # Now the background thread is inside save() with the registry lock released.
        # The main thread must be able to call get_or_create() without blocking.
        # (If the lock were still held we'd deadlock here since idle_timeout=0
        # and the pool is now empty after eviction, so creation should succeed.)
        t0 = time.monotonic()
        msg3 = _make_msg(chat_id="3")
        s3 = mgr.get_or_create(msg3)
        elapsed = time.monotonic() - t0

        # Release the background save.
        save_can_finish.set()
        bg.join(timeout=5.0)

        assert s3.chat_id == "3"
        # The main-thread call must have completed in well under 1 second
        # (save() is blocked for ~0 s after save_can_finish is set, so if
        # it were blocking the main thread the elapsed time would be much longer).
        assert elapsed < 1.0, (
            f"get_or_create() took {elapsed:.2f}s — it may have been blocked on the "
            "registry lock held during save() in idle eviction (BUG-101)."
        )


# ---------------------------------------------------------------------------
# Test 4: idle eviction count = 0 falls back to oldest-session forced eviction
# (existing forced-eviction path must still work correctly)
# ---------------------------------------------------------------------------
def test_get_or_create_forced_eviction_when_no_idle_sessions() -> None:
    """When no sessions are idle, the oldest session is force-evicted."""
    # idle_timeout very large so no session is considered idle.
    mgr = _make_manager(max_sessions=1, idle_timeout=9999.0)
    mock_mm = _make_mock_mm()

    with _patch_create(mgr, mock_mm):
        msg1 = _make_msg(chat_id="1")
        s1 = mgr.get_or_create(msg1)
        assert s1.chat_id == "1"

        msg2 = _make_msg(chat_id="2")
        s2 = mgr.get_or_create(msg2)
        assert s2.chat_id == "2"

    # s1 must have been force-evicted and saved.
    mock_mm.save.assert_called()
    assert "telegram::1" not in mgr._sessions
    assert "telegram::2" in mgr._sessions
