"""Regression tests for ChatSessionManager session cap behavior (Issue #1075).

BUG-1075: Soft session cap allows unbounded concurrent sessions when oldest
sessions are busy. The hard overflow cap prevents unbounded growth under
sustained concurrent load.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from cogtrix_core.assistant.channel import IncomingMessage
from cogtrix_core.assistant.session import (
    _MAX_SESSION_OVERFLOW_MULTIPLIER,
    ChatSession,
    ChatSessionManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(channel: str = "telegram", chat_id: str = "42") -> MagicMock:
    msg = MagicMock(spec=IncomingMessage)
    msg.channel = channel
    msg.chat_id = chat_id
    msg.session_key = f"{channel}::{chat_id}"
    return msg


def _make_manager(max_sessions: int = 5) -> ChatSessionManager:
    return ChatSessionManager(
        config=MagicMock(),
        llm=MagicMock(),
        system_prompt="sys",
        registry=MagicMock(),
        max_sessions=max_sessions,
        idle_timeout=3600.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionCapHardOverflow:
    """Hard overflow cap fires when soft eviction cannot free a slot."""

    def test_hard_cap_fires_at_overflow_threshold(self):
        """When all sessions are busy and count >= max_sessions * 1.5, RuntimeError is raised."""
        mgr = _make_manager(max_sessions=4)
        overflow_limit = int(4 * _MAX_SESSION_OVERFLOW_MULTIPLIER)  # 6

        # Fill sessions up to the overflow limit with busy locks
        for i in range(overflow_limit):
            msg = _make_msg(chat_id=f"chat{i}")
            # Create session directly in registry with a lock already held
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            session.lock.acquire()  # hold the lock — simulates busy session
            mgr._sessions[msg.session_key] = session

        # At overflow limit — next request must fail
        new_msg = _make_msg(chat_id="new_overflow")
        with pytest.raises(RuntimeError) as exc_info:
            mgr.get_or_create(new_msg)

        assert "Session cap exceeded" in str(exc_info.value)
        assert str(overflow_limit) in str(exc_info.value)

    def test_soft_cap_allows_growth_up_to_overflow_threshold(self):
        """Sessions can still be created between max_sessions and the overflow threshold."""
        mgr = _make_manager(max_sessions=4)

        # Fill exactly to max_sessions with busy sessions
        for i in range(4):
            msg = _make_msg(chat_id=f"chat{i}")
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            session.lock.acquire()  # hold lock
            mgr._sessions[msg.session_key] = session

        # Still below overflow threshold — soft cap allows creation (over max)
        msg_new = _make_msg(chat_id="burst_allowed")
        session = mgr.get_or_create(msg_new)
        assert session.session_key == "telegram::burst_allowed"
        assert len(mgr._sessions) == 5

    def test_normal_creation_below_max_sessions(self):
        """Below max_sessions, creation works with no eviction needed."""
        mgr = _make_manager(max_sessions=5)

        msg = _make_msg(chat_id="normal")
        session = mgr.get_or_create(msg)
        assert session.session_key == "telegram::normal"
        assert len(mgr._sessions) == 1

    def test_existing_session_returned_when_at_cap(self):
        """Existing session lookup is unaffected by cap enforcement."""
        mgr = _make_manager(max_sessions=2)
        overflow_limit = int(2 * _MAX_SESSION_OVERFLOW_MULTIPLIER)  # 3

        # Pre-populate at overflow threshold
        for i in range(overflow_limit):
            msg = _make_msg(chat_id=f"chat{i}")
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            session.lock.acquire()
            mgr._sessions[msg.session_key] = session

        # Lookup of existing session still works even when at cap
        existing_msg = _make_msg(chat_id="chat0")
        session = mgr.get_or_create(existing_msg)
        assert session.session_key == "telegram::chat0"

    def test_idle_eviction_clears_slots_before_hard_cap(self):
        """When idle sessions exist, eviction frees a slot before hard cap fires."""
        mgr = _make_manager(max_sessions=4)

        # Fill to exactly max_sessions with idle sessions (no locks held)
        for i in range(4):
            msg = _make_msg(chat_id=f"idle{i}")
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            # Simulate idle session: set last_activity far in the past
            session.last_activity = time.monotonic() - mgr._idle_timeout - 1
            mgr._sessions[msg.session_key] = session

        # At max_sessions, idle eviction should free a slot
        msg_new = _make_msg(chat_id="new_after_idle_eviction")
        session = mgr.get_or_create(msg_new)
        assert session.session_key == "telegram::new_after_idle_eviction"
        # All 4 idle sessions were evicted, new one created — only new session remains
        assert len(mgr._sessions) == 1

    def test_non_busy_oldest_eviction_frees_slot_before_hard_cap(self):
        """When oldest session is not busy, it is evicted and a new slot opens."""
        mgr = _make_manager(max_sessions=4)

        # Fill to max_sessions, oldest has no lock held (can be evicted)
        for i in range(4):
            msg = _make_msg(chat_id=f"chat{i}")
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            mgr._sessions[msg.session_key] = session
        # chat0 is oldest (created first) — lock not held

        msg_new = _make_msg(chat_id="new_after_oldest_eviction")
        session = mgr.get_or_create(msg_new)
        assert session.session_key == "telegram::new_after_oldest_eviction"
        assert "telegram::chat0" not in mgr._sessions  # evicted

    def test_hard_cap_threshold_is_50_percent_over_max(self):
        """Default overflow multiplier is 1.5, meaning hard cap fires at 150% of max."""
        mgr = _make_manager(max_sessions=10)
        hard_cap = int(10 * _MAX_SESSION_OVERFLOW_MULTIPLIER)  # 15

        # Fill to one below the hard cap
        for i in range(hard_cap - 1):
            msg = _make_msg(chat_id=f"chat{i}")
            session = ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=MagicMock(),
            )
            session.lock.acquire()
            mgr._sessions[msg.session_key] = session

        # One more should still be allowed (at threshold, not over)
        msg_new = _make_msg(chat_id="at_threshold")
        session = mgr.get_or_create(msg_new)
        assert session.session_key == "telegram::at_threshold"
        assert len(mgr._sessions) == hard_cap

        # One more — now over threshold — must raise
        msg_over = _make_msg(chat_id="over_threshold")
        with pytest.raises(RuntimeError) as exc_info:
            mgr.get_or_create(msg_over)
        assert "Session cap exceeded" in str(exc_info.value)
