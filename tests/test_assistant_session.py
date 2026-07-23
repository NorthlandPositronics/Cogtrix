"""Unit tests for src/assistant/session.py — ChatSession and ChatSessionManager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
from src.assistant.session import ChatSession, ChatSessionManager
from src.memory.context import MemoryContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_mock_memory_manager() -> MagicMock:
    mm = MagicMock()
    mm.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=None,
    )
    return mm


def _make_manager(
    max_sessions: int = 50,
    idle_timeout: float = 3600.0,
) -> ChatSessionManager:
    """Create a ChatSessionManager with no external dependencies."""
    config = MagicMock()
    config.services = {"assistant": {}}
    llm = MagicMock()
    registry = MagicMock()

    return ChatSessionManager(
        config=config,
        llm=llm,
        system_prompt="sys",
        registry=registry,
        max_sessions=max_sessions,
        idle_timeout=idle_timeout,
    )


def _fake_create_factory(mock_mm: MagicMock):
    """Return a side_effect function for _create_session that builds real ChatSessions."""

    def _fake_create(self_or_msg, msg: IncomingMessage | None = None) -> ChatSession:
        # patch.object with side_effect passes the bound instance as first arg
        # when the method is an instance method; compensate for that.
        if msg is None:
            actual_msg = self_or_msg
        else:
            actual_msg = msg
        return ChatSession(
            session_key=actual_msg.session_key,
            channel=actual_msg.channel,
            chat_id=actual_msg.chat_id,
            memory_manager=mock_mm,
        )

    return _fake_create


# ---------------------------------------------------------------------------
# TestChatSession
# ---------------------------------------------------------------------------


class TestChatSession:
    """Tests for ChatSession dataclass."""

    def test_fields_stored(self):
        """ChatSession stores all provided fields correctly."""
        mm = _make_mock_memory_manager()
        session = ChatSession(
            session_key="telegram::42",
            channel="telegram",
            chat_id="42",
            memory_manager=mm,
        )
        assert session.session_key == "telegram::42"
        assert session.channel == "telegram"
        assert session.chat_id == "42"
        assert session.memory_manager is mm

    def test_lock_is_threading_lock(self):
        """Each ChatSession gets its own threading.Lock."""
        mm = _make_mock_memory_manager()
        s1 = ChatSession(session_key="a::1", channel="a", chat_id="1", memory_manager=mm)
        s2 = ChatSession(session_key="b::2", channel="b", chat_id="2", memory_manager=mm)
        assert isinstance(s1.lock, type(threading.Lock()))
        assert isinstance(s2.lock, type(threading.Lock()))
        assert s1.lock is not s2.lock

    def test_last_activity_defaults_to_recent(self):
        """last_activity is set close to the current monotonic time."""
        before = time.monotonic()
        mm = _make_mock_memory_manager()
        session = ChatSession(session_key="a::1", channel="a", chat_id="1", memory_manager=mm)
        after = time.monotonic()
        assert before <= session.last_activity <= after


# ---------------------------------------------------------------------------
# TestChatSessionManager
# ---------------------------------------------------------------------------


class TestChatSessionManager:
    """Tests for ChatSessionManager."""

    def _patched_create(self, mgr: ChatSessionManager, mock_mm: MagicMock):
        """Patch _create_session on the manager instance to avoid real file/memory I/O."""

        def _fake(msg: IncomingMessage) -> ChatSession:
            return ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=mock_mm,
            )

        return patch.object(mgr, "_create_session", side_effect=_fake)

    def test_get_or_create_new_session(self):
        """get_or_create() creates a new session for an unknown message."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager()
        msg = _make_msg(channel="telegram", chat_id="100")

        with self._patched_create(mgr, mock_mm):
            session = mgr.get_or_create(msg)

        assert session.session_key == "telegram::100"
        assert session.channel == "telegram"
        assert session.chat_id == "100"

    def test_get_or_create_same_key_returns_same_session(self):
        """get_or_create() returns the same session object for the same session_key."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager()
        msg = _make_msg(channel="telegram", chat_id="200")

        with self._patched_create(mgr, mock_mm):
            s1 = mgr.get_or_create(msg)
            s2 = mgr.get_or_create(msg)

        assert s1 is s2

    def test_different_keys_produce_different_sessions(self):
        """Different (channel, chat_id) pairs produce independent sessions."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager()
        msg_a = _make_msg(channel="telegram", chat_id="1")
        msg_b = _make_msg(channel="whatsapp", chat_id="1")

        with self._patched_create(mgr, mock_mm):
            sa = mgr.get_or_create(msg_a)
            sb = mgr.get_or_create(msg_b)

        assert sa is not sb
        assert sa.session_key != sb.session_key

    def test_sessions_have_independent_locks(self):
        """Each session gets its own lock, not a shared one."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager()
        msg_a = _make_msg(channel="telegram", chat_id="1")
        msg_b = _make_msg(channel="telegram", chat_id="2")

        with self._patched_create(mgr, mock_mm):
            sa = mgr.get_or_create(msg_a)
            sb = mgr.get_or_create(msg_b)

        assert sa.lock is not sb.lock

    def test_evict_idle_removes_timed_out_sessions(self):
        """evict_idle() removes sessions whose last_activity exceeds idle_timeout."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(idle_timeout=0.0)
        msg = _make_msg(chat_id="999")

        with self._patched_create(mgr, mock_mm):
            session = mgr.get_or_create(msg)

        # Force the session to appear stale
        session.last_activity = time.monotonic() - 9999

        evicted = mgr.evict_idle()
        assert evicted == 1
        assert msg.session_key not in mgr._sessions

    def test_evict_idle_calls_save_on_evicted_sessions(self):
        """evict_idle() calls memory_manager.save() on each evicted session."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(idle_timeout=0.0)
        msg = _make_msg(chat_id="evict-me")

        with self._patched_create(mgr, mock_mm):
            session = mgr.get_or_create(msg)

        session.last_activity = time.monotonic() - 9999
        mgr.evict_idle()

        mock_mm.save.assert_called()

    def test_evict_idle_leaves_active_sessions(self):
        """evict_idle() does not remove sessions that are still active."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(idle_timeout=3600.0)
        msg = _make_msg(chat_id="active")

        with self._patched_create(mgr, mock_mm):
            mgr.get_or_create(msg)

        evicted = mgr.evict_idle()
        assert evicted == 0
        assert msg.session_key in mgr._sessions

    def test_save_all_calls_save_on_each_session(self):
        """save_all() persists every active session's memory manager."""
        mm_a = _make_mock_memory_manager()
        mm_b = _make_mock_memory_manager()
        managers = [mm_a, mm_b]
        call_idx = [0]

        mgr = _make_manager()

        def _fake(msg: IncomingMessage) -> ChatSession:
            idx = call_idx[0]
            call_idx[0] += 1
            return ChatSession(
                session_key=msg.session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                memory_manager=managers[idx],
            )

        with patch.object(mgr, "_create_session", side_effect=_fake):
            mgr.get_or_create(_make_msg(chat_id="a"))
            mgr.get_or_create(_make_msg(chat_id="b"))

        mgr.save_all()

        mm_a.save.assert_called_once()
        mm_b.save.assert_called_once()

    def test_session_cap_evicts_oldest(self):
        """When max_sessions is reached, the oldest session is evicted."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(max_sessions=2)

        msg_a = _make_msg(chat_id="oldest")
        msg_b = _make_msg(chat_id="middle")
        msg_c = _make_msg(chat_id="newest")

        with self._patched_create(mgr, mock_mm):
            sa = mgr.get_or_create(msg_a)
            # Make sa clearly the oldest
            sa.last_activity = time.monotonic() - 1000
            mgr.get_or_create(msg_b)
            # Adding a third session must evict the oldest
            mgr.get_or_create(msg_c)

        assert msg_a.session_key not in mgr._sessions
        assert len(mgr._sessions) == 2

    # BUG-106 regression tests -----------------------------------------------

    def test_evicted_lock_released_if_create_session_raises(self):
        """BUG-106: evicted session locks must be released even when _create_session raises."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(max_sessions=1)

        # Create one session to fill the cap
        msg_a = _make_msg(chat_id="a")
        msg_b = _make_msg(chat_id="b")

        # First call succeeds normally
        with self._patched_create(mgr, mock_mm):
            sa = mgr.get_or_create(msg_a)
            sa.last_activity = time.monotonic() - 1000  # make it the oldest

        # Second call with a failing _create_session: eviction should have already
        # released sa.lock so it can be acquired here
        def _failing_create(msg: IncomingMessage) -> ChatSession:
            raise RuntimeError("simulated creation failure")

        with patch.object(mgr, "_create_session", side_effect=_failing_create):
            try:
                mgr.get_or_create(msg_b)
            except RuntimeError:
                pass  # expected

        # The evicted session's lock must be released (non-blocking acquire succeeds)
        acquired = sa.lock.acquire(blocking=False)
        assert acquired, "evicted session lock was not released after _create_session raised"
        sa.lock.release()

    def test_evicted_session_save_called_if_create_session_raises(self):
        """BUG-106: memory_manager.save() must be called for evicted sessions even on error."""
        mock_mm = _make_mock_memory_manager()
        mgr = _make_manager(max_sessions=1)

        msg_a = _make_msg(chat_id="a2")
        msg_b = _make_msg(chat_id="b2")

        with self._patched_create(mgr, mock_mm):
            sa = mgr.get_or_create(msg_a)
            sa.last_activity = time.monotonic() - 1000

        def _failing_create(msg: IncomingMessage) -> ChatSession:
            raise RuntimeError("simulated creation failure")

        with patch.object(mgr, "_create_session", side_effect=_failing_create):
            try:
                mgr.get_or_create(msg_b)
            except RuntimeError:
                pass

        # save() must have been called on the evicted session's memory manager
        sa.memory_manager.save.assert_called_once()
