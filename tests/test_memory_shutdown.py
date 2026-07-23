"""Regression tests for graceful memory shutdown handling."""

from __future__ import annotations

import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from cogtrix_core.memory.context import MemoryContext
from cogtrix_core.memory.json_store import JsonFileMemoryStore
from cogtrix_core.memory.manager import BaseMemoryManager
from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager
from cogtrix_core.memory.modes.reasoning import Goal, ReasoningMemoryManager


class _DummyMemoryManager(BaseMemoryManager):
    @property
    def mode_name(self) -> str:
        return "dummy"

    def prepare_context(self, user_input: str) -> MemoryContext:  # noqa: ARG002
        return MemoryContext()

    def update(
        self,
        user_input: str,  # noqa: ARG002
        ai_response: str,  # noqa: ARG002
        agent_messages: list[object] | None = None,  # noqa: ARG002
    ) -> None:
        return None


def _make_manager() -> _DummyMemoryManager:
    return _DummyMemoryManager(
        store=SimpleNamespace(base_path=None),
        session_id="session-1",
    )


def test_shutdown_calls_save_when_background_is_idle():
    manager = _make_manager()
    manager.save = MagicMock()

    manager.shutdown()

    manager.save.assert_called_once_with()
    assert manager._bg_future is None


def test_shutdown_suppresses_keyboard_interrupt_from_save():
    manager = _make_manager()
    manager.save = MagicMock(side_effect=KeyboardInterrupt)

    manager.shutdown()

    manager.save.assert_called_once_with()
    assert manager._bg_future is None


def test_shutdown_suppresses_system_exit_from_save():
    manager = _make_manager()
    manager.save = MagicMock(side_effect=SystemExit)

    manager.shutdown()

    manager.save.assert_called_once_with()
    assert manager._bg_future is None


def test_shutdown_suppresses_os_error_from_save():
    """Real-world exception: disk-full or permission errors during save."""
    manager = _make_manager()
    manager.save = MagicMock(side_effect=OSError("No space left on device"))

    manager.shutdown()

    manager.save.assert_called_once_with()
    assert manager._bg_future is None


def test_shutdown_suppresses_permission_error_from_save():
    """Real-world exception: permission denied during save."""
    manager = _make_manager()
    manager.save = MagicMock(side_effect=PermissionError("Permission denied"))

    manager.shutdown()

    manager.save.assert_called_once_with()
    assert manager._bg_future is None


def test_shutdown_skips_save_when_background_job_is_still_running():
    manager = _make_manager()
    manager.save = MagicMock()
    manager.save_messages_only = MagicMock()
    manager._save_hybrid_meta = MagicMock()
    manager._save_mode_meta = MagicMock()
    manager._save_tier_cache = MagicMock()
    bg_future = MagicMock()
    bg_future.running.return_value = True
    bg_future.done.return_value = False
    manager._bg_future = bg_future

    manager.shutdown()

    manager.save.assert_not_called()
    manager.save_messages_only.assert_called_once_with()
    manager._save_hybrid_meta.assert_called_once_with(block=False)
    manager._save_mode_meta.assert_called_once_with()
    manager._save_tier_cache.assert_called_once_with(block=False)
    bg_future.cancel.assert_called_once_with()
    assert manager._bg_future is None


# ── Test helpers for real memory managers ───────────────────────────────────


class _FakeLLM:
    """Minimal LangChain-compatible LLM for testing background summarization."""

    def invoke(self, messages: list[object]) -> object:
        return SimpleNamespace(content="Fake summary for testing.")


def _make_conversation_manager(
    tmpdir: str,
    session_id: str = "test-session",
    working_memory_size: int = 5,
) -> ConversationMemoryManager:
    store = JsonFileMemoryStore(base_dir=tmpdir)
    return ConversationMemoryManager(
        store=store,
        session_id=session_id,
        config={
            "summarization": True,
            "working_memory_size": working_memory_size,
            "summary_threshold": working_memory_size + 1,
        },
    )


class _PausableConversationManager(ConversationMemoryManager):
    """Manager that pauses inside ``_run_slow_path()`` for deterministic tests.

    Signals ``_slow_path_started`` when the background job begins and
    ``_slow_path_paused`` once it reaches the pause point.  The test thread
    can then call ``shutdown()`` while the background summarizer is
    logically mid-flight, and resume it via ``_slow_path_resume``.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._slow_path_started: threading.Event = threading.Event()
        self._slow_path_paused: threading.Event = threading.Event()
        self._slow_path_resume: threading.Event = threading.Event()

    def _run_slow_path(self, batch: list[object], unsummarized_end: int) -> None:
        self._slow_path_started.set()
        self._slow_path_paused.set()
        self._slow_path_resume.wait()
        super()._run_slow_path(batch, unsummarized_end)


# ── Tests for ConversationMemoryManager shutdown ────────────────────────────


def test_conversation_shutdown_with_active_background_job():
    """Shutdown while background summarizer is running must not deadlock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = _PausableConversationManager(
            store=store,
            session_id="test-shutdown-bg",
            config={
                "summarization": True,
                "working_memory_size": 5,
                "summary_threshold": 6,
            },
        )
        manager.set_llm(_FakeLLM())

        for i in range(15):
            manager.update(f"pre-{i} " + "a" * 400, f"pre-ai-{i} " + "b" * 400)

        assert manager._slow_path_started.wait(
            timeout=10.0
        ), "Timed out waiting for background summarizer to start"
        assert manager._slow_path_paused.wait(
            timeout=10.0
        ), "Timed out waiting for background summarizer to pause"

        bg_fut = manager._bg_future

        # shutdown must return promptly even though bg job is running
        manager.shutdown()

        # Resume the background thread so it can finish naturally.
        manager._slow_path_resume.set()
        if bg_fut is not None:
            try:
                bg_fut.result(timeout=10.0)
            except Exception:
                pass

        assert manager._bg_future is None

        # Messages should have been saved via save_messages_only fast-path
        reloaded = JsonFileMemoryStore(base_dir=tmpdir)
        messages = reloaded.load_history("test-shutdown-bg")
        assert len(messages) == 30, f"Expected 30 messages, got {len(messages)}"


def test_conversation_shutdown_with_save_oserror():
    """shutdown() must complete even when save() raises OSError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_conversation_manager(tmpdir)
        manager.save = MagicMock(side_effect=OSError("disk full"))

        manager.shutdown()

        manager.save.assert_called_once_with()
        assert manager._bg_future is None


def test_conversation_shutdown_with_save_permission_error():
    """shutdown() must complete even when save() raises PermissionError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_conversation_manager(tmpdir)
        manager.save = MagicMock(side_effect=PermissionError("denied"))

        manager.shutdown()

        manager.save.assert_called_once_with()
        assert manager._bg_future is None


# ── Tests for CodeDevelopmentMemoryManager shutdown ─────────────────────────


def test_code_shutdown_persists_file_tracking():
    """CodeDevelopmentMemoryManager shutdown must preserve file state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = CodeDevelopmentMemoryManager(
            store=store,
            session_id="test-code-shutdown",
        )
        manager.update("Look at `cogtrix_core/main.py`", "Okay")

        manager.shutdown()

        assert manager._bg_future is None
        # Mode meta should have been written (file tracking state)
        meta_path = manager._mode_meta_path()
        assert meta_path.exists(), "Mode meta should be persisted after shutdown"


# ── Tests for ReasoningMemoryManager shutdown ───────────────────────────────


def test_reasoning_shutdown_persists_goals():
    """ReasoningMemoryManager shutdown must preserve goal state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = ReasoningMemoryManager(
            store=store,
            session_id="test-reasoning-shutdown",
        )
        manager.update("Goal: fix bug", "Working on it")
        manager._goals.append(Goal(id="g1", description="fix bug"))

        manager.shutdown()

        assert manager._bg_future is None
        # Mode meta should have been written (goal state)
        meta_path = manager._mode_meta_path()
        assert meta_path.exists(), "Mode meta should be persisted after shutdown"
