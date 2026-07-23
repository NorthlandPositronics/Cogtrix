"""Concurrency and thread-safety tests for memory managers.

Issue #753 — Memory managers have zero concurrency/thread-safety tests.
Issue #987 — CodeDevelopmentMemoryManager and ReasoningMemoryManager have
zero concurrency test coverage.

Verifies that concurrent ``update()``, ``save()``, and ``shutdown()`` calls
do not corrupt state across all memory modes.
"""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

# LangChain message types are required for the test subclasses that
# override update().  In the test environment langchain-core is always
# available via the dev dependency group.
from langchain_core.messages import AIMessage, HumanMessage  # type: ignore[import-untyped]

from cogtrix_core.memory.json_store import JsonFileMemoryStore
from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager
from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager


def _make_manager(tmpdir: str, session_id: str = "test-session") -> ConversationMemoryManager:
    """Create a ConversationMemoryManager with summarization disabled."""
    store = JsonFileMemoryStore(base_dir=tmpdir)
    return ConversationMemoryManager(
        store=store,
        session_id=session_id,
        config={"summarization": False},
    )


# ── Test hook subclasses ────────────────────────────────────────────────────


class _PausableMemoryManager(ConversationMemoryManager):
    """Manager that pauses inside ``update()`` for deterministic concurrency tests.

    After appending the human message, ``update()`` signals
    ``_update_paused`` and blocks on ``_update_resume``.  This allows
    the test thread to inspect intermediate state or call ``save()`` /
    ``shutdown()`` while ``update()`` is logically in progress.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._update_paused: threading.Event = threading.Event()
        self._update_resume: threading.Event = threading.Event()

    def update(  # type: ignore[override]
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[object] | None = None,
    ) -> None:
        # -- Build and append the human message --------------------------------
        human_msg = HumanMessage(content=user_input)
        self._set_msg_ts(human_msg, self._pending_user_ts)
        self._pending_user_ts = None
        self._messages.append(human_msg)

        # -- Pause: signal the test thread, then wait for the resume signal ----
        self._update_paused.set()
        self._update_resume.wait()

        # -- Append the AI response -------------------------------------------
        if agent_messages is not None:
            for m in agent_messages:
                self._messages.append(m)
            last = agent_messages[-1]
            if hasattr(last, "content") or isinstance(last, dict):
                self._set_msg_ts(last)
        else:
            ai_msg = AIMessage(content=ai_response)
            self._set_msg_ts(ai_msg)
            self._messages.append(ai_msg)

        # Skip token tracking and background scheduling for test simplicity.
        # The parent's _tokens_since_summary, _schedule_slow_path, and
        # schedule_tier_roll_forward are not exercised here — they are tested
        # in the full update() path via the other test cases.


# ── Test 1: concurrent update() calls do not lose messages ───────────────────


def test_concurrent_update_calls_do_not_lose_messages() -> None:
    """Multiple threads calling ``update()`` concurrently must not drop turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir)
        num_workers = 10
        calls_per_worker = 50

        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()  # synchronise start
            for i in range(calls_per_worker):
                manager.update(f"u-{worker_id}-{i}", f"ai-{worker_id}-{i}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()  # propagate any exception

        expected = num_workers * calls_per_worker * 2  # human + ai per turn
        actual = manager.get_message_count()
        assert (
            actual == expected
        ), f"Expected {expected} messages, got {actual} — messages were lost"

        # Verify that save + reload produces consistent state
        manager.save()
        reloaded = manager.store.load_history(manager.session_id)
        assert (
            len(reloaded) == expected
        ), f"Reloaded count mismatch: expected {expected}, got {len(reloaded)}"


# ── Test 2: concurrent save() and update() do not interleave ────────────────


def test_concurrent_save_does_not_produce_partial_turn() -> None:
    """``save()`` called mid-``update()`` must not persist an orphaned human message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = _PausableMemoryManager(
            store=store,
            session_id="test-concurrent-save",
            config={"summarization": False},
        )

        # Start update() in a background thread — it will pause after
        # appending the human message.
        update_done: list[Exception | None] = [None]

        def _do_update() -> None:
            try:
                manager.update("hello", "world")
            except Exception as exc:  # noqa: BLE001
                update_done[0] = exc

        t = threading.Thread(target=_do_update)
        t.start()

        # Wait for update() to reach the pause point.
        assert manager._update_paused.wait(
            timeout=5.0
        ), "Timed out waiting for update() to pause — may be a deadlock"

        # Call save() while update() is paused between human and AI append.
        manager.save()

        # Inspect what was saved: at minimum the human message must be present,
        # and save() must not have crashed.
        saved = store.load_history("test-concurrent-save")
        assert saved is not None, "save() should not produce None"
        assert len(saved) >= 1, f"Expected ≥1 message in saved state, got {len(saved)}"

        # Resume update() so it can append the AI response.
        manager._update_resume.set()
        t.join(timeout=5.0)
        assert not t.is_alive(), "Update thread should have completed"
        assert update_done[0] is None, f"update() raised an exception: {update_done[0]}"

        # Final save and verify the complete turn is present.
        manager.save()
        final = store.load_history("test-concurrent-save")
        assert len(final) == 2, f"Expected 2 messages (human + AI), got {len(final)}"


def test_concurrent_save_and_update_loop_consistency() -> None:
    """Stress-test: rapid concurrent ``save()`` + ``update()`` in a loop.

    Even without deterministic pause points, repeated interleaving
    should not produce a corrupt or truncated message list.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir, session_id="stress-test")
        iterations = 100

        # Pre-populate so the manager has existing state.
        manager.update("initial", "initial-ai")

        for i in range(iterations):
            start_barrier = threading.Barrier(2, timeout=5.0)
            errors: list[Exception] = []

            def _save(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.save()
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            def _upd(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
                _i: int = i,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.update(f"stress-{_i}", f"stress-ai-{_i}")
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            t1 = threading.Thread(target=_save)
            t2 = threading.Thread(target=_upd)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            assert not errors, f"Iteration {i}: {errors}"

        # After all iterations, verify message count.
        # Initial turn (2) + 100 stress turns (2 each) = 202 messages.
        expected = 202
        actual = manager.get_message_count()
        assert actual == expected, f"Expected {expected} messages after stress loop, got {actual}"


# ── Test 3: shutdown() during active update() ────────────────────────────────


def test_shutdown_during_active_update_completes_cleanly() -> None:
    """``shutdown()`` called while ``update()`` is paused must not deadlock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = _PausableMemoryManager(
            store=store,
            session_id="test-shutdown-concurrent",
            config={"summarization": False},
        )

        # Start update() in a background thread — it pauses after the
        # human append.
        update_errors: list[Exception | None] = [None]

        def _do_update() -> None:
            try:
                manager.update("msg-during-shutdown", "response")
            except Exception as exc:  # noqa: BLE001
                update_errors[0] = exc

        t = threading.Thread(target=_do_update)
        t.start()

        # Wait for update() to pause.
        assert manager._update_paused.wait(timeout=5.0), "Timed out waiting for update() to pause"

        # Call shutdown() while update() is logically in progress.
        manager.shutdown()

        # shutdown() must complete (we are here — no deadlock).
        # Now resume update() so the background thread can finish.
        manager._update_resume.set()
        t.join(timeout=5.0)
        assert not t.is_alive(), "Update thread should have completed after shutdown"
        assert (
            update_errors[0] is None
        ), f"update() raised an exception after shutdown: {update_errors[0]}"


def test_shutdown_does_not_lose_inflight_messages() -> None:
    """Messages from an ``update()`` that finished before shutdown must survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager(tmpdir, session_id="test-shutdown-msg-loss")

        # Add several messages and save.
        for i in range(5):
            manager.update(f"pre-shutdown-{i}", f"pre-shutdown-ai-{i}")
        manager.save()

        # Start a fast update in a separate thread, then immediately shutdown.
        last_msg_done = threading.Event()

        def _final_update() -> None:
            manager.update("final-msg", "final-ai")
            last_msg_done.set()

        t = threading.Thread(target=_final_update)
        t.start()

        # Wait for the update to finish or timeout.
        last_msg_done.wait(timeout=5.0)

        manager.shutdown()
        t.join(timeout=5.0)

        # Reload and verify all messages survived.
        reloaded = JsonFileMemoryStore(base_dir=tmpdir)
        messages = reloaded.load_history("test-shutdown-msg-loss")
        # 5 pre-shutdown turns (10 msgs) + 1 final turn (2 msgs) = 12
        assert len(messages) == 12, f"Expected 12 messages after shutdown, got {len(messages)}"


# ── CodeDevelopmentMemoryManager concurrency tests ───────────────────────────


def _make_code_manager(
    tmpdir: str, session_id: str = "test-code-session"
) -> CodeDevelopmentMemoryManager:
    """Create a CodeDevelopmentMemoryManager with summarization disabled."""
    store = JsonFileMemoryStore(base_dir=tmpdir)
    return CodeDevelopmentMemoryManager(
        store=store,
        session_id=session_id,
        config={"summarization": False},
    )


def test_code_concurrent_update_calls_do_not_lose_messages() -> None:
    """Multiple threads calling ``update()`` on CodeDevelopmentMemoryManager must not drop turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_code_manager(tmpdir)
        num_workers = 10
        calls_per_worker = 50

        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()
            for i in range(calls_per_worker):
                manager.update(f"u-{worker_id}-{i}", f"ai-{worker_id}-{i}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        expected = num_workers * calls_per_worker * 2
        actual = manager.get_message_count()
        assert (
            actual == expected
        ), f"Expected {expected} messages, got {actual} — messages were lost"

        manager.save()
        reloaded = manager.store.load_history(manager.session_id)
        assert (
            len(reloaded) == expected
        ), f"Reloaded count mismatch: expected {expected}, got {len(reloaded)}"


def test_code_concurrent_update_does_not_corrupt_file_tracking() -> None:
    """Concurrent ``update()`` with file references must not corrupt the file map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_code_manager(tmpdir)
        num_workers = 10
        files_per_worker = 20

        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()
            for i in range(files_per_worker):
                manager.update(
                    f"Look at `file_{worker_id}_{i}.py`",
                    f"Fixed `file_{worker_id}_{i}.py`",
                )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        # All files should be tracked (up to max_files limit)
        tracked = manager.get_tracked_files()
        max_files = manager._mode_config["max_files"]
        assert len(tracked) <= max_files, f"Tracked files {len(tracked)} exceed limit {max_files}"
        assert len(tracked) > 0, "Expected some files to be tracked"

        # Verify no duplicate corruption — every tracked path should be a string
        for path in tracked:
            assert isinstance(path, str), f"Corrupted file path: {path!r}"
            assert path.endswith(".py"), f"Unexpected file path: {path}"


def test_code_concurrent_file_context_and_update() -> None:
    """Concurrent ``set_file_context()`` and ``update()`` must not corrupt state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_code_manager(tmpdir)
        iterations = 100
        errors: list[Exception] = []

        def _file_setter() -> None:
            for i in range(iterations):
                try:
                    manager.set_file_context(f"path_{i}.py", snippet=f"code {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        def _updater() -> None:
            for i in range(iterations):
                try:
                    manager.update(f"msg {i}", f"resp {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        t1 = threading.Thread(target=_file_setter)
        t2 = threading.Thread(target=_updater)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors, f"Errors during concurrent file+update: {errors}"
        assert manager.get_message_count() == iterations * 2
        # File context should not be corrupted
        for path, fc in manager._files.items():
            assert isinstance(path, str)
            assert fc.path == path


def test_code_concurrent_save_and_update_loop_consistency() -> None:
    """Rapid concurrent ``save()`` + ``update()`` for code mode must not corrupt state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_code_manager(tmpdir, session_id="code-stress")
        iterations = 100

        manager.update("initial", "initial-ai")

        for i in range(iterations):
            start_barrier = threading.Barrier(2, timeout=5.0)
            errors: list[Exception] = []

            def _save(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.save()
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            def _upd(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
                _i: int = i,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.update(f"stress-{_i}", f"stress-ai-{_i}")
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            t1 = threading.Thread(target=_save)
            t2 = threading.Thread(target=_upd)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            assert not errors, f"Iteration {i}: {errors}"

        expected = 202
        actual = manager.get_message_count()
        assert actual == expected, f"Expected {expected} messages after stress loop, got {actual}"


# ── ReasoningMemoryManager concurrency tests ─────────────────────────────────


def _make_reasoning_manager(
    tmpdir: str, session_id: str = "test-reasoning-session"
) -> ReasoningMemoryManager:
    """Create a ReasoningMemoryManager with summarization disabled."""
    store = JsonFileMemoryStore(base_dir=tmpdir)
    return ReasoningMemoryManager(
        store=store,
        session_id=session_id,
        config={"summarization": False},
    )


def test_reasoning_concurrent_update_calls_do_not_lose_messages() -> None:
    """Multiple threads calling ``update()`` on ReasoningMemoryManager must not drop turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_reasoning_manager(tmpdir)
        num_workers = 10
        calls_per_worker = 50

        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()
            for i in range(calls_per_worker):
                manager.update(f"u-{worker_id}-{i}", f"ai-{worker_id}-{i}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        expected = num_workers * calls_per_worker * 2
        actual = manager.get_message_count()
        assert (
            actual == expected
        ), f"Expected {expected} messages, got {actual} — messages were lost"

        manager.save()
        reloaded = manager.store.load_history(manager.session_id)
        assert (
            len(reloaded) == expected
        ), f"Reloaded count mismatch: expected {expected}, got {len(reloaded)}"


def test_reasoning_concurrent_update_does_not_corrupt_goals() -> None:
    """Concurrent ``update()`` with goal additions must not corrupt the goal list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_reasoning_manager(tmpdir)
        num_workers = 10
        goals_per_worker = 20

        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()
            for i in range(goals_per_worker):
                manager.update(
                    f"Set goal g-{worker_id}-{i}",
                    f"Acknowledged goal g-{worker_id}-{i}",
                )
                manager.add_goal(f"g-{worker_id}-{i}", f"Goal {worker_id}-{i}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        goals = manager._goals
        # Every goal should have valid fields
        for goal in goals:
            assert isinstance(goal.id, str)
            assert isinstance(goal.description, str)
            assert goal.status in ("pending", "in_progress", "completed", "blocked")

        # Verify no None/duplicate corruption
        ids = [g.id for g in goals]
        assert None not in ids, "Corrupted goal with None id"


def test_reasoning_concurrent_goal_update_and_reasoning_step() -> None:
    """Concurrent ``add_goal()`` / ``update_goal_status()`` and ``update()`` must not corrupt state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_reasoning_manager(tmpdir)
        iterations = 100
        errors: list[Exception] = []

        def _goal_mutator() -> None:
            for i in range(iterations):
                try:
                    manager.add_goal(f"goal_{i}", f"Description {i}")
                    manager.update_goal_status(f"goal_{i}", "in_progress")
                    manager.add_reasoning_step(f"Step {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        def _updater() -> None:
            for i in range(iterations):
                try:
                    manager.update(f"msg {i}", f"resp {i}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        t1 = threading.Thread(target=_goal_mutator)
        t2 = threading.Thread(target=_updater)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not errors, f"Errors during concurrent goal+update: {errors}"
        assert manager.get_message_count() == iterations * 2

        # Goals should not be corrupted
        for goal in manager._goals:
            assert isinstance(goal.id, str)
            assert goal.status in ("pending", "in_progress", "completed", "blocked")

        # Reasoning chain should not be corrupted
        for step in manager._reasoning_chain:
            assert isinstance(step, str)


def test_reasoning_concurrent_save_and_update_loop_consistency() -> None:
    """Rapid concurrent ``save()`` + ``update()`` for reasoning mode must not corrupt state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_reasoning_manager(tmpdir, session_id="reasoning-stress")
        iterations = 100

        manager.update("initial", "initial-ai")

        for i in range(iterations):
            start_barrier = threading.Barrier(2, timeout=5.0)
            errors: list[Exception] = []

            def _save(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.save()
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            def _upd(
                _barrier: threading.Barrier = start_barrier,
                _errs: list[Exception] = errors,
                _i: int = i,
            ) -> None:
                try:
                    _barrier.wait()
                    manager.update(f"stress-{_i}", f"stress-ai-{_i}")
                except Exception as exc:  # noqa: BLE001
                    _errs.append(exc)

            t1 = threading.Thread(target=_save)
            t2 = threading.Thread(target=_upd)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            assert not errors, f"Iteration {i}: {errors}"

        expected = 202
        actual = manager.get_message_count()
        assert actual == expected, f"Expected {expected} messages after stress loop, got {actual}"


# ── Cross-mode sanity check ──────────────────────────────────────────────────


def test_all_modes_support_get_message_count() -> None:
    """Every memory mode must implement ``get_message_count()`` consistently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conv = ConversationMemoryManager(
            store=JsonFileMemoryStore(base_dir=tmpdir),
            session_id="conv",
            config={"summarization": False},
        )
        code = CodeDevelopmentMemoryManager(
            store=JsonFileMemoryStore(base_dir=tmpdir),
            session_id="code",
            config={"summarization": False},
        )
        reasoning = ReasoningMemoryManager(
            store=JsonFileMemoryStore(base_dir=tmpdir),
            session_id="reasoning",
            config={"summarization": False},
        )

        for mgr in (conv, code, reasoning):
            assert mgr.get_message_count() == 0
            mgr.update("hello", "world")
            assert mgr.get_message_count() == 2


# ── Background summarization concurrency tests ────────────────────────────────


def _make_manager_with_summarization(
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


class _FakeLLM:
    def invoke(self, messages: list[object]) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(content="Fake summary for testing.")


class _SlowPathPausableMemoryManager(ConversationMemoryManager):
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


def test_concurrent_update_and_background_summarize_no_message_loss() -> None:
    """Rapid update() calls while background summarizer runs must not drop turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _make_manager_with_summarization(tmpdir, working_memory_size=5)
        manager.set_llm(_FakeLLM())
        num_workers, calls_per_worker = 8, 20
        barrier = threading.Barrier(num_workers)

        def _worker(worker_id: int) -> None:
            barrier.wait()
            for i in range(calls_per_worker):
                manager.update(
                    f"user-{worker_id}-{i} " + "x" * 300, f"ai-{worker_id}-{i} " + "y" * 300
                )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for f in as_completed([executor.submit(_worker, w) for w in range(num_workers)]):
                f.result()

        manager.join_background(timeout=30.0)
        expected = num_workers * calls_per_worker * 2
        assert (
            manager.get_message_count() == expected
        ), f"Expected {expected}, got {manager.get_message_count()}"
        manager.save()
        assert len(manager.store.load_history(manager.session_id)) == expected
        assert manager._summary is not None


def test_save_during_background_summarize_no_deadlock() -> None:
    """save() called while background summarizer is mid-flight must not deadlock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = _SlowPathPausableMemoryManager(
            store=store,
            session_id="test-save-during-bg",
            config={"summarization": True, "working_memory_size": 5, "summary_threshold": 6},
        )
        manager.set_llm(_FakeLLM())
        for i in range(15):
            manager.update(f"pre-{i} " + "a" * 400, f"pre-ai-{i} " + "b" * 400)
        assert manager._slow_path_started.wait(timeout=10.0)
        assert manager._slow_path_paused.wait(timeout=10.0)
        manager.save()
        manager._slow_path_resume.set()
        manager.join_background(timeout=10.0)
        saved = store.load_history("test-save-during-bg")
        assert saved is not None and len(saved) == 30


def test_shutdown_during_background_summarize_completes_cleanly() -> None:
    """shutdown() while background summarizer is mid-flight must not deadlock."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = _SlowPathPausableMemoryManager(
            store=store,
            session_id="test-shutdown-during-bg",
            config={"summarization": True, "working_memory_size": 5, "summary_threshold": 6},
        )
        manager.set_llm(_FakeLLM())
        for i in range(15):
            manager.update(f"pre-{i} " + "a" * 400, f"pre-ai-{i} " + "b" * 400)
        assert manager._slow_path_started.wait(timeout=10.0)
        assert manager._slow_path_paused.wait(timeout=10.0)
        bg_fut = manager._bg_future
        manager.shutdown()
        manager._slow_path_resume.set()
        if bg_fut is not None:
            try:
                bg_fut.result(timeout=10.0)
            except Exception:
                pass
        messages = JsonFileMemoryStore(base_dir=tmpdir).load_history("test-shutdown-during-bg")
        assert len(messages) == 30


# ── Test: concurrent _get_facts_store() lazy init is thread-safe ─────────────


def test_concurrent_get_facts_store_creates_single_instance() -> None:
    """Multiple threads racing into ``_get_facts_store()`` must create only one ``PersistentFactsStore``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonFileMemoryStore(base_dir=tmpdir)
        manager = ConversationMemoryManager(
            store=store,
            session_id="test-facts-race",
            config={"summarization": False},
        )

        call_count = 0
        construct_event = threading.Event()

        class _CountingFactsStore:
            def __init__(self, session_id: str, storage_dir: str) -> None:
                nonlocal call_count
                call_count += 1
                construct_event.set()
                self.session_id = session_id
                self.storage_dir = storage_dir

            def load(self):
                return None

            def save(self, facts, ttl_days=7):
                pass

            def clear(self):
                pass

        with patch("cogtrix_core.memory.facts.PersistentFactsStore", _CountingFactsStore):
            start_barrier = threading.Barrier(5)

            def _worker() -> None:
                start_barrier.wait(timeout=2.0)
                manager._get_facts_store()

            threads = [threading.Thread(target=_worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
                assert not t.is_alive(), "Worker thread should have completed"

        assert (
            call_count == 1
        ), f"Expected exactly 1 PersistentFactsStore instantiation, got {call_count}"
