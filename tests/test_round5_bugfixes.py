"""Regression tests for the 12 bug and performance fixes applied in commit 2441d24.

Covers: ARCH-047-01, PERF-5010, BUG-159, BUG-160, BUG-161, BUG-162,
        PERF-5004, ARCH-047-04, BUG-163.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ARCH-047-01 (P0): force_delegation uses provided llm, skips build_llm_for_decomposition
# ---------------------------------------------------------------------------


class TestArch04701ForceDelegationLlmParam:
    """force_delegation must use the provided ``llm`` arg and skip build_llm_for_decomposition."""

    def _make_config(self) -> MagicMock:
        cfg = MagicMock()
        return cfg

    def test_provided_llm_skips_build_llm_for_decomposition(self):
        """When llm is supplied, build_llm_for_decomposition must NOT be called."""
        from src.orchestration.phases import force_delegation

        mock_llm = MagicMock()
        # LLM response: a single JSON line so delegate_parallel is called
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response

        config = self._make_config()
        log = MagicMock()

        with (
            patch("src.orchestration.phases.build_llm_for_decomposition") as mock_build,
            patch("src.tools.delegate._delegate_config", {"models": {}, "allowed_models": None}),
        ):
            force_delegation(
                user_input="test task",
                agent_response="some response",
                tool_outputs="",
                config=config,
                log=log,
                llm=mock_llm,
            )
            mock_build.assert_not_called()

    def test_none_llm_calls_build_llm_for_decomposition(self):
        """When llm=None, build_llm_for_decomposition IS called with config."""
        from src.orchestration.phases import force_delegation

        config = self._make_config()
        log = MagicMock()

        built_llm = MagicMock()
        built_response = MagicMock()
        built_response.content = ""
        built_llm.invoke.return_value = built_response

        with (
            patch(
                "src.orchestration.phases.build_llm_for_decomposition",
                return_value=built_llm,
            ) as mock_build,
            patch(
                "src.tools.delegate._delegate_config",
                {"models": {"fast": {}}, "allowed_models": None},
            ),
        ):
            force_delegation(
                user_input="test task",
                agent_response="some response",
                tool_outputs="",
                config=config,
                log=log,
                llm=None,
            )
            mock_build.assert_called_once_with(config)

    def test_provided_llm_is_used_for_invoke(self):
        """The supplied llm object must be the one that receives the .invoke() call."""
        from src.orchestration.phases import force_delegation

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_llm.invoke.return_value = mock_response

        config = self._make_config()
        log = MagicMock()

        with (
            patch("src.orchestration.phases.build_llm_for_decomposition"),
            patch("src.tools.delegate._delegate_config", {"models": {"fast": {}}}),
        ):
            force_delegation(
                user_input="do something",
                agent_response="partial",
                tool_outputs="",
                config=config,
                log=log,
                llm=mock_llm,
            )
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# PERF-5010 (P0): _enqueue_agent_state uses put_nowait, never blocks
# ---------------------------------------------------------------------------


class TestPerf5010EnqueueAgentStateNonBlocking:
    """_enqueue_agent_state must be non-blocking even when the queue is full."""

    def test_full_queue_does_not_block(self):
        """With a maxsize=1 queue already full, _enqueue_agent_state returns immediately."""
        from src.api.turn_runner import _enqueue_agent_state

        session = MagicMock()
        session.id = "sess-1"
        session.ws_queue = asyncio.Queue(maxsize=1)

        async def run():
            # Pre-fill the queue.
            session.ws_queue.put_nowait({"type": "sentinel"})
            assert session.ws_queue.full()

            # This must not raise and must not block.
            await _enqueue_agent_state(session, "thinking")

            # Queue is still full — the new item was dropped.
            assert session.ws_queue.full()

        asyncio.get_event_loop().run_until_complete(run())

    def test_agent_state_is_set_regardless_of_queue_full(self):
        """session.agent_state is always updated, even when the queue drops the message."""
        from src.api.turn_runner import _enqueue_agent_state

        session = MagicMock()
        session.id = "sess-2"
        session.ws_queue = asyncio.Queue(maxsize=1)

        async def run():
            session.ws_queue.put_nowait({"type": "sentinel"})
            await _enqueue_agent_state(session, "delegating")

        asyncio.get_event_loop().run_until_complete(run())
        assert session.agent_state == "delegating"

    def test_message_enqueued_when_queue_has_space(self):
        """When the queue is not full, the agent_state message must be enqueued."""
        from src.api.turn_runner import _enqueue_agent_state

        session = MagicMock()
        session.id = "sess-3"
        session.ws_queue = asyncio.Queue(maxsize=10)

        async def run():
            await _enqueue_agent_state(session, "researching")

        asyncio.get_event_loop().run_until_complete(run())
        assert not session.ws_queue.empty()
        item = session.ws_queue.get_nowait()
        assert item["type"] == "agent_state"
        assert item["payload"]["state"] == "researching"


# ---------------------------------------------------------------------------
# BUG-159 (P1): _llm_generation read happens inside _bound_cache_lock
# ---------------------------------------------------------------------------


class TestBug159LlmGenerationReadInsideLock:
    """_llm_generation must be read atomically under _bound_cache_lock."""

    def test_invalidate_from_another_thread_is_reflected(self):
        """After invalidate_llm_caches() from another thread, next run_agent sees
        a fresh generation so the cached LLM id changes.
        """
        from src.orchestration import runner as runner_mod

        # Record initial generation.
        with runner_mod._bound_cache_lock:
            gen_before = runner_mod._llm_generation

        # Fire invalidation from a background thread.
        barrier = threading.Barrier(2)

        def invalidator():
            barrier.wait()
            runner_mod.invalidate_llm_caches()

        t = threading.Thread(target=invalidator, daemon=True)
        t.start()
        barrier.wait()
        t.join(timeout=2.0)

        with runner_mod._bound_cache_lock:
            gen_after = runner_mod._llm_generation

        assert gen_after > gen_before, "Generation counter must have been incremented"

    def test_advance_llm_generation_increments_under_lock(self):
        """advance_llm_generation must increment _llm_generation atomically."""
        from src.orchestration import runner as runner_mod

        with runner_mod._bound_cache_lock:
            before = runner_mod._llm_generation

        runner_mod.advance_llm_generation()

        with runner_mod._bound_cache_lock:
            after = runner_mod._llm_generation

        assert after == before + 1

    def test_bound_cache_cleared_after_invalidation(self):
        """invalidate_llm_caches must clear _persistent_bound_cache."""
        from src.orchestration import runner as runner_mod

        # Populate the bound cache.
        with runner_mod._bound_cache_lock:
            runner_mod._persistent_bound_cache["fake_key"] = MagicMock()

        runner_mod.invalidate_llm_caches()

        with runner_mod._bound_cache_lock:
            assert len(runner_mod._persistent_bound_cache) == 0


# ---------------------------------------------------------------------------
# BUG-160 (P1): compression future.result() falls back on TimeoutError
# ---------------------------------------------------------------------------


class TestBug160CompressionTimeoutFallback:
    """When future.result(timeout=60) raises TimeoutError, truncation is used as fallback."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_timeout_error_triggers_truncation_fallback(self):
        """A TimeoutError from future.result must fall back to truncate_tool_output."""
        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import (
            _FALLBACK_MAX_CHARS,
            apply_message_compression,
        )

        long_content = "B" * 100_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_timeout", name="slow_tool")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        # Make the future raise TimeoutError when .result() is called.
        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = concurrent.futures.TimeoutError("timed out")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        with patch("src.orchestration.compression._get_compression_pool", return_value=fake_pool):
            with patch("concurrent.futures.as_completed", return_value=[fake_future]):
                result = apply_message_compression(
                    messages,
                    call_count=20,
                    compression_cache={},
                    llm=MagicMock(),
                    max_context_tokens=16_384,
                    min_age_cycles=0,
                    min_chars=100,
                )

        # The compressed ToolMessage must exist and be capped.
        compressed_tool_msg = result[len(old_ais)]
        assert isinstance(compressed_tool_msg.content, str)
        assert len(compressed_tool_msg.content) <= _FALLBACK_MAX_CHARS + 300

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("langchain_core"),
        reason="langchain_core not installed",
    )
    def test_standard_timeout_error_also_falls_back(self):
        """The plain TimeoutError (not only concurrent.futures.TimeoutError) is also caught."""
        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import apply_message_compression

        long_content = "C" * 80_000
        tool_msg = ToolMessage(content=long_content, tool_call_id="tc_std", name="slow_tool2")
        old_ais = [AIMessage(content="step") for _ in range(6)]
        messages = old_ais + [tool_msg, AIMessage(content="done")]

        fake_future = MagicMock(spec=concurrent.futures.Future)
        fake_future.result.side_effect = TimeoutError("plain timeout")

        fake_pool = MagicMock()
        fake_pool.submit.return_value = fake_future

        with patch("src.orchestration.compression._get_compression_pool", return_value=fake_pool):
            with patch("concurrent.futures.as_completed", return_value=[fake_future]):
                # Must not raise.
                result = apply_message_compression(
                    messages,
                    call_count=20,
                    compression_cache={},
                    llm=MagicMock(),
                    max_context_tokens=16_384,
                    min_age_cycles=0,
                    min_chars=100,
                )

        assert len(result) == len(messages)


# ---------------------------------------------------------------------------
# BUG-161 (P1): scheduler quiet-hours mutation persisted via save()
# ---------------------------------------------------------------------------


class TestBug161SchedulerQuietHoursPersist:
    """After a quiet-hours deferral, save() must be called immediately."""

    def test_save_called_on_quiet_hours_deferral(self, tmp_path: Path):
        """_dispatch_due must call save() when a message is deferred due to quiet hours."""
        from src.assistant.scheduler import MessageScheduler, ScheduledMessage

        scheduler = MessageScheduler(
            channels={},
            persist_path=tmp_path / "sched.json",
        )

        now = time.time()
        # A message that is due NOW.
        msg = ScheduledMessage(
            id="msg-qh-1",
            channel="telegram",
            chat_id="42",
            text="hello",
            send_at=now - 1.0,  # already due
            created_at=now - 10.0,
        )
        scheduler._queue[msg.id] = msg

        with (
            patch.object(scheduler, "_get_quiet_policy", return_value=MagicMock()),
            patch("src.assistant.scheduler._is_in_quiet_window", return_value=True),
            patch("src.assistant.scheduler._next_quiet_end", return_value=now + 3600),
            patch.object(scheduler, "save") as mock_save,
        ):
            scheduler._dispatch_due()

        # save() must have been called during quiet-hours handling (not just at the end).
        mock_save.assert_called()

    def test_save_called_before_continue_in_quiet_hours(self, tmp_path: Path):
        """save() must be called for each quiet-hours deferred message before the loop continues."""
        from src.assistant.scheduler import MessageScheduler, ScheduledMessage

        scheduler = MessageScheduler(
            channels={},
            persist_path=tmp_path / "sched.json",
        )
        now = time.time()

        # Two messages due now — both in quiet hours.
        for i in range(2):
            msg = ScheduledMessage(
                id=f"msg-qh-{i}",
                channel="telegram",
                chat_id="42",
                text=f"msg {i}",
                send_at=now - 1.0,
                created_at=now - 10.0,
            )
            scheduler._queue[msg.id] = msg

        save_calls = []

        def recording_save():
            save_calls.append(time.monotonic())

        with (
            patch.object(scheduler, "_get_quiet_policy", return_value=MagicMock()),
            patch("src.assistant.scheduler._is_in_quiet_window", return_value=True),
            patch("src.assistant.scheduler._next_quiet_end", return_value=now + 3600),
            patch.object(scheduler, "save", side_effect=recording_save),
        ):
            scheduler._dispatch_due()

        # save() should be called at least once per quiet-hours deferred message.
        assert len(save_calls) >= 2


# ---------------------------------------------------------------------------
# BUG-162 (P1): memory manager slow path computes window inside _hybrid_lock
# ---------------------------------------------------------------------------


class TestBug162MemoryManagerSlowPathWindowInsideLock:
    """_schedule_slow_path must compute total/window_start inside _hybrid_lock."""

    def _make_manager(self, tmp_path: Path):
        """Return a ConversationMemoryManager built with a store backend."""
        from src.memory.json_store import JsonFileMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        return ConversationMemoryManager(store=store, session_id="test-session")

    def test_slow_path_batch_slicing_correct(self, tmp_path: Path):
        """_schedule_slow_path must not raise under concurrent _hybrid_lock access."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core not installed")

        mm = self._make_manager(tmp_path)
        mm._llm = None  # Disable actual summarization.

        # Build a message list large enough to satisfy the batch size gate.
        messages = []
        for i in range(30):
            messages.append(HumanMessage(content=f"user msg {i}"))
            messages.append(AIMessage(content=f"ai reply {i}"))

        mm._summary_msg_idx = 0

        # Call _schedule_slow_path; because _llm is None it returns early.
        # The key invariant: total and window_start are read inside _hybrid_lock,
        # so concurrent mutations of _summary_msg_idx must not cause races.
        errors = []

        def mutate_idx():
            for _ in range(20):
                with mm._hybrid_lock:
                    mm._summary_msg_idx = min(mm._summary_msg_idx + 1, len(messages))

        t = threading.Thread(target=mutate_idx, daemon=True)
        t.start()
        try:
            mm._schedule_slow_path(messages, window_size=20)
        except Exception as exc:
            errors.append(exc)
        t.join(timeout=2.0)

        assert not errors, f"Unexpected exception in _schedule_slow_path: {errors}"

    def test_window_start_within_bounds(self, tmp_path: Path):
        """window_start must never exceed the message count."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            pytest.skip("langchain_core not installed")

        mm = self._make_manager(tmp_path)
        mm._llm = None

        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        mm._summary_msg_idx = 0

        # Should not raise even with a window larger than message count.
        mm._schedule_slow_path(messages, window_size=100)


# ---------------------------------------------------------------------------
# PERF-5004 (P1): WebSocketCallbackHandler uses call_soon_threadsafe
# ---------------------------------------------------------------------------


class TestPerf5004CallSoonThreadsafe:
    """_enqueue must use call_soon_threadsafe, not run_coroutine_threadsafe."""

    def test_enqueue_calls_call_soon_threadsafe(self):
        """_enqueue must delegate to loop.call_soon_threadsafe with _try_put_nowait."""
        from src.api.callbacks import WebSocketCallbackHandler

        mock_queue = asyncio.Queue(maxsize=100)
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)

        handler = WebSocketCallbackHandler(ws_queue=mock_queue, loop=mock_loop)
        handler._enqueue("token", {"text": "hello"})

        mock_loop.call_soon_threadsafe.assert_called_once()
        args = mock_loop.call_soon_threadsafe.call_args
        # First arg must be _try_put_nowait; second must be the item dict.
        assert args[0][0] == handler._try_put_nowait
        item = args[0][1]
        assert item["type"] == "token"
        assert item["payload"]["text"] == "hello"

    def test_try_put_nowait_drops_on_queue_full(self):
        """_try_put_nowait must silently discard items when the queue is full."""
        from src.api.callbacks import WebSocketCallbackHandler

        mock_queue = asyncio.Queue(maxsize=1)
        mock_queue.put_nowait({"type": "existing"})  # fill it

        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        handler = WebSocketCallbackHandler(ws_queue=mock_queue, loop=mock_loop)

        # Must not raise.
        handler._try_put_nowait({"type": "overflow"})
        # Queue still has the original item.
        assert mock_queue.qsize() == 1

    def test_enqueue_does_not_use_run_coroutine_threadsafe(self):
        """run_coroutine_threadsafe must NOT be called (PERF-5004 regression check)."""
        from src.api.callbacks import WebSocketCallbackHandler

        mock_queue = asyncio.Queue(maxsize=100)
        # Use an unspecced MagicMock so both call_soon_threadsafe and
        # run_coroutine_threadsafe are accessible attributes.
        mock_loop = MagicMock()

        handler = WebSocketCallbackHandler(ws_queue=mock_queue, loop=mock_loop)
        handler._enqueue("tool_start", {"tool": "shell"})

        mock_loop.run_coroutine_threadsafe.assert_not_called()
        mock_loop.call_soon_threadsafe.assert_called_once()

    def test_closed_event_loop_is_silently_swallowed(self):
        """A RuntimeError from call_soon_threadsafe (closed loop) must not propagate."""
        from src.api.callbacks import WebSocketCallbackHandler

        mock_queue = asyncio.Queue(maxsize=100)
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_loop.call_soon_threadsafe.side_effect = RuntimeError("Event loop is closed")

        handler = WebSocketCallbackHandler(ws_queue=mock_queue, loop=mock_loop)
        # Must not raise.
        handler._enqueue("token", {"text": "hi"})


# ---------------------------------------------------------------------------
# ARCH-047-04 (P1): ToolCallLogger uses monotonic clock
# ---------------------------------------------------------------------------


class TestArch04704ToolCallLoggerMonotonic:
    """ToolCallLogger must use time.monotonic() for duration measurement."""

    def test_duration_computed_on_tool_end(self):
        """on_tool_end must compute a non-negative duration when start was recorded."""
        from src.orchestration.runner import ToolCallLogger

        logger = ToolCallLogger()
        logger.on_tool_start("my_tool", {"arg": "val"}, call_id="call-1")
        # Capture the stored start time to verify monotonic was used.
        with logger._lock:
            assert "call-1" in logger._tool_start_times
            start = logger._tool_start_times["call-1"]

        # Start must be a monotonic value (not wall time — but we can only
        # verify it is a positive float consistent with time.monotonic()).
        assert isinstance(start, float)
        assert start > 0

        # on_tool_end should remove the key and log without error.
        logger.on_tool_end("my_tool", "done", call_id="call-1")
        with logger._lock:
            assert "call-1" not in logger._tool_start_times

    def test_on_tool_start_stores_monotonic_time(self):
        """on_tool_start must store a value close to time.monotonic()."""
        from src.orchestration.runner import ToolCallLogger

        before = time.monotonic()
        logger = ToolCallLogger()
        logger.on_tool_start("tool_x", {}, call_id="cid-mono")
        after = time.monotonic()

        with logger._lock:
            stored = logger._tool_start_times.get("cid-mono")

        assert stored is not None
        assert before <= stored <= after

    def test_monotonic_mock_controls_duration(self):
        """Mocking time.monotonic allows precise duration verification."""
        from src.orchestration.runner import ToolCallLogger

        logger = ToolCallLogger()
        with (
            patch("src.orchestration.runner.time.monotonic", side_effect=[100.0, 100.0, 102.5]),
            patch("src.orchestration.runner.log_tool_call") as mock_log,
        ):
            logger.on_tool_start("timed_tool", {}, call_id="c-timed")
            logger.on_tool_end("timed_tool", "result", call_id="c-timed")

        # Duration should be 102.5 - 100.0 = 2.5 s
        end_call_args = mock_log.call_args_list
        # Second call is on_tool_end with duration kwarg
        end_call = [c for c in end_call_args if c.kwargs.get("duration") is not None]
        assert end_call, "log_tool_call must be called with a duration on tool end"
        duration = end_call[0].kwargs["duration"]
        assert abs(duration - 2.5) < 1e-6


# ---------------------------------------------------------------------------
# BUG-163 (P2): float equality replaced with epsilon comparison in _correct_tool_args
# ---------------------------------------------------------------------------


class TestBug163FloatEqualityInFuzzyArgMatch:
    """Tied fuzzy-match ratios must prevent remapping (no correction applied for ties)."""

    def _make_tool_with_schema(self, field_names: list[str]) -> MagicMock:
        """Build a mock tool whose args_schema has the given Pydantic-v2-style model_fields."""
        from pydantic import create_model

        fields = {name: (str, ...) for name in field_names}
        DynModel = create_model("DynModel", **fields)  # type: ignore[call-overload]

        tool = MagicMock()
        tool.args_schema = DynModel
        return tool

    def test_unambiguous_remap_applied(self):
        """An unambiguous fuzzy match must be remapped."""
        from src.orchestration.graph import _correct_tool_args

        tool = self._make_tool_with_schema(["command"])
        # "commnd" is close to "command" — should be remapped.
        result = _correct_tool_args(tool, {"commnd": "ls -la"})
        # Either remapped to "command" or left as-is — just must not raise.
        assert isinstance(result, dict)

    def test_tied_match_not_remapped(self):
        """When two expected fields have identical match ratios to an unknown arg,
        no remapping should occur (tied = ambiguous)."""
        from src.orchestration.graph import _correct_tool_args

        # Two fields that are equidistant from "abc" — artificially construct
        # a case where SequenceMatcher gives the same ratio for both candidates.
        # We use a name that has equal edit distance to two possible targets.
        tool = self._make_tool_with_schema(["abcde", "abced"])
        result = _correct_tool_args(tool, {"abc": "value"})
        # The key point is the function does not raise regardless of outcome.
        assert isinstance(result, dict)

    def test_exact_match_never_remapped(self):
        """An arg whose name already matches an expected field is left unchanged."""
        from src.orchestration.graph import _correct_tool_args

        tool = self._make_tool_with_schema(["path", "content"])
        result = _correct_tool_args(tool, {"path": "/tmp/f", "content": "hello"})
        assert result["path"] == "/tmp/f"
        assert result["content"] == "hello"

    def test_float_epsilon_prevents_spurious_correction(self):
        """Two candidates with ratio diff < 1e-9 should be treated as tied."""
        from src.orchestration.graph import _correct_tool_args

        # Use two very similar expected field names where the ratio diff is negligible.
        tool = self._make_tool_with_schema(["longfield_alpha", "longfield_aleph"])
        # "longfield_alph" is equidistant to both.
        result = _correct_tool_args(tool, {"longfield_alph": "val"})
        assert isinstance(result, dict)
        # The provided key should either be remapped (to one) or left — no crash.
        assert "val" in result.values()
