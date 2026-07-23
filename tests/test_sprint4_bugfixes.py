"""Sprint 4 regression tests.

Covers:
- BUG-089: flush_all cancels timers created during the flush window
- PERF-1005: module-level compression pool is reused across passes
- PERF-1006: module-level tool executor is reused and UserCancelledRun propagates
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# BUG-089 — flush_all cancels post-snapshot timers
# ---------------------------------------------------------------------------


class TestFlushAllCancelsPostSnapshotTimers:
    """flush_all must cancel any timer added for a key between the initial
    lock snapshot and the _flush(key) call."""

    def _make_buffer(self, handler=None, executor=None):
        from src.assistant.channel import IncomingMessage
        from src.assistant.poller import MessageBuffer

        h = handler or MagicMock()
        ex = executor or MagicMock()
        buf = MessageBuffer(h, ex, debounce_seconds=60.0)
        return buf, IncomingMessage

    def _make_msg(self, IncomingMessage, chat_id="42", channel="telegram"):
        return IncomingMessage(
            channel=channel,
            chat_id=chat_id,
            message_id="m1",
            sender_id="u1",
            sender_name="Alice",
            text="hello",
            timestamp=time.time(),
        )

    def test_flush_all_cancels_timer_added_during_flush(self):
        """A timer inserted while _flush runs for another key must be cancelled."""
        from src.assistant.channel import Channel
        from src.assistant.poller import MessageBuffer

        handler = MagicMock()
        executor = MagicMock()
        buf = MessageBuffer(handler, executor, debounce_seconds=60.0)

        channel_mock = MagicMock(spec=Channel)

        from src.assistant.channel import IncomingMessage

        msg1 = self._make_msg(IncomingMessage, chat_id="100")
        msg2 = self._make_msg(IncomingMessage, chat_id="200")

        buf.add(msg1, channel_mock)
        buf.add(msg2, channel_mock)

        # Inject a side effect so that when _flush("100") is called it adds a
        # new message (and thus a new timer) for key "100".
        new_timer_created: list[threading.Timer] = []
        original_flush = buf._flush

        call_count = [0]

        def patched_flush(key: str) -> None:
            original_flush(key)
            # After flushing key "100", simulate a new message arrival that
            # creates a fresh timer for "100".
            if key == "telegram::100" and call_count[0] == 0:
                call_count[0] += 1
                new_msg = self._make_msg(IncomingMessage, chat_id="100")
                buf.add(new_msg, channel_mock)
                # Record the timer so we can inspect it later
                with buf._lock:
                    t = buf._timers.get("telegram::100")
                    if t:
                        new_timer_created.append(t)

        buf._flush = patched_flush
        buf.flush_all()

        # The post-flush timer re-check must have cancelled the new timer.
        assert len(new_timer_created) == 1, "Expected exactly one new timer to be created"
        # A cancelled threading.Timer is popped from _timers by the post-flush check.
        with buf._lock:
            assert "telegram::100" not in buf._timers

    def test_flush_all_no_new_timer_leaves_state_clean(self):
        """When no new messages arrive during flush, state is clean afterwards."""
        from src.assistant.channel import Channel
        from src.assistant.poller import MessageBuffer

        handler = MagicMock()
        executor = MagicMock()
        buf = MessageBuffer(handler, executor, debounce_seconds=60.0)

        channel_mock = MagicMock(spec=Channel)
        from src.assistant.channel import IncomingMessage

        msg = self._make_msg(IncomingMessage, chat_id="42")
        buf.add(msg, channel_mock)

        buf.flush_all()

        with buf._lock:
            assert buf._timers == {}
            assert buf._buffers == {}


# ---------------------------------------------------------------------------
# PERF-1005 — module-level compression pool
# ---------------------------------------------------------------------------


class TestCompressionPoolReuse:
    """_COMPRESSION_POOL must be a module-level ThreadPoolExecutor."""

    def test_pool_is_thread_pool_executor(self):
        from src.orchestration.compression import _COMPRESSION_POOL

        assert isinstance(_COMPRESSION_POOL, concurrent.futures.ThreadPoolExecutor)

    def test_compression_produces_correct_output(self):
        """Compression via the module-level pool returns compressed content."""
        try:
            from langchain_core.messages import AIMessage, ToolMessage
        except ImportError:
            pytest.skip("langchain_core not installed")

        from src.orchestration.compression import apply_message_compression

        old_ais = [AIMessage(content="step") for _ in range(5)]
        long_content = "x" * 50_000
        tm = ToolMessage(content=long_content, tool_call_id="tc1", name="my_tool")
        msgs = old_ais + [tm, AIMessage(content="done")]

        # Content must be > 20 chars (compress_tool_message minimum) and
        # shorter than the original to avoid the "did not reduce" fallback.
        compressed_text = "compressed summary of tool output result"
        compressed_mock = MagicMock()
        compressed_mock.content = compressed_text
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        cache: dict[str, str] = {}
        result = apply_message_compression(
            msgs,
            call_count=10,
            compression_cache=cache,
            llm=llm,
            max_context_tokens=16_384,
            min_age_cycles=1,
            min_chars=100,
        )

        assert len(result) == len(msgs)
        llm.invoke.assert_called_once()
        # The ToolMessage in the result should have compressed content.
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == compressed_text

    def test_compression_cache_reused_on_second_pass(self):
        """Second call with same tool_call_id reads from cache, no new LLM call."""
        try:
            from langchain_core.messages import AIMessage, ToolMessage
        except ImportError:
            pytest.skip("langchain_core not installed")

        from src.orchestration.compression import apply_message_compression

        old_ais = [AIMessage(content="step") for _ in range(5)]
        long_content = "y" * 50_000
        tm = ToolMessage(content=long_content, tool_call_id="tc_cached", name="my_tool")
        msgs = old_ais + [tm, AIMessage(content="done")]

        # Content must be > 20 chars for compress_tool_message to accept it.
        compressed_mock = MagicMock()
        compressed_mock.content = "cached result summary of tool output"
        llm = MagicMock()
        llm.invoke.return_value = compressed_mock

        cache: dict[str, str] = {}
        # First pass — LLM is called once.
        apply_message_compression(
            msgs,
            call_count=10,
            compression_cache=cache,
            llm=llm,
            max_context_tokens=16_384,
            min_age_cycles=1,
            min_chars=100,
        )
        assert llm.invoke.call_count == 1

        # Second pass with the populated cache — LLM must NOT be called again.
        apply_message_compression(
            msgs,
            call_count=10,
            compression_cache=cache,
            llm=llm,
            max_context_tokens=16_384,
            min_age_cycles=1,
            min_chars=100,
        )
        assert llm.invoke.call_count == 1, "LLM should not be called again when cache is warm"


# ---------------------------------------------------------------------------
# PERF-1006 — module-level tool executor
# ---------------------------------------------------------------------------


class TestToolExecutorReuse:
    """_get_tool_executor must return the same instance across calls."""

    def test_same_instance_returned(self):
        from src.orchestration.graph import _get_tool_executor

        ex1 = _get_tool_executor()
        ex2 = _get_tool_executor()
        assert ex1 is ex2, "_get_tool_executor should return the same instance"

    def test_executor_is_thread_pool(self):
        from src.orchestration.graph import _get_tool_executor

        ex = _get_tool_executor()
        assert isinstance(ex, concurrent.futures.ThreadPoolExecutor)

    def test_user_cancelled_run_propagates(self):
        """UserCancelledRun raised in a task submitted to the module-level executor propagates."""
        from src.agent.safety import UserCancelledRun
        from src.orchestration.graph import _get_tool_executor

        def _cancelling():
            raise UserCancelledRun()

        pool = _get_tool_executor()
        future = pool.submit(_cancelling)
        with pytest.raises(UserCancelledRun):
            future.result()

    def test_user_cancelled_run_raised_after_all_futures(self):
        """When the first future raises UserCancelledRun, remaining futures still
        complete and the exception is detected on result() inspection."""
        from src.agent.safety import UserCancelledRun
        from src.orchestration.graph import _get_tool_executor

        def fast_cancel():
            raise UserCancelledRun()

        def slow_ok():
            time.sleep(0.05)
            return "done"

        pool = _get_tool_executor()
        f1 = pool.submit(fast_cancel)
        f2 = pool.submit(slow_ok)

        cancelled = False
        try:
            f1.result()
        except UserCancelledRun:
            cancelled = True

        # f2 should still complete even though f1 raised.
        result2 = f2.result()
        assert cancelled
        assert result2 == "done"
