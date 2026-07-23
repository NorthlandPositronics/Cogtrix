"""Regression tests for #2061 — the background-summarizer single-flight guard.

The ``_bg_future`` check and the pool ``submit`` used to run outside
``_hybrid_lock``, so two concurrent ``_schedule_slow_path`` callers could both
observe no in-flight job and both submit — the duplicate jobs then raced on
``_summary_msg_idx`` (last-writer-wins could rewind the summary boundary). The
guard + submit must now be atomic under the lock.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from cogtrix_core.memory.modes.conversation import ConversationMemoryManager


class _MockStore:
    def __init__(self) -> None:
        self.data: dict = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages) -> None:
        self.data[session_id] = list(messages)


def _manager() -> ConversationMemoryManager:
    mgr = ConversationMemoryManager(_MockStore(), "s")
    mgr._llm = MagicMock()  # truthy -> scheduling is enabled
    return mgr


def _messages(n: int = 30) -> list[HumanMessage]:
    # Meaningful human messages so the summarization gate passes.
    return [HumanMessage(content=f"meaningful conversation message number {i}") for i in range(n)]


def _running_future() -> Future:
    f: Future = Future()
    f.set_running_or_notify_cancel()  # RUNNING -> not done()
    return f


def test_in_flight_job_skips_resubmit() -> None:
    """A second schedule while a job is in flight must not submit again."""
    mgr = _manager()
    mgr._bg_future = _running_future()
    mgr._bg_submitted_at = time.monotonic()
    fake_pool = MagicMock()
    with patch("cogtrix_core.memory.manager._get_summarization_pool", return_value=fake_pool):
        mgr._schedule_slow_path(_messages(), window_size=5)
    fake_pool.submit.assert_not_called()


def test_concurrent_schedule_submits_exactly_one_job() -> None:
    """Two concurrent schedule calls must submit exactly one background job."""
    mgr = _manager()
    messages = _messages()

    submit_count = 0
    count_lock = threading.Lock()

    def fake_submit(fn, *a, **k):
        nonlocal submit_count
        with count_lock:
            submit_count += 1
        return _running_future()

    fake_pool = MagicMock()
    fake_pool.submit.side_effect = fake_submit
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        mgr._schedule_slow_path(messages, window_size=5)

    with patch("cogtrix_core.memory.manager._get_summarization_pool", return_value=fake_pool):
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert submit_count == 1, f"single-flight violated: {submit_count} jobs submitted"
