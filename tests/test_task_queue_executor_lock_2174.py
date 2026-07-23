"""Regression tests for #2174 — TaskQueue touches the executor under the lock.

``submit()`` previously read ``self._executor`` and dispatched *outside*
``self._lock``, and ``stop()`` shut the executor down and nulled it with no
lock at all. A ``submit()`` racing a ``stop()`` could therefore hit
``RuntimeError: cannot schedule new futures after shutdown`` or silently orphan
a freshly-inserted PENDING task. The fix snapshots the executor under the lock
in ``submit()`` and nulls it under the lock in ``stop()``.
"""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from cogtrix_core.tasks.queue import TaskQueue, TaskStatus


@pytest.fixture()
def queue(tmp_path):
    q = TaskQueue(db_path=tmp_path / "tasks.db", log_dir=tmp_path / "logs")
    yield q
    if q._executor is not None:
        q.stop()


class _CountingLock:
    """Wraps a real lock and counts context-manager acquisitions."""

    def __init__(self, real):
        self._real = real
        self.acquires = 0

    def __enter__(self):
        self.acquires += 1
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)


def test_stop_nulls_executor_under_lock(queue) -> None:
    queue.start()
    tracker = _CountingLock(queue._lock)
    queue._lock = tracker

    before = tracker.acquires
    queue.stop()

    assert queue._executor is None
    assert tracker.acquires > before, "stop() must acquire _lock to null the executor"


def test_submit_snapshots_executor_before_releasing_lock(queue) -> None:
    """submit() must capture the executor *inside* the lock, so a concurrent
    stop() that nulls the field cannot make it skip dispatch (#2174)."""
    queue.start()
    live_executor = queue._executor
    assert live_executor is not None

    dispatched: list[str] = []

    def _recording_submit(_fn, task_id):
        dispatched.append(task_id)
        f: Future = Future()
        f.set_result(None)
        return f

    live_executor.submit = _recording_submit  # type: ignore[method-assign]

    real_lock = queue._lock

    class _NullingLock:
        """Simulates a concurrent stop() nulling _executor the instant the
        submit() critical section releases the lock."""

        def __enter__(self):
            return real_lock.__enter__()

        def __exit__(self, *exc):
            queue._executor = None
            return real_lock.__exit__(*exc)

    queue._lock = _NullingLock()
    try:
        queue.submit("agent", "do work")
    finally:
        queue._lock = real_lock
        queue._executor = live_executor

    assert dispatched, (
        "submit() must dispatch using an executor snapshotted under the lock; "
        "reading self._executor after releasing the lock races a concurrent stop()"
    )


def test_submit_after_stop_leaves_task_pending_without_raising(queue) -> None:
    """After stop(), submit() must not raise and must still persist the task as
    PENDING so a later start() can recover it (#2159 orphan recovery)."""
    queue.start()
    queue.stop()

    task_id = queue.submit("agent", "do work")

    rec = queue.get(task_id)
    assert rec is not None
    assert rec.status == TaskStatus.PENDING
