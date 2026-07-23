"""Tests for src/tasks/queue.py — SQLite-backed background task queue."""

from __future__ import annotations

import threading
import time

import pytest

from src.tasks.queue import (
    TaskQueue,
    TaskStatus,
    cancel_task,
    get_task,
    get_task_queue,
    init_task_queue,
    list_tasks,
    submit_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def queue(tmp_path):
    q = TaskQueue(
        db_path=tmp_path / "tasks.db",
        log_dir=tmp_path / "logs",
    )
    yield q
    # Ensure executor is shut down if start() was called.
    if q._executor is not None:
        q.stop()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton between tests."""
    import src.tasks.queue as _mod

    original = _mod._queue
    yield
    _mod._queue = original


# ---------------------------------------------------------------------------
# submit / get
# ---------------------------------------------------------------------------


class TestSubmitGet:
    def test_submit_returns_string_id(self, queue):
        task_id = queue.submit("my_agent", "do something")
        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID4

    def test_submit_creates_pending_record(self, queue):
        task_id = queue.submit("my_agent", "do something")
        record = queue.get(task_id)
        assert record is not None
        assert record.task_id == task_id
        assert record.agent_name == "my_agent"
        assert record.prompt == "do something"
        assert record.status == TaskStatus.PENDING
        assert record.result == ""
        assert record.error == ""

    def test_get_unknown_returns_none(self, queue):
        assert queue.get("nonexistent-id") is None

    def test_log_path_set_on_submit(self, queue):
        task_id = queue.submit("agent", "prompt")
        record = queue.get(task_id)
        assert record is not None
        assert task_id in record.log_path
        assert record.log_path.endswith(".log")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_returns_all_records(self, queue):
        ids = [queue.submit("a", f"prompt {i}") for i in range(3)]
        records = queue.list()
        assert len(records) == 3
        returned_ids = {r.task_id for r in records}
        assert returned_ids == set(ids)

    def test_list_newest_first(self, queue):
        queue.submit("a", "first")
        time.sleep(0.01)
        queue.submit("a", "second")
        records = queue.list()
        assert records[0].prompt == "second"
        assert records[1].prompt == "first"

    def test_list_filtered_by_status(self, queue):
        id1 = queue.submit("a", "one")
        id2 = queue.submit("a", "two")
        queue.cancel(id1)

        pending = queue.list(status=TaskStatus.PENDING)
        cancelled = queue.list(status=TaskStatus.CANCELLED)

        assert len(pending) == 1
        assert pending[0].task_id == id2

        assert len(cancelled) == 1
        assert cancelled[0].task_id == id1

    def test_list_filter_returns_empty_when_no_match(self, queue):
        queue.submit("a", "p")
        assert queue.list(status=TaskStatus.COMPLETED) == []

    def test_list_limit_respected(self, queue):
        for i in range(10):
            queue.submit("a", f"p{i}")
        assert len(queue.list(limit=5)) == 5


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_pending_returns_true(self, queue):
        task_id = queue.submit("a", "p")
        assert queue.cancel(task_id) is True
        record = queue.get(task_id)
        assert record is not None
        assert record.status == TaskStatus.CANCELLED

    def test_cancel_unknown_returns_false(self, queue):
        assert queue.cancel("does-not-exist") is False

    def test_cancel_completed_returns_false(self, queue):
        # Manually insert a COMPLETED record.
        task_id = queue.submit("a", "p")
        with queue._lock:
            with queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (TaskStatus.COMPLETED.value, task_id),
                )
        assert queue.cancel(task_id) is False

    def test_cancel_running_returns_false(self, queue):
        task_id = queue.submit("a", "p")
        with queue._lock:
            with queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (TaskStatus.RUNNING.value, task_id),
                )
        assert queue.cancel(task_id) is False


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_stop_no_tasks(self, queue):
        queue.start()
        queue.stop()
        assert queue._executor is None

    def test_start_stop_with_pending_tasks(self, queue):
        queue.submit("a", "p1")
        queue.submit("a", "p2")
        queue.start()
        queue.stop()

    def test_stop_without_start_is_safe(self, queue):
        queue.stop()  # should not raise


# ---------------------------------------------------------------------------
# Task execution (end-to-end with real threads)
# ---------------------------------------------------------------------------


class TestExecution:
    def test_submitted_task_reaches_completed(self, tmp_path):
        q = TaskQueue(tmp_path / "tasks.db", tmp_path / "logs")
        q.start()
        try:
            done = threading.Event()

            def _poll():
                for _ in range(50):
                    r = q.get(task_id)
                    if r and r.status == TaskStatus.COMPLETED:
                        done.set()
                        return
                    time.sleep(0.1)

            task_id = q.submit("test_agent", "hello world")
            t = threading.Thread(target=_poll, daemon=True)
            t.start()
            assert done.wait(timeout=10), "Task did not reach COMPLETED within 10 s"

            record = q.get(task_id)
            assert record is not None
            assert record.status == TaskStatus.COMPLETED
            assert record.result != ""
            assert record.started_at is not None
            assert record.finished_at is not None
        finally:
            q.stop()

    def test_result_contains_stub_text(self, tmp_path):
        q = TaskQueue(tmp_path / "tasks.db", tmp_path / "logs")
        q.start()
        try:
            task_id = q.submit("my_agent", "do the thing")
            for _ in range(50):
                r = q.get(task_id)
                if r and r.status == TaskStatus.COMPLETED:
                    break
                time.sleep(0.1)
            record = q.get(task_id)
            assert record is not None
            assert "[stub]" in record.result
            assert "my_agent" in record.result
        finally:
            q.stop()

    def test_cancelled_task_not_executed(self, tmp_path):
        q = TaskQueue(tmp_path / "tasks.db", tmp_path / "logs")
        # Do NOT start the executor — cancel before any worker picks it up.
        task_id = q.submit("a", "p")
        assert q.cancel(task_id) is True
        q.start()
        q.stop()
        record = q.get(task_id)
        assert record is not None
        assert record.status == TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# _append_log
# ---------------------------------------------------------------------------


class TestAppendLog:
    def test_append_creates_file(self, queue):
        task_id = queue.submit("a", "p")
        queue._append_log(task_id, "line one")
        log = queue._log_dir / f"{task_id}.log"
        assert log.exists()
        assert log.read_text(encoding="utf-8") == "line one\n"

    def test_append_accumulates_lines(self, queue):
        task_id = queue.submit("a", "p")
        queue._append_log(task_id, "alpha")
        queue._append_log(task_id, "beta")
        log = queue._log_dir / f"{task_id}.log"
        lines = log.read_text(encoding="utf-8").splitlines()
        assert lines == ["alpha", "beta"]

    def test_log_written_after_completion(self, tmp_path):
        q = TaskQueue(tmp_path / "tasks.db", tmp_path / "logs")
        q.start()
        try:
            task_id = q.submit("a", "prompt")
            for _ in range(50):
                r = q.get(task_id)
                if r and r.status == TaskStatus.COMPLETED:
                    break
                time.sleep(0.1)
            log = tmp_path / "logs" / f"{task_id}.log"
            assert log.exists()
            content = log.read_text(encoding="utf-8")
            assert "[COMPLETED]" in content
        finally:
            q.stop()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_task_queue_raises_before_init(self):
        import src.tasks.queue as _mod

        _mod._queue = None
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_task_queue()

    def test_init_task_queue_returns_instance(self, tmp_path):
        q = init_task_queue(tmp_path / "db.db", tmp_path / "logs")
        assert isinstance(q, TaskQueue)

    def test_module_helpers_use_singleton(self, tmp_path):
        init_task_queue(tmp_path / "db.db", tmp_path / "logs")
        task_id = submit_task("agent", "hello")
        record = get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.PENDING

        records = list_tasks()
        assert any(r.task_id == task_id for r in records)

        assert cancel_task(task_id) is True
        assert get_task(task_id).status == TaskStatus.CANCELLED  # type: ignore[union-attr]

    def test_list_tasks_filter_via_singleton(self, tmp_path):
        init_task_queue(tmp_path / "db.db", tmp_path / "logs")
        submit_task("a", "one")
        tid = submit_task("a", "two")
        cancel_task(tid)

        assert len(list_tasks(status=TaskStatus.PENDING)) == 1
        assert len(list_tasks(status=TaskStatus.CANCELLED)) == 1
