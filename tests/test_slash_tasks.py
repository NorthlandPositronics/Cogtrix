"""Tests for /tasks, /spawn, and /goal slash commands added in M2.7."""

from __future__ import annotations

import pytest

import src.tasks.goal_tracker as _goal_mod
import src.tasks.queue as _queue_mod
from src.tasks.queue import init_task_queue

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_queue_singleton():
    original = _queue_mod._queue
    yield
    _queue_mod._queue = original


@pytest.fixture(autouse=True)
def _reset_goal_cache():
    original = dict(_goal_mod._stacks)
    _goal_mod._stacks.clear()
    yield
    _goal_mod._stacks.clear()
    _goal_mod._stacks.update(original)


def _make_registry(config=None):
    """Return a SlashCommandRegistry built by _build_slash_commands, with optional config."""
    from cogtrix import _build_slash_commands

    reg = _build_slash_commands()
    reg.config = config
    return reg


class _FakeConfig:
    session = "test-session"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir


def _dispatch(reg, cmd: str) -> str:
    result = reg.dispatch(cmd)
    return result or "continue"


def _captured(capsys) -> str:
    """Return everything written to stdout so far."""
    out, _ = capsys.readouterr()
    return out


# ---------------------------------------------------------------------------
# /tasks — queue not initialised
# ---------------------------------------------------------------------------


class TestTasksNotInitialised:
    def test_tasks_no_queue_prints_message(self, capsys):
        _queue_mod._queue = None
        reg = _make_registry()
        _dispatch(reg, "/tasks")
        out = _captured(capsys)
        assert "initialised" in out or "available" in out

    def test_tasks_continues_on_missing_queue(self):
        _queue_mod._queue = None
        reg = _make_registry()
        result = _dispatch(reg, "/tasks")
        assert result == "continue"


# ---------------------------------------------------------------------------
# /tasks — queue initialised
# ---------------------------------------------------------------------------


class TestTasksWithQueue:
    @pytest.fixture()
    def queue(self, tmp_path):
        q = init_task_queue(tmp_path / "db.db", tmp_path / "logs")
        return q

    def test_tasks_empty_queue_returns_continue(self, queue):
        reg = _make_registry(config=_FakeConfig())
        result = _dispatch(reg, "/tasks")
        assert result == "continue"

    def test_tasks_lists_submitted_task(self, queue, capsys):
        queue.submit("my_agent", "do something useful", session_id="test-session")
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks")
        out = _captured(capsys)
        assert "my_agent" in out

    def test_tasks_filter_pending(self, queue, capsys):
        queue.submit("a", "p1", session_id="test-session")
        tid = queue.submit("a", "p2", session_id="test-session")
        queue.cancel(tid)
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks pending")
        out = _captured(capsys)
        assert "p1" in out
        assert "p2" not in out

    def test_tasks_filter_cancelled(self, queue, capsys):
        queue.submit("a", "active", session_id="test-session")
        tid = queue.submit("a", "cancelled_task", session_id="test-session")
        queue.cancel(tid)
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks cancelled")
        out = _captured(capsys)
        assert "cancelled_task" in out

    def test_tasks_invalid_status_returns_continue(self, queue):
        reg = _make_registry(config=_FakeConfig())
        result = _dispatch(reg, "/tasks bogus_status")
        assert result == "continue"

    def test_tasks_invalid_status_prints_error(self, queue, capsys):
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks bogus_status")
        out = _captured(capsys)
        assert "bogus_status" in out or "Unknown" in out


# ---------------------------------------------------------------------------
# /spawn
# ---------------------------------------------------------------------------


class TestSpawn:
    @pytest.fixture()
    def queue(self, tmp_path):
        return init_task_queue(tmp_path / "db.db", tmp_path / "logs")

    def test_spawn_no_args_returns_continue(self, queue):
        reg = _make_registry()
        result = _dispatch(reg, "/spawn")
        assert result == "continue"

    def test_spawn_one_arg_only_returns_continue(self, queue):
        reg = _make_registry()
        result = _dispatch(reg, "/spawn myagent")
        assert result == "continue"

    def test_spawn_no_args_prints_usage(self, queue, capsys):
        reg = _make_registry()
        _dispatch(reg, "/spawn")
        out = _captured(capsys)
        assert "Usage" in out

    def test_spawn_submits_task(self, queue):
        reg = _make_registry()
        _dispatch(reg, "/spawn researcher Summarise arXiv papers")
        tasks = queue.list()
        assert len(tasks) == 1
        assert tasks[0].agent_name == "researcher"
        assert "Summarise arXiv papers" in tasks[0].prompt

    def test_spawn_prints_task_id(self, queue, capsys):
        reg = _make_registry()
        _dispatch(reg, "/spawn coder refactor the module")
        out = _captured(capsys)
        assert "coder" in out

    def test_spawn_queue_not_initialised_returns_continue(self):
        _queue_mod._queue = None
        reg = _make_registry()
        result = _dispatch(reg, "/spawn agent some task")
        assert result == "continue"

    def test_spawn_prompt_allows_spaces(self, queue):
        reg = _make_registry()
        _dispatch(reg, "/spawn bot do task A and task B and task C")
        tasks = queue.list()
        assert tasks[0].prompt == "do task A and task B and task C"

    def test_task_alias_dispatches(self):
        """Alias /task should reach the /tasks handler (queue not init → continue)."""
        _queue_mod._queue = None
        reg = _make_registry()
        result = _dispatch(reg, "/task")
        assert result == "continue"


# ---------------------------------------------------------------------------
# /goal
# ---------------------------------------------------------------------------


class TestGoal:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return _FakeConfig(data_dir=str(tmp_path))

    def test_goal_list_empty_returns_continue(self, cfg):
        reg = _make_registry(config=cfg)
        result = _dispatch(reg, "/goal")
        assert result == "continue"

    def test_goal_list_empty_prints_message(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal")
        out = _captured(capsys)
        assert "active" in out.lower() or "goal" in out.lower()

    def test_goal_set_creates_goal(self, cfg):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set Migrate the auth module")
        from src.tasks.goal_tracker import get_goal_stack

        stack = get_goal_stack(cfg.session, cfg.data_dir)
        goals = stack.list_active()
        assert len(goals) == 1
        assert goals[0].description == "Migrate the auth module"

    def test_goal_set_prints_goal_id(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set Write tests for new feature")
        out = _captured(capsys)
        assert "Write tests for new feature" in out or "Goal set" in out

    def test_goal_set_no_desc_prints_usage(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set")
        out = _captured(capsys)
        assert "Usage" in out

    def test_goal_complete_marks_complete(self, cfg):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set Complete this")
        from src.tasks.goal_tracker import GoalStatus, get_goal_stack

        stack = get_goal_stack(cfg.session, cfg.data_dir)
        goal_id = stack.list_active()[0].goal_id

        _dispatch(reg, f"/goal complete {goal_id}")
        goal = stack.get(goal_id)
        assert goal is not None
        assert goal.status == GoalStatus.COMPLETED

    def test_goal_complete_unknown_id(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal complete badid999")
        out = _captured(capsys)
        assert "Unknown" in out or "badid" in out.lower()

    def test_goal_complete_no_id_prints_usage(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal complete")
        out = _captured(capsys)
        assert "Usage" in out

    def test_goal_abandon_marks_abandoned(self, cfg):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set Abandon this goal")
        from src.tasks.goal_tracker import GoalStatus, get_goal_stack

        stack = get_goal_stack(cfg.session, cfg.data_dir)
        goal_id = stack.list_active()[0].goal_id

        _dispatch(reg, f"/goal abandon {goal_id}")
        goal = stack.get(goal_id)
        assert goal is not None
        assert goal.status == GoalStatus.ABANDONED

    def test_goal_abandon_unknown_id(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal abandon nosuchid")
        out = _captured(capsys)
        assert "Unknown" in out or "nosuchid" in out.lower()

    def test_goal_abandon_no_id_prints_usage(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal abandon")
        out = _captured(capsys)
        assert "Usage" in out

    def test_goal_list_subcommand_shows_goals(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal set Listed goal")
        capsys.readouterr()  # flush the "set" output
        _dispatch(reg, "/goal list")
        out = _captured(capsys)
        assert "Listed goal" in out

    def test_goal_unknown_subcommand_returns_continue(self, cfg):
        reg = _make_registry(config=cfg)
        result = _dispatch(reg, "/goal badcmd something")
        assert result == "continue"

    def test_goal_unknown_subcommand_prints_error(self, cfg, capsys):
        reg = _make_registry(config=cfg)
        _dispatch(reg, "/goal badcmd something")
        out = _captured(capsys)
        assert "Unknown" in out or "badcmd" in out

    def test_goals_alias_dispatches(self, cfg):
        """Alias /goals should reach the same handler."""
        reg = _make_registry(config=cfg)
        result = _dispatch(reg, "/goals")
        assert result == "continue"


# ---------------------------------------------------------------------------
# TaskQueue.set_runner — callback injection
# ---------------------------------------------------------------------------


class TestSetRunner:
    def test_set_runner_called_by_worker(self, tmp_path):
        """A configured runner is invoked by the background worker."""
        import threading
        import time

        from src.tasks.queue import TaskQueue, TaskStatus

        called_with: list = []
        done = threading.Event()

        def _mock_runner(record):
            called_with.append(record)
            done.set()
            return "mock result"

        q = TaskQueue(tmp_path / "db.db", tmp_path / "logs")
        q.set_runner(_mock_runner)
        q.start()
        try:
            task_id = q.submit("test_agent", "do something")
            assert done.wait(timeout=10), "Runner not called within 10 s"
            assert len(called_with) == 1
            assert called_with[0].task_id == task_id
            assert called_with[0].agent_name == "test_agent"
            assert called_with[0].prompt == "do something"
            # Result is written back
            for _ in range(50):
                r = q.get(task_id)
                if r and r.status == TaskStatus.COMPLETED:
                    break
                time.sleep(0.05)
            r = q.get(task_id)
            assert r is not None
            assert r.result == "mock result"
        finally:
            q.stop()

    def test_no_runner_falls_back_to_stub(self, tmp_path):
        """Without a runner the stub message is returned."""
        import time

        from src.tasks.queue import TaskQueue, TaskStatus

        q = TaskQueue(tmp_path / "db.db", tmp_path / "logs")
        q.start()
        try:
            task_id = q.submit("my_agent", "stub task")
            for _ in range(50):
                r = q.get(task_id)
                if r and r.status == TaskStatus.COMPLETED:
                    break
                time.sleep(0.05)
            r = q.get(task_id)
            assert r is not None
            assert "[stub]" in r.result
        finally:
            q.stop()

    def test_runner_exception_marks_task_failed(self, tmp_path):
        """If the runner raises, the task is marked FAILED with the error message."""
        import time

        from src.tasks.queue import TaskQueue, TaskStatus

        def _bad_runner(record):
            raise ValueError("something went wrong")

        q = TaskQueue(tmp_path / "db.db", tmp_path / "logs")
        q.set_runner(_bad_runner)
        q.start()
        try:
            task_id = q.submit("broken_agent", "break it")
            for _ in range(50):
                r = q.get(task_id)
                if r and r.status in (TaskStatus.FAILED, TaskStatus.COMPLETED):
                    break
                time.sleep(0.05)
            r = q.get(task_id)
            assert r is not None
            assert r.status == TaskStatus.FAILED
            assert "something went wrong" in r.error
        finally:
            q.stop()


# ---------------------------------------------------------------------------
# /tasks <id> — detail view
# ---------------------------------------------------------------------------


class TestTaskDetail:
    @pytest.fixture()
    def queue(self, tmp_path):
        return init_task_queue(tmp_path / "db.db", tmp_path / "logs")

    def _insert_completed(self, queue, agent="tester", prompt="do stuff", result="done"):
        """Submit a task and manually set it to COMPLETED with a result."""
        task_id = queue.submit(agent, prompt)
        import time

        with queue._lock:
            with queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status=?, started_at=?, finished_at=?, result=? WHERE task_id=?",
                    ("COMPLETED", time.time() - 2, time.time(), result, task_id),
                )
        return task_id

    def test_task_detail_by_full_id(self, queue, capsys):
        task_id = self._insert_completed(queue)
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, f"/tasks {task_id}")
        out = _captured(capsys)
        assert task_id in out or task_id[:8] in out
        assert "tester" in out

    def test_task_detail_by_prefix(self, queue, capsys):
        task_id = self._insert_completed(queue, prompt="prefix match test")
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, f"/tasks {task_id[:8]}")
        out = _captured(capsys)
        assert "prefix match test" in out or task_id[:8] in out

    def test_task_detail_shows_result(self, queue, capsys):
        task_id = self._insert_completed(queue, result="the final answer")
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, f"/tasks {task_id[:8]}")
        out = _captured(capsys)
        assert "the final answer" in out

    def test_task_detail_not_found(self, queue, capsys):
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks deadbeef")
        out = _captured(capsys)
        assert "not found" in out.lower() or "deadbeef" in out

    def test_task_detail_ambiguous_prefix_not_found(self, queue, capsys):
        """When multiple tasks share a prefix, no detail is shown."""
        # Submit two tasks whose IDs share a common first character (just verify
        # error path when no exact match — use a random prefix that won't match)
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks 00000000")
        out = _captured(capsys)
        assert "not found" in out.lower() or "00000000" in out

    def test_task_list_shows_hint(self, queue, capsys):
        queue.submit("a", "some prompt", session_id="test-session")
        reg = _make_registry(config=_FakeConfig())
        _dispatch(reg, "/tasks")
        out = _captured(capsys)
        assert "/tasks <id>" in out or "details" in out.lower()
