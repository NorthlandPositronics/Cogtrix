"""Tests for src/tools/agent_tools.py — agent spawning and task management tools."""

from __future__ import annotations

import pytest

import src.agent.registry as _reg_mod
import src.tasks.queue as _queue_mod
from src.agent.registry import AgentConfig, AgentRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Give each test a clean, isolated agent registry."""
    original = _reg_mod._registry
    _reg_mod._registry = AgentRegistry()
    yield
    _reg_mod._registry = original


@pytest.fixture(autouse=True)
def _reset_queue():
    """Reset the task-queue singleton after every test."""
    original = _queue_mod._queue
    yield
    _queue_mod._queue = original


@pytest.fixture()
def queue(tmp_path):
    """Initialise a real SQLite-backed TaskQueue for the test."""
    from src.tasks.queue import init_task_queue

    return init_task_queue(
        db_path=tmp_path / "tasks.db",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture()
def agent():
    """Register a single test agent in the module-level registry."""
    _reg_mod._registry.register(AgentConfig(name="test_agent", description="A unit-test agent"))


# ---------------------------------------------------------------------------
# spawn_agent
# ---------------------------------------------------------------------------


class TestSpawnAgent:
    def test_background_false_returns_stub_result(self, agent) -> None:
        """Synchronous spawn returns a stub string without a queue."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("test_agent", "do something")
        assert isinstance(result, str)
        assert len(result) > 0
        # Queue is not initialised → local stub
        assert "test_agent" in result

    def test_background_false_with_queue_returns_stub(self, agent, queue) -> None:
        """Synchronous spawn uses the queue's _run_agent_task stub when available."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("test_agent", "queue task", background=False)
        assert isinstance(result, str)
        assert "[stub]" in result
        assert "test_agent" in result

    def test_background_true_returns_task_id(self, agent, queue) -> None:
        """Background spawn submits to the queue and returns a task_id string."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("test_agent", "background task", background=True)
        assert isinstance(result, str)
        assert "task_id=" in result
        # Extract the UUID from the returned string
        task_id = result.split("task_id=")[-1].strip()
        assert len(task_id) == 36  # UUID4

    def test_background_true_no_queue_returns_error(self, agent) -> None:
        """Background spawn with no queue returns a descriptive error string."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("test_agent", "task", background=True)
        assert "Error" in result
        assert "queue" in result.lower() or "initialised" in result.lower()

    def test_unknown_agent_returns_error(self) -> None:
        """spawn_agent with an unregistered agent name returns an error string."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("ghost_agent", "some task")
        assert "Unknown agent" in result or "unknown agent" in result.lower()
        assert "ghost_agent" in result

    def test_unknown_agent_lists_available(self, agent) -> None:
        """Error message for unknown agent includes the list of known agents."""
        from src.tools.agent_tools import spawn_agent

        result = spawn_agent("nonexistent", "task")
        assert "test_agent" in result


# ---------------------------------------------------------------------------
# get_task_status
# ---------------------------------------------------------------------------


class TestGetTaskStatus:
    def test_known_task_returns_formatted_status(self, agent, queue) -> None:
        """get_task_status returns task_id, agent, status, and elapsed fields."""
        from src.tasks.queue import submit_task
        from src.tools.agent_tools import get_task_status

        task_id = submit_task("test_agent", "check me")
        result = get_task_status(task_id)
        assert "task_id" in result
        assert task_id in result
        assert "test_agent" in result
        assert "status" in result
        assert "elapsed" in result

    def test_unknown_task_id_returns_error(self, queue) -> None:
        """get_task_status returns an error string for an unknown task_id."""
        from src.tools.agent_tools import get_task_status

        result = get_task_status("00000000-0000-0000-0000-000000000000")
        assert "not found" in result.lower() or "Error" in result

    def test_no_queue_returns_error(self) -> None:
        """get_task_status returns an error string when queue is not initialised."""
        from src.tools.agent_tools import get_task_status

        result = get_task_status("any-id")
        assert "Error" in result

    def test_result_preview_truncated_to_500(self, agent, queue) -> None:
        """Result field is truncated to 500 chars with trailing ellipsis."""
        import src.tasks.queue as _qmod
        from src.tasks.queue import TaskStatus, submit_task
        from src.tools.agent_tools import get_task_status

        task_id = submit_task("test_agent", "p")
        # Manually set a long result
        with _qmod._queue._lock:
            with _qmod._queue._connect() as conn:
                long_result = "x" * 600
                conn.execute(
                    "UPDATE tasks SET status=?, result=? WHERE task_id=?",
                    (TaskStatus.COMPLETED.value, long_result, task_id),
                )
        result = get_task_status(task_id)
        assert "..." in result
        # The preview portion is ≤ 500 chars of 'x' chars + label prefix
        lines = {ln.split(":", 1)[0].strip(): ln for ln in result.splitlines()}
        result_line = lines.get("result", "")
        assert len(result_line) < 600


# ---------------------------------------------------------------------------
# get_task_result
# ---------------------------------------------------------------------------


class TestGetTaskResult:
    def test_completed_task_returns_full_result(self, agent, queue) -> None:
        """get_task_result returns the full result string for a COMPLETED task."""
        import src.tasks.queue as _qmod
        from src.tasks.queue import TaskStatus, submit_task
        from src.tools.agent_tools import get_task_result

        task_id = submit_task("test_agent", "finish me")
        the_result = "finished successfully"
        with _qmod._queue._lock:
            with _qmod._queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status=?, result=? WHERE task_id=?",
                    (TaskStatus.COMPLETED.value, the_result, task_id),
                )
        assert get_task_result(task_id) == the_result

    def test_failed_task_returns_error_message(self, agent, queue) -> None:
        import src.tasks.queue as _qmod
        from src.tasks.queue import TaskStatus, submit_task
        from src.tools.agent_tools import get_task_result

        task_id = submit_task("test_agent", "fail me")
        with _qmod._queue._lock:
            with _qmod._queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status=?, error=? WHERE task_id=?",
                    (TaskStatus.FAILED.value, "boom", task_id),
                )
        result = get_task_result(task_id)
        assert "failed" in result.lower()
        assert "boom" in result

    def test_running_task_returns_still_running(self, agent, queue) -> None:
        import src.tasks.queue as _qmod
        from src.tasks.queue import TaskStatus, submit_task
        from src.tools.agent_tools import get_task_result

        task_id = submit_task("test_agent", "run me")
        with _qmod._queue._lock:
            with _qmod._queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status=? WHERE task_id=?",
                    (TaskStatus.RUNNING.value, task_id),
                )
        assert "running" in get_task_result(task_id).lower()

    def test_pending_task_returns_pending_message(self, agent, queue) -> None:
        from src.tasks.queue import submit_task
        from src.tools.agent_tools import get_task_result

        task_id = submit_task("test_agent", "wait for me")
        result = get_task_result(task_id)
        assert "pending" in result.lower()

    def test_unknown_task_returns_error(self, queue) -> None:
        from src.tools.agent_tools import get_task_result

        result = get_task_result("00000000-0000-0000-0000-000000000000")
        assert "not found" in result.lower() or "Error" in result


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_returns_formatted_table(self, agent, queue) -> None:
        """list_tasks returns a header row and data rows."""
        from src.tasks.queue import submit_task
        from src.tools.agent_tools import list_tasks

        submit_task("test_agent", "task one")
        submit_task("test_agent", "task two")
        result = list_tasks()
        lines = result.splitlines()
        # Header + separator + at least two data rows
        assert len(lines) >= 4
        assert "Agent" in lines[0]
        assert "Status" in lines[0]

    def test_filter_by_status(self, agent, queue) -> None:
        """list_tasks with status='PENDING' returns only PENDING tasks."""
        from src.tasks.queue import submit_task
        from src.tools.agent_tools import list_tasks

        submit_task("test_agent", "pending task")
        result = list_tasks(status="PENDING")
        assert "PENDING" in result
        assert "Error" not in result

    def test_invalid_status_returns_error(self, queue) -> None:
        from src.tools.agent_tools import list_tasks

        result = list_tasks(status="BOGUS")
        assert "Error" in result
        assert "BOGUS" in result

    def test_empty_queue_returns_no_tasks_message(self, queue) -> None:
        from src.tools.agent_tools import list_tasks

        result = list_tasks()
        assert "No tasks" in result

    def test_no_queue_returns_error(self) -> None:
        from src.tools.agent_tools import list_tasks

        result = list_tasks()
        assert "Error" in result

    def test_limit_respected(self, agent, queue) -> None:
        """list_tasks(limit=1) returns at most one data row."""
        from src.tasks.queue import submit_task
        from src.tools.agent_tools import list_tasks

        submit_task("test_agent", "a")
        submit_task("test_agent", "b")
        submit_task("test_agent", "c")
        result = list_tasks(limit=1)
        # header + sep + 1 data row = 3 lines
        lines = [ln for ln in result.splitlines() if ln.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------


class TestCancelTask:
    def test_cancels_pending_task(self, agent, queue) -> None:
        """cancel_task returns confirmation for a PENDING task."""
        from src.tasks.queue import TaskStatus, get_task_queue, submit_task
        from src.tools.agent_tools import cancel_task

        task_id = submit_task("test_agent", "cancel me")
        result = cancel_task(task_id)
        assert "cancelled" in result.lower()
        assert get_task_queue().get(task_id).status == TaskStatus.CANCELLED

    def test_cannot_cancel_completed_task(self, agent, queue) -> None:
        import src.tasks.queue as _qmod
        from src.tasks.queue import TaskStatus, submit_task
        from src.tools.agent_tools import cancel_task

        task_id = submit_task("test_agent", "done task")
        with _qmod._queue._lock:
            with _qmod._queue._connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status=? WHERE task_id=?",
                    (TaskStatus.COMPLETED.value, task_id),
                )
        result = cancel_task(task_id)
        assert "Error" in result or "cannot" in result.lower()

    def test_unknown_task_returns_error(self, queue) -> None:
        from src.tools.agent_tools import cancel_task

        result = cancel_task("00000000-0000-0000-0000-000000000000")
        assert "not found" in result.lower() or "Error" in result

    def test_no_queue_returns_error(self) -> None:
        from src.tools.agent_tools import cancel_task

        result = cancel_task("any-id")
        assert "Error" in result


# ---------------------------------------------------------------------------
# TOOL_CONFIGS structure
# ---------------------------------------------------------------------------


class TestToolConfigs:
    def test_has_five_entries(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIGS

        assert len(TOOL_CONFIGS) == 5

    def test_spawn_agent_requires_confirmation(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIGS

        cfg = next(c for c in TOOL_CONFIGS if c["name"] == "spawn_agent")
        assert cfg["requires_confirmation"] is True

    def test_cancel_task_requires_confirmation(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIGS

        cfg = next(c for c in TOOL_CONFIGS if c["name"] == "cancel_task")
        assert cfg["requires_confirmation"] is True

    def test_read_only_tools_do_not_require_confirmation(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIGS

        no_confirm = {"get_task_status", "get_task_result", "list_tasks"}
        for cfg in TOOL_CONFIGS:
            if cfg["name"] in no_confirm:
                assert cfg["requires_confirmation"] is False, cfg["name"]

    def test_all_entries_have_required_keys(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIGS

        required = {"name", "description", "input_schema", "requires_confirmation", "function"}
        for cfg in TOOL_CONFIGS:
            assert required <= cfg.keys(), f"Missing keys in {cfg['name']}"

    def test_tool_config_alias_points_to_spawn_agent(self) -> None:
        from src.tools.agent_tools import TOOL_CONFIG, TOOL_CONFIGS

        assert TOOL_CONFIG is TOOL_CONFIGS[0]
        assert TOOL_CONFIG["name"] == "spawn_agent"
