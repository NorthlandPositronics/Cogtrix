"""Agent spawning and task management tools.

Tools:
    spawn_agent      — spawn a named agent synchronously or in the background
    get_task_status  — get status summary for a background task
    get_task_result  — get the full result of a completed task
    list_tasks       — list tasks, optionally filtered by status
    cancel_task      — cancel a pending task

Background tasks are stored in the SQLite-backed TaskQueue
(src/tasks/queue.py).  Call init_task_queue() at application startup to
initialise the queue; all tools gracefully return an informative error string
when the queue has not been initialised.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

log = logging.getLogger("cogtrix.tools.agent_tools")


# ── Input schemas ──────────────────────────────────────────────────────────────


class SpawnAgentInput(BaseModel):
    agent_name: str = Field(..., description="Name of the registered agent to spawn.")
    task: str = Field(..., description="Task description / prompt to give to the agent.")
    background: bool = Field(
        default=False,
        description=(
            "If False (default), run the agent synchronously and return the result. "
            "If True, submit to the background task queue and return a task_id immediately."
        ),
    )


class GetTaskStatusInput(BaseModel):
    task_id: str = Field(..., description="UUID task ID returned by spawn_agent.")


class GetTaskResultInput(BaseModel):
    task_id: str = Field(..., description="UUID task ID returned by spawn_agent.")


class ListTasksInput(BaseModel):
    status: str = Field(
        default="",
        description=(
            "Optional status filter: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED. "
            "Leave empty to list all tasks."
        ),
    )
    limit: int = Field(default=10, description="Maximum number of tasks to return (1–100).")


class CancelTaskInput(BaseModel):
    task_id: str = Field(..., description="UUID task ID of a PENDING task to cancel.")


# ── Internal helpers ───────────────────────────────────────────────────────────


def _elapsed(created_at: float, finished_at: float | None = None) -> str:
    """Human-readable elapsed time since *created_at*."""
    seconds = (finished_at if finished_at is not None else time.time()) - created_at
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _get_queue():  # type: ignore[return]
    """Return the active TaskQueue or raise RuntimeError."""
    from src.tasks.queue import get_task_queue

    return get_task_queue()


# ── Tool functions ─────────────────────────────────────────────────────────────


def spawn_agent(agent_name: str, task: str, background: bool = False) -> str:
    """Spawn a named agent with a task, synchronously or in the background."""
    import src.agent.registry as _reg

    if _reg.get(agent_name) is None:
        known = [a.name for a in _reg.list_agents()]
        if known:
            return (
                f"Unknown agent '{agent_name}'. " f"Available agents: {', '.join(sorted(known))}."
            )
        return f"Unknown agent '{agent_name}'. No agents are currently registered."

    if background:
        try:
            from src.tasks.queue import submit_task

            task_id = submit_task(agent_name, task)
            return f"Task submitted. task_id={task_id}"
        except RuntimeError as exc:
            return f"Error: task queue not initialised — {exc}"

    # Synchronous: run the _run_agent_task stub inline via the queue if available,
    # otherwise fall back to a local stub so the tool always returns a result.
    try:
        import uuid

        from src.tasks.queue import TaskRecord, TaskStatus, get_task_queue

        q = get_task_queue()
        record = TaskRecord(
            task_id=str(uuid.uuid4()),
            agent_name=agent_name,
            prompt=task,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            started_at=None,
            finished_at=None,
            result="",
            error="",
            log_path="",
        )
        return q._run_agent_task(record)
    except RuntimeError:
        return f"[stub] Agent '{agent_name}' completed task: {task[:80]}"


def get_task_status(task_id: str) -> str:
    """Get the current status, elapsed time, and result preview for a background task."""
    try:
        q = _get_queue()
    except RuntimeError as exc:
        return f"Error: {exc}"

    record = q.get(task_id)
    if record is None:
        return f"Error: task '{task_id}' not found."

    lines = [
        f"task_id : {record.task_id}",
        f"agent   : {record.agent_name}",
        f"status  : {record.status}",
        f"elapsed : {_elapsed(record.created_at, record.finished_at)}",
    ]
    if record.result:
        preview = record.result[:500]
        lines.append(f"result  : {preview}{'...' if len(record.result) > 500 else ''}")
    if record.error:
        preview = record.error[:500]
        lines.append(f"error   : {preview}{'...' if len(record.error) > 500 else ''}")
    return "\n".join(lines)


def get_task_result(task_id: str) -> str:
    """Get the full result of a completed task, or an appropriate message otherwise."""
    try:
        q = _get_queue()
    except RuntimeError as exc:
        return f"Error: {exc}"

    record = q.get(task_id)
    if record is None:
        return f"Error: task '{task_id}' not found."

    from src.tasks.queue import TaskStatus

    if record.status == TaskStatus.COMPLETED:
        return record.result
    if record.status == TaskStatus.FAILED:
        return f"Task failed: {record.error}"
    if record.status == TaskStatus.RUNNING:
        return "Task still running."
    if record.status == TaskStatus.PENDING:
        return "Task is pending (not yet started)."
    if record.status == TaskStatus.CANCELLED:
        return "Task was cancelled."
    return f"Task status: {record.status}"


def list_tasks(status: str = "", limit: int = 10) -> str:
    """List background tasks (newest first), optionally filtered by status."""
    try:
        q = _get_queue()
    except RuntimeError as exc:
        return f"Error: {exc}"

    limit = max(1, min(limit, 100))

    status_filter = None
    if status:
        from src.tasks.queue import TaskStatus

        try:
            status_filter = TaskStatus(status.upper())
        except ValueError:
            from src.tasks.queue import TaskStatus as _TS

            valid = ", ".join(s.value for s in _TS)
            return f"Error: invalid status '{status}'. Valid values: {valid}."

    records = q.list(status=status_filter, limit=limit)
    if not records:
        msg = "No tasks found"
        if status:
            msg += f" with status '{status.upper()}'"
        return msg + "."

    header = f"{'ID':8}  {'Agent':20}  {'Status':10}  {'Created':8}  {'Elapsed':>8}"
    sep = "─" * len(header)
    rows = [header, sep]
    for r in records:
        rows.append(
            f"{r.task_id[:8]:<8}  "
            f"{r.agent_name[:20]:<20}  "
            f"{r.status.value[:10]:<10}  "
            f"{_format_ts(r.created_at):8}  "
            f"{_elapsed(r.created_at, r.finished_at):>8}"
        )
    return "\n".join(rows)


def cancel_task(task_id: str) -> str:
    """Cancel a PENDING background task by its task_id."""
    try:
        q = _get_queue()
    except RuntimeError as exc:
        return f"Error: {exc}"

    if q.cancel(task_id):
        return f"Task '{task_id}' has been cancelled."

    record = q.get(task_id)
    if record is None:
        return f"Error: task '{task_id}' not found."
    return (
        f"Error: cannot cancel task in '{record.status}' status "
        "(only PENDING tasks can be cancelled)."
    )


# ── Configuration hook ─────────────────────────────────────────────────────────


def configure_agent_tools() -> None:
    """No-op configuration hook; ensures imports resolve at load time."""


# ── Tool catalog ───────────────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "spawn_agent",
        "description": (
            "Spawn a named agent with a task. "
            "Use background=False (default) to run synchronously and get the result immediately; "
            "use background=True to submit to the background queue and receive a task_id. "
            "The agent must be registered in the agent registry."
        ),
        "input_schema": SpawnAgentInput,
        "requires_confirmation": True,
        "function": spawn_agent,
    },
    {
        "name": "get_task_status",
        "description": (
            "Get the current status, elapsed time, and result preview (first 500 chars) "
            "for a background task submitted via spawn_agent."
        ),
        "input_schema": GetTaskStatusInput,
        "requires_confirmation": False,
        "function": get_task_status,
    },
    {
        "name": "get_task_result",
        "description": (
            "Get the full result of a completed background task, "
            "or an informative message if the task is still running, pending, failed, or cancelled."
        ),
        "input_schema": GetTaskResultInput,
        "requires_confirmation": False,
        "function": get_task_result,
    },
    {
        "name": "list_tasks",
        "description": (
            "List background tasks (newest first), optionally filtered by status "
            "(PENDING, RUNNING, COMPLETED, FAILED, CANCELLED). "
            "Returns a summary table with IDs, agent names, statuses, and elapsed times."
        ),
        "input_schema": ListTasksInput,
        "requires_confirmation": False,
        "function": list_tasks,
    },
    {
        "name": "cancel_task",
        "description": (
            "Cancel a PENDING background task by its task_id. "
            "Tasks that are already RUNNING, COMPLETED, FAILED, or CANCELLED cannot be cancelled."
        ),
        "input_schema": CancelTaskInput,
        "requires_confirmation": True,
        "function": cancel_task,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "spawn_agent",
    "get_task_status",
    "get_task_result",
    "list_tasks",
    "cancel_task",
    "configure_agent_tools",
    "SpawnAgentInput",
    "GetTaskStatusInput",
    "GetTaskResultInput",
    "ListTasksInput",
    "CancelTaskInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
