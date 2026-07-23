"""Goal tracking tools — set, complete, abandon, and list agent goals.

Tools:
    set_goal      — push a new top-level goal
    add_subgoal   — attach a child goal under an existing goal
    complete_goal — mark a goal COMPLETED
    abandon_goal  — mark a goal ABANDONED
    list_goals    — show all active goals

Configuration:
    TOOL_SETUP(config) is called automatically by ToolRegistry after this
    module is loaded.  It wires _session_id and _data_dir so goal persistence
    uses the correct session directory.  Do not call configure_goal_tools()
    from configure.py or cogtrix.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cogtrix_core.tools.delegate import register_tool_categories

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

from cogtrix_core.tasks.goal_tracker import get_goal_stack

if TYPE_CHECKING:
    from cogtrix_core.config import Config

# ── Module-level state (set by TOOL_SETUP) ────────────────────────────────────

_session_id: str = "default"
_data_dir: Path = Path("data")


# ── Configuration ─────────────────────────────────────────────────────────────


def configure_goal_tools(config: Config, session_id: str = "default") -> None:
    """Wire session_id and data_dir from *config*."""
    global _session_id, _data_dir
    _session_id = session_id
    _data_dir = Path(config.data_dir) if hasattr(config, "data_dir") else Path("data")


def TOOL_SETUP(config: Config) -> None:
    """Called automatically by ToolRegistry after loading this module."""
    configure_goal_tools(
        config,
        session_id=(
            config.default_session_id if hasattr(config, "default_session_id") else "default"
        ),
    )


# ── Input schemas ─────────────────────────────────────────────────────────────


class SetGoalInput(BaseModel):
    description: str = Field(..., description="Description of the top-level goal to track.")


class AddSubgoalInput(BaseModel):
    parent_id: str = Field(..., description="8-character ID of the parent goal.")
    description: str = Field(..., description="Description of the new subgoal.")


class CompleteGoalInput(BaseModel):
    goal_id: str = Field(..., description="8-character ID of the goal to mark completed.")


class AbandonGoalInput(BaseModel):
    goal_id: str = Field(..., description="8-character ID of the goal to abandon.")


class ListGoalsInput(BaseModel):
    pass


# ── Tool functions ────────────────────────────────────────────────────────────


def set_goal(description: str) -> str:
    """Set a new top-level goal for the current session."""
    stack = get_goal_stack(_session_id, _data_dir)
    goal_id = stack.push(description)
    return f"Goal set [{goal_id}]: {description}"


def add_subgoal(parent_id: str, description: str) -> str:
    """Add a child subgoal under an existing goal."""
    stack = get_goal_stack(_session_id, _data_dir)
    try:
        goal_id = stack.add_subgoal(parent_id, description)
    except KeyError:
        return f"Error: parent goal {parent_id!r} not found"
    return f"Subgoal added [{goal_id}] under [{parent_id}]"


def complete_goal(goal_id: str) -> str:
    """Mark a goal as completed."""
    stack = get_goal_stack(_session_id, _data_dir)
    if stack.complete(goal_id):
        return f"Goal [{goal_id}] marked as completed."
    return f"Error: goal {goal_id!r} not found"


def abandon_goal(goal_id: str) -> str:
    """Mark a goal as abandoned."""
    stack = get_goal_stack(_session_id, _data_dir)
    if stack.abandon(goal_id):
        return f"Goal [{goal_id}] marked as abandoned."
    return f"Error: goal {goal_id!r} not found"


def list_goals() -> str:
    """List all active goals and their subgoals."""
    stack = get_goal_stack(_session_id, _data_dir)
    prefix = stack.to_context_prefix()
    return prefix if prefix else "No active goals."


# ── Tool registry entries ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "set_goal",
        "description": (
            "Set a top-level goal to track during this session. "
            "Returns the goal ID that can be used to add subgoals or mark the goal complete."
        ),
        "input_schema": SetGoalInput,
        "requires_confirmation": False,
        "function": set_goal,
        "category": "mutation",
    },
    {
        "name": "add_subgoal",
        "description": (
            "Add a child subgoal under an existing goal. "
            "Provide the parent's 8-character goal ID and a description."
        ),
        "input_schema": AddSubgoalInput,
        "requires_confirmation": False,
        "function": add_subgoal,
        "category": "mutation",
    },
    {
        "name": "complete_goal",
        "description": "Mark a goal or subgoal as completed by its 8-character ID.",
        "input_schema": CompleteGoalInput,
        "requires_confirmation": False,
        "function": complete_goal,
        "category": "mutation",
    },
    {
        "name": "abandon_goal",
        "description": "Mark a goal or subgoal as abandoned (will not be pursued) by its ID.",
        "input_schema": AbandonGoalInput,
        "requires_confirmation": False,
        "function": abandon_goal,
        "category": "mutation",
    },
    {
        "name": "list_goals",
        "description": "List all active goals and their subgoals with status indicators.",
        "input_schema": ListGoalsInput,
        "requires_confirmation": False,
        "function": list_goals,
        "category": "readonly",
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]


register_tool_categories(
    {
        "set_goal": "mutation",
        "add_subgoal": "mutation",
        "complete_goal": "mutation",
        "abandon_goal": "mutation",
        "list_goals": "readonly",
    }
)

__all__ = [
    "TOOL_SETUP",
    "configure_goal_tools",
    "set_goal",
    "add_subgoal",
    "complete_goal",
    "abandon_goal",
    "list_goals",
    "SetGoalInput",
    "AddSubgoalInput",
    "CompleteGoalInput",
    "AbandonGoalInput",
    "ListGoalsInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
