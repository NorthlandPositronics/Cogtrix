"""Tests for cogtrix_core/tools/goal_tools.py — goal-tracking CRUD tools.

Covers: set_goal, add_subgoal, complete_goal, abandon_goal, list_goals.
Issue: #1225 (zero test coverage).

Test isolation strategy:
- Each test uses a unique session_id derived from the pytest `tmp_path` fixture.
- Module globals _session_id / _data_dir are saved and restored by the fixture.
- The global _stacks cache in goal_tracker is cleared between tests.
"""

import pytest

import cogtrix_core.tools.goal_tools as goal_tools_module
from cogtrix_core.tasks.goal_tracker import get_goal_stack


@pytest.fixture
def isolated_goal_tools(tmp_path):
    """Isolate goal_tools globals and the goal_tracker cache for each test.

    Uses tmp_path as the data directory and a unique session_id per test so
    that get_goal_stack() returns a fresh, empty GoalStack every time.
    Cleans up both the module globals and the global _stacks cache after the
    test runs.
    """
    # Unique session id so get_goal_stack() creates a fresh stack, not a cached
    # one from a previous test.
    session_id = f"test-{tmp_path.name}"
    data_dir = tmp_path

    # Save original globals so we can restore them
    orig_session_id = goal_tools_module._session_id
    orig_data_dir = goal_tools_module._data_dir

    # Wire in isolated values
    goal_tools_module._session_id = session_id
    goal_tools_module._data_dir = data_dir

    # Clear any cached stack for this session (could be stale from prior test)
    from cogtrix_core.tasks import goal_tracker

    with goal_tracker._stacks_lock:
        goal_tracker._stacks.pop(session_id, None)

    yield session_id, data_dir

    # Teardown: restore globals and clear cache
    goal_tools_module._session_id = orig_session_id
    goal_tools_module._data_dir = orig_data_dir
    with goal_tracker._stacks_lock:
        goal_tracker._stacks.pop(session_id, None)


# ── set_goal ───────────────────────────────────────────────────────────────────


class TestSetGoal:
    def test_set_goal_creates_goal_and_returns_id(self, isolated_goal_tools) -> None:
        """set_goal returns a 'Goal set [{id}]: {description}' string."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.set_goal("Write the quarterly report")

        assert result.startswith("Goal set [")
        assert result.endswith("]: Write the quarterly report")
        # goal_id is 8 chars between the brackets
        goal_id = result.split("[")[1].split("]")[0]
        assert len(goal_id) == 8

        # Verify the goal was persisted
        stack = get_goal_stack(session_id, data_dir)
        goal = stack.get(goal_id)
        assert goal is not None
        assert goal.description == "Write the quarterly report"
        assert goal.status.value == "ACTIVE"

    def test_set_goal_empty_description_accepted(self, isolated_goal_tools) -> None:
        """set_goal accepts an empty-string description (no validation in tool)."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.set_goal("")

        assert result.startswith("Goal set [")
        goal_id = result.split("[")[1].split("]")[0]
        stack = get_goal_stack(session_id, data_dir)
        assert stack.get(goal_id) is not None

    def test_set_goal_multiple_goals_distinct_ids(self, isolated_goal_tools) -> None:
        """Multiple set_goal calls produce distinct goal IDs."""
        session_id, data_dir = isolated_goal_tools
        result1 = goal_tools_module.set_goal("Goal one")
        result2 = goal_tools_module.set_goal("Goal two")
        result3 = goal_tools_module.set_goal("Goal three")

        id1 = result1.split("[")[1].split("]")[0]
        id2 = result2.split("[")[1].split("]")[0]
        id3 = result3.split("[")[1].split("]")[0]

        assert len({id1, id2, id3}) == 3  # all distinct


# ── add_subgoal ────────────────────────────────────────────────────────────────


class TestAddSubgoal:
    def test_add_subgoal_under_valid_parent(self, isolated_goal_tools) -> None:
        """add_subgoal returns 'Subgoal added [{sub_id}] under [{parent_id}]'."""
        session_id, data_dir = isolated_goal_tools

        parent_result = goal_tools_module.set_goal("Top-level goal")
        parent_id = parent_result.split("[")[1].split("]")[0]

        sub_result = goal_tools_module.add_subgoal(parent_id, "Sub-goal item")

        assert sub_result.startswith("Subgoal added [")
        assert f"under [{parent_id}]" in sub_result
        sub_id = sub_result.split("[")[1].split("]")[0]
        assert len(sub_id) == 8

        stack = get_goal_stack(session_id, data_dir)
        sub_goal = stack.get(sub_id)
        assert sub_goal is not None
        assert sub_goal.description == "Sub-goal item"
        assert sub_goal.parent_id == parent_id

    def test_add_subgoal_nonexistent_parent_returns_error(self, isolated_goal_tools) -> None:
        """add_subgoal returns an error when the parent_id is not found."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.add_subgoal("notexist1", "Some subgoal")

        assert result.startswith("Error: parent goal ")
        assert "notexist1" in result
        assert "not found" in result

    def test_add_subgoal_empty_description_accepted(self, isolated_goal_tools) -> None:
        """add_subgoal accepts an empty description (no tool-level validation)."""
        session_id, data_dir = isolated_goal_tools
        parent_result = goal_tools_module.set_goal("Parent")
        parent_id = parent_result.split("[")[1].split("]")[0]

        result = goal_tools_module.add_subgoal(parent_id, "")
        assert result.startswith("Subgoal added [")


# ── complete_goal ──────────────────────────────────────────────────────────────


class TestCompleteGoal:
    def test_complete_goal_marks_completed(self, isolated_goal_tools) -> None:
        """complete_goal returns success message and updates goal status."""
        session_id, data_dir = isolated_goal_tools

        create_result = goal_tools_module.set_goal("Deliver the package")
        goal_id = create_result.split("[")[1].split("]")[0]

        complete_result = goal_tools_module.complete_goal(goal_id)

        assert complete_result == f"Goal [{goal_id}] marked as completed."

        stack = get_goal_stack(session_id, data_dir)
        goal = stack.get(goal_id)
        assert goal is not None
        assert goal.status.value == "COMPLETED"
        assert goal.completed_at is not None

    def test_complete_goal_nonexistent_returns_error(self, isolated_goal_tools) -> None:
        """complete_goal returns an error when goal_id is not found."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.complete_goal("nomatch1")

        assert result == "Error: goal 'nomatch1' not found"

    def test_complete_goal_already_completed_still_succeeds(self, isolated_goal_tools) -> None:
        """complete_goal on an already-completed goal returns success (idempotent)."""
        session_id, data_dir = isolated_goal_tools

        create_result = goal_tools_module.set_goal("Task")
        goal_id = create_result.split("[")[1].split("]")[0]

        goal_tools_module.complete_goal(goal_id)
        second_result = goal_tools_module.complete_goal(goal_id)

        # The underlying GoalStack.complete() returns True even if already COMPLETED,
        # so the tool returns the success message.
        assert second_result == f"Goal [{goal_id}] marked as completed."


# ── abandon_goal ───────────────────────────────────────────────────────────────


class TestAbandonGoal:
    def test_abandon_goal_marks_abandoned(self, isolated_goal_tools) -> None:
        """abandon_goal returns success message and updates goal status."""
        session_id, data_dir = isolated_goal_tools

        create_result = goal_tools_module.set_goal("Deprecated task")
        goal_id = create_result.split("[")[1].split("]")[0]

        abandon_result = goal_tools_module.abandon_goal(goal_id)

        assert abandon_result == f"Goal [{goal_id}] marked as abandoned."

        stack = get_goal_stack(session_id, data_dir)
        goal = stack.get(goal_id)
        assert goal is not None
        assert goal.status.value == "ABANDONED"
        assert goal.completed_at is not None

    def test_abandon_goal_nonexistent_returns_error(self, isolated_goal_tools) -> None:
        """abandon_goal returns an error when goal_id is not found."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.abandon_goal("notagoal")

        assert result == "Error: goal 'notagoal' not found"


# ── list_goals ─────────────────────────────────────────────────────────────────


class TestListGoals:
    def test_list_goals_empty_returns_no_active_goals(self, isolated_goal_tools) -> None:
        """list_goals returns 'No active goals.' when the stack is empty."""
        session_id, data_dir = isolated_goal_tools
        result = goal_tools_module.list_goals()

        assert result == "No active goals."

    def test_list_goals_with_active_goals_returns_context_prefix(self, isolated_goal_tools) -> None:
        """list_goals returns the formatted ## Active Goals block."""
        session_id, data_dir = isolated_goal_tools

        result1 = goal_tools_module.set_goal("Primary task")
        goal_id = result1.split("[")[1].split("]")[0]
        goal_tools_module.add_subgoal(goal_id, "Subtask A")

        output = goal_tools_module.list_goals()

        assert output.startswith("## Active Goals")
        assert "Primary task" in output
        assert "Subtask A" in output
        assert "[ACTIVE]" in output

    def test_list_goals_completed_goal_not_shown(self, isolated_goal_tools) -> None:
        """list_goals excludes COMPLETED goals from the active block."""
        session_id, data_dir = isolated_goal_tools

        result = goal_tools_module.set_goal("Finished task")
        goal_id = result.split("[")[1].split("]")[0]
        goal_tools_module.complete_goal(goal_id)

        output = goal_tools_module.list_goals()
        # Completed goal should not appear in active output
        assert "Finished task" not in output
        assert output == "No active goals."

    def test_list_goals_abandoned_goal_not_shown(self, isolated_goal_tools) -> None:
        """list_goals excludes ABANDONED goals from the active block."""
        session_id, data_dir = isolated_goal_tools

        result = goal_tools_module.set_goal("Dropped task")
        goal_id = result.split("[")[1].split("]")[0]
        goal_tools_module.abandon_goal(goal_id)

        output = goal_tools_module.list_goals()
        assert "Dropped task" not in output
        assert output == "No active goals."

    def test_list_goals_mixed_states_shows_only_active(self, isolated_goal_tools) -> None:
        """list_goals shows only ACTIVE goals when a mix of states exists."""
        session_id, data_dir = isolated_goal_tools

        # Create one that will be completed
        r1 = goal_tools_module.set_goal("Will be completed")
        id1 = r1.split("[")[1].split("]")[0]
        goal_tools_module.complete_goal(id1)

        # Create one that will be abandoned
        r2 = goal_tools_module.set_goal("Will be abandoned")
        id2 = r2.split("[")[1].split("]")[0]
        goal_tools_module.abandon_goal(id2)

        # Create one that stays active
        goal_tools_module.set_goal("Stays active")

        output = goal_tools_module.list_goals()

        assert "Stays active" in output
        assert "Will be completed" not in output
        assert "Will be abandoned" not in output
