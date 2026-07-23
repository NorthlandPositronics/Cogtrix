"""Tests for src/tasks/goal_tracker.py — GoalStack and get_goal_stack."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.tasks.goal_tracker import Goal, GoalStack, GoalStatus, get_goal_stack

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_stack(tmp_path: Path) -> GoalStack:
    """A fresh GoalStack backed by a temp directory."""
    return GoalStack(session_id="test", data_dir=tmp_path)


# ── GoalStatus ────────────────────────────────────────────────────────────────


def test_goal_status_values() -> None:
    assert GoalStatus.ACTIVE == "ACTIVE"
    assert GoalStatus.COMPLETED == "COMPLETED"
    assert GoalStatus.ABANDONED == "ABANDONED"


# ── push ──────────────────────────────────────────────────────────────────────


def test_push_returns_8char_id(tmp_stack: GoalStack) -> None:
    goal_id = tmp_stack.push("Finish the report")
    assert len(goal_id) == 8


def test_push_creates_active_goal(tmp_stack: GoalStack) -> None:
    goal_id = tmp_stack.push("Write tests")
    goal = tmp_stack.get(goal_id)
    assert goal is not None
    assert goal.status == GoalStatus.ACTIVE
    assert goal.description == "Write tests"
    assert goal.parent_id is None


def test_push_multiple_top_level_goals(tmp_stack: GoalStack) -> None:
    id1 = tmp_stack.push("Goal A")
    id2 = tmp_stack.push("Goal B")
    assert id1 != id2
    assert len(tmp_stack.list_all()) == 2


def test_push_preserves_insertion_order(tmp_stack: GoalStack) -> None:
    id1 = tmp_stack.push("First")
    id2 = tmp_stack.push("Second")
    id3 = tmp_stack.push("Third")
    active = tmp_stack.list_active()
    assert [g.goal_id for g in active] == [id1, id2, id3]


# ── complete ──────────────────────────────────────────────────────────────────


def test_complete_existing_goal(tmp_stack: GoalStack) -> None:
    goal_id = tmp_stack.push("Existing goal")
    result = tmp_stack.complete(goal_id)
    assert result is True
    goal = tmp_stack.get(goal_id)
    assert goal is not None
    assert goal.status == GoalStatus.COMPLETED
    assert goal.completed_at is not None


def test_complete_unknown_goal_returns_false(tmp_stack: GoalStack) -> None:
    result = tmp_stack.complete("nonexist")
    assert result is False


def test_complete_sets_completed_at_timestamp(tmp_stack: GoalStack) -> None:
    before = time.time()
    goal_id = tmp_stack.push("Timed goal")
    tmp_stack.complete(goal_id)
    after = time.time()
    goal = tmp_stack.get(goal_id)
    assert goal is not None
    assert goal.completed_at is not None
    assert before <= goal.completed_at <= after


# ── abandon ───────────────────────────────────────────────────────────────────


def test_abandon_existing_goal(tmp_stack: GoalStack) -> None:
    goal_id = tmp_stack.push("Goal to abandon")
    result = tmp_stack.abandon(goal_id)
    assert result is True
    goal = tmp_stack.get(goal_id)
    assert goal is not None
    assert goal.status == GoalStatus.ABANDONED


def test_abandon_unknown_goal_returns_false(tmp_stack: GoalStack) -> None:
    result = tmp_stack.abandon("nosuchid")
    assert result is False


# ── add_subgoal ───────────────────────────────────────────────────────────────


def test_add_subgoal_attaches_to_parent(tmp_stack: GoalStack) -> None:
    parent_id = tmp_stack.push("Parent goal")
    sub_id = tmp_stack.add_subgoal(parent_id, "Sub-task")
    parent = tmp_stack.get(parent_id)
    sub = tmp_stack.get(sub_id)
    assert parent is not None
    assert sub is not None
    assert sub_id in parent.subgoals
    assert sub.parent_id == parent_id


def test_add_subgoal_unknown_parent_raises(tmp_stack: GoalStack) -> None:
    with pytest.raises(KeyError):
        tmp_stack.add_subgoal("badparent", "Orphan subgoal")


def test_add_subgoal_status_is_active(tmp_stack: GoalStack) -> None:
    parent_id = tmp_stack.push("Parent")
    sub_id = tmp_stack.add_subgoal(parent_id, "Child")
    sub = tmp_stack.get(sub_id)
    assert sub is not None
    assert sub.status == GoalStatus.ACTIVE


# ── get ───────────────────────────────────────────────────────────────────────


def test_get_existing_goal(tmp_stack: GoalStack) -> None:
    goal_id = tmp_stack.push("Findable goal")
    goal = tmp_stack.get(goal_id)
    assert isinstance(goal, Goal)
    assert goal.goal_id == goal_id


def test_get_missing_goal_returns_none(tmp_stack: GoalStack) -> None:
    assert tmp_stack.get("missing1") is None


# ── list_active ───────────────────────────────────────────────────────────────


def test_list_active_excludes_completed(tmp_stack: GoalStack) -> None:
    id_active = tmp_stack.push("Still active")
    id_done = tmp_stack.push("Done")
    tmp_stack.complete(id_done)
    active = tmp_stack.list_active()
    ids = [g.goal_id for g in active]
    assert id_active in ids
    assert id_done not in ids


def test_list_active_excludes_abandoned(tmp_stack: GoalStack) -> None:
    id_active = tmp_stack.push("Active")
    id_drop = tmp_stack.push("Dropped")
    tmp_stack.abandon(id_drop)
    active = tmp_stack.list_active()
    ids = [g.goal_id for g in active]
    assert id_active in ids
    assert id_drop not in ids


def test_list_active_empty_on_fresh_stack(tmp_stack: GoalStack) -> None:
    assert tmp_stack.list_active() == []


# ── clear_completed ───────────────────────────────────────────────────────────


def test_clear_completed_removes_completed_and_abandoned(tmp_stack: GoalStack) -> None:
    id_keep = tmp_stack.push("Keep me")
    id_done = tmp_stack.push("Done")
    id_gone = tmp_stack.push("Abandon me")
    tmp_stack.complete(id_done)
    tmp_stack.abandon(id_gone)
    removed = tmp_stack.clear_completed()
    assert removed == 2
    assert tmp_stack.get(id_keep) is not None
    assert tmp_stack.get(id_done) is None
    assert tmp_stack.get(id_gone) is None


def test_clear_completed_returns_zero_when_nothing_to_clear(tmp_stack: GoalStack) -> None:
    tmp_stack.push("Still active")
    assert tmp_stack.clear_completed() == 0


def test_clear_completed_removes_from_order(tmp_stack: GoalStack) -> None:
    id1 = tmp_stack.push("One")
    id2 = tmp_stack.push("Two")
    tmp_stack.complete(id1)
    tmp_stack.clear_completed()
    active = tmp_stack.list_active()
    assert len(active) == 1
    assert active[0].goal_id == id2


def test_clear_completed_removes_subgoal_refs(tmp_stack: GoalStack) -> None:
    parent_id = tmp_stack.push("Parent")
    sub_id = tmp_stack.add_subgoal(parent_id, "Sub")
    tmp_stack.complete(sub_id)
    tmp_stack.clear_completed()
    parent = tmp_stack.get(parent_id)
    assert parent is not None
    assert sub_id not in parent.subgoals


# ── to_context_prefix ─────────────────────────────────────────────────────────


def test_to_context_prefix_empty_when_no_active(tmp_stack: GoalStack) -> None:
    assert tmp_stack.to_context_prefix() == ""


def test_to_context_prefix_empty_when_all_completed(tmp_stack: GoalStack) -> None:
    gid = tmp_stack.push("Done")
    tmp_stack.complete(gid)
    assert tmp_stack.to_context_prefix() == ""


def test_to_context_prefix_shows_active_goal(tmp_stack: GoalStack) -> None:
    tmp_stack.push("My top goal")
    prefix = tmp_stack.to_context_prefix()
    assert "## Active Goals" in prefix
    assert "My top goal" in prefix
    assert "[ACTIVE]" in prefix


def test_to_context_prefix_shows_subgoals(tmp_stack: GoalStack) -> None:
    parent_id = tmp_stack.push("Top-level")
    tmp_stack.add_subgoal(parent_id, "Sub-level")
    prefix = tmp_stack.to_context_prefix()
    assert "Sub-level" in prefix


def test_to_context_prefix_shows_subgoal_status(tmp_stack: GoalStack) -> None:
    parent_id = tmp_stack.push("Top")
    sub_id = tmp_stack.add_subgoal(parent_id, "Done sub")
    tmp_stack.complete(sub_id)
    prefix = tmp_stack.to_context_prefix()
    # Parent is still active so it should appear; sub shows COMPLETED
    assert "[COMPLETED]" in prefix
    assert "Done sub" in prefix


def test_to_context_prefix_omits_abandoned_top_level(tmp_stack: GoalStack) -> None:
    gid = tmp_stack.push("Dropped goal")
    tmp_stack.abandon(gid)
    assert tmp_stack.to_context_prefix() == ""


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_creates_json_file(tmp_path: Path) -> None:
    stack = GoalStack(session_id="mysession", data_dir=tmp_path)
    stack.push("Persisted goal")
    expected_path = tmp_path / "goals" / "mysession.json"
    assert expected_path.exists()


def test_save_load_round_trip(tmp_path: Path) -> None:
    stack = GoalStack(session_id="roundtrip", data_dir=tmp_path)
    goal_id = stack.push("Round-trip goal")
    sub_id = stack.add_subgoal(goal_id, "Sub-round-trip")
    stack.complete(sub_id)

    # Reload from disk
    stack2 = GoalStack(session_id="roundtrip", data_dir=tmp_path)
    stack2.load()

    assert stack2.get(goal_id) is not None
    assert stack2.get(goal_id).description == "Round-trip goal"  # type: ignore[union-attr]
    sub = stack2.get(sub_id)
    assert sub is not None
    assert sub.status == GoalStatus.COMPLETED
    assert sub_id in stack2.get(goal_id).subgoals  # type: ignore[union-attr]


def test_load_noop_when_file_missing(tmp_path: Path) -> None:
    stack = GoalStack(session_id="missing", data_dir=tmp_path)
    stack.load()  # should not raise
    assert stack.list_all() == []


def test_load_preserves_insertion_order(tmp_path: Path) -> None:
    stack = GoalStack(session_id="order", data_dir=tmp_path)
    ids = [stack.push(f"Goal {i}") for i in range(3)]

    stack2 = GoalStack(session_id="order", data_dir=tmp_path)
    stack2.load()
    active = stack2.list_active()
    assert [g.goal_id for g in active] == ids


def test_save_json_is_valid(tmp_path: Path) -> None:
    stack = GoalStack(session_id="valid_json", data_dir=tmp_path)
    stack.push("JSON test")
    path = tmp_path / "goals" / "valid_json.json"
    raw = json.loads(path.read_text())
    assert "order" in raw
    assert "goals" in raw


# ── get_goal_stack caching ────────────────────────────────────────────────────


def test_get_goal_stack_returns_same_instance(tmp_path: Path) -> None:
    # Use a unique session ID to avoid cross-test cache pollution
    sid = f"cache_test_{id(tmp_path)}"
    s1 = get_goal_stack(sid, tmp_path)
    s2 = get_goal_stack(sid, tmp_path)
    assert s1 is s2


def test_get_goal_stack_different_sessions_are_distinct(tmp_path: Path) -> None:
    base = id(tmp_path)
    s1 = get_goal_stack(f"sess_a_{base}", tmp_path)
    s2 = get_goal_stack(f"sess_b_{base}", tmp_path)
    assert s1 is not s2


def test_get_goal_stack_loads_existing_data(tmp_path: Path) -> None:
    sid = f"preload_{id(tmp_path)}"
    # Write data via direct GoalStack instance
    direct = GoalStack(session_id=sid, data_dir=tmp_path)
    direct.push("Pre-existing goal")

    # get_goal_stack should load from disk the first time
    cached = get_goal_stack(sid, tmp_path)
    assert len(cached.list_all()) == 1
