"""Goal tracking — persistent GoalStack per session.

Classes:
    GoalStatus  — ACTIVE / COMPLETED / ABANDONED
    Goal        — single goal dataclass
    GoalStack   — per-session stack with push/complete/abandon/subgoal ops

Module helpers:
    get_goal_stack(session_id, data_dir) -> GoalStack
        Return (and cache) the GoalStack for a session, loading from disk
        if this is the first access.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from src.utils.atomic_write import atomic_write_json

log = logging.getLogger("cogtrix.tasks.goal_tracker")


# ── Enums & dataclasses ───────────────────────────────────────────────────────


class GoalStatus(StrEnum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


@dataclass
class Goal:
    goal_id: str
    description: str
    status: GoalStatus
    created_at: float
    completed_at: float | None
    subgoals: list[str] = field(default_factory=list)  # child goal_ids
    parent_id: str | None = None


# ── GoalStack ─────────────────────────────────────────────────────────────────


class GoalStack:
    """Per-session ordered goal hierarchy with atomic JSON persistence."""

    def __init__(self, session_id: str, data_dir: str | Path) -> None:
        self._session_id = session_id
        self._data_dir = Path(data_dir)
        self._goals: dict[str, Goal] = {}
        # Insertion order for top-level goals only
        self._order: list[str] = []

    # ── Mutations ─────────────────────────────────────────────────────────────

    def push(self, description: str, parent_id: str | None = None) -> str:
        """Create a new ACTIVE goal and return its goal_id."""
        goal_id = str(uuid.uuid4())[:8]
        goal = Goal(
            goal_id=goal_id,
            description=description,
            status=GoalStatus.ACTIVE,
            created_at=time.time(),
            completed_at=None,
            subgoals=[],
            parent_id=parent_id,
        )
        self._goals[goal_id] = goal
        if parent_id is None:
            self._order.append(goal_id)
        else:
            parent = self._goals.get(parent_id)
            if parent is not None:
                parent.subgoals.append(goal_id)
        self.save()
        return goal_id

    def complete(self, goal_id: str) -> bool:
        """Mark goal COMPLETED. Returns False if goal_id unknown."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = time.time()
        self.save()
        return True

    def abandon(self, goal_id: str) -> bool:
        """Mark goal ABANDONED. Returns False if goal_id unknown."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.status = GoalStatus.ABANDONED
        goal.completed_at = time.time()
        self.save()
        return True

    def add_subgoal(self, parent_id: str, description: str) -> str:
        """Add a child goal under parent_id. Raises KeyError if parent unknown."""
        if parent_id not in self._goals:
            raise KeyError(f"Parent goal {parent_id!r} not found")
        return self.push(description, parent_id=parent_id)

    def clear_completed(self) -> int:
        """Remove all COMPLETED and ABANDONED goals; return count removed."""
        terminal = {GoalStatus.COMPLETED, GoalStatus.ABANDONED}
        to_remove = {gid for gid, g in self._goals.items() if g.status in terminal}
        if not to_remove:
            return 0
        for gid in to_remove:
            del self._goals[gid]
        self._order = [gid for gid in self._order if gid not in to_remove]
        for goal in self._goals.values():
            goal.subgoals = [s for s in goal.subgoals if s not in to_remove]
        self.save()
        return len(to_remove)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def list_active(self) -> list[Goal]:
        """Return only ACTIVE goals, top-level first then child goals."""
        seen: set[str] = set()
        result: list[Goal] = []
        # Top-level goals in insertion order
        for gid in self._order:
            goal = self._goals.get(gid)
            if goal and goal.status == GoalStatus.ACTIVE:
                result.append(goal)
                seen.add(gid)
        # Remaining active goals (subgoals not under an active top-level)
        for goal in self._goals.values():
            if goal.goal_id not in seen and goal.status == GoalStatus.ACTIVE:
                result.append(goal)
        return result

    def list_all(self) -> list[Goal]:
        return list(self._goals.values())

    def to_context_prefix(self) -> str:
        """Return a formatted block for LLM injection; empty string if no active goals.

        Format::

            ## Active Goals
            [ACTIVE] abc12345: Top-level goal description
              [ACTIVE] def67890: Sub-goal description
              [COMPLETED] ghi11111: Another sub-goal
        """
        active_top = [
            self._goals[gid]
            for gid in self._order
            if gid in self._goals and self._goals[gid].status == GoalStatus.ACTIVE
        ]
        if not active_top:
            return ""
        lines = ["## Active Goals"]
        for goal in active_top:
            lines.append(f"[{goal.status}] {goal.goal_id}: {goal.description}")
            for sub_id in goal.subgoals:
                sub = self._goals.get(sub_id)
                if sub is not None:
                    lines.append(f"  [{sub.status}] {sub.goal_id}: {sub.description}")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomically write goal state to {data_dir}/goals/{session_id}.json."""
        path = self._data_dir / "goals" / f"{self._session_id}.json"
        payload: dict = {
            "session_id": self._session_id,
            "order": self._order,
            "goals": {
                gid: {
                    "goal_id": g.goal_id,
                    "description": g.description,
                    "status": str(g.status),
                    "created_at": g.created_at,
                    "completed_at": g.completed_at,
                    "subgoals": g.subgoals,
                    "parent_id": g.parent_id,
                }
                for gid, g in self._goals.items()
            },
        }
        with atomic_write_json(path) as f:
            json.dump(payload, f)

    def load(self) -> None:
        """Load goal state from disk; no-op if the file does not exist."""
        path = self._data_dir / "goals" / f"{self._session_id}.json"
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._order = raw.get("order", [])
            self._goals = {}
            for gid, gdata in raw.get("goals", {}).items():
                completed_at_raw = gdata.get("completed_at")
                self._goals[gid] = Goal(
                    goal_id=gdata["goal_id"],
                    description=gdata["description"],
                    status=GoalStatus(gdata["status"]),
                    created_at=float(gdata["created_at"]),
                    completed_at=float(completed_at_raw) if completed_at_raw is not None else None,
                    subgoals=gdata.get("subgoals", []),
                    parent_id=gdata.get("parent_id"),
                )
        except Exception as exc:
            log.warning("Failed to load goals for session %s: %s", self._session_id, exc)


# ── Module-level session cache ────────────────────────────────────────────────

_stacks: dict[str, GoalStack] = {}


def get_goal_stack(session_id: str, data_dir: str | Path) -> GoalStack:
    """Return the cached GoalStack for *session_id*, creating + loading if absent."""
    if session_id not in _stacks:
        stack = GoalStack(session_id, data_dir)
        stack.load()
        _stacks[session_id] = stack
    return _stacks[session_id]


__all__ = [
    "Goal",
    "GoalStack",
    "GoalStatus",
    "get_goal_stack",
]
