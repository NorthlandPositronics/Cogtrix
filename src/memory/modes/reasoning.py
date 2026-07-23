"""
Reasoning and planning memory mode for strategic thinking.

Optimized for:
- Strategic planning and decision-making
- Complex problem analysis
- Architecture decisions
- Project management

Features:
- Goal hierarchy tracking
- Decision logging with rationale
- Reasoning chain preservation
- Constraint awareness
- Alternatives and dead-end tracking
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.manager import BaseMemoryManager

log = logging.getLogger("cogtrix")

# Optional LangChain imports
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    BaseMessage = None  # type: ignore[misc, assignment]


@dataclass
class Decision:
    """Record of a decision made during reasoning."""

    id: str
    timestamp: datetime
    decision: str
    rationale: str
    alternatives_rejected: list[str] = field(default_factory=list)
    trade_offs: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """A goal in the hierarchy."""

    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    parent_id: str | None = None
    success_criteria: list[str] = field(default_factory=list)


class ReasoningMemoryManager(BaseMemoryManager):
    """
    Memory manager for strategic reasoning and planning.

    Configuration options:
        working_memory_size (int): Messages in context (default: 30)
        track_reasoning (bool): Track reasoning chain (default: True)
        track_decisions (bool): Track decisions (default: True)
        track_goals (bool): Track goals (default: True)
        max_decisions (int): Max decisions to keep (default: 20)
        max_alternatives (int): Max alternatives to track (default: 10)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "working_memory_size": 30,
        "vector_recall_k": 2,
        "track_reasoning": True,
        "track_decisions": True,
        "track_goals": True,
        "max_decisions": 20,
        "max_alternatives": 10,
        "prefix_max_stale_turns": 3,
        "summary_max_age_hours": 24,
        "summary_max_uncovered_tokens": 32_000,
        "distill_on_expire": True,
        "facts_ttl_days": 7,
    }

    def __init__(
        self,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        # Merge defaults so BaseMemoryManager sees them (e.g. vector_recall_k)
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(store, session_id, merged)
        self._mode_config = merged

        # Working memory
        self._messages: list[Any] = []

        # Layer 1: Active reasoning buffer
        self._current_problem: str | None = None
        self._active_hypothesis: str | None = None
        self._reasoning_chain: list[str] = []
        self._open_questions: list[str] = []

        # Layer 2: Scratchpad
        self._alternatives: dict[str, str] = {}  # name -> status
        self._dead_ends: list[str] = []
        self._assumptions: list[str] = []

        # Layer 3: Goal hierarchy
        self._primary_objective: str | None = None
        self._goals: list[Goal] = []
        self._current_phase: str | None = None

        # Layer 4: Decision log
        self._decisions: list[Decision] = []

        # Layer 5: Constraints
        self._constraints: dict[str, list[str]] = {
            "business": [],
            "technical": [],
            "non_negotiables": [],
        }

        # Turn counter and section-freshness tracking for prefix gating (F5).
        # Each section records the turn when it was last modified.
        # Sections older than _prefix_max_stale_turns are omitted from the prefix.
        self._turn_count: int = 0
        self._prefix_max_stale_turns: int = self._mode_config.get("prefix_max_stale_turns", 3)
        self._section_ts: dict[str, int] = {}

        # Lock protecting mode-specific mutable state (_messages, _goals, _alternatives, etc.)
        self._mode_lock = threading.Lock()

    @property
    def mode_name(self) -> str:
        return "reasoning"

    # --- Section freshness helpers (F5 prefix gating) ---

    def _touch_section(self, section: str) -> None:
        """Record that *section* was modified on the current turn.

        CALLER MUST HOLD _mode_lock.
        """
        self._section_ts[section] = self._turn_count

    def _is_section_fresh(self, section: str) -> bool:
        """Return True if *section* was modified within the staleness window.

        CALLER MUST HOLD _mode_lock.
        """
        ts = self._section_ts.get(section)
        if ts is None:
            return False
        return (self._turn_count - ts) <= self._prefix_max_stale_turns

    # --- Public API for reasoning context ---

    def set_problem(self, problem: str) -> None:
        """Define the current problem being analyzed."""
        with self._mode_lock:
            self._current_problem = problem
            self._touch_section("problem")

    def set_hypothesis(self, hypothesis: str) -> None:
        """Set the active hypothesis under consideration."""
        with self._mode_lock:
            self._active_hypothesis = hypothesis
            self._touch_section("hypothesis")

    def add_reasoning_step(self, step: str) -> None:
        """Add a step to the reasoning chain."""
        with self._mode_lock:
            self._reasoning_chain.append(step)
            self._touch_section("reasoning_chain")

    def add_question(self, question: str) -> None:
        """Add an open question."""
        with self._mode_lock:
            self._open_questions.append(question)
            self._touch_section("open_questions")

    def clear_reasoning_chain(self) -> None:
        """Clear the reasoning chain for a new problem."""
        with self._mode_lock:
            self._reasoning_chain = []
            self._active_hypothesis = None

    # --- Alternatives tracking ---

    def add_alternative(self, name: str, status: str = "exploring") -> None:
        """Record an alternative being considered."""
        with self._mode_lock:
            max_alts = self._mode_config["max_alternatives"]
            at_limit = len(self._alternatives) >= max_alts
            if at_limit and name not in self._alternatives:
                # Remove oldest
                if self._alternatives:
                    oldest_key = next(iter(self._alternatives))
                    del self._alternatives[oldest_key]
            self._alternatives[name] = status
            self._touch_section("alternatives")

    def update_alternative(self, name: str, status: str) -> None:
        """Update status of an alternative."""
        with self._mode_lock:
            if name in self._alternatives:
                self._alternatives[name] = status

    def mark_dead_end(self, description: str) -> None:
        """Record a dead end to avoid revisiting."""
        with self._mode_lock:
            self._dead_ends.append(description)
            # Mark matching alternatives as dead_end
            for key in list(self._alternatives.keys()):
                if key.lower() in description.lower():
                    self._alternatives[key] = "dead_end"
            self._touch_section("dead_ends")

    def add_assumption(self, assumption: str) -> None:
        """Record an assumption being made."""
        with self._mode_lock:
            self._assumptions.append(assumption)
            self._touch_section("assumptions")

    # --- Goal tracking ---

    def set_objective(self, objective: str) -> None:
        """Set the primary objective (North Star)."""
        with self._mode_lock:
            self._primary_objective = objective
            self._touch_section("objective")

    def add_goal(
        self,
        goal_id: str,
        description: str,
        parent_id: str | None = None,
        success_criteria: list[str] | None = None,
    ) -> None:
        """Add a goal to the hierarchy."""
        with self._mode_lock:
            goal = Goal(
                id=goal_id,
                description=description,
                parent_id=parent_id,
                success_criteria=success_criteria or [],
            )
            self._goals.append(goal)
            self._touch_section("goals")

    def update_goal_status(self, goal_id: str, status: str) -> None:
        """Update a goal's status."""
        with self._mode_lock:
            for goal in self._goals:
                if goal.id == goal_id:
                    goal.status = status
                    break
            self._touch_section("goals")

    def set_current_phase(self, phase: str) -> None:
        """Set the current phase of the plan."""
        with self._mode_lock:
            self._current_phase = phase
            self._touch_section("phase")

    # --- Decision logging ---

    def log_decision(
        self,
        decision_id: str,
        decision: str,
        rationale: str,
        alternatives_rejected: list[str] | None = None,
        trade_offs: list[str] | None = None,
    ) -> None:
        """Log a decision with rationale."""
        with self._mode_lock:
            dec = Decision(
                id=decision_id,
                timestamp=datetime.now(UTC),
                decision=decision,
                rationale=rationale,
                alternatives_rejected=alternatives_rejected or [],
                trade_offs=trade_offs or [],
            )
            self._decisions.append(dec)

            # Enforce max decisions
            max_dec = self._mode_config["max_decisions"]
            if len(self._decisions) > max_dec:
                self._decisions = self._decisions[-max_dec:]
            self._touch_section("decisions")

    def get_decisions(self) -> list[Decision]:
        """Get all logged decisions."""
        with self._mode_lock:
            return list(self._decisions)

    # --- Constraint tracking ---

    def add_constraint(self, category: str, constraint: str) -> None:
        """Add a constraint (business, technical, non_negotiables)."""
        with self._mode_lock:
            if category in self._constraints:
                self._constraints[category].append(constraint)
                self._touch_section("constraints")

    def get_constraints(self, category: str | None = None) -> dict[str, list[str]]:
        """Get constraints, optionally filtered by category."""
        with self._mode_lock:
            if category:
                return {category: self._constraints.get(category, [])}
            return dict(self._constraints)

    # --- Memory Manager Interface ---

    def load(self) -> None:
        """Load reasoning session from storage, sanitizing bad entries."""
        self._messages = self.store.load_history(self.session_id)
        self._messages = self.sanitize_history(self._messages)
        self._load_hybrid_meta()
        self._check_summary_ttl()
        self._check_summary_token_ttl()
        self._load_tier_cache()
        self._load_mode_meta()
        self._clamp_summary_idx()
        self._initial_mode = self.mode_name
        self._loaded = True

    def _restore_mode_state(self, data: dict) -> None:
        """Restore reasoning-specific state from mode_state.json."""
        with self._mode_lock:
            self._current_problem = data.get("current_problem")
            self._active_hypothesis = data.get("active_hypothesis")
            self._reasoning_chain = data.get("reasoning_chain", [])
            self._open_questions = data.get("open_questions", [])
            self._alternatives = data.get("alternatives", {})
            self._dead_ends = data.get("dead_ends", [])
            self._assumptions = data.get("assumptions", [])
            self._primary_objective = data.get("primary_objective")
            self._current_phase = data.get("current_phase")
            self._constraints = data.get(
                "constraints",
                {"business": [], "technical": [], "non_negotiables": []},
            )
            self._goals = []
            for g_data in data.get("goals", []):
                self._goals.append(
                    Goal(
                        id=g_data["id"],
                        description=g_data["description"],
                        status=g_data.get("status", "pending"),
                        parent_id=g_data.get("parent_id"),
                        success_criteria=g_data.get("success_criteria", []),
                    )
                )
            self._decisions = []
            for d_data in data.get("decisions", []):
                self._decisions.append(
                    Decision(
                        id=d_data["id"],
                        timestamp=datetime.fromisoformat(d_data["timestamp"]),
                        decision=d_data["decision"],
                        rationale=d_data["rationale"],
                        alternatives_rejected=d_data.get("alternatives_rejected", []),
                        trade_offs=d_data.get("trade_offs", []),
                    )
                )
            self._turn_count = data.get("_turn_count", 0)
            self._section_ts = data.get("_section_ts", {})

    def save(self) -> None:
        """Save reasoning session to storage."""
        with self._mode_lock:
            self.store.save_history(self.session_id, self._messages)
        super().save()

    def prepare_context(self, user_input: str) -> MemoryContext:
        """Prepare reasoning-focused context for LLM."""
        # Record the moment the user sent this message.
        # Protected by _hybrid_lock so concurrent prepare_context() calls
        # do not silently overwrite each other's timestamps before update()
        # can consume them (issue #1344).
        with self._hybrid_lock:
            self._captured_user_ts = self._now_ts()
            self._pending_user_ts = self._captured_user_ts

        # ── Mode-specific prefix (objective, goals, decisions …) ─────────
        # Computed before message selection so both the tier-cache and
        # sliding-window paths return a consistent prefix.
        prefix_parts = []

        # Hybrid memory (summary + recall)
        hybrid = self._build_hybrid_prefix(user_input)
        if hybrid:
            prefix_parts.append(hybrid)

        # Acquire mode lock for safe reads of mode-specific state
        with self._mode_lock:
            # Primary objective — always include (small, essential)
            if self._primary_objective:
                prefix_parts.append(f"\U0001f3af **OBJECTIVE:** {self._primary_objective}")

            # Goals — gate by freshness
            if self._goals and self._is_section_fresh("goals"):
                status_icons = {
                    "pending": "\u25cb",
                    "in_progress": "\u25d0",
                    "completed": "\u25cf",
                    "blocked": "\u2717",
                }
                goals_lines = []
                for g in self._goals[-5:]:
                    icon = status_icons.get(g.status, "\u25cb")
                    goals_lines.append(f"  {icon} {g.description}")
                goals_text = "\n".join(goals_lines)
                prefix_parts.append(f"**Goals:**\n{goals_text}")

            # Phase — gate by freshness
            if self._current_phase and self._is_section_fresh("phase"):
                prefix_parts.append(f"**Current Phase:** {self._current_phase}")

            # Constraints — gate by freshness
            if any(self._constraints.values()) and self._is_section_fresh("constraints"):
                constraint_lines = []
                for cat, items in self._constraints.items():
                    if items:
                        items_str = ", ".join(items[:3])
                        constraint_lines.append(f"  {cat.title()}: {items_str}")
                if constraint_lines:
                    constraints_text = "\n".join(constraint_lines)
                    prefix_parts.append(f"**Constraints:**\n{constraints_text}")

            # Problem — gate by freshness
            if self._current_problem and self._is_section_fresh("problem"):
                prefix_parts.append(f"**Problem:** {self._current_problem}")

            # Hypothesis — gate by freshness
            if self._active_hypothesis and self._is_section_fresh("hypothesis"):
                prefix_parts.append(f"**Hypothesis:** {self._active_hypothesis}")

            # Reasoning chain — gate by freshness
            if self._reasoning_chain and self._is_section_fresh("reasoning_chain"):
                chain_lines = []
                for i, step in enumerate(self._reasoning_chain[-5:], 1):
                    chain_lines.append(f"  {i}. {step}")
                chain_text = "\n".join(chain_lines)
                prefix_parts.append(f"**Reasoning Chain:**\n{chain_text}")

            # Alternatives — gate by freshness
            if self._alternatives and self._is_section_fresh("alternatives"):
                alt_lines = []
                for name, status in list(self._alternatives.items())[-5:]:
                    alt_lines.append(f"  [{status}] {name}")
                alt_text = "\n".join(alt_lines)
                prefix_parts.append(f"**Alternatives:**\n{alt_text}")

            # Dead ends — gate by freshness
            if self._dead_ends and self._is_section_fresh("dead_ends"):
                dead_lines = [f"  \u2717 {d}" for d in self._dead_ends[-3:]]
                dead_text = "\n".join(dead_lines)
                prefix_parts.append(f"**Dead Ends (avoid):**\n{dead_text}")

            # Assumptions — gate by freshness
            if self._assumptions and self._is_section_fresh("assumptions"):
                assumption_text = ", ".join(self._assumptions[-3:])
                prefix_parts.append(f"**Assumptions:** {assumption_text}")

            # Decisions — gate by freshness
            if self._decisions and self._is_section_fresh("decisions"):
                dec_lines = []
                for d in self._decisions[-3:]:
                    rationale_short = (
                        d.rationale[:50] + "..." if len(d.rationale) > 50 else d.rationale
                    )
                    dec_lines.append(f"  \u2022 {d.decision} \u2014 {rationale_short}")
                dec_text = "\n".join(dec_lines)
                prefix_parts.append(f"**Recent Decisions:**\n{dec_text}")

            # Open questions — gate by freshness
            if self._open_questions and self._is_section_fresh("open_questions"):
                questions_lines = [f"  ? {q}" for q in self._open_questions[-3:]]
                questions_text = "\n".join(questions_lines)
                prefix_parts.append(f"**Open Questions:**\n{questions_text}")

        # Active GoalStack goals — injected from goal_tracker if available
        try:
            from pathlib import Path

            from src.tasks.goal_tracker import get_goal_stack

            _goal_prefix = get_goal_stack(self.session_id, Path("data")).to_context_prefix()
            if _goal_prefix:
                prefix_parts.append(_goal_prefix)
        except Exception:
            pass  # goal tracking unavailable; do not break reasoning context

        context_prefix = "\n\n".join(prefix_parts) if prefix_parts else None

        # ── Tiered context assembly ──────────────────────────────────────
        with self._hybrid_lock:
            tier_ready = self._tier_cache_ready
            tier_cache = self._tier_cache
            summary = self._summary or ""
            summary_msg_idx = self._summary_msg_idx

        if tier_ready and tier_cache is not None:
            from src.memory.tier_cache import assemble_from_tiers

            assembled, tier_counts = assemble_from_tiers(
                snapshot=tier_cache,
                messages=self._messages,
                summary=summary,
                summary_msg_idx=summary_msg_idx,
            )
            total_tokens = sum(tier_counts.values())

            return MemoryContext(
                messages=assembled,
                system_additions=self.get_system_prompt_additions(),
                context_prefix=context_prefix,
                mode=self.mode_name,
                total_messages_stored=len(self._messages),
                context_messages_count=len(assembled),
                token_estimate=total_tokens,
                tier_token_counts=tier_counts,
                metadata={
                    "has_objective": self._primary_objective is not None,
                    "goal_count": len(self._goals),
                    "decision_count": len(self._decisions),
                    "has_problem": self._current_problem is not None,
                    "reasoning_steps": len(self._reasoning_chain),
                    "alternative_count": len(self._alternatives),
                },
            )

        # ── Sliding window fallback (cold cache) ─────────────────────────
        window_size = self._mode_config["working_memory_size"]
        context_messages = self._messages[-window_size:] if self._messages else []

        # Inject timestamps so the LLM has temporal awareness
        context_messages = self._inject_timestamps(context_messages)

        if log.isEnabledFor(logging.DEBUG):
            token_estimate = self._estimate_tokens(context_messages)
        else:
            token_estimate = 0

        return MemoryContext(
            messages=context_messages,
            system_additions=self.get_system_prompt_additions(),
            context_prefix=context_prefix,
            mode=self.mode_name,
            total_messages_stored=len(self._messages),
            context_messages_count=len(context_messages),
            token_estimate=token_estimate,
            metadata={
                "has_objective": self._primary_objective is not None,
                "goal_count": len(self._goals),
                "decision_count": len(self._decisions),
                "has_problem": self._current_problem is not None,
                "reasoning_steps": len(self._reasoning_chain),
                "alternative_count": len(self._alternatives),
            },
        )

    def update(
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[Any] | None = None,
    ) -> None:
        """Update memory with new turn (full chain if available)."""
        # Capture timestamp under _hybrid_lock (matches lock used in
        # prepare_context to write it — issue #1344).
        with self._hybrid_lock:
            ts_to_apply = self._pending_user_ts
            self._pending_user_ts = None
        with self._mode_lock:
            self._turn_count += 1
            # --- Build the human message --------------------------------
            if HumanMessage is not None:
                human_msg: Any = HumanMessage(content=user_input)
            else:
                human_msg = {"type": "human", "content": user_input}
            self._set_msg_ts(human_msg, ts_to_apply)

            self._messages.append(human_msg)

            # --- Append the agent's messages ---------------------------
            if agent_messages is not None:
                for m in agent_messages:
                    self._messages.append(m)
                last = agent_messages[-1]
                if hasattr(last, "content") or isinstance(last, dict):
                    self._set_msg_ts(last)
            else:
                if AIMessage is not None:
                    ai_msg: Any = AIMessage(content=ai_response)
                else:
                    ai_msg = {"type": "ai", "content": ai_response}
                self._set_msg_ts(ai_msg)
                self._messages.append(ai_msg)

        # Layer-1a: accumulate tokens since last summary update
        from src.memory.manager import _msg_tokens

        with self._hybrid_lock:
            self._tokens_since_summary += _msg_tokens(human_msg)
            if agent_messages is not None:
                self._tokens_since_summary += _msg_tokens(agent_messages[-1])
            else:
                self._tokens_since_summary += _msg_tokens(ai_msg)
            self._check_summary_token_ttl_locked()

        # Incrementally summarize messages outside the sliding window
        window_size = self._mode_config["working_memory_size"]
        self._schedule_slow_path(self._messages, window_size)

        # Schedule tier cache roll-forward only when history exceeds the window.
        if len(self._messages) > window_size:
            try:
                self.schedule_tier_roll_forward(
                    max_context_tokens=getattr(self, "_max_context_tokens", 0) or 128_000,
                    llm=getattr(self, "_compression_llm", None),
                )
            except Exception as exc:
                log.debug("Tier roll-forward scheduling failed: %s", exc)

        # ── Domain-shift detection ─────────────────────────────────────
        # Check if recent conversation patterns indicate a topic-domain shift
        # that warrants resetting the rolling summary. Called outside all locks.
        prompts = self._extract_recent_user_prompts(self._messages, limit=3)
        self._check_domain_shift(prompts)

    def get_system_prompt_additions(self) -> str | None:
        """Return reasoning-mode system prompt additions."""
        return (
            "You are a strategic advisor and reasoning partner. "
            "COMPLETE tasks systematically — do not stop to ask what to do next. "
            "Think step-by-step, document reasoning and decisions. "
            "When given a multi-step task: break it down, execute each part, "
            "then synthesize results into a complete deliverable. "
            "Flag assumptions but keep working toward the goal. "
            "When asked to check, look up, find, search for, or verify anything, "
            "start doing it immediately using available tools — never ask whether "
            "the user wants you to search versus explaining how to search."
        )

    def clear(self) -> None:
        """Clear all reasoning memory.

        Acquires _mode_lock before calling super().clear() (which acquires
        _hybrid_lock), consistent with the standard lock nesting order used
        throughout the memory subsystem (_mode_lock -> _hybrid_lock).
        """
        with self._mode_lock:
            super().clear()
            self._messages = []
            self._current_problem = None
            self._active_hypothesis = None
            self._reasoning_chain = []
            self._open_questions = []
            self._alternatives = {}
            self._dead_ends = []
            self._assumptions = []
            self._primary_objective = None
            self._goals = []
            self._current_phase = None
            self._decisions = []
            self._constraints = {
                "business": [],
                "technical": [],
                "non_negotiables": [],
            }
            self._turn_count = 0
            self._section_ts = {}

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
        with self._mode_lock:
            return len(self._messages)

    def get_stats(self) -> dict[str, Any]:
        """Return reasoning statistics."""
        return {
            **super().get_stats(),
            "total_messages": len(self._messages),
            "working_memory_size": self._mode_config["working_memory_size"],
            "has_objective": self._primary_objective is not None,
            "goal_count": len(self._goals),
            "decision_count": len(self._decisions),
            "has_problem": self._current_problem is not None,
            "reasoning_steps": len(self._reasoning_chain),
            "alternative_count": len(self._alternatives),
            "dead_end_count": len(self._dead_ends),
            "assumption_count": len(self._assumptions),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize reasoning state."""
        from src.memory.json_store import _message_to_dict

        with self._mode_lock:
            base = super().to_dict()

            messages_data = [_message_to_dict(m) for m in self._messages]

            # Serialize goals
            goals_data = []
            for g in self._goals:
                goals_data.append(
                    {
                        "id": g.id,
                        "description": g.description,
                        "status": g.status,
                        "parent_id": g.parent_id,
                        "success_criteria": g.success_criteria,
                    }
                )

            # Serialize decisions
            decisions_data = []
            for d in self._decisions:
                decisions_data.append(
                    {
                        "id": d.id,
                        "timestamp": d.timestamp.isoformat(),
                        "decision": d.decision,
                        "rationale": d.rationale,
                        "alternatives_rejected": d.alternatives_rejected,
                        "trade_offs": d.trade_offs,
                    }
                )

            data = {
                **base,
                "messages": messages_data,
                "current_problem": self._current_problem,
                "active_hypothesis": self._active_hypothesis,
                "reasoning_chain": self._reasoning_chain,
                "open_questions": self._open_questions,
                "alternatives": self._alternatives,
                "dead_ends": self._dead_ends,
                "assumptions": self._assumptions,
                "primary_objective": self._primary_objective,
                "goals": goals_data,
                "current_phase": self._current_phase,
                "decisions": decisions_data,
                "constraints": self._constraints,
            }
            data["_turn_count"] = self._turn_count
            data["_section_ts"] = dict(self._section_ts)
            return data

    def _mode_state_dict(self) -> dict[str, Any]:
        """Persist reasoning-specific state without messages or hybrid data."""
        with self._mode_lock:
            goals_data = []
            for g in self._goals:
                goals_data.append(
                    {
                        "id": g.id,
                        "description": g.description,
                        "status": g.status,
                        "parent_id": g.parent_id,
                        "success_criteria": g.success_criteria,
                    }
                )

            decisions_data = []
            for d in self._decisions:
                decisions_data.append(
                    {
                        "id": d.id,
                        "timestamp": d.timestamp.isoformat(),
                        "decision": d.decision,
                        "rationale": d.rationale,
                        "alternatives_rejected": d.alternatives_rejected,
                        "trade_offs": d.trade_offs,
                    }
                )

            data = {
                "current_problem": self._current_problem,
                "active_hypothesis": self._active_hypothesis,
                "reasoning_chain": self._reasoning_chain,
                "open_questions": self._open_questions,
                "alternatives": self._alternatives,
                "dead_ends": self._dead_ends,
                "assumptions": self._assumptions,
                "primary_objective": self._primary_objective,
                "goals": goals_data,
                "current_phase": self._current_phase,
                "decisions": decisions_data,
                "constraints": self._constraints,
            }
            data["_turn_count"] = self._turn_count
            data["_section_ts"] = dict(self._section_ts)
            return data

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore reasoning state."""
        from src.memory.json_store import _dict_to_message

        super().from_dict(data)

        with self._mode_lock:
            self._messages = [_dict_to_message(d) for d in data.get("messages", [])]

            # Restore reasoning state
            self._current_problem = data.get("current_problem")
            self._active_hypothesis = data.get("active_hypothesis")
            self._reasoning_chain = data.get("reasoning_chain", [])
            self._open_questions = data.get("open_questions", [])
            self._alternatives = data.get("alternatives", {})
            self._dead_ends = data.get("dead_ends", [])
            self._assumptions = data.get("assumptions", [])
            self._primary_objective = data.get("primary_objective")
            self._current_phase = data.get("current_phase")
            self._constraints = data.get(
                "constraints",
                {
                    "business": [],
                    "technical": [],
                    "non_negotiables": [],
                },
            )

            # Restore goals
            self._goals = []
            for g_data in data.get("goals", []):
                self._goals.append(
                    Goal(
                        id=g_data["id"],
                        description=g_data["description"],
                        status=g_data.get("status", "pending"),
                        parent_id=g_data.get("parent_id"),
                        success_criteria=g_data.get("success_criteria", []),
                    )
                )

            # Restore decisions
            self._decisions = []
            for d_data in data.get("decisions", []):
                self._decisions.append(
                    Decision(
                        id=d_data["id"],
                        timestamp=datetime.fromisoformat(d_data["timestamp"]),
                        decision=d_data["decision"],
                        rationale=d_data["rationale"],
                        alternatives_rejected=d_data.get("alternatives_rejected", []),
                        trade_offs=d_data.get("trade_offs", []),
                    )
                )

            self._turn_count = data.get("_turn_count", 0)
            self._section_ts = data.get("_section_ts", {})

            self._loaded = True

    # _estimate_tokens() is inherited from BaseMemoryManager
