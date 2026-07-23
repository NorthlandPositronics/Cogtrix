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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.manager import BaseMemoryManager

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
        working_memory_size (int): Messages in context (default: 6)
        track_reasoning (bool): Track reasoning chain (default: True)
        track_decisions (bool): Track decisions (default: True)
        track_goals (bool): Track goals (default: True)
        max_decisions (int): Max decisions to keep (default: 20)
        max_alternatives (int): Max alternatives to track (default: 10)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "working_memory_size": 6,
        "track_reasoning": True,
        "track_decisions": True,
        "track_goals": True,
        "max_decisions": 20,
        "max_alternatives": 10,
    }

    def __init__(
        self,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(store, session_id, config)
        self._mode_config = {**self.DEFAULT_CONFIG, **(config or {})}

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

    @property
    def mode_name(self) -> str:
        return "reasoning"

    # --- Public API for reasoning context ---

    def set_problem(self, problem: str) -> None:
        """Define the current problem being analyzed."""
        self._current_problem = problem

    def set_hypothesis(self, hypothesis: str) -> None:
        """Set the active hypothesis under consideration."""
        self._active_hypothesis = hypothesis

    def add_reasoning_step(self, step: str) -> None:
        """Add a step to the reasoning chain."""
        self._reasoning_chain.append(step)

    def add_question(self, question: str) -> None:
        """Add an open question."""
        self._open_questions.append(question)

    def clear_reasoning_chain(self) -> None:
        """Clear the reasoning chain for a new problem."""
        self._reasoning_chain = []
        self._active_hypothesis = None

    # --- Alternatives tracking ---

    def add_alternative(self, name: str, status: str = "exploring") -> None:
        """Record an alternative being considered."""
        max_alts = self._mode_config["max_alternatives"]
        at_limit = len(self._alternatives) >= max_alts
        if at_limit and name not in self._alternatives:
            # Remove oldest
            if self._alternatives:
                oldest_key = next(iter(self._alternatives))
                del self._alternatives[oldest_key]
        self._alternatives[name] = status

    def update_alternative(self, name: str, status: str) -> None:
        """Update status of an alternative."""
        if name in self._alternatives:
            self._alternatives[name] = status

    def mark_dead_end(self, description: str) -> None:
        """Record a dead end to avoid revisiting."""
        self._dead_ends.append(description)
        # Mark matching alternatives as dead_end
        for key in list(self._alternatives.keys()):
            if key.lower() in description.lower():
                self._alternatives[key] = "dead_end"

    def add_assumption(self, assumption: str) -> None:
        """Record an assumption being made."""
        self._assumptions.append(assumption)

    # --- Goal tracking ---

    def set_objective(self, objective: str) -> None:
        """Set the primary objective (North Star)."""
        self._primary_objective = objective

    def add_goal(
        self,
        goal_id: str,
        description: str,
        parent_id: str | None = None,
        success_criteria: list[str] | None = None,
    ) -> None:
        """Add a goal to the hierarchy."""
        goal = Goal(
            id=goal_id,
            description=description,
            parent_id=parent_id,
            success_criteria=success_criteria or [],
        )
        self._goals.append(goal)

    def update_goal_status(self, goal_id: str, status: str) -> None:
        """Update a goal's status."""
        for goal in self._goals:
            if goal.id == goal_id:
                goal.status = status
                break

    def set_current_phase(self, phase: str) -> None:
        """Set the current phase of the plan."""
        self._current_phase = phase

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
        dec = Decision(
            id=decision_id,
            timestamp=datetime.now(),
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

    def get_decisions(self) -> list[Decision]:
        """Get all logged decisions."""
        return list(self._decisions)

    # --- Constraint tracking ---

    def add_constraint(self, category: str, constraint: str) -> None:
        """Add a constraint (business, technical, non_negotiables)."""
        if category in self._constraints:
            self._constraints[category].append(constraint)

    def get_constraints(self, category: str | None = None) -> dict[str, list[str]]:
        """Get constraints, optionally filtered by category."""
        if category:
            return {category: self._constraints.get(category, [])}
        return dict(self._constraints)

    # --- Memory Manager Interface ---

    def load(self) -> None:
        """Load reasoning session from storage, sanitizing bad entries."""
        self._messages = self.store.load_history(self.session_id)
        self._messages = self.sanitize_history(self._messages)
        self._loaded = True

    def save(self) -> None:
        """Save reasoning session to storage."""
        self.store.save_history(self.session_id, self._messages)

    def prepare_context(self, user_input: str) -> MemoryContext:
        """Prepare reasoning-focused context for LLM."""
        window_size = self._mode_config["working_memory_size"]
        context_messages = self._messages[-window_size:] if self._messages else []

        prefix_parts = []

        # Primary objective
        if self._primary_objective:
            prefix_parts.append(f"🎯 **OBJECTIVE:** {self._primary_objective}")

        # Current phase and goals
        if self._goals:
            status_icons = {
                "pending": "○",
                "in_progress": "◐",
                "completed": "●",
                "blocked": "✗",
            }
            goals_lines = []
            for g in self._goals[-5:]:
                icon = status_icons.get(g.status, "○")
                goals_lines.append(f"  {icon} {g.description}")
            goals_text = "\n".join(goals_lines)
            prefix_parts.append(f"**Goals:**\n{goals_text}")

        if self._current_phase:
            prefix_parts.append(f"**Current Phase:** {self._current_phase}")

        # Constraints
        if any(self._constraints.values()):
            constraint_lines = []
            for cat, items in self._constraints.items():
                if items:
                    items_str = ", ".join(items[:3])
                    constraint_lines.append(f"  {cat.title()}: {items_str}")
            if constraint_lines:
                constraints_text = "\n".join(constraint_lines)
                prefix_parts.append(f"**Constraints:**\n{constraints_text}")

        # Active reasoning state
        if self._current_problem:
            prefix_parts.append(f"**Problem:** {self._current_problem}")

        if self._active_hypothesis:
            prefix_parts.append(f"**Hypothesis:** {self._active_hypothesis}")

        if self._reasoning_chain:
            chain_lines = []
            for i, step in enumerate(self._reasoning_chain[-5:], 1):
                chain_lines.append(f"  {i}. {step}")
            chain_text = "\n".join(chain_lines)
            prefix_parts.append(f"**Reasoning Chain:**\n{chain_text}")

        # Alternatives
        if self._alternatives:
            alt_lines = []
            for name, status in list(self._alternatives.items())[-5:]:
                alt_lines.append(f"  [{status}] {name}")
            alt_text = "\n".join(alt_lines)
            prefix_parts.append(f"**Alternatives:**\n{alt_text}")

        # Dead ends
        if self._dead_ends:
            dead_lines = [f"  ✗ {d}" for d in self._dead_ends[-3:]]
            dead_text = "\n".join(dead_lines)
            prefix_parts.append(f"**Dead Ends (avoid):**\n{dead_text}")

        # Assumptions
        if self._assumptions:
            assumption_text = ", ".join(self._assumptions[-3:])
            prefix_parts.append(f"**Assumptions:** {assumption_text}")

        # Recent decisions
        if self._decisions:
            dec_lines = []
            for d in self._decisions[-3:]:
                rationale_short = d.rationale[:50] + "..." if len(d.rationale) > 50 else d.rationale
                dec_lines.append(f"  • {d.decision} — {rationale_short}")
            dec_text = "\n".join(dec_lines)
            prefix_parts.append(f"**Recent Decisions:**\n{dec_text}")

        # Open questions
        if self._open_questions:
            questions_lines = [f"  ? {q}" for q in self._open_questions[-3:]]
            questions_text = "\n".join(questions_lines)
            prefix_parts.append(f"**Open Questions:**\n{questions_text}")

        context_prefix = "\n\n".join(prefix_parts) if prefix_parts else None

        return MemoryContext(
            messages=context_messages,
            system_additions=self.get_system_prompt_additions(),
            context_prefix=context_prefix,
            mode=self.mode_name,
            total_messages_stored=len(self._messages),
            context_messages_count=len(context_messages),
            token_estimate=self._estimate_tokens(context_messages),
            metadata={
                "has_objective": self._primary_objective is not None,
                "goal_count": len(self._goals),
                "decision_count": len(self._decisions),
                "has_problem": self._current_problem is not None,
                "reasoning_steps": len(self._reasoning_chain),
                "alternative_count": len(self._alternatives),
            },
        )

    def update(self, user_input: str, ai_response: str) -> None:
        """Update memory with new turn."""
        if HumanMessage is not None:
            self._messages.append(HumanMessage(content=user_input))
            self._messages.append(AIMessage(content=ai_response))
        else:
            self._messages.append({"type": "human", "content": user_input})
            self._messages.append({"type": "ai", "content": ai_response})

    def get_system_prompt_additions(self) -> str | None:
        """Return reasoning-mode system prompt additions."""
        return (
            "You are a strategic advisor and reasoning partner. "
            "COMPLETE tasks systematically - do not stop to ask what to do next. "
            "Think step-by-step, document reasoning and decisions. "
            "When given a multi-step task: break it down, execute each part, "
            "then synthesize results into a complete deliverable. "
            "Flag assumptions but keep working toward the goal. "
            "Avoid dead ends. Stay focused on completing the objective.\n\n"
            "ACCURACY: Base factual claims strictly on data gathered by tools. "
            "Do NOT invent numbers, dates, parameter counts, URLs, or other "
            "specifics not found in tool results. If the information was not "
            "found, say so explicitly.\n\n"
            "IMPORTANT: When the user says 'think deep', 'think deeply', "
            "'deep think', 'analyze thoroughly', or similar phrases, you MUST "
            "invoke the `deep_think` tool — these are explicit tool requests.\n"
            "Also use `deep_think` for decisions with significant trade-offs, "
            "complex strategy questions, or problems that benefit from exploring "
            "multiple approaches.\n"
            "NOTE: `deep_think` runs in isolation and cannot see your "
            "conversation or prior tool results. Always paste the FULL text "
            "of any gathered data into the `context` parameter.\n\n"
            "When a task involves multiple independent research or analysis "
            "subtasks, use `delegate_parallel` to run them concurrently. "
            "Use `delegate_task` to get a second opinion from another model "
            "or to offload routine subtasks while you focus on synthesis."
        )

    def clear(self) -> None:
        """Clear all reasoning memory."""
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

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
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
        base = super().to_dict()

        # Serialize messages
        messages_data = []
        for msg in self._messages:
            if BaseMessage is not None and isinstance(msg, BaseMessage):
                if HumanMessage is not None and isinstance(msg, HumanMessage):
                    msg_type = "human"
                else:
                    msg_type = "ai"
                messages_data.append(
                    {
                        "type": msg_type,
                        "content": msg.content or "",
                    }
                )
            elif isinstance(msg, dict):
                messages_data.append(msg)

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

        return {
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

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore reasoning state."""
        super().from_dict(data)

        # Restore messages
        self._messages = []
        for msg_data in data.get("messages", []):
            if HumanMessage is not None:
                content = msg_data["content"]
                if msg_data.get("type") == "ai":
                    self._messages.append(AIMessage(content=content))
                else:
                    self._messages.append(HumanMessage(content=content))
            else:
                self._messages.append(msg_data)

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

        self._loaded = True

    def _estimate_tokens(self, messages: list[Any]) -> int:
        """
        Rough token estimation for messages.

        Uses simple heuristic: ~4 characters per token.

        Args:
            messages: List of messages

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                total_chars += len(msg.content)
            elif isinstance(msg, dict) and msg.get("content"):
                total_chars += len(msg["content"])
        return total_chars // 4
