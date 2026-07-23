"""Decision accountability and counter-argumentation for Cogtrix.

ADR-0052 Milestone 1: LLM-driven plan generation, counter-plan evaluation,
and decision justification parsing.

Milestone 2 integration points (graph.py):
- Pass session ``llm`` to ``PlanGenerator`` and ``CounterPlanEvaluator``.
- Call ``extract_decision_justification`` on agent responses to log structured output.
- Inject ``ACCOUNTABILITY_PROMPT`` via ``build_system_prompt(...,
  decision_accountability_prompt=ACCOUNTABILITY_PROMPT)`` when
  ``config.decision_accountability["enabled"]`` is True.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

from src.concurrency import invoke_with_timeout
from src.logging_config import get_logger

log = get_logger()

_REFLECTION_LLM_TIMEOUT_SECONDS: int = 60

# ── System-prompt block — injected by M2 when the feature is enabled ──────────

ACCOUNTABILITY_PROMPT = """\
## Decision Accountability

Before taking any action, structure your reasoning explicitly:

---PLAN---
<step-by-step approach to accomplish the task>
---ASSUMPTIONS---
- <assumption 1>
- <assumption 2>
---EVIDENCE---
- <evidence supporting this plan>
---CONFIDENCE---
<number 0.0–10.0>
---END---

Then generate a counter-plan:

---COUNTER-PLAN---
<alternative approach or why the plan might be wrong>
---FLAWS---
- <specific flaw that could cause failure>
---END---

Proceed only when adjusted confidence (base confidence − 1.0 per critical flaw) ≥ 7.0.
When it falls below 7.0, report uncertainty and suggest revisions instead of acting."""

# Prefix used by graph.py when appending the uncertainty note to an agent response.
# Parsers and API consumers can detect accountability warnings by this prefix.
UNCERTAINTY_NOTE_PREFIX = "⚠️ Decision accountability:"


# ── TypedDicts ────────────────────────────────────────────────────────────────


class PlanSnapshot(TypedDict):
    """Snapshot of a generated plan with assumptions, evidence, and confidence."""

    plan: str
    assumptions: list[str]
    evidence: list[str]
    confidence: float  # 0–10 scale
    timestamp: str


class DecisionJustification(TypedDict):
    """Full decision justification: plan, counter-plan, flaws, and proceed gate."""

    plan: PlanSnapshot
    counter_plan: str
    flaws: list[str]
    confidence_adjustment: float
    should_proceed: bool
    timestamp: str


# ── Internal helpers ──────────────────────────────────────────────────────────


def _call_llm(llm: Any, prompt: str) -> str:
    """Invoke *llm* with a single human message; return text content.

    Returns an empty string on any failure — callers must handle fallback.

    Migrated to :func:`src.concurrency.invoke_with_timeout` under #1903 —
    see :doc:`docs/architecture/CONCURRENCY.md` for the policy.  The
    centralized helper provides the shared bounded pool and the safe
    ``shutdown(wait=False)`` semantics that the previous per-call
    ``ThreadPoolExecutor(max_workers=1)`` here re-implemented inline.
    """
    from langchain_core.messages import HumanMessage

    try:
        result = invoke_with_timeout(
            llm.invoke,
            [HumanMessage(content=prompt)],
            timeout=_REFLECTION_LLM_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log.warning(
            "reflection_delegate LLM call timed out after %ds — returning empty fallback",
            _REFLECTION_LLM_TIMEOUT_SECONDS,
        )
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("reflection_delegate LLM call failed: %s", exc, exc_info=True)
        return ""

    content = result.content
    if isinstance(content, list):
        content = " ".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)
    return str(content) if content else ""


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Return the text between *start_marker* and *end_marker*, stripped."""
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    return text[start : end if end != -1 else None].strip()


def _parse_bullet_list(text: str) -> list[str]:
    """Parse lines starting with '- ' or '* ' into a clean list."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            item = line[2:].strip()
            if item:
                items.append(item)
    return items


_NO_FLAWS_PHRASES = frozenset(
    [
        # Short base phrases (as standalone items)
        "no critical flaws",
        "no major flaws",
        "no significant flaws",
        "no flaws",
        "sound plan",
        "plan is sound",
        # Full-phrase variants the LLM commonly emits
        "no critical flaws identified",
        "no critical flaws found",
        "no flaws identified",
        "no flaws found",
        "no major flaws identified",
        "no significant flaws identified",
        "the plan is sound",
        "no issues identified",
        "no issues found",
    ]
)


def _filter_non_flaws(flaws: list[str]) -> list[str]:
    """Drop 'no flaws' placeholder entries produced by well-behaved LLMs.

    Uses normalized exact match (lowercase, trailing punctuation stripped)
    so that a real flaw like "Plan is sound only for trivial inputs — fails on
    edge cases" is not dropped even though it contains "plan is sound".
    """
    result = []
    for f in flaws:
        # Normalize: lowercase and strip trailing punctuation
        f_normalized = f.lower().strip().rstrip(".,;!?")
        if f_normalized not in _NO_FLAWS_PHRASES:
            result.append(f)
    return result


# ── Core classes ──────────────────────────────────────────────────────────────


@dataclass
class PlanGenerator:
    """Generate plans with explicit assumptions and evidence using the session LLM.

    Args:
        llm: The session LLM (passed from build_agent_graph closure in M2).
    """

    llm: Any

    def generate_plan(self, task: str, context: str = "") -> PlanSnapshot:
        """Generate a plan by calling the LLM with a structured prompt.

        Parses the delimited response for plan text, assumptions, evidence, and
        confidence.  Falls back to a minimal plan when the LLM response cannot
        be parsed.
        """
        prompt = (
            "You are an analytical planner. Generate a structured plan for the task below.\n\n"
            f"## Task\n\n{task}\n"
        )
        if context:
            prompt += f"\n## Context\n\n{context}\n"
        prompt += (
            "\n## Response Format\n\n"
            "Use EXACTLY this format — no extra sections:\n\n"
            "---PLAN---\n"
            "<step-by-step approach>\n"
            "---ASSUMPTIONS---\n"
            "- <assumption 1>\n"
            "---EVIDENCE---\n"
            "- <supporting evidence>\n"
            "---CONFIDENCE---\n"
            "<number 0.0–10.0>\n"
            "---END---\n\n"
            "Confidence: 10=certain, 7+=proceed, <7=uncertain, <5=needs revision. "
            "Only include what you genuinely know."
        )

        raw = _call_llm(self.llm, prompt)
        log.debug("reflection_delegate generate_plan raw=%d chars", len(raw))

        plan_text = _extract_section(raw, "---PLAN---", "---ASSUMPTIONS---")
        assumptions_text = _extract_section(raw, "---ASSUMPTIONS---", "---EVIDENCE---")
        evidence_text = _extract_section(raw, "---EVIDENCE---", "---CONFIDENCE---")
        confidence_text = _extract_section(raw, "---CONFIDENCE---", "---END---")

        assumptions = _parse_bullet_list(assumptions_text) or ["No explicit assumptions identified"]
        evidence = _parse_bullet_list(evidence_text) or ["No explicit evidence cited"]

        # Heuristic baseline, overridden by the LLM's self-assessed confidence when valid
        confidence = self._calculate_confidence(plan_text or task, assumptions, evidence)
        try:
            m = re.search(r"\d+(?:\.\d+)?", confidence_text or "")
            if m:
                llm_conf = float(m.group())
                if 0.0 <= llm_conf <= 10.0:
                    confidence = llm_conf
        except ValueError:
            pass

        return PlanSnapshot(
            plan=plan_text or f"Plan for: {task}",
            assumptions=assumptions,
            evidence=evidence,
            confidence=confidence,
            timestamp=self._current_timestamp(),
        )

    def _calculate_confidence(
        self, plan: str, assumptions: list[str], evidence: list[str]
    ) -> float:
        """Heuristic: more evidence raises confidence, more assumptions lower it."""
        base = 5.0
        evidence_bonus = min(5.0, len(evidence) * 0.5)
        assumption_penalty = len(assumptions) * 0.3
        return max(0.0, min(10.0, base + evidence_bonus - assumption_penalty))

    def _current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()


@dataclass
class CounterPlanEvaluator:
    """Evaluate plans by generating LLM-driven counter-arguments.

    Args:
        llm: The session LLM (passed from build_agent_graph closure in M2).
    """

    llm: Any

    def evaluate_plan(self, plan: PlanSnapshot, task: str) -> DecisionJustification:
        """Generate a counter-plan, identify flaws, and decide whether to proceed."""
        counter_plan_raw = self._generate_counter_plan(plan, task)
        flaws = self._identify_flaws(plan, counter_plan_raw)
        confidence_adjustment = self._calculate_confidence_adjustment(flaws)

        # Extract readable counter-plan text (strip delimiters for storage)
        counter_plan_text = (
            _extract_section(counter_plan_raw, "---COUNTER-PLAN---", "---FLAWS---")
            or counter_plan_raw
        )

        return DecisionJustification(
            plan=plan,
            counter_plan=counter_plan_text,
            flaws=flaws,
            confidence_adjustment=confidence_adjustment,
            should_proceed=self._should_proceed(confidence_adjustment, plan["confidence"]),
            timestamp=self._current_timestamp(),
        )

    def _generate_counter_plan(self, plan: PlanSnapshot, task: str) -> str:
        """Call the LLM to produce a counter-plan that challenges the original."""
        assumptions_str = "\n".join(f"- {a}" for a in plan["assumptions"])
        prompt = (
            "You are a critical reviewer. Find weaknesses in the plan below.\n\n"
            f"## Task\n\n{task}\n\n"
            f"## Plan to Critique\n\n{plan['plan']}\n\n"
            f"## Plan Assumptions\n\n{assumptions_str}\n\n"
            "## Response Format\n\n"
            "Use EXACTLY this format:\n\n"
            "---COUNTER-PLAN---\n"
            "<alternative approach or critique>\n"
            "---FLAWS---\n"
            "- <specific flaw that could cause failure>\n"
            "---END---\n\n"
            "Be specific. If the plan is sound, write 'No critical flaws identified' "
            "under ---FLAWS--- (on its own line, as a bullet)."
        )
        raw = _call_llm(self.llm, prompt)
        return raw or f"Counter-plan unavailable for: {plan['plan'][:80]}"

    def _identify_flaws(self, plan: PlanSnapshot, counter_plan: str) -> list[str]:
        """Parse the counter-plan response for a bullet list of flaws."""
        flaws_text = _extract_section(counter_plan, "---FLAWS---", "---END---")
        raw_flaws = (
            _parse_bullet_list(flaws_text) if flaws_text else _parse_bullet_list(counter_plan)
        )
        return _filter_non_flaws(raw_flaws)

    def _calculate_confidence_adjustment(self, flaws: list[str]) -> float:
        """Each identified flaw reduces confidence by 1.0."""
        return -float(len(flaws))

    def _should_proceed(self, confidence_adjustment: float, original_confidence: float) -> bool:
        """Proceed when adjusted confidence meets the minimum threshold (7.0)."""
        return original_confidence + confidence_adjustment >= 7.0

    def _current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()


# ── Output parser for M2 graph.py integration ─────────────────────────────────


def extract_decision_justification(agent_response: str) -> dict[str, Any] | None:
    """Parse structured decision accountability output from an agent response.

    Called by M2 (graph.py) after receiving an agent response to extract and
    log the structured plan/counter-plan/flaws when the agent followed the
    ACCOUNTABILITY_PROMPT format.

    Returns:
        Dict with plan/assumptions/evidence/counter_plan/flaws/confidence/
        confidence_adjustment/should_proceed, or ``None`` when the response
        does not contain the expected structure.
    """
    if "---PLAN---" not in agent_response:
        return None

    plan_text = _extract_section(agent_response, "---PLAN---", "---ASSUMPTIONS---")
    if not plan_text:
        return None

    assumptions = _parse_bullet_list(
        _extract_section(agent_response, "---ASSUMPTIONS---", "---EVIDENCE---")
    )
    evidence = _parse_bullet_list(
        _extract_section(agent_response, "---EVIDENCE---", "---CONFIDENCE---")
    )
    confidence_text = _extract_section(agent_response, "---CONFIDENCE---", "---END---")
    counter_plan = _extract_section(agent_response, "---COUNTER-PLAN---", "---FLAWS---")
    flaws = _filter_non_flaws(
        _parse_bullet_list(_extract_section(agent_response, "---FLAWS---", "---END---"))
    )

    confidence = 7.0
    try:
        m = re.search(r"[\d.]+", confidence_text or "")
        if m:
            parsed = float(m.group())
            if 0.0 <= parsed <= 10.0:
                confidence = parsed
    except ValueError:
        pass

    confidence_adjustment = -float(len(flaws))
    return {
        "plan": plan_text,
        "assumptions": assumptions,
        "evidence": evidence,
        "counter_plan": counter_plan,
        "flaws": flaws,
        "confidence": confidence,
        "confidence_adjustment": confidence_adjustment,
        "should_proceed": (confidence + confidence_adjustment) >= 7.0,
    }


# ── Clarification policy prompts ──────────────────────────────────────────────
#
# CLARIFICATION_POLICY_PROMPT is a reference constant — not injected directly;
# the abbreviated version lives in DEFAULT_SYSTEM_PROMPT in core.py.
# PRE_ACTION_CONFIRMATION_PROMPT is injected via build_system_prompt() when
# config.pre_action_confirmation_enabled is True.

CLARIFICATION_POLICY_PROMPT = """\
## Clarification Policy

**Ask one focused question before acting when ALL of these hold:**
- The action is irreversible (delete, install, deploy, drop, overwrite, format)
- AND the target scope is genuinely ambiguous ("delete the old files" — which files?)
- OR two equally-valid options exist with no default ("set up auth" — OAuth or JWT?)
- OR a prior instruction directly conflicts with this request

**State your assumption and proceed when:**
- Risk is low and any valid interpretation produces an acceptable result
- One option is a clear default given the conversation context
- The task is underspecified but reversible

**Rules:**
- Ask exactly ONE question per response, phrased specifically
  (not "can you clarify?" — "Should I delete only *.log files or everything in /tmp/build/?")
- After asking, stop — do not begin execution
- Surface conflicts explicitly; never resolve them silently"""

PRE_ACTION_CONFIRMATION_PROMPT = """\
## Pre-Action Confirmation

Before any irreversible operation (delete, uninstall, deploy to production, \
drop table/database, overwrite data, format/wipe storage):
1. State exactly what you are about to do and what the consequence is
2. Ask "Shall I proceed?" and wait for explicit confirmation before using any tool

Skip only when the user's message already contains explicit execution consent \
("go ahead", "yes do it", "proceed", "confirmed", "yes install it")."""
