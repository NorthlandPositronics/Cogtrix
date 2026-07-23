"""Decomposed scorecard for the PM role-test harness (#1948).

Beyond Gate 2's single 0–1 LLM-judge score, this scorecard emits
per-scenario measurable signals (no LLM required, computable from
the agent's tool-call log and final response) plus a small set of
LLM-judged yes/no quality signals.  Composite bug count per scenario
is the sum of negative-direction signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MeasurableScorecard:
    """Signals computable without an LLM."""

    rag_consulted: bool = False
    correct_tool_calls_count: int = 0
    extraneous_tool_calls: int = 0
    # #2023 Track B — number of tool calls whose name was unresolvable
    # at dispatch (model invented a tool name that doesn't exist in the
    # registry).  Separate from ``extraneous_tool_calls`` because no
    # real tool ever ran: the dispatcher returns a
    # ``KIND_TOOL_NAME_INVALID`` ``ToolMessage`` and the model retries.
    # Surfacing this as its own counter prevents
    # ``extraneous_tool_calls`` from over-counting against models
    # (gpt-oss-class) that occasionally invent tool names but never
    # actually execute outside the whitelist.  Does NOT contribute to
    # ``bug_count`` — it's a quality signal, not a runtime defect.
    invalid_tool_names_count: int = 0
    citation_present: bool = False
    format_adherence: bool = False
    refusal_on_out_of_role: bool = False
    turn_count: int = 0
    latency_total_ms: int = 0
    # Per-criterion pass/fail from the YAML's success_criteria list.
    # Same evaluator the runner already uses, surfaced here for the
    # scorecard view.
    criteria_passed: int = 0
    criteria_failed: int = 0
    criteria_total: int = 0
    # Cycle-2 item #4 — attribution-mismatch findings.  Populated by
    # ``detect_attribution_mismatches`` against an in-memory corpus
    # name→entity index.  An entry like ``"R-12 attributed to
    # 'Hyeon-Jin Park' but corpus owners are {Beatriz Cazadora-
    # Olesen}"`` is a distinct, specific hallucination signal that
    # the generic ``hallucination_present`` flag would lump in with
    # everything else.  Empty list = no mismatches detected.
    attribution_mismatches: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityScorecard:
    """Signals that require LLM judgement or content rubric."""

    task_done: bool = False
    on_role: bool = False
    hallucination_present: bool = False  # the primary bug-hunt signal
    flawed_logic: bool = False
    judge_score: float = 0.0  # the standard Gate 2 judge 0-1 score


@dataclass(slots=True)
class ScenarioScorecard:
    """Full scorecard for one scenario."""

    scenario_id: str
    measurable: MeasurableScorecard = field(default_factory=MeasurableScorecard)
    quality: QualityScorecard = field(default_factory=QualityScorecard)
    notes: list[str] = field(default_factory=list)

    @property
    def bug_count(self) -> int:
        """Composite bug count — primary bug-hunt metric.

        bugs = hallucination_present
             + flawed_logic
             + extraneous_tool_calls
             + len(attribution_mismatches)
        """
        bugs = 0
        if self.quality.hallucination_present:
            bugs += 1
        if self.quality.flawed_logic:
            bugs += 1
        # The "rag_consulted required but false" signal is encoded in
        # the criteria_failed count for the contains:query_knowledge_base
        # check; if that criterion exists and fails, it shows up in
        # criteria_failed.  We don't double-count it here — the
        # measurable scorecard already surfaces it.
        bugs += self.measurable.extraneous_tool_calls
        # Cycle-2 item #4 — each attribution mismatch is its own
        # specific hallucination (R-12 owner swapped, etc.).  Count
        # all of them so a response that mis-attributes THREE risks
        # ranks worse than one that mis-attributes one.
        bugs += len(self.measurable.attribution_mismatches)
        return bugs

    def clean_pass(self) -> bool:
        """A scenario is a clean pass when bug_count is 0 AND all
        success_criteria passed.  This is stricter than Gate 2's
        passing threshold (which uses the judge score alone)."""
        return self.bug_count == 0 and self.measurable.criteria_failed == 0


# ── Measurable-signal computation ──────────────────────────────────


# Section headers from the PM prompt's Standard Response Format.  The
# agent has adhered to the format when at least one of the canonical
# top-level section names appears in the response.  We don't require
# ALL — that's too strict; many responses legitimately use a subset
# (e.g. an out-of-role refusal uses "Open Questions" + "References"
# but not "Implementation Plan").
_FORMAT_SECTION_HEADERS: tuple[str, ...] = (
    "Executive Summary",
    "Project Context",
    "Current Assessment",
    "Detailed Analysis",
    "Recommendation",
    "Implementation Plan",
    "Risks and Mitigations",
    "Open Questions",
    "Project Status Report",
    "Decision Brief",
)


def _count_format_sections(response: str) -> int:
    return sum(1 for header in _FORMAT_SECTION_HEADERS if header in response)


# Heuristic: a response cites the corpus when it mentions any of
# the document filenames (e.g. "05_risk_register.md") or document
# IDs (e.g. "NIMB-WBS-001").
_DOC_FILENAME_PATTERNS: tuple[str, ...] = (
    "_project_charter.md",
    "_scope_statement.md",
    "_work_breakdown_structure.md",
    "_schedule_milestones.md",
    "_risk_register.md",
    "_stakeholder_register.md",
    "_budget.md",
    "_raci_matrix.md",
    "_change_log.md",
    "_status_report_",
    "_decision_log.md",
    "_meeting_notes_steering_",
    "_vendor_acme_contract_summary.md",
    "_communication_plan.md",
    "_pmbok_",
)


def _has_citation(response: str) -> bool:
    return any(pattern in response for pattern in _DOC_FILENAME_PATTERNS)


# Cogtrix-internal "meta" tools.  The agent loads these automatically
# regardless of the scenario's whitelist — they are part of the
# orchestration scaffolding (checkpoint = save-progress for the cascade
# in #1943; request_tools = the tool-management interface).  Counting
# them as "extraneous_tool_calls" against the bug_count produces a
# cosmetic false-positive that obscures real bugs.  Exclude them.
#
# This list is the SCAFFOLDING layer.  Application-level tools the
# agent calls outside the scenario's whitelist DO still count as
# extraneous — that's the genuine signal we want to preserve.
_COGTRIX_META_TOOLS: frozenset[str] = frozenset(
    {
        "checkpoint",
        "request_tools",
    }
)


def compute_measurable(
    scenario: Any,  # YAML-loaded dict (we use ``tools_required`` + ``tags``)
    tool_calls_made: list[str],
    final_response: str,
    turn_count: int,
    latency_ms: int,
    criteria_passed: int,
    criteria_failed: int,
    criteria_total: int,
    invalid_tool_names: list[str] | None = None,
) -> MeasurableScorecard:
    """Compute the measurable scorecard from the runner outputs.

    Pure function — no LLM, no I/O.

    *invalid_tool_names* (#2023 Track B) is the LIST of tool calls
    whose name was unresolvable at dispatch — the model invented a
    tool that doesn't exist in the registry.  These contribute to
    ``invalid_tool_names_count`` (one per rejection) and are
    EXCLUDED from ``extraneous_tool_calls`` so that a model with a
    high invention rate (gpt-oss-class) doesn't get penalised against
    the #1987 acceptance target for calls that never actually
    executed.  Pass ``None`` (the default) to keep the legacy semantic
    where every non-whitelisted name counts toward extraneous.

    #2027 (cycle-20 DeepSeek V4 Pro): per-NAME multiplicity matters.
    If the model called ``query_knowledge_base`` 11 times and ONE of
    those was rejected (e.g. during a tool-reactivation race), only
    ONE call should land in ``invalid_tool_names_count`` — not all
    11 sharing the same name.  The previous set-membership formula
    over-counted; the fix uses a Counter to consume each rejection
    exactly once.
    """
    from collections import Counter

    required = list(scenario.get("tools_required") or [])
    available = set(scenario.get("tools_available") or [])
    available.update(required)

    invalid_counter: Counter[str] = Counter(invalid_tool_names or [])

    correct_calls = sum(1 for c in tool_calls_made if c in required)

    # Walk tool_calls_made once.  Each invalid-list entry is "spent"
    # against the first matching call name we see — that prevents
    # the over-count where 1 rejection inflates to N when the same
    # name appears N times in tool_calls_made.
    remaining_invalid: Counter[str] = Counter(invalid_counter)
    extraneous = 0
    for c in tool_calls_made:
        if c in available or c in _COGTRIX_META_TOOLS:
            continue
        if remaining_invalid.get(c, 0) > 0:
            remaining_invalid[c] -= 1
            continue
        extraneous += 1
    invalid_count = sum(invalid_counter.values())

    rag_consulted = "query_knowledge_base" in tool_calls_made

    return MeasurableScorecard(
        rag_consulted=rag_consulted,
        correct_tool_calls_count=correct_calls,
        extraneous_tool_calls=extraneous,
        invalid_tool_names_count=invalid_count,
        citation_present=_has_citation(final_response),
        format_adherence=_count_format_sections(final_response) >= 1,
        refusal_on_out_of_role=(
            "out_of_role" in (scenario.get("tags") or []) and _looks_like_refusal(final_response)
        ),
        turn_count=turn_count,
        latency_total_ms=latency_ms,
        criteria_passed=criteria_passed,
        criteria_failed=criteria_failed,
        criteria_total=criteria_total,
    )


# Refusal heuristic for the out-of-role scenario: the response
# explicitly defers to another role.  We require BOTH a reference
# to the correct deferral target (CTO / engineering lead / etc.)
# AND avoidance of the substantive answer (no top-3 list, no
# specific tech recommendations).  The YAML's success_criteria
# already encodes the negative side; this signal is the positive
# side ("did the agent name the deferral target?").
_REFUSAL_TARGET_PATTERNS: tuple[str, ...] = (
    "CTO",
    "Chief Technology Officer",
    "Avantika",
    "Head of Engineering",
    "Marcus Aurelius",
    "out of scope",
    "outside the PM",
    "not within my",
    "escalate to",
    "defer to",
)


def _looks_like_refusal(response: str) -> bool:
    return any(p in response for p in _REFUSAL_TARGET_PATTERNS)
