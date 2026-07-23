"""Gate 2 evaluation judge — LLM-as-judge scoring for domain scenarios.

Usage::

    from tests.evaluation.judge import judge_response, judge_result
    from tests.evaluation.runner import load_scenario

    scenario = load_scenario(Path("scenarios/procurement/po_basic.yaml"))
    result = run_scenario(scenario, model)  # from runner.py

    score = judge_response(scenario, result, judge_model="claude-sonnet-4-6")
    # score is a float in [0.0, 1.0]
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tests.evaluation.runner import (
    EvalResult,
    EvalScenario,
    Turn,
    TurnResult,
    _build_llm,
    get_model,
)

# ── Prompt template ──────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring AI agent responses on Finance/Procurement tasks. "
    "You must respond with valid JSON only, no markdown, no extra text."
)

_JUDGE_HUMAN_TEMPLATE = """You are evaluating an AI agent's response to a Finance/Procurement task.

TASK: {description}
EXPECTED: {expected_outcome}
CRITERIA:
{criteria}

AGENT RESPONSE: {final_response}
TOOLS CALLED: {tools_called}

Score 1.0 if all criteria are fully met, 0.5 if partially met, 0.0 if not met.
Respond with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


# Multi-turn template: one invocation per turn.  The judge sees the
# turn's user prompt + that turn's agent response + that turn's criteria
# in isolation, with the overall scenario description and expected
# outcome as context only.  Turn position is named so the judge can
# weight intermediate vs. final-turn coverage of the criteria.
_JUDGE_MULTI_TURN_HUMAN_TEMPLATE = """You are evaluating turn {turn_idx} of {total_turns} of a multi-turn AI agent workflow on a Finance/Procurement task.

OVERALL SCENARIO: {description}
OVERALL EXPECTED OUTCOME: {expected_outcome}

USER PROMPT FOR THIS TURN: {user_prompt}
CRITERIA FOR THIS TURN:
{criteria}

AGENT RESPONSE ON THIS TURN: {final_response}
TOOLS CALLED ON THIS TURN: {tools_called}

Score 1.0 if this turn fully meets its own criteria, 0.5 if partially, 0.0 if not.
The score is for THIS turn only; the workflow as a whole is aggregated separately.
Respond with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


# ── Public API ───────────────────────────────────────────────────────────────


def judge_response(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str = "claude-sonnet-4-6",
) -> float:
    """Score a single scenario result using an LLM-as-judge.

    For multi-turn scenarios (``len(result.turn_results) > 1`` and equal
    to ``len(scenario.turns)``) each turn is judged independently and
    the per-turn scores are aggregated by a weighted average using each
    turn's ``judge_weight``.  Single-turn scenarios — including legacy
    programmatic ``EvalResult`` constructions with empty
    ``turn_results`` — keep the original single-call behaviour.

    Args:
        scenario: The scenario definition (rubric).
        result: The agent's result for this scenario.
        judge_model: Model id from ``models.yaml`` to use as judge.

    Returns:
        A float in ``[0.0, 1.0]``.  On any failure (missing API key,
        malformed LLM output, etc.) the function returns ``0.0`` rather
        than raising, so that a failed judge call does not crash the
        evaluation pipeline.
    """
    aggregate, _ = _judge_breakdown(scenario, result, judge_model)
    return aggregate


def judge_result(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str = "claude-sonnet-4-6",
) -> EvalResult:
    """Return a *copy* of *result* with ``passed`` set by the judge score.

    For single-turn scenarios the notes string is unchanged: ``judge_score=X.YY``.
    For multi-turn scenarios it carries the per-turn breakdown:
    ``judge_score=X.YY (turns: A.AA, B.BB, ...)``.
    """
    aggregate, per_turn = _judge_breakdown(scenario, result, judge_model)
    passed = aggregate >= 0.5
    notes = result.notes
    if notes:
        notes += " | "
    if len(per_turn) > 1:
        per_turn_str = ", ".join(f"{s:.2f}" for s in per_turn)
        notes += f"judge_score={aggregate:.2f} (turns: {per_turn_str})"
    else:
        notes += f"judge_score={aggregate:.2f}"
    return EvalResult(
        scenario_id=result.scenario_id,
        model_id=result.model_id,
        model_display_name=result.model_display_name,
        passed=passed,
        tool_calls_made=result.tool_calls_made,
        tool_calls_required=result.tool_calls_required,
        turns_used=result.turns_used,
        elapsed_seconds=result.elapsed_seconds,
        final_response=result.final_response,
        error=result.error,
        notes=notes,
        tool_selection_rate=result.tool_selection_rate,
        task_completion=result.task_completion,
        turn_results=result.turn_results,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _judge_breakdown(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str,
) -> tuple[float, list[float]]:
    """Return ``(aggregate_score, per_turn_scores)``.

    Single-turn scenarios produce ``per_turn_scores = [aggregate]``.
    Multi-turn scenarios produce one entry per turn; ``aggregate`` is
    the weighted average using ``Turn.judge_weight``.

    Any failure path (unknown model, missing API key, judge LLM error,
    malformed output) falls back to the heuristic score for the whole
    result and returns ``[fallback]`` regardless of turn count.
    """
    try:
        model_cfg = get_model(judge_model)
    except KeyError:
        fb = _fallback_score(scenario, result)
        return fb, [fb]

    try:
        llm = _build_llm(model_cfg, temperature=0.0)
    except (OSError, ImportError):
        fb = _fallback_score(scenario, result)
        return fb, [fb]

    # Multi-turn dispatch requires both:
    #   1. The runner populated per-turn results (length > 1)
    #   2. Those align 1:1 with scenario.turns
    # Legacy programmatic ``EvalResult`` construction in tests leaves
    # ``turn_results`` empty; we take the single-call path then.
    is_multi_turn = len(result.turn_results) > 1 and len(result.turn_results) == len(scenario.turns)

    if not is_multi_turn:
        prompt = _build_judge_prompt(scenario, result)
        score = _invoke_judge(llm, prompt)
        if score is None:
            fb = _fallback_score(scenario, result)
            return fb, [fb]
        return score, [score]

    per_turn_scores: list[float] = []
    per_turn_weights: list[float] = []
    for turn_idx, (turn, tr) in enumerate(zip(scenario.turns, result.turn_results, strict=True)):
        prompt = _build_judge_prompt_for_turn(
            scenario,
            turn,
            tr,
            turn_idx=turn_idx,
            total_turns=len(scenario.turns),
        )
        score = _invoke_judge(llm, prompt)
        if score is None:
            # One bad turn judgement → fall back to heuristic for the whole result.
            # Consistent with the single-turn path's all-or-nothing failure mode.
            fb = _fallback_score(scenario, result)
            return fb, [fb]
        per_turn_scores.append(score)
        per_turn_weights.append(float(turn.judge_weight))

    total_weight = sum(per_turn_weights)
    if total_weight <= 0:
        # Degenerate case: every turn weighted at zero.  Should be
        # rejected by ``load_scenario`` (non-negative gate), but a
        # programmatic construction could still get here.  Fall back to
        # an unweighted mean so we surface a real number instead of NaN.
        aggregate = sum(per_turn_scores) / len(per_turn_scores)
    else:
        aggregate = (
            sum(s * w for s, w in zip(per_turn_scores, per_turn_weights, strict=True))
            / total_weight
        )

    return aggregate, per_turn_scores


def _invoke_judge(llm: Any, prompt: str) -> float | None:
    """Run the judge LLM once and return a parsed score, or None on failure."""
    try:
        response = llm.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        raw = str(response.content) if hasattr(response, "content") else str(response)
    except Exception:
        return None
    return _parse_score(raw)


def _build_judge_prompt(scenario: EvalScenario, result: EvalResult) -> str:
    """Assemble the human prompt for the judge LLM (single-turn path)."""
    criteria = "\n".join(f"- {c}" for c in scenario.success_criteria)
    tools = ", ".join(result.tool_calls_made) if result.tool_calls_made else "none"
    return _JUDGE_HUMAN_TEMPLATE.format(
        description=scenario.description,
        expected_outcome=scenario.expected_outcome,
        criteria=criteria,
        final_response=result.final_response[:2000],  # cap to keep token count low
        tools_called=tools,
    )


def _build_judge_prompt_for_turn(
    scenario: EvalScenario,
    turn: Turn,
    turn_result: TurnResult,
    *,
    turn_idx: int,
    total_turns: int,
) -> str:
    """Assemble the per-turn human prompt for the judge LLM (multi-turn path)."""
    criteria = "\n".join(f"- {c}" for c in turn.success_criteria)
    tools = ", ".join(turn_result.tool_calls_made) if turn_result.tool_calls_made else "none"
    return _JUDGE_MULTI_TURN_HUMAN_TEMPLATE.format(
        turn_idx=turn_idx + 1,
        total_turns=total_turns,
        description=scenario.description,
        expected_outcome=scenario.expected_outcome,
        user_prompt=turn.user_prompt[:500],
        criteria=criteria,
        final_response=turn_result.final_response[:2000],
        tools_called=tools,
    )


def _parse_score(raw: str) -> float | None:
    """Extract a float score from raw LLM output.

    Tries JSON parsing first, then regex fallback for plain numbers.
    """
    # 1. Try strict JSON parsing
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            score = float(data["score"])
            return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Try to extract JSON from markdown code fences
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 3. Regex fallback: look for a float/integer near "score"
    match = re.search(r'"score"\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    # 4. Last resort: look for a number in [0,1] that appears after the
    #    word "score" on the same line.
    match = re.search(r"score.*?\b([01](?:\.\d+)?)\b", raw, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    return None


def _fallback_score(scenario: EvalScenario, result: EvalResult) -> float:
    """Deterministic heuristic score when the judge LLM is unavailable.

    Uses the same logic as the runner's binary pass/fail:
    - 1.0 if all required tools were called and no error
    - 0.5 if some required tools were called
    - 0.0 otherwise
    """
    if result.error:
        return 0.0
    required = set(scenario.tools_required)
    called = set(result.tool_calls_made)
    if not required:
        return 1.0 if result.final_response else 0.0
    matched = len(called & required)
    if matched == len(required):
        return 1.0
    if matched > 0:
        return 0.5
    return 0.0
