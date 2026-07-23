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

from langchain_core.messages import HumanMessage, SystemMessage

from tests.evaluation.runner import EvalResult, EvalScenario, _build_llm, get_model

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


# ── Public API ───────────────────────────────────────────────────────────────


def judge_response(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str = "claude-sonnet-4-6",
) -> float:
    """Score a single scenario result using an LLM-as-judge.

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
    try:
        model_cfg = get_model(judge_model)
    except KeyError:
        # Unknown judge model — fall back to the simplest safe score.
        return _fallback_score(scenario, result)

    try:
        llm = _build_llm(model_cfg, temperature=0.0)
    except (OSError, ImportError):
        # No API key or missing package — can't call judge LLM.
        return _fallback_score(scenario, result)

    prompt = _build_judge_prompt(scenario, result)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        raw = str(response.content) if hasattr(response, "content") else str(response)
    except Exception:
        return _fallback_score(scenario, result)

    score = _parse_score(raw)
    if score is None:
        return _fallback_score(scenario, result)

    return float(score)


def judge_result(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str = "claude-sonnet-4-6",
) -> EvalResult:
    """Return a *copy* of *result* with ``passed`` set by the judge score.

    The copy also has ``notes`` appended with the judge score.
    """
    score = judge_response(scenario, result, judge_model=judge_model)
    passed = score >= 0.5
    notes = result.notes
    if notes:
        notes += " | "
    notes += f"judge_score={score:.2f}"
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
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


def _build_judge_prompt(scenario: EvalScenario, result: EvalResult) -> str:
    """Assemble the human prompt for the judge LLM."""
    criteria = "\n".join(f"- {c}" for c in scenario.success_criteria)
    tools = ", ".join(result.tool_calls_made) if result.tool_calls_made else "none"
    return _JUDGE_HUMAN_TEMPLATE.format(
        description=scenario.description,
        expected_outcome=scenario.expected_outcome,
        criteria=criteria,
        final_response=result.final_response[:2000],  # cap to keep token count low
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
