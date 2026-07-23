"""Gate 2 quality judge — LLM-as-judge scoring for scenario results.

This module mirrors the Gate 2 evaluation judge used under ``tests/evaluation``
but accepts plain dictionaries so it can score quality-harness outputs without
requiring the evaluation dataclasses.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tests.evaluation.runner import _build_llm, get_model

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring Cogtrix agent behavior on quality "
    "scenarios. You must respond with valid JSON only, no markdown, no extra text."
)

_JUDGE_HUMAN_TEMPLATE = """You are evaluating an AI agent's response to a Cogtrix quality scenario.

TASK: {description}
EXPECTED: {expected_outcome}
CRITERIA:
{criteria}

AGENT RESPONSE: {final_response}
TOOLS CALLED: {tools_called}

Score 1.0 if all criteria are fully met, 0.5 if partially met, 0.0 if not met.
Respond with JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


def score_scenario(result: dict[str, Any], judge_model: str = "claude-sonnet-4-6") -> float:
    """Score a single quality-harness result using an LLM-as-judge.

    The *result* payload is intentionally flexible:

    - a top-level dict may contain ``description``, ``expected_outcome``,
      ``criteria``, ``final_response``, ``tools_called``, ``tools_required``,
      ``tool_calls_made`` and ``error``; or
    - it may nest the scenario details under ``result["scenario"]``.

    If the judge model cannot be loaded or returns malformed output, the
    function falls back to a deterministic heuristic instead of raising.
    """
    try:
        model_cfg = get_model(judge_model)
    except KeyError:
        return _fallback_score(result)

    try:
        llm = _build_llm(model_cfg, temperature=0.0)
    except (OSError, ImportError):
        return _fallback_score(result)

    prompt = _build_judge_prompt(result)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        raw = str(response.content) if hasattr(response, "content") else str(response)
    except Exception:
        return _fallback_score(result)

    score = _parse_score(raw)
    if score is None:
        return _fallback_score(result)

    return float(score)


def _build_judge_prompt(result: dict[str, Any]) -> str:
    """Assemble the human prompt for the judge LLM."""
    scenario = _scenario_payload(result)
    description = _first_text(
        _get_value(result, scenario, "description", "title"), default="(no description)"
    )
    expected_outcome = _first_text(
        _get_value(result, scenario, "expected_outcome", "expected"),
        default="(no expected outcome)",
    )
    criteria = _format_criteria(_get_value(result, scenario, "criteria", "success_criteria"))
    final_response = _first_text(_get_value(result, scenario, "final_response"), default="")
    tools_called = _format_tools(_get_value(result, scenario, "tools_called", "tool_calls_made"))
    return _JUDGE_HUMAN_TEMPLATE.format(
        description=description,
        expected_outcome=expected_outcome,
        criteria=criteria,
        final_response=final_response[:2000],
        tools_called=tools_called,
    )


def _parse_score(raw: str) -> float | None:
    """Extract a float score from raw LLM output.

    Tries JSON parsing first, then regex fallback for plain numbers.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            score = float(data["score"])
            return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    match = re.search(r'"score"\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    match = re.search(r"score.*?\b([01](?:\.\d+)?)\b", raw, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    return None


def _fallback_score(result: dict[str, Any]) -> float:
    """Deterministic heuristic score when the judge LLM is unavailable."""
    scenario = _scenario_payload(result)

    if _first_text(_get_value(result, scenario, "error"), default=""):
        return 0.0

    required = set(_as_text_list(_get_value(result, scenario, "tools_required", "required_tools")))
    called = set(
        _as_text_list(
            _get_value(result, scenario, "tool_calls_made", "tools_called", "called_tools")
        )
    )
    final_response = _first_text(_get_value(result, scenario, "final_response"), default="")

    if not required:
        return 1.0 if final_response else 0.0

    matched = len(called & required)
    if matched == len(required):
        return 1.0
    if matched > 0:
        return 0.5
    return 0.0


def _scenario_payload(result: dict[str, Any]) -> Mapping[str, Any]:
    """Return the nested scenario mapping, or the result itself if needed."""
    scenario = result.get("scenario")
    if isinstance(scenario, Mapping):
        return scenario
    return result


def _get_value(result: dict[str, Any], scenario: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present value among result/scenario keys."""
    for source in (result, scenario):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _first_text(value: Any, default: str = "") -> str:
    """Coerce *value* to a string, preserving empty/falsey values as default."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    text = str(value).strip()
    return text if text else default


def _as_text_list(value: Any) -> list[str]:
    """Coerce strings / sequences into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            text = _first_text(item)
            if text:
                out.append(text)
        return out
    text = _first_text(value)
    return [text] if text else []


def _format_criteria(value: Any) -> str:
    criteria = _as_text_list(value)
    if not criteria:
        return "- (none provided)"
    return "\n".join(f"- {criterion}" for criterion in criteria)


def _format_tools(value: Any) -> str:
    tools = _as_text_list(value)
    return ", ".join(tools) if tools else "none"
