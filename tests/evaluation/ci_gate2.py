"""Gate 2 CI helper — run smoke scenarios and log judge scores.

This module mirrors the existing Gate 2 smoke pytest coverage but emits a
score for each scenario after the harness completes. The score is advisory and
does not change the pass/fail gate yet.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any

from tests.evaluation.runner import (
    _KEY_COVERS,
    _KEY_PRIORITY,
    EvalResult,
    EvalScenario,
    ModelConfig,
    _is_auth_or_quota_error,
    load_all_scenarios,
    run_scenario,
    smoke_models,
)
from tests.quality.judge import score_scenario

_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

# Judge score at or above this threshold is treated as pass. Mirrors the 0.5
# cutoff used by tests/evaluation/judge.py:judge_result. The runner's binary
# substring heuristic in EvalResult.passed is too brittle for natural-language
# variation across models (e.g. "VP" vs "Vice President", "$12,500" vs "12500"),
# so we defer to the judge as the authoritative pass/fail signal.
_JUDGE_PASS_THRESHOLD = 0.5

# D2 cost-ceiling multiplier.  Every scenario YAML carries
# budget_usd_estimate; the actual run is allowed to exceed that estimate by
# up to this multiple.  3× catches runaway loops (e.g. an agent looping
# forever on the same tool eats orders of magnitude more tokens than
# planned) while leaving headroom for normal model variance.  The ceiling
# is only enforced when both estimate and actual cost are non-zero —
# scenarios or models without pricing data opt out.
_COST_CEILING_MULTIPLIER = 3.0


def _eligible_models(
    models: list[ModelConfig],
    active_key: tuple[str, str] | None,
) -> list[ModelConfig]:
    """Filter models to those reachable via the active priority key.

    - OPENROUTER_API_KEY → all models that have openrouter_model_id
    - CEREBRAS_API_KEY   → models whose env_key is CEREBRAS_API_KEY
    - Other native keys  → models whose env_key matches
    - No key             → empty list (all scenarios skipped)
    """
    if active_key is None:
        return []
    key_name, _ = active_key
    covers = _KEY_COVERS.get(key_name, set())
    if "*" in covers:
        # OpenRouter: include any model that has an openrouter_model_id
        return [m for m in models if m.openrouter_model_id]
    return [m for m in models if m.env_key in covers]


def score_result(
    scenario: EvalScenario,
    result: EvalResult,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
) -> float:
    """Score one harness result using the Gate 2 judge."""
    payload: dict[str, Any] = result.to_dict()
    payload["scenario"] = asdict(scenario)
    return score_scenario(payload, judge_model=judge_model)


def _cost_ceiling_breached(scenario: EvalScenario, result: EvalResult) -> bool:
    """Return True iff actual cost exceeds the scenario's budget × multiplier.

    Skipped (returns False) when either the scenario budget or the measured
    cost is zero/negative — a model without input/output prices in
    models.yaml or a provider that omitted usage metadata silently bypasses
    this gate rather than failing on missing data.
    """
    if scenario.budget_usd_estimate <= 0:
        return False
    if result.actual_cost_usd <= 0:
        return False
    return result.actual_cost_usd > _COST_CEILING_MULTIPLIER * scenario.budget_usd_estimate


def _final_passed(scenario: EvalScenario, result: EvalResult, score: float) -> bool:
    """Decide pass/fail for one (scenario, model) outcome — strict gate.

    Issue #1268: a run only passes when ALL of the following hold:

    1. ``result.error is None`` — no auth/timeout/transport failure.
    2. ``result.task_completion`` — every required tool was actually called.
       This is the *structural* check the harness already computes.  Without
       it, a judge that only sees the final-response text can rescue a
       partial-completion run (DeepSeek-V3 calling 1 of 3 tools and
       summarising the partial work was the literal failure that
       motivated this gate).
    3. ``score >= _JUDGE_PASS_THRESHOLD`` — the judge LLM approves the
       *quality* of the final response.  This catches subtle correctness
       issues that the binary tools-called check cannot see.
    4. The D2 cost ceiling has not been breached.

    The judge is a quality check on top of the structural floor, never an
    override.  Any single condition failing flips this to False.
    """
    if result.error is not None:
        return False
    if not result.task_completion:
        return False
    if score < _JUDGE_PASS_THRESHOLD:
        return False
    if _cost_ceiling_breached(scenario, result):
        return False
    return True


def _is_transient_error(exc: Exception) -> bool:
    """Return True for timeout, connection, or temporary provider errors."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "timeout",
            "timed out",
            "connection",
            "connect",
            "temporary",
            "503",
            "502",
            "504",
            "429",
            "rate limit",
            "too many requests",
        )
    )


def _is_empty_response(result: EvalResult) -> bool:
    """Return True for the "model produced nothing usable" flake pattern.

    Symptom (seen on DeepSeek-V3 via OpenRouter): no tool calls were
    made AND the final response is empty AND there is no error message
    to indicate why.  This is structurally indistinguishable from a
    transient provider hiccup and is treated as one for retry purposes.

    A run that produced ANY tool call or ANY non-empty content is NOT
    empty, even when ``task_completion`` is False — that is the
    partial-completion path (issue #1268) which the strict gate
    correctly fails without retry.
    """
    return (
        not result.tool_calls_made
        and not (result.final_response or "").strip()
        and not result.error
    )


def _try_run_with_key(
    scenario: EvalScenario,
    model: ModelConfig,
    key: tuple[str, str],
    emit: Callable[[str], None],
    max_retries: int = 1,
) -> EvalResult | None:
    """Attempt to run scenario+model with one key. Returns None on auth/quota error.

    Retries once on transient errors (timeout, connection, rate-limit) to
    reduce flakiness from temporary provider issues (see issue #1124).

    Auth/quota errors take two shapes:

    1. The exception bubbles out of ``run_scenario`` (rare — most providers
       error inside ``graph.invoke``).
    2. ``run_scenario`` catches the provider exception and returns an
       ``EvalResult`` with ``error`` populated (the common case for OpenRouter
       402, OpenAI 401, etc.).

    Both cases must trigger fallback to the next priority key.
    """
    for attempt in range(max_retries + 1):
        try:
            result = run_scenario(scenario, model, active_key=key)
        except Exception as exc:
            if _is_auth_or_quota_error(exc):
                emit(f"[gate2] KEY_FAIL {key[0]} for {model.id}: {exc}")
                return None
            if attempt < max_retries and _is_transient_error(exc):
                emit(f"[gate2] RETRY {key[0]} for {model.id} (attempt {attempt + 1}): {exc}")
                continue
            raise

        if result.error and _is_auth_or_quota_error(Exception(result.error)):
            emit(f"[gate2] KEY_FAIL {key[0]} for {model.id}: {result.error}")
            return None
        if result.error and attempt < max_retries and _is_transient_error(Exception(result.error)):
            emit(f"[gate2] RETRY {key[0]} for {model.id} (attempt {attempt + 1}): {result.error}")
            continue
        # Empty-response flake (DeepSeek-V3 via OpenRouter occasionally
        # returns no tool calls and no content with no error).  This
        # is structurally indistinguishable from a transient timeout —
        # the model produced nothing usable through no fault of the
        # test — so retry once before reporting failure.  A run that
        # produced ANY tool call or ANY content text is NOT empty and
        # falls through to the strict gate (issue #1268), which is
        # the desired behaviour for genuine partial completion.
        if attempt < max_retries and _is_empty_response(result):
            emit(
                f"[gate2] RETRY {key[0]} for {model.id} (attempt {attempt + 1}): "
                "empty_response (no tool calls, no content)"
            )
            continue
        return result
    # All retries exhausted — return the last result with error annotated.
    if result is not None and result.error:
        result.error = f"[retried {max_retries}x] {result.error}"
    return result


_flushing_print: Callable[..., None] = partial(print, flush=True)


def run_gate2_smoke(
    scenarios: list[EvalScenario] | None = None,
    models: list[ModelConfig] | None = None,
    judge_model: str = _DEFAULT_JUDGE_MODEL,
    emit: Callable[[str], None] = _flushing_print,
) -> int:
    """Run the Gate 2 smoke subset, trying API keys in priority order.

    Priority: OPENROUTER_API_KEY → CEREBRAS_API_KEY → DEEPSEEK_API_KEY
              → OPENAI_API_KEY → ANTHROPIC_API_KEY.

    The first key that is present AND works (no auth/quota error) is used for
    the entire run. If a key is present but fails (wrong key, no credits), the
    next key in order is tried. Scenarios are never run twice.

    Returns:
        0 — all executed scenarios passed (or all keys were exhausted → advisory skip).
        1 — at least one executed scenario failed.
    """
    smoke_scenarios = (
        scenarios
        if scenarios is not None
        else [s for s in load_all_scenarios() if "smoke" in s.tags or not s.tags]
    )
    all_smoke_models = models if models is not None else smoke_models()

    # Build the candidate key list from environment.
    candidate_keys: list[tuple[str, str]] = []
    for key_name in _KEY_PRIORITY:
        value = os.environ.get(key_name, "")
        if value:
            candidate_keys.append((key_name, value))

    if not candidate_keys:
        emit("[gate2] SKIP — no API key set in priority order: " + ", ".join(_KEY_PRIORITY))
        return 0

    any_failures = False

    for scenario in smoke_scenarios:
        for model in all_smoke_models:
            # Try each candidate key in order until one works.
            result: EvalResult | None = None
            for key in candidate_keys:
                eligible = _eligible_models([model], key)
                if not eligible:
                    emit(
                        f"[gate2] SKIP {scenario.id}__{model.id} (key {key[0]} cannot reach this model)"
                    )
                    continue
                result = _try_run_with_key(scenario, model, key, emit)
                if result is not None:
                    break  # key worked — use it

            if result is None:
                emit(f"[gate2] SKIP {scenario.id}__{model.id} (all keys exhausted or ineligible)")
                continue

            score = score_result(scenario, result, judge_model=judge_model)
            # Strict gate: structural completion AND judge approval AND
            # within cost ceiling AND no transport error.  See _final_passed
            # for the rationale (issue #1268).
            final_passed = _final_passed(scenario, result, score)
            cost_ceiling_breached = _cost_ceiling_breached(scenario, result)

            emit(
                "[gate2] "
                f"{scenario.id}__{model.id} "
                f"passed={final_passed} "
                f"score={score:.2f} "
                f"completion={result.task_completion} "
                f"tools={len(result.tool_calls_made)} "
                f"turns={result.turns_used} "
                f"cost_usd={result.actual_cost_usd:.4f} "
                f"budget_usd={scenario.budget_usd_estimate:.4f} "
                f"error={result.error or 'none'}"
            )
            if not result.task_completion:
                missing = sorted(set(result.tool_calls_required) - set(result.tool_calls_made))
                emit(
                    f"[gate2]   PARTIAL_COMPLETION rate={result.tool_selection_rate:.0f}% "
                    f"missing_required_tools={missing}"
                )
            if cost_ceiling_breached:
                emit(
                    f"[gate2]   COST_CEILING actual_usd={result.actual_cost_usd:.4f} > "
                    f"{_COST_CEILING_MULTIPLIER}× budget_usd={scenario.budget_usd_estimate:.4f}"
                )
            if not final_passed:
                any_failures = True
                # On failure, emit the final response (truncated) and the
                # actual tool calls made so the failure is diagnosable from
                # CI logs alone, without needing to re-run locally.
                _final_text = (result.final_response or "").strip()
                if len(_final_text) > 600:
                    _final_text = _final_text[:600] + " …[truncated]"
                emit(f"[gate2]   tools_called={result.tool_calls_made or '[]'}")
                emit(f"[gate2]   final_response={_final_text!r}")

    return 1 if any_failures else 0


def main() -> int:
    """CLI entry point for the Gate 2 CI smoke runner.

    Forces line-buffered stdout so per-scenario progress is visible when the
    runner is invoked via nohup, CI log redirection, or any context where
    Python defaults to block-buffered output.  Without this every
    ``[gate2] ...`` line stays buffered until the whole run finishes,
    making a multi-minute run look like a hang.  ``_flushing_print``
    (the default ``emit``) flushes every individual line as a belt-and-
    braces backup when ``reconfigure`` is unavailable.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    return run_gate2_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
