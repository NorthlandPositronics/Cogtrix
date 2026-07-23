"""Gate 2 CI helper — run smoke scenarios and log judge scores.

This module mirrors the existing Gate 2 smoke pytest coverage but emits a
score for each scenario after the harness completes. The score is advisory and
does not change the pass/fail gate yet.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

# Static LPT bin-pack of *smoke* scenarios into four shards.  Cost proxy is
# each scenario's timeout_seconds; the shards are within ~60 s of each
# other on that metric, and each shard runs its assigned scenarios against
# ALL smoke models.  Expected wall time per shard with 4 parallel CI jobs
# is roughly the original ~17 min / 4 ≈ 4-5 min, dominated by the heaviest
# scenario in the shard.
#
# Only scenarios that pass the smoke filter ("smoke" tag, or no tags) are
# assigned here.  Nightly-only scenarios tagged ``gate2`` without
# ``smoke`` (e.g. finance_budget_variance_report) are intentionally
# excluded because run_gate2_smoke never executes them.
#
# When a new smoke scenario is added to tests/evaluation/scenarios/, it
# MUST be added to one of these shards or the runner will raise
# ScenarioShardError.  Pick the shard with the lowest current
# sum(timeout) to preserve balance.
_SHARD_MAP: dict[str, frozenset[str]] = {
    # 240 + 240 = 480s
    "A": frozenset(
        {
            "regression_recovery_synthesis_no_meta_analysis",
            "regression_no_fabrication_for_unknown_entity",
        }
    ),
    # 120 + 90 + 240 + 180 = 630s
    "B": frozenset(
        {
            "procurement_po_approval_basic",
            "safety_refuse_unauthorized_payment",
            "regression_no_url_fabrication_in_response",
            "regression_web_search_no_external_url_recommendation_on_low_yield",
        }
    ),
    # 120 + 120 + 180 + 180 = 600s
    #   (regression_deepseek_native_tool_call_format bumped 60 → 120 alongside
    #    PR #1999 to give kimi-k2-5 enough room for the PR #1997 retry-backoff
    #    path when Moonshot capacity is exhausted; see scenario YAML comment).
    "C": frozenset(
        {
            "procurement_supplier_registration",
            "regression_deepseek_native_tool_call_format",
            "regression_web_search_synthesis_correctness",
            "regression_web_search_synthesis_disagreement",
        }
    ),
    # 240 + 60 + 240 + 180 (× 2 turns) = 900s worst-case
    #   (finance_invoice_approval_workflow bumped 120 → 240 alongside
    #    PR #2013 to give kimi-k2-5's retry path enough headroom when
    #    Moonshot capacity is degraded; same precedent as PR #1999.)
    # Note: regression_multi_turn_effort_gate_no_carryover's 180s is
    # per-turn (2 turns ⇒ ~360s wall worst-case); shard D's other
    # entries are single-turn so they cap at their listed timeouts.
    "D": frozenset(
        {
            "finance_invoice_approval_workflow",
            "regression_stuck_loop_identical_tool_calls",
            "regression_multi_turn_effort_gate_no_carryover",
            "regression_persist_before_refusing",
        }
    ),
}


class ScenarioShardError(RuntimeError):
    """Raised when a scenario is missing from _SHARD_MAP.

    Triggered when a new scenario yaml is added to tests/evaluation/scenarios/
    without being assigned to one of the four CI shards.  The fix is to add
    the new scenario's id to whichever shard in _SHARD_MAP has the lowest
    sum-of-timeouts and rebalance if needed.
    """


def _filter_scenarios_by_shard(
    scenarios: list[EvalScenario],
    shard: str,
) -> list[EvalScenario]:
    """Return only the scenarios assigned to the given shard letter."""
    if shard not in _SHARD_MAP:
        raise ScenarioShardError(f"unknown shard {shard!r}; expected one of {sorted(_SHARD_MAP)}")

    # Every scenario in the input must be covered by some shard, otherwise
    # the shard split would silently drop a scenario from CI coverage.
    all_assigned: set[str] = set().union(*_SHARD_MAP.values())
    unassigned = [s.id for s in scenarios if s.id not in all_assigned]
    if unassigned:
        raise ScenarioShardError(
            "scenarios missing from _SHARD_MAP in tests/evaluation/ci_gate2.py "
            f"(add each to one shard and rebalance): {sorted(unassigned)}"
        )

    target = _SHARD_MAP[shard]
    return [s for s in scenarios if s.id in target]


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


def _candidate_keys_for_model(
    candidate_keys: list[tuple[str, str]],
    model: ModelConfig,
) -> list[tuple[str, str]]:
    """Reorder candidate keys so the model's *native* env_key is tried first.

    The global ``_KEY_PRIORITY`` defaults to OpenRouter first because that
    one key can reach every model in the registry, simplifying CI setup.
    But OpenRouter is a pass-through router and some provider features do
    not survive cleanly across it.  Two concrete cases observed in
    production CI runs:

    * DeepSeek-V4-Flash uses *thinking mode* and returns a
      ``reasoning_content`` field that the API requires on subsequent
      turns.  OpenRouter's request shape does not always thread it
      back, producing
      ``400 The reasoning_content in the thinking mode must be passed
      back to the API`` after the first tool-call turn.
    * Cerebras's bespoke streaming behaviour also differs from
      OpenRouter's normalised passthrough on some scenarios.  Cerebras
      also has a model-id catalogue distinct from OpenRouter's slug
      namespace (e.g. it carries ``gpt-oss-120b`` directly), so a
      native-API call uses the exact provider-side identifier.

    Native APIs are authoritative for their own models — they emit the
    exact shape the provider expects to see back.  Trying native first
    gives us the best fidelity; falling back to OpenRouter only when the
    native key is unavailable preserves the "one key gets everything to
    work" property for solo developers.

    Per-model escape hatch: when ``model.prefer_openrouter`` is True the
    native promotion is skipped and the candidate-key list is returned
    in the configured priority order (OpenRouter first).  This is for
    models whose native provider has a known integration bug that
    OpenRouter happens to absorb — currently ``deepseek-v4-flash`` while
    issue #1391 (reasoning_content threading) is open.  Each flag flip
    must reference its tracking issue in models.yaml so we know when to
    flip it back off.
    """
    if model.prefer_openrouter:
        return candidate_keys
    native = next((k for k in candidate_keys if k[0] == model.env_key), None)
    if native is None:
        return candidate_keys
    return [native] + [k for k in candidate_keys if k[0] != model.env_key]


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

    Issue #1268 + Bug L follow-up (2026-05-20): a run only passes when
    ALL of the following hold:

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
    5. ``not result.tool_errors_unrecovered`` — no tool error went
       unrecovered.  A scenario where the model produced a graceful
       "could not find" answer alongside a pydantic ValidationError on
       http_get was being reported as a pass before — the error was
       invisible to success_criteria because the criteria only inspected
       the final response text.  Tool errors are real failures the test
       must surface.

       Issue #1787 refinement: an error that the model recovered from on
       a successful retry of the same tool is no longer a hard veto.
       The full ``result.tool_errors`` list still surfaces in the log
       for diagnosis; only ``tool_errors_unrecovered`` gates the pass
       decision.

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
    if result.tool_errors_unrecovered:
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


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return True when the provider says the requested model isn't there.

    Defensive fallback predicate: if a model entry in models.yaml maps to
    a model_id that the provider has since deprecated, retired, or simply
    never exposed to this org (e.g. Cerebras dropping llama-3.3-70b in
    favour of gpt-oss-120b), the native-API call surfaces a
    "model not found" / "model_not_found" / "does not exist" / "unknown
    model" error.  These are NOT auth/quota or transient — they are
    permanent on this provider for this key — but they ARE recoverable
    by falling back to the next priority key (often OpenRouter, which
    may still route via a different inference backend).

    Treating these as fall-through-able rather than raising prevents a
    single provider's catalogue change from breaking every Gate 2 cell
    for the affected model.  Documenting this here because the trigger
    is platform-side, not in our code — a future "why did this start
    failing?" investigation will land on this predicate.
    """
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "model not found",
            "model_not_found",
            "does not exist",
            "unknown model",
            "model is not available",
            "model is unavailable",
            "no such model",
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


def _backoff_before_retry(
    model: ModelConfig,
    reason: str,
    emit: Callable[[str], None],
    sleep: Callable[[float], None],
) -> None:
    """Pause before a retry when the model opts in to a backoff.

    Issue #1994: kimi-k2-5 routed through OpenRouter regularly produces
    empty-response flakes when Moonshot's upstream capacity window
    closes.  An immediate retry hits the same closed window and burns
    the retry budget without giving the capacity a chance to recover.
    Models can opt in via ``ModelConfig.retry_backoff_seconds`` — when
    that value is positive, this helper sleeps for that many seconds
    and emits a diagnostic log line so the wait is visible in CI
    output.  Default 0 keeps the historical immediate-retry behaviour
    for every other model.

    The ``sleep`` callable is injectable so tests can verify the
    backoff path without actually waiting.
    """
    delay = max(0, model.retry_backoff_seconds)
    if delay <= 0:
        return
    emit(f"[gate2] BACKOFF {model.id} sleeping {delay}s before retry " f"(reason={reason})")
    sleep(delay)


def _try_run_with_key(
    scenario: EvalScenario,
    model: ModelConfig,
    key: tuple[str, str],
    emit: Callable[[str], None],
    max_retries: int = 1,
    sleep: Callable[[float], None] = time.sleep,
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
    "Model unavailable" errors (a provider deprecating a model_id, e.g.
    Cerebras dropping llama-3.3-70b) take the same shape and the same
    fallback path — see ``_is_model_unavailable_error``.

    Issue #1994 — per-model retry backoff: models that opt in via
    ``ModelConfig.retry_backoff_seconds`` get a short sleep between the
    failing attempt and the retry, giving upstream capacity windows a
    chance to roll forward.  The ``sleep`` parameter is injected so
    tests can drive the backoff path deterministically.
    """
    for attempt in range(max_retries + 1):
        try:
            result = run_scenario(scenario, model, active_key=key)
        except Exception as exc:
            if _is_auth_or_quota_error(exc) or _is_model_unavailable_error(exc):
                emit(f"[gate2] KEY_FAIL {key[0]} for {model.id}: {exc}")
                return None
            if attempt < max_retries and _is_transient_error(exc):
                emit(f"[gate2] RETRY {key[0]} for {model.id} (attempt {attempt + 1}): {exc}")
                _backoff_before_retry(model, "transient_exception", emit, sleep)
                continue
            raise

        if result.error and (
            _is_auth_or_quota_error(Exception(result.error))
            or _is_model_unavailable_error(Exception(result.error))
        ):
            emit(f"[gate2] KEY_FAIL {key[0]} for {model.id}: {result.error}")
            return None
        if result.error and attempt < max_retries and _is_transient_error(Exception(result.error)):
            emit(f"[gate2] RETRY {key[0]} for {model.id} (attempt {attempt + 1}): {result.error}")
            _backoff_before_retry(model, "transient_error", emit, sleep)
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
            _backoff_before_retry(model, "empty_response", emit, sleep)
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
            # Try each candidate key in order until one works.  Native
            # provider keys (e.g. DEEPSEEK_API_KEY for deepseek-v4-flash)
            # are tried before OpenRouter so model-specific features like
            # DeepSeek's thinking-mode reasoning_content threading work
            # against the authoritative API.  See
            # _candidate_keys_for_model for the full rationale.
            result: EvalResult | None = None
            for key in _candidate_keys_for_model(candidate_keys, model):
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
                # Bug L follow-up (2026-05-20): a scenario that could
                # not run because every priority key was rejected (e.g.
                # OpenRouter 402 "out of credits") must NOT be reported
                # as a silent pass. Before this change, exhaustion fell
                # through with `continue` and `any_failures` stayed
                # False, so Gate 2 reported green even when nothing ran.
                #
                # The unconfigured-CI early-exit (no key set at all)
                # still returns 0 — that case never reaches this point
                # because `candidate_keys` is empty above. Once at
                # least one key was provided and tried, exhaustion is
                # a real failure.
                emit(
                    f"[gate2] FAIL {scenario.id}__{model.id} "
                    "(all keys exhausted or ineligible — scenario could not run)"
                )
                any_failures = True
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
            if result.tool_errors:
                # Surface tool errors prominently in CI logs.  Each entry is
                # already a short "<tool>: <truncated>" line; cap the list
                # at 5 to keep noisy runs scannable without losing signal.
                #
                # Issue #1787: ``unrecovered`` is the count that actually
                # gates pass/fail; ``count`` is the diagnostic total
                # (includes errors the model self-corrected from on a
                # later retry of the same tool).  Showing both makes
                # "the model misfired but recovered" runs obvious in CI.
                _unrec = len(result.tool_errors_unrecovered)
                emit(
                    f"[gate2]   TOOL_ERRORS count={len(result.tool_errors)} "
                    f"unrecovered={_unrec} errors={result.tool_errors[:5]}"
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Gate 2 CI smoke runner.

    Forces line-buffered stdout so per-scenario progress is visible when the
    runner is invoked via nohup, CI log redirection, or any context where
    Python defaults to block-buffered output.  Without this every
    ``[gate2] ...`` line stays buffered until the whole run finishes,
    making a multi-minute run look like a hang.  ``_flushing_print``
    (the default ``emit``) flushes every individual line as a belt-and-
    braces backup when ``reconfigure`` is unavailable.

    Filtering flags (CI matrix fan-out):

    * ``--shard {A,B,C,D}`` runs only scenarios assigned to that shard.
    * ``--model <id>`` runs only the named smoke model.

    Both flags are optional and independent — without either, every smoke
    scenario runs against every smoke model (the pre-matrix behaviour,
    used for local invocations and ad-hoc runs).  CI uses both together
    so each matrix cell is one shard × one model.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(prog="ci_gate2")
    parser.add_argument(
        "--shard",
        choices=sorted(_SHARD_MAP),
        default=None,
        help="Run only the scenarios assigned to this CI shard letter.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run only this smoke model id (must be smoke=true in models.yaml).",
    )
    args = parser.parse_args(argv)

    scenarios: list[EvalScenario] | None = None
    if args.shard is not None:
        all_smoke = [s for s in load_all_scenarios() if "smoke" in s.tags or not s.tags]
        scenarios = _filter_scenarios_by_shard(all_smoke, args.shard)

    models: list[ModelConfig] | None = None
    if args.model is not None:
        all_smoke_models = smoke_models()
        models = [m for m in all_smoke_models if m.id == args.model]
        if not models:
            available = sorted(m.id for m in all_smoke_models)
            raise SystemExit(
                f"--model {args.model!r} is not a smoke model; " f"available: {available}"
            )

    _flushing_print(
        f"[gate2] shard={args.shard or 'ALL'} model={args.model or 'ALL'} "
        f"scenarios={[s.id for s in (scenarios or [])] or 'ALL'}"
    )
    return run_gate2_smoke(scenarios=scenarios, models=models)


if __name__ == "__main__":
    raise SystemExit(main())
