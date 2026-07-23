"""
Gate 2 evaluation CI — pytest entry point for live LLM evaluation.

These tests call REAL LLM APIs and incur API costs.  They are guarded by the
``live_llm`` marker so they never run in the standard fast suite.

Run smoke subset (one model per provider, ~$0.45/run):
    uv run pytest tests/evaluation/ -m "eval and smoke" -v

Run full matrix (all models, nightly):
    uv run pytest tests/evaluation/ -m "eval" -v

Both require the appropriate API keys to be set in the environment.
"""

from __future__ import annotations

import os

import pytest

from tests.evaluation.runner import (
    EvalScenario,
    ModelConfig,
    load_all_scenarios,
    load_model_registry,
    run_scenario,
    smoke_models,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_ALL_SCENARIOS = load_all_scenarios()
_SMOKE_MODELS = smoke_models()
_ALL_MODELS = load_model_registry()

# Smoke scenarios: procurement + finance, critical tag if any.
# If no scenarios are authored yet, smoke suite is empty (no failure).
_SMOKE_SCENARIOS = [s for s in _ALL_SCENARIOS if "smoke" in s.tags or not s.tags]


def _model_has_key(model: ModelConfig) -> bool:
    return bool(os.environ.get(model.env_key))


def _skip_if_no_key(model: ModelConfig) -> None:
    if not _model_has_key(model):
        pytest.skip(f"No API key for {model.display_name} (env: {model.env_key})")


# ── Smoke suite (cheap, per-RC) ───────────────────────────────────────────────


@pytest.mark.live_llm
@pytest.mark.eval
@pytest.mark.smoke
@pytest.mark.parametrize(
    "scenario,model",
    [pytest.param(s, m, id=f"{s.id}__{m.id}") for s in _SMOKE_SCENARIOS for m in _SMOKE_MODELS],
)
def test_smoke_scenario(scenario: EvalScenario, model: ModelConfig) -> None:
    """Gate 2 smoke: one model per provider × smoke scenarios — quick sanity check."""
    _skip_if_no_key(model)
    result = run_scenario(scenario, model)
    assert result.passed, (
        f"Scenario '{scenario.id}' FAILED on {model.display_name}:\n"
        f"  tool_selection_rate={result.tool_selection_rate:.0f}%\n"
        f"  turns_used={result.turns_used}\n"
        f"  error={result.error}\n"
        f"  response={result.final_response[:200]}"
    )


# ── Full matrix (all models × all scenarios, nightly) ────────────────────────


@pytest.mark.live_llm
@pytest.mark.eval
@pytest.mark.parametrize(
    "scenario,model",
    [pytest.param(s, m, id=f"{s.id}__{m.id}") for s in _ALL_SCENARIOS for m in _ALL_MODELS],
)
def test_full_matrix(scenario: EvalScenario, model: ModelConfig) -> None:
    """Gate 2 full matrix: all models × all scenarios."""
    _skip_if_no_key(model)
    result = run_scenario(scenario, model)
    assert result.passed, (
        f"Scenario '{scenario.id}' FAILED on {model.display_name}:\n"
        f"  tool_selection_rate={result.tool_selection_rate:.0f}%\n"
        f"  turns_used={result.turns_used}\n"
        f"  error={result.error}"
    )
