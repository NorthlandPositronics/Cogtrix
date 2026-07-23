"""Tests for ``src.config.schema`` — YAML config validation."""

from __future__ import annotations

import pathlib
import tempfile

import pytest
import yaml

from src.config_schema import (
    ModelConfig,
    ModelRegistry,
    Turn,
    validate_all_scenarios,
    validate_models_yaml,
    validate_scenario,
)

_EVAL_DIR = pathlib.Path(__file__).parent / "evaluation"
_MODELS_YAML = _EVAL_DIR / "models.yaml"
_SCENARIOS_DIR = _EVAL_DIR / "scenarios"


# ── ModelRegistry ────────────────────────────────────────────────────────────


def test_validate_models_yaml_smoke() -> None:
    """The real models.yaml must validate."""
    registry = validate_models_yaml(_MODELS_YAML)
    assert isinstance(registry, ModelRegistry)
    assert registry.default_timeout > 0
    assert registry.smoke_budget_usd > 0
    assert len(registry.models) > 0
    # Every model must have the required fields
    for m in registry.models:
        assert m.id
        assert m.provider
        assert m.display_name
        assert m.tier in ("A", "B", "C")
        assert isinstance(m.smoke, bool)
        assert m.env_key
        assert m.model_id


def test_models_yaml_duplicate_id_rejected() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "default_timeout": 120,
                "smoke_budget_usd": 0.5,
                "models": [
                    {
                        "id": "dup",
                        "provider": "openai",
                        "display_name": "Dup",
                        "tier": "A",
                        "smoke": True,
                        "env_key": "KEY",
                        "model_id": "gpt-4",
                    },
                    {
                        "id": "dup",
                        "provider": "openai",
                        "display_name": "Dup 2",
                        "tier": "B",
                        "smoke": False,
                        "env_key": "KEY",
                        "model_id": "gpt-4",
                    },
                ],
            },
            f,
        )
        f.flush()
        with pytest.raises(ValueError, match="duplicate model id"):
            validate_models_yaml(pathlib.Path(f.name))


def test_models_yaml_invalid_tier_rejected() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "default_timeout": 120,
                "smoke_budget_usd": 0.5,
                "models": [
                    {
                        "id": "bad-tier",
                        "provider": "openai",
                        "display_name": "Bad",
                        "tier": "D",
                        "smoke": True,
                        "env_key": "KEY",
                        "model_id": "gpt-4",
                    },
                ],
            },
            f,
        )
        f.flush()
        with pytest.raises(ValueError, match="tier must be A/B/C"):
            validate_models_yaml(pathlib.Path(f.name))


def test_models_yaml_forward_compatible_extra_fields() -> None:
    """Extra fields on model entries must not break validation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "default_timeout": 120,
                "smoke_budget_usd": 0.5,
                "models": [
                    {
                        "id": "extra",
                        "provider": "openai",
                        "display_name": "Extra",
                        "tier": "A",
                        "smoke": True,
                        "env_key": "KEY",
                        "model_id": "gpt-4",
                        "future_field": 123,
                    },
                ],
            },
            f,
        )
        f.flush()
        registry = validate_models_yaml(pathlib.Path(f.name))
        assert len(registry.models) == 1


# ── EvalScenario ─────────────────────────────────────────────────────────────


def test_validate_all_scenarios_smoke() -> None:
    """All real scenario YAMLs must validate."""
    scenarios, errors = validate_all_scenarios(_SCENARIOS_DIR)
    assert not errors, "\n".join(errors)
    assert len(scenarios) > 0
    for s in scenarios:
        assert s.id
        assert s.domain
        assert s.title
        assert s.description
        assert s.timeout_seconds > 0
        assert s.budget_usd_estimate >= 0
        # Every scenario must have at least one turn (legacy folded into turns)
        assert len(s.turns) > 0 or s.user_prompt


def test_scenario_legacy_shape() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "id": "legacy",
                "domain": "finance",
                "title": "Legacy",
                "description": "A legacy scenario.",
                "user_prompt": "Do something.",
                "success_criteria": ["contains: done"],
                "timeout_seconds": 60,
                "budget_usd_estimate": 0.01,
            },
            f,
        )
        f.flush()
        scenario = validate_scenario(pathlib.Path(f.name))
        assert scenario.user_prompt == "Do something."
        assert scenario.success_criteria == ["contains: done"]


def test_scenario_multi_turn_shape() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "id": "multi",
                "domain": "procurement",
                "title": "Multi",
                "description": "A multi-turn scenario.",
                "turns": [
                    {
                        "user_prompt": "First turn.",
                        "success_criteria": ["contains: first"],
                        "judge_weight": 2.0,
                    },
                    {
                        "user_prompt": "Second turn.",
                        "success_criteria": ["contains: second"],
                    },
                ],
                "timeout_seconds": 90,
                "budget_usd_estimate": 0.02,
            },
            f,
        )
        f.flush()
        scenario = validate_scenario(pathlib.Path(f.name))
        assert len(scenario.turns) == 2
        assert scenario.turns[0].judge_weight == 2.0
        assert scenario.turns[1].judge_weight == 1.0


def test_scenario_mutually_exclusive_turns_and_legacy() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "id": "bad",
                "domain": "finance",
                "title": "Bad",
                "description": "Mutually exclusive.",
                "user_prompt": "Legacy.",
                "turns": [{"user_prompt": "Turn."}],
            },
            f,
        )
        f.flush()
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_scenario(pathlib.Path(f.name))


def test_scenario_missing_prompt_and_turns() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "id": "bad",
                "domain": "finance",
                "title": "Bad",
                "description": "Missing prompt.",
            },
            f,
        )
        f.flush()
        with pytest.raises(ValueError, match="must provide either"):
            validate_scenario(pathlib.Path(f.name))


def test_scenario_forward_compatible_extra_fields() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "id": "extra",
                "domain": "finance",
                "title": "Extra",
                "description": "Has extra fields.",
                "user_prompt": "Do something.",
                "future_field": [1, 2, 3],
            },
            f,
        )
        f.flush()
        scenario = validate_scenario(pathlib.Path(f.name))
        assert scenario.id == "extra"


# ── Unit-level pydantic validation ───────────────────────────────────────────


def test_model_config_pricing_must_be_numeric() -> None:
    """Pydantic coerces strings that look like numbers, but rejects true garbage."""
    with pytest.raises(ValueError):
        ModelConfig(
            id="x",
            provider="p",
            display_name="X",
            tier="A",
            smoke=True,
            env_key="K",
            model_id="m",
            input_price_per_1m="not_a_number",
        )


def test_turn_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Turn(user_prompt="hi", judge_weight=-1.0)
