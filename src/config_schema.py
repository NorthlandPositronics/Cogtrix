"""Pydantic schemas for Cogtrix YAML configuration files.

Forward-compatible: extra fields are ignored so that new keys added to
YAMLs do not break validation.
"""

from __future__ import annotations

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Turn(BaseModel):
    """A single user turn in a multi-turn evaluation scenario."""

    model_config = ConfigDict(extra="ignore")

    user_prompt: str
    success_criteria: list[str] = Field(default_factory=list)
    judge_weight: float = 1.0

    @field_validator("judge_weight")
    @classmethod
    def _non_negative_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError("judge_weight must be non-negative")
        return v


class EvalScenario(BaseModel):
    """Evaluation scenario schema.

    Accepts both legacy single-turn (``user_prompt`` + ``success_criteria``)
    and multi-turn (``turns:``) shapes.  The two shapes are mutually
    exclusive at the YAML level; ``load_scenario`` in the runner normalises
    either into ``turns``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    domain: str
    title: str
    description: str
    # Legacy single-turn fields
    user_prompt: str = ""
    system_prompt: str = ""
    tools_required: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    max_turns: int = 20
    timeout_seconds: int = 120
    tags: list[str] = Field(default_factory=list)
    budget_usd_estimate: float = 0.05
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    tools_available: list[str] = Field(default_factory=list)
    # Multi-turn fields
    turns: list[Turn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _turns_or_legacy(self) -> EvalScenario:
        has_turns = bool(self.turns)
        has_legacy = bool(self.user_prompt) or bool(self.success_criteria)
        if has_turns and has_legacy:
            raise ValueError(
                "`turns:` is mutually exclusive with top-level `user_prompt` / `success_criteria`"
            )
        if not has_turns and not self.user_prompt:
            raise ValueError("must provide either `turns:` or `user_prompt`")
        return self


class ModelConfig(BaseModel):
    """Schema for a single model entry in ``models.yaml``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    provider: str
    display_name: str
    tier: str
    smoke: bool
    env_key: str
    model_id: str
    base_url: str | None = None
    openrouter_model_id: str | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    prefer_openrouter: bool = False

    @field_validator("tier")
    @classmethod
    def _tier_must_be_abc(cls, v: str) -> str:
        if v not in ("A", "B", "C"):
            raise ValueError(f"tier must be A/B/C, got {v!r}")
        return v


class ModelRegistry(BaseModel):
    """Schema for the top-level ``models.yaml`` file."""

    model_config = ConfigDict(extra="ignore")

    default_timeout: int = 120
    smoke_budget_usd: float = 0.50
    models: list[ModelConfig]

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> ModelRegistry:
        seen: set[str] = set()
        for m in self.models:
            if m.id in seen:
                raise ValueError(f"duplicate model id: {m.id!r}")
            seen.add(m.id)
        return self


# ── Validation helpers ────────────────────────────────────────────────────────


def validate_models_yaml(path: pathlib.Path) -> ModelRegistry:
    """Load and validate ``models.yaml``.

    Raises:
        ValueError: on schema violations.
        FileNotFoundError: if *path* does not exist.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return ModelRegistry(**data)


def validate_scenario(path: pathlib.Path) -> EvalScenario:
    """Load and validate a single scenario YAML.

    Raises:
        ValueError: on schema violations.
        FileNotFoundError: if *path* does not exist.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return EvalScenario(**data)


def validate_all_scenarios(
    scenarios_dir: pathlib.Path,
) -> tuple[list[EvalScenario], list[str]]:
    """Validate every ``*.yaml`` under *scenarios_dir*.

    Returns:
        (valid_scenarios, error_messages)
    """
    scenarios: list[EvalScenario] = []
    errors: list[str] = []
    for yaml_file in sorted(scenarios_dir.rglob("*.yaml")):
        # Skip models.yaml if it happens to live here
        if yaml_file.name == "models.yaml":
            continue
        try:
            scenarios.append(validate_scenario(yaml_file))
        except Exception as exc:
            errors.append(f"{yaml_file}: {exc}")
    return scenarios, errors
