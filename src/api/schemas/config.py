"""Configuration and provider/model management schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Provider schemas
# ---------------------------------------------------------------------------


class ProviderOut(BaseModel):
    """A configured LLM provider (connection info only)."""

    name: str = Field(
        ...,
        description="Provider alias used in config (e.g. 'ollama', 'openai').",
        examples=["openai"],
    )
    type: str = Field(
        ...,
        description="Provider type: 'openai', 'ollama', 'anthropic', or 'google'.",
        examples=["openai"],
    )
    base_url: str | None = Field(
        default=None,
        description="API base URL (null uses the provider default).",
        examples=["https://api.openai.com/v1"],
    )
    has_api_key: bool = Field(
        ...,
        description="True when an API key is configured (key value is never returned).",
    )


class ProviderHealthOut(BaseModel):
    """Provider connectivity health check result."""

    name: str = Field(..., description="Provider alias.")
    reachable: bool = Field(..., description="True when the provider responded successfully.")
    latency_ms: int | None = Field(
        default=None,
        description="Round-trip latency in milliseconds; null when unreachable.",
        examples=[142],
    )
    error: str | None = Field(
        default=None,
        description="Error description when unreachable; null on success.",
    )


# ---------------------------------------------------------------------------
# Model schemas
# ---------------------------------------------------------------------------


class ModelOut(BaseModel):
    """A named model entry from the models registry or provider default list."""

    alias: str = Field(
        ...,
        description="Model alias (registry key) or raw model name.",
        examples=["gpt-4.1-mini"],
    )
    provider: str = Field(
        ...,
        description="Provider alias that serves this model.",
        examples=["openai"],
    )
    model_name: str = Field(
        ...,
        description="Actual model name sent to the provider API.",
        examples=["gpt-4.1-mini"],
    )
    num_ctx: int | None = Field(
        default=None,
        description="Context window size (tokens); null uses provider default. Also accepted as context_window in config files.",
        examples=[131072],
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature override; null uses provider default.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Max output tokens override; null uses provider default.",
    )
    is_active: bool = Field(
        ...,
        description="True when this is the currently selected model.",
    )


class ModelSwitchRequest(BaseModel):
    """Request body for POST /api/v1/config/model."""

    model: str = Field(
        ...,
        description="Model alias or 'provider/model_name' string.",
        examples=["gpt-4.1-mini"],
    )


# ---------------------------------------------------------------------------
# Config view / edit
# ---------------------------------------------------------------------------


class ConfigOut(BaseModel):
    """Current application configuration snapshot (sensitive fields masked)."""

    active_model: str | None = Field(
        default=None,
        description="Active model alias from the models registry; null uses provider default.",
        examples=["oss"],
    )
    memory_mode: str = Field(
        ...,
        description="Default memory mode for new sessions.",
        examples=["conversation"],
    )
    prompt_optimizer: bool = Field(
        ...,
        description="Whether the prompt optimizer is enabled globally.",
    )
    parallel_tool_execution: bool = Field(
        ...,
        description="Whether parallel tool execution is enabled globally.",
    )
    context_compression: bool = Field(
        ...,
        description="Whether context compression is enabled globally.",
    )
    debug: bool = Field(..., description="Whether debug logging is active.")
    verbose: bool = Field(..., description="Whether verbose logging is active.")
    config_file_path: str | None = Field(
        default=None,
        description="Path to the loaded config file; null if using defaults only.",
    )
    providers: list[ProviderOut] = Field(
        default_factory=list,
        description="All configured provider entries.",
    )
    models: list[ModelOut] = Field(
        default_factory=list,
        description="All named model registry entries.",
    )
    raw_yaml: str | None = Field(
        default=None,
        description="Raw YAML of the config file (admin only; null for regular users).",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Active system prompt (null if using default).",
    )
    guardrails: dict | None = Field(
        default=None,
        description="Active guardrail configuration (assistant mode).",
    )


class ConfigPatchRequest(BaseModel):
    """Request body for PATCH /api/v1/config.

    All fields are optional; only supplied fields are changed.
    Setting a field to null resets it to the built-in default.
    """

    debug: bool | None = Field(default=None, description="Toggle debug logging.")
    verbose: bool | None = Field(default=None, description="Toggle verbose logging.")
    prompt_optimizer: bool | None = Field(
        default=None,
        description="Toggle the prompt optimizer globally.",
    )
    parallel_tool_execution: bool | None = Field(
        default=None,
        description="Toggle parallel tool execution globally.",
    )
    context_compression: bool | None = Field(
        default=None,
        description="Toggle context compression globally.",
    )


class ConfigReloadResponse(BaseModel):
    """Result of POST /api/v1/config/reload."""

    reloaded: bool = Field(
        ...,
        description="True when the config was successfully reloaded from disk.",
    )
    config_file_path: str | None = Field(
        default=None,
        description="Path of the file that was reloaded; null if using defaults.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during reload.",
    )


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------


class WizardStartRequest(BaseModel):
    """Request body for POST /api/v1/config/wizard."""

    docs_url: str | None = Field(
        default=None,
        description="URL to fetch configuration docs for wizard context (SSRF-guarded).",
    )
    edit_existing: bool = Field(
        default=False,
        description="True to load and edit the existing config file; false to start fresh.",
    )


class WizardStepRequest(BaseModel):
    """Request body for POST /api/v1/config/wizard/{wizard_id}/step.

    Advances the wizard by one step.  ``answer`` contains the user's response
    to the current wizard question.  ``data`` carries structured fields such as
    provider type, API key, or model name depending on the current step.
    """

    answer: str | None = Field(
        default=None,
        description="Free-text answer to the current wizard question.",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Structured step data (e.g. {provider_type: 'openai', api_key: '...'}).",
    )


class WizardStepOut(BaseModel):
    """Current wizard step result."""

    wizard_id: str = Field(
        ...,
        description="Opaque wizard session identifier.",
    )
    step: int = Field(
        ...,
        description="Current step index (0-based).",
        examples=[1],
    )
    total_steps: int = Field(
        ...,
        description="Total number of wizard steps.",
        examples=[3],
    )
    step_name: str = Field(
        ...,
        description="Human-readable step name.",
        examples=["Connect to LLM"],
    )
    question: str | None = Field(
        default=None,
        description="LLM-generated question for the user; null on the final step.",
    )
    yaml_preview: str | None = Field(
        default=None,
        description="Masked YAML preview of the config so far; populated on the final step.",
    )
    complete: bool = Field(
        ...,
        description="True when all steps are done and the config has been saved.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings from this step.",
    )
