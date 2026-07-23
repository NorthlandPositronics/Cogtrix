"""Session request/response schemas.

A *session* is the top-level context that groups conversation history,
memory state, provider/model settings, and tool configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from cogtrix_core.api.schemas.common import ensure_utc

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

MemoryMode = Literal["conversation", "code", "reasoning"]
AgentState = Literal[
    "idle",
    "thinking",
    "analyzing",
    "researching",
    "deep_thinking",
    "writing",
    "delegating",
    "done",
    "error",
]


# ---------------------------------------------------------------------------
# Embedded sub-objects
# ---------------------------------------------------------------------------


class TokenCounts(BaseModel):
    """Token usage snapshot for a session."""

    input_tokens: int = Field(
        ...,
        description="Cumulative input tokens sent to the LLM in this session.",
        examples=[12400],
    )
    output_tokens: int = Field(
        ...,
        description="Cumulative output tokens received from the LLM in this session.",
        examples=[3200],
    )
    context_window: int = Field(
        ...,
        description="Maximum context window for the current model (tokens).",
        examples=[131072],
    )


class SessionConfig(BaseModel):
    """Mutable session configuration settable at creation and via PATCH."""

    agent_name: str | None = Field(
        default=None,
        description="Name of the registered agent to use for this session. If provided, tool filtering based on agent.tools_include/tools_exclude will be applied.",
        examples=["researcher"],
    )
    model: str | None = Field(
        default=None,
        description="Model alias from the models registry or raw model name.",
        examples=["gpt-4.1-mini"],
    )
    memory_mode: MemoryMode | None = Field(
        default=None,
        description="Memory strategy: 'conversation', 'code', or 'reasoning'.",
        examples=["conversation"],
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=32768,
        description="Override the default system prompt for this session.",
    )
    prompt_optimizer: bool | None = Field(
        default=None,
        description="Enable/disable the one-shot prompt rewriter for this session.",
    )
    parallel_tool_execution: bool | None = Field(
        default=None,
        description="Enable/disable parallel tool execution for this session.",
    )
    context_compression: bool | None = Field(
        default=None,
        description="Enable/disable context compression for this session.",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Maximum agent steps (tool calls) per turn.",
        examples=[25],
    )


# ---------------------------------------------------------------------------
# Session resource
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    """Full session representation returned from GET/POST/PATCH endpoints."""

    id: str = Field(
        ...,
        description="UUID v4 uniquely identifying this session.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    name: str = Field(
        ...,
        description="Human-readable session name (editable).",
        examples=["Research — climate policy"],
    )
    owner_id: str = Field(
        ...,
        description="UUID v4 of the user who owns this session.",
    )
    state: AgentState = Field(
        ...,
        description="Current agent state for this session.",
        examples=["idle"],
    )
    config: SessionConfig = Field(
        ...,
        description="Active configuration for this session.",
    )
    token_counts: TokenCounts = Field(
        ...,
        description="Cumulative token usage in this session.",
    )
    active_tools: list[str] = Field(
        default_factory=list,
        description="Names of tools currently loaded in the active set.",
        examples=[["read_file", "web_search"]],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the session was created.",
    )
    updated_at: datetime = Field(
        ...,
        description="UTC timestamp of the last activity in this session.",
    )
    archived_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the session was archived (null if active).",
    )
    workspace_id: str | None = Field(
        default=None,
        description="UUID of the workspace this session belongs to (null for personal sessions).",
    )

    _ensure_utc = field_validator("created_at", "updated_at", "archived_at", mode="before")(
        ensure_utc
    )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


def _default_session_name() -> str:
    return datetime.now(UTC).strftime("Session %Y-%m-%d %H:%M")


class SessionCreateRequest(BaseModel):
    """Request body for POST /api/v1/sessions."""

    name: str = Field(
        default_factory=_default_session_name,
        max_length=256,
        description="Human-readable name for the new session.",
        examples=["Research — climate policy"],
    )
    config: SessionConfig = Field(
        default_factory=SessionConfig,
        description="Initial session configuration; all fields are optional.",
    )
    initial_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to pin into the active set immediately after session creation. "
            "Equivalent to calling PATCH /sessions/{id}/tools with load=[...] right after "
            "creation. Unknown tool names are silently skipped."
        ),
        examples=[["read_file", "write_file", "git_status"]],
    )
    auto_approve_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to auto-approve for the session (no confirmation prompt). "
            "Equivalent to calling PATCH /sessions/{id}/tools with auto_approve=[...]. "
            "Does not require the tool to be in initial_tools."
        ),
        examples=[["git_add", "git_commit"]],
    )
    workspace_id: str | None = Field(
        default=None,
        description="UUID of the workspace to create this session in. "
        "If provided, the caller must be a member of the workspace.",
    )


class SessionPatchRequest(BaseModel):
    """Request body for PATCH /api/v1/sessions/{id}.

    All fields are optional; only supplied fields are updated.
    """

    name: str | None = Field(
        default=None,
        max_length=256,
        description="New session name.",
        examples=["Renamed session"],
    )
    config: SessionConfig | None = Field(
        default=None,
        description="Partial config update; only supplied sub-fields are changed.",
    )
