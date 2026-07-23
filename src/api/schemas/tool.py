"""Tool management schemas.

A *tool* is a capability the agent can invoke.  Tools are either active
(immediately available to the LLM), on-demand (loadable via request_tools),
or disabled (blocked for the session).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

ToolStatus = Literal["active", "on_demand", "disabled", "auto_approved"]


# ---------------------------------------------------------------------------
# Tool resource
# ---------------------------------------------------------------------------


class ToolParameterSchema(BaseModel):
    """JSON Schema description of a single tool parameter."""

    name: str = Field(
        ...,
        description="Parameter name.",
        examples=["query"],
    )
    type: str = Field(
        ...,
        description="JSON Schema type string (string, integer, boolean, array, object).",
        examples=["string"],
    )
    description: str = Field(
        ...,
        description="Human-readable parameter description.",
        examples=["The search query string."],
    )
    required: bool = Field(
        ...,
        description="Whether this parameter must be supplied on every call.",
    )
    default: Any | None = Field(
        default=None,
        description="Default value when the parameter is omitted (null if required).",
    )
    enum: list[Any] | None = Field(
        default=None,
        description="Allowed values when the parameter is an enum; null otherwise.",
    )


class ToolOut(BaseModel):
    """Full tool representation including parameter schema.

    ``status`` reflects the tool's current state within the session:
    - 'active'       — loaded in the agent's active tool set
    - 'on_demand'    — available in the catalog, not yet loaded
    - 'disabled'     — blocked for this session
    - 'auto_approved'— active and pre-approved (no confirmation required)
    """

    name: str = Field(
        ...,
        description="Tool identifier (snake_case).",
        examples=["web_search"],
    )
    description: str = Field(
        ...,
        description="Full tool description shown to the LLM and in the UI.",
        examples=["Search the web using the configured search provider."],
    )
    short_description: str = Field(
        ...,
        description="One-line summary (≤120 chars).",
        examples=["Search the web."],
    )
    status: ToolStatus = Field(
        ...,
        description="Current tool status within the session.",
        examples=["on_demand"],
    )
    requires_confirmation: bool = Field(
        ...,
        description="True when the tool requires human confirmation before execution.",
    )
    parameters: list[ToolParameterSchema] = Field(
        default_factory=list,
        description="Tool input parameter schemas.",
    )
    module: str | None = Field(
        default=None,
        description="Source module name (e.g. 'web_search', 'file_ops').",
        examples=["web_search"],
    )
    is_mcp: bool = Field(
        default=False,
        description="True when this tool comes from an MCP server.",
    )
    mcp_server: str | None = Field(
        default=None,
        description="MCP server name when is_mcp is true; null for built-in tools.",
    )


class ToolSummary(BaseModel):
    """Lightweight tool entry for list responses."""

    name: str = Field(..., description="Tool identifier.", examples=["web_search"])
    short_description: str = Field(
        ...,
        description="One-line summary.",
        examples=["Search the web."],
    )
    status: ToolStatus = Field(
        ...,
        description="Current tool status.",
        examples=["on_demand"],
    )
    requires_confirmation: bool = Field(
        ...,
        description="True when confirmation is required before execution.",
    )
    is_mcp: bool = Field(
        default=False,
        description="True when this tool comes from an MCP server.",
    )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class ToolActionRequest(BaseModel):
    """Request body for PATCH /api/v1/sessions/{id}/tools.

    Supply exactly one of the action fields; the rest should be omitted or null.
    """

    load: list[str] | None = Field(
        default=None,
        description="Tool names to move from on-demand catalog into the active set.",
        examples=[["web_search", "read_file"]],
    )
    unload: list[str] | None = Field(
        default=None,
        description="Tool names to move back from active to on-demand.",
        examples=[["read_file"]],
    )
    enable: list[str] | None = Field(
        default=None,
        description="Tool names to re-enable after being disabled.",
        examples=[["shell"]],
    )
    disable: list[str] | None = Field(
        default=None,
        description="Tool names to disable (block) for this session.",
        examples=[["shell"]],
    )
    auto_approve: list[str] | None = Field(
        default=None,
        description="Tool names to auto-approve (skip confirmation) for this session.",
        examples=[["web_search"]],
    )
    revoke_approval: list[str] | None = Field(
        default=None,
        description="Tool names to remove from the auto-approved set.",
        examples=[["web_search"]],
    )
