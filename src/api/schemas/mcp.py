"""MCP server management schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MCPServerStatus = Literal["connected", "disconnected", "error", "connecting"]


class MCPToolSummary(BaseModel):
    """Lightweight MCP tool entry."""

    name: str = Field(..., description="Tool name as registered by the MCP server.")
    description: str = Field(..., description="One-line tool description.")


class MCPServerOut(BaseModel):
    """An MCP server configuration and runtime status."""

    name: str = Field(
        ...,
        description="Server alias (config key).",
        examples=["my-mcp-server"],
    )
    status: MCPServerStatus = Field(
        ...,
        description="Current connection status.",
        examples=["connected"],
    )
    transport: Literal["stdio", "sse"] = Field(
        ...,
        description="Transport type: stdio (subprocess) or sse (HTTP).",
        examples=["stdio"],
    )
    url: str | None = Field(
        default=None,
        description="SSE endpoint URL (sse transport only).",
    )
    command: str | None = Field(
        default=None,
        description="Subprocess command (stdio transport only).",
        examples=["npx"],
    )
    args: list[str] = Field(
        default_factory=list,
        description="Subprocess arguments (stdio transport only).",
        examples=[["@modelcontextprotocol/server-filesystem", "/data"]],
    )
    requires_confirmation: bool = Field(
        ...,
        description="True when all tools from this server require confirmation before use.",
    )
    tools: list[MCPToolSummary] = Field(
        default_factory=list,
        description="Tools discovered from this server; empty when disconnected.",
    )
    error: str | None = Field(
        default=None,
        description="Last connection error message; null when connected.",
    )
    connected_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the last successful connection.",
    )


class MCPServerAddRequest(BaseModel):
    """Request body for POST /api/v1/mcp/servers."""

    name: str = Field(
        ...,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Server alias (alphanumeric, underscores, hyphens).",
        examples=["my-mcp-server"],
    )
    transport: Literal["stdio", "sse"] = Field(
        ...,
        description="Transport type.",
        examples=["stdio"],
    )
    url: str | None = Field(
        default=None,
        description="SSE endpoint URL (required for sse transport).",
    )
    command: str | None = Field(
        default=None,
        description="Subprocess command (required for stdio transport).",
        examples=["npx"],
    )
    args: list[str] = Field(
        default_factory=list,
        description="Subprocess arguments.",
        examples=[["@modelcontextprotocol/server-filesystem", "/data"]],
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Whether tools from this server require confirmation.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the subprocess.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for SSE transport (e.g. Authorization).",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection timeout in seconds.",
    )
