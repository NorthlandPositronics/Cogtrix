"""System / observability schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SystemInfoOut(BaseModel):
    """System information returned from GET /api/v1/system/info."""

    version: str = Field(
        ...,
        description="Cogtrix version string.",
        examples=["0.1.14"],
    )
    api_version: str = Field(
        ...,
        description="API version prefix.",
        examples=["v1"],
    )
    platform: str = Field(
        ...,
        description="Host OS platform string.",
        examples=["Linux 6.14.0"],
    )
    python_version: str = Field(
        ...,
        description="Python interpreter version.",
        examples=["3.12.2"],
    )
    debug: bool = Field(..., description="True when debug logging is active.")
    verbose: bool = Field(..., description="True when verbose logging is active.")
    uptime_s: float = Field(
        ...,
        description="API server uptime in seconds.",
        examples=[3600.0],
    )
    started_at: datetime = Field(
        ...,
        description="UTC timestamp when the API server started.",
    )


class HealthOut(BaseModel):
    """Liveness check response from GET /api/v1/health."""

    status: Literal["ok"] = Field(
        default="ok",
        description="Always 'ok' for a live server (HTTP 200).",
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the liveness check.",
    )


class ReadinessComponentStatus(BaseModel):
    """Status of a single system component checked during readiness."""

    name: str = Field(..., description="Component name.", examples=["llm_provider"])
    ok: bool = Field(..., description="True when the component is ready.")
    latency_ms: int | None = Field(
        default=None,
        description="Check latency in milliseconds; null when not applicable.",
    )
    detail: str | None = Field(
        default=None,
        description="Human-readable status detail or error message.",
    )


class ReadinessOut(BaseModel):
    """Readiness check response from GET /api/v1/health/ready.

    Returns 200 when all critical components are ready, 503 when any are not.
    """

    ready: bool = Field(
        ...,
        description="True when all critical components are healthy.",
    )
    components: list[ReadinessComponentStatus] = Field(
        ...,
        description="Per-component readiness status.",
    )


class DebugToggleRequest(BaseModel):
    """Request body for POST /api/v1/system/debug."""

    debug: bool | None = Field(
        default=None,
        description="Target debug mode state; null toggles current state.",
    )
    verbose: bool | None = Field(
        default=None,
        description="Target verbose mode state; null leaves unchanged.",
    )


class TurnStatsOut(BaseModel):
    """Response statistics for a completed agent turn."""

    session_id: str = Field(..., description="Session that produced this turn.")
    message_id: str = Field(..., description="UUID of the AI message produced.")
    input_tokens: int = Field(..., description="Total input tokens for this turn.", examples=[320])
    output_tokens: int = Field(
        ...,
        description="Total output tokens for this turn.",
        examples=[88],
    )
    duration_ms: int = Field(
        ...,
        description="Wall-clock duration of the agent turn in milliseconds.",
        examples=[4200],
    )
    tool_calls: int = Field(
        ...,
        description="Number of tool calls made during this turn.",
        examples=[3],
    )
