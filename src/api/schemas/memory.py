"""Memory management schemas.

Memory state reflects the current contents of the session's
``ConversationMemoryManager`` — the sliding window, incremental summary,
and optional vector recall index.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc
from src.api.schemas.session import MemoryMode

# ---------------------------------------------------------------------------
# Memory state snapshot
# ---------------------------------------------------------------------------


class MemoryStateOut(BaseModel):
    """Current memory state for a session.

    ``mode`` is the active memory strategy.
    ``summary`` is the most recent LLM-generated summary of older messages.
    ``window_messages`` is the count of messages kept verbatim in the sliding window.
    ``summarized_messages`` is the count of messages compressed into the summary.
    ``tokens_used`` is the estimated token count of the current context.
    ``vector_recall_enabled`` indicates whether evicted messages are embedded for retrieval.
    ``mode_meta`` contains mode-specific state (goals for reasoning mode, code tasks, etc.).
    """

    session_id: str = Field(
        ...,
        description="UUID v4 of the owning session.",
    )
    mode: MemoryMode = Field(
        ...,
        description="Active memory mode.",
        examples=["conversation"],
    )
    summary: str | None = Field(
        default=None,
        description="LLM-generated summary of messages outside the sliding window; null when empty.",
    )
    window_messages: int = Field(
        ...,
        description="Number of messages kept verbatim in the sliding window.",
        examples=[20],
    )
    summarized_messages: int = Field(
        ...,
        description="Number of messages compressed into the summary.",
        examples=[45],
    )
    tokens_used: int = Field(
        ...,
        description="Estimated token count of the current serialized context.",
        examples=[8400],
    )
    context_window: int = Field(
        ...,
        description="Maximum context window for the current model (tokens).",
        examples=[131072],
    )
    vector_recall_enabled: bool = Field(
        ...,
        description="True when the optional vector recall index is active.",
    )
    mode_meta: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Mode-specific metadata: goals/decisions for reasoning mode, "
            "code tasks for code mode, entities for conversation mode."
        ),
    )
    updated_at: datetime = Field(
        ...,
        description="UTC timestamp of the last memory update.",
    )

    _ensure_utc = field_validator("updated_at", mode="before")(ensure_utc)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class MemoryModeSwitchRequest(BaseModel):
    """Request body for PATCH /api/v1/sessions/{id}/memory."""

    mode: MemoryMode = Field(
        ...,
        description="Target memory mode to switch to.",
        examples=["reasoning"],
    )
