"""Assistant mode (WhatsApp/Telegram daemon) schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Service status
# ---------------------------------------------------------------------------

AssistantStatus = Literal["running", "stopped", "starting", "stopping", "error"]
FilterMode = Literal["none", "allow", "ignore", "blacklist"]
ChannelType = Literal["whatsapp", "telegram"]


class ChannelStatusOut(BaseModel):
    """Runtime status of a single assistant channel."""

    name: str = Field(..., description="Channel name (e.g. 'whatsapp', 'telegram').")
    type: ChannelType = Field(..., description="Channel type.")
    enabled: bool = Field(..., description="True when the channel is configured and enabled.")
    connected: bool = Field(..., description="True when the channel is actively polling.")
    active_chats: int = Field(
        ...,
        description="Number of chat sessions currently in memory.",
        examples=[3],
    )
    poll_interval_s: float = Field(
        ...,
        description="Current adaptive poll interval in seconds.",
        examples=[5.0],
    )
    error: str | None = Field(
        default=None,
        description="Last connection error; null when healthy.",
    )


class AssistantStatusOut(BaseModel):
    """Full assistant service status."""

    status: AssistantStatus = Field(..., description="Overall service status.")
    channels: list[ChannelStatusOut] = Field(
        default_factory=list,
        description="Per-channel runtime status.",
    )
    started_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the service was last started.",
    )
    uptime_s: float | None = Field(
        default=None,
        description="Service uptime in seconds; null when stopped.",
    )


class AssistantStartRequest(BaseModel):
    """Request body for POST /api/v1/assistant/start (admin only)."""

    force_restart: bool = Field(
        default=False,
        description="Stop and restart the service if it is already running.",
    )


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------


class ChatSessionOut(BaseModel):
    """An active assistant chat session (one per chat_id/channel pair)."""

    session_key: str = Field(
        ...,
        description="Composite key: '{channel}::{chat_id}'.",
        examples=["whatsapp::+1234567890"],
    )
    channel: str = Field(..., description="Channel name.")
    chat_id: str = Field(
        ...,
        description="Channel-specific chat identifier.",
        examples=["+1234567890@c.us"],
    )
    display_name: str | None = Field(
        default=None,
        description="Resolved display name from phonebook; null if not in phonebook.",
        examples=["Alice"],
    )
    message_count: int = Field(
        ...,
        description="Total messages in this chat session's memory.",
    )
    last_activity: datetime | None = Field(
        default=None,
        description="UTC timestamp of the last message processed.",
    )
    memory_mode: str = Field(..., description="Active memory mode for this chat session.")
    is_locked: bool = Field(
        ...,
        description="True when a message is currently being processed for this chat.",
    )


# ---------------------------------------------------------------------------
# Scheduled messages
# ---------------------------------------------------------------------------


class ScheduledMessageOut(BaseModel):
    """A pending scheduled message."""

    id: str = Field(..., description="UUID v4 message ID.")
    chat_id: str = Field(..., description="Target chat identifier.")
    channel: str = Field(..., description="Target channel name.")
    recipient: str = Field(
        ...,
        description="Human-readable recipient (resolved phone / name / chat_id).",
    )
    text: str = Field(..., description="Message text to send.")
    send_at: datetime = Field(..., description="Scheduled delivery UTC timestamp.")
    created_at: datetime = Field(..., description="UTC creation timestamp.")
    attempts: int = Field(..., description="Delivery attempt count.", examples=[0])
    max_attempts: int = Field(..., description="Maximum allowed delivery attempts.", examples=[3])
    status: Literal["pending", "firing", "sent", "failed", "cancelled"] = Field(
        ...,
        description="Delivery status.",
    )


class ScheduledMessageEditRequest(BaseModel):
    """Request body for PATCH /api/v1/assistant/scheduled/{id}."""

    text: str | None = Field(
        default=None,
        max_length=4096,
        description="New message text; null leaves unchanged.",
    )
    send_at: datetime | None = Field(
        default=None,
        description="New delivery UTC timestamp; null leaves unchanged.",
    )


# ---------------------------------------------------------------------------
# Deferred messages
# ---------------------------------------------------------------------------


class DeferredRecordOut(BaseModel):
    """A pending deferred re-processing record."""

    session_key: str = Field(
        ...,
        description="Chat session key this record belongs to.",
        examples=["whatsapp::+1234567890"],
    )
    fire_at: datetime = Field(..., description="UTC timestamp when re-processing will fire.")
    pending_messages: list[str] = Field(
        ...,
        description="Message texts queued for the re-processing batch.",
    )
    depth: int = Field(..., description="Current deferral depth.", examples=[1])
    max_depth: int = Field(..., description="Maximum allowed deferral depth.", examples=[3])
    status: Literal["pending", "firing"] = Field(..., description="Record state.")
    created_at: datetime = Field(..., description="UTC creation timestamp.")


# ---------------------------------------------------------------------------
# Phonebook / contacts
# ---------------------------------------------------------------------------


class ContactOut(BaseModel):
    """A phonebook contact entry."""

    name: str = Field(
        ...,
        description="Contact display name (phonebook key).",
        examples=["Alice"],
    )
    identifiers: list[str] = Field(
        ...,
        description="List of channel identifiers for this contact.",
        examples=[["+1234567890", "alice_tg"]],
    )
    channels: list[str] = Field(
        default_factory=list,
        description="Channels this contact is configured for.",
        examples=[["whatsapp", "telegram"]],
    )
    prompt: str | None = Field(
        default=None,
        description="Per-contact system prompt override; null uses the global prompt.",
    )
    filter_mode: FilterMode | None = Field(
        default=None,
        description="Per-contact filter mode override; null uses the channel default.",
    )


# ---------------------------------------------------------------------------
# Guardrails dashboard
# ---------------------------------------------------------------------------


class ViolationRecordOut(BaseModel):
    """A security violation event recorded by the guardrail pipeline."""

    chat_id: str = Field(..., description="Chat that generated the violation.")
    channel: str = Field(..., description="Channel name.")
    violation_type: str = Field(
        ...,
        description="Guard that triggered: 'input', 'encoding', 'tool_call', 'rate_limit', 'llm_judge'.",
        examples=["input"],
    )
    detail: str = Field(..., description="Human-readable violation description.")
    timestamp: datetime = Field(..., description="UTC timestamp of the violation.")


class GuardrailStatusOut(BaseModel):
    """Guardrail pipeline status snapshot."""

    blacklisted_chats: list[str] = Field(
        ...,
        description="Chat IDs currently on the auto-blacklist.",
    )
    total_violations: int = Field(
        ...,
        description="Total recorded violations across all chats.",
        examples=[12],
    )
    recent_violations: list[ViolationRecordOut] = Field(
        default_factory=list,
        description="Most recent 50 violation records.",
    )


# ---------------------------------------------------------------------------
# Knowledge store
# ---------------------------------------------------------------------------


class KnowledgeFactOut(BaseModel):
    """A fact in the shared knowledge store."""

    id: str = Field(..., description="UUID v4 fact identifier.")
    text: str = Field(..., description="Fact text.")
    source_chat: str | None = Field(
        default=None,
        description="Chat session key that produced this fact.",
    )
    source_channel: str | None = Field(
        default=None,
        description="Channel name that produced this fact.",
    )
    created_at: datetime = Field(..., description="UTC extraction timestamp.")
    relevance_score: float | None = Field(
        default=None,
        description="Cosine similarity score from the last retrieval (null when listed directly).",
        examples=[0.87],
    )


class KnowledgeSearchRequest(BaseModel):
    """Request body for POST /api/v1/assistant/knowledge/search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Semantic search query.",
        examples=["What does Alice prefer for breakfast?"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of facts to return.",
        examples=[10],
    )
