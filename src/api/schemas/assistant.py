"""Assistant mode (WhatsApp/Telegram daemon) schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc

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

    _ensure_utc = field_validator("send_at", "created_at", mode="before")(ensure_utc)


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

    _ensure_utc = field_validator("fire_at", "created_at", mode="before")(ensure_utc)


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

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)

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


# ---------------------------------------------------------------------------
# Outbound messaging
# ---------------------------------------------------------------------------


class OutboundRequest(BaseModel):
    """Request body for POST /api/v1/assistant/outbound."""

    contact_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Phonebook contact name to message.",
        examples=["Alice"],
    )
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Operator instructions for the agent (what to tell the contact).",
        examples=["Ask Alice about her availability next week for the project review."],
    )
    channel: str | None = Field(
        default=None,
        description=(
            "Preferred channel ('whatsapp' or 'telegram'). "
            "If null, uses the first available channel for this contact."
        ),
        examples=["whatsapp"],
    )


class OutboundResponse(BaseModel):
    """Response body for POST /api/v1/assistant/outbound."""

    session_key: str = Field(
        ...,
        description="Chat session key used for this outbound message.",
        examples=["whatsapp::+1234567890@c.us"],
    )
    channel: str = Field(..., description="Channel used.", examples=["whatsapp"])
    chat_id: str = Field(
        ..., description="Resolved chat identifier.", examples=["+1234567890@c.us"]
    )
    contact_name: str = Field(..., description="Contact name from the request.", examples=["Alice"])
    response_text: str = Field(
        ...,
        description="The agent-generated message that was sent to the contact.",
    )
    message_id: str | None = Field(
        default=None,
        description="Channel message ID of the sent message; null if delivery failed.",
    )


# ---------------------------------------------------------------------------
# Campaigns (Level 2 outbound)
# ---------------------------------------------------------------------------

CampaignStatus = Literal["draft", "active", "paused", "completed", "cancelled"]
CampaignTargetStatus = Literal["pending", "active", "completed", "failed", "escalated"]


class CampaignTargetIn(BaseModel):
    """A target contact for a campaign."""

    contact_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Phonebook contact name.",
        examples=["Alice"],
    )
    channel: str | None = Field(
        default=None,
        description="Preferred channel. If null, auto-resolved from phonebook.",
        examples=["whatsapp"],
    )


class CampaignTargetOut(BaseModel):
    """Per-target progress within a campaign."""

    contact_name: str = Field(..., description="Contact name.", examples=["Alice"])
    channel: str = Field(..., description="Channel name.", examples=["whatsapp"])
    chat_id: str = Field(..., description="Resolved chat identifier.", examples=["+123@c.us"])
    status: CampaignTargetStatus = Field(..., description="Target status.")
    follow_ups_sent: int = Field(..., description="Number of follow-ups sent.", examples=[0])
    last_outbound_at: datetime | None = Field(
        default=None, description="Last outbound message timestamp."
    )
    last_reply_at: datetime | None = Field(
        default=None, description="Last reply received timestamp."
    )
    completion_reason: str | None = Field(
        default=None, description="Reason for completion/failure/escalation."
    )


class CampaignCreateRequest(BaseModel):
    """Request body for POST /api/v1/assistant/campaigns."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Campaign name.",
        examples=["Spring bike sale outreach"],
    )
    goal: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="What the campaign is trying to achieve.",
        examples=["Schedule a meeting to discuss new bike models"],
    )
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Operator instructions for the agent.",
        examples=["Reach out about our new bike models and the spring sale."],
    )
    targets: list[CampaignTargetIn] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Contacts to target.",
    )
    max_follow_ups: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Maximum follow-ups per target before escalation.",
    )
    follow_up_interval_hours: float = Field(
        default=24.0,
        ge=0.5,
        le=720.0,
        description="Hours between follow-up attempts.",
    )
    auto_launch: bool = Field(
        default=False,
        description="If true, launch immediately after creation.",
    )


class CampaignUpdateRequest(BaseModel):
    """Request body for PATCH /api/v1/assistant/campaigns/{id}."""

    name: str | None = Field(default=None, max_length=256)
    goal: str | None = Field(default=None, max_length=1024)
    instructions: str | None = Field(default=None, max_length=8192)
    status: CampaignStatus | None = Field(
        default=None,
        description="Set to 'paused' or 'cancelled' to change campaign state.",
    )
    max_follow_ups: int | None = Field(default=None, ge=0, le=20)
    follow_up_interval_hours: float | None = Field(default=None, ge=0.5, le=720.0)


class CampaignOut(BaseModel):
    """Full campaign representation."""

    id: str = Field(..., description="Campaign UUID.")
    name: str = Field(..., description="Campaign name.")
    goal: str = Field(..., description="Campaign goal.")
    instructions: str = Field(..., description="Operator instructions.")
    targets: list[CampaignTargetOut] = Field(..., description="Per-target progress.")
    max_follow_ups: int = Field(..., description="Max follow-ups per target.")
    follow_up_interval_hours: float = Field(..., description="Hours between follow-ups.")
    status: CampaignStatus = Field(..., description="Campaign status.")
    created_at: datetime = Field(..., description="UTC creation timestamp.")
    updated_at: datetime = Field(..., description="UTC last update timestamp.")

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _coerce_and_ensure_utc(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            from datetime import UTC
            from datetime import datetime as _dt

            return _dt.fromtimestamp(v, tz=UTC)
        return ensure_utc(v)
