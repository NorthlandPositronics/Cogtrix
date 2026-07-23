"""Message request/response schemas.

Messages are the turn-level records within a session.  Each user message
initiates an agent turn that produces one AI message.  Tool calls emitted
during the turn are embedded in the AI message as ``tool_calls``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc

# ---------------------------------------------------------------------------
# Tool call sub-objects (embedded in AI messages)
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """A single tool invocation embedded inside an AI message."""

    tool_call_id: str = Field(
        ...,
        description="Unique ID assigned by the LLM for this tool call.",
        examples=["call_abc123"],
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool that was invoked.",
        examples=["web_search"],
    )
    input: dict[str, Any] = Field(
        ...,
        description="Arguments passed to the tool.",
    )
    output: str | None = Field(
        default=None,
        description="Tool output string; null if the call is still in progress.",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Tool execution duration in milliseconds.",
        examples=[340],
    )
    error: str | None = Field(
        default=None,
        description="Error string if the tool call failed; null on success.",
    )


# ---------------------------------------------------------------------------
# Message resource
# ---------------------------------------------------------------------------

MessageRole = Literal["user", "assistant", "system", "tool"]


class MessageOut(BaseModel):
    """A single message in the conversation history.

    ``role`` is 'user' for human input, 'assistant' for agent responses,
    'tool' for tool result summaries, and 'system' for injected context.
    ``tool_calls`` is non-empty only on 'assistant' messages where the agent
    invoked one or more tools.
    """

    id: str = Field(
        ...,
        description="UUID v4 uniquely identifying this message.",
        examples=["7a3c1b2e-5d4f-11ee-be56-0242ac120002"],
    )
    session_id: str = Field(
        ...,
        description="UUID v4 of the session this message belongs to.",
    )
    role: MessageRole = Field(
        ...,
        description="Message role: 'user', 'assistant', 'tool', or 'system'.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
        examples=["What is the capital of France?"],
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Tool calls emitted during this message (assistant messages only).",
    )
    token_counts: dict[str, int] | None = Field(
        default=None,
        description="Per-message token breakdown: {input, output}.",
        examples=[{"input": 320, "output": 88}],
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the message was created.",
    )

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class SendMessageRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{id}/messages.

    Initiates an agent turn.  The response is the queued user message;
    the agent reply streams over the WebSocket connection for this session.
    """

    content: str = Field(
        ...,
        min_length=1,
        max_length=65536,
        description="User message text.",
        examples=["What is the capital of France?"],
    )
    mode: Literal["normal", "think", "delegate"] = Field(
        default="normal",
        description=(
            "Agent execution mode: "
            "'normal' for standard turn, "
            "'think' to force the deep-think pipeline, "
            "'delegate' to force research delegation."
        ),
        examples=["normal"],
    )
    optimize_prompt: bool | None = Field(
        default=None,
        description="Override session-level prompt_optimizer for this turn only; null uses session setting.",
    )


class SyncTurnOut(BaseModel):
    """Response body for POST /api/v1/sessions/{id}/messages?sync=true.

    Returned with HTTP 200 once the agent turn has completed.  The caller
    receives the full response text without needing a WebSocket connection.
    """

    message_id: str = Field(..., description="UUID of the AI message created for this turn.")
    text: str = Field(..., description="Assembled agent response text.")
    total_tokens: int = Field(
        ..., description="Total tokens consumed (input + output).", examples=[1800]
    )
    input_tokens: int = Field(..., description="Input tokens for this turn.", examples=[1420])
    output_tokens: int = Field(..., description="Output tokens for this turn.", examples=[380])
    duration_ms: int = Field(
        ..., description="Wall-clock agent turn duration in milliseconds.", examples=[4200]
    )
    tool_calls: int = Field(
        ..., description="Number of tool calls made during this turn.", examples=[3]
    )
    blocked_by_guardrails: bool = Field(
        default=False,
        description=(
            "True when the input was refused by the API content guardrail (#2056) "
            "before the model ran. When true, `text` is a generic refusal and "
            "token/tool counts are zero. Always false unless api.guardrails is enabled."
        ),
    )
    guardrail_reason: str | None = Field(
        default=None,
        description="Guard reason when blocked_by_guardrails is true, else null.",
    )


class ClearHistoryRequest(BaseModel):
    """Request body for DELETE /api/v1/sessions/{id}/messages.

    Optionally keep the last N messages instead of clearing everything.
    """

    keep_last: int | None = Field(
        default=None,
        ge=0,
        description="Keep the last N messages; null or 0 clears all history.",
        examples=[10],
    )


# ---------------------------------------------------------------------------
# Tool confirmation
# ---------------------------------------------------------------------------

ToolConfirmAction = Literal["allow", "deny", "allow_all", "disable", "forbid_all", "cancel"]


class ToolConfirmRequest(BaseModel):
    """Sent by the frontend when the user responds to a tool confirmation prompt.

    This is dispatched via WebSocket (type: ``tool_confirm``) rather than REST,
    but the schema is defined here for reference and OpenAPI documentation.

    Actions map to the CLI equivalents:
    - allow      → y  (allow this invocation once)
    - deny       → n  (deny this invocation; agent may retry)
    - allow_all  → a  (approve this tool for the entire session)
    - disable    → d  (disable this tool for the entire session)
    - forbid_all → f  (block all further tool requests this turn)
    - cancel     → c  (cancel the current agent workflow)
    """

    confirmation_id: str = Field(
        ...,
        description="The confirmation_id from the tool_confirm_request WebSocket message.",
        examples=["conf_3f2504e0"],
    )
    action: ToolConfirmAction = Field(
        ...,
        description="User decision for the pending tool confirmation.",
        examples=["allow"],
    )
