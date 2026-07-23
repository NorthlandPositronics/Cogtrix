"""WebSocket connection manager and message dispatcher.

Each WebSocket connection is scoped to a single session via:
    ws://host/ws/v1/sessions/{session_id}

All messages use a typed envelope:
    {
        "type": "<message_type>",
        "session_id": "<uuid>",
        "payload": { ... },
        "seq": <int>
    }

The server sends ``done`` when the agent turn is complete.
The client sends ``ping`` every 30 s; the server responds with ``pong``.
Dead connections are dropped after 90 s of silence.

WebSocket authentication: the client must include the Bearer token in the
``Authorization`` header (or as ``?token=<jwt>`` query parameter for
environments that cannot set custom headers, e.g., browser WebSocket API).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

log = logging.getLogger("cogtrix.api.ws")

# Seconds to keep buffered messages available for reconnecting clients.
_RECONNECT_BUFFER_TTL = 30.0

# ---------------------------------------------------------------------------
# Typed message envelope
# ---------------------------------------------------------------------------

ServerMessageType = Literal[
    "token",
    "tool_start",
    "tool_end",
    "tool_confirm_request",
    "agent_state",
    "memory_update",
    "error",
    "done",
    "pong",
    "log_line",
]

ClientMessageType = Literal[
    "user_message",
    "tool_confirm",
    "ping",
    "cancel",
]


class ServerMessage(BaseModel):
    """Typed envelope for all server-to-client WebSocket messages.

    ``seq`` is monotonically increasing per connection.  The frontend uses it
    to detect dropped messages and request reconnection.
    """

    type: ServerMessageType = Field(
        ...,
        description="Message type discriminator.",
        examples=["token"],
    )
    session_id: str = Field(
        ...,
        description="UUID v4 of the session this message belongs to.",
    )
    payload: dict[str, Any] = Field(
        ...,
        description="Type-specific payload (see WebSocket protocol documentation).",
    )
    seq: int = Field(
        ...,
        description="Monotonically increasing sequence number per connection.",
        examples=[42],
    )
    ts: str = Field(
        ...,
        description="ISO 8601 UTC server timestamp.",
        examples=["2026-03-04T12:34:56.789Z"],
    )


class ClientMessage(BaseModel):
    """Typed envelope for all client-to-server WebSocket messages."""

    type: ClientMessageType = Field(
        ...,
        description="Message type discriminator.",
        examples=["ping"],
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific payload.",
    )


# ---------------------------------------------------------------------------
# Payload schemas (documented; not enforced on the wire — WS is schema-free)
# ---------------------------------------------------------------------------


class TokenPayload(BaseModel):
    """Payload for type='token' — incremental LLM output token."""

    text: str = Field(
        ...,
        description="Incremental token text to append to the response buffer.",
        examples=[" Paris"],
    )
    final: bool = Field(
        default=False,
        description=(
            "True when tokens are generated after at least one tool call has completed "
            "(final response phase); False for preamble tokens before tools run."
        ),
    )


class ToolStartPayload(BaseModel):
    """Payload for type='tool_start' — agent began executing a tool."""

    tool: str = Field(
        ...,
        description="Tool name being invoked.",
        examples=["web_search"],
    )
    tool_call_id: str = Field(
        ...,
        description="Unique ID for this invocation (links to tool_end).",
        examples=["call_abc123"],
    )
    input: dict[str, Any] = Field(
        ...,
        description="Arguments passed to the tool.",
    )


class ToolEndPayload(BaseModel):
    """Payload for type='tool_end' — tool execution completed."""

    tool: str = Field(..., description="Tool name.")
    tool_call_id: str = Field(..., description="Unique invocation ID (matches tool_start).")
    duration_ms: int = Field(
        ...,
        description="Wall-clock execution time in milliseconds.",
        examples=[340],
    )
    error: str | None = Field(
        default=None,
        description="Error string if the tool failed; null on success.",
    )


class ToolConfirmRequestPayload(BaseModel):
    """Payload for type='tool_confirm_request' — agent awaits user confirmation.

    The frontend must display a confirmation dialog and send back a
    ``tool_confirm`` client message with the user's decision.
    The connection will block until a response is received or the turn is cancelled.
    """

    confirmation_id: str = Field(
        ...,
        description="Opaque ID to echo in the tool_confirm response.",
        examples=["conf_3f2504e0"],
    )
    tool: str = Field(..., description="Tool name requiring confirmation.", examples=["write_file"])
    parameters: dict[str, Any] = Field(
        ...,
        description="Parameters the tool will be called with (sorted, large values last).",
    )
    message: str = Field(
        ...,
        description="Human-readable description of what the tool will do.",
        examples=["Write 2 KB to /home/user/report.md"],
    )


class AgentStatePayload(BaseModel):
    """Payload for type='agent_state' — agent state machine transition."""

    state: Literal[
        "idle",
        "thinking",
        "analyzing",
        "researching",
        "deep_thinking",
        "writing",
        "delegating",
        "done",
        "error",
    ] = Field(
        ...,
        description="New agent state.",
        examples=["thinking"],
    )


class MemoryUpdatePayload(BaseModel):
    """Payload for type='memory_update' — memory compaction occurred."""

    mode: str = Field(
        ...,
        description="Active memory mode after the update.",
        examples=["conversation"],
    )
    tokens_used: int = Field(
        ...,
        description="Estimated context token count after compression.",
        examples=[1200],
    )
    summarized: bool = Field(
        default=False,
        description="True when a summarization pass ran during this update.",
    )


class ErrorPayload(BaseModel):
    """Payload for type='error' — agent-level error (not a connection error)."""

    code: str = Field(
        ...,
        description="Machine-readable error code.",
        examples=["TOOL_EXPANSION_FAILED"],
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
        examples=["web_search could not be loaded."],
    )


class DonePayload(BaseModel):
    """Payload for type='done' — agent turn is complete."""

    message_id: str = Field(
        ...,
        description="UUID of the AI message created for this turn.",
    )
    total_tokens: int = Field(
        ...,
        description="Total tokens consumed in this turn (input + output).",
        examples=[1800],
    )
    input_tokens: int = Field(..., description="Input tokens for this turn.", examples=[1420])
    output_tokens: int = Field(..., description="Output tokens for this turn.", examples=[380])
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


class LogLinePayload(BaseModel):
    """Payload for type='log_line' — live log stream (GET /ws/v1/logs).

    This message type is emitted on the log-stream WebSocket only, not on
    session WebSockets.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        ...,
        description="Log level.",
    )
    logger: str = Field(
        ...,
        description="Logger name.",
        examples=["cogtrix.orchestration.runner"],
    )
    message: str = Field(..., description="Log message text.")
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of the log record.",
    )


# ---------------------------------------------------------------------------
# Client message payloads
# ---------------------------------------------------------------------------


class UserMessagePayload(BaseModel):
    """Payload for type='user_message' — send a message over WS instead of REST."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=65536,
        description="User message text.",
    )
    mode: Literal["normal", "think", "delegate"] = Field(
        default="normal",
        description="Execution mode override.",
    )


class ToolConfirmPayload(BaseModel):
    """Payload for type='tool_confirm' — user decision on a tool confirmation prompt."""

    confirmation_id: str = Field(
        ...,
        description="The confirmation_id from the tool_confirm_request message.",
    )
    action: Literal["allow", "deny", "allow_all", "disable", "forbid_all", "cancel"] = Field(
        ...,
        description="User decision.",
        examples=["allow"],
    )


# ---------------------------------------------------------------------------
# Connection manager stub
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections keyed by session_id.

    Thread-safe for concurrent FastAPI handler coroutines.  Each session may
    have at most one active WebSocket connection; a second connection replaces
    the first (the old connection receives a close).

    The sequence counter is per-connection, not per-session, and resets to 0
    on reconnect.

    A 30-second reconnection buffer is maintained per session: outgoing messages
    are kept in a timestamped deque after disconnect.  On reconnect with
    ``?last_seq=N``, messages with seq > N are replayed.
    """

    def __init__(self) -> None:
        # {session_id: websocket}
        self._connections: dict[str, Any] = {}
        # {session_id: seq_counter}
        self._seq: dict[str, int] = {}
        # {session_id: deque[(seq, ts, json_str)]}
        self._buffers: dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: Any) -> None:
        """Register a new WebSocket connection for a session.

        If an existing connection is present it is closed gracefully before
        the new one is registered.  The sequence counter resets to 0.
        """
        async with self._lock:
            old_ws = self._connections.pop(session_id, None)
            self._connections[session_id] = websocket
            self._seq[session_id] = 0
            # Keep the buffer so reconnecting clients can replay missed messages.
            if session_id not in self._buffers:
                self._buffers[session_id] = deque()

        # Close the displaced connection outside the lock to avoid holding
        # the lock across I/O — other coroutines need the lock for send().
        if old_ws is not None:
            try:
                await old_ws.close(code=1001)
            except Exception:
                pass

    async def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection from the registry.

        The message buffer is kept for ``_RECONNECT_BUFFER_TTL`` seconds to
        support reconnects; it will be pruned naturally on the next ``send`` call
        or when the session is evicted.
        """
        async with self._lock:
            self._connections.pop(session_id, None)

    async def send(
        self, session_id: str, message_type: ServerMessageType, payload: dict[str, Any]
    ) -> None:
        """Send a typed message to the WebSocket for the given session.

        Increments the sequence counter automatically.
        No-op when no connection exists for the session, but the message is still
        buffered so it can be replayed on reconnect.

        Lock is held only for the minimal critical section: allocating the seq
        number and reading the ws/buf references.  JSON serialisation (CPU-bound
        Pydantic work) and the network send (I/O) both happen outside the lock.
        A second brief lock acquisition appends the built message to the replay
        buffer; two lock acquisitions per send is cheaper than holding one lock
        across serialisation for all concurrent senders.
        """
        # Phase 1: allocate seq and capture mutable state references.
        async with self._lock:
            seq = self._seq.get(session_id, 0)
            self._seq[session_id] = seq + 1
            self._buffers.setdefault(session_id, deque())
            ws = self._connections.get(session_id)

        # CPU-bound serialisation outside the lock so other coroutines can send.
        json_str = self._build_message(session_id, message_type, payload, seq)

        # Phase 2: append to replay buffer (separate brief lock acquisition).
        # Re-fetch buf from _buffers instead of using the Phase-1 reference — a
        # concurrent connect() call may have replaced the deque between the two
        # lock acquisitions, and appending to the stale deque would lose this
        # message from replay_missed.
        async with self._lock:
            current_buf = self._buffers.setdefault(session_id, deque())
            current_buf.append((seq, datetime.now(UTC).timestamp(), json_str))
            self._prune_buffer(current_buf)

        # Network I/O always outside the lock.
        if ws is not None:
            try:
                await ws.send_text(json_str)
            except Exception as exc:
                log.debug("WebSocket send failed for session %s: %s", session_id, exc)

    async def replay_missed(self, session_id: str, last_seq: int) -> None:
        """Replay buffered messages with seq > last_seq to the current connection.

        Called immediately after a successful reconnect so the client catches up
        on messages it missed during the disconnect window.
        """
        async with self._lock:
            ws = self._connections.get(session_id)
            buf = list(self._buffers.get(session_id, deque()))

        if ws is None:
            return

        for seq, _ts, json_str in buf:
            if seq > last_seq:
                try:
                    await ws.send_text(json_str)
                except Exception as exc:
                    log.debug("Replay send failed for session %s seq %d: %s", session_id, seq, exc)
                    break

    def _build_message(
        self, session_id: str, message_type: ServerMessageType, payload: dict[str, Any], seq: int
    ) -> str:
        """Serialise a ServerMessage to JSON."""
        now = datetime.now(UTC)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        msg = ServerMessage(
            type=message_type,
            session_id=session_id,
            payload=payload,
            seq=seq,
            ts=ts,
        )
        return msg.model_dump_json()

    @staticmethod
    def _prune_buffer(buf: deque) -> None:
        """Remove entries older than the reconnect buffer TTL from the left."""
        cutoff = datetime.now(UTC).timestamp() - _RECONNECT_BUFFER_TTL
        while buf and buf[0][1] < cutoff:
            buf.popleft()


# Module-level singleton used by route handlers.
manager = ConnectionManager()
