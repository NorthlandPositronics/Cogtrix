# Cogtrix WebSocket Protocol

Version: v1
Endpoint: `ws://host/ws/v1/sessions/{session_id}`
Log stream: `ws://host/ws/v1/logs`

Related documents:
- `docs/API/OVERVIEW.md` — API orientation, quick start, authentication model
- `docs/API/CLIENT_CONTRACT.md` — TypeScript types and full WebSocket usage example (`SessionSocket` class)
- `docs/API/WEBUI_DEVELOPMENT_GUIDE.md` — React integration patterns for streaming

---

## 1. Overview

WebSockets are used exclusively for real-time streaming surfaces:
- Token-by-token agent output streaming (session WebSocket)
- Tool execution progress (session WebSocket)
- Tool confirmation dialogs (session WebSocket)
- Live log streaming (log WebSocket, admin only)

All other operations use the REST API.

---

## 2. Authentication

The JWT bearer token (or `cgx_live_` API key) may be provided via either of
two paths. Both reach the same downstream validation pipeline; pick whichever
matches your client environment.

### 2.1 Authorization header — CLI / SDK clients

```
Authorization: Bearer <jwt>
```

Case-insensitive on the scheme (``bearer``/``Bearer``/``BEARER`` all accepted).
This is the canonical path for CLI tools, server-side SDKs, and any HTTP
client that can set custom request headers on the WebSocket upgrade.

### 2.2 Sec-WebSocket-Protocol — browser clients (#1887)

```
Sec-WebSocket-Protocol: bearer, <jwt>
```

The browser `WebSocket` constructor does not allow setting custom headers
on the upgrade request; the only browser-portable way to attach auth to
a WebSocket connection is the `protocols` argument:

```js
const ws = new WebSocket(url, ["bearer", token]);
```

The server extracts the second list element as the token when the first
is `bearer` (case-insensitive). Per RFC 6455 the server echoes the
selected subprotocol back on accept (`Sec-WebSocket-Protocol: bearer`) —
without that echo Chromium / Firefox close the connection client-side
with 1002 (Protocol error).

When **both** paths are present, the `Authorization` header wins; no
subprotocol is echoed in the response.

#### Operator note — handshake-header logging

The `Sec-WebSocket-Protocol` header is logged by some reverse proxies
that do not redact it the way `Authorization` is conventionally
redacted (nginx `$http_sec_websocket_protocol`, for example, is
captured by default in many configurations). TLS protects the value
in transit; this concern is **server-side logging only**.

If your ingress logs handshake headers, add a redaction rule for
`Sec-WebSocket-Protocol` containing `bearer,` — or strip the header
from access logs entirely. Authorization-header path clients are
unaffected.

### 2.3 Close codes

If the token is missing, malformed, or invalid the server closes with:

- Close code `4001` — unauthorized (missing token, invalid token, revoked API key, inactive user)
- Close code `4003` — forbidden (valid token but wrong role / ownership)
- Close code `4004` — session not found (auth succeeded, no such session id)

---

## 3. Message Envelope

All messages in both directions use this JSON envelope:

### Server → Client

```json
{
  "type": "<message_type>",
  "session_id": "<uuid>",
  "payload": { ... },
  "seq": 42,
  "ts": "2026-03-04T12:34:56.789Z"
}
```

| Field      | Type   | Description |
|------------|--------|-------------|
| type       | string | Message type discriminator (see Section 4) |
| session_id | string | UUID v4 of the session this message belongs to |
| payload    | object | Type-specific payload (see Section 4) |
| seq        | int    | Monotonically increasing per-connection sequence number |
| ts         | string | ISO 8601 UTC server timestamp |

### Client → Server

```json
{
  "type": "<message_type>",
  "payload": { ... }
}
```

---

## 4. Message Types

### 4.1 Server → Client Messages

#### `token` — Incremental LLM Output Token

Emitted once per output token during agent generation.
The frontend appends `text` to the response buffer.

```json
{
  "type": "token",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "text": " Paris",
    "final": true
  },
  "seq": 42,
  "ts": "2026-03-04T12:34:56.789Z"
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| text          | string | Incremental token text |
| final         | bool   | `true` when this token is part of the **final response** (after all tool calls complete). `false` during preamble text before tool calls. Use this to distinguish intermediate reasoning from the actual answer. Only meaningful when `tool_call_count > 0`; `false` until the first tool call is seen. |

**Streaming of the post-tool final answer (#2251 / #2264).** By default, the
post-tool **final** answer is *not* streamed token-by-token — it is delivered as a
single `token` frame (`final: true`) at turn end. This avoids a double-render when
the server's verification-recovery loop discards a fully-generated answer and
regenerates it. A client that handles the [`discard_pending`](#discard_pending--drop-the-in-progress-answer)
frame (below) can opt in to **live** final-answer streaming by setting
`stream_final_answer: true` on the send-message request
(`POST /api/v1/sessions/{id}/messages`); the server then streams `final: true`
tokens incrementally and emits `discard_pending` when it supersedes a partial.

---

#### `discard_pending` — Drop the In-Progress Answer

Emitted only when the client opted in via `stream_final_answer: true` (#2264).
Signals that the final-answer tokens streamed so far this turn were **discarded**
by server-side verification-recovery and a fresh generation will stream next. The
client MUST **clear its in-progress assistant buffer** on this frame; at most one
is sent before the surviving generation streams. The turn-ending `done` frame text
remains authoritative — reconcile the rendered text against it.

```json
{
  "type": "discard_pending",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {},
  "seq": 51,
  "ts": "2026-03-04T12:34:56.789Z"
}
```

(No payload fields.)

---

#### `tool_start` — Tool Execution Began

Emitted when the agent invokes a tool.

```json
{
  "type": "tool_start",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "tool_name": "web_search",
    "tool_call_id": "call_abc123",
    "input": {
      "query": "climate policy 2025"
    }
  },
  "seq": 43,
  "ts": "2026-03-04T12:34:57.001Z"
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| tool_name     | string | Tool name |
| tool_call_id  | string | Unique invocation ID (links to tool_end) |
| input         | object | Arguments passed to the tool |

---

#### `tool_end` — Tool Execution Completed

Emitted when a tool returns (success or error).

```json
{
  "type": "tool_end",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "tool_name": "web_search",
    "tool_call_id": "call_abc123",
    "duration_ms": 340,
    "error": null
  },
  "seq": 44,
  "ts": "2026-03-04T12:34:57.341Z"
}
```

| Payload field | Type        | Description |
|---------------|-------------|-------------|
| tool_name     | string      | Tool name |
| tool_call_id  | string      | Unique invocation ID (matches tool_start) |
| duration_ms   | int         | Execution time in milliseconds |
| error         | string/null | Error description on failure; null on success |

---

#### `tool_confirm_request` — Tool Awaiting User Confirmation

Emitted when a safety-wrapped tool requires human approval before execution.
**The agent is blocked until the client sends a `tool_confirm` response.**

The frontend must display a confirmation dialog immediately.

```json
{
  "type": "tool_confirm_request",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "confirmation_id": "conf_3f2504e0",
    "tool": "write_file",
    "parameters": {
      "path": "/home/user/report.md",
      "content": "# Climate Report\n..."
    },
    "message": "Write 2 KB to /home/user/report.md"
  },
  "seq": 45,
  "ts": "2026-03-04T12:34:58.001Z"
}
```

| Payload field   | Type   | Description |
|-----------------|--------|-------------|
| confirmation_id | string | Opaque ID to echo in the tool_confirm response |
| tool            | string | Tool requiring confirmation |
| parameters      | object | Tool call parameters (large values sorted last) |
| message         | string | Human-readable description of the action |

---

#### `agent_state` — Agent State Machine Transition

Emitted when the agent transitions between execution phases.

```json
{
  "type": "agent_state",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "state": "thinking"
  },
  "seq": 46,
  "ts": "2026-03-04T12:34:58.050Z"
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| state         | string | One of: idle, thinking, analyzing, researching, deep_thinking, writing, delegating, done, error |

State transitions by mode:

**Normal mode:**
```
idle → thinking → done
```

**Think mode** (forced deep reasoning):
```
idle → thinking → analyzing → researching (if web tools used) → deep_thinking → done
```

**Delegate mode** (forced task delegation):
```
idle → thinking → delegating → done
```

---

#### `memory_update` — Memory Compaction Occurred

Emitted when the background memory subsystem runs a summarization or compression pass.

```json
{
  "type": "memory_update",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "mode": "conversation",
    "tokens_used": 1200,
    "summarized": true
  },
  "seq": 47,
  "ts": "2026-03-04T12:34:58.200Z"
}
```

| Payload field | Type    | Description |
|---------------|---------|-------------|
| mode          | string  | Active memory mode |
| tokens_used   | int     | Estimated context token count after update |
| summarized    | boolean | True when a LLM summarization pass ran |

---

#### `error` — Agent-Level Error

Emitted when the agent encounters an error during the turn (not a WebSocket protocol error).
The connection stays open; the frontend should display the error in the chat UI.

```json
{
  "type": "error",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "code": "TOOL_EXPANSION_FAILED",
    "message": "web_search could not be loaded: API key not configured."
  },
  "seq": 48,
  "ts": "2026-03-04T12:34:58.300Z"
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| code          | string | Machine-readable error code |
| message       | string | Human-readable description safe to display |

---

#### `done` — Agent Turn Complete

Emitted when the agent turn finishes (successfully or after an error recovery).
Always the last message for a turn.

```json
{
  "type": "done",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {
    "message_id": "7a3c1b2e-5d4f-11ee-be56-0242ac120002",
    "total_tokens": 1800,
    "input_tokens": 1420,
    "output_tokens": 380,
    "duration_ms": 4200,
    "tool_calls": 3,
    "text": "The capital of France is Paris."
  },
  "seq": 49,
  "ts": "2026-03-04T12:34:59.200Z"
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| message_id    | string | UUID of the AI message created |
| total_tokens  | int    | Total tokens for this turn |
| input_tokens  | int    | Input tokens |
| output_tokens | int    | Output tokens |
| duration_ms   | int    | Wall-clock turn duration in milliseconds |
| tool_calls    | int    | Number of tool invocations |
| text          | string | Full assembled agent response text for this turn |

---

#### `pong` — Keepalive Response

Response to a client `ping` message.

```json
{
  "type": "pong",
  "session_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
  "payload": {},
  "seq": 50,
  "ts": "2026-03-04T12:35:00.001Z"
}
```

---

#### `log_line` — Live Log Record (log stream only)

Emitted on the `/ws/v1/logs` endpoint only. **Note:** log stream messages are plain JSON dicts — they do NOT use the common `ServerMessage` envelope (no `session_id`, `seq`, or `ts` fields).

```json
{
  "type": "log_line",
  "level": "INFO",
  "logger": "cogtrix.orchestration.runner",
  "message": "Agent turn completed in 4.2s",
  "timestamp": "2026-03-04T12:34:59.200Z"
}
```

---

### 4.2 Client → Server Messages

#### `user_message` — Send a Message Over WebSocket

Alternative to the REST `POST /api/v1/sessions/{id}/messages`.
Useful for low-latency chat UIs that want to avoid an extra HTTP round-trip.

```json
{
  "type": "user_message",
  "payload": {
    "text": "What is the capital of France?",
    "mode": "normal"
  }
}
```

| Payload field | Type   | Description |
|---------------|--------|-------------|
| text          | string | User message text (1–65536 chars) |
| mode          | string | normal, think, or delegate |

---

#### `tool_confirm` — User Decision on Tool Confirmation

Must be sent in response to a `tool_confirm_request` message.

```json
{
  "type": "tool_confirm",
  "payload": {
    "confirmation_id": "conf_3f2504e0",
    "action": "allow"
  }
}
```

| Payload field   | Type   | Description |
|-----------------|--------|-------------|
| confirmation_id | string | The confirmation_id from the tool_confirm_request |
| action          | string | allow, deny, allow_all, disable, forbid_all, or cancel |

Action semantics (mirrors CLI options):

| Action     | CLI key | Description |
|------------|---------|-------------|
| allow      | y       | Allow this invocation once |
| deny       | n       | Deny this invocation; agent may retry |
| allow_all  | a       | Auto-approve this tool for the entire session |
| disable    | d       | Disable this tool for the entire session |
| forbid_all | f       | Block all further tool requests this turn |
| cancel     | c       | Cancel the current agent workflow entirely |

---

#### `ping` — Keepalive

Must be sent every 30 seconds.  Connections silent for 90 seconds are dropped.

```json
{
  "type": "ping",
  "payload": {}
}
```

---

#### `cancel` — Cancel Current Agent Turn

Signals the server to abort the in-progress agent turn.
The server transitions to `agent_state: idle` first, then sends an `error` message with code `CANCELLED`.
A `done` message is **not** sent. The connection remains open for the next turn.

```json
{
  "type": "cancel",
  "payload": {}
}
```

---

## 5. Connection Lifecycle

```
CLIENT                                    SERVER
  |                                          |
  |  ---- WS connect + JWT ----------->     |
  |                                          |  validate token & session ownership
  |  <--- agent_state (idle) ----------      |
  |                                          |
  |  ---- user_message / POST REST --->      |  [or message sent via REST]
  |                                          |
  |  <--- agent_state (thinking) ------      |
  |  <--- tool_start (web_search) -----      |
  |  <--- tool_confirm_request --------      |  [if tool needs confirmation]
  |  ---- tool_confirm (allow) ------->      |
  |  <--- tool_end (web_search) -------      |
  |  <--- agent_state (analyzing) -----      |  [think mode: classifying task]
  |  <--- agent_state (researching) ---      |  [think mode: research delegate]
  |  <--- agent_state (deep_thinking) -      |  [think mode: deep reasoning]
  |  <--- agent_state (delegating) ----      |  [delegate mode: parallel delegation]
  |  <--- agent_state (writing) -------      |
  |  <--- token ("The capital") -------      |
  |  <--- token (" is Paris") ---------      |
  |  <--- memory_update ---------------      |  [if summarization ran]
  |  <--- agent_state (done) ----------      |
  |  <--- done -------------------------      |
  |                                          |
  |  ---- ping ------------------------>     |  [every 30s]
  |  <--- pong -------------------------      |
  |                                          |
  |  [... next turn ...]                     |
```

---

## 6. Error Handling

### Sending a message while a turn is in progress

If a `user_message` arrives while an agent turn is already running, the server sends
an `error` payload with code `TURN_IN_PROGRESS`. The connection remains open and the
in-progress turn is unaffected. Wait for the `done` message before sending another message.

### Agent crashes mid-stream

If the agent raises an unrecoverable exception during a turn:
1. Server sends `type: error` with a descriptive error code and message.
2. Server transitions to `agent_state: error`, then `agent_state: idle`.
3. The connection remains open for the next turn.

### WebSocket protocol errors

| Close code | Meaning |
|------------|---------|
| 4000       | Session registry unavailable — server is still starting up |
| 4001       | Unauthorized — no token, invalid signature |
| 4003       | Forbidden — valid token, wrong role or session ownership |
| 4004       | Session not found — session does not exist or was archived |
| 1000       | Normal closure |
| 1001       | Server going away (shutdown); also used when a second connection replaces the first |
| 1011       | Internal server error |

---

## 7. Reconnection Strategy

The `seq` field enables the frontend to detect dropped messages and recover:

1. Store `last_seen_seq` in memory (reset to -1 on new page load).
2. On reconnect, send `?last_seq=<last_seen_seq>` as a query parameter.
3. The server replays buffered messages with `seq > last_seq` (buffer kept 30 s post-disconnect).
4. If `last_seq` is too old (buffer expired), the server sends the current state only.

Recommended reconnect strategy:
- Immediate reconnect on first disconnect.
- Exponential backoff: 1s → 2s → 4s → 8s → 16s → cap at 30s.
- Stop retrying after 10 consecutive failures; show the user an error.
- On successful reconnect, fetch message history via REST to fill any gap.

---

## 8. Log Stream WebSocket (`/ws/v1/logs`)

Admin-only endpoint for live log streaming.

```
ws://host/ws/v1/logs?token=<jwt>&level=INFO
```

Query parameters:
- `token` — JWT bearer token (admin required).
- `level` — minimum log level to stream: DEBUG, INFO, WARNING, ERROR (default INFO).

Streams `log_line` messages as they are emitted.
Same keepalive timing applies (drop after 90 s of silence). **Note:** the log stream uses a plain text `"ping"` string for keepalive (not the `ClientMessage` JSON envelope used by session WebSockets). The server responds with `{"type": "pong"}`.
