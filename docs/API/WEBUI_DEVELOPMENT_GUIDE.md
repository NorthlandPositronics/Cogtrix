# Cogtrix Web UI Development Guide

Audience: React developers building the Cogtrix web frontend
API version: v1
Last updated: 2026-03-27

Related documents:
- `docs/API/OVERVIEW.md` — API orientation, quick start, environment variables (start here if new to the API)
- Live OpenAPI 3.1 schema at `/api/v1/openapi.json` (Swagger UI at `/api/v1/docs`) — authoritative source for request/response shapes
- `docs/API/CLIENT_CONTRACT.md` — TypeScript types and API client patterns (start here for code)
- `docs/API/WEBSOCKET_PROTOCOL.md` — WebSocket message types, lifecycle diagrams, and reconnection strategy

---

## 1. Architecture Overview

### Backend stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI (async) |
| Database | SQLAlchemy async + aiosqlite (SQLite by default) |
| Migrations | Alembic |
| Auth | JWT bearer tokens (access + refresh) |
| Real-time | WebSocket (native FastAPI) |
| Agent orchestration | LangGraph `StateGraph` (`src/orchestration/`) |
| Memory | Hybrid sliding window + incremental summary + optional vector recall (`src/memory/`) |

### Frontend target

The frontend is a React + TypeScript single-page application built with Vite.

### Communication model

```
Browser ──── REST (CRUD, config, file upload) ────► FastAPI /api/v1
Browser ──── WebSocket (streaming, events)   ────► FastAPI /ws/v1
```

REST handles everything that is not time-critical: creating sessions, loading message history, managing tools, configuration, and file uploads. WebSocket handles everything real-time: token-by-token agent output, tool activity events, tool confirmation dialogs, and agent state transitions.

### Authentication

The API uses JWT bearer tokens. Two tokens are issued on login:

- **Access token** — short-lived (default 1 hour), sent as `Authorization: Bearer <token>` on every REST request
- **Refresh token** — long-lived (default 30 days), used only to obtain a new access token when the old one expires

For WebSocket connections, the browser `WebSocket` API does not allow custom headers.
Route through a reverse proxy that injects the ``Authorization`` header (e.g., from a
query param or HttpOnly cookie), or use a dedicated auth endpoint that sets an HttpOnly
session cookie accepted by the backend. The session WebSocket no longer accepts
``?token=`` (removed in #1128 to prevent token exposure in access logs).

### API versioning

All REST endpoints are under `/api/v1/`. All WebSocket endpoints are under `/ws/v1/`. The version prefix is included in every path shown in this document.

---

## 2. Environment Setup

### Frontend environment variables

Create a `.env.local` file in the Vite project root:

```
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

`CLIENT_CONTRACT.md` (Section 1) shows how these are consumed in `src/lib/api/config.ts`.

### Backend setup

```bash
# Generate a JWT secret (required — no default)
export COGTRIX_JWT_SECRET=$(openssl rand -hex 32)

# Apply database migrations
uv run python -m alembic upgrade head

# Start the API server with hot reload
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The first user to call `POST /api/v1/auth/register` is automatically granted the `admin` role. This uses an atomic `INSERT…SELECT` so concurrent registration requests cannot both receive admin.

---

## 3. Page Map and Component Hierarchy

This section maps every significant API surface to a frontend page and its primary components.

### 3.1 Login / Register page (`/login`, `/register`)

**Purpose:** Authenticate the user and store the token pair.

**API calls:**

| Action | Endpoint |
|--------|----------|
| Register new account | `POST /api/v1/auth/register` |
| Login | `POST /api/v1/auth/login` |
| Refresh access token (interceptor) | `POST /api/v1/auth/refresh` |

**Key behaviours:**
- On successful login or register, store the token pair (see Section 4.1).
- Redirect to `/sessions` on success.
- Display field-level validation errors from the `details` field of a 422 response.
- The first registered user becomes admin automatically — no separate admin setup step is needed.

---

### 3.2 Dashboard / Sessions list (`/sessions`)

**Purpose:** Show the user's sessions and let them create or archive sessions.

**API calls:**

| Action | Endpoint |
|--------|----------|
| List sessions (paginated) | `GET /api/v1/sessions` |
| Create new session | `POST /api/v1/sessions` |
| Archive session | `DELETE /api/v1/sessions/{id}` |
| Restore archived session | `POST /api/v1/sessions/{id}/restore` |

**Key behaviours:**
- Use cursor-based pagination (`next_cursor` / `cursor` query parameter). See Section 4.3 for the infinite-scroll pattern.
- Each session card shows `name`, `state` (idle / thinking / ...), `config.model`, and `updated_at`.
- The `state` field of type `AgentState` reflects the current execution status. Use it to show a live indicator on sessions that are actively running.
- Session creation accepts an optional `name` and `config` object (`SessionCreateRequest`). The config can override model, memory mode, and feature flags per session.

---

### 3.3 Session / Chat page (`/sessions/{id}`)

This is the most complex page. It combines REST calls for history and session management with a persistent WebSocket connection for real-time streaming.

#### Message list

| Action | Endpoint |
|--------|----------|
| Load message history | `GET /api/v1/sessions/{id}/messages` |
| Clear history | `DELETE /api/v1/sessions/{id}/messages` |

Load history in reverse chronological order using cursor pagination (newest first). As the user scrolls up, fetch older pages. Each `MessageOut` includes `role`, `content`, `tool_calls`, `token_counts`, and `created_at`.

#### Sending a message

Two equivalent paths exist:

1. **REST** — `POST /api/v1/sessions/{id}/messages` with a `SendMessageRequest` body. The agent turn is queued; streaming begins over the WebSocket immediately after the HTTP response.
2. **WebSocket** — send a `user_message` client message directly over the open WebSocket. Saves one HTTP round-trip.

The `mode` field controls the reasoning pipeline:
- `normal` — standard agent run
- `think` — forces the deep-think pipeline (extended reasoning)
- `delegate` — forces task delegation to a sub-agent

#### Real-time streaming

Connect to `ws://host/ws/v1/sessions/{id}` with the token in the ``Authorization`` header (injected by a reverse proxy or via cookie — see Section 1.2 Authentication). Keep the connection alive. See Section 4.2 for the full streaming flow and Section 4.4 for reconnection.

Events to handle:

| Event | UI action |
|-------|-----------|
| `agent_state` | Update the agent state indicator in the header |
| `token` | Append text to the streaming response bubble |
| `tool_start` | Add an expanding tool activity row |
| `tool_end` | Update the tool activity row with duration / error |
| `tool_confirm_request` | Open the confirmation modal (blocks agent) |
| `memory_update` | Update the memory panel token counter |
| `error` | Display an error inline in the chat |
| `done` | Finalise the response bubble; update token stats |

#### Tool confirmation dialog

When a `tool_confirm_request` arrives, the agent is blocked. Open a modal immediately showing:
- Tool name and human-readable description (`message` field)
- Parameters (large values like `content` are sorted last by the server)
- Six action buttons: Allow once / Deny / Allow all / Disable / Forbid all / Cancel

Send the user's choice back as a `tool_confirm` client message with the matching `confirmation_id`. The agent resumes once the response is received.

Actions mirror the CLI confirmation options:

| Button label | Action value | Effect |
|--------------|-------------|--------|
| Allow once | `allow` | Permits this invocation only |
| Deny | `deny` | Blocks this invocation; agent may retry |
| Allow all | `allow_all` | Auto-approves this tool for the session |
| Disable | `disable` | Blocks this tool for the entire session |
| Forbid all | `forbid_all` | Blocks all further tool requests this turn |
| Cancel | `cancel` | Aborts the agent workflow entirely |

The server resets the cancel state between turns, so a `cancel` action only affects the current turn — it does not persist to future turns.

#### Agent state indicator

Display a persistent status badge in the session header driven by `agent_state` messages:

| State | Suggested label | Description |
|-------|----------------|-------------|
| `idle` | Ready | No active turn |
| `thinking` | Thinking... | Agent is processing the initial prompt |
| `analyzing` | Analyzing... | Think mode: classifying task type |
| `researching` | Researching... | Think mode: research delegate fetching URLs |
| `deep_thinking` | Deep thinking... | Think mode: extended multi-branch reasoning |
| `writing` | Writing... | Agent is producing output |
| `delegating` | Delegating... | Delegate mode: parallel sub-agent execution |
| `done` | Done | Turn completed |
| `error` | Error | Turn failed |

Transition order by mode:

- **Normal:** `idle → thinking → writing → done`
- **Think:** `idle → thinking → analyzing → researching → deep_thinking → done`
- **Delegate:** `idle → thinking → delegating → done`

#### Memory panel

| Action | Endpoint |
|--------|----------|
| Get memory state | `GET /api/v1/sessions/{id}/memory` |
| Switch memory mode | `PATCH /api/v1/sessions/{id}/memory` |
| Clear memory | `DELETE /api/v1/sessions/{id}/memory` |

`MemoryStateOut` includes: active mode, summary text, window message count, summarized message count, token usage, and context window size. Display as a collapsible sidebar panel. Refresh after each `memory_update` WebSocket event.

Memory modes: `conversation` (default), `code`, `reasoning`.

#### Tools sidebar

| Action | Endpoint |
|--------|----------|
| Get session tool status | `GET /api/v1/sessions/{id}/tools` |
| Manage tools (load / enable / disable / approve) | `PATCH /api/v1/sessions/{id}/tools` |
| Browse all tools | `GET /api/v1/tools` |
| Get tool details | `GET /api/v1/tools/{name}` |

Each `ToolSummary` has a `status` field: `active`, `pinned`, `on_demand`, `disabled`, `auto_approved`. A `ToolActionRequest` lets you send multiple operations in a single call (load, unload, enable, disable, auto_approve, revoke_approval).

---

### 3.4 Settings page (`/settings`)

**Purpose:** View and modify the global Cogtrix configuration, manage providers and models, and manage MCP servers.

#### Configuration

| Action | Endpoint | Auth |
|--------|----------|------|
| Get current config | `GET /api/v1/config` | bearer |
| Update config flags | `PATCH /api/v1/config` | admin |
| Reload config from disk | `POST /api/v1/config/reload` | admin |
| List providers | `GET /api/v1/config/providers` | bearer |
| Get provider details | `GET /api/v1/config/providers/{name}` | bearer |
| Add provider | `POST /api/v1/config/providers` | admin |
| Update provider | `PATCH /api/v1/config/providers/{name}` | admin |
| Delete provider | `DELETE /api/v1/config/providers/{name}` | admin |
| Provider health check | `POST /api/v1/config/providers/{name}/health` | bearer |
| List models | `GET /api/v1/config/models` | bearer |
| Switch active model | `POST /api/v1/config/model` | admin |

`ConfigOut` includes `active_model` (the current model alias), memory mode defaults, feature flag booleans (`prompt_optimizer`, `parallel_tool_execution`, `context_compression`, `debug`, `verbose`), and embedded `providers` and `models` lists. The `raw_yaml` field is populated only for admin users.

`PATCH /api/v1/config` accepts a `ConfigPatchRequest` with boolean flag overrides. Non-admin users receive 403 on this endpoint.

#### MCP servers

| Action | Endpoint | Auth |
|--------|----------|------|
| List MCP servers | `GET /api/v1/mcp/servers` | bearer |
| Add server | `POST /api/v1/mcp/servers` | admin |
| Get server details | `GET /api/v1/mcp/servers/{name}` | bearer |
| Remove server | `DELETE /api/v1/mcp/servers/{name}` | admin |
| Restart server | `POST /api/v1/mcp/servers/{name}/restart` | admin |

MCP tools discovered from connected servers appear in the on-demand tool pool. They are loaded and used the same way as built-in tools.

#### Setup wizard

The setup wizard is a multi-step flow for writing or editing the Cogtrix YAML config file.

| Action | Endpoint | Auth |
|--------|----------|------|
| Start wizard | `POST /api/v1/config/wizard` | admin |
| Advance a step | `POST /api/v1/config/wizard/{id}/step` | admin |
| Cancel | `DELETE /api/v1/config/wizard/{id}` | admin |

Each `WizardStepOut` response includes `step`, `total_steps`, `step_name`, a `question` for the user, an optional `yaml_preview`, and a `complete` flag. Display the question and collect the user's answer in a form. Submit with `WizardStepRequest.answer`. When `complete` is `true`, the wizard has written the config file.

---

### 3.5 Admin panel (`/admin`)

Restrict this page to users with `role: 'admin'`.

| Action | Endpoint |
|--------|----------|
| System info | `GET /api/v1/system/info` |
| Toggle debug logging | `POST /api/v1/system/debug` |
| Live log stream | `WS /ws/v1/logs?token=<jwt>&level=INFO` |
| Guardrail status | `GET /api/v1/assistant/guardrails` |
| Remove from blacklist | `DELETE /api/v1/assistant/guardrails/blacklist/{id}` |
| List users | `GET /api/v1/users` |
| Create user | `POST /api/v1/users` |
| Update user role | `PATCH /api/v1/users/{id}` |
| Delete user | `DELETE /api/v1/users/{id}` |

The live log stream (`/ws/v1/logs`) emits `log_line` messages. **Note:** log stream messages are
plain JSON objects -- they are NOT wrapped in the standard `ServerMessage` envelope (no
`session_id`, `seq`, or `ts` fields). Each message is `{ type: "log_line", level, logger, message,
timestamp }`. Connect with a minimum level filter (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Keepalive
rules for the log stream: ping every 30 seconds, connection dropped after 90 seconds of silence
(hardcoded for log stream; separate from the 300-second session WebSocket timeout). The log stream
uses a plain text `"ping"` string for keepalive (not the `ClientMessage` JSON envelope used by
session WebSockets).

**User management** — all four `/api/v1/users` endpoints require admin role. The create endpoint validates `EmailStr`, username pattern `^[a-zA-Z0-9_-]+$` (3–64 chars), password min 8 chars, and role `user` or `admin`. The first registered user is automatically elected as admin via an atomic DB operation.

---

### 3.6 Assistant dashboard (`/assistant`)

Relevant only when the Cogtrix instance is running in `--assistant` mode (headless WhatsApp/Telegram daemon).

| Action | Endpoint | Auth |
|--------|----------|------|
| Service status | `GET /api/v1/assistant/status` | bearer |
| Start service | `POST /api/v1/assistant/start` | admin |
| Stop service | `POST /api/v1/assistant/stop` | admin |
| List active chats | `GET /api/v1/assistant/chats` | bearer |
| Per-chat message history | `GET /api/v1/assistant/chats/{key}/messages` | bearer |
| List scheduled messages | `GET /api/v1/assistant/scheduled` | bearer |
| Edit scheduled message | `PATCH /api/v1/assistant/scheduled/{id}` | admin |
| Cancel scheduled message | `DELETE /api/v1/assistant/scheduled/{id}` | admin |
| List deferred records | `GET /api/v1/assistant/deferred` | bearer |
| Cancel deferred record | `DELETE /api/v1/assistant/deferred/{key}` | bearer |
| List phonebook contacts | `GET /api/v1/assistant/contacts` | bearer |
| List knowledge facts | `GET /api/v1/assistant/knowledge` | bearer |
| Semantic search in knowledge | `POST /api/v1/assistant/knowledge/search` | bearer |
| Delete knowledge fact | `DELETE /api/v1/assistant/knowledge/{id}` | admin |
| Send outbound message | `POST /api/v1/assistant/outbound` | admin |
| List campaigns | `GET /api/v1/assistant/campaigns` | bearer |
| Create campaign | `POST /api/v1/assistant/campaigns` | admin |
| Get campaign | `GET /api/v1/assistant/campaigns/{id}` | bearer |
| Update campaign | `PATCH /api/v1/assistant/campaigns/{id}` | admin |
| Delete campaign | `DELETE /api/v1/assistant/campaigns/{id}` | admin |
| Launch campaign | `POST /api/v1/assistant/campaigns/{id}/launch` | admin |

The assistant maintains per-chat memory isolation. Each `(channel, chat_id)` pair has its own conversation history that is never shared with other chats.

`POST /api/v1/assistant/stop` is an async operation — the server wraps blocking shutdown calls with `asyncio.to_thread` to avoid blocking the event loop. The UI should show a "stopping..." indicator and poll `GET /api/v1/assistant/status` until `running` becomes `false`.

---

### 3.7 RAG / Documents page (`/documents`)

| Action | Endpoint | Auth |
|--------|----------|------|
| Upload and ingest document | `POST /api/v1/rag/documents` | admin |
| List documents | `GET /api/v1/rag/documents` | bearer |
| Get document details | `GET /api/v1/rag/documents/{id}` | bearer |
| Delete document | `DELETE /api/v1/rag/documents/{id}` | admin |
| Semantic search | `POST /api/v1/rag/search` | bearer |

Document upload uses `multipart/form-data`. The server enforces file size limits; validate file types client-side before sending (PDF, plain text, Markdown, CSV). Ingestion is asynchronous — the 201 response is returned before processing completes. Poll the document list or use a status indicator while ingestion runs in the background.

Document IDs are UUIDs validated server-side. The `GET /api/v1/rag/documents` endpoint uses compound `(created_at, id)` keyset cursors for correct pagination ordering.

---

### 3.8 Workflows page (`/workflows`)

Relevant only when assistant mode is active. Workflows bundle a system prompt, per-workflow knowledge base, and tool policy into reusable configurations that can be bound to specific chats.

| Action | Endpoint | Auth |
|--------|----------|------|
| List workflows | `GET /api/v1/assistant/workflows` | bearer |
| Get workflow | `GET /api/v1/assistant/workflows/{id}` | bearer |
| Create workflow | `POST /api/v1/assistant/workflows` | admin |
| Update workflow | `PUT /api/v1/assistant/workflows/{id}` | admin |
| Delete workflow | `DELETE /api/v1/assistant/workflows/{id}` | admin |
| List workflow documents | `GET /api/v1/assistant/workflows/{id}/documents` | bearer |
| Upload workflow document | `POST /api/v1/assistant/workflows/{id}/documents` | admin |
| Delete workflow document | `DELETE /api/v1/assistant/workflows/{id}/documents/{doc_id}` | admin |
| List chat bindings | `GET /api/v1/assistant/workflows/bindings` | bearer |
| Bind chat to workflow | `PUT /api/v1/assistant/workflows/bindings/{session_key}` | admin |
| Unbind chat | `DELETE /api/v1/assistant/workflows/bindings/{session_key}` | admin |

**Component hierarchy:**

```
WorkflowsPage
├── WorkflowList          — table of all workflows with name, description, binding count
│   └── WorkflowRow       — click to open detail view; admin badge for CRUD actions
├── WorkflowDetail        — selected workflow: edit form + document list + binding list
│   ├── WorkflowForm      — name, description, system prompt (textarea), tool policy toggles
│   ├── DocumentList      — per-workflow uploaded documents with upload/delete
│   └── BindingList       — chats bound to this workflow; bind/unbind controls
└── WorkflowCreateDialog  — modal for new workflow (ID must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$)
```

**Key behaviors:**

- Workflow IDs are immutable after creation (used as directory names)
- Document uploads use `multipart/form-data`; allowed extensions: `.pdf`, `.txt`, `.md`, `.markdown`, `.csv`; max 50 MB
- The `auto_detect` section configures keyword/regex-based automatic workflow assignment for new chats
- Resolution order for chat→workflow: explicit binding → contact_prompts fallback → auto-detect → no match (global defaults)

---

## 4. Key Integration Patterns

### 4.1 Token Refresh Flow

Token lifetimes (from the OpenAPI spec):
- Access token: **1 hour** by default
- Refresh token: **30 days** by default

Do not rely on token lifetime values for proactive refresh. Instead, intercept 401 responses:

1. Any request returns 401 with `error.code === 'TOKEN_EXPIRED'`.
2. Pause the failing request.
3. Call `POST /api/v1/auth/refresh` with the stored refresh token.
4. On success: store the new token pair, retry the original request with the new access token.
5. On failure (refresh token expired or invalid): clear stored tokens and redirect to `/login`.

Prevent multiple concurrent requests from each triggering a separate refresh. Use a single in-flight promise and queue other failing requests until the refresh resolves:

```typescript
// src/lib/api/client.ts

let refreshPromise: Promise<void> | null = null;

async function refreshAccessToken(): Promise<void> {
  if (refreshPromise) return refreshPromise;
  refreshPromise = doRefresh().finally(() => { refreshPromise = null; });
  return refreshPromise;
}
```

The full request function with retry logic is shown in `docs/API/CLIENT_CONTRACT.md`, Section 2.3.

---

### 4.2 Streaming Agent Output

This is the primary real-time interaction pattern. Follow these steps in order:

**Step 1 — Establish the WebSocket connection**

Connect when the user opens a session. Keep the connection alive across multiple turns. Use `last_seq` on reconnect for message replay:

```typescript
// The reverse proxy injects the Authorization header from a cookie/query param.
// Do NOT use ?token= — the session WebSocket no longer supports it (see #1128).
const url = `${WS_V1}/sessions/${sessionId}?last_seq=${lastSeq}`;
const ws = new WebSocket(url);
```

**Step 2 — Send the message**

Either via REST (`POST /api/v1/sessions/{id}/messages`) or directly over the WebSocket (`user_message` client message). The `mode` field selects `normal`, `think`, or `delegate`.

**Step 3 — Accumulate tokens**

Append each `token` payload's `text` field to the response buffer. Render the buffer as Markdown. Update the UI on each token to show streaming output.

**Step 4 — Show tool activity**

On `tool_start`: add a collapsible activity row showing the tool name and its input arguments.
On `tool_end`: update the row with `duration_ms`. If `error` is non-null, show it in red.

**Step 5 — Handle tool confirmation**

On `tool_confirm_request`: open a blocking modal with the tool name, `message`, and `parameters`. Do not dismiss the modal until the user selects an action. Send `tool_confirm` immediately on selection. The agent remains blocked until you respond.

**Step 6 — Mark the turn complete**

On `done`: finalise the response bubble. Display token stats from the payload (`input_tokens`, `output_tokens`, `duration_ms`, `tool_calls`). Reset the streaming buffer. Update the session's `state` to `idle`.

The `SessionSocket` class in `docs/API/CLIENT_CONTRACT.md`, Section 4, provides a complete working implementation.

---

### 4.3 Cursor-Based Pagination

All list endpoints return a `CursorPage<T>` envelope:

```typescript
interface CursorPage<T> {
  items: T[];
  next_cursor: string | null;
  has_more: boolean;
  total: number | null;
}
```

To load the next page, pass the `next_cursor` value as the `cursor` query parameter. When `has_more` is `false` or `next_cursor` is `null`, you have reached the end.

For message history (infinite scroll upward), load older pages as the user scrolls toward the top of the list. Prepend items to the existing list.

The `useInfiniteList` hook pattern is shown in `docs/API/CLIENT_CONTRACT.md`, Section 5.

TanStack Query (`useInfiniteQuery`) integrates directly with this pattern:

```typescript
const query = useInfiniteQuery({
  queryKey: ['messages', sessionId],
  queryFn: ({ pageParam }) =>
    api.messages.list(sessionId, { cursor: pageParam ?? undefined, limit: 50 }),
  getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  initialPageParam: null,
});
```

---

### 4.4 WebSocket Reconnection

The `seq` field on every server message enables gap-free reconnection:

1. Store `lastSeq` in component state, initialised to `-1`.
2. On each message, update `lastSeq = msg.seq`.
3. On disconnect: attempt immediate reconnect, then use exponential backoff: 1 s → 2 s → 4 s → 8 s → 16 s, capped at 30 s.
4. Include `?last_seq=${lastSeq}` on the reconnect URL. The server replays buffered messages with `seq > lastSeq` for up to 30 seconds after disconnect.
5. After a successful reconnect, also fetch missing messages via `GET /api/v1/sessions/{id}/messages` to fill any gap beyond the server's replay buffer.
6. Stop retrying after 10 consecutive failures. Show the user an explicit error with a manual reconnect button.

Full reconnection strategy is documented in `docs/API/WEBSOCKET_PROTOCOL.md`, Section 7.

---

### 4.5 Error Handling

All REST errors use a consistent envelope:

```typescript
interface APIError {
  code: string;      // machine-readable
  message: string;   // human-readable, safe to display
  details?: Record<string, unknown> | null;
}
```

**HTTP status code handling:**

| Status | Error code | Recommended UI action |
|--------|-----------|----------------------|
| 400 | `INVALID_CURSOR` | Restart pagination from the first page |
| 400 | `VALIDATION_ERROR` | Show field-level errors from `details` |
| 401 | `UNAUTHORIZED` | Redirect to `/login` |
| 401 | `TOKEN_EXPIRED` | Silently refresh token, retry request |
| 403 | `FORBIDDEN` | Show "Access denied" — do not redirect |
| 404 | `*_NOT_FOUND` | Show inline "Not found" message |
| 409 | `*_ALREADY_*` | Show "Already exists" toast |
| 422 | `VALIDATION_ERROR` | Show field-level validation errors |
| 422 | `PROVIDER_UNREACHABLE` | Show an "LLM provider is offline" banner |
| 500 | `INTERNAL_ERROR` | Show generic toast; log `meta.request_id` for support |

**WebSocket close code handling:**

| Close code | Meaning | UI action |
|------------|---------|-----------|
| 4000 | Session registry unavailable | Show "Server starting up" toast; retry after delay |
| 4001 | Unauthorized | Redirect to `/login` |
| 4003 | Forbidden | Show "Access denied", redirect to sessions list |
| 4004 | Session not found | Show "Session not found", redirect to sessions list |
| 1001 | Server shutting down / replaced by new connection | Show "Server is restarting", poll `/api/v1/health/ready` |
| 1011 | Internal server error | Show error, attempt reconnect |

The error display pattern from `docs/API/CLIENT_CONTRACT.md`, Section 6, shows a concrete `handleApiError()` implementation.

---

## 5. State Management Recommendations

### Auth state

Use a React context (not a global module) to hold the token pair and the current user profile. Expose:
- `login(credentials)` — calls `POST /auth/login`, stores tokens, redirects to `/sessions`
- `logout()` — calls `POST /auth/logout`, clears tokens, redirects to `/login`
- `user: UserOut | null` — populated by `GET /auth/me` after login
- `isAdmin: boolean` — derived from `user.role === 'admin'`

Store the access token in memory. Use an HttpOnly cookie for the refresh token where possible to limit XSS exposure. See `docs/API/CLIENT_CONTRACT.md`, Section 2.2, for the in-memory token store pattern.

### Session list

Use TanStack Query with `stale-while-revalidate`. Sessions change infrequently; a 30-second stale time is appropriate. Invalidate the query on `POST /sessions` (create) and `DELETE /sessions/{id}` (archive).

### Active session messages

Keep message state local to the session page component. On mount, fetch the first page of history via REST. Append new messages as they arrive via the WebSocket `done` event (the finalised message with its `message_id`). On unmount, disconnect the WebSocket.

### WebSocket connection

Encapsulate the WebSocket lifecycle in a custom hook (`useSessionSocket`). The hook should:
- Connect on mount, disconnect on unmount
- Manage reconnection logic with exponential backoff
- Expose `sendMessage()` and `confirmTool()` methods
- Expose a `state` value (`connecting`, `open`, `reconnecting`, `closed`)

The `SessionSocket` class in `docs/API/CLIENT_CONTRACT.md`, Section 4, can be wrapped directly in this hook.

### Tool confirmation

Keep a `pendingConfirmation: ToolConfirmRequestPayload | null` state in the session page. Set it on `tool_confirm_request`; clear it after the user responds. The modal is rendered conditionally from this state.

### Config and provider data

Fetch config via `GET /api/v1/config` on the settings page mount. Use TanStack Query with a long stale time (5 minutes) since config rarely changes at runtime. Invalidate on successful `PATCH /config` or `POST /config/reload`.

---

## 6. API Quick Reference

### Health (no auth)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness — is the process alive? |
| GET | `/api/v1/health/ready` | Readiness — is the DB connected? |

### Authentication

| Method | Path | Auth | Page |
|--------|------|------|------|
| POST | `/api/v1/auth/register` | none | Register |
| POST | `/api/v1/auth/login` | none | Login |
| POST | `/api/v1/auth/refresh` | none | Interceptor (all pages) |
| POST | `/api/v1/auth/logout` | bearer | Any |
| GET | `/api/v1/auth/me` | bearer | Header / profile |
| GET | `/api/v1/auth/api-keys` | bearer | Settings |
| POST | `/api/v1/auth/api-keys` | bearer | Settings |
| DELETE | `/api/v1/auth/api-keys/{id}` | bearer | Settings |

### Sessions

| Method | Path | Auth | Page |
|--------|------|------|------|
| POST | `/api/v1/sessions` | bearer | Dashboard |
| GET | `/api/v1/sessions` | bearer | Dashboard |
| GET | `/api/v1/sessions/{id}` | bearer | Session page |
| PATCH | `/api/v1/sessions/{id}` | bearer | Session page |
| DELETE | `/api/v1/sessions/{id}` | bearer | Dashboard |
| POST | `/api/v1/sessions/{id}/restore` | bearer | Dashboard (archived sessions) |

### Messages

| Method | Path | Auth | Page |
|--------|------|------|------|
| POST | `/api/v1/sessions/{id}/messages` | bearer | Session page |
| GET | `/api/v1/sessions/{id}/messages` | bearer | Session page |
| DELETE | `/api/v1/sessions/{id}/messages` | bearer | Session page |
| WS | `/ws/v1/sessions/{id}` | token= param | Session page |

### Memory

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/sessions/{id}/memory` | bearer | Session page (panel) |
| DELETE | `/api/v1/sessions/{id}/memory` | bearer | Session page (panel) |
| PATCH | `/api/v1/sessions/{id}/memory` | bearer | Session page (panel) |

### Tools

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/tools` | bearer | Tools sidebar |
| GET | `/api/v1/tools/{name}` | bearer | Tools sidebar |
| GET | `/api/v1/sessions/{id}/tools` | bearer | Tools sidebar |
| PATCH | `/api/v1/sessions/{id}/tools` | bearer | Tools sidebar |

### Configuration

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/config` | bearer | Settings |
| PATCH | `/api/v1/config` | admin | Settings |
| POST | `/api/v1/config/reload` | admin | Settings |
| GET | `/api/v1/config/providers` | bearer | Settings |
| GET | `/api/v1/config/providers/{name}` | bearer | Settings |
| POST | `/api/v1/config/providers` | admin | Settings | Add a new provider at runtime |
| PATCH | `/api/v1/config/providers/{name}` | admin | Settings | Update provider connection details |
| DELETE | `/api/v1/config/providers/{name}` | admin | Settings | Remove a provider |
| POST | `/api/v1/config/providers/{name}/health` | bearer | Settings |
| GET | `/api/v1/config/models` | bearer | Settings |
| POST | `/api/v1/config/model` | admin | Settings |
| POST | `/api/v1/config/wizard` | admin | Settings (wizard) |
| POST | `/api/v1/config/wizard/{id}/step` | admin | Settings (wizard) |
| DELETE | `/api/v1/config/wizard/{id}` | admin | Settings (wizard) |

### MCP Servers

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/mcp/servers` | bearer | Settings |
| POST | `/api/v1/mcp/servers` | admin | Settings |
| GET | `/api/v1/mcp/servers/{name}` | bearer | Settings |
| DELETE | `/api/v1/mcp/servers/{name}` | admin | Settings |
| POST | `/api/v1/mcp/servers/{name}/restart` | admin | Settings |

### Assistant

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/assistant/status` | bearer | Assistant dashboard |
| POST | `/api/v1/assistant/start` | admin | Assistant dashboard |
| POST | `/api/v1/assistant/stop` | admin | Assistant dashboard |
| POST | `/api/v1/assistant/outbound` | admin | Assistant dashboard |
| GET | `/api/v1/assistant/chats` | bearer | Assistant dashboard |
| GET | `/api/v1/assistant/chats/{key}/messages` | bearer | Assistant dashboard |
| GET | `/api/v1/assistant/scheduled` | bearer | Assistant dashboard |
| PATCH | `/api/v1/assistant/scheduled/{id}` | admin | Assistant dashboard |
| DELETE | `/api/v1/assistant/scheduled/{id}` | admin | Assistant dashboard |
| GET | `/api/v1/assistant/deferred` | bearer | Assistant dashboard |
| DELETE | `/api/v1/assistant/deferred/{key}` | bearer | Assistant dashboard |
| GET | `/api/v1/assistant/contacts` | bearer | Assistant dashboard |
| GET | `/api/v1/assistant/guardrails` | admin | Admin panel |
| DELETE | `/api/v1/assistant/guardrails/blacklist/{id}` | admin | Admin panel |
| GET | `/api/v1/assistant/knowledge` | bearer | Assistant dashboard |
| POST | `/api/v1/assistant/knowledge/search` | bearer | Assistant dashboard |
| DELETE | `/api/v1/assistant/knowledge/{id}` | admin | Assistant dashboard |
| GET | `/api/v1/assistant/campaigns` | bearer | Assistant dashboard |
| POST | `/api/v1/assistant/campaigns` | admin | Assistant dashboard |
| GET | `/api/v1/assistant/campaigns/{id}` | bearer | Assistant dashboard |
| PATCH | `/api/v1/assistant/campaigns/{id}` | admin | Assistant dashboard |
| DELETE | `/api/v1/assistant/campaigns/{id}` | admin | Assistant dashboard |
| POST | `/api/v1/assistant/campaigns/{id}/launch` | admin | Assistant dashboard |

### RAG / Documents

| Method | Path | Auth | Page |
|--------|------|------|------|
| POST | `/api/v1/rag/documents` | admin | Documents page |
| GET | `/api/v1/rag/documents` | bearer | Documents page |
| GET | `/api/v1/rag/documents/{id}` | bearer | Documents page |
| DELETE | `/api/v1/rag/documents/{id}` | admin | Documents page |
| POST | `/api/v1/rag/search` | bearer | Documents page |

### Workflows

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/assistant/workflows` | bearer | Workflows page |
| GET | `/api/v1/assistant/workflows/{id}` | bearer | Workflows page |
| POST | `/api/v1/assistant/workflows` | admin | Workflows page |
| PUT | `/api/v1/assistant/workflows/{id}` | admin | Workflows page |
| DELETE | `/api/v1/assistant/workflows/{id}` | admin | Workflows page |
| GET | `/api/v1/assistant/workflows/{id}/documents` | bearer | Workflows page |
| POST | `/api/v1/assistant/workflows/{id}/documents` | admin | Workflows page |
| DELETE | `/api/v1/assistant/workflows/{id}/documents/{doc_id}` | admin | Workflows page |
| GET | `/api/v1/assistant/workflows/bindings` | bearer | Workflows page |
| PUT | `/api/v1/assistant/workflows/bindings/{session_key}` | admin | Workflows page |
| DELETE | `/api/v1/assistant/workflows/bindings/{session_key}` | admin | Workflows page |

### Users (Admin)

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/users` | admin | Admin panel |
| POST | `/api/v1/users` | admin | Admin panel |
| PATCH | `/api/v1/users/{id}` | admin | Admin panel |
| DELETE | `/api/v1/users/{id}` | admin | Admin panel |

### System

| Method | Path | Auth | Page |
|--------|------|------|------|
| GET | `/api/v1/system/info` | bearer | Admin panel |
| POST | `/api/v1/system/debug` | admin | Admin panel |
| WS | `/ws/v1/logs` | admin token= | Admin panel |

---

## 7. WebSocket Message Quick Reference

### Server to Client

| Type | Key payload fields | When emitted |
|------|--------------------|-------------|
| `token` | `text: string`, `final: boolean` | Once per output token during generation; `final` is `true` when emitting the final response (after all tool calls complete), `false` during preamble text |
| `tool_start` | `tool`, `tool_call_id`, `input` | When the agent invokes a tool |
| `tool_end` | `tool`, `tool_call_id`, `duration_ms`, `error` | When a tool returns |
| `tool_confirm_request` | `confirmation_id`, `tool`, `parameters`, `message` | Tool requires user approval; agent blocked until response |
| `agent_state` | `state` (idle / thinking / analyzing / researching / deep_thinking / writing / delegating / done / error) | On each state transition |
| `memory_update` | `mode`, `tokens_used`, `summarized` | After background summarization or compression |
| `error` | `code`, `message` | Agent-level error; connection stays open |
| `done` | `message_id`, `total_tokens`, `input_tokens`, `output_tokens`, `duration_ms`, `tool_calls` | Turn complete; always the final message for a turn |
| `pong` | (empty) | In response to client `ping` |
| `log_line` | `level`, `logger`, `message`, `timestamp` | Log stream endpoint only |

### Client to Server

| Type | Key payload fields | Purpose |
|------|-------------------|---------|
| `user_message` | `text`, `mode` (normal / think / delegate) | Send a message or trigger an agent turn |
| `tool_confirm` | `confirmation_id`, `action` | Respond to a `tool_confirm_request` |
| `ping` | (empty) | Keepalive; send every 30 seconds |
| `cancel` | (empty) | Abort the in-progress agent turn |

All messages use the JSON envelope described in `docs/API/WEBSOCKET_PROTOCOL.md`, Section 3. Server messages include `session_id`, `seq`, and `ts` fields. Client messages include only `type` and `payload`.

---

## 8. Security Considerations

### Token storage

Store the access token in memory (a module-level variable or React context), never in `localStorage` or `sessionStorage`. Both are readable by any JavaScript on the page.

Store the refresh token in an HttpOnly, Secure, SameSite=Strict cookie when your deployment allows it. If you must store it in memory, be aware it will be lost on page reload and the user will need to log in again.

### CSRF

No CSRF protection is needed for bearer-token APIs. The browser's same-origin policy prevents cross-origin pages from reading the `Authorization` header value.

### XSS

Agent responses are Markdown text. Render them through a safe Markdown renderer (for example, `react-markdown` with the HTML plugin disabled). Do not inject any API string directly into `dangerouslySetInnerHTML`.

Tool parameters displayed in confirmation dialogs may contain arbitrary user-supplied content. Treat all strings from the API as untrusted text.

### Rate limiting

The API enforces rate limits. Respect HTTP 429 responses: read the `Retry-After` header if present, back off, and retry. Do not retry 429 responses in a tight loop.

### File uploads

Validate file types and sizes client-side before submitting to `POST /api/v1/rag/documents`. The server enforces its own size limit, but client-side validation gives faster feedback. The server does not execute uploaded content; it extracts text for embedding.

### Admin-only endpoints

The `PATCH /api/v1/config`, `POST /api/v1/config/reload`, and several other endpoints require `role: 'admin'`. Check `user.role` from the auth context before rendering admin controls. The server enforces this independently — a 403 from an admin endpoint means the current user is not an admin, not that authentication failed.

### Session deletion and WebSocket cleanup

When a session is deleted via `DELETE /api/v1/sessions/{id}`, the server automatically disconnects any active WebSocket connection for that session before archiving it. The frontend should handle the close event gracefully — show a "Session archived" message rather than triggering reconnection.

### WebSocket token expiry

The WebSocket connection uses the access token provided at connect time. If the token expires during a long-running session, the server will close the connection. Handle the close event, refresh the token via `POST /api/v1/auth/refresh`, and reconnect with the new token.

---

## 9. TypeScript File Layout

`CLIENT_CONTRACT.md` defines the canonical file layout. A brief summary:

```
src/lib/api/
  config.ts          — API_BASE_URL, WS_BASE_URL, API_V1, WS_V1
  tokens.ts          — in-memory token store (setTokens, getAccessToken, clearTokens)
  client.ts          — request() with 401 intercept + retry
  pagination.ts      — fetchAllPages, useInfiniteList
  sessionSocket.ts   — SessionSocket class
  types/
    common.ts        — APIError, APIResponse, CursorPage, ResponseMeta
    auth.ts          — RegisterRequest, LoginRequest, TokenPair, UserOut, APIKeyOut
    session.ts       — SessionOut, SessionCreateRequest, SessionConfig, AgentState
    message.ts       — MessageOut, SendMessageRequest, ToolCallRecord, ToolConfirmAction
    tool.ts          — ToolOut, ToolSummary, ToolActionRequest, ToolStatus
    memory.ts        — MemoryStateOut, MemoryModeSwitchRequest
    config.ts        — ConfigOut, ConfigPatchRequest, ProviderOut, ModelOut, WizardStepOut
    websocket.ts     — ServerMessage, ClientMessage, all payload types
```

Full type definitions are in `docs/API/CLIENT_CONTRACT.md`, Section 3.
