# Cogtrix API Client Contract

Audience: React frontend developers (`web_coder` agent)
API version: v1
Last updated: 2026-03-27

Related documents:
- `docs/API/OVERVIEW.md` — API orientation, quick start, environment variables, session lifecycle
- `docs/API/WEBSOCKET_PROTOCOL.md` — full WebSocket message catalogue and reconnection strategy
- `docs/API/WEBUI_DEVELOPMENT_GUIDE.md` — page-by-page React integration guide

---

## 1. Configuration

```typescript
// src/lib/api/config.ts

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
export const WS_BASE_URL  = import.meta.env.VITE_WS_BASE_URL  ?? 'ws://localhost:8000';

export const API_V1  = `${API_BASE_URL}/api/v1`;
export const WS_V1   = `${WS_BASE_URL}/ws/v1`;
```

---

## 2. Authentication

### 2.1 How to attach the JWT

Include the access token in every REST request:

```typescript
headers: {
  'Authorization': `Bearer ${accessToken}`,
  'Content-Type': 'application/json',
}
```

For WebSocket connections the browser API does not allow custom headers.
Use a reverse proxy that injects the ``Authorization`` header from a cookie
or query param — the session WebSocket no longer accepts ``?token=`` (removed
in #1128 to prevent token exposure in access logs).  The log stream WebSocket
(``/ws/v1/logs``) still accepts ``?token=`` for admin auth.

```typescript
// The proxy injects Authorization from a cookie/query param; do NOT use ?token=
new WebSocket(`${WS_V1}/sessions/${sessionId}`);
```

### 2.2 Token storage

Store both tokens in memory (not localStorage) to avoid XSS exposure.
Persist the refresh token in an HttpOnly cookie when possible.

```typescript
// src/lib/api/tokens.ts
let accessToken: string | null = null;
let refreshToken: string | null = null;

export function setTokens(pair: TokenPair): void {
  accessToken = pair.access_token;
  refreshToken = pair.refresh_token;
}
export function getAccessToken(): string | null { return accessToken; }
export function clearTokens(): void { accessToken = null; refreshToken = null; }
```

### 2.3 Authentication behaviour notes

- **First registered user**: The first `POST /api/v1/auth/register` call receives `role: admin`. All subsequent registrations receive `role: user`.
- **Login username field**: The `username` field in `POST /api/v1/auth/login` accepts either a username or an email address.
- **Refresh token rotation**: `POST /api/v1/auth/refresh` invalidates the submitted refresh token before issuing a new pair. If the response is lost (e.g. network timeout), the old token cannot be reused — the client must re-login.
- **Logout scope**: `POST /api/v1/auth/logout` revokes **only the single refresh token** supplied in the request body (`{"refresh_token": "..."}`). To sign a user out across all devices, call `POST /api/v1/auth/logout-all` instead — that endpoint requires the user's current password and invalidates every refresh token issued to the account.
- **API keys**: API key management endpoints (`GET/POST/DELETE /api/v1/auth/api-keys`) are implemented, but API key authentication is not yet wired into request validation. Use JWT bearer tokens for all requests.

### 2.4 Handling 401 / TOKEN_EXPIRED

When any request returns 401 with error code `TOKEN_EXPIRED`:
1. Pause the failing request.
2. Call `POST /api/v1/auth/refresh` with the stored refresh token.
3. If refresh succeeds, store the new token pair and retry the original request.
4. If refresh fails (400/401), clear tokens and redirect to `/login`.

```typescript
// src/lib/api/client.ts
async function request<T>(url: string, opts: RequestInit): Promise<T> {
  const res = await fetch(url, { ...opts, headers: authHeaders() });
  if (res.status === 401) {
    const body = await res.json() as APIResponse<null>;
    if (body.error?.code === 'TOKEN_EXPIRED') {
      await refreshAccessToken();               // throws if refresh fails
      const retry = await fetch(url, { ...opts, headers: authHeaders() });
      return (await retry.json() as APIResponse<T>).data!;
    }
    throw new ApiError(body.error!);
  }
  const body = await res.json() as APIResponse<T>;
  if (body.error) throw new ApiError(body.error);
  return body.data!;
}
```

---

## 3. TypeScript Types

### 3.1 Shared envelope types

```typescript
// src/lib/api/types/common.ts

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface ResponseMeta {
  request_id: string;
  timestamp: string; // ISO 8601 UTC
}

export interface APIResponse<T> {
  data: T | null;
  error: APIError | null;
  meta: ResponseMeta;
}

export interface CursorPage<T> {
  items: T[];
  next_cursor: string | null;
  has_more: boolean;
  total: number | null;
}
```

### 3.2 Authentication types

```typescript
// src/lib/api/types/auth.ts

export interface RegisterRequest {
  username: string;   // 3–64 chars, [a-zA-Z0-9_-]
  email: string;
  password: string;   // 8–128 chars; must include lowercase, uppercase, digit, and special character
}

export interface LoginRequest {
  username: string;   // username OR email address
  password: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;  // seconds
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface UserOut {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'user';
  created_at: string;
}

export interface APIKeyCreateRequest {
  label: string;            // max 128 chars
  expires_in_days?: number | null;  // 1–3650; null = no expiry
}

export interface APIKeyOut {
  id: string;
  label: string;
  key: string | null;        // only present on creation response
  key_prefix: string;        // first 12 chars; always present
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
}
```

### 3.3 Session types

```typescript
// src/lib/api/types/session.ts

export type MemoryMode = 'conversation' | 'code' | 'reasoning';
export type AgentState = 'idle' | 'thinking' | 'analyzing' | 'researching' | 'deep_thinking' | 'writing' | 'delegating' | 'done' | 'error';

export interface TokenCounts {
  input_tokens: number;
  output_tokens: number;
  context_window: number;
}

export interface SessionConfig {
  model?: string | null;
  memory_mode?: MemoryMode | null;
  system_prompt?: string | null;     // max 32768 chars
  prompt_optimizer?: boolean | null;
  parallel_tool_execution?: boolean | null;
  context_compression?: boolean | null;
  max_steps?: number | null;         // 1–200
  agent_name?: string | null;        // named-agent config to apply at session start
}

export interface SessionOut {
  id: string;
  name: string;
  owner_id: string;
  state: AgentState;
  config: SessionConfig;
  token_counts: TokenCounts;
  active_tools: string[];
  created_at: string;
  updated_at: string;
  archived_at: string | null;
}

export interface SessionCreateRequest {
  name?: string;           // max 256 chars; defaults to "Session YYYY-MM-DD HH:MM"
  config?: SessionConfig;
  initial_tools?: string[];      // tool names to pin into active set immediately
  auto_approve_tools?: string[]; // tool names to auto-approve (no confirmation prompt)
}

export interface SessionPatchRequest {
  name?: string | null;    // max 256 chars
  config?: SessionConfig | null;
}
```

### 3.4 Message types

```typescript
// src/lib/api/types/message.ts

export interface ToolCallRecord {
  tool_call_id: string;
  tool_name: string;
  input: Record<string, unknown>;
  output: string | null;
  duration_ms: number | null;
  error: string | null;
}

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export interface MessageOut {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  tool_calls: ToolCallRecord[];
  token_counts: { input: number; output: number } | null;
  created_at: string;
}

export interface SendMessageRequest {
  content: string;                           // 1–65536 chars
  mode?: 'normal' | 'think' | 'delegate';
  optimize_prompt?: boolean | null;          // null uses session-level setting
}

export interface ClearHistoryRequest {
  keep_last?: number | null;   // 0 or null clears all; positive N keeps last N messages
}

export type ToolConfirmAction = 'allow' | 'deny' | 'allow_all' | 'disable' | 'forbid_all' | 'cancel';
```

### 3.5 Tool types

```typescript
// src/lib/api/types/tool.ts

export type ToolStatus = 'active' | 'on_demand' | 'disabled' | 'auto_approved' | 'pinned';

export interface ToolParameterSchema {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default: unknown | null;
  enum: unknown[] | null;
}

export interface ToolOut {
  name: string;
  description: string;
  short_description: string;
  status: ToolStatus;
  requires_confirmation: boolean;
  parameters: ToolParameterSchema[];
  module: string | null;
  is_mcp: boolean;
  mcp_server: string | null;
}

export interface ToolSummary {
  name: string;
  short_description: string;
  status: ToolStatus;
  requires_confirmation: boolean;
  is_mcp: boolean;
}

export interface ToolActionRequest {
  load?: string[] | null;
  unload?: string[] | null;
  enable?: string[] | null;
  disable?: string[] | null;
  auto_approve?: string[] | null;
  revoke_approval?: string[] | null;
}
```

### 3.6 Memory types

```typescript
// src/lib/api/types/memory.ts

export interface MemoryStateOut {
  session_id: string;
  mode: MemoryMode;
  summary: string | null;
  window_messages: number;
  summarized_messages: number;
  tokens_used: number;
  context_window: number;
  vector_recall_enabled: boolean;
  mode_meta: Record<string, unknown>;
  updated_at: string;
}

export interface MemoryModeSwitchRequest {
  mode: MemoryMode;
}
```

### 3.7 Config / provider / model types

```typescript
// src/lib/api/types/config.ts

export interface ProviderOut {
  name: string;
  type: 'openai' | 'ollama' | 'anthropic' | 'google';
  base_url: string | null;
  has_api_key: boolean;
}

export interface ProviderHealthOut {
  name: string;
  reachable: boolean;
  latency_ms: number | null;
  error: string | null;
}

export interface ModelOut {
  alias: string;
  provider: string;
  model_name: string;
  num_ctx: number | null;
  temperature: number | null;
  max_tokens: number | null;
  is_active: boolean;
}

/** POST /api/v1/config/providers request body (admin only) */
export interface ProviderCreateRequest {
  /** Unique alias (1–64 chars, pattern: ^[a-zA-Z0-9][a-zA-Z0-9_-]*$) */
  name: string;
  type: 'openai' | 'ollama' | 'anthropic' | 'google';
  base_url?: string | null;
  api_key?: string | null;
}

/** PATCH /api/v1/config/providers/{name} request body (admin only) */
export interface ProviderPatchRequest {
  /** null clears the override and reverts to the provider-type default */
  base_url?: string | null;
  /** null leaves the key unchanged; empty string removes the key */
  api_key?: string | null;
}

/** POST /api/v1/config/model request body */
export interface ModelSwitchRequest {
  model: string;   // model alias or 'provider/model_name' shorthand
}

export interface ConfigOut {
  /**
   * Active model alias from the models registry (maps to active_model_alias in config).
   * Use POST /api/v1/config/model to change. null means the provider's default model.
   * To find the active model object: models.find(m => m.is_active).
   */
  active_model: string | null;
  memory_mode: string;
  prompt_optimizer: boolean;
  parallel_tool_execution: boolean;
  context_compression: boolean;
  delegate_enabled: boolean;         // /delegate command surface — when false, hide it in the UI
  debug: boolean;
  verbose: boolean;
  config_file_path: string | null;
  providers: ProviderOut[];
  /** Named model registry entries. is_active=true on the currently selected model. */
  models: ModelOut[];
  raw_yaml: string | null;           // admin only; null for non-admin users
  system_prompt: string | null;      // null when using the built-in default
  guardrails: Record<string, unknown> | null;  // assistant guardrail config; null when not set
}

export interface ConfigPatchRequest {
  debug?: boolean | null;
  verbose?: boolean | null;
  prompt_optimizer?: boolean | null;
  parallel_tool_execution?: boolean | null;
  context_compression?: boolean | null;
}

/** Returned by POST /api/v1/config/reload */
export interface ConfigReloadResponse {
  reloaded: boolean;
  config_file_path: string | null;
  warnings: string[];
}

export interface WizardStartRequest {
  docs_url?: string | null;
  edit_existing?: boolean;
}

export interface WizardStepRequest {
  answer?: string | null;
  data?: Record<string, unknown> | null;
}

export interface WizardStepOut {
  wizard_id: string;
  step: number;
  total_steps: number;
  step_name: string;
  question: string | null;
  yaml_preview: string | null;
  complete: boolean;
  /**
   * True when the LLM has produced a YAML config block and the user must
   * explicitly accept or reject it before the wizard advances to the save step.
   * When true, render Accept/Cancel buttons instead of a free-text answer input.
   * Only set during step 1 (Configure).
   */
  requires_acceptance: boolean;
  warnings: string[];
}
```

### 3.8 Workflow types

```typescript
// src/lib/api/types/workflow.ts

export interface WorkflowToolPolicy {
  excluded_tools: string[];
  additional_approved_tools: string[];
}

export interface WorkflowAutoDetect {
  enabled: boolean;
  keywords: string[];
  patterns: string[];
  min_confidence: number;
}

export interface WorkflowOut {
  id: string;
  name: string;
  description: string;
  system_prompt: string | null;
  system_prompt_file: string | null;
  knowledge_base: boolean;
  tool_policy: WorkflowToolPolicy;
  auto_detect: WorkflowAutoDetect;
  created_at: string;
  updated_at: string;
}

/** Used with POST /api/v1/assistant/workflows — schema name: WorkflowCreate */
export interface WorkflowCreateRequest {
  id: string;              // URL-safe slug matching ^[a-zA-Z0-9][a-zA-Z0-9_-]*$
  name: string;
  description?: string;
  system_prompt?: string | null;
  system_prompt_file?: string | null;
  knowledge_base?: boolean;
  tool_policy?: Partial<WorkflowToolPolicy>;
  auto_detect?: Partial<WorkflowAutoDetect>;
}

/** Used with PUT /api/v1/assistant/workflows/{id} — schema name: WorkflowUpdate — all fields optional; unset = unchanged. */
export interface WorkflowUpdateRequest {
  name?: string | null;
  description?: string | null;
  system_prompt?: string | null;
  system_prompt_file?: string | null;
  knowledge_base?: boolean | null;
  tool_policy?: Partial<WorkflowToolPolicy> | null;
  auto_detect?: Partial<WorkflowAutoDetect> | null;
}

/** Used with PUT /api/v1/assistant/workflows/bindings/{session_key} */
export interface BindWorkflowRequest {
  workflow_id: string;     // workflow ID to bind to this session key
}

/** Returned by GET /api/v1/assistant/workflows/bindings and PUT .../bindings/{session_key} */
export interface WorkflowBindingOut {
  session_key: string;     // e.g. "whatsapp::14155551234@c.us"
  workflow_id: string;
  assigned_at: string;     // ISO 8601 UTC
  assigned_by: string | null;
}

export interface WorkflowDocumentOut {
  doc_id: string;              // UUID assigned at upload time
  filename: string;            // original uploaded filename
  size_bytes: number;
  content_type: string | null; // MIME type inferred from filename; null if unknown
  status: string | null;       // present on upload response only (e.g. "saved"); null in list
}
```

### 3.9 Assistant mode types

```typescript
// src/lib/api/types/assistant.ts

export type AssistantStatus = 'running' | 'stopped' | 'starting' | 'stopping' | 'error';
export type ChannelType = 'whatsapp' | 'telegram';
export type FilterMode = 'none' | 'allow' | 'ignore' | 'blacklist';
export type CampaignStatus = 'draft' | 'active' | 'paused' | 'completed' | 'cancelled';
export type CampaignTargetStatus = 'pending' | 'active' | 'completed' | 'failed' | 'escalated';
export type ScheduledMessageStatus = 'pending' | 'firing' | 'sent' | 'failed' | 'cancelled';

export interface ChannelStatusOut {
  name: string;
  type: ChannelType;
  enabled: boolean;
  connected: boolean;
  active_chats: number;
  poll_interval_s: number;
  error: string | null;
}

export interface AssistantStatusOut {
  status: AssistantStatus;
  channels: ChannelStatusOut[];
  started_at: string | null;   // ISO 8601 UTC
  uptime_s: number | null;
}

export interface AssistantStartRequest {
  force_restart?: boolean;
}

export interface OutboundRequest {
  contact_name: string;        // 1–256 chars
  instructions: string;        // 1–8192 chars
  channel?: 'whatsapp' | 'telegram' | null;
}

export interface OutboundResponse {
  session_key: string;
  channel: string;
  chat_id: string;
  contact_name: string;
  response_text: string;
  message_id: string | null;
}

export type SimulateDirection = 'inbound' | 'outbound';

/** Used with POST /api/v1/assistant/simulate — schema name: SimulateRequest */
export interface SimulateRequest {
  channel: string;              // logical channel name, e.g. 'whatsapp' or 'telegram'
  chat_id: string;              // 1–512 chars, e.g. '+1234567890@c.us'
  message: string;              // 1–8192 chars — user message (inbound) or context (outbound)
  direction?: SimulateDirection; // default 'inbound'
  instructions?: string | null; // 1–8192 chars — operator instructions (outbound only)
  sender_name?: string;         // default 'Simulator', max 256 chars
  sender_id?: string;           // default 'simulator', max 256 chars
  persist?: boolean;            // default false — save turn to session memory if true
}

/** Returned by POST /api/v1/assistant/simulate */
export interface SimulateOut {
  channel: string;
  chat_id: string;
  session_key: string;          // "channel::chat_id"
  direction: SimulateDirection;
  response: string;             // agent response text; empty string when suppressed
  suppressed: boolean;          // true when agent called suppress_reply
  deferred: boolean;            // true when agent called defer_processing
  blocked_by_guardrails: boolean; // true when input was blocked before reaching the agent
  guardrail_reason: string | null; // reason string when blocked_by_guardrails is true
  duration_ms: number;          // wall-clock time for the full pipeline (LLM included)
  memory_persisted: boolean;    // true when persist=true and memory was saved successfully
}

export interface ChatSessionOut {
  session_key: string;       // "channel::chat_id"
  channel: string;
  chat_id: string;
  display_name: string | null;     // resolved from phonebook; null if not in phonebook
  message_count: number;           // total messages in this chat session's memory
  last_activity: string | null;    // ISO 8601 UTC timestamp of last processed message
  memory_mode: string;             // active memory mode ("conversation" | "code" | "reasoning")
  is_locked: boolean;              // true when a message is currently being processed
}

export interface ContactOut {
  name: string;
  identifiers: string[];           // list of channel identifiers for this contact
  channels: string[];              // channels this contact appears in (e.g. ["whatsapp", "telegram"])
  prompt: string | null;           // per-contact system prompt override; null uses global prompt
  filter_mode: FilterMode | null;
}

export interface ViolationRecordOut {
  chat_id: string;
  channel: string;
  violation_type: string;    // "input" | "encoding" | "tool_call" | "rate_limit" | "llm_judge"
  detail: string;            // human-readable violation description
  timestamp: string;         // ISO 8601 UTC
}

export interface GuardrailStatusOut {
  blacklisted_chats: string[];     // chat IDs currently on the auto-blacklist
  total_violations: number;        // total recorded violations across all chats
  recent_violations: ViolationRecordOut[];  // most recent 50 violation records
}

export interface DeferredRecordOut {
  session_key: string;              // "channel::chat_id"
  fire_at: string;                  // ISO 8601 UTC — when re-processing will fire
  pending_messages: string[];       // message texts queued for the re-processing batch
  depth: number;                    // current deferral depth
  max_depth: number;                // maximum allowed deferral depth
  status: 'pending' | 'firing';
  created_at: string;               // ISO 8601 UTC
}

export interface KnowledgeFactOut {
  id: string;                       // fact hash identifier (not a UUID)
  text: string;                     // "entity: fact" format
  source_chat: string | null;       // session key that produced this fact
  source_channel: string | null;    // channel name that produced this fact
  created_at: string;               // ISO 8601 UTC
  relevance_score: number | null;   // cosine similarity from last retrieval; null when listed directly
}

export interface KnowledgeSearchRequest {
  query: string;    // 1–1024 chars
  top_k?: number;   // 1–100, default 10
}

export interface CampaignTargetIn {
  contact_name: string;
  channel?: string | null;
}

export interface CampaignTargetOut {
  contact_name: string;
  channel: string;
  chat_id: string;
  status: CampaignTargetStatus;
  follow_ups_sent: number;
  last_outbound_at: string | null;
  last_reply_at: string | null;
  completion_reason: string | null;
}

export interface CampaignCreateRequest {
  name: string;                         // 1–256 chars
  goal: string;                         // 1–1024 chars
  instructions: string;                 // 1–8192 chars
  targets: CampaignTargetIn[];          // 1–100 items
  max_follow_ups?: number;              // 0–20, default 3
  follow_up_interval_hours?: number;    // 0.5–720, default 24
  auto_launch?: boolean;                // default false
}

export interface CampaignUpdateRequest {
  name?: string | null;
  goal?: string | null;
  instructions?: string | null;
  /**
   * Any CampaignStatus value is accepted (draft, active, paused, completed, cancelled).
   * To programmatically pause a campaign use 'paused'; to cancel use 'cancelled'.
   */
  status?: CampaignStatus | null;
  max_follow_ups?: number | null;
  follow_up_interval_hours?: number | null;
}

export interface CampaignOut {
  id: string;
  name: string;
  goal: string;
  instructions: string;
  targets: CampaignTargetOut[];
  max_follow_ups: number;
  follow_up_interval_hours: number;
  status: CampaignStatus;
  created_at: string;
  updated_at: string;
}

export interface ScheduledMessageOut {
  id: string;
  chat_id: string;
  channel: string;
  recipient: string;
  text: string;
  send_at: string;
  created_at: string;
  attempts: number;
  max_attempts: number;
  status: ScheduledMessageStatus;
}

export interface ScheduledMessageEditRequest {
  text?: string | null;          // max 4096 chars
  send_at?: string | null;       // ISO 8601 UTC
}
```

### 3.10 User management types

```typescript
// src/lib/api/types/user.ts

export interface UserCreateRequest {
  username: string;        // ^[a-zA-Z0-9_-]+$ (3–64 chars)
  email: string;           // valid email
  password: string;        // 8–128 chars; must include lowercase, uppercase, digit, and special character
  role?: 'user' | 'admin'; // default: 'user'
}

export interface UserUpdateRequest {
  role?: 'user' | 'admin' | null;
}
```

### 3.11 WebSocket message types

```typescript
// src/lib/api/types/websocket.ts

// --- Server → Client ---

export type ServerMessageType =
  | 'token'
  | 'tool_start'
  | 'tool_end'
  | 'tool_confirm_request'
  | 'agent_state'
  | 'memory_update'
  | 'error'
  | 'done'
  | 'pong'
  | 'log_line';

export interface ServerMessage<P = Record<string, unknown>> {
  type: ServerMessageType;
  session_id: string;
  payload: P;
  seq: number;
  ts: string;  // ISO 8601 UTC
}

export interface TokenPayload {
  text: string;
  /**
   * True when this token is part of the final text response (tool calls have been seen
   * and none are currently in-flight). False for preamble tokens emitted before any tool call.
   */
  final: boolean;
}
export interface ToolStartPayload   { tool: string; tool_call_id: string; input: Record<string, unknown> }
export interface ToolEndPayload     { tool: string; tool_call_id: string; duration_ms: number; error: string | null }
export interface ToolConfirmRequestPayload {
  confirmation_id: string;
  tool: string;
  parameters: Record<string, unknown>;
  message: string;
}
export interface AgentStatePayload  { state: AgentState }
export interface MemoryUpdatePayload { mode: string; tokens_used: number; summarized: boolean }
export interface ErrorPayload       { code: string; message: string }
export interface DonePayload {
  message_id: string;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  duration_ms: number;
  tool_calls: number;
  text?: string;           // full assembled agent response text for this turn
}

/**
 * Payload for 'log_line' messages on the /ws/v1/logs stream.
 * Note: log_line messages on /ws/v1/logs are sent as raw JSON objects —
 * they are NOT wrapped in the standard ServerMessage envelope.
 * Each message is a plain JSON object with these fields.
 */
export interface LogLinePayload {
  type: 'log_line';
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  logger: string;
  message: string;
  timestamp: string;  // ISO 8601 UTC
}

// --- Client → Server ---

export type ClientMessageType = 'user_message' | 'tool_confirm' | 'ping' | 'cancel';

export interface ClientMessage<P = Record<string, unknown>> {
  type: ClientMessageType;
  payload: P;
}

export interface UserMessagePayload { text: string; mode?: 'normal' | 'think' | 'delegate' }
export interface ToolConfirmPayload  { confirmation_id: string; action: ToolConfirmAction }
```

### 3.12 MCP server types

```typescript
// src/lib/api/types/mcp.ts

export type MCPServerStatus = 'connected' | 'disconnected' | 'error' | 'connecting';
export type MCPTransport = 'stdio' | 'sse';

export interface MCPToolSummary {
  name: string;
  description: string;
}

export interface MCPServerOut {
  name: string;
  status: MCPServerStatus;
  transport: MCPTransport;
  url: string | null;        // SSE only
  command: string | null;    // stdio only
  args: string[];            // stdio only
  requires_confirmation: boolean;
  tools: MCPToolSummary[];
  error: string | null;
  connected_at: string | null;  // ISO 8601 UTC
}

/** POST /api/v1/mcp/servers — currently returns 501 NOT_IMPLEMENTED */
export interface MCPServerAddRequest {
  name: string;              // ^[a-zA-Z0-9_-]+$ (max 64 chars)
  transport: MCPTransport;
  url?: string | null;
  command?: string | null;
  args?: string[];
  requires_confirmation?: boolean;
  env?: Record<string, string> | null;
}
```

### 3.13 System and health types

```typescript
// src/lib/api/types/system.ts

export interface SystemInfoOut {
  version: string;
  api_version: string;
  platform: string;
  python_version: string;
  debug: boolean;
  verbose: boolean;
  verbosity: number;       // 0=normal, 1=debug, 2=verbose, 3=trace
  uptime_s: number;
  started_at: string;  // ISO 8601 UTC
}

export interface DebugToggleRequest {
  debug?: boolean | null;    // null = toggle current state
  verbose?: boolean | null;  // null = leave unchanged
  verbosity?: number | null; // 0–3; supersedes debug/verbose when provided
}

export interface HealthOut {
  status: 'ok';
  timestamp: string;  // ISO 8601 UTC
}

export interface ReadinessComponentStatus {
  name: string;
  ok: boolean;
  latency_ms: number | null;
  detail: string | null;
}

export interface ReadinessOut {
  ready: boolean;
  components: ReadinessComponentStatus[];
}
```

### 3.14 RAG / document types

```typescript
// src/lib/api/types/rag.ts

export type IngestStatus = 'pending' | 'processing' | 'indexed' | 'failed';

export interface DocumentOut {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  chunk_count: number;
  status: IngestStatus;
  error: string | null;
  ingested_at: string;  // ISO 8601 UTC
  created_at: string;   // ISO 8601 UTC
}

export interface RAGSearchRequest {
  query: string;                   // 1–2048 chars
  top_k?: number;                  // 1–50, default 5
  document_ids?: string[] | null;  // null = search all documents
}

export interface RAGChunkOut {
  document_id: string;
  document_name: string;
  chunk_index: number;
  text: string;
  score: number;  // cosine similarity 0.0–1.0
}

export interface RAGSearchResponse {
  query: string;
  chunks: RAGChunkOut[];
  total_documents_searched: number;
}
```

---

## 4. WebSocket Usage — Full Example

```typescript
// src/lib/api/sessionSocket.ts

import type {
  ServerMessage, ClientMessage,
  TokenPayload, ToolConfirmRequestPayload, DonePayload,
  ToolConfirmPayload, AgentStatePayload,
} from './types/websocket';

export class SessionSocket {
  private ws: WebSocket | null = null;
  private lastSeq = -1;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  constructor(
    private sessionId: string,
    private getToken: () => string,
    private handlers: {
      onToken: (text: string, final: boolean) => void;
      onToolStart: (tool: string, input: Record<string, unknown>) => void;
      onToolEnd: (tool: string, durationMs: number, error: string | null) => void;
      onToolConfirmRequest: (payload: ToolConfirmRequestPayload) => void;
      onAgentState: (state: string) => void;
      onDone: (payload: DonePayload) => void;
      onError: (code: string, message: string) => void;
      onDisconnect: () => void;
    }
  ) {}

  connect(): void {
    // The reverse proxy injects Authorization from a cookie or query param.
    // Do NOT use ?token= for session WebSocket — it is no longer supported (see #1128).
    const url = `${WS_V1}/sessions/${this.sessionId}?last_seq=${this.lastSeq}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      // Start keepalive ping every 30s
      this.pingInterval = setInterval(() => this.send({ type: 'ping', payload: {} }), 30_000);
    };

    this.ws.onmessage = (event: MessageEvent<string>) => {
      const msg = JSON.parse(event.data) as ServerMessage;
      this.lastSeq = msg.seq;

      switch (msg.type) {
        case 'token': {
          const p = msg.payload as TokenPayload;
          this.handlers.onToken(p.text, p.final);
          break;
        }
        case 'tool_start':
          this.handlers.onToolStart(
            (msg.payload as any).tool,
            (msg.payload as any).input
          );
          break;
        case 'tool_end':
          this.handlers.onToolEnd(
            (msg.payload as any).tool,
            (msg.payload as any).duration_ms,
            (msg.payload as any).error
          );
          break;
        case 'tool_confirm_request':
          this.handlers.onToolConfirmRequest(msg.payload as ToolConfirmRequestPayload);
          break;
        case 'agent_state':
          this.handlers.onAgentState((msg.payload as AgentStatePayload).state);
          break;
        case 'done':
          this.handlers.onDone(msg.payload as DonePayload);
          break;
        case 'error':
          this.handlers.onError((msg.payload as any).code, (msg.payload as any).message);
          break;
      }
    };

    this.ws.onclose = () => {
      if (this.pingInterval) clearInterval(this.pingInterval);
      this.handlers.onDisconnect();
    };
  }

  sendMessage(text: string, mode: 'normal' | 'think' | 'delegate' = 'normal'): void {
    this.send({ type: 'user_message', payload: { text, mode } });
  }

  confirmTool(confirmationId: string, action: ToolConfirmPayload['action']): void {
    this.send({ type: 'tool_confirm', payload: { confirmation_id: confirmationId, action } });
  }

  cancel(): void {
    this.send({ type: 'cancel', payload: {} });
  }

  disconnect(): void {
    this.ws?.close(1000);
  }

  private send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }
}
```

---

## 5. Pagination — Infinite Scroll Pattern

All list endpoints support cursor-based pagination.
Use the `next_cursor` value as the `cursor` query parameter in the next request.

**Important**: The `GET /api/v1/sessions/{id}/messages` endpoint uses a raw message UUID as its cursor, not a base64-encoded opaque cursor. All other paginated endpoints use opaque base64url cursors. Always treat cursors as opaque strings — do not parse them.

```typescript
// src/lib/api/pagination.ts

export async function fetchAllPages<T>(
  fetchPage: (cursor: string | null) => Promise<CursorPage<T>>
): Promise<T[]> {
  const items: T[] = [];
  let cursor: string | null = null;

  do {
    const page = await fetchPage(cursor);
    items.push(...page.items);
    cursor = page.next_cursor;
  } while (cursor !== null);

  return items;
}

// React hook for infinite scroll:
export function useInfiniteList<T>(
  fetchFn: (cursor: string | null) => Promise<CursorPage<T>>
) {
  const [items, setItems] = React.useState<T[]>([]);
  const [cursor, setCursor] = React.useState<string | null>(null);
  const [hasMore, setHasMore] = React.useState(true);
  const [loading, setLoading] = React.useState(false);

  const loadMore = React.useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const page = await fetchFn(cursor);
      setItems(prev => [...prev, ...page.items]);
      setCursor(page.next_cursor);
      setHasMore(page.has_more);
    } finally {
      setLoading(false);
    }
  }, [cursor, hasMore, loading, fetchFn]);

  return { items, hasMore, loading, loadMore };
}
```

---

## 6. Error Handling Guide

### HTTP Status Codes

| Status | Error Code                  | UI action |
|--------|-----------------------------|-----------|
| 400    | `BAD_REQUEST`               | Show inline error (e.g. self-demote/self-delete on user management) |
| 400/422 | `INVALID_CURSOR`           | Restart pagination from the beginning. Note: session list and RAG endpoints return 400; assistant, scheduled, knowledge, and deferred list endpoints return 422. |
| 400    | `INVALID_DOCUMENT_ID`       | Show "Invalid document ID" inline error |
| 400    | `CONTACT_NOT_FOUND`         | Show "Contact not found in phonebook" inline error |
| 400    | `CHANNEL_NOT_AVAILABLE`     | Show "Channel not available for this contact" inline error |
| 400    | `VALIDATION_ERROR`          | Display field-level validation errors from `details.fields` (see below) |
| 401    | `UNAUTHORIZED`              | Redirect to /login |
| 401    | `TOKEN_EXPIRED`             | Silently refresh token and retry |
| 403    | `FORBIDDEN`                 | Show "Access denied" toast |
| 404    | `*_NOT_FOUND`               | Show "Not found" inline message |
| 409    | `SESSION_NAME_DUPLICATE`    | Show "A session with this name already exists" inline error |
| 409    | `TURN_IN_PROGRESS`          | Show "Please wait for the current response to complete" toast; retry after `done` |
| 409    | `ASSISTANT_ALREADY_RUNNING` | Show "Assistant is already running" toast |
| 409    | `ASSISTANT_NOT_RUNNING`     | Show "Assistant is not running" toast |
| 409    | `CONFLICT`                  | Show "Already exists" inline error (e.g. duplicate username on user creation, duplicate workflow) |
| 409    | `PROVIDER_IN_USE`           | Show "Provider is referenced by one or more models and cannot be deleted" inline error |
| 409    | `CAMPAIGNS_NOT_AVAILABLE`   | Show "Campaign manager not available" toast |
| 409    | `CAMPAIGN_NOT_LAUNCHABLE`   | Show "Campaign cannot be launched (must be in draft or paused state)" toast |
| 410    | `GONE`                      | Endpoint removed — update client code (e.g. POST /config/provider) |
| 413    | `VALIDATION_ERROR`          | Show "File too large (max 50 MB)" error (413 responses use code `VALIDATION_ERROR`, not `PAYLOAD_TOO_LARGE`) |
| 415    | `VALIDATION_ERROR`          | Show "Unsupported file type" error — allowed: `.pdf`, `.txt`, `.md`, `.markdown`, `.csv` (415 responses use code `VALIDATION_ERROR`, not `UNSUPPORTED_MEDIA_TYPE`) |
| 422    | `VALIDATION_ERROR`          | Display field-level errors (same format as 400) |
| 422    | `PROVIDER_UNREACHABLE`      | Wizard step 0/1 connection test failed or LLM call failed — show inline error, allow retry |
| 422    | `MODEL_NOT_FOUND`           | `PATCH /sessions/{id}` — model alias not found in registry — show "Model not found" inline error |
| 422    | `WIZARD_STEP_ERROR`         | Wizard step could not complete — show error message from `error.message` |
| 422    | `CONFIG_INVALID`            | Config reload or wizard save rejected the YAML — show details |
| 500    | `INGEST_FAILED`             | Show "Document processing failed" error toast |
| 500    | `MEMORY_CLEAR_FAILED`       | Show "Could not clear memory" error toast |
| 501    | `NOT_IMPLEMENTED`           | Feature not yet available; show informational toast |
| 503    | `SERVICE_UNAVAILABLE`       | Server dependencies not ready (config/registry missing, or assistant service failed to start) — show "Service temporarily unavailable" banner; retry later |
| 503    | `ASSISTANT_START_FAILED`    | Assistant service start failed — show "Assistant failed to start" error with `error.message` for details |
| 503    | `MCP_RESTART_FAILED`        | Show "MCP server restart failed" error toast |
| 500    | `INTERNAL_ERROR`            | Show generic error toast; log `meta.request_id` |

### WebSocket Close Codes

| Close code | UI action |
|------------|-----------|
| 4000       | Session registry unavailable (server starting) — retry after delay |
| 4001       | Redirect to /login |
| 4003       | Show "Access denied" and redirect to sessions list |
| 4004       | Session not found — navigate to sessions list |
| 1001       | Server going away (shutdown) or replaced by new connection — attempt reconnect |
| 1011       | Show "Server error" and attempt reconnect |

### Validation error format

When the error code is `VALIDATION_ERROR`, the `details` field contains a `fields` object keyed by field name. Each field maps to an array of error objects with a stable `code` and a human-readable `message` safe to display directly in the UI.

```typescript
// src/lib/api/types/validation.ts

export interface FieldError {
  code: ValidationCode;
  message: string;  // Human-readable — display directly in the UI
}

// Nested fields use nested objects (e.g. config.max_steps):
// details.fields.config.max_steps = [{ code: "OUT_OF_RANGE", message: "..." }]
export type FieldErrors = Record<string, FieldError[] | Record<string, FieldError[]>>;

export interface ValidationDetails {
  fields: FieldErrors;
}

export type ValidationCode =
  | 'TOO_SHORT'       // min_length constraint violated
  | 'TOO_LONG'        // max_length constraint violated
  | 'INVALID_FORMAT'  // pattern constraint violated
  | 'INVALID_VALUE'   // value_error (e.g. invalid email)
  | 'REQUIRED'        // required field missing
  | 'OUT_OF_RANGE'    // ge/le/gt/lt constraint violated
  | 'INVALID_CHOICE'  // Literal/enum value not in allowed set
  | 'INVALID_JSON'    // malformed JSON body
  | 'TYPE_MISMATCH'   // wrong type (e.g. string where int expected)
  | 'INVALID';        // catch-all for unrecognized constraint types
```

**Example response:**
```json
{
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request body or query parameter validation failed.",
    "details": {
      "fields": {
        "username": [
          { "code": "INVALID_FORMAT", "message": "Username may only contain letters, digits, hyphens, and underscores." }
        ],
        "password": [
          { "code": "TOO_SHORT", "message": "Password must be at least 8 characters." }
        ]
      }
    }
  },
  "meta": { "request_id": "...", "timestamp": "..." }
}
```

**Rendering field errors in React:**
```typescript
function renderFieldErrors(details: ValidationDetails | null, fieldName: string): string[] {
  if (!details?.fields) return [];
  const errors = details.fields[fieldName];
  if (Array.isArray(errors)) return errors.map(e => e.message);
  return []; // nested object — drill deeper for sub-fields
}
```

### Error display pattern

```typescript
function handleApiError(error: APIError): void {
  switch (error.code) {
    case 'TOKEN_EXPIRED':
      // handled by the client interceptor — never reaches here
      break;
    case 'SESSION_NOT_FOUND':
      navigate('/sessions');
      toast.error('Session not found');
      break;
    case 'PROVIDER_UNREACHABLE':
      showBanner('LLM provider is offline. Check configuration.');
      break;
    case 'SERVICE_UNAVAILABLE':
      showBanner('Service temporarily unavailable. Please try again shortly.');
      break;
    case 'INTERNAL_ERROR':
      toast.error(`Server error (ID: ${requestId})`);
      break;
    default:
      toast.error(error.message);
  }
}
```

---

## 7. Endpoint Reference Summary

### Authentication

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| POST | /api/v1/auth/register | none | 201 | Register new account |
| POST | /api/v1/auth/login | none | 200 | Login, receive token pair; `username` field accepts username or email |
| POST | /api/v1/auth/refresh | none | 200 | Refresh access token |
| POST | /api/v1/auth/logout | bearer | 200 | Revoke the **single** refresh token in the request body |
| POST | /api/v1/auth/logout-all | bearer | 200 | Revoke **all** refresh tokens for the user (requires password confirmation) |
| GET  | /api/v1/auth/me | bearer | 200 | Get current user profile |
| GET  | /api/v1/auth/api-keys | bearer | 200 | List API keys (`?cursor=`, `?limit=` 1–100, default 20) |
| POST | /api/v1/auth/api-keys | bearer | 201 | Create API key |
| DELETE | /api/v1/auth/api-keys/{id} | bearer | 200 | Revoke API key |

**Error codes:** `VALIDATION_ERROR` (409 on duplicate username/email for register), `UNAUTHORIZED`, `TOKEN_EXPIRED`, `NOT_FOUND`, `FORBIDDEN`.

### Sessions

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| POST | /api/v1/sessions | bearer | 201 | Create session |
| GET | /api/v1/sessions | bearer | 200 | List sessions (`?cursor=`, `?limit=` 1–100 default 20, `?include_archived=true`) |
| GET | /api/v1/sessions/{id} | bearer | 200 | Get session details |
| PATCH | /api/v1/sessions/{id} | bearer | 200 | Update session name/config |
| DELETE | /api/v1/sessions/{id} | bearer | 200 | Archive session (soft delete); add `?permanent=true` to hard-delete the row and all messages (non-recoverable) |
| POST | /api/v1/sessions/{id}/restore | bearer | 200 | Restore an archived session (clears `archived_at`) |
| POST | /api/v1/sessions/{id}/restore | bearer | 200 | Restore an archived session (clears `archived_at`) |

**Error codes:**
- `SESSION_NOT_FOUND` (404), `FORBIDDEN` (403), `SESSION_NAME_DUPLICATE` (409)
- `PATCH`: `TURN_IN_PROGRESS` (409) when an agent turn is in progress; `MODEL_NOT_FOUND` (422) when `config.model` refers to an unknown model alias.
- `DELETE ?permanent=true`: also returns 404 `SESSION_NOT_FOUND` after disconnect if the session was already hard-deleted.

### Messages

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| POST | /api/v1/sessions/{id}/messages | bearer | 202 | Send message, queue agent turn (`mode`: `normal`, `think`, `delegate`) |
| GET | /api/v1/sessions/{id}/messages | bearer | 200 | List history (`?cursor=<uuid>`, `?limit=` 1–200 default 50) |
| DELETE | /api/v1/sessions/{id}/messages | bearer | 200 | Clear history (optional body: `ClearHistoryRequest`) |
| WS | /ws/v1/sessions/{id} | token= | — | Stream agent output |

**Message history cursor note:** the `cursor` for `GET .../messages` is a raw message UUID, not a base64-encoded cursor. Pass the `id` of the oldest message on the current page to fetch the page before it.

**Error codes:**
- `SESSION_NOT_FOUND` (404), `FORBIDDEN` (403), `TURN_IN_PROGRESS` (409)

**WebSocket (`/ws/v1/sessions/{id}`):**
- Query params: `token=<jwt>`, `last_seq=<n>` (for reconnect/replay)
- Messages with `seq > last_seq` are replayed on reconnect (30-second replay buffer)
- Idle timeout: 90 seconds (server closes with 1001 on silence)
- Send `ping` every 30 seconds; server responds with `pong`
- Close codes: `4000` (registry unavailable), `4001` (unauthorized/expired), `4003` (forbidden — not session owner), `4004` (session not found)

### Memory

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/sessions/{id}/memory | bearer | 200 | Get memory state |
| DELETE | /api/v1/sessions/{id}/memory | bearer | 200 | Clear memory |
| PATCH | /api/v1/sessions/{id}/memory | bearer | 200 | Switch memory mode |

**Error codes:** `SESSION_NOT_FOUND` (404), `FORBIDDEN` (403), `MEMORY_CLEAR_FAILED` (500).

### Tools

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/tools | bearer | 200 | List all tools (`?search=`, `?include_mcp=true`, `?cursor=`, `?limit=` 1–500 default 100) |
| GET | /api/v1/tools/{name} | bearer | 200 | Get tool details |
| GET | /api/v1/sessions/{id}/tools | bearer | 200 | Get session tool status — returns `list[ToolSummary]` (not paginated) |
| PATCH | /api/v1/sessions/{id}/tools | bearer | 200 | Manage session tools |

**Error codes:** `TOOL_NOT_FOUND` (404), `SESSION_NOT_FOUND` (404), `TOOL_ALREADY_ACTIVE` (409), `TOOL_ALREADY_DISABLED` (409), `TOOL_EXPANSION_FAILED` (422).

### Configuration

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/config | bearer | 200 | Get config snapshot (`raw_yaml` is admin-only; null for regular users) |
| PATCH | /api/v1/config | admin | 200 | Partial config update |
| POST | /api/v1/config/reload | admin | 200 | Reload config from disk |
| GET | /api/v1/config/providers | bearer | 200 | List providers |
| GET | /api/v1/config/providers/{name} | bearer | 200 | Get provider |
| POST | /api/v1/config/providers | admin | 201 | Add a new provider at runtime; body: `ProviderCreateRequest` |
| PATCH | /api/v1/config/providers/{name} | admin | 200 | Update provider `base_url` or `api_key`; body: `ProviderPatchRequest` |
| DELETE | /api/v1/config/providers/{name} | admin | 200 | Remove a provider (409 if any model references it) |
| POST | /api/v1/config/provider | admin | 410 | **Removed** — always returns 410 `GONE`; use `POST /config/providers` instead |
| POST | /api/v1/config/providers/{name}/health | bearer | 200 | Provider health check |
| GET | /api/v1/config/models | bearer | 200 | List models (`?provider=name` to filter) |
| POST | /api/v1/config/model | admin | 200 | Switch active model; body: `ModelSwitchRequest` |
| POST | /api/v1/config/wizard | admin | 201 | Start setup wizard — returns `WizardStepOut` with `step: 0` |
| POST | /api/v1/config/wizard/{id}/step | admin | 200 | Advance wizard one step |
| DELETE | /api/v1/config/wizard/{id} | admin | 200 | Cancel and discard wizard session |

**Error codes:**
- `NOT_FOUND` (404) for missing provider/wizard session
- `CONFLICT` (409) for `POST /config/providers` when the provider name already exists
- `PROVIDER_IN_USE` (409) for `DELETE /config/providers/{name}` when a model references the provider
- `CONFIG_INVALID` (422) for `reload` if YAML is invalid, or for `POST /config/providers` with an unknown provider type
- `PROVIDER_UNREACHABLE` (422) for wizard step 0 connection failure or step 1 LLM failure
- `WIZARD_STEP_ERROR` (422) for other wizard step failures
- `GONE` (410) for deprecated `POST /config/provider`

### MCP Servers

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/mcp/servers | bearer | 200 | List MCP servers |
| POST | /api/v1/mcp/servers | admin | 501 | Add MCP server (**not yet implemented — always returns 501**) |
| GET | /api/v1/mcp/servers/{name} | bearer | 200 | Get server details |
| DELETE | /api/v1/mcp/servers/{name} | admin | 501 | Remove server (**not yet implemented — always returns 501**) |
| POST | /api/v1/mcp/servers/{name}/restart | admin | 200 | Restart server |

**Error codes:** `MCP_SERVER_NOT_FOUND` (404), `MCP_RESTART_FAILED` (503), `NOT_IMPLEMENTED` (501).

### Assistant Mode

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/assistant/status | bearer | 200 | Service status |
| POST | /api/v1/assistant/start | admin | 200 | Start service |
| POST | /api/v1/assistant/stop | admin | 200 | Stop service |
| POST | /api/v1/assistant/outbound | admin | 200 | Send operator-initiated outbound message |
| GET | /api/v1/assistant/chats | bearer | 200 | List chat sessions (`?cursor=`, `?limit=` 1–200 default 50, `?channel=whatsapp\|telegram`) |
| GET | /api/v1/assistant/chats/{key}/messages | bearer | 200 | Per-chat message history (`?cursor=`, `?limit=`) |
| GET | /api/v1/assistant/scheduled | bearer | 200 | List scheduled messages (`?cursor=`, `?limit=`, `?channel=`, `?chat_id=`) |
| PATCH | /api/v1/assistant/scheduled/{id} | admin | 200 | Edit scheduled message |
| DELETE | /api/v1/assistant/scheduled/{id} | admin | 200 | Cancel scheduled message |
| GET | /api/v1/assistant/deferred | bearer | 200 | List deferred records — returns `list[DeferredRecordOut]` (flat, not paginated; `?channel=` to filter) |
| DELETE | /api/v1/assistant/deferred/{key} | bearer | 200 | Cancel deferred record |
| GET | /api/v1/assistant/contacts | bearer | 200 | List phonebook contacts |
| GET | /api/v1/assistant/guardrails | admin | 200 | Guardrail status |
| DELETE | /api/v1/assistant/guardrails/blacklist/{chat_id} | admin | 200 | Remove from blacklist |
| GET | /api/v1/assistant/knowledge | bearer | 200 | List knowledge facts (`?cursor=`, `?limit=`, `?source_chat=`) |
| POST | /api/v1/assistant/knowledge/search | bearer | 200 | Semantic search over facts |
| DELETE | /api/v1/assistant/knowledge/{fact_id} | admin | 200 | Delete fact |
| GET | /api/v1/assistant/campaigns | bearer | 200 | List campaigns (`?status_filter=`) |
| POST | /api/v1/assistant/campaigns | admin | 200 | Create campaign |
| GET | /api/v1/assistant/campaigns/{id} | bearer | 200 | Get campaign details |
| PATCH | /api/v1/assistant/campaigns/{id} | admin | 200 | Update campaign |
| DELETE | /api/v1/assistant/campaigns/{id} | admin | 200 | Delete campaign |
| POST | /api/v1/assistant/campaigns/{id}/launch | admin | 200 | Launch campaign |
| POST | /api/v1/assistant/simulate | admin | 200 | Simulate a message turn without delivery |

**Error codes for assistant endpoints:**
- `ASSISTANT_ALREADY_RUNNING` (409) — `POST /start` when already running and `force_restart=false`
- `ASSISTANT_NOT_RUNNING` (409) — `POST /stop` and all endpoints requiring a running service
- `SERVICE_UNAVAILABLE` (503) — `POST /start` when app config or tool registry is missing; `POST /outbound` and `POST /simulate` when handler is not initialised
- `ASSISTANT_START_FAILED` (503) — `POST /start` when service initialisation throws
- `CONTACT_NOT_FOUND` (400) — `POST /outbound` when contact not in phonebook
- `CHANNEL_NOT_AVAILABLE` (400) — `POST /outbound` when resolved channel is not active
- `SCHEDULED_MSG_NOT_FOUND` (404) — `PATCH/DELETE /scheduled/{id}`
- `DEFERRED_MSG_NOT_FOUND` (404) — `DELETE /deferred/{key}`
- `NOT_FOUND` (404) — `DELETE /guardrails/blacklist/{chat_id}`; `GET .../chats/{key}/messages` when chat not found
- `FACT_NOT_FOUND` (404) — `DELETE /knowledge/{fact_id}`
- `CAMPAIGNS_NOT_AVAILABLE` (409) — all `/campaigns` endpoints when the campaign manager is not initialised
- `CAMPAIGN_NOT_FOUND` (404) — individual campaign endpoints
- `CAMPAIGN_NOT_LAUNCHABLE` (409) — `POST /campaigns/{id}/launch` when campaign is not in draft or paused state

**Notes:**
- `GET /assistant/chats/{key}/messages` requires the assistant to be running; returns `409 ASSISTANT_NOT_RUNNING` if not.
- `POST /assistant/campaigns` returns **200** (not 201).
- `KnowledgeSearchRequest`: `query` max 1024 chars, `top_k` 1–100 default 10. These differ from `RAGSearchRequest` (query max 2048, top_k 1–50 default 5).

### Workflows

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| POST | /api/v1/assistant/workflows | admin | 201 | Create workflow |
| GET | /api/v1/assistant/workflows | bearer | 200 | List workflows (paginated) |
| GET | /api/v1/assistant/workflows/bindings | bearer | 200 | List all chat-to-workflow bindings |
| GET | /api/v1/assistant/workflows/{id} | bearer | 200 | Get workflow |
| PUT | /api/v1/assistant/workflows/{id} | admin | 200 | Replace workflow definition (all fields optional; unset fields unchanged) |
| DELETE | /api/v1/assistant/workflows/{id} | admin | 200 | Delete workflow |
| POST | /api/v1/assistant/workflows/{id}/documents | admin | 202 | Upload document to workflow knowledge base |
| GET | /api/v1/assistant/workflows/{id}/documents | bearer | 200 | List workflow documents |
| DELETE | /api/v1/assistant/workflows/{id}/documents/{doc_id} | admin | 200 | Delete workflow document |
| PUT | /api/v1/assistant/workflows/bindings/{session_key} | admin | 200 | Bind chat to workflow (`session_key` = `channel::chat_id`) |
| DELETE | /api/v1/assistant/workflows/bindings/{session_key} | admin | 200 | Remove binding |

**Error codes:** `CONFLICT` (409 — duplicate workflow on create), `NOT_FOUND` (404), `SERVICE_UNAVAILABLE` (503 — workflow registry not initialised), `VALIDATION_ERROR` (413 file too large, 415 unsupported type, 400 path traversal on document upload).

### User Management

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/users | admin | 200 | List all users |
| POST | /api/v1/users | admin | 201 | Create user |
| PATCH | /api/v1/users/{id} | admin | 200 | Update user role |
| DELETE | /api/v1/users/{id} | admin | 200 | Delete user |

**Error codes:** `CONFLICT` (409 — duplicate username/email on create), `NOT_FOUND` (404), `BAD_REQUEST` (400 — attempting to demote or delete yourself).

### RAG / Documents

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| POST | /api/v1/rag/documents | admin | 202 | Upload and ingest document (multipart/form-data) |
| GET | /api/v1/rag/documents | bearer | 200 | List documents (`?cursor=`, `?limit=` 1–200 default 50, `?status=`) |
| GET | /api/v1/rag/documents/{id} | bearer | 200 | Get document details |
| DELETE | /api/v1/rag/documents/{id} | admin | 200 | Delete document |
| POST | /api/v1/rag/search | bearer | 200 | Semantic search |

**Error codes:** `DOCUMENT_NOT_FOUND` (404), `INVALID_DOCUMENT_ID` (400), `VALIDATION_ERROR` (413 file too large >50 MB, 415 unsupported type — both use `VALIDATION_ERROR` code, not `PAYLOAD_TOO_LARGE`/`UNSUPPORTED_MEDIA_TYPE`).

**Allowed upload types:** `.pdf`, `.txt`, `.md`, `.markdown`, `.csv`. Maximum size: 50 MB.

### Health (unauthenticated)

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/health | none | 200 | Liveness check — always 200 if server is up |
| GET | /api/v1/health/ready | none | 200/503 | Readiness check — 503 when any critical component fails |

**Note:** `/health/ready` returns 503 with `ReadinessOut` (not an error envelope) when not ready. Check `data.ready` and `data.components` to identify which component failed.

### System

| Method | Path | Auth | Status | Description |
|--------|------|------|--------|-------------|
| GET | /api/v1/system/info | bearer | 200 | System info |
| POST | /api/v1/system/debug | admin | 200 | Toggle debug/verbose logging (optional body) |
| WS | /ws/v1/logs | admin | — | Live log stream |

**Live log WebSocket (`/ws/v1/logs`):**
- Query params: `token=<jwt>`, `level=INFO` (default; valid: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- Close codes: `4001` (no token or invalid token), `4003` (not admin)
- Idle timeout: 90 seconds
- **Log messages are NOT wrapped in the standard `ServerMessage` envelope.** Each message is a raw JSON object matching `LogLinePayload`: `{ type: "log_line", level, logger, message, timestamp }`.

---

## 8. Server Configuration Notes

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COGTRIX_JWT_SECRET` | **yes** | JWT signing secret (min 32 chars). Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `COGTRIX_DB_URL` | no | Database URL (default `sqlite+aiosqlite:///data/api/cogtrix.db`; use `asyncpg://` for PostgreSQL) |
| `COGTRIX_CORS_ORIGINS` | no | Comma-separated allowed origins (overrides defaults: `localhost:5173`, `localhost:3000`, `app.cogtrix.ai`) |
| `COGTRIX_API_HOST` | no | Bind host (default `0.0.0.0`) |
| `COGTRIX_API_PORT` | no | Bind port (default `8000`) |
| `COGTRIX_API_WORKERS` | no | Number of uvicorn workers (default `1`) |
| `COGTRIX_CONFIG_FILE` | no | Path to config file (JSON or YAML) |
| `COGTRIX_DEBUG` | no | Enable debug logging when set to any non-empty value |

### CLI flags

The API server supports these command-line flags via `python -m src.api`:

```bash
python -m src.api                              # defaults
python -m src.api --debug                      # debug logging to cogtrix-api.log
python -m src.api --log                        # info logging to cogtrix-api.log
python -m src.api --log-file /tmp/api.log      # custom log file
python -m src.api --config-file prod.yaml      # explicit config
python -m src.api --host 0.0.0.0 --port 9000   # custom bind address
python -m src.api --workers 4                  # multiple workers
python -m src.api --reload                     # auto-reload (development)
```

### Interactive API explorer

The server exposes auto-generated API documentation:

| URL | Description |
|-----|-------------|
| `/api/v1/docs` | Swagger UI |
| `/api/v1/redoc` | ReDoc |
| `/api/v1/openapi.json` | OpenAPI 3.x schema |

### Assistant auto-start

If `services.assistant.auto_start: true` is set in the config file, the assistant
service starts automatically at boot. Calling `POST /api/v1/assistant/start` when
the service is already running returns `409 ASSISTANT_ALREADY_RUNNING`.

---

## 9. Setup Wizard Flow

The setup wizard is a 3-step interactive session that generates a validated YAML config file.

**Step 0 — Connect to LLM** (`POST /api/v1/config/wizard`)

Start the wizard. Returns `WizardStepOut` with `step: 0`, `step_name: "Connect to LLM"`,
and a `question` prompt. Store the `wizard_id`.

**Step 1 — Advance to Configure** (`POST /api/v1/config/wizard/{id}/step`)

Submit `data: { provider_type, api_key, base_url, model }`. The server tests the connection.

- On connection test failure or LLM invocation failure: `422 PROVIDER_UNREACHABLE` — show the error inline and allow the user to retry.
- On success: returns `WizardStepOut` with `step: 1`, `step_name: "Configure"`, and the first
  LLM-generated question in `question`.

**Step 1 loop — Configure Q&A** (repeating `POST /api/v1/config/wizard/{id}/step`)

Submit `answer: "<user response>"` for each question.

When the LLM produces a YAML config block, the response includes:
- `requires_acceptance: true`
- `yaml_preview`: masked YAML (secrets replaced with `***`)
- `question`: explanatory text without the YAML block

Render Accept/Cancel buttons instead of a free-text input.

To accept: re-submit with `data: { accept: true }` plus any free-text `answer`.
To reject: continue with a normal `answer` requesting changes.

If the LLM invocation fails during step 1 Q&A: `422 PROVIDER_UNREACHABLE`.
If the step cannot be completed for other reasons: `422 WIZARD_STEP_ERROR`.

**Step 2 — Save** (triggered automatically on accept)

The server validates the YAML, writes `~/.cogtrix.yaml`, reloads config, and returns
`WizardStepOut` with `complete: true`. The `yaml_preview` field holds the final masked config.

- On invalid YAML: `422 WIZARD_STEP_ERROR` or `422 CONFIG_INVALID`.

To cancel at any point: `DELETE /api/v1/config/wizard/{id}` — no file is written.
The wizard session expires after 30 minutes of inactivity.

```typescript
// Minimal wizard flow
async function runWizard(providerType: string, apiKey: string, model: string) {
  // Start
  const start = await api.post<WizardStepOut>('/config/wizard', { edit_existing: false });
  const { wizard_id } = start;

  // Step 0 → Step 1: connect
  let step = await api.post<WizardStepOut>(`/config/wizard/${wizard_id}/step`, {
    data: { provider_type: providerType, api_key: apiKey, model },
  });

  // Step 1 loop: answer questions until requires_acceptance
  while (!step.complete) {
    if (step.requires_acceptance) {
      // Show yaml_preview to user; await Accept/Cancel
      const accepted = await showAcceptDialog(step.yaml_preview);
      if (!accepted) {
        step = await api.post<WizardStepOut>(`/config/wizard/${wizard_id}/step`, {
          answer: "Please revise the configuration.",
        });
        continue;
      }
      step = await api.post<WizardStepOut>(`/config/wizard/${wizard_id}/step`, {
        answer: "Accept",
        data: { accept: true },
      });
    } else {
      const userAnswer = await promptUser(step.question);
      step = await api.post<WizardStepOut>(`/config/wizard/${wizard_id}/step`, {
        answer: userAnswer,
      });
    }
  }

  return step.yaml_preview; // complete: true — config saved
}
```
