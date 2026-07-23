# Cogtrix API — Overview

API version: v1
Last updated: 2026-04-29

---

## What the API is

The Cogtrix API is a FastAPI application that exposes the full Cogtrix agent platform over HTTP and WebSocket. It is the backend for the React web frontend and supports programmatic access via JWT bearer tokens.

Two transport layers are used:

- **REST** (`/api/v1/`) — create and manage sessions, send messages, load tools, configure the system, manage users, upload documents, and control the assistant daemon.
- **WebSocket** (`/ws/v1/`) — receive token-by-token agent output, tool activity events, tool confirmation dialogs, agent state transitions, and live log streams in real time.

Everything that is not time-critical uses REST. Everything real-time uses WebSocket.

---

## Documents in this directory

| Document | Audience | Purpose |
|----------|----------|---------|
| `OVERVIEW.md` (this file) | All | Entry point — start here |
| `CLIENT_CONTRACT.md` | Frontend developers | TypeScript types, endpoint reference, error codes, WebSocket usage examples, pagination, wizard flow |
| `WEBSOCKET_PROTOCOL.md` | Frontend developers | Full WebSocket message catalogue, connection lifecycle diagrams, reconnection strategy |
| `WEBUI_DEVELOPMENT_GUIDE.md` | React frontend developers | Page-by-page integration guide, component hierarchy, state management patterns |
| `ENTERPRISE.md` | Enterprise integrators | Enterprise-tier endpoints (SAML, LDAP, JIT, teams, workspaces, cross-workspace bus) |
| `OPENAPI.yaml` / `OPENAPI.json` | All | Machine-readable OpenAPI 3.x schema — authoritative source for all request/response shapes |

---

## Quick start

### 1. Prerequisites

Generate a JWT secret (required — no default):

```bash
export COGTRIX_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
```

### 2. Apply database migrations

```bash
uv run python -m alembic upgrade head
```

### 3. Start the API server

```bash
# Development (with auto-reload)
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Using the module entrypoint
uv run python -m src.api --host 0.0.0.0 --port 8000

# With debug logging
uv run python -m src.api --debug
```

### 4. Register the first user (admin)

The first user to register automatically receives the `admin` role. This is handled atomically in the database — no manual setup step is needed.

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@example.com","password":"s3cur3P@ss"}'
```

The response includes an access token and a refresh token. Use the access token in the `Authorization: Bearer <token>` header for all subsequent requests.

### 5. Explore the interactive docs

| URL | Tool |
|-----|------|
| `http://localhost:8000/api/v1/docs` | Swagger UI |
| `http://localhost:8000/api/v1/redoc` | ReDoc |
| `http://localhost:8000/api/v1/openapi.json` | OpenAPI 3.x JSON schema |

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COGTRIX_JWT_SECRET` | **yes** | — | JWT signing secret. Minimum 32 characters. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `COGTRIX_DB_URL` | no | `sqlite+aiosqlite:///data/api/cogtrix.db` | Database URL. Use `postgresql+asyncpg://` for PostgreSQL in production. |
| `COGTRIX_CORS_ORIGINS` | no | `http://localhost:5173, http://localhost:3000` | Comma-separated allowed CORS origins. Also settable via the `api.cors_origins` config key or a Helm value (env var wins). The default is localhost-only — set your real web UI origin(s) for any non-local deployment, e.g. `https://cogtrix.ai,https://www.cogtrix.ai`. |
| `COGTRIX_API_HOST` | no | `0.0.0.0` | Bind host. |
| `COGTRIX_API_PORT` | no | `8000` | Bind port. |
| `COGTRIX_API_WORKERS` | no | `1` | Number of uvicorn workers. |
| `COGTRIX_CONFIG_FILE` | no | auto-detected | Path to Cogtrix config file (JSON or YAML). |
| `COGTRIX_DEBUG` | no | — | Set to any non-empty value to enable debug logging. |

---

## Authentication model

The API uses JWT bearer tokens with a two-token model:

- **Access token** — short-lived (1 hour). Send in every REST request as `Authorization: Bearer <token>`. For WebSocket connections, pass in the `Authorization` header — see `WEBSOCKET_PROTOCOL.md` for browser-environment workarounds (the browser ``WebSocket`` API does not support custom headers).
- **Refresh token** — long-lived (30 days). Used only to renew the access token. Never send this on ordinary API calls.

When the access token expires, any API call returns `401` with `error.code === "TOKEN_EXPIRED"`. The frontend should silently call `POST /api/v1/auth/refresh` with the stored refresh token, receive a new token pair, and retry the original request — without prompting the user to log in.

For the complete token refresh flow and implementation pattern, see `CLIENT_CONTRACT.md` Section 2.

---

## Response envelope

Every REST response — success or error — uses the same JSON envelope:

```json
{
  "data": { ... } | null,
  "error": { "code": "...", "message": "...", "details": null } | null,
  "meta": { "request_id": "...", "timestamp": "2026-03-27T12:00:00Z" }
}
```

On success, `data` is populated and `error` is `null`. On failure, `data` is `null` and `error` is populated. `meta` is always present and includes a `request_id` UUID for support traces.

Paginated list responses use an additional `CursorPage<T>` wrapper inside `data`:

```json
{
  "data": {
    "items": [...],
    "next_cursor": "<opaque>" | null,
    "has_more": true | false,
    "total": 42 | null
  }
}
```

Pass `next_cursor` as the `?cursor=` query parameter to fetch the next page. Treat cursors as opaque strings — never parse them.

---

## Resource groups at a glance

| Group | Base path | Purpose |
|-------|-----------|---------|
| Health | `/api/v1/health` | Liveness and readiness probes (no auth) |
| Auth | `/api/v1/auth` | Register, login, token refresh, API key management |
| Sessions | `/api/v1/sessions` | Create and manage conversation sessions |
| Messages | `/api/v1/sessions/{id}/messages` | Send messages and retrieve history |
| Memory | `/api/v1/sessions/{id}/memory` | Inspect and control the session memory subsystem |
| Tools | `/api/v1/tools`, `/api/v1/sessions/{id}/tools` | Browse the tool catalog; load/enable/disable per session |
| Configuration | `/api/v1/config` | Read and update global settings, providers, and models |
| MCP Servers | `/api/v1/mcp/servers` | Manage Model Context Protocol server connections |
| RAG / Documents | `/api/v1/rag` | Upload documents for retrieval-augmented generation |
| Assistant Mode | `/api/v1/assistant` | Control the headless WhatsApp/Telegram daemon |
| Workflows | `/api/v1/assistant/workflows` | Manage workflow definitions and chat bindings |
| Users | `/api/v1/users` | Admin-only user management |
| System | `/api/v1/system` | System info and debug controls |
| Agents | `/api/v1/agents` | Named agent configurations |
| Tasks | `/api/v1/tasks` | Background task queue |
| Teams | `/api/v1/teams` | Team management (org-scoped, admin) |
| Workspaces | `/api/v1/workspaces` | Workspace management (org-scoped, admin) |
| Cross-Workspace | `/api/v1/cross-workspace` | Inter-workspace messaging |
| Plans | `/api/v1/plans` | Subscription plan definitions |
| Usage | `/api/v1/usage` | Usage metering and records |
| Enforcement | `/api/v1/enforcement` | Plan limit enforcement status |
| Billing | `/api/v1/billing` | Stripe checkout and subscriptions |
| SAML | `/api/v1/saml` | SAML 2.0 SSO |
| LDAP | `/api/v1/ldap` | LDAP/AD sync |
| JIT | `/api/v1/jit` | Just-in-time user provisioning |
| SCIM | `/scim/v2/` | SCIM 2.0 user provisioning |
| WebSocket — Session | `/ws/v1/sessions/{id}` | Real-time agent output and event streaming |
| WebSocket — Logs | `/ws/v1/logs` | Live log streaming (admin only) |

---

## Session lifecycle

```
POST /api/v1/sessions              → creates session record (idle state)
                                     ↓
First request to the session       → session is warmed: memory loaded,
                                     LLM created, AgentRunConfig built
                                     ↓
POST /api/v1/sessions/{id}/messages
  or WebSocket user_message        → agent turn runs; streaming over WebSocket
                                     ↓
                          [30 minutes idle]
                                     ↓
Background eviction                → memory saved; session removed from RAM
  (state preserved in DB)
                                     ↓
DELETE /api/v1/sessions/{id}       → soft-delete (archived_at set)
  ?permanent=true                  → hard-delete (non-recoverable)
POST /api/v1/sessions/{id}/restore → clear archived_at (restore soft-deleted session)
```

---

## Agent execution modes

When sending a message, the `mode` field selects the reasoning pipeline:

| Mode | Description | WebSocket state sequence |
|------|-------------|--------------------------|
| `normal` | Standard agent run | `idle → thinking → done` |
| `think` | Forces the deep-think pipeline (extended multi-branch reasoning, optional research delegate) | `idle → thinking → analyzing → researching → deep_thinking → done` |
| `delegate` | Forces task delegation to parallel sub-agents | `idle → thinking → delegating → done` |

---

## Docker usage

The `docker-entrypoint.sh` in the `docker/` directory intercepts `api` or `--api` as the first argument, runs `alembic upgrade head`, and starts uvicorn:

```bash
docker run -e COGTRIX_JWT_SECRET=<secret> \
  -p 8000:8000 \
  cogtrix api
```

The `docker/Dockerfile` healthcheck probes `GET /api/v1/health` and exits immediately in CLI mode (no sentinel file) so orchestrators do not mark CLI containers as unhealthy.

---

## Where to go next

- Building the React frontend? Start with `CLIENT_CONTRACT.md` for TypeScript types, then `WEBUI_DEVELOPMENT_GUIDE.md` for the full page-by-page integration guide.
- Integrating with the WebSocket in detail? Read `WEBSOCKET_PROTOCOL.md`.
- Integrating with enterprise features (SAML, SCIM, billing, workspaces, teams)? Read `ENTERPRISE.md`.
- Exploring interactively? Open `http://localhost:8000/api/v1/docs` (Swagger UI) with the server running.
