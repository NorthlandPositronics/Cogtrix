# Migration Guide: v0.x → v1.0

This guide covers every breaking change, new requirement, and recommended
action for operators upgrading a Cogtrix deployment from any v0.x release to
v1.0.

---

## 1. Overview

Cogtrix v1.0 is the first production-stability release. It ships:

- PostgreSQL support as the production-grade database backend (M5.1)
- Redis-backed session presence for horizontal scaling (M5.2)
- OIDC/SSO token validation alongside local JWT auth (M5.3)
- Structured NDJSON audit log (M5.4)
- Per-user resource quotas (M5.5)
- Multi-arch Docker image (amd64 + arm64) with signed attestations (M7.5)
- A pluggable tool architecture for external tool authors (M2.8)

**Breaking changes summary:**

| Area | What changed |
|------|-------------|
| Config keys | `provider`/`model` top-level keys deprecated; use `models.default` |
| Environment vars | `COGTRIX_PROVIDER` removed; use `COGTRIX_MODEL` |
| CLI flag | `--provider` / `-p` removed; use `--model` / `-m` |
| API endpoint | `POST /config/provider` returns 410 Gone; use `POST /config/providers` |
| New required secret | `COGTRIX_JWT_SECRET` (≥ 32 chars) mandatory in API mode |

All other changes are additive. Existing v0.x config files continue to work
with automatic migration of deprecated keys.

---

## 2. Database Changes (M5.1)

### Default database

The API layer defaults to **SQLite** (`./data/api/cogtrix.db`) for development
and single-instance deployments. For production, PostgreSQL is the recommended
backend.

### Switching to PostgreSQL

Set the `COGTRIX_DB_URL` environment variable to a `postgresql+asyncpg://` URL:

```bash
export COGTRIX_DB_URL="postgresql+asyncpg://cogtrix:secret@db-host:5432/cogtrix"
```

Install the async PostgreSQL driver:

```bash
pip install "cogtrix[postgresql]"
# or with uv:
uv sync --extra postgresql
```

The application validates the connection at startup and raises a clear error
if it cannot connect. Check:

1. The database server is running and reachable.
2. `COGTRIX_DB_URL` is correct (password, host, port, database name).
3. The database user has `CONNECT` and `CREATE TABLE` privileges.

### Keeping SQLite for development

Omit `COGTRIX_DB_URL` or set it explicitly:

```bash
export COGTRIX_DB_URL="sqlite+aiosqlite:///./data/api/cogtrix.db"
```

You can also relocate the SQLite file via `COGTRIX_DATA_DIR`:

```bash
export COGTRIX_DATA_DIR="/var/cogtrix"
# DB will be at /var/cogtrix/api/cogtrix.db
```

### Schema migration

Run Alembic migrations before starting the server after an upgrade:

```bash
alembic upgrade head
```

The Docker entrypoint runs `alembic upgrade head` automatically when started
in API mode (`docker run ... api`).

---

## 3. Configuration Changes

### Deprecated top-level keys

The old flat `provider`/`model` format is deprecated. Keys are auto-migrated
at load time (no immediate action required), but you should update your config
before v1.0 goes final to avoid warnings.

**Before (v0.x):**

```yaml
provider: ollama
model: qwen3:8b
temperature: 0.6
```

**After (v1.0):**

```yaml
providers:
  local:
    type: ollama
    base_url: http://localhost:11434

models:
  default: local/qwen3:8b
  local-fast:
    provider: local
    model: qwen3:8b
    temperature: 0.6
```

### Removed environment variable

`COGTRIX_PROVIDER` is no longer recognized. Use `COGTRIX_MODEL` instead:

```bash
# Before
export COGTRIX_PROVIDER=openai

# After — specify a model alias or provider/model shorthand
export COGTRIX_MODEL=openai/gpt-4.1-mini
```

### Removed CLI flag

`--provider` / `-p` is removed. Use `--model` / `-m` with a model alias:

```bash
# Before
cogtrix.py --provider openai

# After
cogtrix.py --model openai/gpt-4.1-mini
```

### New configuration sections (M5.x)

Add these sections to `.cogtrix.yaml` as needed. All are optional with safe
defaults.

```yaml
# Redis session presence (M5.2) — optional, enables horizontal scaling
redis_url: "redis://localhost:6379/0"

# OIDC/SSO (M5.3) — optional
oidc:
  enabled: false
  issuer: "https://your-idp.example.com/realms/cogtrix"
  audience: "cogtrix-api"

# Audit log (M5.4) — enabled by default
audit_log:
  enabled: true
  path: "data/audit/audit.log"

# Per-user quotas (M5.5) — all unlimited by default
quotas:
  token_budget_per_day: null     # e.g. 500000
  requests_per_hour: null        # e.g. 60
  max_concurrent_sessions: null  # e.g. 5
```

### New required secret in API mode

`COGTRIX_JWT_SECRET` must be set to a random string of at least 32 characters.
The server refuses to start without it:

```bash
export COGTRIX_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

### Full .env example

```bash
# .env — v1.0 reference

# Required in API mode
COGTRIX_JWT_SECRET=change-me-to-a-random-32-char-string-here

# Database (defaults to SQLite if not set)
COGTRIX_DB_URL=postgresql+asyncpg://cogtrix:secret@localhost:5432/cogtrix

# Data directory (relocates SQLite file and audit log)
COGTRIX_DATA_DIR=/var/cogtrix

# Redis (optional; enables horizontal scaling)
# COGTRIX_REDIS_URL=redis://localhost:6379/0

# Model selection
COGTRIX_MODEL=openai/gpt-4.1-mini

# API host / port
COGTRIX_API_HOST=0.0.0.0
COGTRIX_API_PORT=8000
COGTRIX_API_WORKERS=4
```

---

## 4. Authentication Changes (M5.3)

### OIDC/SSO integration

v1.0 adds support for identity providers that issue RS256 or ES256 JWT tokens
(Keycloak, Auth0, Okta, Azure AD, Google Workspace, etc.).

When OIDC is enabled, the API accepts both:
- Local JWT tokens issued by Cogtrix's own `POST /auth/login` endpoint.
- OIDC ID tokens issued by the configured identity provider.

Enable OIDC in `.cogtrix.yaml`:

```yaml
oidc:
  enabled: true
  issuer: "https://your-idp.example.com/realms/cogtrix"
  audience: "cogtrix-api"
  # Optional: override JWKS URI (auto-discovered from issuer by default)
  # jwks_uri: "https://your-idp.example.com/realms/cogtrix/protocol/openid-connect/certs"
  # Optional: claim that holds role information (default: "roles")
  role_claim: "roles"
  # Optional: role assigned when claim is absent (default: "user")
  default_role: "user"
```

### Role mapping

The `role_claim` in the OIDC token is mapped to Cogtrix roles:
- If the claim contains `"admin"`, the user receives admin access.
- Otherwise, the first role in the list (or `default_role`) is used.

### Fallback behavior

If OIDC is not configured (`oidc.enabled: false`, the default), the API
continues to use local JWT authentication exclusively. No action is required
for deployments that do not use SSO.

---

## 5. Session Storage Changes (M5.2)

### Redis session presence

v1.0 introduces an optional Redis store that tracks session last-activity
timestamps. This enables correct idle eviction across multiple API instances
(horizontal scaling) without sharing in-memory state.

**What is stored in Redis:** only the last-activity timestamp per session ID.
Live session objects (LLM client, memory manager, asyncio queue) remain in
process memory — they cannot be serialized.

### Enabling Redis

Set `redis_url` in `.cogtrix.yaml`:

```yaml
redis_url: "redis://localhost:6379/0"
# TLS: redis_url: "rediss://redis.example.com:6380/0"
```

Install the async Redis client:

```bash
pip install "cogtrix[redis]"
# or with uv:
uv sync --extra redis
```

### Behavior without Redis

If `redis_url` is empty (the default) or the Redis server is unreachable at
startup, Cogtrix falls back to in-process session tracking with a warning log.
All single-instance deployments continue to work without any Redis configuration.
Redis connectivity errors during operation are caught and logged at DEBUG level
— they do not cause request failures.

---

## 6. Audit Log (M5.4)

### Overview

v1.0 writes a structured audit trail for tool calls, user actions,
configuration changes, and authentication events. The log is stored as
**NDJSON** (newline-delimited JSON) — one event per line — so it can be
tailed, grepped, and parsed by log aggregators (Splunk, Elastic, Loki, etc.).

### Default location

```
data/audit/audit.log
```

Each line is a JSON object with these fields:

```json
{
  "event_id": "3f2504e0-...",
  "timestamp": "2026-03-29T12:00:00Z",
  "category": "tool_call",
  "action": "write_file",
  "actor": "user:alice",
  "status": "ok",
  "detail": { "path": "report.md" },
  "duration_ms": 42
}
```

Categories: `tool_call`, `user_action`, `config_change`, `auth`, `system`.

### Configuration

```yaml
audit_log:
  enabled: true               # set to false to disable
  path: "data/audit/audit.log"   # absolute or relative to CWD
```

Override the path via `COGTRIX_DATA_DIR` — the audit log is placed under
`<COGTRIX_DATA_DIR>/audit/audit.log` when the env var is set.

### Disabling the audit log

```yaml
audit_log:
  enabled: false
```

When disabled, no file is opened and no disk writes occur. This setting is
suitable for resource-constrained environments or local development where
audit trails are not required.

---

## 7. Resource Quotas (M5.5)

### Overview

v1.0 adds per-user quota enforcement. Three independent limits are available:

| Quota | Config key | Default |
|-------|-----------|---------|
| Daily token budget | `quotas.token_budget_per_day` | unlimited |
| Requests per hour | `quotas.requests_per_hour` | unlimited |
| Concurrent sessions | `quotas.max_concurrent_sessions` | unlimited |

All limits default to `null` (unlimited). No action is required if you do not
need usage caps.

### Configuring limits

```yaml
quotas:
  token_budget_per_day: 500000     # tokens consumed per user per calendar day
  requests_per_hour: 60            # API requests per rolling hour
  max_concurrent_sessions: 5       # live warmed sessions per user
```

### Behavior when a limit is exceeded

The API returns `HTTP 429 Too Many Requests` with a JSON error body:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Request rate limit exceeded: 60/60 requests in the last hour."
  }
}
```

Codes: `RATE_LIMIT_EXCEEDED`, `TOKEN_BUDGET_EXCEEDED`, `SESSION_LIMIT_EXCEEDED`.

### Raising or disabling limits

Set the value to a higher integer or back to `null` to remove the cap:

```yaml
quotas:
  token_budget_per_day: null    # no daily token cap
  requests_per_hour: 200        # increase hourly request cap
```

---

## 8. Plugin and Tool Format Changes (M2.8)

### New pluggable tool architecture

v1.0 introduces a formal plugin contract. All external tool modules must
expose `TOOL_CONFIGS` and `TOOL_SETUP`.

**Required interface:**

```python
# ~/.cogtrix/tools/my_tool.py

TOOL_CONFIGS = [
    {
        "name": "my_action",
        "description": "Does something useful.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text."}
            },
            "required": ["input"],
        },
    }
]


def TOOL_SETUP(config) -> None:
    """Called once after the tool is loaded. Configure API keys, clients, etc."""
    api_key = config.services.get("my_service", {}).get("api_key", "")
    _configure(api_key)


def is_configured() -> bool:
    """Optional. Return False to hide this tool when not configured."""
    return bool(_api_key)


def my_action(input: str) -> str:
    return f"result: {input}"
```

### Drop-in directory

Place `.py` files in `~/.cogtrix/tools/` (or any path in `config.tool_dirs`).
They are loaded automatically at startup with no installation step.

```yaml
# .cogtrix.yaml
tool_dirs:
  - "~/.cogtrix/tools"
  - "/opt/company/cogtrix-tools"
```

### Installable packages

Publish a pip package that declares the `cogtrix.tools` entry point:

```toml
# pyproject.toml
[project.entry-points."cogtrix.tools"]
my_tools = "my_package.tools_module"
```

Cogtrix discovers it via `importlib.metadata.entry_points`.

### Updating existing custom plugins

If you have custom tool modules from v0.x, add the following to each:

1. Rename your tool registration dict(s) to a list called `TOOL_CONFIGS`.
2. Move any `configure_*` startup logic into a `TOOL_SETUP(config)` function.
3. Add an `is_configured()` function if the tool requires credentials.

Built-in tools in `src/tools/` are unchanged — this migration only applies to
external/custom plugins.

---

## 9. Docker Image (M7.5)

### Multi-arch image

v1.0 ships a multi-architecture image supporting **amd64** and **arm64**
(Apple Silicon, Raspberry Pi, ARM cloud instances). The image is built with
`docker buildx bake` and signed with SLSA attestations.

Pull the new image:

```bash
docker pull ghcr.io/northlandpositronics/cogtrix:v0.2.0
# or use the rolling tag:
docker pull ghcr.io/northlandpositronics/cogtrix:latest
```

### Running in API mode

```bash
docker run -d \
  --name cogtrix-api \
  -p 8000:8000 \
  -e COGTRIX_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" \
  -e COGTRIX_DB_URL="postgresql+asyncpg://cogtrix:secret@db:5432/cogtrix" \
  -v cogtrix-data:/data \
  ghcr.io/northlandpositronics/cogtrix:v0.2.0 api
```

The entrypoint automatically runs `alembic upgrade head` before starting
uvicorn when the first argument is `api` or `--api`.

### docker-compose example

```yaml
version: "3.9"
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: cogtrix
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: cogtrix
    volumes:
      - pg-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  api:
    image: ghcr.io/northlandpositronics/cogtrix:v0.2.0
    command: api
    ports:
      - "8000:8000"
    environment:
      COGTRIX_JWT_SECRET: "change-me-to-a-random-32-char-string-here"
      COGTRIX_DB_URL: "postgresql+asyncpg://cogtrix:secret@db:5432/cogtrix"
    volumes:
      - cogtrix-data:/data
      - ./cogtrix.yaml:/app/.cogtrix.yaml:ro
    depends_on:
      db:
        condition: service_healthy

volumes:
  pg-data:
  cogtrix-data:
```

The image's healthcheck probes `GET /api/v1/health` and reports healthy once
the API is accepting requests. CLI-mode containers exit the healthcheck
immediately with code 0 so they are never marked unhealthy.

---

## 10. Step-by-Step Upgrade Checklist

Follow these steps in order when upgrading a running deployment.

1. **Back up your database.**
   - SQLite: copy `data/api/cogtrix.db` to a safe location.
   - PostgreSQL: `pg_dump cogtrix > cogtrix_backup_$(date +%Y%m%d).sql`

2. **Review configuration changes.**
   - Replace deprecated `provider`/`model` top-level keys with the
     `providers:` / `models:` sections (see Section 3).
   - Remove any reference to `COGTRIX_PROVIDER`; use `COGTRIX_MODEL`.
   - Add `COGTRIX_JWT_SECRET` (required in API mode).
   - Add `redis_url`, `oidc`, `audit_log`, and `quotas` sections as needed.

3. **Pull the new Docker image or reinstall the package.**
   ```bash
   # Docker
   docker pull ghcr.io/northlandpositronics/cogtrix:v0.2.0

   # Python package
   pip install --upgrade "cogtrix[postgresql,redis]"
   # or:
   uv sync
   ```

4. **Run database migrations.**
   ```bash
   # Standalone
   alembic upgrade head

   # Docker (migration runs automatically on start in API mode)
   docker run --rm -e COGTRIX_DB_URL=... ghcr.io/northlandpositronics/cogtrix:v0.2.0 \
     alembic upgrade head
   ```

5. **Update custom tool plugins** (if any) to use `TOOL_CONFIGS` and
   `TOOL_SETUP(config)` as described in Section 8.

6. **Start the new version** and verify with the health check endpoint:
   ```bash
   curl http://localhost:8000/api/v1/health
   # Expected: {"data": {"status": "ok", ...}, "error": null}
   ```

7. **Verify authentication** by logging in via `POST /api/v1/auth/login` and
   confirming that access tokens work on a protected endpoint.

8. **Check the audit log** to confirm events are being written:
   ```bash
   tail -f data/audit/audit.log | python3 -m json.tool
   ```

9. **Update any API clients** that called `POST /config/provider` — this
   endpoint now returns `410 Gone`. Use `POST /config/providers` instead.

---

*For questions or issues, open a GitHub issue at
https://github.com/NorthlandPositronics/Cogtrix/issues*
