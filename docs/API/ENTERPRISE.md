# Enterprise API Reference

Audience: Enterprise integrators and admin frontend developers
API version: v1
Last updated: 2026-05-01

Related documents:
- `docs/API/OVERVIEW.md` — API orientation, quick start, authentication model
- `docs/API/CLIENT_CONTRACT.md` — TypeScript types and REST/WebSocket usage patterns
- `docs/API/WEBSOCKET_PROTOCOL.md` — full WebSocket message catalogue

---

## Table of Contents

1. [Authentication & Authorization](#1-authentication--authorization)
2. [Common Patterns](#2-common-patterns)
3. [Agents](#3-agents)
4. [Tasks](#4-tasks)
5. [SAML](#5-saml)
6. [LDAP](#6-ldap)
7. [Teams](#7-teams)
8. [JIT Provisioning](#8-jit-provisioning)
9. [Workspaces](#9-workspaces)
10. [Cross-Workspace](#10-cross-workspace)
11. [Plans](#11-plans)
12. [Usage Metering](#12-usage-metering)
13. [Enforcement](#13-enforcement)
14. [Billing](#14-billing)
15. [SCIM 2.0](#15-scim-20)
16. [Users](#16-users)

---

## 1. Authentication & Authorization

Enterprise endpoints use the same JWT bearer tokens as the core API.

### 1.1 Bearer Token

Include the access token in every REST request:

```
Authorization: Bearer <jwt>
Content-Type: application/json
```

### 1.2 Role Requirements

| Requirement | Description |
|-------------|-------------|
| `bearer` | Any authenticated user (role `user` or `admin`). |
| `admin` | Admin role required. Some admin endpoints also require the caller to belong to the target organization (org-scoped). |
| `none` | No authentication required. |

### 1.3 Org Context

Most enterprise endpoints are **org-scoped**: admin callers can only manage resources within their own organization. The org is resolved automatically from the JWT token. Endpoints that are org-scoped will return `FORBIDDEN` or `NOT_FOUND` when the resource does not belong to the caller's org.

### 1.4 SCIM Authentication

SCIM endpoints use the same JWT bearer token as the rest of the API, but require an `admin` role. The expected `Content-Type` is `application/scim+json`.

---

## 2. Common Patterns

### 2.1 APIResponse Envelope

All REST responses (except SCIM and SAML metadata) use the standard envelope:

```json
{
  "data": { ... },
  "error": null,
  "meta": {
    "request_id": "...",
    "timestamp": "2026-05-01T12:34:56.789Z"
  }
}
```

On error, `data` is `null` and `error` contains `code`, `message`, and optional `details`.

### 2.2 Cursor Pagination

List endpoints that support pagination use cursor-based semantics (not offset). Responses return a `CursorPage[T]`:

| Field | Type | Description |
|-------|------|-------------|
| `items` | `T[]` | Items on the current page. |
| `next_cursor` | `string \| null` | Opaque cursor to pass as `?cursor=` for the next page. |
| `has_more` | `boolean` | `true` when additional pages exist. |
| `total` | `number \| null` | Total item count (may be `null` when expensive). |

**Query parameters:**
- `cursor` — opaque cursor from the previous page's `next_cursor`.
- `limit` — page size (varies per endpoint; see individual endpoint tables).

### 2.3 Error Codes

Enterprise endpoints share the canonical error codes defined in `src/api/schemas/common.py`. The most common ones are:

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `UNAUTHORIZED` | 401 | Missing or invalid bearer token. |
| `TOKEN_EXPIRED` | 401 | JWT expired; refresh and retry. |
| `FORBIDDEN` | 403 | Authenticated user lacks permission. |
| `NOT_FOUND` | 404 | Resource does not exist or is outside the caller's org. |
| `CONFLICT` | 409 | Resource already exists (duplicate name/slug/email). |
| `VALIDATION_ERROR` | 400/422 | Request body or query parameter validation failed. |
| `INTERNAL_ERROR` | 500 | Unexpected server error. |
| `NOT_IMPLEMENTED` | 501 | Feature not yet available. |
| `SERVICE_UNAVAILABLE` | 503 | Required subsystem is not configured or not installed. |

Additional endpoint-specific codes are documented per route group below.

---

## 3. Agents

**Base path:** `/api/v1/agents`

Named agent configuration endpoints. Returns agents loaded from the config file and `AGENTS.md`.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `bearer` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/agents` | 200 | List all registered agents. |
| GET | `/api/v1/agents/{name}` | 200 | Get a single agent by name. |

**Schemas:**

```python
class AgentOut(BaseModel):
    name: str
    description: str
    system_prompt: str
    tools_include: list[str]
    tools_exclude: list[str]
    model_alias: str
    memory_mode: str
    max_steps: int
    temperature: float
```

**Error codes:** `AGENT_NOT_FOUND` (404).

---

## 4. Tasks

**Base path:** `/api/v1/tasks`

Background task queue endpoints. Tasks are submitted to named agents and processed asynchronously.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `bearer` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| POST | `/api/v1/tasks` | 202 | Submit a new background task. |
| GET | `/api/v1/tasks` | 200 | List tasks (`?status=`, `?limit=` 1–200 default 50). |
| GET | `/api/v1/tasks/{task_id}` | 200 | Get a single task by ID. |
| DELETE | `/api/v1/tasks/{task_id}` | 200 | Cancel a pending task. |
| GET | `/api/v1/tasks/{task_id}/log` | 200 | Stream the raw text log for a task. |

**Schemas:**

```python
class TaskCreateRequest(BaseModel):
    agent_name: str   # 1–128 chars
    prompt: str       # 1–8192 chars

class TaskOut(BaseModel):
    task_id: str
    agent_name: str
    prompt: str
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    result: str
    error: str
    log_path: str
    user_id: str = ""
    org_id: str | None = None
```

**Error codes:**
- `TASK_QUEUE_UNAVAILABLE` (503) — task queue module not loaded or not initialised.
- `TASK_NOT_FOUND` (404) — task ID does not exist.
- `TASK_ACCESS_DENIED` (403) — task belongs to another user.
- `TASK_NOT_CANCELLABLE` (409) — task is not in `PENDING` state.
- `INVALID_STATUS` (400) — unknown status filter value.

---

## 5. SAML

**Base path:** `/api/v1/saml`

SAML 2.0 Service Provider routes. Requires the `[saml]` optional extra (`pip install cogtrix[saml]`).

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| GET /metadata | `none` |
| GET /sso | `none` |
| POST /acs | `none` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/saml/metadata` | 200 | Return SP metadata XML. |
| GET | `/api/v1/saml/sso` | 302 | Build SAMLRequest and redirect to IdP. |
| POST | `/api/v1/saml/acs` | 200 | Assertion Consumer Service; returns `{"access_token": "...", "token_type": "bearer"}`. |

**Request parameters:**
- `POST /acs` accepts `application/x-www-form-urlencoded`:
  - `SAMLResponse` (required) — Base64-encoded SAMLResponse from the IdP.
  - `RelayState` (optional) — echoed back by the IdP.

**Error codes:**
- `SAML_NOT_CONFIGURED` (503) — SAML config is missing.
- `SAML_NOT_INSTALLED` (503) — `python3-saml` is not installed.
- `SAML_METADATA_ERROR` (500) — metadata generation failed.
- `SAML_INVALID_RESPONSE` (401) — response validation failed.
- `USER_ACCOUNT_CONFLICT` (422) — cross-org email collision (opaque).

---

## 6. LDAP

**Base path:** `/api/v1/ldap`

LDAP/Active Directory sync routes. Requires the `[ldap]` optional extra (`pip install cogtrix[ldap]`).

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `admin` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/ldap/status` | 200 | Check LDAP configuration and library availability. |
| POST | `/api/v1/ldap/sync` | 200 | Trigger a full user sync run. |

**Response schemas:**

`GET /status` returns:
```json
{
  "configured": true,
  "ldap3_installed": true,
  "server_url": "ldaps://ldap.example.com:636",
  "search_base": "ou=users,dc=example,dc=com"
}
```

`POST /sync` returns:
```json
{
  "added": 0,
  "updated": 0,
  "skipped": 0,
  "total_processed": 0,
  "errors": 0,
  "success": true
}
```

**Error codes:**
- `LDAP_NOT_CONFIGURED` (503) — LDAP sync is not configured.
- `LDAP_NOT_INSTALLED` (503) — `ldap3` is not installed.

---

## 7. Teams

**Base path:** `/api/v1/teams`

Team management routes. Org-scoped and admin-only.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `admin` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/teams` | 200 | List teams in the caller's org. |
| POST | `/api/v1/teams` | 201 | Create a team. |
| GET | `/api/v1/teams/{id}` | 200 | Get a team. |
| PATCH | `/api/v1/teams/{id}` | 200 | Update a team. |
| DELETE | `/api/v1/teams/{id}` | 200 | Delete a team. |
| GET | `/api/v1/teams/{id}/members` | 200 | List team members. |
| POST | `/api/v1/teams/{id}/members` | 201 | Add a member. |
| DELETE | `/api/v1/teams/{id}/members/{uid}` | 200 | Remove a member. |

**Schemas:**

```python
class TeamOut(BaseModel):
    id: str
    org_id: str
    name: str
    description: str | None
    member_count: int
    created_at: datetime

class TeamCreate(BaseModel):
    name: str       # 1–128 chars
    description: str | None   # max 512 chars

class TeamUpdate(BaseModel):
    name: str | None
    description: str | None

class MemberOut(BaseModel):
    user_id: str
    username: str
    email: str
    role: str       # "member" or "admin"
    joined_at: datetime

class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "member"   # "member" or "admin"
```

**Error codes:**
- `CONFLICT` (409) — team name already exists, or user is already a member.
- `NOT_FOUND` (404) — team or user not found in this org.

---

## 8. JIT Provisioning

**Base path:** `/api/v1/jit`

Just-in-time user provisioning admin routes.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `admin` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/jit/status` | 200 | JIT configuration summary. |
| POST | `/api/v1/jit/test` | 200 | Dry-run: check whether an email would be allowed. |

**Response schemas:**

`GET /status` returns:
```json
{
  "configured": true,
  "enabled": true,
  "allowed_domains": ["example.com"],
  "default_role": "user",
  "org_id": "...",
  "auto_team_id": "...",
  "max_users": 100,
  "deactivate_unknown": false
}
```
When not configured, returns `{"enabled": false, "configured": false}`.

`POST /test` accepts `{"email": "..."}` and returns:
```json
{
  "allowed": true,
  "email": "user@example.com",
  "reason": "domain allowed"
}
```

---

## 9. Workspaces

**Base path:** `/api/v1/workspaces`

Workspace management routes. Admin-only and org-scoped. Includes typed workspace config endpoints.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `admin` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/workspaces` | 200 | List workspaces in the caller's org. |
| POST | `/api/v1/workspaces` | 201 | Create a workspace. |
| GET | `/api/v1/workspaces/{id}` | 200 | Get a workspace. |
| PATCH | `/api/v1/workspaces/{id}` | 200 | Update a workspace. |
| DELETE | `/api/v1/workspaces/{id}` | 200 | Delete a workspace. |
| GET | `/api/v1/workspaces/{id}/members` | 200 | List workspace members. |
| POST | `/api/v1/workspaces/{id}/members` | 201 | Add a member. |
| DELETE | `/api/v1/workspaces/{id}/members/{uid}` | 200 | Remove a member. |
| GET | `/api/v1/workspaces/{id}/config` | 200 | Read typed workspace config. |
| PATCH | `/api/v1/workspaces/{id}/config` | 200 | Update workspace config (partial). |

**Schemas:**

```python
class WorkspaceOut(BaseModel):
    id: str
    org_id: str
    name: str
    description: str | None
    settings: dict[str, Any] | None
    member_count: int
    is_active: bool
    created_at: datetime

class WorkspaceCreate(BaseModel):
    name: str       # 1–128 chars
    description: str | None   # max 512 chars
    settings: dict[str, Any] | None

class WorkspaceUpdate(BaseModel):
    name: str | None
    description: str | None
    settings: dict[str, Any] | None
    is_active: bool | None

class WorkspaceMemberOut(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    joined_at: datetime

class AddWorkspaceMemberRequest(BaseModel):
    user_id: str
    role: str = "member"   # "member" or "admin"
```

**Workspace config fields** (`GET /config` and `PATCH /config`):

| Field | Type | Description |
|-------|------|-------------|
| `model_override` | `string \| null` | LLM model alias for sessions in this workspace. |
| `system_prompt` | `string \| null` | System prompt prepended to all sessions. |
| `tool_policy` | `string \| null` | `"all"`, `"none"`, or comma-separated tool names. |
| `max_context_tokens` | `integer \| null` | Context window override. |
| `rate_limit_multiplier` | `number` | Multiply the org-level rate limit by this factor (default `1.0`). |

`PATCH /config` merges keys into the existing config. Keys set to `null` are removed.

**Error codes:**
- `CONFLICT` (409) — workspace name already exists or user already a member.
- `NOT_FOUND` (404) — workspace or user not found in this org.

---

## 10. Cross-Workspace

**Base path:** `/api/v1/cross-workspace`

Cross-workspace agent communication routes. Messages are sent between workspaces within the same organization.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| POST /messages | `bearer` + org context |
| GET /inbox/{ws_id} | `bearer` + org context |
| DELETE /inbox/{ws_id}/{msg_id} | `bearer` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| POST | `/api/v1/cross-workspace/messages` | 201 | Send a message from one workspace to another. |
| GET | `/api/v1/cross-workspace/inbox/{ws_id}` | 200 | Read inbox for a workspace (`?limit=` 1–200 default 50). |
| DELETE | `/api/v1/cross-workspace/inbox/{ws_id}/{msg_id}` | 200 | Delete a message from the inbox. |

**Request schema:**

```python
class SendMessageRequest(BaseModel):
    from_workspace_id: str
    to_workspace_id: str
    subject: str      # 1–128 chars
    body: dict        # default {}
```

**Error codes:**
- `CROSS_WS_DISABLED` (503) — cross-workspace communication is disabled.
- `CROSS_ORG_BLOCKED` (403) — source and destination are in different orgs, or caller does not belong to the source workspace.
- `POLICY_DENIED` (403) — the workspace pair is blocked by policy.
- `NOT_FOUND` (404) — workspace or message not found.

---

## 11. Plans

**Base path:** `/api/v1/plans`

Plan (subscription tier) management. Public listing is available to any authenticated user; mutations require admin.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| GET /plans | `bearer` |
| GET /plans/{id} | `bearer` |
| POST /plans | `admin` |
| PATCH /plans/{id} | `admin` |
| DELETE /plans/{id} | `admin` |
| PATCH /organizations/{id}/plan | `admin` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/plans` | 200 | List public plans. |
| GET | `/api/v1/plans/{id}` | 200 | Get a plan. |
| POST | `/api/v1/plans` | 201 | Create a plan. |
| PATCH | `/api/v1/plans/{id}` | 200 | Update a plan. |
| DELETE | `/api/v1/plans/{id}` | 200 | Deactivate (soft-delete) a plan. |
| PATCH | `/api/v1/organizations/{id}/plan` | 200 | Assign a plan to an organization. Body: `{"plan_id": "..."}`. |

**Schemas:**

```python
class PlanLimits(BaseModel):
    max_users: int = 0                 # 0 = unlimited
    max_workspaces: int = 0
    max_api_calls_per_month: int = 0
    max_storage_gb: int = 0

class PlanOut(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    price_monthly_cents: int
    price_annual_cents: int
    limits: PlanLimits
    is_active: bool
    is_public: bool
    created_at: datetime

class PlanCreate(BaseModel):
    name: str            # 1–64 chars
    slug: str            # 1–32 chars, pattern: ^[a-z0-9]+(?:-[a-z0-9]+)*$
    description: str | None
    price_monthly_cents: int = 0   # ge=0
    price_annual_cents: int = 0    # ge=0
    limits: PlanLimits = Field(default_factory=PlanLimits)
    is_public: bool = True

class PlanUpdate(BaseModel):
    name: str | None
    description: str | None
    price_monthly_cents: int | None   # ge=0
    price_annual_cents: int | None    # ge=0
    limits: PlanLimits | None
    is_active: bool | None
    is_public: bool | None
```

**Error codes:**
- `CONFLICT` (409) — plan slug already exists.
- `NOT_FOUND` (404) — plan or organization not found.
- `VALIDATION_ERROR` (422) — missing `plan_id` in org assignment body.

---

## 12. Usage Metering

**Base path:** `/api/v1/usage`

Usage metering and reporting for the caller's organization.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| GET /summary | `bearer` + org context |
| GET /records | `admin` + org context |
| POST /record | `admin` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/usage/summary` | 200 | Monthly usage summary (`?year=`, `?month=`). |
| GET | `/api/v1/usage/records` | 200 | Recent raw usage records (`?event_type=`, `?limit=` 1–500 default 50). |
| POST | `/api/v1/usage/record` | 201 | Manually record a usage event. |

**Response schemas:**

`GET /summary` returns:
```json
{
  "org_id": "...",
  "period": "2026-05",
  "totals": { ... }
}
```

`GET /records` returns a list of:
```json
{
  "id": "...",
  "event_type": "api_call",
  "quantity": 1,
  "workspace_id": "...",
  "user_id": "...",
  "period": "2026-05",
  "recorded_at": "2026-05-01T12:34:56.789Z"
}
```

`POST /record` accepts:
```json
{
  "event_type": "api_call",
  "quantity": 1,
  "workspace_id": "...",
  "user_id": "..."
}
```

**Error codes:** `VALIDATION_ERROR` (422) — `event_type` is required.

---

## 13. Enforcement

**Base path:** `/api/v1/enforcement`

Plan enforcement status for the caller's organization.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `bearer` |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/enforcement/status` | 200 | Current plan limits and live usage counters. |

**Response schema:**

```json
{
  "plan": "pro",
  "limits": {
    "users": 10,
    "workspaces": 5,
    "api_calls_per_month": 10000,
    "storage_gb": 50
  },
  "usage": {
    "users": 3,
    "workspaces": 2,
    "api_calls_this_month": 1240
  },
  "headroom": {
    "can_add_user": true,
    "can_add_workspace": true,
    "can_make_api_call": true
  }
}
```

---

## 14. Billing

**Base path:** `/api/v1/billing`

Stripe billing integration. Webhook endpoint has no JWT auth (uses Stripe signature verification).

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| POST /checkout | `admin` + org context |
| GET /portal | `admin` + org context |
| GET /subscription | `bearer` + org context |
| POST /webhook | `none` (Stripe signature) |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| POST | `/api/v1/billing/checkout` | 200 | Create a Stripe Checkout Session. |
| GET | `/api/v1/billing/portal` | 200 | Create a Stripe Customer Portal session URL. |
| GET | `/api/v1/billing/subscription` | 200 | Current subscription summary. |
| POST | `/api/v1/billing/webhook` | 200 | Stripe webhook receiver. |

**Request schemas:**

```python
class CheckoutRequest(BaseModel):
    plan_slug: str
    success_url: str
    cancel_url: str
```

**Response schemas:**

`POST /checkout` returns `{"checkout_url": "https://checkout.stripe.com/..."}`.

`GET /portal` returns `{"portal_url": "https://billing.stripe.com/..."}`.

`GET /subscription` returns:
```json
{
  "plan": "pro",
  "status": "active",
  "stripe_customer_id": "cus_...",
  "stripe_subscription_id": "sub_..."
}
```

**Error codes:**
- `INVALID_PLAN` (400) — plan slug not found or inactive.
- `NO_STRIPE_CUSTOMER` (400) — org has no Stripe customer yet; complete checkout first.

---

## 15. SCIM 2.0

**Base path:** `/scim/v2`

SCIM 2.0 provisioning endpoints (RFC 7644). Org-scoped and admin-only. Responses use `application/scim+json` and do **not** use the standard `APIResponse` envelope.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| All | `admin` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/scim/v2/ServiceProviderConfig` | 200 | Server capabilities. |
| GET | `/scim/v2/Users` | 200 | List users (`?filter=`, `?startIndex=`, `?count=` 1–200 default 100). |
| POST | `/scim/v2/Users` | 201 | Create user. |
| GET | `/scim/v2/Users/{id}` | 200 | Get user. |
| PUT | `/scim/v2/Users/{id}` | 200 | Full replace user. |
| PATCH | `/scim/v2/Users/{id}` | 200 | Partial update (RFC 7644 §3.5.2). |
| DELETE | `/scim/v2/Users/{id}` | 204 | Deactivate user (soft-delete). |

**SCIM schemas:**

```python
class SCIMEmail(BaseModel):
    value: str
    primary: bool = False
    type: str | None = None

class SCIMUserCreate(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:schemas:core:2.0:User"]
    userName: str
    name: SCIMName | None = None
    displayName: str | None = None
    emails: list[SCIMEmail] = []
    active: bool = True
    externalId: str | None = None
    password: str | None = None

class SCIMUserReplace(SCIMUserCreate):
    """Same fields as create; used for PUT."""

class SCIMPatchOp(BaseModel):
    op: Literal["add", "replace", "remove"]
    path: str | None = None
    value: Any = None

class SCIMPatch(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:api:messages:2.0:PatchOp"]
    Operations: list[SCIMPatchOp]

class SCIMListResponse(BaseModel):
    schemas: list[str] = ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
    totalResults: int
    startIndex: int = 1
    itemsPerPage: int
    Resources: list[SCIMUser]
```

**Supported filters:**
- `username eq "alice"`
- `emails.value eq "alice@example.com"`
- `active eq true`

**Error codes:**
- `uniqueness` (409) — user already exists in this org.
- `uniqueness` (422) — cross-org conflict (opaque to prevent enumeration).
- `invalidFilter` (400) — unsupported or malformed filter.
- `invalidValue` (503) — SCIM base URL not configured.

---

## 16. Users

**Base path:** `/api/v1/users`

User management endpoints. Admin-only for list/create/update/delete; quota endpoint is available to any authenticated user.

**Auth requirements:**

| Endpoint | Auth |
|----------|------|
| GET /users/me/quota | `bearer` |
| GET /users | `admin` + org context |
| POST /users | `admin` + org context |
| PATCH /users/{id} | `admin` + org context |
| DELETE /users/{id} | `admin` + org context |

**Endpoints:**

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/api/v1/users/me/quota` | 200 | Get my quota limits and usage. |
| GET | `/api/v1/users` | 200 | List all users in the caller's org. |
| POST | `/api/v1/users` | 201 | Create a user. |
| PATCH | `/api/v1/users/{id}` | 200 | Update user role. |
| DELETE | `/api/v1/users/{id}` | 200 | Delete a user. |

**Schemas:**

```python
class UserOut(BaseModel):
    id: str
    username: str
    email: str
    role: str          # "admin" or "user"
    created_at: datetime

class UserCreateRequest(BaseModel):
    username: str      # 3–64 chars, pattern: ^[a-zA-Z0-9_-]+$
    email: str         # valid email
    password: str      # 8–128 chars; must include lowercase, uppercase, digit, and special character
    role: str = "user" # "admin" or "user"

class UserUpdateRequest(BaseModel):
    role: str | None   # "admin" or "user"
```

**Error codes:**
- `FORBIDDEN` (403) — caller is not an admin.
- `CONFLICT` (409) — username or email already exists.
- `BAD_REQUEST` (400) — self-demotion (`PATCH` own role to `user`) or self-deletion.
- `NOT_FOUND` (404) — user not found in this org.
- `VALIDATION_ERROR` (422) — invalid role value.

---

*End of Enterprise API Reference*
