"""Organization schemas (Enterprise Phase 1)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc

_VALID_PLANS = frozenset({"free", "pro", "team", "enterprise"})
_SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class OrganizationOut(BaseModel):
    """A tenant organization."""

    id: str = Field(
        ...,
        description="UUID v4 of the organization.",
        examples=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
    )
    name: str = Field(
        ...,
        description="Human-readable organization name (unique).",
        examples=["Acme Corp"],
    )
    slug: str = Field(
        ...,
        description="URL-safe identifier (unique, lowercase, hyphens only).",
        examples=["acme-corp"],
    )
    plan: str = Field(
        ...,
        description="Subscription plan: free, pro, team, or enterprise.",
        examples=["enterprise"],
    )
    settings: dict[str, Any] | None = Field(
        default=None,
        description="Org-level config overrides (arbitrary JSON object).",
    )
    is_active: bool = Field(
        ...,
        description="Whether the organization is active (soft-delete flag).",
    )
    created_at: datetime = Field(..., description="UTC timestamp of creation.")
    updated_at: datetime = Field(..., description="UTC timestamp of last update.")

    _ensure_created_utc = field_validator("created_at", mode="before")(ensure_utc)
    _ensure_updated_utc = field_validator("updated_at", mode="before")(ensure_utc)

    @field_validator("settings", mode="before")
    @classmethod
    def _parse_settings(cls, v: Any) -> dict[str, Any] | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class OrganizationCreate(BaseModel):
    """Request body for creating an organization."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable organization name (unique).",
        examples=["Acme Corp"],
    )
    slug: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=_SLUG_PATTERN,
        description="URL-safe slug (lowercase letters, digits, hyphens; unique).",
        examples=["acme-corp"],
    )
    plan: str = Field(
        default="free",
        description="Subscription plan: free, pro, team, or enterprise.",
        examples=["free"],
    )
    settings: dict[str, Any] | None = Field(
        default=None,
        description="Optional org-level config overrides.",
    )

    @field_validator("plan")
    @classmethod
    def _validate_plan(cls, v: str) -> str:
        if v not in _VALID_PLANS:
            raise ValueError(f"plan must be one of {sorted(_VALID_PLANS)}")
        return v


class OrganizationUpdate(BaseModel):
    """Request body for updating an organization (all fields optional)."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="New human-readable name.",
    )
    plan: str | None = Field(
        default=None,
        description="New subscription plan.",
    )
    settings: dict[str, Any] | None = Field(
        default=None,
        description="New settings blob (replaces existing).",
    )
    is_active: bool | None = Field(
        default=None,
        description="Activate or soft-delete the organization.",
    )

    @field_validator("plan")
    @classmethod
    def _validate_plan(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_PLANS:
            raise ValueError(f"plan must be one of {sorted(_VALID_PLANS)}")
        return v


class OrgSummary(BaseModel):
    """Lightweight organization record for admin list views."""

    id: str = Field(..., description="UUID v4 of the organization.")
    name: str = Field(..., description="Human-readable organization name.")
    slug: str = Field(..., description="URL-safe identifier.")
    status: str = Field(..., description="Organization status: active, inactive, or suspended.")
    plan: str = Field(..., description="Subscription plan.")
    member_count: int = Field(..., description="Number of users in the organization.")
    created_at: datetime = Field(..., description="UTC timestamp of creation.")

    _ensure_created_utc = field_validator("created_at", mode="before")(ensure_utc)


class AdminStats(BaseModel):
    """Global system statistics for admin dashboards."""

    total_orgs: int = Field(..., description="Total number of organizations.")
    active_sessions: int = Field(..., description="Number of non-archived sessions.")
    total_users: int = Field(..., description="Total number of registered users.")
    mcp_server_count: int = Field(..., description="Number of configured MCP servers.")


class OrgUsage(BaseModel):
    """Aggregated usage metrics for a single organization."""

    org_id: str = Field(..., description="UUID v4 of the organization.")
    from_date: str | None = Field(
        default=None, description="Start of the query range (ISO 8601 date)."
    )
    to_date: str | None = Field(default=None, description="End of the query range (ISO 8601 date).")
    total_api_calls: int = Field(0, description="Total API calls.")
    total_sessions: int = Field(0, description="Total sessions created.")
    total_users_provisioned: int = Field(0, description="Total users provisioned.")
    total_storage_kb: int = Field(0, description="Total storage writes in KB.")
    total_workspaces: int = Field(0, description="Total workspaces created.")


class OrgAuditLog(BaseModel):
    """Audit log entries for an organization (stub until DB audit table exists)."""

    entries: list[dict[str, Any]] = Field(default_factory=list, description="Audit log entries.")
    note: str = Field(
        "",
        description="Human-readable note about audit log availability.",
    )


class ImpersonateRequest(BaseModel):
    """Request body to start an impersonation session."""

    user_id: str = Field(
        ...,
        description="UUID of the organization member to impersonate.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Business reason for impersonation (audited).",
    )
    duration_minutes: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Impersonation session lifetime in minutes (default 30, max 120).",
    )


class ImpersonateResponse(BaseModel):
    """Response when an impersonation session is created successfully."""

    impersonation_token: str = Field(
        ...,
        description="JWT to use for subsequent requests as the impersonated user.",
    )
    expires_at: datetime = Field(..., description="UTC timestamp when the session expires.")
    impersonated_user_id: str = Field(..., description="UUID of the impersonated user.")
    org_id: str = Field(..., description="UUID of the target organization.")

    _ensure_expires_utc = field_validator("expires_at", mode="before")(ensure_utc)


class AuditLogEntryOut(BaseModel):
    """Single audit log entry for compliance export."""

    id: str = Field(..., description="UUID of the audit entry.")
    actor_id: str = Field(..., description="UUID of the user who performed the action.")
    impersonated_by: str | None = Field(
        default=None, description="UUID of the superadmin if this was an impersonated action."
    )
    action: str = Field(..., description="Action code (e.g., impersonation.start).")
    resource_type: str = Field(..., description="Type of resource affected.")
    resource_id: str | None = Field(default=None, description="UUID of the affected resource.")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured details about the action."
    )
    created_at: datetime = Field(..., description="UTC timestamp of the action.")

    _ensure_created_utc = field_validator("created_at", mode="before")(ensure_utc)
