"""Team and TeamMembership schemas (Enterprise Phase 1 — task 1.2.4)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc

_VALID_MEMBER_ROLES = frozenset({"member", "admin"})


class TeamOut(BaseModel):
    """A team within an organization."""

    id: str = Field(..., description="UUID v4 of the team.")
    org_id: str = Field(..., description="UUID v4 of the owning organization.")
    name: str = Field(..., description="Team name (unique within the org).")
    description: str | None = Field(default=None, description="Optional description.")
    member_count: int = Field(default=0, description="Number of members.")
    created_at: datetime = Field(..., description="UTC creation timestamp.")

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)


class TeamCreate(BaseModel):
    """Request body for POST /api/v1/teams."""

    name: str = Field(..., min_length=1, max_length=128, description="Team name.")
    description: str | None = Field(default=None, max_length=512)


class TeamUpdate(BaseModel):
    """Request body for PATCH /api/v1/teams/{id}."""

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)


class MemberOut(BaseModel):
    """A user's membership record within a team."""

    user_id: str = Field(..., description="UUID v4 of the user.")
    username: str = Field(..., description="Username of the member.")
    email: str = Field(..., description="Email address of the member.")
    role: str = Field(..., description="Membership role: member or admin.")
    joined_at: datetime = Field(..., description="UTC timestamp when the user joined.")

    _ensure_utc = field_validator("joined_at", mode="before")(ensure_utc)


class AddMemberRequest(BaseModel):
    """Request body for POST /api/v1/teams/{id}/members."""

    user_id: str = Field(..., description="UUID v4 of the user to add.")
    role: str = Field(default="member", description="Membership role: member or admin.")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in _VALID_MEMBER_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_MEMBER_ROLES)}")
        return v
