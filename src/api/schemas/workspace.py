"""Workspace schemas (Enterprise Phase 1 — task 1.3.1)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc


class WorkspaceOut(BaseModel):
    id: str = Field(..., description="UUID v4 of the workspace.")
    org_id: str = Field(..., description="Owning organization UUID.")
    name: str = Field(..., description="Workspace name (unique within the org).")
    description: str | None = Field(default=None)
    settings: dict[str, Any] | None = Field(default=None, description="Workspace config overrides.")
    member_count: int = Field(default=0)
    is_active: bool
    created_at: datetime

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)

    @field_validator("settings", mode="before")
    @classmethod
    def _parse_settings(cls, v: Any) -> dict[str, Any] | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return json.loads(v)
        return v


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    settings: dict[str, Any] | None = Field(default=None)


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    settings: dict[str, Any] | None = Field(default=None)
    is_active: bool | None = Field(default=None)


class WorkspaceMemberOut(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    joined_at: datetime

    _ensure_utc = field_validator("joined_at", mode="before")(ensure_utc)


class AddWorkspaceMemberRequest(BaseModel):
    user_id: str
    role: str = Field(default="member")

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in {"member", "admin"}:
            raise ValueError("role must be 'member' or 'admin'")
        return v
