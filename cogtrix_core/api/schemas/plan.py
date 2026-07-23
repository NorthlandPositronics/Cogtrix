"""Plan schemas (Enterprise Phase 1 — task 1.4.1)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cogtrix_core.api.schemas.common import ensure_utc


class PlanLimits(BaseModel):
    """Quantitative limits for a subscription plan.  0 = unlimited."""

    max_users: int = 0
    max_workspaces: int = 0
    max_api_calls_per_month: int = 0
    max_storage_gb: int = 0


class PlanOut(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None = None
    price_monthly_cents: int
    price_annual_cents: int
    limits: PlanLimits
    is_active: bool
    is_public: bool
    created_at: datetime

    _ensure_utc = field_validator("created_at", mode="before")(ensure_utc)

    @field_validator("limits", mode="before")
    @classmethod
    def _parse_limits(cls, v: Any) -> PlanLimits:
        if isinstance(v, PlanLimits):
            return v
        if isinstance(v, str):
            try:
                return PlanLimits(**json.loads(v))
            except Exception:
                return PlanLimits()
        if isinstance(v, dict):
            return PlanLimits(**v)
        return PlanLimits()


class PlanCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    slug: str = Field(..., min_length=1, max_length=32, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    description: str | None = None
    price_monthly_cents: int = Field(default=0, ge=0)
    price_annual_cents: int = Field(default=0, ge=0)
    limits: PlanLimits = Field(default_factory=PlanLimits)
    is_public: bool = True


class PlanUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = None
    price_monthly_cents: int | None = Field(default=None, ge=0)
    price_annual_cents: int | None = Field(default=None, ge=0)
    limits: PlanLimits | None = None
    is_active: bool | None = None
    is_public: bool | None = None
