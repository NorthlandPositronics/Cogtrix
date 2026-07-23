"""Plan enforcement — check usage against plan limits (Enterprise Phase 1 — task 1.4.4).

Provides ``check_plan_limit`` and ``require_plan_capacity`` FastAPI dependencies
that gate resource creation behind subscription plan limits.

Limits are read from the org's active ``Plan`` (via ``plan_id`` FK).
A limit value of ``0`` means **unlimited** (enterprise tier behaviour).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.db.engine import get_db
from cogtrix_core.api.db.repositories.organization import OrganizationRepository
from cogtrix_core.api.db.repositories.plans import PlanRepository
from cogtrix_core.api.db.repositories.usage import UsageRepository
from cogtrix_core.api.org_context import OrgContext, get_org_context, require_org_context

log = logging.getLogger("cogtrix.api.enforcement")


# ---------------------------------------------------------------------------
# Resolved limit snapshot
# ---------------------------------------------------------------------------


@dataclass
class PlanLimitSnapshot:
    """Current limits and usage for a single org.

    Attributes:
        plan_slug:                Active plan slug (e.g. ``"pro"``).
        max_users:                User seat cap  (0 = unlimited).
        max_workspaces:           Workspace cap  (0 = unlimited).
        max_api_calls_per_month:  Monthly API call cap  (0 = unlimited).
        max_storage_gb:           Storage cap in GB  (0 = unlimited).
        current_users:            Users currently in the org.
        current_workspaces:       Workspaces currently in the org.
        current_api_calls:        API calls recorded this calendar month.
    """

    plan_slug: str
    max_users: int
    max_workspaces: int
    max_api_calls_per_month: int
    max_storage_gb: int
    current_users: int
    current_workspaces: int
    current_api_calls: int

    def within_limit(self, limit: int, current: int) -> bool:
        """Return True when *current* is within *limit* (0 = unlimited)."""
        return limit == 0 or current < limit

    @property
    def can_add_user(self) -> bool:
        return self.within_limit(self.max_users, self.current_users)

    @property
    def can_add_workspace(self) -> bool:
        return self.within_limit(self.max_workspaces, self.current_workspaces)

    @property
    def can_make_api_call(self) -> bool:
        return self.within_limit(self.max_api_calls_per_month, self.current_api_calls)


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


async def get_plan_limit_snapshot(
    org_id: str,
    db: AsyncSession,
) -> PlanLimitSnapshot:
    """Load the active plan limits and current usage for *org_id*.

    Falls back to fully-unlimited limits when the org has no plan assigned.
    """
    from datetime import UTC, datetime

    from sqlalchemy import func, select

    from cogtrix_core.api.db.models import User, Workspace
    from cogtrix_core.api.db.repositories.usage import EVENT_API_CALL

    now = datetime.now(UTC)

    # --- Load plan limits ---
    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(org_id)
    limits = {
        "max_users": 0,
        "max_workspaces": 0,
        "max_api_calls_per_month": 0,
        "max_storage_gb": 0,
    }
    plan_slug = "free"

    if org is not None and org.plan_id:
        plan_repo = PlanRepository(db)
        plan = await plan_repo.get_by_id(org.plan_id)
        if plan is not None and plan.limits:
            try:
                limits = {**limits, **json.loads(plan.limits)}
            except (json.JSONDecodeError, TypeError):
                pass
            plan_slug = plan.slug

    # --- Load current usage ---
    user_count_result = await db.execute(
        select(func.count()).select_from(User).where(User.org_id == org_id)
    )
    current_users = user_count_result.scalar_one()

    ws_count_result = await db.execute(
        select(func.count())
        .select_from(Workspace)
        .where(Workspace.org_id == org_id, Workspace.is_active.is_(True))
    )
    current_workspaces = ws_count_result.scalar_one()

    usage_repo = UsageRepository(db)
    current_api_calls = await usage_repo.count_for_period(
        org_id, EVENT_API_CALL, now.year, now.month
    )

    return PlanLimitSnapshot(
        plan_slug=plan_slug,
        max_users=int(limits.get("max_users", 0)),
        max_workspaces=int(limits.get("max_workspaces", 0)),
        max_api_calls_per_month=int(limits.get("max_api_calls_per_month", 0)),
        max_storage_gb=int(limits.get("max_storage_gb", 0)),
        current_users=current_users,
        current_workspaces=current_workspaces,
        current_api_calls=current_api_calls,
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_enforcement_snapshot(
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
) -> PlanLimitSnapshot:
    """FastAPI dependency: resolve the plan limit snapshot for the caller's org."""
    return await get_plan_limit_snapshot(ctx.org_id, db)  # type: ignore[arg-type]


def _quota_exceeded(resource: str, current: int, limit: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "code": "QUOTA_EXCEEDED",
            "message": (
                f"Plan quota exceeded for '{resource}': "
                f"{current}/{limit} used. Upgrade your plan to continue."
            ),
            "resource": resource,
            "current": current,
            "limit": limit,
        },
    )


async def require_user_capacity(
    snap: PlanLimitSnapshot = Depends(get_enforcement_snapshot),
) -> PlanLimitSnapshot:
    """Raise 402 when the org has reached its user seat limit."""
    if not snap.can_add_user:
        raise _quota_exceeded("users", snap.current_users, snap.max_users)
    return snap


async def require_workspace_capacity(
    snap: PlanLimitSnapshot = Depends(get_enforcement_snapshot),
) -> PlanLimitSnapshot:
    """Raise 402 when the org has reached its workspace limit."""
    if not snap.can_add_workspace:
        raise _quota_exceeded("workspaces", snap.current_workspaces, snap.max_workspaces)
    return snap


async def require_api_call_capacity(
    snap: PlanLimitSnapshot = Depends(get_enforcement_snapshot),
) -> PlanLimitSnapshot:
    """Raise 402 when the org has exhausted its monthly API call quota."""
    if not snap.can_make_api_call:
        raise _quota_exceeded(
            "api_calls_per_month", snap.current_api_calls, snap.max_api_calls_per_month
        )
    return snap


async def maybe_require_api_call_capacity(
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Enforce plan API call quota when the user belongs to an org.

    No-op for free-tier users without an org.  Use this on public-facing
    endpoints (session creation, message sending) where org membership is
    optional but plan quotas should still apply for org users.
    """
    if not ctx.has_org:
        return
    snap = await get_plan_limit_snapshot(ctx.org_id, db)  # type: ignore[arg-type]
    if not snap.can_make_api_call:
        raise _quota_exceeded(
            "api_calls_per_month", snap.current_api_calls, snap.max_api_calls_per_month
        )
