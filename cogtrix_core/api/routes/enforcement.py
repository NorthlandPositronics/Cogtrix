"""Plan enforcement routes (Enterprise Phase 1 — task 1.4.4).

Endpoints:
    GET /api/v1/enforcement/status   — current plan limits and usage for caller's org
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from cogtrix_core.api.auth import TokenData, get_current_user
from cogtrix_core.api.plan_enforcement import PlanLimitSnapshot, get_enforcement_snapshot
from cogtrix_core.api.schemas.common import APIResponse

router = APIRouter(prefix="/enforcement", tags=["Plan Enforcement"])


@router.get(
    "/status",
    response_model=APIResponse[dict],
    summary="Plan enforcement status",
)
async def enforcement_status(
    snap: PlanLimitSnapshot = Depends(get_enforcement_snapshot),
    _: TokenData = Depends(get_current_user),
) -> APIResponse[dict]:
    """Return the current plan limits and live usage counters for the caller's org."""
    return APIResponse(
        data={
            "plan": snap.plan_slug,
            "limits": {
                "users": snap.max_users,
                "workspaces": snap.max_workspaces,
                "api_calls_per_month": snap.max_api_calls_per_month,
                "storage_gb": snap.max_storage_gb,
            },
            "usage": {
                "users": snap.current_users,
                "workspaces": snap.current_workspaces,
                "api_calls_this_month": snap.current_api_calls,
            },
            "headroom": {
                "can_add_user": snap.can_add_user,
                "can_add_workspace": snap.can_add_workspace,
                "can_make_api_call": snap.can_make_api_call,
            },
        }
    )
