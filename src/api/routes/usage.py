"""Usage metering routes (Enterprise Phase 1 — task 1.4.2).

Endpoints:
    GET /api/v1/usage/summary    — current-month totals for caller's org
    GET /api/v1/usage/records    — recent raw usage records
    POST /api/v1/usage/record    — manually record a usage event (admin)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.db.engine import get_db
from src.api.db.repositories.usage import UsageRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.schemas.common import APIResponse

log = logging.getLogger("cogtrix.api.usage")

router = APIRouter(prefix="/usage", tags=["Usage Metering"])


@router.get("/summary", response_model=APIResponse[dict], summary="Monthly usage summary")
async def usage_summary(
    year: int | None = Query(default=None, description="Year (defaults to current)."),
    month: int | None = Query(default=None, description="Month 1–12 (defaults to current)."),
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(get_current_user),
) -> APIResponse[dict]:
    """Return aggregated usage totals for the caller's org."""
    now = datetime.now(UTC)
    yr = year or now.year
    mo = month or now.month
    repo = UsageRepository(db)
    totals = await repo.get_monthly_totals(ctx.org_id, yr, mo)  # type: ignore[arg-type]
    return APIResponse(
        data={
            "org_id": ctx.org_id,
            "period": f"{yr}-{mo:02d}",
            "totals": totals,
        }
    )


@router.get("/records", response_model=APIResponse[list[dict]], summary="Recent usage records")
async def usage_records(
    event_type: str | None = Query(default=None, description="Filter by event type."),
    limit: int = Query(default=50, ge=1, le=500),
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[list[dict]]:
    """Return recent raw usage records for the caller's org (admin only)."""
    repo = UsageRepository(db)
    records = await repo.get_recent_records(
        ctx.org_id,  # type: ignore[arg-type]
        event_type=event_type,
        limit=limit,
    )
    return APIResponse(
        data=[
            {
                "id": r.id,
                "event_type": r.event_type,
                "quantity": r.quantity,
                "workspace_id": r.workspace_id,
                "user_id": r.user_id,
                "period": f"{r.period_year}-{r.period_month:02d}",
                "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
            }
            for r in records
        ]
    )


@router.post(
    "/record", response_model=APIResponse[dict], status_code=201, summary="Record usage event"
)
async def record_usage(
    body: dict,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Manually record a usage event (admin/integration use)."""
    event_type = body.get("event_type", "")
    if not event_type:
        from fastapi import HTTPException
        from fastapi import status as _status

        raise HTTPException(
            status_code=_status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "VALIDATION_ERROR", "message": "event_type required."},
        )
    repo = UsageRepository(db)
    rec = await repo.record(
        org_id=ctx.org_id,  # type: ignore[arg-type]
        event_type=event_type,
        quantity=int(body.get("quantity", 1)),
        workspace_id=body.get("workspace_id"),
        user_id=body.get("user_id"),
    )
    await db.commit()
    return APIResponse(data={"id": rec.id, "event_type": rec.event_type, "quantity": rec.quantity})
