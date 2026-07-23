"""Plan management routes (Enterprise Phase 1 — task 1.4.1).

Endpoints:
    GET    /api/v1/plans                 — list public plans (authenticated)
    GET    /api/v1/plans/{id}            — get plan
    POST   /api/v1/plans                 — create plan (admin)
    PATCH  /api/v1/plans/{id}            — update plan (admin)
    DELETE /api/v1/plans/{id}            — deactivate plan (admin)
    PATCH  /api/v1/organizations/{id}/plan — assign plan to org (admin)
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.db.engine import get_db
from src.api.db.repositories.organization import OrganizationRepository
from src.api.db.repositories.plans import PlanRepository
from src.api.org_context import OrgContext, assert_same_org, get_org_context
from src.api.schemas.common import APIResponse
from src.api.schemas.plan import PlanCreate, PlanOut, PlanUpdate

log = logging.getLogger("cogtrix.api.plans")

router = APIRouter(prefix="/plans", tags=["Plans"])
org_plan_router = APIRouter(prefix="/organizations", tags=["Plans"])


def _to_out(plan) -> PlanOut:
    return PlanOut(
        id=plan.id,
        name=plan.name,
        slug=plan.slug,
        description=plan.description,
        price_monthly_cents=plan.price_monthly_cents,
        price_annual_cents=plan.price_annual_cents,
        limits=plan.limits,
        is_active=plan.is_active,
        is_public=plan.is_public,
        created_at=plan.created_at,
    )


@router.get("", response_model=APIResponse[list[PlanOut]], summary="List plans")
async def list_plans(
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(get_current_user),
) -> APIResponse[list[PlanOut]]:
    """Return all active, public plans."""
    repo = PlanRepository(db)
    plans = await repo.list_all(public_only=True)
    return APIResponse(data=[_to_out(p) for p in plans])


@router.get("/{plan_id}", response_model=APIResponse[PlanOut], summary="Get plan")
async def get_plan(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(get_current_user),
) -> APIResponse[PlanOut]:
    repo = PlanRepository(db)
    plan = await repo.get_by_id(plan_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Plan not found."},
        )
    return APIResponse(data=_to_out(plan))


@router.post("", response_model=APIResponse[PlanOut], status_code=201, summary="Create plan")
async def create_plan(
    body: PlanCreate,
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[PlanOut]:
    repo = PlanRepository(db)
    if await repo.get_by_slug(body.slug) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Plan slug '{body.slug}' already exists."},
        )
    plan = await repo.create(
        plan_id=str(uuid.uuid4()),
        name=body.name,
        slug=body.slug,
        description=body.description,
        price_monthly_cents=body.price_monthly_cents,
        price_annual_cents=body.price_annual_cents,
        limits=body.limits.model_dump(),
        is_public=body.is_public,
    )
    await db.commit()
    return APIResponse(data=_to_out(plan))


@router.patch("/{plan_id}", response_model=APIResponse[PlanOut], summary="Update plan")
async def update_plan(
    plan_id: str,
    body: PlanUpdate,
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[PlanOut]:
    repo = PlanRepository(db)
    plan = await repo.update(
        plan_id,
        name=body.name,
        description=body.description,
        price_monthly_cents=body.price_monthly_cents,
        price_annual_cents=body.price_annual_cents,
        limits=body.limits.model_dump() if body.limits else None,
        is_active=body.is_active,
        is_public=body.is_public,
    )
    if plan is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "Plan not found."}
        )
    await db.commit()
    return APIResponse(data=_to_out(plan))


@router.delete("/{plan_id}", response_model=APIResponse[PlanOut], summary="Deactivate plan")
async def deactivate_plan(
    plan_id: str,
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[PlanOut]:
    """Soft-deactivate a plan (does not delete orgs using it)."""
    repo = PlanRepository(db)
    plan = await repo.update(plan_id, is_active=False)
    if plan is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "Plan not found."}
        )
    await db.commit()
    return APIResponse(data=_to_out(plan))


# ---------------------------------------------------------------------------
# Organization plan assignment
# ---------------------------------------------------------------------------


@org_plan_router.patch(
    "/{org_id}/plan",
    response_model=APIResponse[dict],
    summary="Assign plan to org",
)
async def assign_org_plan(
    org_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
    ctx: OrgContext = Depends(get_org_context),
) -> APIResponse[dict]:
    """Assign a plan to an organization.  Body: ``{"plan_id": "<uuid>"}``."""
    plan_id = body.get("plan_id")
    if not plan_id:
        raise HTTPException(
            status_code=422, detail={"code": "VALIDATION_ERROR", "message": "plan_id required."}
        )

    plan_repo = PlanRepository(db)
    plan = await plan_repo.get_by_id(plan_id)
    if plan is None or not plan.is_active:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "Plan not found."}
        )

    # Enforce org-scoped access BEFORE org lookup to prevent org enumeration
    # and ensure isolation check happens first even for non-existent orgs.
    assert_same_org(ctx, org_id, admin_bypass=False)

    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(org_id)
    if org is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "Organization not found."}
        )

    org.plan = plan.slug
    org.plan_id = plan.id
    await db.commit()
    log.info("Assigned plan %s to org %s", plan.slug, org_id)
    return APIResponse(data={"org_id": org_id, "plan_id": plan_id, "plan_slug": plan.slug})
