"""Plan repository — CRUD for Plan records."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import Plan


class PlanRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        plan_id: str,
        name: str,
        slug: str,
        description: str | None = None,
        price_monthly_cents: int = 0,
        price_annual_cents: int = 0,
        stripe_price_id: str | None = None,
        limits: dict[str, Any] | None = None,
        is_public: bool = True,
    ) -> Plan:
        plan = Plan(
            id=plan_id,
            name=name,
            slug=slug,
            description=description,
            price_monthly_cents=price_monthly_cents,
            price_annual_cents=price_annual_cents,
            stripe_price_id=stripe_price_id,
            limits=json.dumps(limits) if limits else None,
            is_public=is_public,
        )
        self._db.add(plan)
        await self._db.flush()
        await self._db.refresh(plan)
        return plan

    async def get_by_id(self, plan_id: str) -> Plan | None:
        result = await self._db.execute(select(Plan).where(Plan.id == plan_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Plan | None:
        result = await self._db.execute(select(Plan).where(Plan.slug == slug))
        return result.scalar_one_or_none()

    async def list_all(
        self, *, include_inactive: bool = False, public_only: bool = False
    ) -> list[Plan]:
        stmt = select(Plan)
        if not include_inactive:
            stmt = stmt.where(Plan.is_active.is_(True))
        if public_only:
            stmt = stmt.where(Plan.is_public.is_(True))
        result = await self._db.execute(stmt.order_by(Plan.price_monthly_cents))
        return list(result.scalars().all())

    async def update(
        self,
        plan_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        price_monthly_cents: int | None = None,
        price_annual_cents: int | None = None,
        stripe_price_id: str | None = None,
        limits: dict[str, Any] | None = None,
        is_active: bool | None = None,
        is_public: bool | None = None,
    ) -> Plan | None:
        plan = await self.get_by_id(plan_id)
        if plan is None:
            return None
        if name is not None:
            plan.name = name
        if description is not None:
            plan.description = description
        if price_monthly_cents is not None:
            plan.price_monthly_cents = price_monthly_cents
        if price_annual_cents is not None:
            plan.price_annual_cents = price_annual_cents
        if stripe_price_id is not None:
            plan.stripe_price_id = stripe_price_id
        if limits is not None:
            plan.limits = json.dumps(limits)
        if is_active is not None:
            plan.is_active = is_active
        if is_public is not None:
            plan.is_public = is_public
        await self._db.flush()
        await self._db.refresh(plan)
        return plan

    async def delete(self, plan_id: str) -> bool:
        result = await self._db.execute(delete(Plan).where(Plan.id == plan_id))
        return result.rowcount > 0
