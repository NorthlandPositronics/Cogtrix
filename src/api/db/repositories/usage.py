"""Usage repository — record and query metered usage events."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import UsageRecord

# ---------------------------------------------------------------------------
# Supported event types
# ---------------------------------------------------------------------------

EVENT_API_CALL = "api_call"
EVENT_SESSION_CREATED = "session_created"
EVENT_USER_PROVISIONED = "user_provisioned"
EVENT_STORAGE_WRITE_KB = "storage_write_kb"
EVENT_WORKSPACE_CREATED = "workspace_created"


class UsageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record(
        self,
        *,
        org_id: str,
        event_type: str,
        quantity: int = 1,
        workspace_id: str | None = None,
        user_id: str | None = None,
        at: datetime | None = None,
    ) -> UsageRecord:
        """Insert a usage event and return it."""
        now = at or datetime.now(UTC)
        rec = UsageRecord(
            org_id=org_id,
            event_type=event_type,
            quantity=quantity,
            workspace_id=workspace_id,
            user_id=user_id,
            period_year=now.year,
            period_month=now.month,
            recorded_at=now,
        )
        self._db.add(rec)
        await self._db.flush()
        return rec

    async def get_monthly_totals(self, org_id: str, year: int, month: int) -> dict[str, int]:
        """Return {event_type: total_quantity} for the given org and period."""
        result = await self._db.execute(
            select(UsageRecord.event_type, func.sum(UsageRecord.quantity))
            .where(
                UsageRecord.org_id == org_id,
                UsageRecord.period_year == year,
                UsageRecord.period_month == month,
            )
            .group_by(UsageRecord.event_type)
        )
        return {row[0]: row[1] for row in result.all()}

    async def get_recent_records(
        self,
        org_id: str,
        *,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        """Return recent usage records for an org, newest-first."""
        stmt = (
            select(UsageRecord)
            .where(UsageRecord.org_id == org_id)
            .order_by(UsageRecord.recorded_at.desc())
            .limit(limit)
        )
        if event_type:
            stmt = stmt.where(UsageRecord.event_type == event_type)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def count_for_period(self, org_id: str, event_type: str, year: int, month: int) -> int:
        """Return the total quantity for a specific event type and period."""
        result = await self._db.execute(
            select(func.coalesce(func.sum(UsageRecord.quantity), 0)).where(
                UsageRecord.org_id == org_id,
                UsageRecord.event_type == event_type,
                UsageRecord.period_year == year,
                UsageRecord.period_month == month,
            )
        )
        return result.scalar_one()

    async def aggregate_by_org(
        self,
        org_id: str,
        *,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> dict[str, int]:
        """Return {event_type: total_quantity} for the given org and optional date range."""
        stmt = (
            select(UsageRecord.event_type, func.sum(UsageRecord.quantity))
            .where(UsageRecord.org_id == org_id)
            .group_by(UsageRecord.event_type)
        )
        if from_date is not None:
            stmt = stmt.where(UsageRecord.recorded_at >= from_date)
        if to_date is not None:
            stmt = stmt.where(UsageRecord.recorded_at <= to_date)
        result = await self._db.execute(stmt)
        return {row[0]: row[1] for row in result.all()}
