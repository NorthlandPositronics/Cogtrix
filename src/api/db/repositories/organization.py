"""Organization repository — CRUD operations for the Organization ORM model."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import Organization, User


class OrganizationRepository:
    """Data access layer for Organization records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        org_id: str,
        name: str,
        slug: str,
        plan: str = "free",
        settings: dict | None = None,
    ) -> Organization:
        """Insert a new Organization row and return it."""
        org = Organization(
            id=org_id,
            name=name,
            slug=slug,
            plan=plan,
            settings=json.dumps(settings) if settings is not None else None,
        )
        self._db.add(org)
        await self._db.flush()
        await self._db.refresh(org)
        return org

    async def get_by_id(self, org_id: str) -> Organization | None:
        """Return the organization with the given UUID, or None."""
        result = await self._db.execute(select(Organization).where(Organization.id == org_id))
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Organization | None:
        """Return the organization with the given slug, or None."""
        result = await self._db.execute(select(Organization).where(Organization.slug == slug))
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Organization | None:
        """Return the organization with the given name, or None."""
        result = await self._db.execute(select(Organization).where(Organization.name == name))
        return result.scalar_one_or_none()

    async def list_all(self, *, include_inactive: bool = False) -> list[Organization]:
        """Return all organizations ordered by creation time."""
        stmt = select(Organization).order_by(Organization.created_at)
        if not include_inactive:
            stmt = stmt.where(Organization.is_active.is_(True))
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def list_orgs(
        self,
        *,
        after_id: str | None = None,
        limit: int = 20,
        name_filter: str | None = None,
        status_filter: str | None = None,
        plan_filter: str | None = None,
        created_after: datetime | None = None,
    ) -> list[Organization]:
        """Return organizations with optional filters and cursor pagination.

        Args:
            after_id:  Cursor — return rows after this org id (by created_at).
            limit:     Maximum rows to return.
            name_filter: Substring match on org name (case-insensitive).
            status_filter: Exact match on status.
            plan_filter: Exact match on plan.
            created_after: Return orgs created on or after this datetime.
        """
        stmt = select(Organization).order_by(Organization.id)

        if after_id:
            stmt = stmt.where(Organization.id > after_id)
        if name_filter:
            stmt = stmt.where(Organization.name.ilike(f"%{name_filter}%"))
        if status_filter:
            stmt = stmt.where(Organization.status == status_filter)
        if plan_filter:
            stmt = stmt.where(Organization.plan == plan_filter)
        if created_after:
            stmt = stmt.where(Organization.created_at >= created_after)

        stmt = stmt.limit(limit + 1)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def count_orgs(
        self,
        *,
        name_filter: str | None = None,
        status_filter: str | None = None,
        plan_filter: str | None = None,
        created_after: datetime | None = None,
    ) -> int:
        """Return total count of organizations matching the filters."""
        stmt = select(func.count()).select_from(Organization)
        if name_filter:
            stmt = stmt.where(Organization.name.ilike(f"%{name_filter}%"))
        if status_filter:
            stmt = stmt.where(Organization.status == status_filter)
        if plan_filter:
            stmt = stmt.where(Organization.plan == plan_filter)
        if created_after:
            stmt = stmt.where(Organization.created_at >= created_after)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def count_users_per_org(self, org_ids: list[str]) -> dict[str, int]:
        """Return a mapping org_id -> user count for the given org ids."""
        if not org_ids:
            return {}
        stmt = (
            select(User.org_id, func.count().label("cnt"))
            .where(User.org_id.in_(org_ids))
            .group_by(User.org_id)
        )
        result = await self._db.execute(stmt)
        return {row[0]: row[1] for row in result.all()}

    async def update(
        self,
        org_id: str,
        *,
        name: str | None = None,
        plan: str | None = None,
        settings: dict | None = None,
        is_active: bool | None = None,
    ) -> Organization | None:
        """Partially update an organization; return the updated row or None."""
        values: dict = {}
        if name is not None:
            values["name"] = name
        if plan is not None:
            values["plan"] = plan
        if settings is not None:
            values["settings"] = json.dumps(settings)
        if is_active is not None:
            values["is_active"] = is_active
        if not values:
            return await self.get_by_id(org_id)
        stmt = (
            update(Organization)
            .where(Organization.id == org_id)
            .values(**values)
            .returning(Organization)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete(self, org_id: str) -> bool:
        """Hard-delete the organization; return True if a row was deleted."""
        stmt = delete(Organization).where(Organization.id == org_id)
        result = await self._db.execute(stmt)
        return result.rowcount > 0

    async def list_users(self, org_id: str) -> list[User]:
        """Return all users belonging to the given organization."""
        result = await self._db.execute(
            select(User).where(User.org_id == org_id).order_by(User.created_at)
        )
        return list(result.scalars().all())

    async def count_users(self, org_id: str) -> int:
        """Return the number of users in the given organization."""
        from sqlalchemy import func

        result = await self._db.execute(
            select(func.count()).select_from(User).where(User.org_id == org_id)
        )
        return result.scalar_one()

    async def ensure_default_org(self) -> Organization:
        """Return the default organization, creating it if it does not exist.

        Used at application startup and in tests to guarantee the compatibility
        default org (slug='default') is always present.  Idempotent: calling
        this multiple times always returns the same row.
        """
        import uuid as _uuid

        DEFAULT_SLUG = "default"
        existing = await self.get_by_slug(DEFAULT_SLUG)
        if existing is not None:
            return existing

        return await self.create(
            org_id=str(_uuid.uuid4()),
            name="Default Organization",
            slug=DEFAULT_SLUG,
            plan="free",
        )
