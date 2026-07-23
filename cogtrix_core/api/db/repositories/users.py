"""User repository — CRUD operations for the User ORM model."""

from __future__ import annotations

from sqlalchemy import case, delete, func, insert, literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.db.models import User


class UserRepository:
    """Data access layer for User records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
        role: str = "user",
        org_id: str | None = None,
    ) -> User:
        """Insert a new User row and return it."""
        user = User(
            id=user_id,
            username=username,
            email=email.lower(),
            password_hash=password_hash,
            role=role,
            org_id=org_id,
        )
        self._db.add(user)
        await self._db.flush()
        await self._db.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Return the user with the given UUID, or None."""
        result = await self._db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_id_and_org(self, user_id: str, org_id: str) -> User | None:
        """Return the user with the given UUID scoped to an org, or None."""
        result = await self._db.execute(
            select(User).where(User.id == user_id, User.org_id == org_id)
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str, org_id: str | None = None) -> User | None:
        """Return the user with the given username (case-sensitive), or None."""
        stmt = select(User).where(User.username == username)
        if org_id is not None:
            stmt = stmt.where(User.org_id == org_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str, org_id: str | None = None) -> User | None:
        """Return the user with the given email (case-insensitive), or None."""
        stmt = select(User).where(func.lower(User.email) == email.lower())
        if org_id is not None:
            stmt = stmt.where(User.org_id == org_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def set_active(self, user_id: str, is_active: bool) -> User | None:
        """Set the active flag for a user and return the updated row."""
        stmt = update(User).where(User.id == user_id).values(is_active=is_active).returning(User)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_password(self, user_id: str, password_hash: str) -> User | None:
        """Set a user's password hash and return the updated row (#2065)."""
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(password_hash=password_hash)
            .returning(User)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_with_role_election(
        self,
        *,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
    ) -> User:
        """Insert a user, atomically assigning ``admin`` iff no users exist yet.

        Uses ``INSERT … SELECT CASE WHEN`` so the count check and the insert
        happen in a single statement — eliminates the race where two concurrent
        registrations both see count == 0 and both get ``admin``.
        """
        role_subq = (
            select(case((func.count() == 0, literal("admin")), else_=literal("user")))
            .select_from(User)
            .scalar_subquery()
        )

        stmt = (
            insert(User)
            .values(
                id=user_id,
                username=username,
                email=email.lower(),
                password_hash=password_hash,
                role=role_subq,
            )
            .returning(User)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def list_all(self) -> list[User]:
        """Return all users ordered by creation time."""
        result = await self._db.execute(select(User).order_by(User.created_at))
        return list(result.scalars().all())

    async def list_by_org(self, org_id: str) -> list[User]:
        """Return all users belonging to the given organization."""
        result = await self._db.execute(
            select(User).where(User.org_id == org_id).order_by(User.created_at)
        )
        return list(result.scalars().all())

    async def assign_org(self, user_id: str, org_id: str | None) -> User | None:
        """Set or clear the org_id for a user; return the updated user or None."""
        stmt = update(User).where(User.id == user_id).values(org_id=org_id).returning(User)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_role(self, user_id: str, role: str) -> User | None:
        """Update the role of the user with the given UUID; return the updated user or None."""
        stmt = update(User).where(User.id == user_id).values(role=role).returning(User)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_all(self) -> int:
        """Return the total number of registered users."""
        result = await self._db.execute(select(func.count()).select_from(User))
        return result.scalar_one()

    async def delete(self, user_id: str) -> bool:
        """Delete the user with the given UUID; return True if a row was deleted."""
        stmt = delete(User).where(User.id == user_id)
        result = await self._db.execute(stmt)
        return result.rowcount > 0
