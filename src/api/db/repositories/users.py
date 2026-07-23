"""User repository — CRUD operations for the User ORM model."""

from __future__ import annotations

from sqlalchemy import case, func, insert, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import User


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
    ) -> User:
        """Insert a new User row and return it."""
        user = User(
            id=user_id,
            username=username,
            email=email.lower(),
            password_hash=password_hash,
            role=role,
        )
        self._db.add(user)
        await self._db.flush()
        await self._db.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Return the user with the given UUID, or None."""
        result = await self._db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Return the user with the given username (case-sensitive), or None."""
        result = await self._db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Return the user with the given email (case-insensitive), or None."""
        result = await self._db.execute(select(User).where(func.lower(User.email) == email.lower()))
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
