"""Session repository — CRUD for ApiSessionRecord ORM model."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import ApiSessionRecord, Message


class SessionRepository:
    """Data access layer for ApiSessionRecord rows."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        config_json: str = "{}",
        workspace_id: str | None = None,
    ) -> ApiSessionRecord:
        """Insert a new session row and return it."""
        record = ApiSessionRecord(
            user_id=user_id,
            name=name,
            config_json=config_json,
            workspace_id=workspace_id,
            token_counts_json=json.dumps(
                {"input_tokens": 0, "output_tokens": 0, "context_window": 0}
            ),
            active_tools_json="[]",
            state="idle",
        )
        self._db.add(record)
        await self._db.flush()
        await self._db.refresh(record)
        return record

    async def get_by_id(self, session_id: str) -> ApiSessionRecord | None:
        """Return the session with the given UUID, or None."""
        result = await self._db.execute(
            select(ApiSessionRecord).where(ApiSessionRecord.id == session_id)
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        *,
        after_id: str | None = None,
        limit: int = 20,
        include_archived: bool = False,
    ) -> list[ApiSessionRecord]:
        """Return a page of sessions for the user, newest first.

        ``after_id`` is the cursor: the last ``id`` seen on the previous page.
        """
        query = select(ApiSessionRecord).where(ApiSessionRecord.user_id == user_id)

        if not include_archived:
            query = query.where(ApiSessionRecord.archived_at.is_(None))

        query = query.order_by(ApiSessionRecord.updated_at.desc(), ApiSessionRecord.id.desc())

        if after_id is not None:
            cursor_result = await self._db.execute(
                select(ApiSessionRecord.updated_at, ApiSessionRecord.id).where(
                    ApiSessionRecord.id == after_id
                )
            )
            cursor_row = cursor_result.one_or_none()
            if cursor_row is not None:
                cursor_updated_at, cursor_id = cursor_row
                query = query.where(
                    (ApiSessionRecord.updated_at < cursor_updated_at)
                    | (
                        (ApiSessionRecord.updated_at == cursor_updated_at)
                        & (ApiSessionRecord.id < cursor_id)
                    )
                )

        query = query.limit(limit + 1)
        result = await self._db.execute(query)
        return list(result.scalars().all())

    async def list_all(
        self,
        *,
        after_id: str | None = None,
        limit: int = 20,
        include_archived: bool = False,
    ) -> list[ApiSessionRecord]:
        """Return a page of all sessions (admin use), newest first."""
        query = select(ApiSessionRecord)

        if not include_archived:
            query = query.where(ApiSessionRecord.archived_at.is_(None))

        query = query.order_by(ApiSessionRecord.updated_at.desc(), ApiSessionRecord.id.desc())

        if after_id is not None:
            cursor_result = await self._db.execute(
                select(ApiSessionRecord.updated_at, ApiSessionRecord.id).where(
                    ApiSessionRecord.id == after_id
                )
            )
            cursor_row = cursor_result.one_or_none()
            if cursor_row is not None:
                cursor_updated_at, cursor_id = cursor_row
                query = query.where(
                    (ApiSessionRecord.updated_at < cursor_updated_at)
                    | (
                        (ApiSessionRecord.updated_at == cursor_updated_at)
                        & (ApiSessionRecord.id < cursor_id)
                    )
                )

        query = query.limit(limit + 1)
        result = await self._db.execute(query)
        return list(result.scalars().all())

    async def update(self, session_id: str, **fields) -> ApiSessionRecord | None:
        """Update arbitrary columns on a session row and return the refreshed record."""
        fields["updated_at"] = datetime.now(UTC)
        await self._db.execute(
            update(ApiSessionRecord).where(ApiSessionRecord.id == session_id).values(**fields)
        )
        await self._db.flush()
        result = await self._db.execute(
            select(ApiSessionRecord).where(ApiSessionRecord.id == session_id)
        )
        return result.scalar_one_or_none()

    async def archive(self, session_id: str) -> None:
        """Set archived_at to now for the given session."""
        now = datetime.now(UTC)
        await self._db.execute(
            update(ApiSessionRecord)
            .where(ApiSessionRecord.id == session_id)
            .values(archived_at=now, updated_at=now)
        )
        await self._db.flush()

    async def restore(self, session_id: str) -> None:
        """Clear archived_at, making the session visible in default listings again."""
        now = datetime.now(UTC)
        await self._db.execute(
            update(ApiSessionRecord)
            .where(ApiSessionRecord.id == session_id)
            .values(archived_at=None, updated_at=now)
        )
        await self._db.flush()

    async def hard_delete(self, session_id: str) -> bool:
        """Permanently delete a session and all its messages (non-recoverable).

        Returns True if the session existed and was deleted, False otherwise.
        """
        await self._db.execute(delete(Message).where(Message.session_id == session_id))
        result = await self._db.execute(
            delete(ApiSessionRecord).where(ApiSessionRecord.id == session_id)
        )
        await self._db.flush()
        return result.rowcount > 0

    async def count_by_user(self, user_id: str) -> int:
        """Return the total number of active (non-archived) sessions for a user."""
        result = await self._db.execute(
            select(func.count())
            .select_from(ApiSessionRecord)
            .where(
                ApiSessionRecord.user_id == user_id,
                ApiSessionRecord.archived_at.is_(None),
            )
        )
        return result.scalar_one()

    async def count_all_active(self) -> int:
        """Return the total number of active (non-archived) sessions across all users."""
        result = await self._db.execute(
            select(func.count())
            .select_from(ApiSessionRecord)
            .where(ApiSessionRecord.archived_at.is_(None))
        )
        return result.scalar_one()

    async def name_exists_for_user(
        self, user_id: str, name: str, *, exclude_id: str | None = None
    ) -> bool:
        """Check if an active (non-archived) session with this name exists for the user."""
        query = (
            select(func.count())
            .select_from(ApiSessionRecord)
            .where(
                ApiSessionRecord.user_id == user_id,
                ApiSessionRecord.name == name,
                ApiSessionRecord.archived_at.is_(None),
            )
        )
        if exclude_id is not None:
            query = query.where(ApiSessionRecord.id != exclude_id)
        result = await self._db.execute(query)
        return result.scalar_one() > 0
