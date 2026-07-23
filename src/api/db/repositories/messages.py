"""Message repository — CRUD for Message ORM model."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import Message


class MessageRepository:
    """Data access layer for Message records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        session_id: str,
        role: str,
        content_json: str,
        tool_calls_json: str | None = None,
    ) -> Message:
        """Insert a new message row and return it."""
        msg = Message(
            session_id=session_id,
            role=role,
            content_json=content_json,
            tool_calls_json=tool_calls_json,
        )
        self._db.add(msg)
        await self._db.flush()
        await self._db.refresh(msg)
        return msg

    async def list_by_session(
        self,
        session_id: str,
        *,
        after_id: str | None = None,
        limit: int = 50,
    ) -> list[Message]:
        """Return a page of messages for the session, oldest first.

        ``after_id`` is the cursor: the last ``id`` seen on the previous page.
        """
        query = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )

        if after_id is not None:
            cursor_result = await self._db.execute(
                select(Message.created_at, Message.id).where(Message.id == after_id)
            )
            cursor_row = cursor_result.one_or_none()
            if cursor_row is not None:
                cursor_created_at, cursor_id = cursor_row
                query = query.where(
                    (Message.created_at > cursor_created_at)
                    | ((Message.created_at == cursor_created_at) & (Message.id > cursor_id))
                )

        query = query.limit(limit + 1)
        result = await self._db.execute(query)
        return list(result.scalars().all())

    async def delete_by_session(self, session_id: str) -> int:
        """Delete all messages for a session. Returns the number of deleted rows."""
        result = await self._db.execute(delete(Message).where(Message.session_id == session_id))
        await self._db.flush()
        return result.rowcount
