"""API key repository — hash-based lookup with cursor-paginated listing."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.db.models import ApiKey


class ApiKeyRepository:
    """Data access layer for ApiKey records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        key_id: str,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        label: str,
        expires_at=None,
    ) -> ApiKey:
        """Insert a new ApiKey row and return it."""
        key = ApiKey(
            id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            label=label,
            expires_at=expires_at,
        )
        self._db.add(key)
        await self._db.flush()
        await self._db.refresh(key)
        return key

    async def get_by_id(self, key_id: str) -> ApiKey | None:
        """Return the API key with the given UUID, or None."""
        result = await self._db.execute(select(ApiKey).where(ApiKey.id == key_id))
        return result.scalar_one_or_none()

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        """Return the API key matching the given SHA-256 hash, or None."""
        result = await self._db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
        return result.scalar_one_or_none()

    async def list_for_user(
        self,
        user_id: str,
        *,
        after_id: str | None = None,
        limit: int = 20,
    ) -> list[ApiKey]:
        """Return a page of API keys for the user, ordered by created_at desc.

        ``after_id`` is the cursor: the last ``id`` seen on the previous page.
        """
        query = (
            select(ApiKey)
            .where(ApiKey.user_id == user_id, ApiKey.revoked.is_(False))
            .order_by(ApiKey.created_at.desc(), ApiKey.id.desc())
        )
        if after_id is not None:
            # Find the created_at of the cursor row, then filter below it
            cursor_result = await self._db.execute(
                select(ApiKey.created_at, ApiKey.id).where(ApiKey.id == after_id)
            )
            cursor_row = cursor_result.one_or_none()
            if cursor_row is not None:
                cursor_created_at, cursor_id = cursor_row
                query = query.where(
                    (ApiKey.created_at < cursor_created_at)
                    | ((ApiKey.created_at == cursor_created_at) & (ApiKey.id < cursor_id))
                )

        query = query.limit(limit + 1)
        result = await self._db.execute(query)
        return list(result.scalars().all())

    async def revoke(self, key_id: str) -> None:
        """Mark the API key with the given UUID as revoked."""
        await self._db.execute(update(ApiKey).where(ApiKey.id == key_id).values(revoked=True))

    async def update_last_used(self, key_id: str, last_used_at) -> None:
        """Update the last_used_at timestamp for an API key."""
        await self._db.execute(
            update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=last_used_at)
        )
