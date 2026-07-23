"""Refresh token repository — hash-based lookup with single-use rotation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import RefreshToken


class RefreshTokenRepository:
    """Data access layer for RefreshToken records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        token_id: str,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> RefreshToken:
        """Insert a new RefreshToken row and return it."""
        token = RefreshToken(
            id=token_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self._db.add(token)
        await self._db.flush()
        await self._db.refresh(token)
        return token

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        """Return the refresh token matching the given SHA-256 hash, or None."""
        result = await self._db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def rotate_and_get(self, token_hash: str) -> RefreshToken | None:
        """Atomically check-and-revoke a refresh token.

        Uses a conditional UPDATE to avoid the TOCTOU window in the refresh rotation
        flow: UPDATE ... WHERE token_hash=? AND revoked=False RETURNING *.
        Exactly one concurrent request will match and get the token record; any
        others will get None (token already revoked by the winner).

        Returns:
            RefreshToken — the token was found and is now marked revoked.
            None — token not found OR was already revoked (caller should reject).
        """
        result = await self._db.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash, RefreshToken.revoked.is_(False))
            .values(revoked=True)
            .returning(RefreshToken)
        )
        return result.scalar_one_or_none()

    async def revoke(self, token_id: str) -> None:
        """Mark a single refresh token as revoked."""
        await self._db.execute(
            update(RefreshToken).where(RefreshToken.id == token_id).values(revoked=True)
        )

    async def revoke_all_for_user(self, user_id: str) -> None:
        """Revoke all active refresh tokens belonging to a user."""
        await self._db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False))
            .values(revoked=True)
        )
