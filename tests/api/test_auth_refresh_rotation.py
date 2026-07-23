"""Regression tests for refresh token rotation TOCTOU (issue #952).

Covers:
  - RefreshTokenRepository.rotate_and_get(): atomic check-and-revoke
  - Concurrent refresh requests with the same token: exactly one succeeds
  - rotate_and_get() returns None when token already revoked (not found or used)
  - rotate_and_get() returns token record when token is valid (and marks it revoked)
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from src.api.db.repositories.tokens import RefreshTokenRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token_record(
    token_id: str | None = None,
    user_id: str = "user-1",
    token_hash: str | None = None,
    revoked: bool = False,
    expires_at: datetime | None = None,
):
    """Return a RefreshToken-like MagicMock."""
    record = MagicMock()
    record.id = token_id or str(uuid.uuid4())
    record.user_id = user_id
    record.token_hash = token_hash or hashlib.sha256(b"test-token").hexdigest()
    record.revoked = revoked
    record.expires_at = expires_at or (datetime.now(UTC) + timedelta(days=7))
    return record


# ---------------------------------------------------------------------------
# rotate_and_get — unit tests (mock-based)
# ---------------------------------------------------------------------------


class TestRotateAndGetUnit:
    """Test rotate_and_get() with mocked database session."""

    def _build_mock_result(self, returned_record: MagicMock | None):
        """Build a mock sqlalchemy result object for .execute()."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=returned_record)
        return mock_result

    def _build_mock_db(self, result: MagicMock):
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=result)
        return mock_db

    @pytest.mark.asyncio
    async def test_returns_token_when_valid_and_revokes(self):
        """rotate_and_get() returns the token and marks it as revoked in the DB."""
        token = _make_token_record(token_id="token-1", token_hash="abc123")
        mock_result = self._build_mock_result(token)
        mock_db = self._build_mock_db(mock_result)

        repo = RefreshTokenRepository(mock_db)
        result = await repo.rotate_and_get("abc123")

        assert result is token
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        compiled = call_args[0][0]  # first positional arg = compiled SQLAlchemy statement
        # Verify the statement has the revoked=False condition
        assert "revoked" in str(compiled).lower()

    @pytest.mark.asyncio
    async def test_returns_none_when_token_not_found(self):
        """rotate_and_get() returns None when no matching token exists."""
        mock_result = self._build_mock_result(None)
        mock_db = self._build_mock_db(mock_result)

        repo = RefreshTokenRepository(mock_db)
        result = await repo.rotate_and_get("nonexistent-hash")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_token_already_revoked(self):
        """rotate_and_get() returns None when the token was already revoked.

        This is the critical TOCTOU scenario: a second concurrent request that
        arrives after the first request has already rotated the token should
        receive None and be rejected — not succeed with a second rotation.
        """
        mock_result = self._build_mock_result(None)  # RETURNING matches zero rows
        mock_db = self._build_mock_db(mock_result)

        repo = RefreshTokenRepository(mock_db)
        result = await repo.rotate_and_get("already-rotated-hash")

        assert result is None


# ---------------------------------------------------------------------------
# rotate_and_get — concurrent integration test (real SQLite session)
# ---------------------------------------------------------------------------


class TestRotateAndGetConcurrent:
    """Test that concurrent rotate_and_get() calls correctly serialise via DB.

    Uses a real in-memory SQLite session to exercise the actual UPDATE ... RETURNING
    SQL. SQLite serialises writes at the database level, so this test verifies the
    SQL statement is correct and that exactly one of two concurrent calls succeeds.
    """

    @pytest.fixture(autouse=True)
    def _setup_db(self):
        """Create an in-memory SQLite database with a RefreshToken table."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def _init():
            async with engine.begin() as conn:
                # Create a minimal refresh_tokens table matching the real schema
                await conn.execute(text("""
                        CREATE TABLE refresh_tokens (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            token_hash TEXT NOT NULL UNIQUE,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL,
                            revoked BOOLEAN NOT NULL DEFAULT 0
                        )
                        """))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_init())

        self._engine = engine
        self._factory = factory
        self._loop = loop
        yield
        loop.run_until_complete(engine.dispose())
        loop.close()

    @pytest.mark.asyncio
    async def test_concurrent_rotate_only_one_succeeds(self):
        """Exactly one of two concurrent rotate_and_get() calls succeeds.

        This reproduces the TOCTOU window: before the fix, both calls could pass
        the SELECT check and both proceed to revoke, allowing double use.
        With rotate_and_get(), the conditional UPDATE serialises access — exactly
        one call matches the non-revoked row and succeeds; the other gets None.
        """
        token_hash = hashlib.sha256(b"shared-test-token").hexdigest()
        token_id = str(uuid.uuid4())

        # Insert a valid, non-revoked token
        async with self._factory() as session:
            await session.execute(
                text("""
                    INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, revoked)
                    VALUES (:id, :user_id, :token_hash, :expires_at, 0)
                    """),
                {
                    "id": token_id,
                    "user_id": "user-concurrent",
                    "token_hash": token_hash,
                    "expires_at": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
                },
            )
            await session.commit()

        # Fire two concurrent rotate_and_get() calls
        async def call_rotate():
            async with self._factory() as session:
                repo = RefreshTokenRepository(session)
                return await repo.rotate_and_get(token_hash)

        results = await asyncio.gather(call_rotate(), call_rotate())

        # Exactly one succeeded, one got None
        successes = [r for r in results if r is not None]
        failures = [r for r in results if r is None]

        assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
        assert len(failures) == 1, f"Expected exactly 1 failure, got {len(failures)}"

        # The successful result should have the right token hash and user
        assert successes[0].token_hash == token_hash
        assert successes[0].user_id == "user-concurrent"
        assert successes[0].revoked is True  # rotate_and_get marks it revoked in the UPDATE

    @pytest.mark.asyncio
    async def test_rotate_and_get_idempotent_on_missing_token(self):
        """Both concurrent calls return None when the token doesn't exist."""
        nonexistent_hash = hashlib.sha256(b"never-existed-token").hexdigest()

        async def call_rotate():
            async with self._factory() as session:
                repo = RefreshTokenRepository(session)
                return await repo.rotate_and_get(nonexistent_hash)

        results = await asyncio.gather(call_rotate(), call_rotate())

        assert all(r is None for r in results), "Both should return None for missing token"


# ---------------------------------------------------------------------------
# auth.py refresh endpoint — usage verification
# ---------------------------------------------------------------------------


class TestRefreshEndpointUsesAtomicRotation:
    """Verify the refresh endpoint uses rotate_and_get instead of get_by_hash + revoke."""

    def test_refresh_endpoint_no_longer_uses_separate_revoke(self):
        """The refresh() function must not call get_by_hash() + revoke() separately.

        This test guards against regression: if someone refactors refresh() back to
        the non-atomic pattern, this test will fail.
        """
        import ast
        from pathlib import Path

        auth_path = Path(__file__).parent.parent.parent / "src" / "api" / "routes" / "auth.py"
        source = auth_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the refresh() function
        refresh_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "refresh":
                refresh_func = node
                break

        assert refresh_func is not None, "refresh() function not found in auth.py"

        # Collect all attribute accesses and calls
        calls_in_refresh = []
        for node in ast.walk(refresh_func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls_in_refresh.append((node.func.attr, getattr(node.func, "value", None)))

        # Build a set of method names called on token_repo
        token_repo_calls = set()
        for attr, value in calls_in_refresh:
            if isinstance(value, ast.Name) and value.id == "token_repo":
                token_repo_calls.add(attr)

        # rotate_and_get must be used; get_by_hash + revoke separate pattern must NOT appear
        assert "rotate_and_get" in token_repo_calls, (
            "refresh() must call token_repo.rotate_and_get() for atomic rotation. "
            f"token_repo calls found: {token_repo_calls}"
        )
        assert not ("get_by_hash" in token_repo_calls and "revoke" in token_repo_calls), (
            "refresh() must not use token_repo.get_by_hash() + revoke() separately — "
            "that pattern has a TOCTOU window (issue #952). Use rotate_and_get() instead."
        )
