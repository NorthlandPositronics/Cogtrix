"""API test fixtures shared across all API test files."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="module")
def app():
    """FastAPI app backed by a fresh in-memory SQLite database.

    Scoped at module level to avoid the ~24s per-test startup cost
    of creating a new engine, creating all tables, and building the
    FastAPI app from scratch (function scope was the previous default).
    A single app + DB is shared across all tests in the module,
    reducing the API integration suite from ~28 min to ~10 min.

    Test isolation: each test function runs inside a FastAPI
    TestClient request context with dependency-overridden get_db
    sessions, so request-scoped state is still clean.  The
    _reset_rate_limit_counters autouse fixture (tests/conftest.py)
    runs before every function and clears rate-limit counters.
    Modules that need true isolation between test classes can
    override this fixture with scope="class" or scope="function"
    in their local conftest.
    """
    from src.api.app import create_app
    from src.api.db.engine import Base, get_db

    # Get the current event loop to ensure we're using the same loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop.run_until_complete(_create())

    jwt_secret = os.environ.get("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": jwt_secret}):
        _app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        _app.dependency_overrides[get_db] = _override
        _app.state.test_session_factory = factory
        yield _app

    loop.run_until_complete(engine.dispose())
    loop.close()


@pytest.fixture()
def client(app):
    """TestClient wrapping the shared app fixture."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
