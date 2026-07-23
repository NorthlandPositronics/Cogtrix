"""API test fixtures shared across all API test files."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="function")
def app():
    """FastAPI app backed by a fresh in-memory SQLite database."""
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
