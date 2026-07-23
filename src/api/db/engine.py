"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

The database URL is read from the ``COGTRIX_DB_URL`` environment variable.
Defaults to ``sqlite+aiosqlite:///./data/api/cogtrix.db`` for development.

In production, set ``COGTRIX_DB_URL`` to a ``postgresql+asyncpg://`` URL.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite+aiosqlite:///./data/api/cogtrix.db"

_DB_URL: str = os.environ.get("COGTRIX_DB_URL", _DEFAULT_DB_URL)

# Ensure the parent directory exists for the default SQLite path
if _DB_URL == _DEFAULT_DB_URL:
    Path("data/api").resolve().mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_connect_args: dict = {}
if _DB_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_async_engine(
    _DB_URL,
    echo=False,
    connect_args=_connect_args,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Yield an async database session, closing it on exit.

    Usage in FastAPI endpoints::

        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
