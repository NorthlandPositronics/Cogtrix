"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

The database URL is resolved in priority order:

1. ``COGTRIX_DB_URL`` environment variable — full URL, takes precedence over everything.
2. ``COGTRIX_DATA_DIR`` environment variable — relocates the SQLite file to
   ``<COGTRIX_DATA_DIR>/api/cogtrix.db`` (same env var read by ``src/config.py``).
3. ``data_dir`` from the Cogtrix config file — read at startup so that
   ``data_dir: /data/cogtrix`` in ``.cogtrix.yaml`` puts the DB at
   ``/data/cogtrix/api/cogtrix.db`` without requiring any env var.
4. Built-in default ``./data/api/cogtrix.db`` (relative to the working directory).

For PostgreSQL in production set ``COGTRIX_DB_URL`` to a
``postgresql+asyncpg://`` URL.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------


def _resolve_default_db_url() -> str:
    """Return the default SQLite DB URL, honouring data_dir from config."""
    # Priority 1: explicit COGTRIX_DATA_DIR env var
    data_dir_env = os.environ.get("COGTRIX_DATA_DIR", "").strip()
    if data_dir_env:
        return f"sqlite+aiosqlite:///{data_dir_env}/api/cogtrix.db"

    # Priority 2: data_dir from config file
    try:
        from src.config import load_config

        cfg = load_config()
        return f"sqlite+aiosqlite:///{cfg.data_dir}/api/cogtrix.db"
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read data_dir from config, using built-in default: %s", exc)

    # Priority 3: built-in default (relative to CWD)
    return "sqlite+aiosqlite:///./data/api/cogtrix.db"


_DB_URL: str = os.environ.get("COGTRIX_DB_URL") or _resolve_default_db_url()

# Ensure the parent directory exists for any SQLite URL
if _DB_URL.startswith("sqlite"):
    _db_file = Path(_DB_URL.split("///", 1)[-1])
    _db_file.parent.mkdir(parents=True, exist_ok=True)

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
