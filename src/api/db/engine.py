"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

The database URL is resolved in priority order:

1. ``COGTRIX_DB_URL`` environment variable — full URL, takes precedence over everything.
2. ``COGTRIX_DATA_DIR`` environment variable — relocates the SQLite file to
   ``<COGTRIX_DATA_DIR>/api/cogtrix.db`` (same env var read by ``src/config.py``).
3. ``data_dir`` from the Cogtrix config file — read at first engine access so
   that ``data_dir: /data/cogtrix`` in ``.cogtrix.yaml`` puts the DB at
   ``/data/cogtrix/api/cogtrix.db`` without requiring any env var.
4. Built-in default ``./data/api/cogtrix.db`` (relative to the working directory).

For PostgreSQL in production set ``COGTRIX_DB_URL`` to a
``postgresql+asyncpg://`` URL.

Import-time side effects: none.  Config read, parent-directory ``mkdir``, and
engine construction are all deferred to first access of ``engine`` /
``AsyncSessionLocal`` (PEP 562 module ``__getattr__``).  This keeps
``import src.api.db.engine`` safe in read-only environments, doc generators,
and import-graph tools.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy state — engine and session factory resolved on first attribute access
# via __getattr__ below.  The DB URL is special: ``COGTRIX_DB_URL`` is
# captured AT MODULE IMPORT TIME (cheap read of os.environ — no filesystem
# access), matching pre-refactor behavior.  Tests rely on that early capture
# so the engine binds to :memory: (set by tests/conftest.py before any test
# module is imported) instead of capturing a file-backed URL that a
# later-collected test module overrides (e.g. tests/test_api_phase3.py).
# Only the config-file fallback path (when no env var is set) is deferred —
# it's the only branch that actually performs filesystem I/O.
# ---------------------------------------------------------------------------

_db_url: str | None = os.environ.get("COGTRIX_DB_URL") or None
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


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


def _get_db_url() -> str:
    """Return the resolved DB URL, computing the config-file fallback lazily."""
    global _db_url
    if _db_url is None:
        _db_url = _resolve_default_db_url()
    return _db_url


def _connect_args_for(url: str) -> dict[str, Any]:
    """Return SQLAlchemy ``connect_args`` for the given URL.

    ``check_same_thread`` is a SQLite-only kwarg; passing it to asyncpg
    raises TypeError.  Kept as a top-level helper so tests can validate the
    dialect mapping without building a full engine.
    """
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def _build_engine() -> AsyncEngine:
    url = _get_db_url()

    # Ensure the parent directory exists for any SQLite URL.  Done here, not
    # at module import, so a read-only filesystem doesn't break ``import``.
    if url.startswith("sqlite"):
        Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)

    return create_async_engine(url, echo=False, connect_args=_connect_args_for(url))


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


def __getattr__(name: str) -> Any:
    # PEP 562: makes ``from src.api.db.engine import engine, AsyncSessionLocal``
    # work without paying the construction cost at import time.
    if name == "engine":
        return _get_engine()
    if name == "AsyncSessionLocal":
        return _get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models) — must be a real class.
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


# ---------------------------------------------------------------------------
# Connection validation (called from lifespan)
# ---------------------------------------------------------------------------


def _sanitize_url(url: str) -> str:
    """Remove password from DB URL for safe logging/error messages."""
    return re.sub(r":[^:@/]+@", ":***@", url)


async def validate_connection() -> None:
    """Test the database connection at startup.

    Raises a clear, operator-actionable ``RuntimeError`` if the connection
    fails — rather than letting a cryptic SQLAlchemy traceback surface on
    the first request.

    Called from the FastAPI lifespan context manager after table creation.
    """
    url = _get_db_url()
    try:
        async with _get_engine().connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception as exc:
        if "postgresql" in url:
            hint = (
                f"Cannot connect to PostgreSQL at {_sanitize_url(url)}. "
                "Check: (1) DB server is running, (2) COGTRIX_DB_URL is correct, "
                "(3) asyncpg is installed: pip install 'cogtrix[postgresql]', "
                f"(4) user has CONNECT privilege. Original error: {exc}"
            )
        else:
            hint = (
                f"Cannot connect to SQLite at {_sanitize_url(url)}. "
                f"Check: directory exists and is writable. Original error: {exc}"
            )
        raise RuntimeError(hint) from exc


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Yield an async database session, closing it on exit.

    Usage in FastAPI endpoints::

        async def my_endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = _get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
