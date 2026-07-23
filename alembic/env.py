"""Alembic environment — wired to the async SQLAlchemy engine from cogtrix_core.api.db.engine."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Alembic Config object
config = context.config

# Set up loggers from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so their tables appear in target_metadata
import cogtrix_core.api.db.models  # noqa: E402, F401 — registers model classes on Base.metadata
from cogtrix_core.api.db.engine import Base, _get_db_url  # noqa: E402

target_metadata = Base.metadata

# Database URL: single source of truth with the API runtime. ``_get_db_url``
# resolves the priority chain (#1877):
#   1. ``COGTRIX_DB_URL`` env var
#   2. ``COGTRIX_DATA_DIR`` env var → ``<DATA_DIR>/api/cogtrix.db``
#   3. ``data_dir`` from the Cogtrix config file
#   4. Built-in default ``./data/api/cogtrix.db``
# The previous direct ``os.environ.get(..., "./data/api/cogtrix.db")``
# fallback only honoured (1) and (4), so when the Docker image set
# ``COGTRIX_DATA_DIR=/data`` (the canonical mount point) without
# ``COGTRIX_DB_URL`` the API runtime resolved to ``/data/api/cogtrix.db``
# while Alembic resolved to ``./data/api/cogtrix.db`` relative to
# ``WORKDIR=/app`` — i.e. a non-writable ``/app/data/api/`` — and the
# migration failed with ``sqlite3.OperationalError: unable to open
# database file`` before Uvicorn could start.
_DB_URL: str = _get_db_url()


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emit SQL to stdout)."""
    context.configure(
        url=_DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[type-arg]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations online using the async engine."""
    connectable = create_async_engine(_DB_URL, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
