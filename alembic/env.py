"""Alembic environment — wired to the async SQLAlchemy engine from src.api.db.engine."""

from __future__ import annotations

import asyncio
import os
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
import src.api.db.models  # noqa: E402, F401 — registers model classes on Base.metadata
from src.api.db.engine import Base  # noqa: E402

target_metadata = Base.metadata

# Database URL: prefer env var over alembic.ini
_DB_URL: str = os.environ.get("COGTRIX_DB_URL", "sqlite+aiosqlite:///./data/api/cogtrix.db")


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
