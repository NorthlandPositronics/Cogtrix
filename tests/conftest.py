"""Root test configuration — fixtures shared across the entire test suite."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator

# Set required environment variables before any src.api module is imported.
os.environ.setdefault(
    "COGTRIX_JWT_SECRET",
    "testsecret_mustbe32chars_minimum00",
)
os.environ.setdefault(
    "COGTRIX_DB_URL",
    "sqlite+aiosqlite:///:memory:",
)

# Install the runtime warning filters BEFORE any langchain/langgraph
# import.  ``_bootstrap`` imports ``langchain_core`` and then installs
# ``ignore`` filters for the upstream PendingDeprecationWarning emitted
# by langgraph's ``Reviver()`` call.  pyproject.toml's
# ``[tool.pytest.ini_options].filterwarnings`` only applies during test
# execution — warnings raised at import/collection time bypass it, so
# the runtime filter from ``_bootstrap`` is what actually suppresses
# the langgraph warning here.
import src._bootstrap  # noqa: F401, E402, I001 — must precede pytest imports

# isort: split — keep ``_bootstrap`` import above ``pytest`` so warning
# filters install before any LangChain/LangGraph import in test collection.
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

# Force-import the engine module so its module-level ``_db_url`` capture (a
# side-effect-free ``os.environ.get`` at import time) snapshots
# COGTRIX_DB_URL while it is still ``:memory:``, BEFORE any test_*.py
# module is collected.  Otherwise a test file collected first that
# hard-overrides COGTRIX_DB_URL at module load (e.g.
# ``tests/test_api_phase3.py:65``) would let the global engine bind to a
# file-backed URL — defeating per-test isolation for any code path that
# bypasses the ``get_db`` dependency override (e.g. ``_api_client()`` in
# test_api_phase5.py and test_api_rag_config_system.py).
import src.api.db.engine  # noqa: F401, E402

# ── Session-scoped environment ──────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _cogtrix_test_env() -> None:
    """Set required environment variables for the test suite.

    Must run before any import that reads these values at module level.
    """
    os.environ.setdefault(
        "COGTRIX_JWT_SECRET",
        "testsecret_mustbe32chars_minimum00",
    )
    os.environ.setdefault(
        "COGTRIX_DB_URL",
        "sqlite+aiosqlite:///:memory:",
    )


@pytest.fixture(autouse=True)
def _reset_rate_limit_counters() -> None:
    """Clear in-memory rate-limit counters before every test.

    Prevents inter-test interference: tests that call the login/refresh/acs
    endpoints many times (e.g. module-scoped fixture suites) would otherwise
    accumulate hits across functions and trigger 429 responses unexpectedly.
    """
    try:
        from src.api.rate_limit import reset_rate_limits

        reset_rate_limits()
    except Exception:  # noqa: BLE001
        pass


# ── Shared database fixtures ────────────────────────────────────────────────


def _make_engine():
    """Return a fresh in-memory SQLite async engine."""
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture()
def engine():
    """Yield a fresh in-memory SQLite engine with tables created.

    Uses sync_engine.dispose() for teardown to ensure the underlying
    aiosqlite worker threads receive the close signal regardless of which
    asyncio event loop owns the connections.  The async dispose path
    (asyncio.run(eng.dispose())) creates a *new* event loop and cannot
    signal threads that were started in the TestClient's anyio portal,
    causing those threads to hang in tx.get() during teardown.
    """
    from src.api.db.engine import Base

    eng = _make_engine()

    async def _setup():
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())
    yield eng
    asyncio.run(eng.dispose())


@pytest.fixture()
def async_session_factory(engine):
    """Yield an async_sessionmaker bound to the test engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture()
def sf(async_session_factory):
    """Alias for the shared async_session_factory."""
    return async_session_factory


@pytest.fixture()
def session_factory(async_session_factory):
    """Alias for the shared async_session_factory."""
    return async_session_factory


@pytest_asyncio.fixture()
async def db_session(engine) -> AsyncGenerator[AsyncSession]:
    """Yield a fresh async session for each test."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
