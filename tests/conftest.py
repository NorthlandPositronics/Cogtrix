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

# pytest-xdist (-n auto) creates multi-threaded worker processes.
# test_python_exec.py uses multiprocessing.Process with the default
# fork start method, but fork() is unsafe in a multi-threaded parent
# (Python emits DeprecationWarning, and thread inheritance can cause
# deadlocks).  Force spawn before any test imports multiprocessing.
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

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


@pytest.fixture(autouse=True)
def _reset_secret_env_cache() -> None:
    """Clear the process-level secret-env cache before every test (#2233).

    ``src.config`` caches secret env values so they survive re-resolution after
    the #2223 unset. That cache is process-global, so without this reset a key
    set via ``monkeypatch.setenv`` in one test would leak into later tests (e.g.
    flipping provider-default assertions). Mirrors the rate-limit reset above.
    """
    try:
        from src.config import _reset_secret_env_cache as _reset

        _reset()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(autouse=True)
def _reset_cached_config() -> None:
    """Drop the process-wide cached Config before every test (#2101).

    ``get_cached_config()`` resolves config once and caches it process-globally so
    the environment is read a single time. Without a per-test reset, the config a
    test resolves (under its own ``monkeypatch.setenv`` / temp config file) would
    leak into later tests via the cache. Mirrors the secret-env reset above.
    """
    try:
        from src.config import reset_cached_config

        reset_cached_config()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(autouse=True)
def _restore_required_env():
    """Keep the globally-required JWT secret present for every test (#2102).

    ``load_config()`` (default ``unset_secrets=True``) now ALSO unsets
    ``COGTRIX_JWT_SECRET`` from ``os.environ`` after reading it (the #2102
    hardening). It is test infrastructure relied on by the whole API suite (JWT
    signing/validation). Crucially, the module-scoped ``app`` fixture
    (``tests/api/conftest.py``) creates the app once — unsetting the var — while a
    function-scoped ``client`` re-enters the FastAPI lifespan per test and
    re-reads the secret; combined with the per-test secret-cache reset above, the
    secret would be missing at lifespan startup. Re-assert it both BEFORE (so each
    lifespan entry resolves it from ``os.environ``) and AFTER each test.
    ``COGTRIX_DB_URL`` is intentionally NOT unset by the loader, so it needs no
    restoration here. The production unset behaviour is still verified directly in
    ``tests/test_env_unset_jwt_db_2102.py``.
    """
    os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
    yield
    os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")


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


# ---------------------------------------------------------------------------
# Shared API app/client fixtures (hoisted from tests/api/conftest.py for #2211)
#
# pytest 9.1 resolves fixture overrides by collection-tree visibility, and the
# API CI shards feed a mixed-directory file list (tests/api/, tests/test_api_*,
# tests/regression/) into one ``-n auto --dist=loadfile`` xdist run. A per-subdir
# ``tests/api/conftest.py`` then stops resolving ``app``/``client`` for the
# ``tests/api/`` files inside that mixed run ("fixture 'client' not found"),
# which blocked the pytest 9.1.1 bump (#2196). Defining them at the top level
# makes them visible to every shard file regardless of directory. Files that
# ship their own ``app``/``client`` (the ``tests/test_api_*`` suites and
# ``test_self_improving_loop_features``) override these by proximity and are
# unaffected. Imports stay lazy so a non-API run never pulls in FastAPI just for
# these.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """FastAPI app backed by a fresh in-memory SQLite database.

    Scoped at module level to avoid the ~24s per-test startup cost of creating a
    new engine, all tables, and the FastAPI app. A single app + DB is shared
    across all tests in the module; each test still runs inside a TestClient
    request context with dependency-overridden ``get_db`` sessions, so
    request-scoped state stays clean. Modules needing stricter isolation can
    override this with ``scope="class"``/``"function"`` in their own conftest.
    """
    from unittest.mock import patch

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from src.api.app import create_app
    from src.api.db.engine import Base, get_db

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
    """TestClient wrapping the shared module-scoped ``app`` fixture."""
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
