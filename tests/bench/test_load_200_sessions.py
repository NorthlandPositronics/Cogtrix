"""Load test: 200 concurrent API sessions — p95 < 5 s.

Design
------
* Fixture (module-scope) — registers 200 users *sequentially* with bcrypt
  mocked to a fast SHA-256 stub so setup does not dominate the benchmark.
  Collects one access token per user.
* Benchmark test — fires all 200 GET /auth/me calls simultaneously using a
  threading.Barrier so every thread starts its HTTP call at the same instant.
  Measures p50 / p95 / p99 response latency and asserts p95 < 5 s.

No live LLM or network I/O is required — this exercises the FastAPI /
SQLAlchemy / aiosqlite stack only.

Run:
    uv run pytest tests/bench/test_load_200_sessions.py -v -s -m benchmark
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import os
import statistics
import threading
import time
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db.engine import Base, get_db  # noqa: E402

CONCURRENT_SESSIONS = 200
P95_THRESHOLD_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Fast password stub — replaces bcrypt so setup does not take 20+ seconds
# ---------------------------------------------------------------------------


def _fast_hash(plaintext: str) -> str:
    """SHA-256 stub: same interface as hash_password, bcrypt-free (test only)."""
    return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()


def _fast_verify(plaintext: str, hashed: str) -> bool:
    return hashed == _fast_hash(plaintext)


# ---------------------------------------------------------------------------
# Module-scoped fixture: one app + in-memory DB + 200 pre-seeded tokens
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def session_tokens():
    """
    Spin up the app, register 200 unique users (bcrypt mocked), and return
    a list of 200 Bearer access tokens to use in the benchmark.
    """
    from src.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())

    _app = create_app()

    # Disable rate limiting so fixture can register all 200 users sequentially.
    if hasattr(_app.state, "limiter"):
        _app.state.limiter.enabled = False

    async def _override():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    _app.dependency_overrides[get_db] = _override

    tokens: list[str] = []

    with (
        patch("src.api.routes.auth.hash_password", side_effect=_fast_hash),
        patch("src.api.routes.auth.verify_password", side_effect=_fast_verify),
        TestClient(_app, raise_server_exceptions=False) as client,
    ):
        for i in range(CONCURRENT_SESSIONS):
            resp = client.post(
                "/api/v1/auth/register",
                json={
                    "username": f"loaduser{i:04d}",
                    "email": f"loaduser{i:04d}@loadtest.example",
                    "password": "Password1!",
                },
            )
            assert resp.status_code == 201, f"Setup failed for user {i}: {resp.text}"
            tokens.append(resp.json()["data"]["access_token"])

        yield client, tokens

    asyncio.run(engine.dispose())


# ---------------------------------------------------------------------------
# Benchmark: 200 concurrent GET /auth/me requests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.timeout(60)
def test_200_concurrent_sessions_p95(session_tokens: tuple) -> None:
    """200 simultaneous authenticated requests; assert p95 response time < 5 s."""
    client, tokens = session_tokens

    durations: list[float] = [0.0] * CONCURRENT_SESSIONS
    errors: list[Exception | None] = [None] * CONCURRENT_SESSIONS

    # Barrier ensures all threads fire their HTTP call at the same instant.
    barrier = threading.Barrier(CONCURRENT_SESSIONS + 1)

    def call_me(i: int) -> None:
        barrier.wait()
        t0 = time.monotonic()
        try:
            client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {tokens[i]}"},
            )
        except Exception as exc:
            errors[i] = exc
        finally:
            durations[i] = time.monotonic() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_SESSIONS) as pool:
        futures = [pool.submit(call_me, i) for i in range(CONCURRENT_SESSIONS)]
        barrier.wait()  # release all workers simultaneously
        concurrent.futures.wait(futures)

    for i, exc in enumerate(errors):
        if exc is not None:
            raise RuntimeError(f"Session {i} raised: {exc}") from exc

    sorted_d = sorted(durations)
    p50 = statistics.median(sorted_d)
    p95_idx = max(0, int(len(sorted_d) * 0.95) - 1)
    p99_idx = max(0, int(len(sorted_d) * 0.99) - 1)
    p95 = sorted_d[p95_idx]
    p99 = sorted_d[p99_idx]

    print(
        f"\nLoad test: {CONCURRENT_SESSIONS} concurrent sessions (GET /auth/me)"
        f"\n  p50={p50:.3f}s  p95={p95:.3f}s  p99={p99:.3f}s"
        f"  max={max(durations):.3f}s  min={min(durations):.3f}s"
    )

    assert p95 < P95_THRESHOLD_SECONDS, f"p95={p95:.3f}s exceeds {P95_THRESHOLD_SECONDS}s threshold"
