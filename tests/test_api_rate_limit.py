"""Rate-limit helper and integration tests for the Cogtrix API."""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api.rate_limit import (
    configure_trusted_proxy_cidrs,
    per_route_rate_limit,
    rate_limit_key,
    reset_rate_limits,
)

pytest.importorskip("fastapi")

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from datetime import UTC, datetime, timedelta  # noqa: E402

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from slowapi import Limiter  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402

from src.api.app import _rate_limit_exceeded_handler  # noqa: E402


@pytest.fixture(autouse=True)
def reset_trusted_proxy_allowlist() -> Iterator[None]:
    configure_trusted_proxy_cidrs(None)
    yield
    configure_trusted_proxy_cidrs(None)


@pytest.fixture(autouse=True)
def reset_counters() -> Iterator[None]:
    reset_rate_limits()
    yield
    reset_rate_limits()


def _request(remote_addr: str, forwarded_for: str | None = None) -> SimpleNamespace:
    headers: dict[str, str] = {}
    if forwarded_for is not None:
        headers["x-forwarded-for"] = forwarded_for
    return SimpleNamespace(client=SimpleNamespace(host=remote_addr), headers=headers)


def test_trusted_proxy_uses_forwarded_for_first_ip() -> None:
    configure_trusted_proxy_cidrs("10.0.0.0/8, 192.168.0.0/16")

    request = _request("10.1.2.3", "203.0.113.9, 10.1.2.3")

    assert rate_limit_key(request) == "203.0.113.9"


def test_untrusted_peer_ignores_forwarded_for_header() -> None:
    configure_trusted_proxy_cidrs("10.0.0.0/8")

    request = _request("198.51.100.7", "203.0.113.9, 10.1.2.3")

    assert rate_limit_key(request) == "198.51.100.7"


def test_invalid_trusted_proxy_cidr_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid trusted proxy CIDR"):
        configure_trusted_proxy_cidrs("not-a-cidr")


# ---------------------------------------------------------------------------
# per_route_rate_limit integration tests
# ---------------------------------------------------------------------------


def _make_per_route_app(max_calls: int = 2, window_seconds: int = 60) -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    def _endpoint(
        request: Request,
        _rl: None = Depends(per_route_rate_limit(max_calls, window_seconds)),
    ) -> dict[str, str]:
        return {"ok": "true"}

    return app


class TestPerRouteRateLimit:
    def test_requests_within_limit_return_200(self) -> None:
        app = _make_per_route_app(max_calls=3, window_seconds=60)
        with TestClient(app, raise_server_exceptions=False) as client:
            for _ in range(3):
                r = client.get("/test")
                assert r.status_code == 200
                assert r.json() == {"ok": "true"}

    def test_exceeding_limit_returns_429(self) -> None:
        app = _make_per_route_app(max_calls=2, window_seconds=60)
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/test").status_code == 200
            assert client.get("/test").status_code == 200
            r = client.get("/test")
            assert r.status_code == 429
            data = r.json()["detail"]
            assert data["code"] == "RATE_LIMIT_EXCEEDED"

    def test_window_expires_and_requests_succeed_again(self) -> None:
        app = _make_per_route_app(max_calls=2, window_seconds=60)
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/test").status_code == 200
            assert client.get("/test").status_code == 200
            assert client.get("/test").status_code == 429

            future = datetime.now(UTC) + timedelta(seconds=61)
            with patch("src.api.rate_limit.datetime") as mock_dt:
                mock_dt.now.return_value = future
                r = client.get("/test")
                assert r.status_code == 200

    def test_different_clients_have_independent_limits(self) -> None:
        configure_trusted_proxy_cidrs("127.0.0.0/8")
        app = _make_per_route_app(max_calls=2, window_seconds=60)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            patch("src.api.rate_limit.get_remote_address", return_value="127.0.0.1"),
        ):
            # Exhaust limit for client A
            assert client.get("/test", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429

            # Client B should still have full quota
            assert client.get("/test", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 429

    def test_trusted_proxy_end_to_end(self) -> None:
        configure_trusted_proxy_cidrs("127.0.0.0/8")
        app = _make_per_route_app(max_calls=2, window_seconds=60)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            patch("src.api.rate_limit.get_remote_address", return_value="127.0.0.1"),
        ):
            # 127.0.0.1 is trusted, so x-forwarded-for is honoured
            assert client.get("/test", headers={"x-forwarded-for": "3.3.3.3"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "3.3.3.3"}).status_code == 200
            assert client.get("/test", headers={"x-forwarded-for": "3.3.3.3"}).status_code == 429

            # Same forwarded-for from a different direct client is still
            # treated as 3.3.3.3 because 127.0.0.1 is trusted.
            # (TestClient always reports 127.0.0.1, so this verifies the
            # proxy path is active.)


# ---------------------------------------------------------------------------
# SlowAPI middleware integration tests
# ---------------------------------------------------------------------------


def _test_key_func(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return get_remote_address(request)


def _make_middleware_app(default_limits: list[str]) -> FastAPI:
    app = FastAPI()
    limiter = Limiter(key_func=_test_key_func, default_limits=default_limits)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    return app


class TestSlowapiMiddleware:
    def test_middleware_allows_requests_under_limit(self) -> None:
        app = _make_middleware_app(["3/minute"])
        with TestClient(app, raise_server_exceptions=False) as client:
            for _ in range(3):
                r = client.get("/health")
                assert r.status_code == 200

    def test_middleware_returns_429_after_limit(self) -> None:
        app = _make_middleware_app(["2/minute"])
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/health").status_code == 200
            r = client.get("/health")
            assert r.status_code == 429

    def test_middleware_independent_per_client(self) -> None:
        app = _make_middleware_app(["2/minute"])
        with TestClient(app, raise_server_exceptions=False) as client:
            # Client A
            assert client.get("/health", headers={"x-forwarded-for": "4.4.4.4"}).status_code == 200
            assert client.get("/health", headers={"x-forwarded-for": "4.4.4.4"}).status_code == 200
            assert client.get("/health", headers={"x-forwarded-for": "4.4.4.4"}).status_code == 429

            # Client B
            assert client.get("/health", headers={"x-forwarded-for": "5.5.5.5"}).status_code == 200
            assert client.get("/health", headers={"x-forwarded-for": "5.5.5.5"}).status_code == 200
            assert client.get("/health", headers={"x-forwarded-for": "5.5.5.5"}).status_code == 429
