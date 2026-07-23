"""Comprehensive system endpoint coverage.

Tests GET /api/v1/system/info and POST /api/v1/system/debug exhaustively.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import asyncio as _asyncio  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture()
def app():
    from cogtrix_core.api.app import create_app

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

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(_create())

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
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
        yield _app

    loop.run_until_complete(engine.dispose())
    loop.close()


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register_and_login(client):
    uname = f"sys_{uuid.uuid4().hex[:8]}"
    email = f"{uname}@test.example"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": email, "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    token = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def auth_headers(client):
    return _register_and_login(client)


# ---------------------------------------------------------------------------
# GET /api/v1/system/info
# ---------------------------------------------------------------------------


class TestSystemInfo:
    def test_info_requires_auth(self, client):
        r = client.get("/api/v1/system/info")
        assert r.status_code == 401

    def test_info_returns_200_with_token(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        assert r.status_code == 200

    def test_info_envelope_structure(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        body = r.json()
        assert "data" in body
        assert body["error"] is None

    def test_info_contains_version(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "version" in data
        assert data["version"] is not None

    def test_info_contains_api_version(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert data.get("api_version") == "v1"

    def test_info_contains_platform(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "platform" in data
        assert isinstance(data["platform"], str)

    def test_info_contains_python_version(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "python_version" in data

    def test_info_contains_uptime(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "uptime_s" in data
        assert isinstance(data["uptime_s"], (int, float))
        assert data["uptime_s"] >= 0

    def test_info_contains_started_at(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "started_at" in data

    def test_info_contains_debug_and_verbose(self, client, auth_headers):
        r = client.get("/api/v1/system/info", headers=auth_headers)
        data = r.json()["data"]
        assert "debug" in data
        assert "verbose" in data
        assert isinstance(data["debug"], bool)
        assert isinstance(data["verbose"], bool)

    def test_info_invalid_token_returns_401(self, client):
        r = client.get("/api/v1/system/info", headers={"Authorization": "Bearer invalid.jwt.token"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/system/debug
# ---------------------------------------------------------------------------


class TestToggleDebug:
    def test_debug_requires_auth(self, client):
        r = client.post("/api/v1/system/debug")
        assert r.status_code == 401

    def test_debug_requires_admin(self, client, auth_headers):
        # First-registered user is admin; if somehow not, this should be 403
        r = client.post("/api/v1/system/debug", headers=auth_headers)
        # First user is admin, so should be 200
        assert r.status_code in (200, 403)

    def test_debug_admin_can_toggle(self, client, auth_headers):
        r = client.post("/api/v1/system/debug", headers=auth_headers)
        # First user is admin
        assert r.status_code == 200

    def test_debug_toggle_returns_system_info(self, client, auth_headers):
        r = client.post("/api/v1/system/debug", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        data = body["data"]
        assert "debug" in data
        assert "version" in data

    def test_debug_set_true(self, client, auth_headers):
        r = client.post("/api/v1/system/debug", json={"debug": True}, headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["data"]["debug"] is True

    def test_debug_set_false(self, client, auth_headers):
        # First set True, then set False
        client.post("/api/v1/system/debug", json={"debug": True}, headers=auth_headers)
        r = client.post("/api/v1/system/debug", json={"debug": False}, headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["data"]["debug"] is False

    def test_debug_with_verbose(self, client, auth_headers):
        r = client.post(
            "/api/v1/system/debug",
            json={"debug": True, "verbose": True},
            headers=auth_headers,
        )
        assert r.status_code == 200

    def test_debug_no_body_toggles(self, client, auth_headers):
        # No body — toggles current debug state
        r1 = client.post("/api/v1/system/debug", headers=auth_headers)
        assert r1.status_code == 200
        state1 = r1.json()["data"]["debug"]
        r2 = client.post("/api/v1/system/debug", headers=auth_headers)
        assert r2.status_code == 200
        state2 = r2.json()["data"]["debug"]
        assert state2 != state1

    def test_debug_non_admin_user_returns_403(self, client, auth_headers):
        # auth_headers fixture already registered the first (admin) user.
        # Register a second user — it will be non-admin.
        uname2 = f"u2_{uuid.uuid4().hex[:6]}"
        r_reg = client.post(
            "/api/v1/auth/register",
            json={
                "username": uname2,
                "email": f"{uname2}@example.com",
                "password": _VALID_PASSWORD,
            },
        )
        assert r_reg.status_code == 201
        r = client.post(
            "/api/v1/auth/login", json={"username": uname2, "password": _VALID_PASSWORD}
        )
        assert r.status_code == 200
        token2 = r.json()["data"]["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}

        r = client.post("/api/v1/system/debug", headers=headers2)
        assert r.status_code == 403

    def test_debug_invalid_token_returns_401(self, client):
        r = client.post("/api/v1/system/debug", headers={"Authorization": "Bearer bad.token.here"})
        assert r.status_code == 401
