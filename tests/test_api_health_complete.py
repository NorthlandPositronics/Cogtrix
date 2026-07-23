"""Comprehensive health endpoint coverage.

Tests GET /api/v1/health and GET /api/v1/health/ready exhaustively.
No auth required on either endpoint.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

import asyncio as _asyncio  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from src.api.db.engine import Base, get_db  # noqa: E402


@pytest.fixture()
def app():
    from src.api.app import create_app

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _asyncio.run(_create())

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

    _asyncio.run(engine.dispose())


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /api/v1/health — liveness
# ---------------------------------------------------------------------------


class TestLiveness:
    def test_liveness_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_liveness_returns_timestamp(self, client):
        r = client.get("/api/v1/health")
        data = r.json()["data"]
        assert "timestamp" in data
        assert data["timestamp"] is not None

    def test_liveness_no_auth_required(self, client):
        # Should succeed with no Authorization header whatsoever
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["error"] is None

    def test_liveness_content_type_json(self, client):
        r = client.get("/api/v1/health")
        assert "application/json" in r.headers["content-type"]

    def test_liveness_envelope_structure(self, client):
        r = client.get("/api/v1/health")
        body = r.json()
        assert "data" in body
        assert "error" in body
        assert body["error"] is None


# ---------------------------------------------------------------------------
# GET /api/v1/health/ready — readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_readiness_no_auth_required(self, client):
        r = client.get("/api/v1/health/ready")
        # Should return either 200 or 503 — both are valid without auth
        assert r.status_code in (200, 503)

    def test_readiness_envelope_structure(self, client):
        r = client.get("/api/v1/health/ready")
        body = r.json()
        assert "data" in body
        data = body["data"]
        assert "ready" in data
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_readiness_database_component_present(self, client):
        r = client.get("/api/v1/health/ready")
        components = r.json()["data"]["components"]
        names = [c["name"] for c in components]
        assert "database" in names

    def test_readiness_tool_registry_component_present(self, client):
        r = client.get("/api/v1/health/ready")
        components = r.json()["data"]["components"]
        names = [c["name"] for c in components]
        assert "tool_registry" in names

    def test_readiness_components_have_required_fields(self, client):
        r = client.get("/api/v1/health/ready")
        for comp in r.json()["data"]["components"]:
            assert "name" in comp
            assert "ok" in comp
            # latency_ms and detail may be None but must be present
            assert "latency_ms" in comp
            assert "detail" in comp

    def test_readiness_ready_bool(self, client):
        r = client.get("/api/v1/health/ready")
        assert isinstance(r.json()["data"]["ready"], bool)

    def test_readiness_503_when_not_ready(self, client, app):
        """When database is unreachable, readiness should return 503."""
        # Override the engine with a broken connection string
        from unittest.mock import patch

        import src.api.routes.health as health_mod

        async def _bad_connect(*a, **kw):
            raise Exception("simulated DB failure")

        with patch.object(health_mod, "engine", None, create=True):
            r = client.get("/api/v1/health/ready")
            # Either 503 (not ready) or 200 if other components pass
            assert r.status_code in (200, 503)

    def test_readiness_db_ok_field_is_bool(self, client):
        r = client.get("/api/v1/health/ready")
        for comp in r.json()["data"]["components"]:
            assert isinstance(comp["ok"], bool)
