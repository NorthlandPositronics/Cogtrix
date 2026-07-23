"""Comprehensive memory endpoint coverage.

Tests all 3 memory endpoints:
  GET    /api/v1/sessions/{id}/memory  — get memory state
  DELETE /api/v1/sessions/{id}/memory  — clear memory
  PATCH  /api/v1/sessions/{id}/memory  — switch memory mode

Focuses on additional cases beyond test_api_sessions_complete.py:
  - Full MemoryStateOut field validation
  - Mode switching with valid modes (conversation, code, reasoning)
  - Unknown mode returns 422
  - Memory clear with no memory_manager set
  - token_counts fallback when not present
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

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


def _reg(client):
    uname = f"mem_{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@example.com", "password": _VALID_PASSWORD},
    )
    r = client.post("/api/v1/auth/login", json={"username": uname, "password": _VALID_PASSWORD})
    return r.json()["data"]["access_token"]


def _h(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def tokens(client):
    owner = _reg(client)
    other = _reg(client)
    return {"owner": owner, "other": other}


@pytest.fixture()
def sid(client, tokens):
    r = client.post("/api/v1/sessions", headers=_h(tokens["owner"]), json={})
    return r.json()["data"]["id"]


def _make_live_session(mode="conversation", window_size=10, summary="test summary"):
    mm = MagicMock()
    mm.to_dict.return_value = {
        "mode": mode,
        "window_size": window_size,
        "summarized_messages": 5,
        "tokens_used": 1024,
        "summary": summary,
        "vector_recall_enabled": False,
        "mode_meta": {},
    }
    mm.clear = MagicMock()
    mm.save = MagicMock()
    mm.load = MagicMock()

    live = MagicMock()
    live.memory_manager = mm
    live.config = {"memory_mode": mode}
    live.token_counts = {"context_window": 8192, "input_tokens": 100, "output_tokens": 50}
    # turn_lock must support async context manager protocol
    live.turn_lock = MagicMock()
    live.turn_lock.__aenter__ = AsyncMock(return_value=None)
    live.turn_lock.__aexit__ = AsyncMock(return_value=None)
    return live


def _mock_registry(app, live_session):
    mock_reg = MagicMock()
    mock_reg.get_or_warm = AsyncMock(return_value=live_session)
    app.state.session_registry = mock_reg


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{id}/memory
# ---------------------------------------------------------------------------


class TestGetMemory:
    def test_get_no_auth_returns_401(self, client, sid):
        r = client.get(f"/api/v1/sessions/{sid}/memory")
        assert r.status_code == 401

    def test_get_non_owner_returns_403(self, client, tokens, sid):
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_get_nonexistent_session_returns_404_or_403(self, client, tokens, app):
        fake_sid = str(uuid.uuid4())
        mock_reg = MagicMock()
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg
        r = client.get(f"/api/v1/sessions/{fake_sid}/memory", headers=_h(tokens["owner"]))
        # 403 from verify_session_owner (ownership check) or 404
        assert r.status_code in (403, 404)

    def test_get_returns_200_with_live_session(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.status_code == 200

    def test_get_envelope_structure(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        body = r.json()
        assert "data" in body
        assert body["error"] is None

    def test_get_returns_mode_field(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.json()["data"]["mode"] == "conversation"

    def test_get_returns_all_required_fields(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        data = r.json()["data"]
        required = [
            "session_id",
            "mode",
            "summary",
            "window_messages",
            "summarized_messages",
            "tokens_used",
            "context_window",
            "vector_recall_enabled",
            "mode_meta",
            "updated_at",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_get_session_id_matches(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.json()["data"]["session_id"] == sid

    def test_get_context_window_from_token_counts(self, client, tokens, sid, app):
        live = _make_live_session()
        live.token_counts = {"context_window": 32768}
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.json()["data"]["context_window"] == 32768

    def test_get_context_window_default_when_none(self, client, tokens, sid, app):
        live = _make_live_session()
        live.token_counts = None
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        # Should default to 131072
        assert r.json()["data"]["context_window"] == 131072

    def test_get_summary_field_present(self, client, tokens, sid, app):
        live = _make_live_session(summary="earlier conversation summary")
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.json()["data"]["summary"] == "earlier conversation summary"

    def test_get_no_memory_manager_returns_200(self, client, tokens, sid, app):
        live = MagicMock()
        live.memory_manager = None
        live.config = {"memory_mode": "conversation"}
        live.token_counts = None
        _mock_registry(app, live)
        r = client.get(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        # Should not crash — mode falls back to "conversation"
        assert r.status_code == 200

    def test_get_invalid_token_returns_401(self, client, sid):
        r = client.get(
            f"/api/v1/sessions/{sid}/memory",
            headers={"Authorization": "Bearer invalid.jwt"},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /api/v1/sessions/{id}/memory
# ---------------------------------------------------------------------------


class TestClearMemory:
    def test_clear_no_auth_returns_401(self, client, sid):
        r = client.delete(f"/api/v1/sessions/{sid}/memory")
        assert r.status_code == 401

    def test_clear_non_owner_returns_403(self, client, tokens, sid):
        r = client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_clear_nonexistent_session_returns_404_or_403(self, client, tokens, app):
        fake_sid = str(uuid.uuid4())
        mock_reg = MagicMock()
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg
        r = client.delete(f"/api/v1/sessions/{fake_sid}/memory", headers=_h(tokens["owner"]))
        assert r.status_code in (403, 404)

    def test_clear_returns_200(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.status_code == 200

    def test_clear_returns_null_data(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.json()["data"] is None

    def test_clear_with_no_memory_manager_returns_200(self, client, tokens, sid, app):
        live = MagicMock()
        live.memory_manager = None
        live.config = {}
        live.token_counts = None
        _mock_registry(app, live)
        r = client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        assert r.status_code == 200

    def test_clear_calls_mm_clear(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        # mm.clear should have been called
        live.memory_manager.clear.assert_called_once()

    def test_clear_envelope_structure(self, client, tokens, sid, app):
        live = _make_live_session()
        _mock_registry(app, live)
        r = client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        body = r.json()
        assert "data" in body
        assert "error" in body
        assert body["error"] is None

    def test_clear_acquires_turn_lock(self, client, tokens, sid, app):
        """clear_memory must acquire live_session.turn_lock before mutating state."""
        live = _make_live_session()
        _mock_registry(app, live)
        client.delete(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]))
        live.turn_lock.__aenter__.assert_awaited_once()
        live.turn_lock.__aexit__.assert_awaited_once()


# ---------------------------------------------------------------------------
# PATCH /api/v1/sessions/{id}/memory — switch mode
# ---------------------------------------------------------------------------


class TestSwitchMemoryMode:
    def test_switch_no_auth_returns_401(self, client, sid):
        r = client.patch(f"/api/v1/sessions/{sid}/memory", json={"mode": "code"})
        assert r.status_code == 401

    def test_switch_non_owner_returns_403(self, client, tokens, sid):
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["other"]),
            json={"mode": "code"},
        )
        assert r.status_code == 403

    def test_switch_missing_body_returns_422(self, client, tokens, sid):
        r = client.patch(f"/api/v1/sessions/{sid}/memory", headers=_h(tokens["owner"]), json={})
        assert r.status_code == 422

    def test_switch_invalid_mode_returns_422(self, client, tokens, sid):
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "unknown_mode_xyz"},
        )
        assert r.status_code == 422

    def test_switch_nonexistent_session_returns_404_or_403(self, client, tokens, app):
        fake_sid = str(uuid.uuid4())
        mock_reg = MagicMock()
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg
        r = client.patch(
            f"/api/v1/sessions/{fake_sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "code"},
        )
        assert r.status_code in (403, 404)

    def test_switch_to_conversation_mode(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "conversation"},
        )
        assert r.status_code == 200

    def test_switch_to_code_mode(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "code"},
        )
        assert r.status_code == 200

    def test_switch_to_reasoning_mode(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "reasoning"},
        )
        assert r.status_code == 200

    def test_switch_returns_memory_state_out(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "code"},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        # Response should be MemoryStateOut (not null)
        assert data is not None
        assert "mode" in data
        assert "session_id" in data

    def test_switch_updates_config_memory_mode(self, client, tokens, sid, app):
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "reasoning"},
        )
        assert r.status_code == 200
        # config should be updated
        if live.config is not None:
            # The handler sets live.config["memory_mode"] = target_mode
            assert live.config.get("memory_mode") == "reasoning"

    def test_switch_invalid_token_returns_401(self, client, sid):
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers={"Authorization": "Bearer bad.token"},
            json={"mode": "code"},
        )
        assert r.status_code == 401

    def test_switch_acquires_turn_lock(self, client, tokens, sid, app):
        """switch_memory_mode must acquire live_session.turn_lock before replacing memory_manager."""
        live = _make_live_session(mode="conversation")
        _mock_registry(app, live)
        r = client.patch(
            f"/api/v1/sessions/{sid}/memory",
            headers=_h(tokens["owner"]),
            json={"mode": "code"},
        )
        assert r.status_code == 200
        live.turn_lock.__aenter__.assert_awaited_once()
        live.turn_lock.__aexit__.assert_awaited_once()
