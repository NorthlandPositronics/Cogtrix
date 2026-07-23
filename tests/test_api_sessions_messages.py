"""Comprehensive API tests for session and message endpoints.

Covers:
- POST   /api/v1/sessions              (TestSessionCreate)
- GET    /api/v1/sessions              (TestSessionList)
- GET    /api/v1/sessions/{id}         (TestSessionGet)
- PATCH  /api/v1/sessions/{id}         (TestSessionUpdate)
- DELETE /api/v1/sessions/{id}         (TestSessionDelete)
- POST   /api/v1/sessions/{id}/messages (TestMessageSend)
- GET    /api/v1/sessions/{id}/messages (TestMessageList)
- DELETE /api/v1/sessions/{id}/messages (TestMessageClear)

All tests use an in-memory SQLite database and mock the session registry /
warm_session so no real LLM provider is required.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Environment — must be set before any src.api import
# ---------------------------------------------------------------------------

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from src.api.auth import create_access_token  # noqa: E402
from src.api.db import models as _models  # noqa: E402, F401
from src.api.db.engine import Base  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def http_db_engine():
    """In-memory SQLite engine for HTTP tests; recreated per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def _build_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.get_cached = AsyncMock(return_value=None)
    registry.put = AsyncMock()
    registry.remove = AsyncMock()
    registry.get_or_warm = AsyncMock(return_value=None)
    registry.start_eviction_loop = MagicMock()
    registry.stop_eviction_loop = AsyncMock()
    return registry


@pytest_asyncio.fixture()
async def test_app(http_db_engine):
    """Yield a (TestClient, session_factory) pair wired to the in-memory DB."""
    from src.api.app import create_app
    from src.api.db.engine import get_db

    session_factory = async_sessionmaker(http_db_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as db:
            yield db

    mock_registry = _build_mock_registry()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.state.session_registry = mock_registry
    app.state.config = None
    app.state.tool_registry = None

    with (
        patch("src.api.routes.sessions.warm_session", new_callable=AsyncMock) as mock_warm,
        patch("src.config.load_config", side_effect=Exception("no config in tests")),
    ):

        async def _fake_warm(record, app_state):
            from src.api.session_bridge import ApiSession

            return ApiSession(
                id=record.id,
                user_id=record.user_id,
                name=record.name,
                config={},
                agent_state="idle",
            )

        mock_warm.side_effect = _fake_warm

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, session_factory


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _token(user_id: str, role: str = "user") -> str:
    return create_access_token(user_id, role)


async def _create_user(session_factory, user_id: str | None = None, role: str = "user") -> str:
    uid = user_id or str(uuid.uuid4())
    async with session_factory() as db:
        repo = UserRepository(db)
        await repo.create(
            user_id=uid,
            username=f"u_{uid[:8]}",
            email=f"{uid[:8]}@test.com",
            password_hash="hashed",
            role=role,
        )
        await db.commit()
    return uid


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_session(client: TestClient, token: str, **kwargs) -> dict:
    """POST /api/v1/sessions and return the response JSON data dict."""
    body: dict = {"name": kwargs.get("name", "Test session")}
    if "config" in kwargs:
        body["config"] = kwargs["config"]
    resp = client.post("/api/v1/sessions", json=body, headers=_auth(token))
    return resp


# ---------------------------------------------------------------------------
# TestSessionCreate
# ---------------------------------------------------------------------------


class TestSessionCreate:
    """POST /api/v1/sessions"""

    @pytest.mark.asyncio
    async def test_happy_path_minimal(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"name": "My session"},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["name"] == "My session"
        assert data["state"] == "idle"
        assert data["owner_id"] == uid
        assert "id" in data

    @pytest.mark.asyncio
    async def test_happy_path_all_config_fields(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={
                "name": "Full config",
                "config": {
                    "model": "gpt-4.1-mini",
                    "max_steps": 50,
                    "parallel_tool_execution": True,
                    "context_compression": False,
                    "memory_mode": "reasoning",
                },
            },
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 201
        cfg = resp.json()["data"]["config"]
        assert cfg["model"] == "gpt-4.1-mini"
        assert cfg["max_steps"] == 50

    @pytest.mark.asyncio
    async def test_name_at_max_length(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        name = "x" * 256
        resp = client.post(
            "/api/v1/sessions",
            json={"name": name},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["name"] == name

    @pytest.mark.asyncio
    async def test_name_exceeds_max_length(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"name": "x" * 257},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_uses_defaults(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post("/api/v1/sessions", json={}, headers=_auth(_token(uid)))
        assert resp.status_code == 201
        assert resp.json()["data"]["name"].startswith("Session ")

    @pytest.mark.asyncio
    async def test_max_steps_minimum_valid(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"config": {"max_steps": 1}},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["config"]["max_steps"] == 1

    @pytest.mark.asyncio
    async def test_max_steps_maximum_valid(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"config": {"max_steps": 200}},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 201
        assert resp.json()["data"]["config"]["max_steps"] == 200

    @pytest.mark.asyncio
    async def test_max_steps_zero_invalid(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"config": {"max_steps": 0}},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_max_steps_over_limit_invalid(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"config": {"max_steps": 201}},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_max_steps_negative_invalid(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions",
            json={"config": {"max_steps": -1}},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, test_app) -> None:
        client, _ = test_app
        resp = client.post("/api/v1/sessions", json={"name": "No auth"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_response_envelope_shape(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post(
            "/api/v1/sessions", json={"name": "Envelope"}, headers=_auth(_token(uid))
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "data" in body
        assert "error" in body
        assert body["error"] is None

    @pytest.mark.asyncio
    async def test_created_session_has_token_counts(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post("/api/v1/sessions", json={}, headers=_auth(_token(uid)))
        assert resp.status_code == 201
        tc = resp.json()["data"]["token_counts"]
        assert "input_tokens" in tc
        assert "output_tokens" in tc
        assert "context_window" in tc

    @pytest.mark.asyncio
    async def test_created_session_has_timestamps(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.post("/api/v1/sessions", json={}, headers=_auth(_token(uid)))
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["created_at"] is not None
        assert data["updated_at"] is not None
        assert data["archived_at"] is None


# ---------------------------------------------------------------------------
# TestSessionList
# ---------------------------------------------------------------------------


class TestSessionList:
    """GET /api/v1/sessions"""

    @pytest.mark.asyncio
    async def test_empty_list(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.get("/api/v1/sessions", headers=_auth(_token(uid)))
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["items"] == []
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_multiple_sessions_listed(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        for i in range(3):
            client.post("/api/v1/sessions", json={"name": f"S{i}"}, headers=_auth(tok))
        resp = client.get("/api/v1/sessions", headers=_auth(tok))
        assert resp.status_code == 200
        assert len(resp.json()["data"]["items"]) == 3

    @pytest.mark.asyncio
    async def test_pagination_first_page(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        for i in range(5):
            client.post("/api/v1/sessions", json={"name": f"S{i}"}, headers=_auth(tok))
        resp = client.get("/api/v1/sessions?limit=3", headers=_auth(tok))
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert len(body["items"]) == 3
        assert body["has_more"] is True
        assert body["next_cursor"] is not None

    @pytest.mark.asyncio
    async def test_pagination_second_page(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        for i in range(5):
            client.post("/api/v1/sessions", json={"name": f"S{i}"}, headers=_auth(tok))
        first = client.get("/api/v1/sessions?limit=3", headers=_auth(tok)).json()["data"]
        cursor = first["next_cursor"]
        resp2 = client.get(f"/api/v1/sessions?limit=3&cursor={cursor}", headers=_auth(tok))
        assert resp2.status_code == 200
        body2 = resp2.json()["data"]
        assert len(body2["items"]) == 2
        assert body2["has_more"] is False

    @pytest.mark.asyncio
    async def test_pagination_no_overlap_between_pages(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        for i in range(5):
            client.post("/api/v1/sessions", json={"name": f"S{i}"}, headers=_auth(tok))
        first = client.get("/api/v1/sessions?limit=3", headers=_auth(tok)).json()["data"]
        cursor = first["next_cursor"]
        second = client.get(f"/api/v1/sessions?limit=3&cursor={cursor}", headers=_auth(tok)).json()[
            "data"
        ]
        ids1 = {s["id"] for s in first["items"]}
        ids2 = {s["id"] for s in second["items"]}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_include_archived_true_shows_archived(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        create_resp = client.post("/api/v1/sessions", json={"name": "Archived"}, headers=_auth(tok))
        sid = create_resp.json()["data"]["id"]
        client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok))

        resp = client.get("/api/v1/sessions?include_archived=true", headers=_auth(tok))
        assert resp.status_code == 200
        ids = [s["id"] for s in resp.json()["data"]["items"]]
        assert sid in ids

    @pytest.mark.asyncio
    async def test_include_archived_false_excludes_archived(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        create_resp = client.post(
            "/api/v1/sessions", json={"name": "Will archive"}, headers=_auth(tok)
        )
        sid = create_resp.json()["data"]["id"]
        client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok))

        resp = client.get("/api/v1/sessions", headers=_auth(tok))
        assert resp.status_code == 200
        ids = [s["id"] for s in resp.json()["data"]["items"]]
        assert sid not in ids

    @pytest.mark.asyncio
    async def test_invalid_cursor_returns_400(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.get(
            "/api/v1/sessions?cursor=!!!notbase64!!!",
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_non_admin_sees_own_sessions_only(self, test_app) -> None:
        client, sf = test_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        for i in range(2):
            client.post("/api/v1/sessions", json={"name": f"U1-S{i}"}, headers=_auth(tok1))
        client.post("/api/v1/sessions", json={"name": "U2-S0"}, headers=_auth(tok2))
        resp = client.get("/api/v1/sessions", headers=_auth(tok1))
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 2
        assert all(s["owner_id"] == uid1 for s in items)

    @pytest.mark.asyncio
    async def test_admin_sees_all_sessions(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        admin_id = await _create_user(sf, role="admin")
        tok = _token(uid)
        admin_tok = _token(admin_id, role="admin")
        client.post("/api/v1/sessions", json={"name": "User"}, headers=_auth(tok))
        client.post("/api/v1/sessions", json={"name": "Admin"}, headers=_auth(admin_tok))
        resp = client.get("/api/v1/sessions", headers=_auth(admin_tok))
        assert resp.status_code == 200
        assert len(resp.json()["data"]["items"]) >= 2

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, test_app) -> None:
        client, _ = test_app
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestSessionGet
# ---------------------------------------------------------------------------


class TestSessionGet:
    """GET /api/v1/sessions/{id}"""

    @pytest.mark.asyncio
    async def test_happy_path(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "Detail"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.get(f"/api/v1/sessions/{sid}", headers=_auth(tok))
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == sid
        assert data["name"] == "Detail"
        assert "config" in data
        assert "token_counts" in data

    @pytest.mark.asyncio
    async def test_not_found(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.get("/api/v1/sessions/nonexistent-id", headers=_auth(_token(uid)))
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_non_owner_forbidden(self, test_app) -> None:
        client, sf = test_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post("/api/v1/sessions", json={"name": "Private"}, headers=_auth(tok1)).json()[
            "data"
        ]["id"]

        resp = client.get(f"/api/v1/sessions/{sid}", headers=_auth(tok2))
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_admin_can_get_any_session(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        admin_id = await _create_user(sf, role="admin")
        tok = _token(uid)
        admin_tok = _token(admin_id, role="admin")
        sid = client.post(
            "/api/v1/sessions", json={"name": "User session"}, headers=_auth(tok)
        ).json()["data"]["id"]

        resp = client.get(f"/api/v1/sessions/{sid}", headers=_auth(admin_tok))
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == sid

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, test_app) -> None:
        client, _ = test_app
        resp = client.get("/api/v1/sessions/some-id")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestSessionUpdate
# ---------------------------------------------------------------------------


class TestSessionUpdate:
    """PATCH /api/v1/sessions/{id}"""

    @pytest.mark.asyncio
    async def test_update_name(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "Original"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.patch(
            f"/api/v1/sessions/{sid}",
            json={"name": "Renamed"},
            headers=_auth(tok),
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_update_config_model(self, test_app) -> None:
        # app.state.config is None (load_config patched out in fixture) so
        # model-alias validation is skipped and any string passes through.
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post(
            "/api/v1/sessions",
            json={"name": "S", "config": {"model": "oss"}},
            headers=_auth(tok),
        ).json()["data"]["id"]

        resp = client.patch(
            f"/api/v1/sessions/{sid}",
            json={"config": {"model": "some-model-alias"}},
            headers=_auth(tok),
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["config"]["model"] == "some-model-alias"

    @pytest.mark.asyncio
    async def test_update_config_model_invalid_alias_returns_422(self, test_app) -> None:
        # Inject a mock config so alias validation is active; unknown aliases → 422.
        from unittest.mock import MagicMock

        from src.config import ConfigError

        client, sf = test_app
        mock_cfg = MagicMock()
        mock_cfg.resolve_llm_config_for.side_effect = ConfigError("alias not found")
        client.app.state.config = mock_cfg

        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post(
            "/api/v1/sessions",
            json={"name": "S2"},
            headers=_auth(tok),
        ).json()[
            "data"
        ]["id"]

        resp = client.patch(
            f"/api/v1/sessions/{sid}",
            json={"config": {"model": "nonexistent-alias-xyz"}},
            headers=_auth(tok),
        )
        assert resp.status_code == 422
        err = resp.json().get("error") or {}
        assert err.get("code") == "MODEL_NOT_FOUND"

        # Restore so subsequent tests in the fixture scope are unaffected.
        client.app.state.config = None

    @pytest.mark.asyncio
    async def test_patch_preserves_existing_config(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post(
            "/api/v1/sessions",
            json={"name": "S", "config": {"model": "gpt-4.1-mini", "max_steps": 10}},
            headers=_auth(tok),
        ).json()["data"]["id"]

        resp = client.patch(
            f"/api/v1/sessions/{sid}",
            json={"config": {"max_steps": 25}},
            headers=_auth(tok),
        )
        assert resp.status_code == 200
        cfg = resp.json()["data"]["config"]
        assert cfg["max_steps"] == 25
        assert cfg.get("model") == "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_not_found(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.patch(
            "/api/v1/sessions/does-not-exist",
            json={"name": "X"},
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_non_owner_forbidden(self, test_app) -> None:
        client, sf = test_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post(
            "/api/v1/sessions", json={"name": "Protected"}, headers=_auth(tok1)
        ).json()["data"]["id"]

        resp = client.patch(
            f"/api/v1/sessions/{sid}",
            json={"name": "Hacked"},
            headers=_auth(tok2),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, test_app) -> None:
        client, _ = test_app
        resp = client.patch("/api/v1/sessions/some-id", json={"name": "X"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_patch_body_is_no_op(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "NoChange"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.patch(f"/api/v1/sessions/{sid}", json={}, headers=_auth(tok))
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "NoChange"


# ---------------------------------------------------------------------------
# TestSessionDelete
# ---------------------------------------------------------------------------


class TestSessionDelete:
    """DELETE /api/v1/sessions/{id}"""

    @pytest.mark.asyncio
    async def test_happy_path_archives_session(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post(
            "/api/v1/sessions", json={"name": "To delete"}, headers=_auth(tok)
        ).json()["data"]["id"]

        resp = client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok))
        assert resp.status_code == 200
        assert resp.json()["data"] is None

    @pytest.mark.asyncio
    async def test_not_found(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        resp = client.delete("/api/v1/sessions/nonexistent", headers=_auth(_token(uid)))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_non_owner_forbidden(self, test_app) -> None:
        client, sf = test_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post(
            "/api/v1/sessions", json={"name": "Protected"}, headers=_auth(tok1)
        ).json()["data"]["id"]

        resp = client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok2))
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_can_delete_any_session(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        admin_id = await _create_user(sf, role="admin")
        tok = _token(uid)
        admin_tok = _token(admin_id, role="admin")
        sid = client.post(
            "/api/v1/sessions", json={"name": "User session"}, headers=_auth(tok)
        ).json()["data"]["id"]

        resp = client.delete(f"/api/v1/sessions/{sid}", headers=_auth(admin_tok))
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, test_app) -> None:
        client, _ = test_app
        resp = client.delete("/api/v1/sessions/some-id")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_deleted_session_not_in_list(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "Gone"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok))

        list_resp = client.get("/api/v1/sessions", headers=_auth(tok))
        items = list_resp.json()["data"]["items"]
        assert not any(s["id"] == sid for s in items)

    @pytest.mark.asyncio
    async def test_deleted_session_appears_with_include_archived(self, test_app) -> None:
        client, sf = test_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post(
            "/api/v1/sessions", json={"name": "Archive me"}, headers=_auth(tok)
        ).json()["data"]["id"]

        client.delete(f"/api/v1/sessions/{sid}", headers=_auth(tok))

        list_resp = client.get("/api/v1/sessions?include_archived=true", headers=_auth(tok))
        ids = [s["id"] for s in list_resp.json()["data"]["items"]]
        assert sid in ids


# ---------------------------------------------------------------------------
# TestMessageSend — POST /api/v1/sessions/{id}/messages
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def message_app(http_db_engine):
    """TestClient fixture with a real session_registry mock supporting get_or_warm."""
    from src.api.app import create_app
    from src.api.db.engine import get_db
    from src.api.session_bridge import ApiSession
    from src.orchestration.session_state import SessionState

    session_factory = async_sessionmaker(http_db_engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as db:
            yield db

    # Build a session cache that is keyed by session_id so get_or_warm can
    # return pre-created sessions.
    _sessions: dict[str, ApiSession] = {}

    async def _fake_get_cached(session_id: str):
        return _sessions.get(session_id)

    async def _fake_get_or_warm(session_id: str, db):
        return _sessions.get(session_id)

    async def _fake_put(sess: ApiSession) -> None:
        _sessions[sess.id] = sess

    async def _fake_remove(session_id: str) -> None:
        _sessions.pop(session_id, None)

    mock_registry = MagicMock()
    mock_registry.get_cached = AsyncMock(side_effect=_fake_get_cached)
    mock_registry.get_or_warm = AsyncMock(side_effect=_fake_get_or_warm)
    mock_registry.put = AsyncMock(side_effect=_fake_put)
    mock_registry.remove = AsyncMock(side_effect=_fake_remove)
    mock_registry.start_eviction_loop = MagicMock()
    mock_registry.stop_eviction_loop = AsyncMock()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.state.session_registry = mock_registry
    app.state.config = None
    app.state.tool_registry = None

    async def _fake_warm(record, app_state):
        sess = ApiSession(
            id=record.id,
            user_id=record.user_id,
            name=record.name,
            config={},
            agent_state="idle",
            session_state=SessionState(no_confirm=True),
        )
        _sessions[sess.id] = sess
        return sess

    with patch("src.api.routes.sessions.warm_session", new_callable=AsyncMock) as mw:
        mw.side_effect = _fake_warm

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, session_factory, _sessions


class TestMessageSend:
    """POST /api/v1/sessions/{id}/messages"""

    @pytest.mark.asyncio
    async def test_happy_path_returns_202(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sess_resp = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok))
        sid = sess_resp.json()["data"]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Hello"},
            headers=_auth(tok),
        )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_response_contains_user_message(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Test message"},
            headers=_auth(tok),
        )
        assert resp.status_code == 202
        msg = resp.json()["data"]
        assert msg["role"] == "user"
        assert msg["session_id"] == sid
        assert "id" in msg
        assert "created_at" in msg

    @pytest.mark.asyncio
    async def test_empty_content_returns_422(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": ""},
            headers=_auth(tok),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_mode_normal_accepted(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Hello", "mode": "normal"},
            headers=_auth(tok),
        )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_mode_think_accepted(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        # Patch run_message_turn to prevent the background asyncio task from
        # spawning deep_think threads that outlive the test fixture.  This test
        # only verifies the 202 response; agent execution is covered by live_llm tests.
        with patch("src.api.routes.messages.run_message_turn", new_callable=AsyncMock):
            resp = client.post(
                f"/api/v1/sessions/{sid}/messages",
                json={"content": "Think hard", "mode": "think"},
                headers=_auth(tok),
            )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_mode_delegate_accepted(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Delegate this", "mode": "delegate"},
            headers=_auth(tok),
        )
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_422(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Hello", "mode": "superfast"},
            headers=_auth(tok),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        resp = client.post(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            json={"content": "Hello"},
            headers=_auth(tok),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, message_app) -> None:
        client, _, _ = message_app
        resp = client.post(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            json={"content": "Hello"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_owner_cannot_send_message(self, message_app) -> None:
        client, sf, _ = message_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok1)).json()[
            "data"
        ]["id"]

        resp = client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Hello"},
            headers=_auth(tok2),
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestMessageList — GET /api/v1/sessions/{id}/messages
# ---------------------------------------------------------------------------


class TestMessageList:
    """GET /api/v1/sessions/{id}/messages"""

    @pytest.mark.asyncio
    async def test_empty_history(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["items"] == []
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_history_contains_sent_messages(self, message_app) -> None:
        # Seed messages directly via DB to avoid the 409 TURN_IN_PROGRESS race
        # that occurs when sending a second message before the first task completes.
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        from src.api.db.repositories.messages import MessageRepository

        async with sf() as db:
            repo = MessageRepository(db)
            await repo.create(
                session_id=sid, role="user", content_json=json.dumps({"text": "First"})
            )
            await repo.create(
                session_id=sid, role="user", content_json=json.dumps({"text": "Second"})
            )
            await db.commit()

        resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 2
        assert all(m["session_id"] == sid for m in items)
        assert all(m["role"] == "user" for m in items)

    @pytest.mark.asyncio
    async def test_cursor_pagination(self, message_app) -> None:
        # Seed 5 messages via DB to avoid the 409 TURN_IN_PROGRESS race.
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        from src.api.db.repositories.messages import MessageRepository

        async with sf() as db:
            repo = MessageRepository(db)
            for i in range(5):
                await repo.create(
                    session_id=sid,
                    role="user",
                    content_json=json.dumps({"text": f"msg {i}"}),
                )
            await db.commit()

        resp1 = client.get(f"/api/v1/sessions/{sid}/messages?limit=3", headers=_auth(tok))
        assert resp1.status_code == 200
        body1 = resp1.json()["data"]
        assert len(body1["items"]) == 3
        assert body1["has_more"] is True
        assert body1["next_cursor"] is not None

        cursor = body1["next_cursor"]
        resp2 = client.get(
            f"/api/v1/sessions/{sid}/messages?limit=3&cursor={cursor}",
            headers=_auth(tok),
        )
        assert resp2.status_code == 200
        body2 = resp2.json()["data"]
        assert len(body2["items"]) == 2

        ids1 = {m["id"] for m in body1["items"]}
        ids2 = {m["id"] for m in body2["items"]}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        resp = client.get(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, message_app) -> None:
        client, _, _ = message_app
        resp = client.get(f"/api/v1/sessions/{uuid.uuid4()}/messages")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_owner_forbidden(self, message_app) -> None:
        client, sf, _ = message_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok1)).json()[
            "data"
        ]["id"]

        resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok2))
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_message_fields_present(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"content": "Check fields"},
            headers=_auth(tok),
        )
        resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert resp.status_code == 200
        msg = resp.json()["data"]["items"][0]
        for field in ("id", "session_id", "role", "content", "created_at", "tool_calls"):
            assert field in msg, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# TestMessageClear — DELETE /api/v1/sessions/{id}/messages
# ---------------------------------------------------------------------------


class TestMessageClear:
    """DELETE /api/v1/sessions/{id}/messages"""

    @pytest.mark.asyncio
    async def test_happy_path_clears_all(self, message_app) -> None:
        # Seed messages via DB to avoid 409 TURN_IN_PROGRESS from sequential REST sends.
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        from src.api.db.repositories.messages import MessageRepository

        async with sf() as db:
            repo = MessageRepository(db)
            for i in range(3):
                await repo.create(
                    session_id=sid,
                    role="user",
                    content_json=json.dumps({"text": f"msg {i}"}),
                )
            await db.commit()

        resp = client.delete(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert resp.status_code == 200
        assert resp.json()["data"] is None

        list_resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert list_resp.json()["data"]["items"] == []

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        resp = client.delete(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers=_auth(_token(uid)),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_auth_returns_401(self, message_app) -> None:
        client, _, _ = message_app
        resp = client.delete(f"/api/v1/sessions/{uuid.uuid4()}/messages")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_owner_forbidden(self, message_app) -> None:
        client, sf, _ = message_app
        uid1 = await _create_user(sf)
        uid2 = await _create_user(sf)
        tok1, tok2 = _token(uid1), _token(uid2)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok1)).json()[
            "data"
        ]["id"]

        resp = client.delete(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok2))
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_keep_last_partial_clear(self, message_app) -> None:
        # Seed messages via DB to avoid 409 TURN_IN_PROGRESS from sequential REST sends.
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        from src.api.db.repositories.messages import MessageRepository

        async with sf() as db:
            repo = MessageRepository(db)
            for i in range(5):
                await repo.create(
                    session_id=sid,
                    role="user",
                    content_json=json.dumps({"text": f"msg {i}"}),
                )
            await db.commit()

        resp = client.request(
            "DELETE",
            f"/api/v1/sessions/{sid}/messages",
            json={"keep_last": 2},
            headers=_auth(tok),
        )
        assert resp.status_code == 200

        list_resp = client.get(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        items = list_resp.json()["data"]["items"]
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_clear_empty_history_is_no_op(self, message_app) -> None:
        client, sf, _ = message_app
        uid = await _create_user(sf)
        tok = _token(uid)
        sid = client.post("/api/v1/sessions", json={"name": "S"}, headers=_auth(tok)).json()[
            "data"
        ]["id"]

        resp = client.delete(f"/api/v1/sessions/{sid}/messages", headers=_auth(tok))
        assert resp.status_code == 200
