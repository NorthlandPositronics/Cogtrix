"""Comprehensive session/messages/memory/tools coverage filling gaps.

New cases not in existing test files:
  Sessions:
    - MODEL_NOT_FOUND 422 on patch with invalid model alias
    - SESSION_NAME_DUPLICATE 409 on patch with conflicting name
    - Malformed cursor 400
    - include_archived=true returns archived sessions
    - non-owner delete 403, nonexistent delete 404
  Messages:
    - clear_history with keep_last > 0
    - clear_history with keep_last = 0 (explicit body)
    - list_messages auth/ownership/404 edge cases
    - send_message empty content 422, invalid mode 422
  Memory:
    - get/clear/switch forbidden for non-owner
    - session 404 for unknown session
    - switch missing body 422
  Tools:
    - list with/without registry
    - search filter match/no-match
    - exclude MCP
    - session tools auth/ownership/404 edge cases
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

from src.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


# ---------------------------------------------------------------------------
# App fixture — function-scope for isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    from src.api.app import create_app

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _asyncio.run(_setup())

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


def _register(client, username=None, password=_VALID_PASSWORD):
    if username is None:
        username = f"u_{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@ex.com", "password": password},
    )
    assert r.status_code == 201, f"register failed: {r.text}"
    return r.json()["data"]["access_token"]


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def _create_session(client, token, name=None):
    body = {}
    if name:
        body["name"] = name
    return client.post("/api/v1/sessions", headers=_h(token), json=body)


@pytest.fixture()
def tokens(client):
    """Two fresh user tokens (owner and other)."""
    owner_token = _register(client)
    other_token = _register(client)
    return {"owner": owner_token, "other": other_token}


# ---------------------------------------------------------------------------
# Sessions — extra coverage
# ---------------------------------------------------------------------------


class TestSessionCreateExtra:
    def test_session_name_conflict_returns_409(self, client, tokens):
        name = f"same_{uuid.uuid4().hex[:6]}"
        _create_session(client, tokens["owner"], name=name)
        r = _create_session(client, tokens["owner"], name=name)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "SESSION_NAME_DUPLICATE"

    def test_different_user_can_use_same_name(self, client, tokens):
        name = f"shared_{uuid.uuid4().hex[:6]}"
        _create_session(client, tokens["owner"], name=name)
        r = _create_session(client, tokens["other"], name=name)
        assert r.status_code == 201


class TestSessionPatchExtra:
    def test_patch_invalid_model_alias_returns_422(self, client, tokens, app):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]

        mock_cfg = MagicMock()
        mock_cfg.resolve_llm_config_for.side_effect = ValueError("not found")
        app.state.config = mock_cfg

        r = client.patch(
            f"/api/v1/sessions/{sid}",
            headers=_h(tokens["owner"]),
            json={"config": {"model": "nonexistent-alias"}},
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "MODEL_NOT_FOUND"

    def test_patch_name_duplicate_returns_409(self, client, tokens):
        name_a = f"na_{uuid.uuid4().hex[:6]}"
        name_b = f"nb_{uuid.uuid4().hex[:6]}"
        _create_session(client, tokens["owner"], name=name_a)
        session_b = _create_session(client, tokens["owner"], name=name_b)
        sid_b = session_b.json()["data"]["id"]

        r = client.patch(
            f"/api/v1/sessions/{sid_b}",
            headers=_h(tokens["owner"]),
            json={"name": name_a},
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "SESSION_NAME_DUPLICATE"

    def test_patch_non_owner_returns_403(self, client, tokens):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]
        r = client.patch(
            f"/api/v1/sessions/{sid}",
            headers=_h(tokens["other"]),
            json={"name": "hijack"},
        )
        assert r.status_code == 403


class TestSessionListExtra:
    def test_invalid_cursor_returns_400(self, client, tokens):
        r = client.get(
            "/api/v1/sessions?cursor=not-valid-base64!!!",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_CURSOR"

    def test_include_archived_false_hides_deleted(self, client, tokens):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]
        client.delete(f"/api/v1/sessions/{sid}", headers=_h(tokens["owner"]))

        r = client.get(
            "/api/v1/sessions?include_archived=false",
            headers=_h(tokens["owner"]),
        )
        ids = [s["id"] for s in r.json()["data"]["items"]]
        assert sid not in ids

    def test_include_archived_true_shows_deleted(self, client, tokens):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]
        client.delete(f"/api/v1/sessions/{sid}", headers=_h(tokens["owner"]))

        r = client.get(
            "/api/v1/sessions?include_archived=true",
            headers=_h(tokens["owner"]),
        )
        ids = [s["id"] for s in r.json()["data"]["items"]]
        assert sid in ids

    def test_limit_clamped_to_1(self, client, tokens):
        for _ in range(3):
            _create_session(client, tokens["owner"])
        r = client.get(
            "/api/v1/sessions?limit=1",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data["items"]) <= 1

    def test_limit_over_max_still_succeeds(self, client, tokens):
        r = client.get(
            "/api/v1/sessions?limit=99999",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200


class TestSessionDeleteExtra:
    def test_delete_non_owner_returns_403(self, client, tokens):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]
        r = client.delete(f"/api/v1/sessions/{sid}", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_delete_nonexistent_returns_404(self, client, tokens):
        r = client.delete(
            f"/api/v1/sessions/{uuid.uuid4()}",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"


# ---------------------------------------------------------------------------
# Messages — extra coverage
# ---------------------------------------------------------------------------


class TestMessagesExtra:
    @pytest.fixture()
    def session_id(self, client, tokens):
        r = _create_session(client, tokens["owner"])
        return r.json()["data"]["id"]

    def test_list_messages_empty(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["items"] == []
        assert data["has_more"] is False

    def test_list_messages_no_auth_returns_401(self, client, session_id):
        r = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert r.status_code == 401

    def test_list_messages_non_owner_returns_403(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["other"]),
        )
        assert r.status_code == 403

    def test_list_messages_nonexistent_session_returns_404(self, client, tokens):
        r = client.get(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"

    def test_list_messages_limit_min_1(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/messages?limit=1",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200

    def test_list_messages_limit_max_200(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/messages?limit=200",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200

    def test_list_messages_limit_over_max_returns_422(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/messages?limit=201",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 422

    def test_clear_history_no_auth_returns_401(self, client, session_id):
        r = client.delete(f"/api/v1/sessions/{session_id}/messages")
        assert r.status_code == 401

    def test_clear_history_non_owner_returns_403(self, client, tokens, session_id):
        r = client.delete(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["other"]),
        )
        assert r.status_code == 403

    def test_clear_history_nonexistent_session_returns_404(self, client, tokens):
        r = client.delete(
            f"/api/v1/sessions/{uuid.uuid4()}/messages",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 404

    def test_clear_history_no_body_succeeds(self, client, tokens, session_id):
        r = client.delete(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_clear_history_keep_last_zero(self, client, tokens, session_id):
        # TestClient wraps httpx — use request() for DELETE with body
        r = client.request(
            "DELETE",
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 0},
        )
        assert r.status_code == 200

    def test_clear_history_keep_last_positive(self, client, tokens, session_id):
        r = client.request(
            "DELETE",
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
            json={"keep_last": 5},
        )
        assert r.status_code == 200

    def test_send_message_no_auth_returns_401(self, client, session_id, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg
        r = client.post(f"/api/v1/sessions/{session_id}/messages", json={"content": "hi"})
        assert r.status_code == 401

    def test_send_message_empty_content_returns_422(self, client, tokens, session_id, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
            json={"content": ""},
        )
        assert r.status_code == 422

    def test_send_message_invalid_mode_returns_422(self, client, tokens, session_id, app):
        mock_reg = MagicMock()
        mock_reg.get_cached = AsyncMock(return_value=None)
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            headers=_h(tokens["owner"]),
            json={"content": "hi", "mode": "invalid_mode"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Memory — extra coverage
# ---------------------------------------------------------------------------


class TestMemoryExtra:
    @pytest.fixture()
    def session_id(self, client, tokens):
        r = _create_session(client, tokens["owner"])
        return r.json()["data"]["id"]

    def _mock_registry_with_live_session(self, app):
        mm = MagicMock()
        mm.to_dict.return_value = {"mode": "conversation", "window_size": 10}
        mm.clear = MagicMock()
        mm.save = MagicMock()

        live = MagicMock()
        live.memory_manager = mm
        live.config = {"memory_mode": "conversation"}
        live.token_counts = {"context_window": 4096, "input_tokens": 0, "output_tokens": 0}

        mock_reg = MagicMock()
        mock_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_reg
        return live

    def test_get_memory_no_auth_returns_401(self, client, session_id):
        r = client.get(f"/api/v1/sessions/{session_id}/memory")
        assert r.status_code == 401

    def test_get_memory_non_owner_returns_403(self, client, tokens, session_id):
        r = client.get(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["other"]),
        )
        assert r.status_code == 403

    def test_get_memory_unknown_session_returns_403_or_404(self, client, tokens, app):
        mock_reg = MagicMock()
        mock_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_reg

        r = client.get(
            f"/api/v1/sessions/{uuid.uuid4()}/memory",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code in (403, 404)

    def test_get_memory_returns_mode_and_stats(self, client, tokens, session_id, app):
        self._mock_registry_with_live_session(app)

        r = client.get(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert "mode" in data
        assert "window_messages" in data
        assert "summarized_messages" in data
        assert "tokens_used" in data
        assert "context_window" in data
        assert "vector_recall_enabled" in data

    def test_clear_memory_no_auth_returns_401(self, client, session_id):
        r = client.delete(f"/api/v1/sessions/{session_id}/memory")
        assert r.status_code == 401

    def test_clear_memory_non_owner_returns_403(self, client, tokens, session_id):
        r = client.delete(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["other"]),
        )
        assert r.status_code == 403

    def test_clear_memory_success(self, client, tokens, session_id, app):
        self._mock_registry_with_live_session(app)
        r = client.delete(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        assert r.json()["data"] is None

    def test_switch_memory_mode_no_auth_returns_401(self, client, session_id):
        r = client.patch(
            f"/api/v1/sessions/{session_id}/memory",
            json={"mode": "code"},
        )
        assert r.status_code == 401

    def test_switch_memory_mode_non_owner_returns_403(self, client, tokens, session_id):
        r = client.patch(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["other"]),
            json={"mode": "code"},
        )
        assert r.status_code == 403

    def test_switch_memory_mode_missing_body_returns_422(self, client, tokens, session_id):
        r = client.patch(
            f"/api/v1/sessions/{session_id}/memory",
            headers=_h(tokens["owner"]),
            json={},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Tools — extra coverage
# ---------------------------------------------------------------------------


def _make_mock_registry(tool_names=("search", "write_file"), mcp_tools=("mcp_tool",)):
    mock_reg = MagicMock()

    tools_dict = {}
    for name in tool_names:
        t = MagicMock()
        t.description = f"A tool called {name}"
        t.args_schema = None
        tools_dict[name] = t
    for name in mcp_tools:
        t = MagicMock()
        t.description = f"MCP tool {name}"
        t.args_schema = None
        tools_dict[name] = t

    mock_reg.tools = tools_dict
    mock_reg.is_mcp_tool = lambda n: n in mcp_tools
    mock_reg.requires_confirmation = lambda n: n in ("write_file",)
    mock_reg.get_tool_server = lambda n: "test_server" if n in mcp_tools else None
    return mock_reg


class TestListToolsExtra:
    def test_list_tools_with_no_registry_returns_empty(self, client, tokens, app):
        app.state.tool_registry = None
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["items"] == []

    def test_list_tools_with_registry(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data["items"]) >= 1
        assert data["total"] >= 1

    def test_list_tools_search_filter_match(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools?search=search", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        items = r.json()["data"]["items"]
        assert all(
            "search" in item["name"].lower() or "search" in item["short_description"].lower()
            for item in items
        )

    def test_list_tools_search_no_match_empty(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools?search=xyzzy_no_match", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert r.json()["data"]["items"] == []

    def test_list_tools_exclude_mcp(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools?include_mcp=false", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        names = [item["name"] for item in r.json()["data"]["items"]]
        assert "mcp_tool" not in names

    def test_list_tools_no_auth_returns_401(self, client, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools")
        assert r.status_code == 401

    def test_list_tools_limit_clamped(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools?limit=99999", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert len(r.json()["data"]["items"]) <= 500

    def test_list_tools_invalid_cursor_returns_400(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get(
            "/api/v1/tools?cursor=!!!invalid!!!",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_CURSOR"


class TestGetToolExtra:
    def test_get_tool_success(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools/search", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "search"
        assert "description" in data
        assert "parameters" in data

    def test_get_tool_not_found_returns_404(self, client, tokens, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools/nonexistent_tool", headers=_h(tokens["owner"]))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_get_tool_no_auth_returns_401(self, client, app):
        app.state.tool_registry = _make_mock_registry()
        r = client.get("/api/v1/tools/search")
        assert r.status_code == 401

    def test_get_tool_no_registry_returns_404(self, client, tokens, app):
        app.state.tool_registry = None
        r = client.get("/api/v1/tools/search", headers=_h(tokens["owner"]))
        assert r.status_code == 404


class TestSessionToolsExtra:
    @pytest.fixture()
    def session_with_live(self, client, tokens, app):
        session_r = _create_session(client, tokens["owner"])
        sid = session_r.json()["data"]["id"]

        ss = MagicMock()
        ss.denials = set()
        ss.loaded_tools = set()
        ss.pinned_tools = set()
        ss.approvals = set()
        ss.all_tool_originals = {}

        live = MagicMock()
        live.session_state = ss
        live.run_config = None
        live.turn_lock = _asyncio.Lock()

        mock_sess_reg = MagicMock()
        mock_sess_reg.get_or_warm = AsyncMock(return_value=live)
        app.state.session_registry = mock_sess_reg
        app.state.tool_registry = _make_mock_registry()
        return sid, live

    def test_get_session_tools_success(self, client, tokens, session_with_live):
        sid, _ = session_with_live
        r = client.get(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
        )
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)

    def test_get_session_tools_no_auth_returns_401(self, client, session_with_live):
        sid, _ = session_with_live
        r = client.get(f"/api/v1/sessions/{sid}/tools")
        assert r.status_code == 401

    def test_get_session_tools_non_owner_returns_403(self, client, tokens, session_with_live):
        sid, _ = session_with_live
        r = client.get(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["other"]),
        )
        assert r.status_code == 403

    def test_patch_session_tools_disable_nonexistent_returns_404(
        self, client, tokens, session_with_live
    ):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"disable": ["totally_nonexistent_tool"]},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_patch_session_tools_load_nonexistent_returns_404(
        self, client, tokens, session_with_live
    ):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"load": ["totally_nonexistent_tool"]},
        )
        assert r.status_code == 404

    def test_patch_session_tools_no_auth_returns_401(self, client, session_with_live):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            json={"disable": ["search"]},
        )
        assert r.status_code == 401

    def test_patch_session_tools_auto_approve_nonexistent_returns_404(
        self, client, tokens, session_with_live
    ):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"auto_approve": ["no_such_tool"]},
        )
        assert r.status_code == 404

    def test_patch_session_tools_enable_nonexistent_no_error(
        self, client, tokens, session_with_live
    ):
        # enable doesn't validate existence — just removes from denials
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"enable": ["does_not_exist"]},
        )
        assert r.status_code == 200

    def test_patch_session_tools_revoke_approval_nonexistent_no_error(
        self, client, tokens, session_with_live
    ):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"revoke_approval": ["does_not_exist"]},
        )
        assert r.status_code == 200

    def test_patch_session_tools_non_owner_returns_403(self, client, tokens, session_with_live):
        sid, _ = session_with_live
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["other"]),
            json={"disable": ["search"]},
        )
        assert r.status_code == 403
