"""Comprehensive tools endpoint coverage.

Tests all 4 tool endpoints:
  GET  /api/v1/tools                     — list all tools
  GET  /api/v1/tools/{name}              — get tool details
  GET  /api/v1/sessions/{id}/tools       — session tool status
  PATCH /api/v1/sessions/{id}/tools      — manage session tool state

Focuses on cases not covered in test_api_sessions_complete.py:
  - All PATCH actions: load, unload, enable, disable, auto_approve, revoke_approval
  - Pagination in list tools
  - Tool detail parameter schema extraction
  - Status classification: disabled, pinned, auto_approved, active, on_demand
  - run_config sync when loading/disabling
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

from src.api.db.engine import Base, get_db  # noqa: E402

_VALID_PASSWORD = "TestPass1!"


@pytest.fixture()
def app():
    from src.api.app import create_app

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
    uname = f"tls_{uuid.uuid4().hex[:8]}"
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


def _make_registry(
    tool_names=("web_search", "write_file", "shell"),
    mcp_names=("mcp_browser",),
):
    tools = {}
    for name in tool_names:
        t = MagicMock()
        t.name = name
        t.description = f"Tool description for {name}. Does things."
        t.args_schema = None
        tools[name] = t
    for name in mcp_names:
        t = MagicMock()
        t.name = name
        t.description = f"MCP tool {name}"
        t.args_schema = None
        tools[name] = t

    registry = MagicMock()
    registry.tools = tools
    registry.is_mcp_tool = lambda n: n in mcp_names
    registry.requires_confirmation = lambda n: n in ("write_file", "shell")
    registry.get_tool_server = lambda n: "mcp_server" if n in mcp_names else None
    return registry


def _make_live_session(tool_names=("web_search", "write_file", "shell", "mcp_browser")):
    from src.orchestration.session_state import SessionState

    # Use a real SessionState so is_denied() / deny_tool() / allow_tool() work correctly.
    ss = SessionState()

    live = MagicMock()
    live.session_state = ss
    live.run_config = None
    live.turn_lock = _asyncio.Lock()
    return live


def _setup_session_tools(app, live_session, registry):
    mock_sess_reg = MagicMock()
    mock_sess_reg.get_or_warm = AsyncMock(return_value=live_session)
    app.state.session_registry = mock_sess_reg
    app.state.tool_registry = registry


# ---------------------------------------------------------------------------
# GET /api/v1/tools — list all tools
# ---------------------------------------------------------------------------


class TestListTools:
    def test_list_requires_auth(self, client, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools")
        assert r.status_code == 401

    def test_list_returns_200(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200

    def test_list_envelope(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        body = r.json()
        assert "data" in body
        assert body["error"] is None
        page = body["data"]
        assert "items" in page
        assert "has_more" in page
        assert "next_cursor" in page
        assert "total" in page

    def test_list_no_registry_returns_empty(self, client, tokens, app):
        app.state.tool_registry = None
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert r.json()["data"]["items"] == []
        assert r.json()["data"]["total"] == 0

    def test_list_returns_all_tools(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        page = r.json()["data"]
        assert page["total"] == 4  # 3 + 1 mcp

    def test_list_item_fields(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools", headers=_h(tokens["owner"]))
        item = r.json()["data"]["items"][0]
        assert "name" in item
        assert "short_description" in item
        assert "status" in item
        assert "requires_confirmation" in item
        assert "is_mcp" in item

    def test_list_search_filter_matches_name(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?search=web_search", headers=_h(tokens["owner"]))
        items = r.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["name"] == "web_search"

    def test_list_search_no_match_empty(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?search=nonexistent_xyz_tool", headers=_h(tokens["owner"]))
        assert r.json()["data"]["items"] == []
        assert r.json()["data"]["total"] == 0

    def test_list_exclude_mcp(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?include_mcp=false", headers=_h(tokens["owner"]))
        names = [i["name"] for i in r.json()["data"]["items"]]
        assert "mcp_browser" not in names
        assert "web_search" in names

    def test_list_include_mcp_true_shows_mcp(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?include_mcp=true", headers=_h(tokens["owner"]))
        names = [i["name"] for i in r.json()["data"]["items"]]
        assert "mcp_browser" in names

    def test_list_limit_1_returns_one_item(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?limit=1", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        page = r.json()["data"]
        assert len(page["items"]) == 1
        assert page["has_more"] is True
        assert page["next_cursor"] is not None

    def test_list_cursor_pagination(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r1 = client.get("/api/v1/tools?limit=2", headers=_h(tokens["owner"]))
        cursor = r1.json()["data"]["next_cursor"]
        assert cursor is not None
        r2 = client.get(f"/api/v1/tools?limit=2&cursor={cursor}", headers=_h(tokens["owner"]))
        assert r2.status_code == 200
        page2 = r2.json()["data"]
        assert len(page2["items"]) >= 1

    def test_list_invalid_cursor_returns_400(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?cursor=notvalidbase64!!!", headers=_h(tokens["owner"]))
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_CURSOR"

    def test_list_requires_confirmation_field(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?search=write_file", headers=_h(tokens["owner"]))
        item = r.json()["data"]["items"][0]
        assert item["requires_confirmation"] is True

    def test_list_mcp_flag_set_for_mcp_tools(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools?search=mcp_browser", headers=_h(tokens["owner"]))
        item = r.json()["data"]["items"][0]
        assert item["is_mcp"] is True


# ---------------------------------------------------------------------------
# GET /api/v1/tools/{name} — get tool details
# ---------------------------------------------------------------------------


class TestGetTool:
    def test_get_requires_auth(self, client, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/web_search")
        assert r.status_code == 401

    def test_get_returns_200(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/web_search", headers=_h(tokens["owner"]))
        assert r.status_code == 200

    def test_get_returns_tool_out_fields(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/web_search", headers=_h(tokens["owner"]))
        data = r.json()["data"]
        required = [
            "name",
            "description",
            "short_description",
            "status",
            "requires_confirmation",
            "parameters",
            "is_mcp",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_get_returns_correct_name(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/write_file", headers=_h(tokens["owner"]))
        assert r.json()["data"]["name"] == "write_file"

    def test_get_requires_confirmation_for_write_file(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/write_file", headers=_h(tokens["owner"]))
        assert r.json()["data"]["requires_confirmation"] is True

    def test_get_not_requires_confirmation_for_web_search(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/web_search", headers=_h(tokens["owner"]))
        assert r.json()["data"]["requires_confirmation"] is False

    def test_get_mcp_tool_is_mcp_true(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/mcp_browser", headers=_h(tokens["owner"]))
        data = r.json()["data"]
        assert data["is_mcp"] is True
        assert data["mcp_server"] == "mcp_server"

    def test_get_nonexistent_tool_returns_404(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/totally_unknown_tool", headers=_h(tokens["owner"]))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_get_no_registry_returns_404(self, client, tokens, app):
        app.state.tool_registry = None
        r = client.get("/api/v1/tools/web_search", headers=_h(tokens["owner"]))
        assert r.status_code == 404

    def test_get_parameters_is_list(self, client, tokens, app):
        app.state.tool_registry = _make_registry()
        r = client.get("/api/v1/tools/web_search", headers=_h(tokens["owner"]))
        params = r.json()["data"]["parameters"]
        assert isinstance(params, list)


# ---------------------------------------------------------------------------
# GET /api/v1/sessions/{id}/tools — session tool status
# ---------------------------------------------------------------------------


class TestGetSessionTools:
    @pytest.fixture()
    def session_setup(self, client, tokens, sid, app):
        registry = _make_registry()
        live = _make_live_session()
        _setup_session_tools(app, live, registry)
        return sid, live, registry

    def test_get_requires_auth(self, client, session_setup):
        sid, _, _ = session_setup
        r = client.get(f"/api/v1/sessions/{sid}/tools")
        assert r.status_code == 401

    def test_get_non_owner_returns_403(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["other"]))
        assert r.status_code == 403

    def test_get_returns_list(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)

    def test_get_lists_all_registry_tools(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        names = [i["name"] for i in r.json()["data"]]
        assert "web_search" in names
        assert "write_file" in names

    def test_get_no_registry_returns_empty_list(self, client, tokens, sid, app):
        app.state.tool_registry = None
        mock_sess_reg = MagicMock()
        mock_sess_reg.get_or_warm = AsyncMock(return_value=None)
        app.state.session_registry = mock_sess_reg
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_get_disabled_tool_shows_disabled_status(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.denials.add("write_file")
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        items = {i["name"]: i for i in r.json()["data"]}
        assert items["write_file"]["status"] == "disabled"

    def test_get_pinned_tool_shows_pinned_status(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.pinned_tools.add("web_search")
        live.session_state.loaded_tools.add("web_search")
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        items = {i["name"]: i for i in r.json()["data"]}
        assert items["web_search"]["status"] == "pinned"

    def test_get_active_tool_shows_active_status(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.loaded_tools.add("shell")
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        items = {i["name"]: i for i in r.json()["data"]}
        assert items["shell"]["status"] == "active"

    def test_get_auto_approved_tool_shows_auto_approved(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.loaded_tools.add("web_search")
        live.session_state.add_approval("web_search")
        r = client.get(f"/api/v1/sessions/{sid}/tools", headers=_h(tokens["owner"]))
        items = {i["name"]: i for i in r.json()["data"]}
        assert items["web_search"]["status"] == "auto_approved"


# ---------------------------------------------------------------------------
# PATCH /api/v1/sessions/{id}/tools — manage tool state
# ---------------------------------------------------------------------------


class TestPatchSessionTools:
    @pytest.fixture()
    def session_setup(self, client, tokens, sid, app):
        registry = _make_registry()
        live = _make_live_session()
        _setup_session_tools(app, live, registry)
        return sid, live, registry

    def test_patch_requires_auth(self, client, session_setup):
        sid, _, _ = session_setup
        r = client.patch(f"/api/v1/sessions/{sid}/tools", json={"disable": ["web_search"]})
        assert r.status_code == 401

    def test_patch_non_owner_returns_403(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["other"]),
            json={"disable": ["web_search"]},
        )
        assert r.status_code == 403

    def test_patch_returns_list(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"disable": ["web_search"]},
        )
        assert r.status_code == 200
        assert isinstance(r.json()["data"], list)

    def test_patch_load_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"load": ["web_search"]},
        )
        assert r.status_code == 200
        assert "web_search" in live.session_state.pinned_tools
        assert "web_search" in live.session_state.loaded_tools

    def test_patch_unload_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.pinned_tools.add("web_search")
        live.session_state.loaded_tools.add("web_search")
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"unload": ["web_search"]},
        )
        assert r.status_code == 200
        assert "web_search" not in live.session_state.pinned_tools

    def test_patch_disable_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"disable": ["write_file"]},
        )
        assert r.status_code == 200
        assert "write_file" in live.session_state.denials

    def test_patch_enable_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.denials.add("web_search")
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"enable": ["web_search"]},
        )
        assert r.status_code == 200
        assert "web_search" not in live.session_state.denials

    def test_patch_auto_approve_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"auto_approve": ["web_search"]},
        )
        assert r.status_code == 200
        assert "web_search" in live.session_state.get_approvals_snapshot()

    def test_patch_revoke_approval_tool(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        live.session_state.add_approval("web_search")
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"revoke_approval": ["web_search"]},
        )
        assert r.status_code == 200
        assert "web_search" not in live.session_state.get_approvals_snapshot()

    def test_patch_load_nonexistent_tool_returns_404(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"load": ["totally_unknown_tool"]},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_patch_disable_nonexistent_tool_returns_404(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"disable": ["totally_unknown_tool"]},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_patch_auto_approve_nonexistent_returns_404(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"auto_approve": ["totally_unknown_tool"]},
        )
        assert r.status_code == 404

    def test_patch_enable_nonexistent_no_error(self, client, tokens, session_setup):
        # enable only removes from denials — no existence check
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"enable": ["totally_unknown_tool"]},
        )
        assert r.status_code == 200

    def test_patch_revoke_approval_nonexistent_no_error(self, client, tokens, session_setup):
        # revoke_approval only removes from approvals — no existence check
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"revoke_approval": ["totally_unknown_tool"]},
        )
        assert r.status_code == 200

    def test_patch_empty_body_returns_200(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={},
        )
        assert r.status_code == 200

    def test_patch_multiple_actions_same_request(self, client, tokens, session_setup):
        sid, live, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={
                "load": ["web_search"],
                "disable": ["shell"],
                "auto_approve": ["write_file"],
            },
        )
        assert r.status_code == 200
        assert "web_search" in live.session_state.loaded_tools
        assert "shell" in live.session_state.denials
        assert "write_file" in live.session_state.get_approvals_snapshot()

    def test_patch_returns_updated_status(self, client, tokens, session_setup):
        sid, _, _ = session_setup
        r = client.patch(
            f"/api/v1/sessions/{sid}/tools",
            headers=_h(tokens["owner"]),
            json={"disable": ["write_file"]},
        )
        assert r.status_code == 200
        items = {i["name"]: i for i in r.json()["data"]}
        assert items["write_file"]["status"] == "disabled"
