"""Comprehensive MCP and config endpoint coverage.

MCP endpoints:
    GET    /api/v1/mcp/servers             — list
    POST   /api/v1/mcp/servers             — add (201); 409 on conflict, 503 on persist failure
    GET    /api/v1/mcp/servers/{name}      — get detail
    DELETE /api/v1/mcp/servers/{name}      — remove (204)
    POST   /api/v1/mcp/servers/{name}/restart — restart

Config endpoints:
    GET    /api/v1/config                  — get config
    PATCH  /api/v1/config                  — patch (admin)
    POST   /api/v1/config/reload           — reload (admin)
    GET    /api/v1/config/providers        — list providers
    GET    /api/v1/config/providers/{name} — get provider
    POST   /api/v1/config/provider         — deprecated 410
    POST   /api/v1/config/providers/{name}/health — provider health
    GET    /api/v1/config/models           — list models
    POST   /api/v1/config/model            — switch model (admin)
    POST   /api/v1/config/wizard           — start wizard (admin)
    POST   /api/v1/config/wizard/{id}/step — advance step
    DELETE /api/v1/config/wizard/{id}      — cancel wizard
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.app import create_app  # noqa: E402
from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_VALID_PASSWORD = "TestPass1!"  # lowercase + uppercase + digit + special


@pytest.fixture()
def app():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    loop.run_until_complete(_setup())

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


# Keep app_engine as an alias so test methods that accept (app_engine) still work.
@pytest.fixture()
def app_engine(app):
    return app, None


@pytest.fixture()
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def tokens(client):
    admin_uname = f"adm_{uuid.uuid4().hex[:6]}"
    admin_r = client.post(
        "/api/v1/auth/register",
        json={
            "username": admin_uname,
            "email": f"{admin_uname}@ex.com",
            "password": _VALID_PASSWORD,
        },
    )
    assert admin_r.status_code == 201, f"admin register failed: {admin_r.text}"
    admin_token = admin_r.json()["data"]["access_token"]

    user_uname = f"usr_{uuid.uuid4().hex[:6]}"
    user_r = client.post(
        "/api/v1/auth/register",
        json={
            "username": user_uname,
            "email": f"{user_uname}@ex.com",
            "password": _VALID_PASSWORD,
        },
    )
    assert user_r.status_code == 201, f"user register failed: {user_r.text}"
    user_token = user_r.json()["data"]["access_token"]

    return {"admin": admin_token, "user": user_token}


def _ah(tokens):
    return {"Authorization": f"Bearer {tokens['admin']}"}


def _uh(tokens):
    return {"Authorization": f"Bearer {tokens['user']}"}


# ---------------------------------------------------------------------------
# MCP endpoints
# ---------------------------------------------------------------------------


class TestMCPListServers:
    def test_no_mcp_client_returns_empty(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        app.state.mcp_manager = None
        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_with_mcp_client_returns_servers(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "my_server": {
                "command": "python",
                "args": ["-m", "myserver"],
                "requires_confirmation": True,
            }
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        assert r.status_code == 200
        items = r.json()["data"]
        assert len(items) == 1
        assert items[0]["name"] == "my_server"
        assert items[0]["transport"] == "stdio"

    def test_sse_transport_detected(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "sse_server": {"url": "http://localhost:8080/sse", "requires_confirmation": False}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        items = r.json()["data"]
        assert any(s["transport"] == "sse" for s in items)

    def test_connected_server_status_via_mcp_manager(self, client, tokens, app_engine):
        """#2151 — the routes must read app.state.mcp_manager (what the lifespan
        actually sets), so a connected server's runtime status/tools surface."""
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "live": {"command": "python", "args": ["-m", "srv"], "requires_confirmation": True}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = [
            {
                "name": "live",
                "connected": True,
                "tool_count": 1,
                "tools": ["do_thing"],
                "transport": "stdio",
                "endpoint": "python",
            }
        ]
        app.state.mcp_manager = mcp

        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        assert r.status_code == 200
        item = r.json()["data"][0]
        assert item["name"] == "live"
        assert item["status"] == "connected"
        assert [t["name"] for t in item["tools"]] == ["do_thing"]

    def test_requires_auth(self, client, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        app.state.mcp_manager = None
        r = client.get("/api/v1/mcp/servers")
        assert r.status_code == 401

    def test_non_admin_can_list(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        app.state.mcp_manager = None
        r = client.get("/api/v1/mcp/servers", headers=_uh(tokens))
        assert r.status_code == 200


class TestMCPAddServer:
    def test_add_stdio_success_returns_201(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {}
        app.state.config = cfg
        app.state.mcp_manager = None

        with patch("cogtrix_core.api.routes.mcp._persist_mcp_servers"):
            r = client.post(
                "/api/v1/mcp/servers",
                headers=_ah(tokens),
                json={
                    "name": "new_server",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "srv"],
                },
            )
        assert r.status_code == 201
        data = r.json()["data"]
        assert data["name"] == "new_server"
        assert data["transport"] == "stdio"
        assert data["command"] == "python"

    def test_add_conflict_returns_409(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {"new_server": {"command": "python"}}
        app.state.config = cfg
        r = client.post(
            "/api/v1/mcp/servers",
            headers=_ah(tokens),
            json={"name": "new_server", "transport": "stdio", "command": "python"},
        )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_add_persist_failure_returns_503(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {}
        app.state.config = cfg
        app.state.mcp_manager = None

        with patch(
            "cogtrix_core.api.routes.mcp._persist_mcp_servers",
            side_effect=RuntimeError("No config file path"),
        ):
            r = client.post(
                "/api/v1/mcp/servers",
                headers=_ah(tokens),
                json={"name": "new_server", "transport": "stdio", "command": "python"},
            )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        # In-memory state must be rolled back on failure
        assert "new_server" not in cfg.mcp_servers

    def test_non_admin_returns_403(self, client, tokens):
        r = client.post(
            "/api/v1/mcp/servers",
            headers=_uh(tokens),
            json={"name": "new_server", "transport": "stdio", "command": "python"},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/v1/mcp/servers", json={"name": "x", "transport": "stdio"})
        assert r.status_code == 401


class TestMCPGetServer:
    @pytest.fixture(autouse=True)
    def _setup_mcp(self, client, app_engine):
        # client dependency ensures TestClient lifespan runs before we override app.state.config
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "node_server": {"command": "node", "args": ["server.js"], "requires_confirmation": True}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

    def test_get_existing_server(self, client, tokens):
        r = client.get("/api/v1/mcp/servers/node_server", headers=_ah(tokens))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "node_server"
        assert data["command"] == "node"

    def test_get_nonexistent_returns_404(self, client, tokens):
        r = client.get("/api/v1/mcp/servers/nonexistent", headers=_ah(tokens))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "MCP_SERVER_NOT_FOUND"

    def test_no_mcp_client_404(self, client, tokens, app_engine):
        app, _ = app_engine
        # "any_server" is not in the config (only "node_server" from autouse fixture)
        app.state.mcp_manager = None
        r = client.get("/api/v1/mcp/servers/any_server", headers=_ah(tokens))
        assert r.status_code == 404

    def test_requires_auth(self, client):
        r = client.get("/api/v1/mcp/servers/node_server")
        assert r.status_code == 401


class TestMCPRemoveServer:
    @pytest.fixture(autouse=True)
    def _setup_mcp(self, client, app_engine):
        # client dependency ensures TestClient lifespan runs before we override app.state.config
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "py_server": {"command": "py", "args": [], "requires_confirmation": True}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

    def test_remove_existing_returns_204(self, client, tokens):
        with patch("cogtrix_core.api.routes.mcp._persist_mcp_servers"):
            r = client.delete("/api/v1/mcp/servers/py_server", headers=_ah(tokens))
        assert r.status_code == 204
        assert r.content == b""

    def test_remove_nonexistent_returns_404(self, client, tokens):
        r = client.delete("/api/v1/mcp/servers/no_such", headers=_ah(tokens))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "MCP_SERVER_NOT_FOUND"

    def test_non_admin_returns_403(self, client, tokens):
        r = client.delete("/api/v1/mcp/servers/py_server", headers=_uh(tokens))
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.delete("/api/v1/mcp/servers/py_server")
        assert r.status_code == 401

    def test_remove_persist_failure_returns_503(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "py_server": {"command": "py", "args": [], "requires_confirmation": True}
        }
        app.state.config = cfg
        app.state.mcp_manager = None

        with patch(
            "cogtrix_core.api.routes.mcp._persist_mcp_servers",
            side_effect=RuntimeError("No config file path"),
        ):
            r = client.delete("/api/v1/mcp/servers/py_server", headers=_ah(tokens))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
        # In-memory state must be rolled back on failure
        assert "py_server" in cfg.mcp_servers


class TestMCPRestartServer:
    def _setup_mcp_with_restart(self, app_engine, succeed=True):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "restart_srv": {"command": "srv", "args": [], "requires_confirmation": True}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        # #2151: the route now calls restart(...) (not the never-existed
        # restart_server) and re-registers the rebuilt tool dict it returns.
        if succeed:
            mcp.restart = MagicMock(return_value={})
        else:
            mcp.restart = MagicMock(side_effect=RuntimeError("fail"))
        app.state.mcp_manager = mcp

    def test_restart_success(self, client, tokens, app_engine):
        self._setup_mcp_with_restart(app_engine, succeed=True)
        r = client.post("/api/v1/mcp/servers/restart_srv/restart", headers=_ah(tokens))
        assert r.status_code == 200
        assert r.json()["data"]["name"] == "restart_srv"

    def test_restart_failure_returns_503(self, client, tokens, app_engine):
        self._setup_mcp_with_restart(app_engine, succeed=False)
        r = client.post("/api/v1/mcp/servers/restart_srv/restart", headers=_ah(tokens))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MCP_RESTART_FAILED"

    def test_restart_nonexistent_returns_404(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {}
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp
        r = client.post("/api/v1/mcp/servers/nope/restart", headers=_ah(tokens))
        assert r.status_code == 404

    def test_non_admin_returns_403(self, client, tokens, app_engine):
        self._setup_mcp_with_restart(app_engine, succeed=True)
        r = client.post("/api/v1/mcp/servers/restart_srv/restart", headers=_uh(tokens))
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client, app_engine):
        app, _ = app_engine
        r = client.post("/api/v1/mcp/servers/restart_srv/restart")
        assert r.status_code == 401

    def test_restart_uses_asyncio_to_thread(self, client, tokens, app_engine):
        """Regression for #1198: the restart call must be offloaded via asyncio.to_thread.

        #2151: the route now calls ``restart(name, builtin_tool_names=...)`` —
        the blocking manager call still runs off the event loop.
        """
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {
            "restart_srv": {"command": "srv", "args": [], "requires_confirmation": True}
        }
        app.state.config = cfg
        mcp = MagicMock()
        mcp.get_server_info.return_value = []
        restart_mock = MagicMock(return_value={})
        mcp.restart = restart_mock
        app.state.mcp_manager = mcp

        with patch(
            "cogtrix_core.api.routes.mcp.asyncio.to_thread", new_callable=AsyncMock, return_value={}
        ) as mock_to_thread:
            r = client.post("/api/v1/mcp/servers/restart_srv/restart", headers=_ah(tokens))
            assert r.status_code == 200
            mock_to_thread.assert_awaited_once()
            call = mock_to_thread.await_args
            assert call.args[0] is restart_mock
            assert call.args[1] == "restart_srv"
            assert "builtin_tool_names" in call.kwargs


class TestMCPRuntimeWiring:
    """#2151 / #2153 — add/delete/restart must drive the live MCPManager AND
    mirror the change into the live tool registry, then reconcile warm sessions
    — not merely persist YAML.
    """

    def _wire(self, app):
        app.state.tool_registry = SimpleNamespace(tools={}, tool_metadata={})
        app.state.pinned_mcp_tool_names = set()
        sr = MagicMock()
        sr.reconcile_tools = AsyncMock(return_value=0)
        app.state.session_registry = sr
        return sr

    @staticmethod
    def _mcp_tool(server):
        t = MagicMock()
        t.metadata = {"source": "mcp", "server": server, "requires_confirmation": True}
        return t

    def test_add_connects_registers_and_reconciles(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {}
        app.state.config = cfg
        sr = self._wire(app)

        mcp = MagicMock()
        mcp.connect_all.return_value = {"do_thing": self._mcp_tool("new_srv")}
        mcp.get_server_info.return_value = [
            {
                "name": "new_srv",
                "connected": True,
                "tool_count": 1,
                "tools": ["do_thing"],
                "transport": "stdio",
                "endpoint": "python",
            }
        ]
        app.state.mcp_manager = mcp

        with patch("cogtrix_core.api.routes.mcp._persist_mcp_servers"):
            r = client.post(
                "/api/v1/mcp/servers",
                headers=_ah(tokens),
                json={
                    "name": "new_srv",
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "srv"],
                },
            )
        assert r.status_code == 201, r.text
        mcp.connect_all.assert_called_once()
        # Tools mirrored into the live registry + pinned set.
        assert "do_thing" in app.state.tool_registry.tools
        assert "do_thing" in app.state.pinned_mcp_tool_names
        # Active sessions refreshed and real runtime status reported.
        sr.reconcile_tools.assert_awaited_once()
        assert r.json()["data"]["status"] == "connected"

    def test_delete_disconnects_unregisters_and_reconciles(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {"gone": {"command": "py", "args": [], "requires_confirmation": True}}
        app.state.config = cfg
        sr = self._wire(app)
        # A tool from this server is currently registered + pinned.
        app.state.tool_registry.tools["do_thing"] = self._mcp_tool("gone")
        app.state.tool_registry.tool_metadata["do_thing"] = {
            "source": "mcp",
            "server": "gone",
            "pin": True,
        }
        app.state.pinned_mcp_tool_names.add("do_thing")

        mcp = MagicMock()
        mcp.disconnect.return_value = True
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

        with patch("cogtrix_core.api.routes.mcp._persist_mcp_servers"):
            r = client.delete("/api/v1/mcp/servers/gone", headers=_ah(tokens))
        assert r.status_code == 204
        mcp.disconnect.assert_called_once_with("gone")
        # Tool revoked from the live registry + pinned set.
        assert "do_thing" not in app.state.tool_registry.tools
        assert "do_thing" not in app.state.pinned_mcp_tool_names
        sr.reconcile_tools.assert_awaited_once()

    def test_restart_reregisters_tools_and_reconciles(self, client, tokens, app_engine):
        app, _ = app_engine
        cfg = _make_mock_config()
        cfg.mcp_servers = {"r": {"command": "py", "args": [], "requires_confirmation": True}}
        app.state.config = cfg
        sr = self._wire(app)
        # Stale tool present before restart.
        app.state.tool_registry.tools["old_tool"] = self._mcp_tool("r")
        app.state.tool_registry.tool_metadata["old_tool"] = {
            "source": "mcp",
            "server": "r",
            "pin": True,
        }
        app.state.pinned_mcp_tool_names.add("old_tool")

        mcp = MagicMock()
        mcp.restart.return_value = {"new_tool": self._mcp_tool("r")}
        mcp.get_server_info.return_value = []
        app.state.mcp_manager = mcp

        r = client.post("/api/v1/mcp/servers/r/restart", headers=_ah(tokens))
        assert r.status_code == 200
        mcp.restart.assert_called_once()
        # Old tool purged, rebuilt tool registered.
        assert "old_tool" not in app.state.tool_registry.tools
        assert "old_tool" not in app.state.pinned_mcp_tool_names
        assert "new_tool" in app.state.tool_registry.tools
        sr.reconcile_tools.assert_awaited_once()


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------


def _make_mock_config(
    providers=None,
    models=None,
    active_model="default",
    mcp_servers=None,
):
    cfg = MagicMock()
    cfg.active_model_alias = active_model
    cfg.memory_mode = "conversation"
    cfg.prompt_optimizer = True
    cfg.parallel_tool_execution = True
    cfg.context_compression = True
    cfg.debug = False
    cfg.verbose = False
    cfg.config_file_path = None
    cfg.system_prompt = None
    cfg.services = {}

    # Providers
    if providers is None:
        providers = {}
    cfg.providers = providers

    # Models
    if models is None:
        models = {}
    cfg.models = models

    # MCP servers — source of truth for MCP routes
    cfg.mcp_servers = {} if mcp_servers is None else mcp_servers

    return cfg


class TestGetConfig:
    def test_authenticated_user_gets_config(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.get("/api/v1/config", headers=_uh(tokens))
        assert r.status_code == 200
        data = r.json()["data"]
        assert "active_model" in data
        assert "memory_mode" in data
        assert "providers" in data
        assert "models" in data

    def test_admin_gets_config(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.get("/api/v1/config", headers=_ah(tokens))
        assert r.status_code == 200

    def test_non_admin_raw_yaml_is_none(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.get("/api/v1/config", headers=_uh(tokens))
        assert r.json()["data"]["raw_yaml"] is None

    def test_no_auth_returns_401(self, client):
        r = client.get("/api/v1/config")
        assert r.status_code == 401

    def test_config_boolean_fields_present(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.get("/api/v1/config", headers=_uh(tokens))
        data = r.json()["data"]
        assert isinstance(data["prompt_optimizer"], bool)
        assert isinstance(data["parallel_tool_execution"], bool)
        assert isinstance(data["context_compression"], bool)
        assert isinstance(data["debug"], bool)
        assert isinstance(data["verbose"], bool)


class TestPatchConfig:
    def test_admin_toggles_debug(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.patch(
            "/api/v1/config",
            headers=_ah(tokens),
            json={"debug": True},
        )
        assert r.status_code == 200
        assert r.json()["data"]["debug"] is True

    def test_admin_toggles_multiple_flags(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.patch(
            "/api/v1/config",
            headers=_ah(tokens),
            json={"verbose": True, "prompt_optimizer": False, "context_compression": False},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["verbose"] is True
        assert data["prompt_optimizer"] is False
        assert data["context_compression"] is False

    def test_non_admin_returns_403(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.patch(
            "/api/v1/config",
            headers=_uh(tokens),
            json={"debug": True},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.patch("/api/v1/config", json={"debug": True})
        assert r.status_code == 401

    def test_patch_no_config_returns_500(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = None
        r = client.patch(
            "/api/v1/config",
            headers=_ah(tokens),
            json={"debug": False},
        )
        assert r.status_code == 500

    def test_patch_parallel_tool_execution(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config()
        r = client.patch(
            "/api/v1/config",
            headers=_ah(tokens),
            json={"parallel_tool_execution": False},
        )
        assert r.status_code == 200
        assert r.json()["data"]["parallel_tool_execution"] is False


class TestReloadConfig:
    def test_admin_triggers_reload(self, client, tokens):
        # load_config is imported locally in the handler; patch at source module
        with patch("cogtrix_core.config.load_config") as mock_load:
            mock_load.return_value = _make_mock_config()
            r = client.post("/api/v1/config/reload", headers=_ah(tokens))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["reloaded"] is True
        assert "config_file_path" in data
        assert "warnings" in data

    def test_non_admin_returns_403(self, client, tokens):
        r = client.post("/api/v1/config/reload", headers=_uh(tokens))
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/v1/config/reload")
        assert r.status_code == 401

    def test_reload_config_failure_returns_422(self, client, tokens):
        with patch("cogtrix_core.config.load_config") as mock_load:
            mock_load.side_effect = RuntimeError("bad config")
            r = client.post("/api/v1/config/reload", headers=_ah(tokens))
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "CONFIG_INVALID"


class TestListProviders:
    def test_list_providers_authenticated(self, client, tokens, app_engine):
        app, _ = app_engine
        pc = MagicMock()
        pc.type = "openai"
        pc.base_url = None
        pc.api_key = "sk-test"
        app.state.config = _make_mock_config(providers={"openai": pc})
        r = client.get("/api/v1/config/providers", headers=_uh(tokens))
        assert r.status_code == 200
        items = r.json()["data"]
        assert isinstance(items, list)

    def test_list_providers_no_config(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = None
        r = client.get("/api/v1/config/providers", headers=_uh(tokens))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_no_auth_returns_401(self, client):
        r = client.get("/api/v1/config/providers")
        assert r.status_code == 401


class TestGetProvider:
    def test_get_existing_provider(self, client, tokens, app_engine):
        app, _ = app_engine
        pc = MagicMock()
        pc.type = "ollama"
        pc.base_url = "http://localhost:11434"
        pc.api_key = None
        app.state.config = _make_mock_config(providers={"ollama": pc})
        r = client.get("/api/v1/config/providers/ollama", headers=_uh(tokens))
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "ollama"
        assert data["type"] == "ollama"

    def test_get_nonexistent_returns_404(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config(providers={})
        r = client.get("/api/v1/config/providers/nope", headers=_uh(tokens))
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_no_auth_returns_401(self, client):
        r = client.get("/api/v1/config/providers/openai")
        assert r.status_code == 401

    def test_has_api_key_masked(self, client, tokens, app_engine):
        app, _ = app_engine
        pc = MagicMock()
        pc.type = "openai"
        pc.base_url = None
        pc.api_key = "sk-secret-key"
        app.state.config = _make_mock_config(providers={"openai": pc})
        r = client.get("/api/v1/config/providers/openai", headers=_uh(tokens))
        data = r.json()["data"]
        assert data["has_api_key"] is True
        # The raw key should NOT be in the response
        assert "sk-secret-key" not in str(r.text)


class TestSwitchProvider:
    def test_deprecated_endpoint_returns_410(self, client, tokens):
        r = client.post(
            "/api/v1/config/provider",
            headers=_ah(tokens),
            json={"provider": "ollama"},
        )
        assert r.status_code == 410
        assert r.json()["error"]["code"] == "GONE"

    def test_non_admin_returns_403(self, client, tokens):
        r = client.post(
            "/api/v1/config/provider",
            headers=_uh(tokens),
            json={"provider": "ollama"},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/v1/config/provider", json={"provider": "ollama"})
        assert r.status_code == 401


class TestProviderHealth:
    def test_provider_not_found_returns_404(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = _make_mock_config(providers={})
        r = client.post(
            "/api/v1/config/providers/nonexistent/health",
            headers=_uh(tokens),
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/v1/config/providers/openai/health")
        assert r.status_code == 401

    def test_provider_health_reachable(self, client, tokens, app):
        pc = MagicMock()
        pc.type = "openai"
        pc.base_url = None
        pc.api_key = None
        pc.get_base_url = lambda: None
        app.state.config = _make_mock_config(providers={"openai": pc})

        # create_chat_model and get_default_model are imported locally
        with (
            patch("cogtrix_core.providers.create_chat_model") as mock_create,
            patch("cogtrix_core.providers.get_default_model") as mock_default,
        ):
            mock_default.return_value = "gpt-4"
            mock_create.return_value = MagicMock()
            r = client.post(
                "/api/v1/config/providers/openai/health",
                headers=_uh(tokens),
            )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["name"] == "openai"
        assert data["reachable"] is True
        assert data["latency_ms"] >= 0

    def test_provider_health_unreachable(self, client, tokens, app):
        pc = MagicMock()
        pc.type = "openai"
        pc.base_url = None
        pc.api_key = None
        pc.get_base_url = lambda: None
        app.state.config = _make_mock_config(providers={"openai": pc})

        with (
            patch("cogtrix_core.providers.create_chat_model") as mock_create,
            patch("cogtrix_core.providers.get_default_model") as mock_default,
        ):
            mock_default.return_value = "gpt-4"
            mock_create.side_effect = ConnectionError("cannot reach")
            r = client.post(
                "/api/v1/config/providers/openai/health",
                headers=_uh(tokens),
            )
        assert r.status_code == 200
        data = r.json()["data"]
        assert data["reachable"] is False
        assert data["error"] is not None


class TestListModels:
    def test_list_models_no_config(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.config = None
        r = client.get("/api/v1/config/models", headers=_uh(tokens))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_list_models_with_models(self, client, tokens, app_engine):
        app, _ = app_engine
        mc = MagicMock()
        mc.provider = "openai"
        mc.model = "gpt-4"
        mc.temperature = 0.7
        mc.context_window = 128000
        mc.max_tokens = 4096
        cfg = _make_mock_config(models={"gpt4": mc})
        app.state.config = cfg
        r = client.get("/api/v1/config/models", headers=_uh(tokens))
        assert r.status_code == 200
        items = r.json()["data"]
        assert len(items) >= 1
        assert items[0]["alias"] == "gpt4"
        assert items[0]["provider"] == "openai"

    def test_list_models_filter_by_provider(self, client, tokens, app_engine):
        app, _ = app_engine
        mc1 = MagicMock()
        mc1.provider = "openai"
        mc1.model = "gpt-4"
        mc1.temperature = 0.7
        mc1.context_window = None
        mc1.max_tokens = None
        mc2 = MagicMock()
        mc2.provider = "ollama"
        mc2.model = "qwen3"
        mc2.temperature = 0.5
        mc2.context_window = None
        mc2.max_tokens = None
        cfg = _make_mock_config(models={"gpt4": mc1, "qwen3": mc2})
        app.state.config = cfg
        r = client.get("/api/v1/config/models?provider=openai", headers=_uh(tokens))
        assert r.status_code == 200
        items = r.json()["data"]
        assert all(m["provider"] == "openai" for m in items)

    def test_no_auth_returns_401(self, client):
        r = client.get("/api/v1/config/models")
        assert r.status_code == 401


class TestSwitchModel:
    def test_admin_can_switch_model(self, client, tokens, app):
        mc = MagicMock()
        mc.provider = "openai"
        mc.model = "gpt-4"
        mc.temperature = 0.7
        mc.context_window = None
        mc.max_tokens = None
        cfg = _make_mock_config(models={"gpt4": mc})
        app.state.config = cfg

        # _resolve_model and invalidate_llm_caches are imported locally
        with (
            patch("cogtrix_core.config._resolve_model") as mock_resolve,
            patch("cogtrix_core.orchestration.runner.invalidate_llm_caches"),
        ):
            mock_resolve.return_value = None
            r = client.post(
                "/api/v1/config/model",
                headers=_ah(tokens),
                json={"model": "gpt4"},
            )
        assert r.status_code == 200
        assert r.json()["data"] is not None

    def test_non_admin_returns_403(self, client, tokens):
        r = client.post(
            "/api/v1/config/model",
            headers=_uh(tokens),
            json={"model": "gpt4"},
        )
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/v1/config/model", json={"model": "gpt4"})
        assert r.status_code == 401

    def test_missing_model_field_returns_422(self, client, tokens):
        r = client.post("/api/v1/config/model", headers=_ah(tokens), json={})
        assert r.status_code == 422


class TestWizardEndpoints:
    @pytest.fixture(autouse=True)
    def _isolate_wizard_sessions(self):
        """Restore the module-global wizard session store after each test.

        Several tests here start a wizard via the real route (mutating
        ``src.api.routes.config._wizard_sessions``) without cancelling/completing
        it, so the session lingers. In a single-process run that residue leaks into
        later files and breaks order-dependent assertions — e.g.
        ``test_wizard_sessions_dict_access_is_lock_protected`` (#2247). Snapshot the
        store before each test and clear-and-restore it afterwards so collection
        order can't matter.
        """
        import cogtrix_core.api.routes.config as _config_mod

        snapshot = dict(_config_mod._wizard_sessions)
        try:
            yield
        finally:
            _config_mod._wizard_sessions.clear()
            _config_mod._wizard_sessions.update(snapshot)

    def test_start_wizard_non_admin_returns_403(self, client, tokens):
        r = client.post(
            "/api/v1/config/wizard",
            headers=_uh(tokens),
            json={"edit_existing": False},
        )
        assert r.status_code == 403

    def test_start_wizard_no_auth_returns_401(self, client):
        r = client.post("/api/v1/config/wizard", json={"edit_existing": False})
        assert r.status_code == 401

    def test_advance_wizard_nonexistent_id_returns_404(self, client, tokens):
        r = client.post(
            f"/api/v1/config/wizard/{uuid.uuid4()}/step",
            headers=_ah(tokens),
            json={"answer": "some answer"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_cancel_wizard_nonexistent_id_returns_404(self, client, tokens):
        r = client.delete(
            f"/api/v1/config/wizard/{uuid.uuid4()}",
            headers=_ah(tokens),
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "NOT_FOUND"

    def test_cancel_wizard_no_auth_returns_401(self, client):
        r = client.delete(f"/api/v1/config/wizard/{uuid.uuid4()}")
        assert r.status_code == 401

    def test_cancel_wizard_non_admin_returns_403(self, client, tokens):
        r = client.delete(
            f"/api/v1/config/wizard/{uuid.uuid4()}",
            headers=_uh(tokens),
        )
        assert r.status_code == 403

    def test_advance_wizard_non_admin_returns_403(self, client, tokens):
        r = client.post(
            f"/api/v1/config/wizard/{uuid.uuid4()}/step",
            headers=_uh(tokens),
            json={"answer": "test"},
        )
        assert r.status_code == 403

    def test_start_wizard_admin_with_detect_env_mocked(self, client, tokens):
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {"openai_key": None, "ollama": False}
            mock_load.return_value = ""
            r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert r.status_code == 201
        data = r.json()["data"]
        assert "wizard_id" in data
        assert data["step"] == 0
        assert data["complete"] is False
        wid = data["wizard_id"]

        # Now cancel it
        with patch("cogtrix_core.api.routes.config._wizard_detect_env"):
            r2 = client.delete(
                f"/api/v1/config/wizard/{wid}",
                headers=_ah(tokens),
            )
        assert r2.status_code == 200

    def test_advance_wizard_step0_connection_failure(self, client, tokens):
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        wid = start_r.json()["data"]["wizard_id"]

        with patch("cogtrix_core.api.routes.config._wizard_test_connection") as mock_test:
            mock_test.side_effect = ConnectionError("no provider")
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "openai", "model": "gpt-4"}},
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "PROVIDER_UNREACHABLE"

    def test_advance_wizard_step0_success_no_substitute_keyerror(self, client, tokens):
        """Step 0 success must NOT raise KeyError from _WIZARD_SYSTEM_PROMPT.substitute().

        Regression for a bug where the API wizard's substitute() call was missing
        required template fields (bootstrap_type, bootstrap_base_url, bootstrap_has_key,
        production_context) that were present in the CLI wizard but omitted here.
        Template.substitute() raises KeyError for any missing $placeholder.
        """
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert start_r.status_code == 201
        wid = start_r.json()["data"]["wizard_id"]

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                return_value=(MagicMock(), None),
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs content"),
            patch(
                "cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Hi, I can help!"
            ),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "openai", "model": "gpt-4o", "api_key": "sk-test"}},
            )

        # If substitute() raised KeyError it would be a 500; a 200 with step==1 proves all
        # template fields were supplied correctly.
        assert r.status_code == 200, f"unexpected error: {r.text}"
        data = r.json()["data"]
        assert data["step"] == 1
        assert data["complete"] is False

    def test_advance_wizard_step0_success_seeds_human_message(self, client, tokens):
        """Step 0 must seed messages with HumanMessage('Start.') at index 1.

        Regression for a missing seed message that caused HTTP 400 errors on strict
        OpenAI-compatible backends (vLLM, LiteLLM) that reject a messages list
        containing only a SystemMessage.
        """
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert start_r.status_code == 201
        wid = start_r.json()["data"]["wizard_id"]

        captured_messages: list = []

        def _capture_invoke(_, messages):
            captured_messages.extend(messages)
            return "Hi, I can help!"

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                return_value=(MagicMock(), None),
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs content"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", side_effect=_capture_invoke),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "openai", "model": "gpt-4o", "api_key": "sk-test"}},
            )

        assert r.status_code == 200, f"unexpected error: {r.text}"
        assert len(captured_messages) >= 2, "messages list must have at least 2 entries"

        # Index 0 is SystemMessage (system prompt), index 1 must be HumanMessage("Start.")
        from langchain_core.messages import (  # type: ignore[import-untyped]
            HumanMessage,
            SystemMessage,
        )

        assert isinstance(
            captured_messages[0], SystemMessage
        ), f"messages[0] must be SystemMessage, got {type(captured_messages[0])}"
        assert isinstance(
            captured_messages[1], HumanMessage
        ), f"messages[1] must be HumanMessage, got {type(captured_messages[1])}"
        assert (
            captured_messages[1].content == "Start."
        ), f"seed message content must be 'Start.', got {captured_messages[1].content!r}"

    def test_advance_wizard_step0_invoke_failure_falls_back_to_default_question(
        self, client, tokens
    ):
        """If the first LLM invocation raises after a successful connection probe, the
        wizard must fall back to the default question and advance to step 1 (soft-fail).

        Phase 1 (LLM object creation) is the hard-fail gate for PROVIDER_UNREACHABLE.
        Any subsequent _wizard_invoke_llm failure (context overflow, transient error)
        is logged as a warning and the wizard continues so the user can still configure.
        """
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert start_r.status_code == 201
        wid = start_r.json()["data"]["wizard_id"]

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                return_value=(MagicMock(), None),
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch(
                "cogtrix_core.api.routes.config._wizard_invoke_llm",
                side_effect=Exception(
                    "Error code: 400 - {'error': {'message': 'No connected db.'}}"
                ),
            ),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={
                    "data": {
                        "provider_type": "openai",
                        "model": "qwen35",
                        "base_url": "http://192.168.1.1/v1",
                    }
                },
            )

        assert r.status_code == 200, f"expected 200 with fallback question, got: {r.text}"
        data = r.json()["data"]
        assert data["step"] == 1
        assert data["question"] is not None
        assert "Welcome" in data["question"]

    def test_advance_wizard_step0_null_content_falls_back_to_default_question(self, client, tokens):
        """If the LLM returns None/empty content (reasoning models), the wizard must
        return the default first question rather than an empty or None question field.
        """
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert start_r.status_code == 201
        wid = start_r.json()["data"]["wizard_id"]

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                return_value=(MagicMock(), None),
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value=""),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={
                    "data": {
                        "provider_type": "openai",
                        "model": "qwen35",
                        "base_url": "http://192.168.1.1/v1",
                    }
                },
            )

        assert r.status_code == 200, f"empty content must not fail the wizard: {r.text}"
        data = r.json()["data"]
        assert data["step"] == 1
        from cogtrix_core.api.routes.config import _WIZARD_DEFAULT_FIRST_QUESTION

        assert data["question"] == _WIZARD_DEFAULT_FIRST_QUESTION


class TestResolveApiKeyFromExisting:
    """_resolve_api_key_from_existing — key lookup from existing YAML."""

    def _fn(
        self, yaml_text: str, *, provider_name: str | None = None, base_url: str | None = None
    ) -> str | None:
        from cogtrix_core.api.routes.config import _resolve_api_key_from_existing

        return _resolve_api_key_from_existing(
            yaml_text, provider_name=provider_name, base_url=base_url
        )

    _YAML = """
providers:
  spark:
    type: openai
    base_url: "http://192.168.70.254:8080/v1"
    api_key: "sk-correct-key"
  openai:
    type: openai
    api_key: "sk-openai-key"
"""

    def test_resolves_by_provider_name(self) -> None:
        assert self._fn(self._YAML, provider_name="spark") == "sk-correct-key"

    def test_resolves_by_base_url_when_name_missing(self) -> None:
        assert self._fn(self._YAML, base_url="http://192.168.70.254:8080/v1") == "sk-correct-key"

    def test_name_takes_priority_over_base_url(self) -> None:
        # provider_name "openai" matches openai entry, not spark — even though base_url matches spark
        assert (
            self._fn(self._YAML, provider_name="openai", base_url="http://192.168.70.254:8080/v1")
            == "sk-openai-key"
        )

    def test_returns_none_when_no_match(self) -> None:
        assert self._fn(self._YAML, provider_name="unknown", base_url="http://other/v1") is None

    def test_returns_none_on_empty_yaml(self) -> None:
        assert self._fn("", provider_name="spark") is None

    def test_returns_none_on_invalid_yaml(self) -> None:
        assert self._fn(":: bad yaml ::", provider_name="spark") is None

    def test_step0_uses_existing_api_key_when_none_submitted(self, client, tokens) -> None:
        """Step 0 must resolve the api_key from existing config when the WebUI omits it."""
        existing_yaml = """
providers:
  spark:
    type: openai
    base_url: "http://192.168.70.254:8080/v1"
    api_key: "sk-resolved-from-config"
"""
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env") as mock_env,
            patch("cogtrix_core.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = existing_yaml
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert start_r.status_code == 201
        wid = start_r.json()["data"]["wizard_id"]

        captured_key: list[str | None] = []

        def _capture_test(provider_type, model, api_key, base_url):
            captured_key.append(api_key)
            return MagicMock(), None

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture_test
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch(
                "cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="First question?"
            ),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                # No api_key in payload — must be resolved from existing config
                json={
                    "data": {
                        "provider_type": "openai",
                        "provider_name": "spark",
                        "model": "qwen35",
                        "base_url": "http://192.168.70.254:8080/v1",
                    }
                },
            )

        assert r.status_code == 200, r.text
        assert captured_key == [
            "sk-resolved-from-config"
        ], f"api_key must be resolved from existing config, got {captured_key!r}"


class TestResolveWizardProvider:
    """Unit tests for _resolve_wizard_provider()."""

    @staticmethod
    def _fn(provider_name, api_key=None, base_url=None, env=None):
        from cogtrix_core.api.routes.config import _resolve_wizard_provider

        return _resolve_wizard_provider(provider_name, api_key, base_url, env or {})

    def test_native_openai_unchanged(self):
        native, url, key = self._fn("openai", api_key="sk-x", base_url="http://localhost")
        assert native == "openai"
        assert url == "http://localhost"
        assert key == "sk-x"

    def test_native_anthropic_unchanged(self):
        native, url, key = self._fn("anthropic", api_key="ak-x")
        assert native == "anthropic"
        assert key == "ak-x"

    def test_groq_preset_resolves_to_openai(self):
        native, url, key = self._fn("groq", api_key="groq-key")
        assert native == "openai"
        assert url == "https://api.groq.com/openai/v1"
        assert key == "groq-key"

    def test_xai_preset_resolves_to_openai(self):
        native, url, key = self._fn("xai", api_key="xai-key")
        assert native == "openai"
        assert url == "https://api.x.ai/v1"
        assert key == "xai-key"

    def test_preset_custom_base_url_takes_priority(self):
        native, url, key = self._fn("groq", base_url="http://my-groq-proxy")
        assert native == "openai"
        assert url == "http://my-groq-proxy"

    def test_preset_key_resolved_from_env(self):
        native, url, key = self._fn("groq", env={"GROQ_API_KEY": "env-groq-key"})
        assert native == "openai"
        assert key == "env-groq-key"

    def test_preset_explicit_key_overrides_env(self):
        native, url, key = self._fn("xai", api_key="direct-key", env={"XAI_API_KEY": "env-key"})
        assert key == "direct-key"

    def test_deepseek_preset_resolves_to_openai(self):
        native, url, key = self._fn("deepseek", api_key="ds-key")
        assert native == "openai"
        assert url == "https://api.deepseek.com/v1"
        assert key == "ds-key"

    def test_deepseek_preset_key_resolved_from_env(self):
        native, url, key = self._fn("deepseek", env={"DEEPSEEK_API_KEY": "env-ds-key"})
        assert native == "openai"
        assert key == "env-ds-key"

    def test_deepseek_preset_explicit_key_overrides_env(self):
        native, url, key = self._fn(
            "deepseek", api_key="direct-key", env={"DEEPSEEK_API_KEY": "env-key"}
        )
        assert key == "direct-key"

    def test_deepseek_preset_custom_base_url_takes_priority(self):
        native, url, key = self._fn("deepseek", base_url="http://my-deepseek-proxy")
        assert native == "openai"
        assert url == "http://my-deepseek-proxy"

    def test_unknown_provider_returned_unchanged(self):
        native, url, key = self._fn("unknownprovider", api_key="k", base_url="http://x")
        assert native == "unknownprovider"
        assert url == "http://x"
        assert key == "k"


class TestWizardStep0PresetResolution:
    """Integration tests: Step 0 correctly resolves preset providers to native types."""

    def _start_wizard(self, client, tokens):
        with (
            patch("cogtrix_core.api.routes.config._wizard_detect_env", return_value={}),
            patch("cogtrix_core.api.routes.config._wizard_load_existing", return_value=None),
        ):
            r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        assert r.status_code == 201
        return r.json()["data"]["wizard_id"]

    def test_step0_groq_resolves_to_openai(self, client, tokens):
        """provider_type='groq' must pass native_type='openai' to _wizard_test_connection."""
        wid = self._start_wizard(client, tokens)
        captured: list[str] = []

        def _capture(provider_type, model, api_key, base_url):
            captured.append(provider_type)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "groq", "api_key": "sk-groq"}},
            )

        assert r.status_code == 200, r.text
        assert captured == ["openai"], f"expected openai, got {captured!r}"

    def test_step0_xai_resolves_to_openai(self, client, tokens):
        """provider_type='xai' must pass native_type='openai' to _wizard_test_connection."""
        wid = self._start_wizard(client, tokens)
        captured: list[str] = []

        def _capture(provider_type, model, api_key, base_url):
            captured.append(provider_type)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "xai", "api_key": "sk-xai"}},
            )

        assert r.status_code == 200, r.text
        assert captured == ["openai"], f"expected openai, got {captured!r}"

    def test_step0_groq_uses_preset_base_url(self, client, tokens):
        """Step 0 with 'groq' must forward Groq's base_url to _wizard_test_connection."""
        wid = self._start_wizard(client, tokens)
        captured_url: list[str | None] = []

        def _capture(provider_type, model, api_key, base_url):
            captured_url.append(base_url)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "groq", "api_key": "sk-groq"}},
            )

        assert r.status_code == 200, r.text
        assert captured_url == [
            "https://api.groq.com/openai/v1"
        ], f"unexpected base_url: {captured_url!r}"

    def test_step0_groq_uses_preset_default_model_when_none_given(self, client, tokens):
        """Step 0 with 'groq' and no model must use the Groq preset's default model."""
        wid = self._start_wizard(client, tokens)
        captured_model: list[str] = []

        def _capture(provider_type, model, api_key, base_url):
            captured_model.append(model)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                # no "model" field — must default to preset model
                json={"data": {"provider_type": "groq", "api_key": "sk-groq"}},
            )

        assert r.status_code == 200, r.text
        assert captured_model == [
            "llama-3.3-70b-versatile"
        ], f"unexpected model: {captured_model!r}"

    def test_step0_deepseek_resolves_to_openai(self, client, tokens):
        """provider_type='deepseek' must pass native_type='openai' to _wizard_test_connection."""
        wid = self._start_wizard(client, tokens)
        captured: list[str] = []

        def _capture(provider_type, model, api_key, base_url):
            captured.append(provider_type)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "deepseek", "api_key": "sk-ds"}},
            )

        assert r.status_code == 200, r.text
        assert captured == ["openai"], f"expected openai, got {captured!r}"

    def test_step0_deepseek_uses_preset_base_url(self, client, tokens):
        """Step 0 with 'deepseek' must forward DeepSeek's base_url to _wizard_test_connection."""
        wid = self._start_wizard(client, tokens)
        captured_url: list[str | None] = []

        def _capture(provider_type, model, api_key, base_url):
            captured_url.append(base_url)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "deepseek", "api_key": "sk-ds"}},
            )

        assert r.status_code == 200, r.text
        assert captured_url == [
            "https://api.deepseek.com/v1"
        ], f"unexpected base_url: {captured_url!r}"

    def test_step0_deepseek_uses_preset_default_model_when_none_given(self, client, tokens):
        """Step 0 with 'deepseek' and no model must use deepseek-chat as default."""
        wid = self._start_wizard(client, tokens)
        captured_model: list[str] = []

        def _capture(provider_type, model, api_key, base_url):
            captured_model.append(model)
            return MagicMock(), None

        with (
            patch("cogtrix_core.api.routes.config._wizard_test_connection", side_effect=_capture),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                # no "model" field — must default to preset model
                json={"data": {"provider_type": "deepseek", "api_key": "sk-ds"}},
            )

        assert r.status_code == 200, r.text
        assert captured_model == ["deepseek-chat"], f"unexpected model: {captured_model!r}"

    def test_step0_bootstrap_info_type_is_native(self, client, tokens):
        """ws['bootstrap_info']['type'] must be 'openai', not 'groq'."""
        wid = self._start_wizard(client, tokens)

        with (
            patch(
                "cogtrix_core.api.routes.config._wizard_test_connection",
                return_value=(MagicMock(), None),
            ),
            patch("cogtrix_core.api.routes.config._wizard_load_docs", return_value="docs"),
            patch("cogtrix_core.api.routes.config._wizard_invoke_llm", return_value="Q?"),
        ):
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "groq", "api_key": "sk-groq"}},
            )

        assert r.status_code == 200, r.text
        # Inspect the stored wizard session state via the internal registry
        from cogtrix_core.api.routes.config import _wizard_sessions

        ws = _wizard_sessions.get(wid)
        assert ws is not None
        assert (
            ws["bootstrap_info"]["type"] == "openai"
        ), f"bootstrap_info type should be 'openai', got {ws['bootstrap_info']['type']!r}"
