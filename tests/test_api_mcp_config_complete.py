"""Comprehensive MCP and config endpoint coverage.

MCP endpoints:
    GET    /api/v1/mcp/servers             — list
    POST   /api/v1/mcp/servers             — add (501)
    GET    /api/v1/mcp/servers/{name}      — get detail
    DELETE /api/v1/mcp/servers/{name}      — remove (501 when found)
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
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_VALID_PASSWORD = "TestPass1!"  # lowercase + uppercase + digit + special


@pytest.fixture()
def app():
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
        app.state.mcp_client = None
        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        assert r.status_code == 200
        assert r.json()["data"] == []

    def test_with_mcp_client_returns_servers(self, client, tokens, app_engine):
        app, _ = app_engine
        sc = MagicMock()
        sc.url = None
        sc.command = "python"
        sc.args = ["-m", "myserver"]
        sc.requires_confirmation = True

        mcp = MagicMock()
        mcp.servers = {"my_server": sc}
        app.state.mcp_client = mcp

        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        assert r.status_code == 200
        items = r.json()["data"]
        assert len(items) == 1
        assert items[0]["name"] == "my_server"
        assert items[0]["transport"] == "stdio"

    def test_sse_transport_detected(self, client, tokens, app_engine):
        app, _ = app_engine
        sc = MagicMock()
        sc.url = "http://localhost:8080/sse"
        sc.command = None
        sc.args = []
        sc.requires_confirmation = False

        mcp = MagicMock()
        mcp.servers = {"sse_server": sc}
        app.state.mcp_client = mcp

        r = client.get("/api/v1/mcp/servers", headers=_ah(tokens))
        items = r.json()["data"]
        assert any(s["transport"] == "sse" for s in items)

    def test_requires_auth(self, client, app_engine):
        app, _ = app_engine
        app.state.mcp_client = None
        r = client.get("/api/v1/mcp/servers")
        assert r.status_code == 401

    def test_non_admin_can_list(self, client, tokens, app_engine):
        app, _ = app_engine
        app.state.mcp_client = None
        r = client.get("/api/v1/mcp/servers", headers=_uh(tokens))
        assert r.status_code == 200


class TestMCPAddServer:
    def test_add_returns_501(self, client, tokens):
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
        assert r.status_code == 501
        assert r.json()["error"]["code"] == "NOT_IMPLEMENTED"

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
    def _setup_mcp(self, app_engine):
        app, _ = app_engine
        sc = MagicMock()
        sc.url = None
        sc.command = "node"
        sc.args = ["server.js"]
        sc.requires_confirmation = True
        mcp = MagicMock()
        mcp.servers = {"node_server": sc}
        app.state.mcp_client = mcp

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
        app.state.mcp_client = None
        r = client.get("/api/v1/mcp/servers/any_server", headers=_ah(tokens))
        assert r.status_code == 404

    def test_requires_auth(self, client):
        r = client.get("/api/v1/mcp/servers/node_server")
        assert r.status_code == 401


class TestMCPRemoveServer:
    @pytest.fixture(autouse=True)
    def _setup_mcp(self, app_engine):
        app, _ = app_engine
        sc = MagicMock()
        sc.url = None
        sc.command = "py"
        sc.args = []
        sc.requires_confirmation = True
        mcp = MagicMock()
        mcp.servers = {"py_server": sc}
        app.state.mcp_client = mcp

    def test_remove_existing_returns_501(self, client, tokens):
        r = client.delete("/api/v1/mcp/servers/py_server", headers=_ah(tokens))
        assert r.status_code == 501
        assert r.json()["error"]["code"] == "NOT_IMPLEMENTED"

    def test_remove_nonexistent_returns_404(self, client, tokens):
        r = client.delete("/api/v1/mcp/servers/no_such", headers=_ah(tokens))
        assert r.status_code == 404

    def test_non_admin_returns_403(self, client, tokens):
        r = client.delete("/api/v1/mcp/servers/py_server", headers=_uh(tokens))
        assert r.status_code == 403

    def test_no_auth_returns_401(self, client):
        r = client.delete("/api/v1/mcp/servers/py_server")
        assert r.status_code == 401


class TestMCPRestartServer:
    def _setup_mcp_with_restart(self, app_engine, succeed=True):
        app, _ = app_engine
        sc = MagicMock()
        sc.url = None
        sc.command = "srv"
        sc.args = []
        sc.requires_confirmation = True
        mcp = MagicMock()
        mcp.servers = {"restart_srv": sc}
        if succeed:
            mcp.restart_server = MagicMock(return_value=None)
        else:
            mcp.restart_server = MagicMock(side_effect=RuntimeError("fail"))
        app.state.mcp_client = mcp

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
        mcp = MagicMock()
        mcp.servers = {}
        app.state.mcp_client = mcp
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


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------


def _make_mock_config(
    providers=None,
    models=None,
    active_model="default",
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
        with patch("src.config.load_config") as mock_load:
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
        with patch("src.config.load_config") as mock_load:
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
            patch("src.providers.create_chat_model") as mock_create,
            patch("src.providers.get_default_model") as mock_default,
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
            patch("src.providers.create_chat_model") as mock_create,
            patch("src.providers.get_default_model") as mock_default,
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
            patch("src.config._resolve_model") as mock_resolve,
            patch("src.orchestration.runner.invalidate_llm_caches"),
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
            patch("src.api.routes.config._wizard_detect_env") as mock_env,
            patch("src.api.routes.config._wizard_load_existing") as mock_load,
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
        with patch("src.api.routes.config._wizard_detect_env"):
            r2 = client.delete(
                f"/api/v1/config/wizard/{wid}",
                headers=_ah(tokens),
            )
        assert r2.status_code == 200

    def test_advance_wizard_step0_connection_failure(self, client, tokens):
        with (
            patch("src.api.routes.config._wizard_detect_env") as mock_env,
            patch("src.api.routes.config._wizard_load_existing") as mock_load,
        ):
            mock_env.return_value = {}
            mock_load.return_value = ""
            start_r = client.post(
                "/api/v1/config/wizard",
                headers=_ah(tokens),
                json={"edit_existing": False},
            )
        wid = start_r.json()["data"]["wizard_id"]

        with patch("src.api.routes.config._wizard_test_connection") as mock_test:
            mock_test.side_effect = ConnectionError("no provider")
            r = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                headers=_ah(tokens),
                json={"data": {"provider_type": "openai", "model": "gpt-4"}},
            )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "PROVIDER_UNREACHABLE"
