"""Phase 4 API tests: Tools, Memory, Config, MCP, System endpoints.

Tests cover:
- Tool listing with search filter and include_mcp filter
- Tool detail with 404 for missing tool
- Per-session tool status classification
- PATCH session tools: load, unload, enable, disable, auto_approve, revoke_approval
- Memory GET/DELETE/PATCH
- Config GET/PATCH/reload/providers/models
- System info and debug toggle
- MCP server listing (empty and with a mock client)

State injection strategy:
    The FastAPI lifespan sets app.state.* during TestClient startup.
    We override app.state inside the `with TestClient(app) as client:` block
    (after lifespan has run) so our mocks win.
"""

from __future__ import annotations

import binascii
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.api imports
# ---------------------------------------------------------------------------

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

# ---------------------------------------------------------------------------
# Imports after env setup
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from src.api.auth import create_access_token  # noqa: E402

# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_tool_registry() -> MagicMock:
    """Build a mock ToolRegistry with two tools (one MCP, one not)."""
    mock_tool = MagicMock()
    mock_tool.description = "A test tool\nWith more details."
    mock_tool.args_schema = None

    mock_mcp_tool = MagicMock()
    mock_mcp_tool.description = "An MCP tool\nProvided by external server."
    mock_mcp_tool.args_schema = None

    registry = MagicMock()
    registry.tools = {"test_tool": mock_tool, "another_tool": mock_mcp_tool}
    registry.tool_metadata = {
        "test_tool": {"requires_confirmation": False, "source": None},
        "another_tool": {"requires_confirmation": True, "source": "mcp", "server": "my-mcp"},
    }
    registry.requires_confirmation.side_effect = lambda n: registry.tool_metadata.get(n, {}).get(
        "requires_confirmation", False
    )
    registry.is_mcp_tool.side_effect = (
        lambda n: registry.tool_metadata.get(n, {}).get("source") == "mcp"
    )
    registry.get_tool_server.side_effect = lambda n: registry.tool_metadata.get(n, {}).get("server")
    return registry


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.provider = "ollama"
    cfg.model = None
    cfg.memory_mode = "conversation"
    cfg.prompt_optimizer = True
    cfg.parallel_tool_execution = True
    cfg.context_compression = True
    cfg.debug = False
    cfg.verbose = False
    cfg.config_file_path = None
    cfg.providers = {}
    cfg.models = {}
    cfg.mcp_servers = {}
    return cfg


def _make_live_session(session_id: str) -> MagicMock:
    """Build a mock ApiSession."""
    from src.orchestration.session_state import SessionState

    ss = SessionState(no_confirm=True)
    live = MagicMock()
    live.id = session_id
    live.session_state = ss
    live.memory_manager = None
    live.config = {"memory_mode": "conversation"}
    live.token_counts = {"context_window": 131072, "input_tokens": 0, "output_tokens": 0}
    return live


def _make_session_registry(live_session: MagicMock) -> MagicMock:
    sr = MagicMock()

    async def _async_get(sid: str, db: object) -> MagicMock | None:
        if sid == live_session.id:
            return live_session
        return None

    sr.get_or_warm = _async_get
    return sr


@contextmanager
def _api_client(
    extra_state: dict | None = None,
) -> Iterator[tuple[TestClient, MagicMock, MagicMock, str, str]]:
    """Context manager yielding (client, registry, config, admin_token, user_token).

    State is injected *inside* the TestClient context so it overwrites whatever
    the lifespan put on app.state.
    """
    from src.api.app import create_app

    registry = _make_tool_registry()
    config = _make_config()
    admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
    user_token = create_access_token(user_id=str(uuid.uuid4()), role="user")

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        # Override state AFTER lifespan startup
        app.state.tool_registry = registry
        app.state.config = config
        app.state.session_registry = None
        app.state.mcp_client = None
        if extra_state:
            for k, v in extra_state.items():
                setattr(app.state, k, v)
        yield client, registry, config, admin_token, user_token


# ---------------------------------------------------------------------------
# Tool endpoint tests
# ---------------------------------------------------------------------------


class TestToolEndpoints:
    def test_list_tools_returns_all(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get("/api/v1/tools", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["data"]["items"] is not None
        assert len(data["data"]["items"]) == 2

    def test_list_tools_search_filter(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools",
                params={"search": "test"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert len(items) == 1
        assert items[0]["name"] == "test_tool"

    def test_list_tools_exclude_mcp(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools",
                params={"include_mcp": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert all(not item["is_mcp"] for item in items)
        assert len(items) == 1

    def test_list_tools_total_count(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get("/api/v1/tools", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 2

    def test_get_tool_detail(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools/test_tool",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["name"] == "test_tool"
        assert data["is_mcp"] is False

    def test_get_mcp_tool_detail(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools/another_tool",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["is_mcp"] is True
        assert data["mcp_server"] == "my-mcp"

    def test_get_tool_not_found(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_list_tools_requires_auth(self) -> None:
        with _api_client() as (client, *_):
            resp = client.get("/api/v1/tools")
        assert resp.status_code == 401

    def test_list_tools_invalid_cursor(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/tools",
                params={"cursor": "not-valid-base64!!!"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_CURSOR"


# ---------------------------------------------------------------------------
# Session tool status tests
# ---------------------------------------------------------------------------


class TestSessionToolEndpoints:
    def test_get_session_tools_not_found_when_no_registry(self) -> None:
        sid = str(uuid.uuid4())
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                f"/api/v1/sessions/{sid}/tools",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404

    def test_get_session_tools_status_classification(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        live.session_state.loaded_tools.add("test_tool")
        live.session_state.denials.add("another_tool")
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.get(
                f"/api/v1/sessions/{sid}/tools",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "active"
        assert items["another_tool"] == "disabled"

    def test_patch_session_tools_load(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"load": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "active"

    def test_patch_session_tools_disable(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        live.session_state.loaded_tools.add("test_tool")
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"disable": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "disabled"

    def test_patch_session_tools_auto_approve(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"auto_approve": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "auto_approved"

    def test_patch_session_tools_revoke_approval(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        live.session_state.approvals.add("test_tool")
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"revoke_approval": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "on_demand"

    def test_patch_session_tools_tool_not_found(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"load": ["nonexistent_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "TOOL_NOT_FOUND"

    def test_patch_session_tools_enable(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        live.session_state.denials.add("test_tool")
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"enable": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "on_demand"

    def test_patch_session_tools_unload(self) -> None:
        sid = str(uuid.uuid4())
        live = _make_live_session(sid)
        live.session_state.loaded_tools.add("test_tool")
        sr = _make_session_registry(live)

        with _api_client(extra_state={"session_registry": sr}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.patch(
                f"/api/v1/sessions/{sid}/tools",
                json={"unload": ["test_tool"]},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        items = {item["name"]: item["status"] for item in resp.json()["data"]}
        assert items["test_tool"] == "on_demand"


# ---------------------------------------------------------------------------
# Memory endpoint tests
# ---------------------------------------------------------------------------


class TestMemoryEndpoints:
    def _make_memory_client(self) -> tuple:
        sid = str(uuid.uuid4())
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")

        mock_mm = MagicMock()
        mock_mm.to_dict.return_value = {
            "mode": "conversation",
            "summary": None,
            "window_size": 10,
            "summarized_messages": 5,
            "tokens_used": 1000,
            "vector_recall_enabled": False,
            "mode_meta": {},
        }

        live = _make_live_session(sid)
        live.memory_manager = mock_mm
        sr = _make_session_registry(live)

        return sid, admin_token, mock_mm, live, sr

    def test_get_memory(self) -> None:
        sid, admin_token, mock_mm, live, sr = self._make_memory_client()
        with _api_client(extra_state={"session_registry": sr}) as (client, *_):
            resp = client.get(
                f"/api/v1/sessions/{sid}/memory",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["session_id"] == sid
        assert data["mode"] == "conversation"
        assert data["window_messages"] == 10
        assert data["summarized_messages"] == 5
        assert data["tokens_used"] == 1000

    def test_clear_memory_calls_clear(self) -> None:
        sid, admin_token, mock_mm, live, sr = self._make_memory_client()
        mock_mm.clear = MagicMock()
        with _api_client(extra_state={"session_registry": sr}) as (client, *_):
            resp = client.delete(
                f"/api/v1/sessions/{sid}/memory",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        mock_mm.clear.assert_called_once()

    def test_clear_memory_no_clear_method(self) -> None:
        """When memory manager has no clear(), fall back to resetting _messages."""
        sid, admin_token, mock_mm, live, sr = self._make_memory_client()
        # Remove the 'clear' attribute so hasattr returns False
        mock_mm_obj = MagicMock(spec=["to_dict", "_messages", "_summary"])
        mock_mm_obj.to_dict.return_value = mock_mm.to_dict.return_value
        mock_mm_obj._messages = ["msg1"]
        mock_mm_obj._summary = "old"
        live.memory_manager = mock_mm_obj

        with _api_client(extra_state={"session_registry": sr}) as (client, *_):
            resp = client.delete(
                f"/api/v1/sessions/{sid}/memory",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200

    def test_switch_memory_mode(self) -> None:
        sid, admin_token, mock_mm, live, sr = self._make_memory_client()
        new_mm = MagicMock()
        new_mm.to_dict.return_value = {
            "mode": "reasoning",
            "summary": None,
            "window_size": 0,
            "summarized_messages": 0,
            "tokens_used": 0,
            "vector_recall_enabled": False,
            "mode_meta": {},
        }
        with (
            _api_client(extra_state={"session_registry": sr}) as (client, *_),
            patch("src.memory.MemoryFactory") as mock_factory,
            patch("src.memory.JsonFileMemoryStore"),
        ):
            mock_factory.is_registered.return_value = True
            mock_factory.create.return_value = new_mm
            resp = client.patch(
                f"/api/v1/sessions/{sid}/memory",
                json={"mode": "reasoning"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert live.memory_manager is new_mm

    def test_memory_session_not_found(self) -> None:
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        with _api_client() as (client, *_):
            resp = client.get(
                f"/api/v1/sessions/{uuid.uuid4()}/memory",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "SESSION_NOT_FOUND"

    def test_memory_requires_auth(self) -> None:
        with _api_client() as (client, *_):
            resp = client.get(f"/api/v1/sessions/{uuid.uuid4()}/memory")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Config endpoint tests
# ---------------------------------------------------------------------------


class TestConfigEndpoints:
    def test_get_config(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["provider"] == "ollama"
        assert data["prompt_optimizer"] is True
        assert data["context_compression"] is True

    def test_get_config_user_no_raw_yaml(self) -> None:
        with _api_client() as (client, registry, config, admin_token, user_token):
            resp = client.get(
                "/api/v1/config",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["raw_yaml"] is None

    def test_patch_config_debug_toggle(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.patch(
                "/api/v1/config",
                json={"debug": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert config.debug is True

    def test_patch_config_multiple_flags(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.patch(
                "/api/v1/config",
                json={"debug": False, "verbose": True, "prompt_optimizer": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert config.verbose is True
        assert config.prompt_optimizer is False

    def test_patch_config_requires_admin(self) -> None:
        with _api_client() as (client, registry, config, admin_token, user_token):
            resp = client.patch(
                "/api/v1/config",
                json={"debug": True},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403

    def test_reload_config(self) -> None:
        mock_new_cfg = MagicMock()
        mock_new_cfg.config_file_path = None
        with (
            _api_client() as (client, registry, config, admin_token, _),
            patch("src.config.Config", return_value=mock_new_cfg),
        ):
            resp = client.post(
                "/api/v1/config/reload",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["reloaded"] is True

    def test_reload_config_requires_admin(self) -> None:
        with _api_client() as (client, registry, config, admin_token, user_token):
            resp = client.post(
                "/api/v1/config/reload",
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403

    def test_list_providers(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/config/providers",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        providers = resp.json()["data"]
        assert isinstance(providers, list)

    def test_list_models_empty(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/config/models",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_wizard_start_returns_501(self) -> None:
        from fastapi.testclient import TestClient as _TC

        from src.api.app import create_app

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        with _TC(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/config/wizard",
                json={},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 501

    def test_wizard_step_returns_501(self) -> None:
        from fastapi.testclient import TestClient as _TC

        from src.api.app import create_app

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        wid = str(uuid.uuid4())
        with _TC(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/config/wizard/{wid}/step",
                json={"answer": "ollama"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 501

    def test_wizard_cancel_returns_501(self) -> None:
        from fastapi.testclient import TestClient as _TC

        from src.api.app import create_app

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        wid = str(uuid.uuid4())
        with _TC(app, raise_server_exceptions=False) as client:
            resp = client.delete(
                f"/api/v1/config/wizard/{wid}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 501

    def test_get_provider_not_found(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/config/providers/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404

    def test_switch_provider_known_type(self) -> None:
        with (
            _api_client() as (client, registry, config, admin_token, _),
            patch("src.orchestration.runner.invalidate_llm_caches", return_value=None),
        ):
            resp = client.post(
                "/api/v1/config/provider",
                json={"provider": "openai"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200

    def test_switch_model(self) -> None:
        with (
            _api_client() as (client, registry, config, admin_token, _),
            patch("src.orchestration.runner.invalidate_llm_caches", return_value=None),
        ):
            resp = client.post(
                "/api/v1/config/model",
                json={"model": "gpt-4.1-mini"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert config.model == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# MCP endpoint tests
# ---------------------------------------------------------------------------


class TestMCPEndpoints:
    def test_list_mcp_servers_empty_when_no_client(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/mcp/servers",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_mcp_servers_with_client(self) -> None:
        from src.mcp_client import MCPServerConfig

        mock_client = MagicMock()
        sc = MCPServerConfig(name="my-server", command="npx", args=["some-pkg"])
        mock_client.servers = {"my-server": sc}

        with _api_client(extra_state={"mcp_client": mock_client}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.get(
                "/api/v1/mcp/servers",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["name"] == "my-server"
        assert data[0]["transport"] == "stdio"
        assert data[0]["status"] == "connected"

    def test_get_mcp_server_details(self) -> None:
        from src.mcp_client import MCPServerConfig

        mock_client = MagicMock()
        sc = MCPServerConfig(name="srv1", command="python", args=["-m", "server"])
        mock_client.servers = {"srv1": sc}

        with _api_client(extra_state={"mcp_client": mock_client}) as (
            client,
            registry,
            config,
            admin_token,
            _,
        ):
            resp = client.get(
                "/api/v1/mcp/servers/srv1",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["name"] == "srv1"

    def test_get_mcp_server_not_found(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/mcp/servers/nonexistent",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MCP_SERVER_NOT_FOUND"

    def test_add_mcp_server_returns_501(self) -> None:
        from fastapi.testclient import TestClient as _TC

        from src.api.app import create_app

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        with _TC(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/mcp/servers",
                json={"name": "test", "transport": "stdio", "command": "npx"},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 501

    def test_remove_mcp_server_returns_501(self) -> None:
        from src.mcp_client import MCPServerConfig

        mock_client = MagicMock()
        sc = MCPServerConfig(name="srv1", command="python", args=[])
        mock_client.servers = {"srv1": sc}

        from fastapi.testclient import TestClient as _TC

        from src.api.app import create_app

        app = create_app()
        admin_token = create_access_token(user_id=str(uuid.uuid4()), role="admin")
        with _TC(app, raise_server_exceptions=False) as client:
            app.state.mcp_client = mock_client
            resp = client.delete(
                "/api/v1/mcp/servers/srv1",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 501

    def test_list_mcp_servers_requires_auth(self) -> None:
        with _api_client() as (client, *_):
            resp = client.get("/api/v1/mcp/servers")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# System endpoint tests
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    def test_system_info(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "version" in data
        assert "uptime_s" in data
        assert data["api_version"] == "v1"
        assert data["uptime_s"] >= 0
        assert "python_version" in data
        assert "platform" in data

    def test_system_info_version_matches(self) -> None:
        from src._version import __version__

        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.get(
                "/api/v1/system/info",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.json()["data"]["version"] == __version__

    def test_system_info_requires_auth(self) -> None:
        with _api_client() as (client, *_):
            resp = client.get("/api/v1/system/info")
        assert resp.status_code == 401

    def test_toggle_debug_enables_debug(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert config.debug is True

    def test_toggle_debug_with_verbose(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": False, "verbose": True},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        assert config.verbose is True

    def test_toggle_debug_requires_admin(self) -> None:
        with _api_client() as (client, registry, config, admin_token, user_token):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": True},
                headers={"Authorization": f"Bearer {user_token}"},
            )
        assert resp.status_code == 403

    def test_toggle_debug_returns_system_info(self) -> None:
        with _api_client() as (client, registry, config, admin_token, _):
            resp = client.post(
                "/api/v1/system/debug",
                json={"debug": False},
                headers={"Authorization": f"Bearer {admin_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "version" in data
        assert "uptime_s" in data


# ---------------------------------------------------------------------------
# Pagination helper tests
# ---------------------------------------------------------------------------


class TestPaginationHelpers:
    def test_encode_decode_cursor(self) -> None:
        from src.api.pagination import decode_cursor, encode_cursor

        original = "some-tool-name"
        encoded = encode_cursor(original)
        assert encoded != original
        assert decode_cursor(encoded) == original

    def test_paginate_list_basic(self) -> None:
        from src.api.pagination import paginate_list

        items = ["a", "b", "c", "d", "e"]
        page, next_cursor, has_more = paginate_list(items, None, 3)
        assert page == ["a", "b", "c"]
        assert has_more is True
        assert next_cursor is not None

    def test_paginate_list_last_page(self) -> None:
        from src.api.pagination import paginate_list

        items = ["a", "b"]
        page, next_cursor, has_more = paginate_list(items, None, 10)
        assert page == ["a", "b"]
        assert has_more is False
        assert next_cursor is None

    def test_paginate_list_with_cursor(self) -> None:
        from src.api.pagination import paginate_list

        items = ["a", "b", "c", "d"]
        page, next_cursor, has_more = paginate_list(items, "b", 2)
        assert page == ["c", "d"]
        assert has_more is False

    def test_paginate_list_limit_clamped(self) -> None:
        from src.api.pagination import paginate_list

        items = list(range(100))
        page, _, _ = paginate_list(items, None, 0)  # 0 becomes 1
        assert len(page) == 1

    def test_decode_cursor_invalid(self) -> None:
        from src.api.pagination import decode_cursor

        # "!!!" characters make base64 decoding produce non-UTF8 bytes -> UnicodeDecodeError
        # OR binascii.Error for truly malformed padding — either is expected
        with pytest.raises((binascii.Error, UnicodeDecodeError, ValueError)):
            decode_cursor("not-valid-base64!!!")
