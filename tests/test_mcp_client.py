"""Tests for src/mcp_client.py."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.mcp_client import (
    MCPConnection,
    MCPManager,
    MCPServerConfig,
    _result_to_str,
    json_schema_to_pydantic,
)

# ── MCPServerConfig ────────────────────────────────────────────────────────────


class TestMCPServerConfig:
    def test_default_values(self):
        cfg = MCPServerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.command is None
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.url is None
        assert cfg.headers is None
        assert cfg.requires_confirmation is True
        assert cfg.timeout == 30

    def test_stdio_config(self):
        cfg = MCPServerConfig(name="stdio-server", command="npx", args=["-y", "mcp-server"])
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "mcp-server"]
        assert cfg.url is None

    def test_sse_config(self):
        cfg = MCPServerConfig(name="sse-server", url="http://localhost:8080/sse")
        assert cfg.url == "http://localhost:8080/sse"
        assert cfg.command is None

    def test_custom_timeout_and_confirmation(self):
        cfg = MCPServerConfig(name="server", command="cmd", timeout=60, requires_confirmation=False)
        assert cfg.timeout == 60
        assert cfg.requires_confirmation is False

    def test_env_and_headers(self):
        cfg = MCPServerConfig(
            name="server",
            url="http://example.com",
            env={"MY_VAR": "val"},
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.env == {"MY_VAR": "val"}
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_args_default_is_independent_per_instance(self):
        cfg1 = MCPServerConfig(name="a")
        cfg2 = MCPServerConfig(name="b")
        cfg1.args.append("x")
        assert cfg2.args == []


# ── json_schema_to_pydantic ───────────────────────────────────────────────────


class TestJsonSchemaToPydantic:
    def test_simple_string_property(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        model = json_schema_to_pydantic("search_tool", schema)
        fields = model.model_fields
        assert "query" in fields
        assert fields["query"].annotation is str

    def test_integer_property(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        model = json_schema_to_pydantic("counter", schema)
        assert model.model_fields["count"].annotation is int

    def test_float_property(self):
        schema = {
            "type": "object",
            "properties": {"ratio": {"type": "number"}},
            "required": ["ratio"],
        }
        model = json_schema_to_pydantic("ratiotool", schema)
        assert model.model_fields["ratio"].annotation is float

    def test_boolean_property(self):
        schema = {
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
            "required": ["verbose"],
        }
        model = json_schema_to_pydantic("btool", schema)
        assert model.model_fields["verbose"].annotation is bool

    def test_array_property(self):
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
        }
        model = json_schema_to_pydantic("listtool", schema)
        assert model.model_fields["items"].annotation is list

    def test_object_property(self):
        schema = {
            "type": "object",
            "properties": {"data": {"type": "object"}},
            "required": ["data"],
        }
        model = json_schema_to_pydantic("objtool", schema)
        assert model.model_fields["data"].annotation is dict

    def test_unknown_type_defaults_to_str(self):
        schema = {
            "type": "object",
            "properties": {"mystery": {"type": "exotic_type"}},
            "required": ["mystery"],
        }
        model = json_schema_to_pydantic("tool", schema)
        assert model.model_fields["mystery"].annotation is str

    def test_required_field_has_no_default(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        model = json_schema_to_pydantic("t", schema)
        field = model.model_fields["q"]
        assert field.is_required()

    def test_optional_field_has_none_default(self):
        schema = {"type": "object", "properties": {"opt": {"type": "string"}}}
        model = json_schema_to_pydantic("t", schema)
        field = model.model_fields["opt"]
        assert not field.is_required()
        assert field.default is None

    def test_no_required_key_all_optional(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        model = json_schema_to_pydantic("t", schema)
        assert not model.model_fields["a"].is_required()
        assert not model.model_fields["b"].is_required()

    def test_empty_properties(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic("empty_tool", schema)
        assert model.model_fields == {}

    def test_no_properties_key(self):
        schema = {"type": "object"}
        model = json_schema_to_pydantic("noprops", schema)
        assert model.model_fields == {}

    def test_tool_name_sanitization_special_chars(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic("my-tool/name", schema)
        assert re.search(r"[^A-Za-z0-9_]", model.__name__) is None

    def test_tool_name_sanitization_leading_digit(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic("123tool", schema)
        assert model.__name__[0] == "_"

    def test_class_name_derived_from_tool_name(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic("MyTool", schema)
        assert "MyTool" in model.__name__

    def test_empty_tool_name_fallback(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic("---", schema)
        assert model.__name__ == "MCPToolInput"

    def test_field_description_preserved(self):
        schema = {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "The search query"}},
            "required": ["q"],
        }
        model = json_schema_to_pydantic("t", schema)
        desc = model.model_fields["q"].description
        assert desc == "The search query"

    def test_multiple_fields_mixed_required(self):
        schema = {
            "type": "object",
            "properties": {
                "req": {"type": "string"},
                "opt": {"type": "integer"},
            },
            "required": ["req"],
        }
        model = json_schema_to_pydantic("mixed", schema)
        assert model.model_fields["req"].is_required()
        assert not model.model_fields["opt"].is_required()


# ── _result_to_str ─────────────────────────────────────────────────────────────


def _make_result(content: list[Any], is_error: bool = False) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.isError = is_error
    return result


def _make_text(text: str) -> MagicMock:
    item = MagicMock(spec=["text"])
    item.text = text
    return item


def _make_image(mime: str) -> MagicMock:
    item = MagicMock(spec=["mimeType"])
    item.mimeType = mime
    return item


class TestResultToStr:
    def test_single_text_content(self):
        result = _make_result([_make_text("hello world")])
        assert _result_to_str(result) == "hello world"

    def test_multiple_text_blocks_joined_with_newline(self):
        result = _make_result([_make_text("line1"), _make_text("line2")])
        assert _result_to_str(result) == "line1\nline2"

    def test_image_content_formatted(self):
        result = _make_result([_make_image("image/png")])
        assert _result_to_str(result) == "[image: image/png]"

    def test_mixed_text_and_image(self):
        result = _make_result([_make_text("caption"), _make_image("image/jpeg")])
        assert _result_to_str(result) == "caption\n[image: image/jpeg]"

    def test_is_error_prefix(self):
        result = _make_result([_make_text("something went wrong")], is_error=True)
        assert _result_to_str(result) == "Error: something went wrong"

    def test_empty_content_list(self):
        result = _make_result([])
        assert _result_to_str(result) == ""

    def test_is_error_with_empty_content(self):
        result = _make_result([], is_error=True)
        assert _result_to_str(result) == "Error: "

    def test_fallback_str_for_unknown_item(self):
        class _Unknown:
            def __str__(self) -> str:
                return "<unknown>"

        result = _make_result([_Unknown()])
        assert "<unknown>" in _result_to_str(result)

    def test_none_content_treated_as_empty(self):
        result = MagicMock()
        result.content = None
        result.isError = False
        assert _result_to_str(result) == ""


# ── MCPManager helpers ────────────────────────────────────────────────────────


def _make_mock_tool(name: str = "test_tool", description: str = "A test tool") -> MagicMock:
    mock_tool = MagicMock()
    mock_tool.name = name
    mock_tool.description = description
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }
    return mock_tool


# ── MCPManager ────────────────────────────────────────────────────────────────


class TestMCPManager:
    def test_connect_all_empty_config_returns_empty_dict(self):
        manager = MCPManager()
        with patch("src.mcp_client.MCP_AVAILABLE", True):
            result = manager.connect_all([])
        assert result == {}

    def test_connect_all_mcp_unavailable_returns_empty(self):
        manager = MCPManager()
        with patch("src.mcp_client.MCP_AVAILABLE", False):
            result = manager.connect_all([MCPServerConfig(name="s", command="cmd")])
        assert result == {}

    def test_connect_all_successful_connection_returns_tools(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="myserver", command="mcp-server")
        mock_tool = _make_mock_tool("search", "Search the web")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg])

        manager.close_all()
        assert "search" in tools

    def test_connect_all_failed_server_skipped_continues(self):
        manager = MCPManager()
        cfg_bad = MCPServerConfig(name="bad", command="bad-cmd")
        cfg_good = MCPServerConfig(name="good", command="good-cmd")
        mock_tool = _make_mock_tool("good_tool", "A good tool")

        call_count = [0]

        async def fake_connect(self_conn: MCPConnection) -> None:
            call_count[0] += 1
            if self_conn._config.name == "bad":
                raise ConnectionError("refused")
            self_conn._tools = [mock_tool]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg_bad, cfg_good])

        manager.close_all()
        assert "good_tool" in tools
        assert call_count[0] == 2

    def test_connect_all_tool_name_collision_prefixed(self):
        manager = MCPManager()
        cfg1 = MCPServerConfig(name="server1", command="cmd1")
        cfg2 = MCPServerConfig(name="server2", command="cmd2")
        tool_a = _make_mock_tool("duplicate_tool", "Tool from server1")
        tool_b = _make_mock_tool("duplicate_tool", "Tool from server2")

        async def fake_connect(self_conn: MCPConnection) -> None:
            if self_conn._config.name == "server1":
                self_conn._tools = [tool_a]
            else:
                self_conn._tools = [tool_b]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg1, cfg2])

        manager.close_all()
        assert "duplicate_tool" in tools
        assert "server2_duplicate_tool" in tools

    def test_call_tool_unknown_server_returns_error(self):
        manager = MCPManager()
        result = manager.call_tool("nonexistent", "some_tool", {})
        assert result.startswith("Error:")
        assert "nonexistent" in result

    def test_call_tool_connected_server_returns_result(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="myserver", command="cmd")
        mock_tool = _make_mock_tool("mytool")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        async def fake_call_tool(_self_conn: MCPConnection, _name: str, _args: dict) -> str:
            return "tool result"

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "call_tool", fake_call_tool):
                    manager.connect_all([cfg])
                    result = manager.call_tool("myserver", "mytool", {"query": "test"})

        manager.close_all()
        assert result == "tool result"

    def test_call_tool_passes_name_unchanged(self):
        # The closure in _create_mcp_tool_wrapper always captures the original
        # MCP tool name, so call_tool() must forward whatever name it receives
        # without stripping any server prefix.
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        mock_tool = _make_mock_tool("original")

        received_name: list[str] = []

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        async def fake_call_tool(_self_conn: MCPConnection, name: str, _args: dict) -> str:
            received_name.append(name)
            return "ok"

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "call_tool", fake_call_tool):
                    manager.connect_all([cfg])
                    manager.call_tool("srv", "original", {})

        manager.close_all()
        assert received_name == ["original"]

    def test_call_tool_timeout_returns_error(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="slow", command="cmd", timeout=1)
        mock_tool = _make_mock_tool("slow_tool")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                manager.connect_all([cfg])

        with patch.object(manager, "_run", side_effect=TimeoutError()):
            result = manager.call_tool("slow", "slow_tool", {})

        manager.close_all()
        assert "timed out" in result.lower() or result.startswith("Error:")

    def test_close_all_clears_connections(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="s", command="cmd")
        mock_tool = _make_mock_tool()

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        async def fake_close(_self_conn: MCPConnection) -> None:
            pass

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "close", fake_close):
                    manager.connect_all([cfg])
                    assert len(manager._connections) == 1
                    manager.close_all()

        assert manager._connections == {}
        assert manager._tool_server_map == {}

    def test_get_server_info_returns_correct_structure(self):
        manager = MCPManager()
        cfg_stdio = MCPServerConfig(name="stdio_srv", command="mcp-bin", args=["--port", "9000"])
        cfg_sse = MCPServerConfig(name="sse_srv", url="http://localhost:8080/sse")
        mock_tool = _make_mock_tool("t1")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        async def fake_close(_self_conn: MCPConnection) -> None:
            pass

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "close", fake_close):
                    manager.connect_all([cfg_stdio, cfg_sse])

        info = manager.get_server_info()
        manager.close_all()

        assert len(info) == 2
        names = {e["name"] for e in info}
        assert "stdio_srv" in names
        assert "sse_srv" in names

        for entry in info:
            assert "connected" in entry
            assert "tool_count" in entry
            assert "tools" in entry
            assert "transport" in entry
            assert "endpoint" in entry

        stdio_entry = next(e for e in info if e["name"] == "stdio_srv")
        sse_entry = next(e for e in info if e["name"] == "sse_srv")
        assert stdio_entry["transport"] == "stdio"
        assert sse_entry["transport"] == "sse"

    def test_get_server_info_disconnected_server(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="offline", command="cmd")
        manager._configs["offline"] = cfg

        info = manager.get_server_info()
        assert len(info) == 1
        assert info[0]["connected"] is False
        assert info[0]["tool_count"] == 0
        assert info[0]["tools"] == []

    def test_get_server_info_endpoint_for_stdio(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="s", command="my-binary")
        manager._configs["s"] = cfg
        info = manager.get_server_info()
        assert info[0]["endpoint"] == "my-binary"

    def test_get_server_info_endpoint_for_sse(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="s", url="http://example.com/sse")
        manager._configs["s"] = cfg
        info = manager.get_server_info()
        assert info[0]["endpoint"] == "http://example.com/sse"

    def test_ensure_loop_starts_background_thread(self):
        manager = MCPManager()
        assert manager._loop is None
        assert manager._thread is None
        manager._ensure_loop()
        assert manager._loop is not None
        assert manager._thread is not None
        assert manager._thread.is_alive()
        manager._loop.call_soon_threadsafe(manager._loop.stop)
        manager._thread.join(timeout=2)

    def test_ensure_loop_idempotent(self):
        manager = MCPManager()
        manager._ensure_loop()
        loop1 = manager._loop
        thread1 = manager._thread
        manager._ensure_loop()
        assert manager._loop is loop1
        assert manager._thread is thread1
        manager._loop.call_soon_threadsafe(manager._loop.stop)
        manager._thread.join(timeout=2)


# ── _create_mcp_tool_wrapper ───────────────────────────────────────────────────


class TestCreateMcpToolWrapper:
    def _build_tool_wrapper(
        self,
        tool_name: str = "search",
        description: str = "Search the web",
        schema: dict | None = None,
        server_name: str = "myserver",
        requires_confirmation: bool = True,
    ):
        from src.mcp_client import _create_mcp_tool_wrapper

        if schema is None:
            schema = {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Query"}},
                "required": ["query"],
            }
        mock_tool = _make_mock_tool(tool_name, description)
        mock_tool.inputSchema = schema
        cfg = MCPServerConfig(
            name=server_name,
            command="cmd",
            requires_confirmation=requires_confirmation,
        )
        manager = MagicMock(spec=MCPManager)
        manager.call_tool.return_value = "search results"
        return _create_mcp_tool_wrapper(mock_tool, server_name, cfg, manager), manager

    def test_creates_structured_tool_with_correct_name(self):
        tool, _ = self._build_tool_wrapper(tool_name="web_search")
        assert tool is not None
        assert tool.name == "web_search"

    def test_creates_structured_tool_with_correct_description(self):
        tool, _ = self._build_tool_wrapper(description="A web search tool")
        assert tool is not None
        assert tool.description == "A web search tool"

    def test_tool_metadata_source_is_mcp(self):
        tool, _ = self._build_tool_wrapper()
        assert tool is not None
        assert tool.metadata["source"] == "mcp"

    def test_tool_metadata_contains_server_name(self):
        tool, _ = self._build_tool_wrapper(server_name="myserver")
        assert tool is not None
        assert tool.metadata["server"] == "myserver"

    def test_tool_metadata_requires_confirmation_true(self):
        tool, _ = self._build_tool_wrapper(requires_confirmation=True)
        assert tool is not None
        assert tool.metadata["requires_confirmation"] is True

    def test_tool_metadata_requires_confirmation_false(self):
        tool, _ = self._build_tool_wrapper(requires_confirmation=False)
        assert tool is not None
        assert tool.metadata["requires_confirmation"] is False

    def test_calling_tool_invokes_manager_call_tool(self):
        tool, manager = self._build_tool_wrapper(tool_name="mytool", server_name="srv")
        assert tool is not None
        result = tool.invoke({"query": "hello"})
        manager.call_tool.assert_called_once_with("srv", "mytool", {"query": "hello"})
        assert result == "search results"

    def test_returns_none_on_schema_failure(self):
        from src.mcp_client import _create_mcp_tool_wrapper

        bad_tool = MagicMock()
        bad_tool.name = "broken"
        bad_tool.description = "broken tool"
        bad_tool.inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cfg = MCPServerConfig(name="s", command="cmd")
        manager = MagicMock(spec=MCPManager)

        with patch("src.mcp_client.json_schema_to_pydantic", side_effect=Exception("schema error")):
            result = _create_mcp_tool_wrapper(bad_tool, "s", cfg, manager)

        assert result is None

    def test_fallback_description_when_none(self):
        from src.mcp_client import _create_mcp_tool_wrapper

        mock_tool = MagicMock()
        mock_tool.name = "no_desc"
        mock_tool.description = None
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        cfg = MCPServerConfig(name="srv", command="cmd")
        manager = MagicMock(spec=MCPManager)

        tool = _create_mcp_tool_wrapper(mock_tool, "srv", cfg, manager)
        assert tool is not None
        assert "no_desc" in tool.description or "srv" in tool.description

    def test_tool_args_schema_has_expected_field(self):
        tool, _ = self._build_tool_wrapper()
        assert tool is not None
        assert "query" in tool.args_schema.model_fields


# ── Config integration ─────────────────────────────────────────────────────────


class TestConfigMcpServers:
    def test_config_has_mcp_servers_field(self):
        from src.config import Config

        cfg = Config()
        assert hasattr(cfg, "mcp_servers")
        assert cfg.mcp_servers == {}

    def test_apply_config_file_parses_mcp_servers(self, tmp_path: Path):
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(
            "mcp_servers:\n"
            "  github:\n"
            "    command: npx\n"
            "    args:\n"
            "      - -y\n"
            '      - "@modelcontextprotocol/server-github"\n'
            "  webfetch:\n"
            "    url: http://localhost:8080/sse\n"
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert "github" in config.mcp_servers
        assert "webfetch" in config.mcp_servers
        assert config.mcp_servers["github"]["command"] == "npx"
        assert config.mcp_servers["webfetch"]["url"] == "http://localhost:8080/sse"

    def test_apply_config_file_no_mcp_servers_leaves_empty(self, tmp_path: Path):
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text("provider: ollama\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.mcp_servers == {}

    def test_apply_config_file_mcp_servers_not_dict_ignored(self, tmp_path: Path):
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text("mcp_servers: not_a_dict\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.mcp_servers == {}


# ── Registry helpers ───────────────────────────────────────────────────────────


class TestRegistryHelpers:
    def _make_registry_with_mcp_tool(self, tool_name: str = "mcp_search", server: str = "srv"):
        from src.registry import ToolRegistry

        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.tool_metadata = {}

        mock_lc_tool = MagicMock()
        mock_lc_tool.name = tool_name
        registry.tools[tool_name] = mock_lc_tool
        registry.tool_metadata[tool_name] = {
            "source": "mcp",
            "server": server,
            "requires_confirmation": True,
        }
        return registry

    def _make_registry_with_regular_tool(self, tool_name: str = "read_file"):
        from src.registry import ToolRegistry

        registry = ToolRegistry.__new__(ToolRegistry)
        registry.tools = {}
        registry.tool_metadata = {}

        mock_lc_tool = MagicMock()
        mock_lc_tool.name = tool_name
        registry.tools[tool_name] = mock_lc_tool
        registry.tool_metadata[tool_name] = {
            "source": "builtin",
            "requires_confirmation": False,
        }
        return registry

    def test_is_mcp_tool_returns_true_for_mcp_tool(self):
        registry = self._make_registry_with_mcp_tool("mcp_search")
        assert registry.is_mcp_tool("mcp_search") is True

    def test_is_mcp_tool_returns_false_for_regular_tool(self):
        registry = self._make_registry_with_regular_tool("read_file")
        assert registry.is_mcp_tool("read_file") is False

    def test_is_mcp_tool_returns_false_for_unknown_tool(self):
        registry = self._make_registry_with_mcp_tool()
        assert registry.is_mcp_tool("nonexistent_tool") is False

    def test_get_tool_server_returns_server_name_for_mcp_tool(self):
        registry = self._make_registry_with_mcp_tool("mcp_search", server="github")
        assert registry.get_tool_server("mcp_search") == "github"

    def test_get_tool_server_returns_none_for_regular_tool(self):
        registry = self._make_registry_with_regular_tool("read_file")
        assert registry.get_tool_server("read_file") is None

    def test_get_tool_server_returns_none_for_unknown_tool(self):
        registry = self._make_registry_with_mcp_tool()
        assert registry.get_tool_server("nonexistent") is None
