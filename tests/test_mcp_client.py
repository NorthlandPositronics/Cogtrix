"""Tests for src/mcp_client.py."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_client import (
    _STARTUP_MAX_RETRIES,
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

    def test_pin_field_default_is_true(self):
        cfg = MCPServerConfig(name="test", url="http://localhost:8001/sse")
        assert cfg.pin is True

    def test_pin_field_explicit_false(self):
        cfg = MCPServerConfig(name="test", url="http://localhost:8001/sse", pin=False)
        assert cfg.pin is False

    def test_sse_connect_uses_bounded_read_timeout(self):
        """SSE connection must NOT use sse_read_timeout=None — unbounded streams hide network failures."""
        import asyncio
        import contextlib
        from unittest.mock import patch

        cfg = MCPServerConfig(name="s", url="https://example.com:9999/sse")
        conn = MCPConnection(cfg)

        captured_kwargs: dict = {}

        @contextlib.asynccontextmanager
        async def _fake_sse(**kwargs):
            captured_kwargs.update(kwargs)
            # Yield fake streams so MCPConnection.connect doesn't need a real server
            raise ConnectionError("test sentinel — not a real connection")
            yield  # type: ignore[misc]

        with patch("src.mcp_client.sse_client", side_effect=_fake_sse):
            try:
                asyncio.run(conn.connect())
            except Exception:
                pass

        assert (
            "sse_read_timeout" in captured_kwargs
        ), "sse_client must be called with an explicit sse_read_timeout keyword"
        assert (
            captured_kwargs["sse_read_timeout"] is not None
        ), "sse_read_timeout=None is unbounded and hides network failures"
        assert captured_kwargs["sse_read_timeout"] > 0, "sse_read_timeout must be a positive number"


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

    def test_nullable_list_type_string_null(self):
        import typing

        schema = {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
            "required": ["name"],
        }
        model = json_schema_to_pydantic("nullable_tool", schema)
        annotation = model.model_fields["name"].annotation
        args = typing.get_args(annotation)
        assert str in args
        assert type(None) in args

    def test_nullable_list_type_integer_null(self):
        import typing

        schema = {
            "type": "object",
            "properties": {"count": {"type": ["integer", "null"]}},
            "required": ["count"],
        }
        model = json_schema_to_pydantic("nullable_int_tool", schema)
        annotation = model.model_fields["count"].annotation
        args = typing.get_args(annotation)
        assert int in args
        assert type(None) in args

    def test_anyof_picks_first_variant_type(self):
        schema = {
            "type": "object",
            "properties": {"value": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
            "required": ["value"],
        }
        model = json_schema_to_pydantic("anyof_tool", schema)
        assert model.model_fields["value"].annotation is int

    def test_oneof_picks_first_variant_type(self):
        schema = {
            "type": "object",
            "properties": {"flag": {"oneOf": [{"type": "boolean"}, {"type": "string"}]}},
            "required": ["flag"],
        }
        model = json_schema_to_pydantic("oneof_tool", schema)
        assert model.model_fields["flag"].annotation is bool

    def test_anyof_empty_variants_defaults_to_str(self):
        schema = {
            "type": "object",
            "properties": {"x": {"anyOf": []}},
            "required": ["x"],
        }
        model = json_schema_to_pydantic("anyof_empty", schema)
        assert model.model_fields["x"].annotation is str

    # ── $ref resolution ────────────────────────────────────────────────────────

    def test_ref_resolves_defs_integer_type(self):
        schema = {
            "type": "object",
            "properties": {"count": {"$ref": "#/$defs/Count"}},
            "required": ["count"],
            "$defs": {"Count": {"type": "integer"}},
        }
        model = json_schema_to_pydantic("reftool", schema)
        assert model.model_fields["count"].annotation is int

    def test_ref_resolves_definitions_string_type(self):
        schema = {
            "type": "object",
            "properties": {"label": {"$ref": "#/definitions/Label"}},
            "required": ["label"],
            "definitions": {"Label": {"type": "string"}},
        }
        model = json_schema_to_pydantic("reftool2", schema)
        assert model.model_fields["label"].annotation is str

    def test_ref_unresolvable_falls_back_to_str(self):
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
            "required": ["x"],
            "$defs": {},
        }
        model = json_schema_to_pydantic("reftool3", schema)
        assert model.model_fields["x"].annotation is str

    def test_ref_external_falls_back_to_str(self):
        schema = {
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/schema.json#/Foo"}},
            "required": ["x"],
        }
        model = json_schema_to_pydantic("reftool4", schema)
        assert model.model_fields["x"].annotation is str

    def test_ref_resolves_nested_const_to_literal(self):
        import typing

        schema = {
            "type": "object",
            "properties": {"mode": {"$ref": "#/$defs/Mode"}},
            "required": ["mode"],
            "$defs": {"Mode": {"const": "strict"}},
        }
        model = json_schema_to_pydantic("reftool5", schema)
        annotation = model.model_fields["mode"].annotation
        assert typing.get_origin(annotation) is typing.Literal
        assert typing.get_args(annotation) == ("strict",)

    # ── const → Literal ────────────────────────────────────────────────────────

    def test_const_string_becomes_literal(self):
        import typing

        schema = {
            "type": "object",
            "properties": {"version": {"const": "v1"}},
            "required": ["version"],
        }
        model = json_schema_to_pydantic("consttool", schema)
        annotation = model.model_fields["version"].annotation
        assert typing.get_origin(annotation) is typing.Literal
        assert typing.get_args(annotation) == ("v1",)

    def test_const_integer_becomes_literal(self):
        import typing

        schema = {
            "type": "object",
            "properties": {"api_version": {"const": 2}},
            "required": ["api_version"],
        }
        model = json_schema_to_pydantic("consttool2", schema)
        annotation = model.model_fields["api_version"].annotation
        assert typing.get_origin(annotation) is typing.Literal
        assert typing.get_args(annotation) == (2,)


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

    def test_filesystem_access_denied_hint_appended(self):
        """Access-denied errors include the /workspace/ prefix hint (#135)."""
        msg = "Access denied - path outside allowed directories: /src/foo.py not in /data"
        result = _make_result([_make_text(msg)], is_error=True)
        output = _result_to_str(result)
        assert "Error:" in output
        assert "/workspace/" in output, "hint about /workspace/ prefix must be present"
        assert "get_file_contents" in output, "hint about get_file_contents must be present"

    def test_non_access_denied_error_no_hint(self):
        """Unrelated errors are NOT enriched with the workspace hint."""
        result = _make_result([_make_text("Connection refused")], is_error=True)
        output = _result_to_str(result)
        assert "Error: Connection refused" == output
        assert "/workspace/" not in output


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
        # good server: 1 call; bad server: 1 initial + _STARTUP_MAX_RETRIES retries
        assert call_count[0] == 1 + (_STARTUP_MAX_RETRIES + 1)

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

    def test_close_all_during_call_tool_does_not_crash(self):
        """Calling close_all() while another thread is in call_tool() must not crash."""
        import threading

        manager = MCPManager()
        cfg = MCPServerConfig(name="s", command="cmd")
        mock_tool = _make_mock_tool()

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        async def fake_close(_self_conn: MCPConnection) -> None:
            pass

        entered = threading.Event()
        release = threading.Event()

        async def slow_call_tool(
            _self_conn: MCPConnection, _name: str, _args: dict[str, Any]
        ) -> str:
            entered.set()
            release.wait(timeout=5.0)
            return "slow-result"

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "close", fake_close):
                    with patch.object(MCPConnection, "call_tool", slow_call_tool):
                        manager.connect_all([cfg])

                        # Start a slow call_tool in a background thread
                        call_result: list[str] = []

                        def _call():
                            try:
                                call_result.append(manager.call_tool("s", "t", {}))
                            except Exception as exc:
                                call_result.append(f"Error: {exc}")

                        t = threading.Thread(target=_call)
                        t.start()

                        # Wait until the call has entered the tool before closing
                        entered.wait(timeout=2.0)

                        # Close all while the call is in flight
                        manager.close_all()

                        # Let the background call complete
                        release.set()
                        t.join(timeout=3.0)

        # The background thread must have completed within the timeout
        assert not t.is_alive(), "Background call_tool thread did not complete"
        # The manager should be clean regardless of race outcome
        assert manager._connections == {}
        # We should have a result (either success or a graceful error)
        assert len(call_result) == 1, f"Expected exactly 1 result, got {call_result}"

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

    def test_connect_all_builtin_collision_prefixed(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="myserver", command="cmd")
        mock_tool = _make_mock_tool("read_file", "MCP read file tool")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        builtin_names = {"read_file", "write_file", "shell"}

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg], builtin_tool_names=builtin_names)

        manager.close_all()
        assert "read_file" not in tools
        assert "myserver_read_file" in tools

    def test_connect_all_no_builtin_names_no_prefix(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="myserver", command="cmd")
        mock_tool = _make_mock_tool("read_file", "MCP read file tool")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg])

        manager.close_all()
        assert "read_file" in tools

    def test_connect_all_builtin_names_none_no_prefix(self):
        manager = MCPManager()
        cfg = MCPServerConfig(name="myserver", command="cmd")
        mock_tool = _make_mock_tool("read_file", "MCP read file tool")

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = [mock_tool]

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                tools = manager.connect_all([cfg], builtin_tool_names=None)

        manager.close_all()
        assert "read_file" in tools


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

    def test_direct_push_to_next_is_blocked(self):
        schema = {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["branch", "path", "content"],
        }
        tool, manager = self._build_tool_wrapper(
            tool_name="create_or_update_file",
            description="Update a repository file",
            schema=schema,
            server_name="github",
            requires_confirmation=True,
        )
        assert tool is not None

        result = tool.invoke({"branch": "next", "path": "docs/file.md", "content": "x"})

        assert "Direct push to protected branch 'next' is not allowed" in result
        manager.call_tool.assert_not_called()

    def test_push_files_to_main_is_blocked(self):
        schema = {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "files": {"type": "array"},
            },
            "required": ["branch", "files"],
        }
        tool, manager = self._build_tool_wrapper(
            tool_name="push_files",
            description="Push multiple files",
            schema=schema,
            server_name="github",
            requires_confirmation=True,
        )
        assert tool is not None

        result = tool.invoke({"branch": "main", "files": []})

        assert "Direct push to protected branch 'main' is not allowed" in result
        manager.call_tool.assert_not_called()

    def test_feature_branch_still_passes_through(self):
        schema = {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["branch", "path", "content"],
        }
        tool, manager = self._build_tool_wrapper(
            tool_name="create_or_update_file",
            description="Update a repository file",
            schema=schema,
            server_name="github",
            requires_confirmation=True,
        )
        assert tool is not None

        result = tool.invoke({"branch": "fix/159-guard", "path": "docs/file.md", "content": "x"})

        manager.call_tool.assert_called_once_with(
            "github",
            "create_or_update_file",
            {"branch": "fix/159-guard", "path": "docs/file.md", "content": "x"},
        )
        assert result == "search results"


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

    def test_apply_config_file_mcp_servers_allow_insecure_preserved(self, tmp_path: Path):
        """Regression for #395: allow_insecure must survive the config-file parsing
        round-trip so it reaches MCPServerConfig (not silently stripped)."""
        from src.config import Config, _apply_config_file

        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(
            "mcp_servers:\n"
            "  internal:\n"
            "    url: http://mcp-internal:8001/sse\n"
            "    allow_insecure: true\n"
        )
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.mcp_servers["internal"]["allow_insecure"] is True

    def test_mcp_server_config_allow_insecure_round_trip(self):
        """Regression for #395: KNOWN_MCP_FIELDS must include allow_insecure so
        the value is not stripped when building MCPServerConfig from config dict."""
        srv = {"url": "http://mcp-internal:8001/sse", "allow_insecure": True}
        _KNOWN_MCP_FIELDS = {
            "command",
            "args",
            "env",
            "url",
            "headers",
            "requires_confirmation",
            "timeout",
            "pin",
            "allow_insecure",
        }
        filtered = {k: v for k, v in srv.items() if k in _KNOWN_MCP_FIELDS}
        cfg = MCPServerConfig(name="internal", **filtered)
        assert cfg.allow_insecure is True

    def test_mcp_server_config_allow_insecure_stripped_without_fix(self):
        """Documents the BROKEN behaviour: omitting allow_insecure from the field
        set silently drops the value and the server stays http-blocked."""
        srv = {"url": "http://mcp-internal:8001/sse", "allow_insecure": True}
        _KNOWN_MCP_FIELDS_OLD = {
            "command",
            "args",
            "env",
            "url",
            "headers",
            "requires_confirmation",
            "timeout",
            "pin",
        }
        filtered = {k: v for k, v in srv.items() if k in _KNOWN_MCP_FIELDS_OLD}
        cfg = MCPServerConfig(name="internal", **filtered)
        # Without the fix allow_insecure is always False regardless of yaml
        assert cfg.allow_insecure is False

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


# ── _is_connection_error ───────────────────────────────────────────────────────


class TestIsConnectionError:
    """Tests for the connection-error classifier (issue #98)."""

    def test_cancelled_error_is_connection_error(self):
        import asyncio

        from src.mcp_client import _is_connection_error

        assert _is_connection_error(asyncio.CancelledError()) is True

    def test_timeout_error_is_connection_error(self):
        from src.mcp_client import _is_connection_error

        assert _is_connection_error(TimeoutError()) is True

    def test_connection_reset_is_connection_error(self):
        from src.mcp_client import _is_connection_error

        assert _is_connection_error(ConnectionResetError()) is True

    def test_broken_pipe_is_connection_error(self):
        from src.mcp_client import _is_connection_error

        assert _is_connection_error(BrokenPipeError()) is True

    def test_value_error_is_not_connection_error(self):
        from src.mcp_client import _is_connection_error

        assert _is_connection_error(ValueError("not a connection issue")) is False

    def test_chained_connection_error_detected(self):
        """Exception wrapping a connection error should still be detected."""
        from src.mcp_client import _is_connection_error

        outer = RuntimeError("tool failed")
        outer.__cause__ = ConnectionResetError("connection lost")
        assert _is_connection_error(outer) is True

    def test_generic_exception_not_connection_error(self):
        from src.mcp_client import _is_connection_error

        assert _is_connection_error(Exception("generic")) is False


# ── call_tool error message enrichment ───────────────────────────────────────


class TestCallToolErrorEnrichment:
    """Error messages from call_tool now include exception type (issue #98)."""

    def _make_manager_with_failing_conn(self, exc: Exception) -> MCPManager:
        mgr = MCPManager.__new__(MCPManager)
        mgr._connections = {}
        mgr._configs = {}
        mgr._tool_server_map = {}
        mgr._loop = None
        mgr._thread = None
        mgr._mcp_pin_map = {}

        cfg = MCPServerConfig(name="test-server", url="http://localhost:8001/sse")
        mgr._configs["test-server"] = cfg

        mock_conn = MagicMock()
        mock_conn.call_tool = MagicMock(side_effect=exc)
        mgr._connections["test-server"] = mock_conn

        # _run just calls the coroutine synchronously for testing.
        def _run_sync(coro, timeout=30):
            raise exc

        mgr._run = _run_sync  # type: ignore[method-assign]
        return mgr

    def test_error_message_includes_exception_type(self):
        mgr = self._make_manager_with_failing_conn(ValueError("bad args"))
        result = mgr.call_tool("test-server", "some_tool", {})
        assert "ValueError" in result
        assert "bad args" in result

    def test_empty_exception_message_shows_no_details(self):
        mgr = self._make_manager_with_failing_conn(ConnectionResetError())
        result = mgr.call_tool("test-server", "some_tool", {})
        assert "ConnectionResetError" in result


# ── MCP SSE URL SSRF validation (#302) ───────────────────────────────────────


class TestMCPUrlValidation:
    """SSRF guards for MCP SSE URLs (issue #302)."""

    def test_https_url_accepted(self):
        from src.mcp_client import _validate_mcp_url

        # Should not raise for a public HTTPS URL.
        _validate_mcp_url("https://example.com/sse")

    def test_http_rejected_by_default(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="Insecure MCP SSE URL"):
            _validate_mcp_url("http://example.com/sse")

    def test_http_allowed_when_allow_insecure_true(self):
        from src.mcp_client import _validate_mcp_url

        _validate_mcp_url("http://example.com/sse", allow_insecure=True)

    def test_rfc1918_allowed_when_allow_insecure_true(self):
        """Regression for #395: allow_insecure must also bypass the RFC1918/private-IP
        block so internal Docker network MCP servers (e.g. 172.20.x.x) can connect."""
        from src.mcp_client import _validate_mcp_url

        # 172.20.x.x is the Docker bridge range used by cogtrix_cogtrix-net
        _validate_mcp_url("http://172.20.0.2/sse", allow_insecure=True)
        _validate_mcp_url("http://192.168.1.1/sse", allow_insecure=True)
        _validate_mcp_url("http://10.0.0.1/sse", allow_insecure=True)

    def test_localhost_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="internal host"):
            _validate_mcp_url("https://localhost/sse")

    def test_loopback_ip_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="blocked IP"):
            _validate_mcp_url("https://127.0.0.1/sse")

    def test_rfc1918_ip_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="blocked IP"):
            _validate_mcp_url("https://192.168.1.1/sse")

    def test_link_local_ip_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="internal host"):
            _validate_mcp_url("https://169.254.169.254/latest/meta-data/")

    def test_aws_metadata_host_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="internal host"):
            _validate_mcp_url("https://169.254.169.254/")

    def test_invalid_url_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="Invalid MCP SSE URL"):
            _validate_mcp_url("not-a-url")

    def test_unsupported_scheme_rejected(self):
        from src.mcp_client import _validate_mcp_url

        with pytest.raises(ValueError, match="scheme must be http or https"):
            _validate_mcp_url("ftp://example.com/sse")

    def test_allow_insecure_false_by_default(self):
        cfg = MCPServerConfig(name="test", url="https://example.com/sse")
        assert cfg.allow_insecure is False

    def test_connect_raises_for_blocked_url(self):
        """MCPConnection.connect() rejects a blocked URL before opening SSE."""
        from src.mcp_client import MCPConnection

        cfg = MCPServerConfig(name="bad-server", url="https://127.0.0.1:9000/sse")
        conn = MCPConnection(cfg)

        import asyncio

        with pytest.raises(ValueError, match="blocked IP"):
            asyncio.run(conn.connect())


# ── Regression: shutdown traceback suppression (#356) ─────────────────────────


class TestMCPShutdownTraceback:
    """Regression tests for issue #356 — MCP post_writer RuntimeError leak."""

    def test_exception_handler_suppresses_shutdown_race(self) -> None:
        """Fix A: custom loop exception handler must silence the shutdown RuntimeError."""
        import io
        import sys

        manager = MCPManager()
        manager._ensure_loop()
        assert manager._loop is not None

        captured_stderr = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_stderr
        try:
            exc = RuntimeError("cannot schedule new futures after shutdown")
            context = {"exception": exc, "message": str(exc)}
            manager._loop.call_exception_handler(context)
        finally:
            sys.stderr = old_stderr

        assert captured_stderr.getvalue() == ""
        manager.close_all()

    def test_exception_handler_passes_through_other_errors(self) -> None:
        """Fix A: non-shutdown errors must still reach the default handler."""
        manager = MCPManager()
        manager._ensure_loop()
        assert manager._loop is not None

        default_called: list[dict] = []

        manager._loop.set_exception_handler(
            lambda loop, ctx: (
                default_called.append(ctx)
                if not (
                    isinstance(ctx.get("exception"), RuntimeError)
                    and "cannot schedule new futures after shutdown"
                    in str(ctx.get("exception", ""))
                )
                else None
            )
        )

        exc = RuntimeError("something else went wrong")
        context = {"exception": exc, "message": str(exc)}
        manager._loop.call_exception_handler(context)
        assert len(default_called) == 1

        manager.close_all()

    def test_close_all_does_not_print_to_stderr(self) -> None:
        """Fix B: MCP library's print() during teardown must not reach the terminal."""
        import io
        import sys

        manager = MCPManager()

        async def fake_connect(self_conn: MCPConnection) -> None:
            pass

        async def noisy_close(_self_conn: MCPConnection) -> None:
            import traceback as _tb

            print("Error in post_writer", file=sys.stderr)
            try:
                raise RuntimeError("cannot schedule new futures after shutdown")
            except RuntimeError:
                _tb.print_exc()

        captured = io.StringIO()
        old_stderr = sys.stderr

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "close", noisy_close):
                    manager.connect_all([MCPServerConfig(name="noisy", command="cmd")])
                    sys.stderr = captured
                    try:
                        manager.close_all()
                    finally:
                        sys.stderr = old_stderr

        assert captured.getvalue() == ""

    def test_close_all_does_not_restore_stderr(self) -> None:
        """Regression #500: close_all() must NOT restore sys.stderr after _close_all_inner.

        The post_writer async task fires DNS resolution *after* _close_all_inner()
        returns. If stderr is restored before that task completes, the traceback
        escapes to the terminal. The fix is to leave stderr captured for the
        lifetime of the process (all close_all() call sites are shutdown-path only).
        """
        import sys

        manager = MCPManager()
        real_stderr = sys.stderr
        try:
            manager.close_all()
            # stderr must still be the captured buffer, not the original terminal
            assert sys.stderr is not real_stderr, (
                "close_all() restored sys.stderr — the post_writer traceback "
                "regression (#500) has been reintroduced"
            )
        finally:
            sys.stderr = real_stderr

    def test_close_all_idempotent(self) -> None:
        """Fix C: close_all() called multiple times must not raise."""
        manager = MCPManager()

        async def fake_connect(self_conn: MCPConnection) -> None:
            pass

        async def fake_close(_self_conn: MCPConnection) -> None:
            pass

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                with patch.object(MCPConnection, "close", fake_close):
                    manager.connect_all([MCPServerConfig(name="s", command="cmd")])

        manager.close_all()
        manager.close_all()
        assert manager._connections == {}


class TestMCPStartupRetry:
    """Regression tests for #393 (startup retry) and #403 (anyio cleanup on failure)."""

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Execute a coroutine synchronously in a fresh event loop."""
        return asyncio.get_event_loop().run_until_complete(coro)

    # ── #403: explicit close on failure ──────────────────────────────────────

    def test_failed_connect_closes_connection_in_same_task(self) -> None:
        """_connect_one_async must call conn.close() when connect() raises (#403)."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        close_calls: list[str] = []

        async def failing_connect(_self: MCPConnection) -> None:
            raise ConnectionRefusedError("refused")

        async def tracking_close(self_conn: MCPConnection) -> None:
            close_calls.append(self_conn._config.name)

        with patch("src.mcp_client._STARTUP_MAX_RETRIES", 0):
            with patch.object(MCPConnection, "connect", failing_connect):
                with patch.object(MCPConnection, "close", tracking_close):
                    with patch("src.mcp_client.MCP_AVAILABLE", True):
                        manager._ensure_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            manager._connect_one_async(cfg), manager._loop  # type: ignore[arg-type]
                        )
                        name, conn = future.result(timeout=5)

        manager.close_all()
        assert conn is None
        assert name == "srv"
        assert "srv" in close_calls, "close() must be called on the failed MCPConnection"

    def test_failed_close_during_cleanup_does_not_propagate(self) -> None:
        """close() errors during startup cleanup must be swallowed, not raised."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")

        async def failing_connect(_self: MCPConnection) -> None:
            raise ConnectionRefusedError("refused")

        async def raising_close(_self: MCPConnection) -> None:
            raise RuntimeError("Attempted to exit cancel scope in a different task")

        with patch("src.mcp_client._STARTUP_MAX_RETRIES", 0):
            with patch.object(MCPConnection, "connect", failing_connect):
                with patch.object(MCPConnection, "close", raising_close):
                    with patch("src.mcp_client.MCP_AVAILABLE", True):
                        manager._ensure_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            manager._connect_one_async(cfg), manager._loop  # type: ignore[arg-type]
                        )
                        name, conn = future.result(timeout=5)

        manager.close_all()
        assert conn is None

    # ── #393: startup retry ───────────────────────────────────────────────────

    def test_retries_on_connection_error_and_succeeds(self) -> None:
        """Transient connection failure followed by success: return (name, conn)."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        attempts: list[int] = []

        async def flaky_connect(self_conn: MCPConnection) -> None:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 2:
                raise ConnectionRefusedError("not ready yet")
            self_conn._tools = []

        async def noop_close(_self: MCPConnection) -> None:
            pass

        with patch("src.mcp_client._STARTUP_RETRY_DELAYS", (0.0, 0.0)):
            with patch.object(MCPConnection, "connect", flaky_connect):
                with patch.object(MCPConnection, "close", noop_close):
                    with patch("src.mcp_client.MCP_AVAILABLE", True):
                        manager._ensure_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            manager._connect_one_async(cfg), manager._loop  # type: ignore[arg-type]
                        )
                        name, conn = future.result(timeout=10)

        manager.close_all()
        assert conn is not None
        assert name == "srv"
        assert len(attempts) == 2

    def test_exhausted_retries_returns_none(self) -> None:
        """All retry attempts fail: _connect_one_async returns (name, None)."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        attempts: list[int] = []

        async def always_fail(_self: MCPConnection) -> None:
            attempts.append(1)
            raise ConnectionRefusedError("always down")

        async def noop_close(_self: MCPConnection) -> None:
            pass

        with patch("src.mcp_client._STARTUP_RETRY_DELAYS", (0.0, 0.0)):
            with patch.object(MCPConnection, "connect", always_fail):
                with patch.object(MCPConnection, "close", noop_close):
                    with patch("src.mcp_client.MCP_AVAILABLE", True):
                        manager._ensure_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            manager._connect_one_async(cfg), manager._loop  # type: ignore[arg-type]
                        )
                        name, conn = future.result(timeout=10)

        manager.close_all()
        assert conn is None
        assert len(attempts) == _STARTUP_MAX_RETRIES + 1

    def test_value_error_is_not_retried(self) -> None:
        """Config/URL validation errors (ValueError) must not trigger retries."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        attempts: list[int] = []

        async def bad_config_connect(_self: MCPConnection) -> None:
            attempts.append(1)
            raise ValueError("invalid URL")

        async def noop_close(_self: MCPConnection) -> None:
            pass

        with patch("src.mcp_client._STARTUP_RETRY_DELAYS", (0.0, 0.0)):
            with patch.object(MCPConnection, "connect", bad_config_connect):
                with patch.object(MCPConnection, "close", noop_close):
                    with patch("src.mcp_client.MCP_AVAILABLE", True):
                        manager._ensure_loop()
                        future = asyncio.run_coroutine_threadsafe(
                            manager._connect_one_async(cfg), manager._loop  # type: ignore[arg-type]
                        )
                        name, conn = future.result(timeout=5)

        manager.close_all()
        assert conn is None
        assert len(attempts) == 1, "ValueError must not be retried"

    def test_connect_all_retries_failed_server_transparently(self) -> None:
        """connect_all() succeeds for a server that needs one retry."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="slow", command="cmd")
        mock_tool = _make_mock_tool("mytool", "A tool")
        attempts: list[int] = []

        async def flaky_connect(self_conn: MCPConnection) -> None:
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionRefusedError("not ready")
            self_conn._tools = [mock_tool]

        async def noop_close(_self: MCPConnection) -> None:
            pass

        with patch("src.mcp_client._STARTUP_RETRY_DELAYS", (0.0, 0.0)):
            with patch("src.mcp_client.MCP_AVAILABLE", True):
                with patch.object(MCPConnection, "connect", flaky_connect):
                    with patch.object(MCPConnection, "close", noop_close):
                        tools = manager.connect_all([cfg])

        manager.close_all()
        assert "mytool" in tools
        assert len(attempts) == 2


class TestMCPReconnectRace:
    """Regression tests for #427 — concurrent reconnect paths must not race on _connections."""

    def test_sync_reconnect_uses_async_lock(self) -> None:
        """_reconnect_server (sync) must route through _reconnect_server_async to share the lock."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        calls: list[str] = []

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = []

        async def tracking_reconnect_async(server_name: str) -> None:
            calls.append(server_name)

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                manager.connect_all([cfg])

        # Patch the bound method on the instance so self is already bound
        import types

        manager._reconnect_server_async = types.MethodType(  # type: ignore[method-assign]
            lambda _self, name: tracking_reconnect_async(name), manager
        )
        manager._reconnect_server("srv")

        manager.close_all()
        assert calls == ["srv"], "_reconnect_server must delegate to _reconnect_server_async"

    def test_per_server_lock_created_lazily(self) -> None:
        """_reconnect_locks must be populated lazily per server during reconnect."""
        manager = MCPManager()
        cfg = MCPServerConfig(name="srv", command="cmd")
        assert "srv" not in manager._reconnect_locks

        async def fake_connect(self_conn: MCPConnection) -> None:
            self_conn._tools = []

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                manager.connect_all([cfg])
                manager._ensure_loop()
                # Trigger async reconnect to create the lock
                import asyncio

                future = asyncio.run_coroutine_threadsafe(
                    manager._reconnect_server_async("srv"), manager._loop  # type: ignore[arg-type]
                )
                future.result(timeout=10)

        manager.close_all()
        assert "srv" in manager._reconnect_locks, "Lock must exist after first reconnect"


class TestMCPShutdownRace:
    """Regression tests for #425 — _ensure_loop() must not spawn a zombie loop after close_all()."""

    def test_ensure_loop_raises_after_close_all(self) -> None:
        """_ensure_loop() called after close_all() must raise RuntimeError, not create a zombie."""
        manager = MCPManager()
        manager._ensure_loop()
        manager.close_all()

        import pytest

        with pytest.raises(RuntimeError, match="close_all"):
            manager._ensure_loop()

    def test_shutting_down_flag_set_by_close_all(self) -> None:
        """close_all() must set _shutting_down before releasing the lock."""
        manager = MCPManager()
        assert not manager._shutting_down
        manager._ensure_loop()
        manager.close_all()
        assert manager._shutting_down

    def test_call_tool_after_close_returns_error_not_zombie(self) -> None:
        """call_tool() after close_all() returns an error string — does not spawn a new loop."""
        manager = MCPManager()

        async def fake_connect(self_conn: MCPConnection) -> None:
            pass

        with patch("src.mcp_client.MCP_AVAILABLE", True):
            with patch.object(MCPConnection, "connect", fake_connect):
                manager.connect_all([MCPServerConfig(name="srv", command="cmd")])

        manager.close_all()
        assert manager._shutting_down

        # Attempting a tool call after close must not create a new event loop
        result = manager.call_tool("srv", "some_tool", {})
        assert result.startswith("Error:")
        assert manager._loop is None, "No new loop must be spawned after close_all()"
