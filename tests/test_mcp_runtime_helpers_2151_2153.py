"""Unit tests for the MCP runtime-wiring helpers (#2151 / #2153).

Covers the pure registry helpers in ``cogtrix_core/api/mcp_runtime.py``, the
session-reconciliation eviction logic, and ``MCPManager.disconnect``'s
no-loop early-out. All fast — no FAISS, no real MCP event loop, no network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from cogtrix_core.api.mcp_runtime import (
    mcp_builtin_tool_names,
    register_mcp_tools,
    unregister_mcp_server_tools,
)


class _FakeRegistry:
    """Minimal stand-in for ToolRegistry — just the two dicts the helpers touch."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}
        self.tool_metadata: dict[str, dict] = {}


def _mcp_tool(server: str, requires_confirmation: bool = True) -> MagicMock:
    t = MagicMock()
    t.metadata = {
        "source": "mcp",
        "server": server,
        "requires_confirmation": requires_confirmation,
    }
    return t


class TestBuiltinToolNames:
    def test_excludes_mcp_tools(self) -> None:
        reg = _FakeRegistry()
        reg.tools = {"shell": object(), "weather": object(), "do_thing": object()}
        reg.tool_metadata = {
            "shell": {"source": "builtin"},
            "weather": {},  # missing source → treated as non-mcp builtin
            "do_thing": {"source": "mcp", "server": "srv"},
        }
        assert mcp_builtin_tool_names(reg) == {"shell", "weather"}

    def test_none_registry_returns_empty(self) -> None:
        assert mcp_builtin_tool_names(None) == set()


class TestRegisterMcpTools:
    def test_registers_with_metadata_and_pin(self) -> None:
        reg = _FakeRegistry()
        pinned: set[str] = set()
        tools = {"do_thing": _mcp_tool("srv")}
        register_mcp_tools(reg, pinned, tools, {"srv": True})

        assert "do_thing" in reg.tools
        meta = reg.tool_metadata["do_thing"]
        assert meta["source"] == "mcp"
        assert meta["server"] == "srv"
        assert meta["pin"] is True
        assert "do_thing" in pinned

    def test_unpinned_server_not_added_to_pinned(self) -> None:
        reg = _FakeRegistry()
        pinned: set[str] = set()
        register_mcp_tools(reg, pinned, {"t": _mcp_tool("big")}, {"big": False})
        assert reg.tool_metadata["t"]["pin"] is False
        assert "t" not in pinned

    def test_reregister_unpinned_discards_stale_pin(self) -> None:
        # A tool previously pinned, re-registered with pin=False, must leave pinned.
        reg = _FakeRegistry()
        pinned: set[str] = {"t"}
        register_mcp_tools(reg, pinned, {"t": _mcp_tool("big")}, {"big": False})
        assert "t" not in pinned

    def test_none_registry_is_noop(self) -> None:
        pinned: set[str] = set()
        register_mcp_tools(None, pinned, {"t": _mcp_tool("s")}, {"s": True})
        assert pinned == set()


class TestUnregisterMcpServerTools:
    def test_removes_only_target_server(self) -> None:
        reg = _FakeRegistry()
        pinned: set[str] = set()
        register_mcp_tools(reg, pinned, {"a_tool": _mcp_tool("A")}, {"A": True})
        register_mcp_tools(reg, pinned, {"b_tool": _mcp_tool("B")}, {"B": True})
        reg.tools["shell"] = object()
        reg.tool_metadata["shell"] = {"source": "builtin"}

        removed = unregister_mcp_server_tools(reg, pinned, "A")

        assert removed == ["a_tool"]
        assert "a_tool" not in reg.tools
        assert "a_tool" not in reg.tool_metadata
        assert "a_tool" not in pinned
        # Other server + builtin untouched
        assert "b_tool" in reg.tools
        assert "b_tool" in pinned
        assert "shell" in reg.tools

    def test_unknown_server_removes_nothing(self) -> None:
        reg = _FakeRegistry()
        pinned: set[str] = set()
        register_mcp_tools(reg, pinned, {"a_tool": _mcp_tool("A")}, {"A": True})
        assert unregister_mcp_server_tools(reg, pinned, "ghost") == []
        assert "a_tool" in reg.tools


class TestReconcileTools:
    def _registry(self):
        from cogtrix_core.api.session_bridge import ApiSessionRegistry

        reg = ApiSessionRegistry(app_state=MagicMock())
        return reg

    def test_evicts_idle_skips_in_flight(self) -> None:
        reg = self._registry()

        done_task = MagicMock()
        done_task.done.return_value = True
        running_task = MagicMock()
        running_task.done.return_value = False

        idle = SimpleNamespace(
            id="idle", last_activity=0.0, turn_task=done_task, memory_manager=None
        )
        warm_no_task = SimpleNamespace(
            id="warm", last_activity=0.0, turn_task=None, memory_manager=None
        )
        busy = SimpleNamespace(
            id="busy", last_activity=0.0, turn_task=running_task, memory_manager=None
        )
        reg._sessions = {"idle": idle, "warm": warm_no_task, "busy": busy}

        evicted = asyncio.run(reg.reconcile_tools())

        assert evicted == 2
        assert "idle" not in reg._sessions
        assert "warm" not in reg._sessions
        # The in-flight session keeps its snapshot until its turn ends.
        assert "busy" in reg._sessions

    def test_empty_registry_returns_zero(self) -> None:
        reg = self._registry()
        assert asyncio.run(reg.reconcile_tools()) == 0


class TestManagerDisconnectEarlyOut:
    def test_unknown_server_returns_false_without_loop(self) -> None:
        from cogtrix_core.mcp_client import MCPManager

        mgr = MCPManager()
        # No connect ever happened → no background loop. disconnect must not
        # spin one up just to report "nothing to do".
        assert mgr.disconnect("never_connected") is False
        assert mgr._loop is None

    def test_purges_stray_tool_server_mapping(self) -> None:
        from cogtrix_core.mcp_client import MCPManager

        mgr = MCPManager()
        mgr._tool_server_map = {"orphan_tool": "ghost"}
        assert mgr.disconnect("ghost") is False
        assert "orphan_tool" not in mgr._tool_server_map
