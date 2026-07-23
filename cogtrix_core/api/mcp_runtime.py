"""Runtime registration helpers for MCP tools in the API process.

These helpers are the single source of truth for how MCP-discovered tools are
mirrored into the live ``ToolRegistry`` (``app.state.tool_registry``) and the
``app.state.pinned_mcp_tool_names`` set. Both the lifespan startup
(``cogtrix_core/api/app.py``) and the runtime ``/mcp`` routes (``cogtrix_core/api/routes/mcp.py``)
go through them, so the add/restart/delete paths stay byte-for-byte consistent
with what startup builds — see #2151 / #2153.

All functions here are synchronous and perform no ``await``. On the API's
single-threaded event loop that makes each call atomic with respect to session
warming (which snapshots ``tool_registry.tools`` via ``dict(...)``): no other
coroutine can observe a half-mutated registry because control never yields
mid-mutation.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "mcp_builtin_tool_names",
    "register_mcp_tools",
    "unregister_mcp_server_tools",
]


def mcp_builtin_tool_names(tool_registry: Any) -> set[str]:
    """Return the names of all *non-MCP* tools currently in the registry.

    Used as the collision base when connecting/restarting a server: a freshly
    connected MCP tool must not shadow a built-in tool (or an already-registered
    MCP tool — callers union those in separately). Mirrors the startup call,
    which passes the registry's tool names taken *before* any MCP registration.
    """
    if tool_registry is None:
        return set()
    metadata: dict[str, dict[str, Any]] = getattr(tool_registry, "tool_metadata", {}) or {}
    names: set[str] = set()
    for name in getattr(tool_registry, "tools", {}) or {}:
        if metadata.get(name, {}).get("source") != "mcp":
            names.add(name)
    return names


def register_mcp_tools(
    tool_registry: Any,
    pinned_mcp_tool_names: set[str],
    mcp_tools: dict[str, Any],
    pin_map: dict[str, bool],
) -> None:
    """Register MCP-discovered LangChain tools into the live registry.

    Mirrors the lifespan startup registration loop so the runtime add/restart
    routes produce identical registry + pin state.

    Args:
        tool_registry: the live ``ToolRegistry`` (or ``None`` — no-op).
        pinned_mcp_tool_names: the shared pinned set (mutated in place).
        mcp_tools: name → LangChain tool, as returned by ``connect_all`` /
            ``restart`` / ``get_langchain_tools``.
        pin_map: server name → pin flag (from each ``MCPServerConfig.pin``).
    """
    if tool_registry is None:
        return
    for tool_name, tool_obj in mcp_tools.items():
        tool_registry.tools[tool_name] = tool_obj
        srv_name = (getattr(tool_obj, "metadata", None) or {}).get("server", "")
        tool_registry.tool_metadata[tool_name] = {
            "requires_confirmation": (getattr(tool_obj, "metadata", None) or {}).get(
                "requires_confirmation", True
            ),
            "source": "mcp",
            "server": srv_name,
            "pin": pin_map.get(srv_name, True),
        }
        if pin_map.get(srv_name, True):
            pinned_mcp_tool_names.add(tool_name)
        else:
            # A previously-pinned tool may be re-registered with pin=False.
            pinned_mcp_tool_names.discard(tool_name)


def unregister_mcp_server_tools(
    tool_registry: Any,
    pinned_mcp_tool_names: set[str],
    server_name: str,
) -> list[str]:
    """Remove every MCP tool belonging to ``server_name`` from the registry.

    Purges ``tools``, ``tool_metadata`` and the pinned set so the model can no
    longer bind the tools — used by the delete and restart routes (restart
    purges before re-registering the rebuilt set). Returns the removed display
    names.
    """
    if tool_registry is None:
        return []
    metadata: dict[str, dict[str, Any]] = getattr(tool_registry, "tool_metadata", {}) or {}
    removed: list[str] = [
        name
        for name, meta in list(metadata.items())
        if meta.get("source") == "mcp" and meta.get("server", "") == server_name
    ]
    for name in removed:
        tool_registry.tools.pop(name, None)
        tool_registry.tool_metadata.pop(name, None)
        pinned_mcp_tool_names.discard(name)
    return removed
