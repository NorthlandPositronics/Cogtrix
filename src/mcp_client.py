"""
MCP (Model Context Protocol) client manager for Cogtrix.

Bridges the synchronous Cogtrix agent loop with the async MCP SDK by running
a persistent asyncio event loop on a background daemon thread. MCP tools are
discovered at connection time and exposed as LangChain StructuredTool objects.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from src.logging_config import get_logger

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.types import ImageContent, TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore[misc, assignment]
    StdioServerParameters = None  # type: ignore[misc, assignment]
    sse_client = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    ImageContent = None  # type: ignore[misc, assignment]
    TextContent = None  # type: ignore[misc, assignment]
    Tool = None  # type: ignore[misc, assignment]

try:
    from langchain_core.tools import StructuredTool

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    StructuredTool = None  # type: ignore[misc, assignment]
    _LANGCHAIN_AVAILABLE = False

try:
    from pydantic import BaseModel, create_model
    from pydantic import Field as PydanticField

    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore[misc, assignment]
    PydanticField = None  # type: ignore[assignment]
    create_model = None  # type: ignore[assignment]
    _PYDANTIC_AVAILABLE = False


# ── Config dataclass ─────────────────────────────────────────────────────────


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    requires_confirmation: bool = True
    timeout: int = 30


# ── Schema conversion ─────────────────────────────────────────────────────────

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def json_schema_to_pydantic(name: str, schema: dict[str, Any]) -> type:
    """
    Convert a JSON Schema dict to a Pydantic BaseModel class.

    Args:
        name: Tool name (used to derive the model class name).
        schema: JSON Schema dict (typically from MCP Tool.inputSchema).

    Returns:
        A dynamically created Pydantic BaseModel subclass.
    """
    if not _PYDANTIC_AVAILABLE:
        raise ImportError("pydantic is required")

    class_name = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "MCPToolInput"
    if class_name[0].isdigit():
        class_name = f"_{class_name}"

    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required_fields: set[str] = set(schema.get("required", []) or [])

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        raw_type = prop_schema.get("type", "string")
        has_null = False
        if isinstance(raw_type, list):
            non_null = [t for t in raw_type if t != "null"]
            python_type = _JSON_SCHEMA_TYPE_MAP.get(non_null[0] if non_null else "string", str)
            if "null" in raw_type:
                has_null = True
                python_type = Optional[python_type]  # noqa: UP045
        elif "anyOf" in prop_schema or "oneOf" in prop_schema:
            variants = prop_schema.get("anyOf") or prop_schema.get("oneOf") or []
            non_null = [v for v in variants if v.get("type") != "null"]
            has_null = len(non_null) < len(variants)
            first_type = non_null[0].get("type", "string") if non_null else "string"
            python_type = _JSON_SCHEMA_TYPE_MAP.get(first_type, str)
            if has_null:
                python_type = Optional[python_type]  # noqa: UP045
        else:
            python_type = _JSON_SCHEMA_TYPE_MAP.get(raw_type, str)
        description = prop_schema.get("description", "")

        if prop_name in required_fields and not has_null:
            field_definitions[prop_name] = (python_type, PydanticField(description=description))
        else:
            field_definitions[prop_name] = (
                Optional[python_type],  # noqa: UP045
                PydanticField(default=None, description=description),
            )

    return create_model(class_name, **field_definitions)  # type: ignore[call-overload]


# ── MCP result → string ───────────────────────────────────────────────────────


def _result_to_str(result: Any) -> str:
    """Serialize a CallToolResult to a plain string."""
    parts: list[str] = []
    for item in result.content or []:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "mimeType"):
            parts.append(f"[image: {item.mimeType}]")
        else:
            parts.append(str(item))

    text = "\n".join(parts)
    if result.isError:
        text = f"Error: {text}"
    return text


# ── Connection class ──────────────────────────────────────────────────────────


class MCPConnection:
    """Manages a single MCP server connection inside the manager's event loop."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._session: Any | None = None
        self._exit_stack: Any | None = None
        self._tools: list[Any] = []

    @property
    def tools(self) -> list[Any]:
        return self._tools

    async def connect(self) -> None:
        """
        Open transport, start ClientSession, initialize, and list tools.

        Uses AsyncExitStack to manage the nested async context managers so
        that both the transport and the session are closed together on close().
        """
        import contextlib

        self._exit_stack = contextlib.AsyncExitStack()
        await self._exit_stack.__aenter__()

        cfg = self._config
        if cfg.url is not None:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(url=cfg.url, headers=cfg.headers)
            )
        elif cfg.command is not None:
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        else:
            raise ValueError(
                f"MCPServerConfig '{cfg.name}' requires either 'command' (stdio) or 'url' (SSE)"
            )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        list_result = await self._session.list_tools()
        self._tools = list_result.tools or []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Call a tool on the MCP server and return the result as a string.

        Args:
            name: MCP tool name.
            arguments: Tool arguments dict.

        Returns:
            String representation of the tool result.
        """
        if self._session is None:
            return "Error: MCP connection not established"
        result = await self._session.call_tool(name, arguments)
        return _result_to_str(result)

    async def close(self) -> None:
        """Clean up the session and transport."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception as exc:
                get_logger().warning("MCP: error during connection cleanup: %s", exc)
            self._exit_stack = None
        self._session = None


# ── Tool wrapper ──────────────────────────────────────────────────────────────


def _create_mcp_tool_wrapper(
    mcp_tool: Any,
    server_name: str,
    config: MCPServerConfig,
    manager: MCPManager,
) -> Any:
    """
    Create a LangChain StructuredTool that delegates calls to MCPManager.call_tool().

    Args:
        mcp_tool: MCP Tool object from list_tools().
        server_name: Name of the server that owns this tool.
        config: Server config (used for requires_confirmation and timeout).
        manager: MCPManager instance used for sync-safe call dispatch.

    Returns:
        A LangChain StructuredTool, or None if creation fails.
    """
    if not _LANGCHAIN_AVAILABLE or not _PYDANTIC_AVAILABLE:
        return None

    log = get_logger()
    tool_name: str = mcp_tool.name
    tool_description: str = (
        mcp_tool.description or f"MCP tool '{tool_name}' on server '{server_name}'"
    )
    input_schema_dict: dict[str, Any] = mcp_tool.inputSchema or {}

    try:
        args_schema = json_schema_to_pydantic(tool_name, input_schema_dict)
    except Exception as exc:
        log.warning("MCP tool '%s': failed to build args schema: %s", tool_name, exc)
        return None

    def _call_fn(**kwargs: Any) -> str:
        return manager.call_tool(server_name, tool_name, kwargs)

    _call_fn.__name__ = tool_name
    _call_fn.__doc__ = tool_description

    try:
        tool = StructuredTool.from_function(
            func=_call_fn,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            metadata={
                "source": "mcp",
                "server": server_name,
                "requires_confirmation": config.requires_confirmation,
            },
        )
        return tool
    except Exception as exc:
        log.warning("MCP tool '%s': StructuredTool creation failed: %s", tool_name, exc)
        return None


# ── Manager class ─────────────────────────────────────────────────────────────


class MCPManager:
    """
    Manages all MCP server connections and provides a synchronous interface.

    A single daemon thread runs the event loop so that async MCP calls can be
    issued from the synchronous Cogtrix agent loop via run_coroutine_threadsafe.
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._tool_server_map: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ── Internal event loop management ───────────────────────────────────────

    def _ensure_loop(self) -> None:
        """Start the background event loop thread if it is not already running."""
        if self._loop is not None and not self._loop.is_closed():
            return

        self._loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            assert self._loop is not None
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True, name="mcp-event-loop")
        self._thread.start()

    def _run(self, coro: Any, timeout: int = 30) -> Any:
        """
        Schedule *coro* on the background event loop and block until it finishes.

        Args:
            coro: Coroutine to execute.
            timeout: Maximum seconds to wait for the result.

        Returns:
            Return value of the coroutine.

        Raises:
            TimeoutError: If the coroutine does not complete within *timeout* seconds.
            Exception: Propagates any exception raised inside the coroutine.
        """
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def connect_all(
        self,
        configs: list[MCPServerConfig],
        builtin_tool_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Connect to all configured MCP servers and return LangChain tools.

        Servers that fail to connect are skipped with a warning — they do not
        prevent other servers from being used.

        Args:
            configs: List of MCPServerConfig objects.
            builtin_tool_names: Optional set of built-in tool names to check for
                collisions against, in addition to other MCP tools.

        Returns:
            Dict mapping tool name → LangChain StructuredTool for every tool
            discovered across all servers.
        """
        if not MCP_AVAILABLE:
            log = get_logger()
            log.warning("MCP SDK not installed; no MCP servers will be connected")
            return {}

        log = get_logger()
        self._ensure_loop()
        all_tools: dict[str, Any] = {}

        for cfg in configs:
            self._configs[cfg.name] = cfg
            conn = MCPConnection(cfg)
            try:
                self._run(conn.connect(), timeout=cfg.timeout)
                self._connections[cfg.name] = conn
                log.info("MCP: connected to server '%s' (%d tools)", cfg.name, len(conn.tools))
            except Exception as exc:
                log.warning("MCP: failed to connect to server '%s': %s", cfg.name, exc)
                continue

            for mcp_tool in conn.tools:
                original_name: str = mcp_tool.name
                tool_name = original_name

                if tool_name in all_tools or (
                    builtin_tool_names and tool_name in builtin_tool_names
                ):
                    prefixed = f"{cfg.name}_{tool_name}"
                    if prefixed in all_tools:
                        log.error(
                            "MCP: tool '%s' from server '%s': prefixed name '%s' also collides;"
                            " skipping tool",
                            original_name,
                            cfg.name,
                            prefixed,
                        )
                        continue
                    log.warning(
                        "MCP: tool name collision '%s' from server '%s'; renamed to '%s'",
                        tool_name,
                        cfg.name,
                        prefixed,
                    )
                    tool_name = prefixed

                lc_tool = _create_mcp_tool_wrapper(mcp_tool, cfg.name, cfg, self)
                if lc_tool is None:
                    log.warning(
                        "MCP: skipping tool '%s' from server '%s' (wrapper creation failed)",
                        original_name,
                        cfg.name,
                    )
                    continue

                # Rename StructuredTool if we had a collision
                if tool_name != original_name:
                    lc_tool.name = tool_name

                all_tools[tool_name] = lc_tool
                self._tool_server_map[tool_name] = cfg.name

        return all_tools

    def call_tool(self, server_name: str, mcp_tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Call an MCP tool synchronously.

        ``mcp_tool_name`` is the original MCP-internal tool name as returned by
        ``list_tools()`` — never a collision-prefixed display name.  The closure
        created in ``_create_mcp_tool_wrapper`` always captures and passes the
        original name, so no prefix-stripping is needed here.

        Args:
            server_name: Name of the MCP server.
            mcp_tool_name: Original MCP tool name (no collision prefix).
            arguments: Dict of tool arguments.

        Returns:
            String result from the MCP tool, or an error string.
        """
        log = get_logger()
        conn = self._connections.get(server_name)
        if conn is None:
            return f"Error: MCP server '{server_name}' is not connected"

        cfg = self._configs.get(server_name)
        timeout = cfg.timeout if cfg else 30

        try:
            return self._run(conn.call_tool(mcp_tool_name, arguments), timeout=timeout)
        except TimeoutError:
            return f"Error: MCP tool '{mcp_tool_name}' timed out after {timeout}s"
        except Exception as exc:
            log.error(
                "MCP tool call '%s' on server '%s' failed: %s", mcp_tool_name, server_name, exc
            )
            return f"Error: {exc}"

    def close_all(self) -> None:
        """Close all MCP connections and stop the background event loop."""
        log = get_logger()
        for name, conn in self._connections.items():
            try:
                self._run(conn.close(), timeout=10)
            except Exception as exc:
                log.warning("MCP: error closing connection '%s': %s", name, exc)
        self._connections.clear()
        self._tool_server_map.clear()

        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    def restart(
        self,
        server_name: str | None = None,
        builtin_tool_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Reconnect one or all MCP servers and return rebuilt LangChain tools.

        Args:
            server_name: If given, reconnect only this server. Otherwise restart all.
            builtin_tool_names: Optional set of built-in (non-MCP) tool names to
                include in collision detection when rebuilding tools.

        Returns:
            Dict mapping tool name to LangChain StructuredTool for every tool
            available on the restarted servers.
        """
        log = get_logger()
        if server_name is not None:
            targets = [server_name] if server_name in self._configs else []
            if not targets:
                log.warning("MCP: cannot restart unknown server '%s'", server_name)
                return {}
        else:
            targets = list(self._configs.keys())

        for name in targets:
            # Purge stale tool-server mappings for this server
            for key in [k for k, v in self._tool_server_map.items() if v == name]:
                del self._tool_server_map[key]
            old_conn = self._connections.pop(name, None)
            if old_conn is not None:
                try:
                    self._run(old_conn.close(), timeout=10)
                except Exception as exc:
                    log.warning("MCP: error closing '%s' during restart: %s", name, exc)

            cfg = self._configs[name]
            new_conn = MCPConnection(cfg)
            try:
                self._run(new_conn.connect(), timeout=cfg.timeout)
                self._connections[name] = new_conn
                log.info("MCP: reconnected server '%s' (%d tools)", name, len(new_conn.tools))
            except Exception as exc:
                log.warning("MCP: restart of server '%s' failed: %s", name, exc)

        # Collect tool names from servers NOT being restarted
        known_tool_names: set[str] = set()
        for tool_name_key, srv in self._tool_server_map.items():
            if srv not in targets:
                known_tool_names.add(tool_name_key)

        new_tools: dict[str, Any] = {}
        for name in targets:
            if name in self._connections:
                server_tools = self.get_langchain_tools(
                    server_name=name,
                    builtin_tool_names=builtin_tool_names,
                    known_tool_names=known_tool_names,
                )
                known_tool_names |= set(server_tools)
                new_tools.update(server_tools)
        return new_tools

    def get_langchain_tools(
        self,
        server_name: str | None = None,
        builtin_tool_names: set[str] | None = None,
        known_tool_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build LangChain StructuredTool objects for currently connected servers.

        Applies the same collision-rename logic as ``connect_all()`` so that
        tool names returned here are consistent with the registry.

        Args:
            server_name: If given, return tools only for that server.
                         If ``None``, return tools for all connected servers.
            builtin_tool_names: Optional set of built-in (non-MCP) tool names to
                include in collision detection.
            known_tool_names: Optional set of already-registered tool names from
                other MCP servers to include in collision detection.

        Returns:
            Dict mapping display tool name → LangChain StructuredTool.
        """
        log = get_logger()
        all_tools: dict[str, Any] = {}
        targets = [server_name] if server_name else list(self._connections.keys())

        for name in targets:
            conn = self._connections.get(name)
            cfg = self._configs.get(name)
            if conn is None or cfg is None:
                continue

            for mcp_tool in conn.tools:
                original_name: str = mcp_tool.name
                tool_name = original_name

                collision_names = set(all_tools)
                if builtin_tool_names:
                    collision_names |= builtin_tool_names
                if known_tool_names:
                    collision_names |= known_tool_names

                if tool_name in collision_names:
                    prefixed = f"{name}_{tool_name}"
                    if prefixed in collision_names:
                        log.error(
                            "MCP: tool '%s' from server '%s': prefixed name '%s' also collides;"
                            " skipping tool",
                            original_name,
                            name,
                            prefixed,
                        )
                        continue
                    log.warning(
                        "MCP: tool name collision '%s' from server '%s'; renamed to '%s'",
                        tool_name,
                        name,
                        prefixed,
                    )
                    tool_name = prefixed

                lc_tool = _create_mcp_tool_wrapper(mcp_tool, name, cfg, self)
                if lc_tool is None:
                    continue
                if tool_name != original_name:
                    lc_tool.name = tool_name
                all_tools[tool_name] = lc_tool
                self._tool_server_map[tool_name] = name

        return all_tools

    def get_server_info(self) -> list[dict[str, Any]]:
        """
        Return a summary of all known servers (for the /mcp command).

        Returns:
            List of dicts with keys: name, connected, tool_count, tools, transport.
        """
        info: list[dict[str, Any]] = []
        for name, cfg in self._configs.items():
            conn = self._connections.get(name)
            transport = "sse" if cfg.url else "stdio"
            info.append(
                {
                    "name": name,
                    "connected": conn is not None,
                    "tool_count": len(conn.tools) if conn else 0,
                    "tools": [t.name for t in conn.tools] if conn else [],
                    "transport": transport,
                    "endpoint": cfg.url or cfg.command or "",
                }
            )
        return info


__all__ = [
    "MCP_AVAILABLE",
    "MCPServerConfig",
    "MCPConnection",
    "MCPManager",
    "json_schema_to_pydantic",
]
