"""
MCP (Model Context Protocol) client manager for Cogtrix.

Bridges the synchronous Cogtrix agent loop with the async MCP SDK by running
a persistent asyncio event loop on a background daemon thread. MCP tools are
discovered at connection time and exposed as LangChain StructuredTool objects.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

# CI trigger fix: blank line added to force CI re-run
import ipaddress
import os
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from src.logging_config import get_logger
from src.tools.error_sanitizer import sanitize_error

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


# ── Config field allowlists (shared with config loaders) ─────────────────────

# Fields forwarded to ``MCPServerConfig`` as constructor kwargs. Loaders
# filter the user's per-server YAML dict against this set so unknown
# fields don't reach the dataclass constructor (which would raise
# ``TypeError`` on unexpected keywords).
KNOWN_MCP_FIELDS: frozenset[str] = frozenset(
    {
        # Stdio transport
        "command",
        "args",
        "env",
        # SSE transport
        "url",
        "headers",
        # Cogtrix-side semantics
        "requires_confirmation",
        "timeout",
        "pin",
        "allow_insecure",
    }
)

# Fields the user legitimately keeps in their per-server YAML for human
# reference (linking the cogtrix config to an external server's own
# settings — typically docker-compose volume roots or out-of-band
# auth metadata) but that Cogtrix does NOT consume. Loaders accept
# these without warning AND without forwarding them to MCPServerConfig.
#
# cogtrix52 surfaced ``allowed_directories``: the user pasted a comment
# in their YAML explaining 'these must match the roots passed to
# server-filesystem in docker-compose.yml' and Cogtrix faithfully
# emitted ``MCP server 'filesystem': ignoring unknown config keys:
# allowed_directories`` on every startup. The field is doc-only —
# it lives in YAML for human cross-reference, nothing programmatic
# reads it. Acknowledging that explicitly here is cleaner than
# either rejecting it (false positive) or forwarding it (no slot in
# MCPServerConfig).
#
# When extending: each entry MUST have a written reason in the
# comment above it. If a field starts being consumed by Cogtrix,
# move it to ``KNOWN_MCP_FIELDS`` and wire it through MCPServerConfig.
DOC_ONLY_MCP_FIELDS: frozenset[str] = frozenset(
    {
        "allowed_directories",
    }
)


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
    pin: bool = True
    allow_insecure: bool = False


# ── SSRF validation helpers ───────────────────────────────────────────────────


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if *ip_str* resolves to a non-public address."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_mcp_url(url: str, allow_insecure: bool = False) -> None:
    """Raise ValueError if *url* is unsafe for MCP SSE connections.

    Checks:
      1. Scheme must be ``https`` (``http`` only when *allow_insecure* is True).
      2. Host must not resolve to loopback, RFC1918, link-local, or multicast.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid MCP SSE URL: {url}")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"MCP SSE URL scheme must be http or https, got: {parsed.scheme}")

    if parsed.scheme == "http" and not allow_insecure:
        raise ValueError(
            f"Insecure MCP SSE URL ({url}) rejected. "
            "Set allow_insecure=True to permit http:// connections."
        )

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError(f"Invalid MCP SSE URL (no host): {url}")

    # When allow_insecure is explicitly set, skip all IP/host blocking.
    # The caller has acknowledged that the target may be on an internal network.
    if allow_insecure:
        return

    # Block well-known internal hostnames by name (defense-in-depth).
    blocked_hosts = {
        "localhost",
        "metadata.google.internal",
        "instance-data",
        "169.254.169.254",
    }
    if hostname.lower() in blocked_hosts:
        raise ValueError(f"MCP SSE URL points to internal host: {hostname}")

    # If hostname is already a raw IP literal, validate it directly.
    if _is_blocked_ip(hostname):
        raise ValueError(f"MCP SSE URL resolves to blocked IP: {hostname}")

    # Resolve hostname and validate returned addresses.
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"MCP SSE URL host cannot be resolved: {hostname}") from exc

    for _family, _socktype, _proto, _canonname, sockaddr in addrinfo:
        ip_str = str(sockaddr[0])
        if _is_blocked_ip(ip_str):
            raise ValueError(f"MCP SSE URL resolves to blocked IP: {ip_str}")


async def _validate_mcp_url_off_loop(url: str, allow_insecure: bool = False) -> None:
    """Async wrapper that runs :func:`_validate_mcp_url` off the event loop.

    ``_validate_mcp_url`` performs a blocking ``socket.getaddrinfo`` lookup.
    Calling it directly from ``MCPConnection.connect`` (which runs on the
    background MCP event loop) would block that loop — and therefore every
    other concurrent server connect plus the heartbeat — for the duration of
    a slow/wedged DNS resolution (#2154). Offload it to the default executor
    so the lookup runs on a worker thread; any ``ValueError`` it raises
    propagates unchanged to the awaiter.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _validate_mcp_url, url, allow_insecure)


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

_PROTECTED_BRANCH_NAMES: frozenset[str] = frozenset({"next", "main"})
_PROTECTED_BRANCH_TOOL_NAMES: frozenset[str] = frozenset({"create_or_update_file", "push_files"})


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> type:
    """Resolve a JSON Schema local $ref pointer to a Python type.

    Only ``#/...`` pointers are supported.  External refs (``http://...``,
    relative paths) fall back to ``str`` with a warning.
    """
    if not ref.startswith("#/"):
        get_logger().warning("MCP: unsupported external $ref %r — falling back to str", ref)
        return str
    parts = ref[2:].split("/")
    node: Any = root_schema
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return str
        node = node[part]
    if not isinstance(node, dict):
        return str
    if "const" in node:
        return Literal[node["const"]]  # type: ignore[return-value]
    raw_type = node.get("type", "string")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        return _JSON_SCHEMA_TYPE_MAP.get(non_null[0] if non_null else "string", str)
    return _JSON_SCHEMA_TYPE_MAP.get(raw_type, str)


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
        elif "allOf" in prop_schema:
            # Use the first concrete type from allOf subschemas
            sub_type = "string"
            for sub in prop_schema.get("allOf", []):
                if sub.get("type") in _JSON_SCHEMA_TYPE_MAP:
                    sub_type = sub["type"]
                    break
            python_type = _JSON_SCHEMA_TYPE_MAP.get(sub_type, str)
        elif "enum" in prop_schema:
            # enum values — all values as str (Pydantic Literal is impractical for
            # dynamic schema construction; str covers the common case)
            python_type = str
        elif "$ref" in prop_schema:
            python_type = _resolve_ref(prop_schema["$ref"], schema)
        elif "const" in prop_schema:
            python_type = Literal[prop_schema["const"]]  # type: ignore[assignment]
        else:
            python_type = _JSON_SCHEMA_TYPE_MAP.get(raw_type)
            if python_type is None:
                get_logger().warning(
                    "MCP tool '%s': unsupported JSON Schema type '%s' for property '%s', "
                    "falling back to str",
                    name,
                    raw_type,
                    prop_name,
                )
                python_type = str
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
        # Enrich filesystem access-denied errors with actionable guidance so the
        # agent can self-correct instead of retrying with the same wrong path.
        if "path outside allowed directories" in text or (
            "Access denied" in text and "allowed directories" in text
        ):
            workspace_prefix = _get_workspace_prefix()
            text += (
                f"\nHint: For Cogtrix workspace files prefix the path with {workspace_prefix}/ "
                f"(e.g. {workspace_prefix}/src/tools/foo.py, "
                f"{workspace_prefix}/.github/workflows/ci.yml). "
                "For GitHub-hosted content use get_file_contents instead of read_text_file."
            )
    return text


def _reject_protected_branch_write(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Block direct writes to protected branches."""
    if tool_name not in _PROTECTED_BRANCH_TOOL_NAMES:
        return None

    branch = arguments.get("branch")
    if not isinstance(branch, str):
        return None

    branch_name = branch.strip().lower()
    if branch_name not in _PROTECTED_BRANCH_NAMES:
        return None

    return (
        f"Error: Direct push to protected branch '{branch_name}' is not allowed. "
        "Create a feature branch and open a PR instead."
    )


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

    @property
    def session(self) -> Any:
        return self._session

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
            # Prevent SSRF via admin-controlled MCP config. Run the validation
            # (which does a blocking getaddrinfo) off the event loop so a slow
            # DNS lookup can't stall the MCP loop / heartbeat (#2154).
            await _validate_mcp_url_off_loop(cfg.url, allow_insecure=cfg.allow_insecure)
            # sse_client(timeout=N) sets the connect/write/pool timeout.
            # sse_client(sse_read_timeout=X) controls how long to wait between
            # 3600s balances two failure modes: the default 300s fires on
            # legitimate idle gaps between tool calls in long sessions, while
            # None disables liveness detection entirely — leaving half-open
            # TCP connections (NAT idle-drop, server crash without FIN)
            # invisible to the client for up to the OS keepalive period (~2h).
            # One hour is long enough for any realistic idle period and short
            # enough to detect network failures within a session.
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(
                    url=cfg.url,
                    headers=cfg.headers,
                    timeout=float(cfg.timeout),
                    sse_read_timeout=3600,
                )
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
            except RuntimeError as exc:
                # anyio cancel scopes are task-local.  When close() runs in a
                # different asyncio Task than connect() (e.g. run_coroutine_threadsafe
                # creates a new task per call), anyio raises this RuntimeError.
                # The connection is being replaced immediately, so log at DEBUG only.
                if "cancel scope" in str(exc):
                    get_logger().debug("MCP: connection cleanup (anyio task boundary): %s", exc)
                else:
                    get_logger().warning("MCP: error during connection cleanup: %s", exc)
            except Exception as exc:
                get_logger().warning("MCP: error during connection cleanup: %s", exc)
            self._exit_stack = None
        self._session = None


# ── Filesystem path normalisation ────────────────────────────────────────────

# Known MCP filesystem tool names (explicit allowlist to avoid matching
# GitHub/remote tools such as create_or_update_file).
_FILESYSTEM_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "read_text_file",
        "write_file",
        "edit_file",
        "create_file",
        "delete_file",
        "move_file",
        "copy_file",
        "search_files",
        "list_files",
        "get_file_info",
        "create_directory",
        "list_directory",
        "directory_tree",
        "delete_directory",
        "move_directory",
        "read_directory",
        "write_directory",
    }
)

_PATH_ARG_NAMES: frozenset[str] = frozenset(
    {"path", "file_path", "directory", "dir", "src", "dest", "source", "target"}
)

_DEFAULT_MCP_WORKSPACE_PREFIX = "/workspace"
_MCP_WORKSPACE_PREFIX_ENV = "COGTRIX_MCP_WORKSPACE_PREFIX"


def _get_workspace_prefix() -> str:
    """Resolve filesystem path prefix for MCP tools.

    Reads ``COGTRIX_MCP_WORKSPACE_PREFIX`` and normalizes it to an absolute
    path without trailing slash. Invalid values (empty or root ``/``) fall back
    to ``/workspace``.
    """
    raw_prefix = os.getenv(_MCP_WORKSPACE_PREFIX_ENV, _DEFAULT_MCP_WORKSPACE_PREFIX).strip()
    if not raw_prefix:
        return _DEFAULT_MCP_WORKSPACE_PREFIX

    normalized = raw_prefix if raw_prefix.startswith("/") else f"/{raw_prefix}"
    normalized = normalized.rstrip("/")
    if not normalized or normalized == "/":
        return _DEFAULT_MCP_WORKSPACE_PREFIX
    return normalized


def _is_filesystem_tool(tool_name: str) -> bool:
    """Return True when *tool_name* is a known filesystem MCP tool.

    Handles collision-prefixed names (e.g. ``filesystem_read_file``) by
    checking both exact match and suffix match.
    """
    lowered = tool_name.lower()
    if lowered in _FILESYSTEM_TOOL_NAMES:
        return True
    return any(lowered.endswith(f"_{name}") for name in _FILESYSTEM_TOOL_NAMES)


def _normalize_mcp_paths(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Auto-prepend ``/workspace/`` to relative paths for filesystem MCP tools.

    Only touches arguments whose names are in ``_PATH_ARG_NAMES`` and whose
    values are strings that do not already start with ``/``.
    """
    if not _is_filesystem_tool(tool_name):
        return kwargs
    workspace_prefix = _get_workspace_prefix()
    normalized: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _PATH_ARG_NAMES and isinstance(v, str) and v and not v.startswith("/"):
            normalized[k] = f"{workspace_prefix}/{v.lstrip('/')}"
        else:
            normalized[k] = v
    return normalized


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
        # Strip None values before forwarding — MCP servers (e.g. GitHub) use
        # strict Zod schemas that reject null for optional parameters.  The LLM
        # may include optional params as None/null when it doesn't intend to set
        # them; omitting them entirely matches the MCP server's expectations.
        guard_error = _reject_protected_branch_write(tool_name, kwargs)
        if guard_error is not None:
            return guard_error
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        clean_kwargs = _normalize_mcp_paths(tool_name, clean_kwargs)
        return manager.call_tool(server_name, tool_name, clean_kwargs)

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


# ── Connection error classifier ──────────────────────────────────────────────

_CONNECTION_ERROR_TYPES: frozenset[str] = frozenset(
    {
        # asyncio / anyio
        "CancelledError",
        "TimeoutError",
        # httpx / httpcore
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "LocalProtocolError",
        # stdlib / OS
        "ConnectionResetError",
        "ConnectionRefusedError",
        "BrokenPipeError",
        "ConnectionAbortedError",
    }
)


def _is_connection_error(exc: Exception) -> bool:
    """Return True when *exc* indicates a broken or half-open SSE connection.

    Checks both the direct exception type and the entire ``__cause__`` /
    ``__context__`` chain so that wrapped exceptions are caught too.
    """
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in _CONNECTION_ERROR_TYPES:
            return True
        # Walk the chain: explicit cause first, then implicit context.
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return False


# ── Startup retry constants ───────────────────────────────────────────────────

# Number of additional connection attempts after the first failure at startup.
# Applied per-server; retries run concurrently across servers.
_STARTUP_MAX_RETRIES: int = 2
# Seconds to wait before each successive retry attempt (index = attempt number).
_STARTUP_RETRY_DELAYS: tuple[float, ...] = (2.0, 5.0)


# ── Manager class ─────────────────────────────────────────────────────────────


class MCPManager:
    """
    Manages all MCP server connections and provides a synchronous interface.

    A single daemon thread runs the event loop so that async MCP calls can be
    issued from the synchronous Cogtrix agent loop via run_coroutine_threadsafe.
    """

    # Interval between heartbeat pings. Must be shorter than the SSE server's
    # keepalive interval (30 s in docker-compose) so the connection is exercised
    # before the server closes the idle SSE stream.
    _HEARTBEAT_INTERVAL: int = 20  # seconds

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._tool_server_map: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        # Unified shutdown guard: set to True by close_all() under _loop_lock
        # before teardown starts. _ensure_loop() checks this flag (under the
        # same lock) and _reconnect_server_async() checks it before reconnecting,
        # so both loop-start and reconnect paths are blocked by one coordinated
        # state transition (#425, #546).
        self._shutting_down: bool = False
        # Track in-flight concurrent.futures.Future objects created by _run()
        # so close_all() can cancel them before stopping the event loop.
        # Without this, a coroutine scheduled via run_coroutine_threadsafe()
        # may still be executing when the loop is stopped, leaving it
        # unreferenced and triggering "RuntimeWarning: coroutine was never
        # awaited" at GC.
        self._pending_futures: set[concurrent.futures.Future[Any]] = set()
        # Heartbeat runs as a native asyncio Task inside the background loop so
        # that list_tools() and reconnect close/connect calls share the same task
        # context, avoiding anyio cancel-scope task-locality violations.
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Per-server asyncio.Lock prevents the heartbeat coroutine and the
        # call_tool() error-handler from racing on _connections for the same
        # server (#427).  Locks are created lazily inside the event loop.
        self._reconnect_locks: dict[str, asyncio.Lock] = {}
        # Signals when the manager has finished rediscovering tools after a
        # connect/reconnect cycle.  Callers may wait on this to avoid binding
        # the model against stale tool state during MCP churn.
        self.tools_ready = threading.Event()
        self.tools_ready.set()
        # Guard against auto-reconnect races during shutdown.
        # API alias for restart method (used by mcp.py routes)
        self.restart_server = self.restart

    # ── Internal event loop management ───────────────────────────────────────

    def _ensure_loop(self) -> None:
        """Start the background event loop thread if it is not already running."""
        with self._loop_lock:
            if self._shutting_down:
                raise RuntimeError(
                    "MCP event loop cannot be started after close_all() has been called"
                )
            if self._loop is not None and not self._loop.is_closed():
                return
            self._loop = asyncio.new_event_loop()

            # Suppress the benign shutdown-race RuntimeError so it never reaches
            # the terminal.  When the thread-pool executor is torn down while an
            # MCP SSE DNS lookup is still in flight the asyncio default handler
            # would print the full traceback.  Route it to DEBUG instead.
            _log = get_logger()

            def _exception_handler(
                loop: asyncio.AbstractEventLoop,
                context: dict,  # type: ignore[type-arg]
            ) -> None:
                exc = context.get("exception")
                if isinstance(
                    exc, RuntimeError
                ) and "cannot schedule new futures after shutdown" in str(exc):
                    _log.debug("MCP: suppressed shutdown race in background loop: %s", exc)
                    return
                loop.default_exception_handler(context)

            self._loop.set_exception_handler(_exception_handler)

            def _run_loop() -> None:
                asyncio.set_event_loop(self._loop)
                assert self._loop is not None
                self._loop.run_forever()

            self._thread = threading.Thread(target=_run_loop, daemon=True, name="mcp-event-loop")
            self._thread.start()
            self._start_heartbeat()

    def _start_heartbeat(self) -> None:
        """Schedule the heartbeat coroutine on the background event loop.

        Running as a native asyncio Task means list_tools() and reconnect
        close/connect calls all execute inside the same task, avoiding anyio
        cancel-scope task-locality violations that fire when the heartbeat ran
        as a threading.Thread calling _run() (which creates a new Task per ping).
        """
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        async def _schedule() -> None:
            self._heartbeat_task = asyncio.ensure_future(self._heartbeat_coro())

        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(_schedule(), self._loop).result(timeout=5)

    async def _reconnect_server_async(self, server_name: str) -> None:
        """Async reconnect — used by the heartbeat coroutine AND the sync error handler.

        A per-server asyncio.Lock serialises concurrent reconnect attempts so
        that the heartbeat and the call_tool() error-handler never race on
        ``_connections`` for the same server (#427).  Without the lock both
        paths pop the old connection and start ``await conn.connect()``;
        whichever finishes last overwrites the winner's result and leaves the
        winner's ``_exit_stack`` unclosed, leaking file descriptors.
        """
        log = get_logger()
        if self._shutting_down:
            log.debug("MCP: skipping reconnect for '%s' — manager is shutting down", server_name)
            return
        cfg = self._configs.get(server_name)
        if cfg is None:
            raise RuntimeError(f"Unknown server '{server_name}'")
        # Lazy-create the per-server lock inside the event loop so it is
        # always bound to the correct running loop.
        lock = self._reconnect_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            if self._shutting_down:
                return
            old_conn = self._connections.pop(server_name, None)
            if old_conn is not None:
                try:
                    await old_conn.close()
                except Exception as exc:
                    log.debug("MCP: error closing stale connection for '%s': %s", server_name, exc)
            new_conn = MCPConnection(cfg)
            try:
                await asyncio.wait_for(new_conn.connect(), timeout=30)
            except Exception:
                # Clean up partial connection so exit stacks and file
                # descriptors are not leaked if connect() fails mid-way.
                try:
                    await new_conn.close()
                except Exception:
                    pass
                raise
            self._connections[server_name] = new_conn
            log.info(
                "MCP: auto-reconnected server '%s' (%d tools)", server_name, len(new_conn.tools)
            )

    async def _heartbeat_coro(self) -> None:
        """Ping each MCP server every _HEARTBEAT_INTERVAL seconds.

        Runs as a long-lived asyncio Task so all connection operations share
        the correct task context, preventing anyio cancel-scope violations.
        Exits when there are no connections AND no configs, or when the
        manager is shutting down.
        """
        log = get_logger()
        while True:
            try:
                await asyncio.sleep(self._HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            # Stop if the manager was garbage-collected without close_all().
            if self._shutting_down or (not self._connections and not self._configs):
                break
            for name, conn in list(self._connections.items()):
                try:
                    await conn.session.list_tools()
                    log.debug("MCP: heartbeat ok for server '%s'", name)
                except Exception as exc:
                    log.warning(
                        "MCP: heartbeat failed for server '%s' (%s) — will reconnect",
                        name,
                        exc,
                    )
                    try:
                        await self._reconnect_server_async(name)
                    except Exception as reconnect_exc:
                        log.error(
                            "MCP: heartbeat reconnect for '%s' failed: %s",
                            name,
                            reconnect_exc,
                        )

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
        with self._loop_lock:
            loop = self._loop
            if loop is None or loop.is_closed():
                coro.close()
                raise RuntimeError("MCP event loop is not running or has been closed")
            future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
            self._pending_futures.add(future)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise
        finally:
            self._pending_futures.discard(future)

    # ── Public API ────────────────────────────────────────────────────────────

    async def _connect_one_async(self, cfg: MCPServerConfig) -> tuple[str, MCPConnection | None]:
        """
        Connect to a single MCP server asynchronously, with startup retry.

        On failure the failed MCPConnection is explicitly closed within this
        task so that anyio cancel scopes are exited from the correct task
        context.  Without explicit close(), Python's async-generator GC
        finalizer creates a new task for cleanup, crossing the anyio task
        boundary and raising "Attempted to exit cancel scope in a different
        task" (#403).

        Connection errors are retried up to _STARTUP_MAX_RETRIES times with
        increasing back-off to handle containers that aren't ready yet (#393).
        ValueError (URL validation) is never retried.

        Returns:
            Tuple of (server_name, MCPConnection) on success, or
            (server_name, None) after all retries are exhausted.
        """
        log = get_logger()
        for attempt in range(_STARTUP_MAX_RETRIES + 1):
            conn = MCPConnection(cfg)
            try:
                await asyncio.wait_for(conn.connect(), timeout=cfg.timeout)
                if attempt > 0:
                    log.info("MCP: connected to server '%s' after %d retries", cfg.name, attempt)
                return cfg.name, conn
            except Exception as exc:
                # ExceptionGroup (TaskGroup) hides the real cause in ``.exceptions``.
                # Walk down to the deepest non-group exception so the log shows the
                # actual error (e.g. ``ConnectError`` / ``FileNotFoundError``)
                # rather than the wrapper. The attribute is ``exceptions`` (a
                # tuple) per the BaseExceptionGroup spec — earlier code used
                # ``__exceptions__`` which doesn't exist; the loop never
                # iterated and the cause stayed the wrapper, so every log
                # line showed "ExceptionGroup: unhandled errors in a
                # TaskGroup" instead of the real reason, and the fail-fast
                # ``isinstance(cause, FileNotFoundError | PermissionError)``
                # check below was effectively dead code.
                cause: BaseException = exc
                while isinstance(cause, BaseExceptionGroup) and cause.exceptions:
                    cause = cause.exceptions[0]

                # Explicitly close the failed connection within this task so
                # anyio cancel scopes are exited in the correct task context (#403).
                try:
                    await conn.close()
                except Exception as close_exc:
                    log.debug("MCP: cleanup after failed connect for '%s': %s", cfg.name, close_exc)

                # Classify fail-fast causes: missing executable, bad config,
                # permanent DNS NXDOMAIN — retrying won't help and we
                # shouldn't keep banging on it.
                #
                # ``socket.gaierror`` covers DNS failures from stdlib paths;
                # ``httpx``/``httpcore`` ConnectError instances with
                # ``[Errno -2] Name or service not known`` carry the same
                # signal at the HTTP layer (cogtrix52: ``mcp-github`` /
                # ``mcp-filesystem`` / ``mcp-slack`` not resolvable from
                # the cogtrix container, three retries do nothing useful).
                import socket as _socket

                fail_fast = (
                    isinstance(exc, ValueError)
                    or isinstance(cause, FileNotFoundError | PermissionError)
                    or isinstance(cause, _socket.gaierror)
                    or (
                        # ConnectError that wraps an NXDOMAIN-class lookup
                        # failure — match on the message because httpx wraps
                        # gaierror in its own ConnectError subclass.
                        type(cause).__name__ == "ConnectError"
                        and "[Errno -2]" in str(cause)
                    )
                )

                # Network-error classes the WARNING line above (or below)
                # already names in full. Suppressing the multi-frame
                # traceback dump for these saves ~125 lines of library
                # plumbing per failure — every frame is httpx → httpcore →
                # anyio → mcp.sse, no Cogtrix code, zero additional
                # diagnostic value. The full traceback STILL fires for
                # unexpected exception classes so genuinely-novel
                # failures keep their forensic trail.
                _network_error_classes = (
                    "ConnectError",
                    "ConnectionRefusedError",
                    "ConnectionResetError",
                    "TimeoutError",
                    "gaierror",
                    "EOFError",
                )
                _is_known_network_error = type(cause).__name__ in _network_error_classes

                if fail_fast:
                    log.warning(
                        "MCP: server '%s' will not be available: %s: %s",
                        cfg.name,
                        type(cause).__name__,
                        cause,
                    )
                    if not _is_known_network_error:
                        log.debug(
                            "MCP: full traceback for server '%s' connect failure",
                            cfg.name,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                    break

                if attempt < _STARTUP_MAX_RETRIES:
                    delay = _STARTUP_RETRY_DELAYS[attempt]
                    log.warning(
                        "MCP: failed to connect to server '%s': %s: %s "
                        "— retrying in %.0fs (attempt %d/%d)",
                        cfg.name,
                        type(cause).__name__,
                        cause,
                        delay,
                        attempt + 1,
                        _STARTUP_MAX_RETRIES + 1,
                    )
                    await asyncio.sleep(delay)
                else:
                    # On final failure: one-line warning. Skip the
                    # multi-frame traceback for known network error classes
                    # (same rationale as the fail_fast branch above:
                    # ~125 lines of library plumbing per failure, zero
                    # Cogtrix-side signal). Unexpected exception classes
                    # still get the full traceback.
                    log.warning(
                        "MCP: server '%s' unavailable after %d attempts: %s: %s",
                        cfg.name,
                        _STARTUP_MAX_RETRIES + 1,
                        type(cause).__name__,
                        cause,
                    )
                    if not _is_known_network_error:
                        log.debug(
                            "MCP: full traceback for server '%s' final failure",
                            cfg.name,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )

        return cfg.name, None

    async def _connect_all_async(
        self, configs: list[MCPServerConfig]
    ) -> list[tuple[str, MCPConnection | None]]:
        """Connect to all servers concurrently and return per-server results."""
        raw = await asyncio.gather(
            *[self._connect_one_async(cfg) for cfg in configs],
            return_exceptions=True,
        )
        results: list[tuple[str, MCPConnection | None]] = []
        for cfg, outcome in zip(configs, raw, strict=True):
            if isinstance(outcome, BaseException):
                get_logger().warning(
                    "MCP: unexpected error connecting to server '%s': %s",
                    cfg.name,
                    outcome,
                    exc_info=outcome,  # preserve full traceback
                )
                results.append((cfg.name, None))
            else:
                results.append(outcome)
        return results

    def connect_all(
        self,
        configs: list[MCPServerConfig],
        builtin_tool_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Connect to all configured MCP servers and return LangChain tools.

        Servers that fail to connect are skipped with a warning — they do not
        prevent other servers from being used. All servers are connected
        concurrently to minimise startup latency.

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
            self.tools_ready.set()
            return {}

        log = get_logger()
        self._ensure_loop()
        self.tools_ready.clear()

        for cfg in configs:
            self._configs[cfg.name] = cfg

        max_timeout = max((cfg.timeout for cfg in configs), default=30) if configs else 30
        try:
            results: list[tuple[str, MCPConnection | None]] = self._run(
                self._connect_all_async(configs),
                timeout=max_timeout + 5,
            )
        except Exception:
            # Unblock any waiters even when connection fails, so the
            # orchestration graph doesn't hang indefinitely.
            self.tools_ready.set()
            raise

        for server_name, conn in results:
            if conn is not None:
                self._connections[server_name] = conn
                log.info("MCP: connected to server '%s' (%d tools)", server_name, len(conn.tools))

        # Names that MCP servers must never shadow — they are core platform
        # primitives that could be exploited if an untrusted server overwrites them.
        _RESERVED_TOOL_NAMES: frozenset[str] = frozenset(
            {"request_tools", "checkpoint", "query_knowledge_base"}
        )

        all_tools: dict[str, Any] = {}
        for cfg in configs:
            conn = self._connections.get(cfg.name)
            if conn is None:
                continue

            for mcp_tool in conn.tools:
                original_name: str = mcp_tool.name
                tool_name = original_name

                if tool_name in _RESERVED_TOOL_NAMES:
                    log.warning(
                        "MCP: tool '%s' from server '%s' uses a reserved name and will be skipped",
                        tool_name,
                        cfg.name,
                    )
                    continue

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

                if tool_name != original_name:
                    lc_tool.name = tool_name
                    loaded_as_note = f"(loaded as {tool_name})"
                    if loaded_as_note not in lc_tool.description:
                        lc_tool.description = (
                            f"{lc_tool.description.rstrip()} {loaded_as_note}".strip()
                        )

                all_tools[tool_name] = lc_tool
                self._tool_server_map[tool_name] = cfg.name

        # Signal readiness AFTER wrappers and _tool_server_map are fully built.
        # Do NOT move this earlier — the orchestration graph unblocks on this
        # event and immediately starts binding tools.
        self.tools_ready.set()
        return all_tools

    def _reconnect_server(self, server_name: str) -> None:
        """Force-reconnect a single server without rebuilding LangChain tools.

        Delegates to ``_reconnect_server_async`` so that the per-server lock
        is shared with the heartbeat coroutine, eliminating the race that
        leaked ``_exit_stack`` file-descriptors (#427).
        """
        self.tools_ready.clear()
        try:
            self._run(self._reconnect_server_async(server_name), timeout=60)
        finally:
            self.tools_ready.set()

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
            call_coro = conn.call_tool(mcp_tool_name, arguments)
            try:
                return self._run(call_coro, timeout=timeout)
            except TimeoutError:
                # _run() raised before consuming the coroutine — prevent
                # "RuntimeWarning: coroutine was never awaited" at GC.
                call_coro.close()
                return f"Error: MCP tool '{mcp_tool_name}' timed out after {timeout}s"
            except Exception:
                call_coro.close()
                raise
        except Exception as exc:
            log.error(
                "MCP tool call '%s' on server '%s' failed: %s",
                mcp_tool_name,
                server_name,
                sanitize_error(exc),
                exc_info=True,
            )
            # Auto-reconnect on connection-layer failures and retry once.
            if _is_connection_error(exc):
                log.info(
                    "MCP: connection error on server '%s' — attempting auto-reconnect",
                    server_name,
                )
                try:
                    self._reconnect_server(server_name)
                    # Refresh conn reference after reconnect.
                    new_conn = self._connections.get(server_name)
                    if new_conn is not None:
                        retry_coro = new_conn.call_tool(mcp_tool_name, arguments)
                        try:
                            return self._run(retry_coro, timeout=timeout)
                        except Exception:
                            retry_coro.close()
                            raise
                except Exception as reconnect_exc:
                    log.error(
                        "MCP: auto-reconnect for server '%s' failed: %s",
                        server_name,
                        reconnect_exc,
                    )
            return f"Error: MCP tool call failed: {sanitize_error(exc)}"

    def close_all(self) -> None:
        """Close all MCP connections and stop the background event loop."""
        import io
        import sys

        log = get_logger()
        # Set shutdown state under the loop lock BEFORE any teardown so a
        # concurrent _ensure_loop() cannot race in and create a zombie loop.
        with self._loop_lock:
            self._shutting_down = True
        # The MCP SSE library prints "Error in post_writer" + a full traceback
        # to sys.stderr when its background task hits a shutdown race.  Capture
        # that output and redirect it to the debug log so users never see it.
        _stderr_buf = io.StringIO()
        sys.stderr = _stderr_buf
        try:
            self._close_all_inner(log)
        finally:
            # Do NOT restore sys.stderr here. The post_writer background task
            # fires async DNS resolution after _close_all_inner() returns, so
            # restoring stderr before that task completes lets the traceback
            # escape to the terminal. The process is always shutting down at
            # this call site, so leaving stderr captured is safe.
            _captured = _stderr_buf.getvalue().strip()
            if _captured:
                log.debug("MCP: suppressed shutdown stderr output:\n%s", _captured)

    def _close_all_inner(self, log: Any) -> None:
        # Cancel the heartbeat task before closing connections so it doesn't
        # attempt reconnects on connections we are intentionally closing.
        self.tools_ready.clear()
        task = self._heartbeat_task
        if task is not None and not task.done() and self._loop is not None:

            async def _cancel_heartbeat() -> None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            try:
                asyncio.run_coroutine_threadsafe(_cancel_heartbeat(), self._loop).result(timeout=5)
            except Exception as exc:
                log.debug("MCP: error cancelling heartbeat task: %s", exc)
        self._heartbeat_task = None

        if self._loop is not None and not self._loop.is_closed():
            # Cancel all pending tasks before stopping so that background
            # coroutines like the MCP SSE post_writer do not attempt network
            # I/O after the loop is stopped (→ RuntimeError: cannot schedule
            # new futures after shutdown).
            async def _cancel_all() -> None:
                # Close connection transports first while the loop is still alive.
                # This prevents SSE post_writer cleanup from attempting DNS/executor
                # work after loop shutdown (#504).
                for name, conn in list(self._connections.items()):
                    exit_stack = getattr(conn, "_exit_stack", None)
                    if exit_stack is None:
                        continue
                    closed_here = False
                    try:
                        await exit_stack.aclose()
                        closed_here = True
                    except RuntimeError as exc:
                        if "cancel scope" in str(exc):
                            log.debug(
                                "MCP: connection '%s' pre-close hit anyio task boundary: %s",
                                name,
                                exc,
                            )
                        else:
                            log.warning("MCP: error pre-closing connection '%s': %s", name, exc)
                    except Exception as exc:
                        log.warning("MCP: error pre-closing connection '%s': %s", name, exc)
                    finally:
                        if closed_here:
                            conn._exit_stack = None
                            conn._session = None

                # Exclude ourselves to avoid self-cancellation.
                #
                # Python 3.13 changed Task.cancel() to propagate recursively
                # into nested child tasks (_GatheringFuture).  With 3+ MCP SSE
                # servers the anyio/httpx/sse_client task hierarchy easily
                # exceeds 1000 frames.  The RecursionError fires in an asyncio
                # event-loop callback (_asyncio.TaskStepMethWrapper), NOT in
                # this coroutine, so try/except alone cannot catch it.
                #
                # Fix: (1) temporarily raise sys.setrecursionlimit so the chain
                # completes; (2) pass msg=None to Task.cancel() which skips the
                # recursive child-propagation path in Python 3.13.
                import sys

                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
                if not tasks:
                    return
                old_limit = sys.getrecursionlimit()
                sys.setrecursionlimit(max(old_limit, 10_000))
                try:
                    for t in tasks:
                        try:
                            t.cancel(msg=None)  # msg=None skips child propagation in 3.13
                        except RecursionError:
                            pass
                    try:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    except RecursionError:
                        pass
                finally:
                    sys.setrecursionlimit(old_limit)

            try:
                future = asyncio.run_coroutine_threadsafe(_cancel_all(), self._loop)
                future.result(timeout=5)
            except Exception as exc:
                log.debug("MCP: error cancelling pending tasks during shutdown: %s", exc)

            # Close connections AFTER background tasks are cancelled/awaited so
            # the SSE post_writer task does not write to an already-closed connection.
            for name, conn in list(self._connections.items()):
                if getattr(conn, "_exit_stack", None) is None:
                    continue
                try:
                    future = asyncio.run_coroutine_threadsafe(conn.close(), self._loop)
                    future.result(timeout=10)
                except Exception as exc:
                    log.warning("MCP: error closing connection '%s': %s", name, exc)
            self._connections.clear()
            self._tool_server_map.clear()

            # Cancel and drain all in-flight _run() futures before stopping
            # the loop.  Each future wraps a coroutine scheduled via
            # run_coroutine_threadsafe; the asyncio Tasks they back were
            # already cancelled above, but cancelling the concurrent
            # Future ensures result() unblocks.  We drain with a short
            # timeout so any surviving futures resolve before the loop
            # stops, preventing "RuntimeWarning: coroutine was never
            # awaited" at GC.
            with self._loop_lock:
                pending = self._pending_futures.copy()
                self._pending_futures.clear()
            for f in pending:
                f.cancel()
            for f in pending:
                try:
                    f.result(timeout=2)
                except Exception:
                    pass

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                get_logger().warning(
                    "MCP background thread did not stop within 5s — "
                    "in-flight MCP calls may be dangling. Proceeding anyway."
                )
            self._thread = None

        # Close the loop after the thread has stopped to release fd resources.
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception as exc:
                get_logger().debug("MCP: error closing event loop: %s", exc)
        self._loop = None

    def restart(
        self,
        server_name: str | None = None,
        builtin_tool_names: set[str] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """
        Reconnect one or all MCP servers and return rebuilt LangChain tools.

        Args:
            server_name: If given, reconnect only this server. Otherwise restart all.
            builtin_tool_names: Optional set of built-in (non-MCP) tool names to
                include in collision detection when rebuilding tools.
            timeout: Overall timeout in seconds for the entire restart operation.

        Returns:
            Dict mapping tool name to LangChain StructuredTool for every tool
            available on the restarted servers.
        """
        log = get_logger()

        # Ensure the background loop exists — restart can be invoked from the
        # CLI/API at any time, including after a transient teardown. Fail
        # cleanly (return {}) if the manager is shutting down (#2152).
        try:
            self._ensure_loop()
        except RuntimeError as exc:
            log.warning("MCP: cannot restart — event loop unavailable: %s", exc)
            return {}

        if server_name is not None:
            targets = [server_name] if server_name in self._configs else []
        else:
            targets = list(self._configs.keys())

        # The inner per-server connects are bounded by each cfg.timeout
        # (default 30s). The overall budget must cover them (+ a close
        # allowance) or the outer wait_for would cancel mid-connect and the
        # restart would spuriously report a timeout. Honour a larger explicit
        # caller timeout (#2152).
        overall_timeout = timeout
        if targets:
            overall_timeout = max(
                timeout,
                sum(self._configs[t].timeout for t in targets) + 10.0 * len(targets),
            )

        async def _do_restart() -> dict[str, Any]:
            """Inner async function — runs ON the MCP event loop, so it awaits
            connect/close directly. Routing them through ``self._run()`` (which
            blocks on ``future.result()``) would deadlock the loop thread
            because the awaited coroutine needs the same loop to progress
            (#2152)."""
            if server_name is not None and not targets:
                log.warning("MCP: cannot restart unknown server '%s'", server_name)
                return {}

            self.tools_ready.clear()
            for name in targets:
                # Purge stale tool-server mappings for this server
                for key in [k for k, v in self._tool_server_map.items() if v == name]:
                    del self._tool_server_map[key]
                old_conn = self._connections.pop(name, None)
                if old_conn is not None:
                    try:
                        await asyncio.wait_for(old_conn.close(), timeout=10)
                    except Exception as exc:
                        log.warning("MCP: error closing '%s' during restart: %s", name, exc)

                cfg = self._configs[name]
                new_conn = MCPConnection(cfg)
                try:
                    await asyncio.wait_for(new_conn.connect(), timeout=cfg.timeout)
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
            self.tools_ready.set()
            return new_tools

        # Run with overall timeout to prevent hangs
        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(_do_restart(), timeout=overall_timeout), self._loop
            )
            return future.result(timeout=overall_timeout + 1.0)
        except (TimeoutError, concurrent.futures.TimeoutError):
            log.error("MCP: restart operation timed out after %.1f seconds", overall_timeout)
            return {}

    def disconnect(self, server_name: str, timeout: float = 15.0) -> bool:
        """Disconnect and fully remove a single MCP server at runtime.

        Closes the live connection (stdio subprocess / SSE socket) and purges
        the server from ``_connections``, ``_configs`` and ``_tool_server_map``
        so its tools can no longer be resolved or called. Returns True if the
        server had a live connection that was closed, False if it was unknown
        or already disconnected.

        Deadlock-safe like ``restart()`` (#2152): the inner coroutine runs ON
        the MCP event loop and ``await``s ``conn.close()`` directly rather than
        routing through ``self._run()`` (which blocks on ``future.result()`` and
        would wedge the loop thread).
        """
        log = get_logger()

        # Nothing known about this server — purge any stray mapping and bail
        # without forcing a background loop into existence.
        if server_name not in self._connections and server_name not in self._configs:
            for key in [k for k, v in self._tool_server_map.items() if v == server_name]:
                del self._tool_server_map[key]
            return False

        try:
            self._ensure_loop()
        except RuntimeError as exc:
            # Shutting down / no loop: still purge in-memory state so the
            # manager reflects the caller's intent.
            log.warning(
                "MCP: cannot cleanly disconnect '%s' — event loop unavailable: %s",
                server_name,
                exc,
            )
            was_connected = self._connections.pop(server_name, None) is not None
            self._configs.pop(server_name, None)
            for key in [k for k, v in self._tool_server_map.items() if v == server_name]:
                del self._tool_server_map[key]
            return was_connected

        async def _do_disconnect() -> bool:
            """Runs ON the MCP event loop — awaits close() directly (#2152)."""
            for key in [k for k, v in self._tool_server_map.items() if v == server_name]:
                del self._tool_server_map[key]
            conn = self._connections.pop(server_name, None)
            self._configs.pop(server_name, None)
            if conn is None:
                return False
            try:
                await asyncio.wait_for(conn.close(), timeout=10)
            except Exception as exc:
                log.warning("MCP: error closing '%s' during disconnect: %s", server_name, exc)
            return True

        try:
            future = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(_do_disconnect(), timeout=timeout), self._loop
            )
            return future.result(timeout=timeout + 1.0)
        except (TimeoutError, concurrent.futures.TimeoutError):
            log.error("MCP: disconnect of '%s' timed out after %.1f seconds", server_name, timeout)
            return False

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
    "DOC_ONLY_MCP_FIELDS",
    "KNOWN_MCP_FIELDS",
    "MCPConnection",
    "MCPManager",
    "MCPServerConfig",
    "MCP_AVAILABLE",
    "json_schema_to_pydantic",
]
