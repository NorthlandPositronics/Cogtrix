"""MCP (Model Context Protocol) server management endpoints.

Endpoints:
    GET    /api/v1/mcp/servers             — list configured MCP servers
    POST   /api/v1/mcp/servers             — add a new MCP server config (admin)
    GET    /api/v1/mcp/servers/{name}      — get a single server's details
    DELETE /api/v1/mcp/servers/{name}      — remove an MCP server config (admin)
    POST   /api/v1/mcp/servers/{name}/restart — restart a server connection (admin)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.schemas.common import APIResponse
from src.api.schemas.mcp import MCPServerAddRequest, MCPServerOut, MCPToolSummary

log = logging.getLogger("cogtrix.api.mcp")

router = APIRouter(prefix="/mcp", tags=["MCP Servers"])

# Keys written into the config file per server entry.
# 'transport' is NOT stored — it is inferred from 'url' (sse) vs 'command' (stdio).
_KNOWN_MCP_CONFIG_KEYS = frozenset(
    {"command", "args", "env", "url", "headers", "requires_confirmation", "timeout"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_config(request: Request) -> Any:
    return getattr(request.app.state, "config", None)


def _get_mcp_client(request: Request) -> Any:
    # The lifespan startup stores the MCPManager on ``app.state.mcp_manager``
    # (src/api/app.py); ``app.state.mcp_client`` was never set in production, so
    # reading it made every /mcp route operate on None (#2151).
    return getattr(request.app.state, "mcp_manager", None)


def _get_mcp_servers(cfg: Any) -> dict[str, dict[str, Any]]:
    """Return the mcp_servers dict from config, or an empty dict."""
    if cfg is None:
        return {}
    return dict(getattr(cfg, "mcp_servers", {}) or {})


def _runtime_info_map(mcp_client: Any) -> dict[str, dict[str, Any]]:
    """Build a name→info dict from MCPManager.get_server_info().

    Returns an empty dict on any failure so callers always get a safe value.
    """
    if mcp_client is None:
        return {}
    try:
        entries = mcp_client.get_server_info()
        return {e["name"]: e for e in entries}
    except Exception:
        return {}


def _config_entry_to_out(
    name: str,
    entry: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> MCPServerOut:
    """Convert a config dict entry (and optional runtime info) to MCPServerOut."""
    url = entry.get("url")
    command = entry.get("command")
    transport: str = "sse" if url else "stdio"

    if runtime is not None:
        connected = bool(runtime.get("connected", False))
        srv_status = "connected" if connected else "disconnected"
        tool_names: list[str] = list(runtime.get("tools") or [])
        tools = [MCPToolSummary(name=t, description="") for t in tool_names]
        error: str | None = runtime.get("error") or None
    else:
        srv_status = "disconnected"
        tools = []
        error = None

    return MCPServerOut(
        name=name,
        status=srv_status,  # type: ignore[arg-type]
        transport=transport,  # type: ignore[arg-type]
        url=url,
        command=command,
        args=list(entry.get("args") or []),
        requires_confirmation=bool(entry.get("requires_confirmation", True)),
        tools=tools,
        error=error,
        connected_at=None,
    )


def _request_to_config_entry(body: MCPServerAddRequest) -> dict[str, Any]:
    """Convert an MCPServerAddRequest to a config dict entry (no 'transport' key)."""
    entry: dict[str, Any] = {"requires_confirmation": body.requires_confirmation}
    if body.url is not None:
        entry["url"] = body.url
    if body.command is not None:
        entry["command"] = body.command
    if body.args:
        entry["args"] = list(body.args)
    if body.env:
        entry["env"] = dict(body.env)
    if body.headers:
        entry["headers"] = dict(body.headers)
    if body.timeout != 30:
        entry["timeout"] = body.timeout
    return entry


def _persist_mcp_servers(cfg: Any) -> None:
    """Atomically write cfg.mcp_servers back to the YAML config file.

    Raises RuntimeError if no config file path is set or if I/O fails.
    """
    config_path_raw: Path | None = getattr(cfg, "config_file_path", None)
    if config_path_raw is None:
        raise RuntimeError("No config file path configured; cannot persist MCP server changes.")

    config_path = Path(config_path_raw)
    try:
        raw = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        data: dict[str, Any] = yaml.safe_load(raw) or {}
    except Exception as exc:
        raise RuntimeError(f"Failed to read config file: {exc}") from exc

    if not isinstance(data, dict):
        data = {}

    mcp_servers: dict[str, Any] = dict(getattr(cfg, "mcp_servers", {}) or {})
    if mcp_servers:
        data["mcp_servers"] = mcp_servers
    else:
        data.pop("mcp_servers", None)

    dir_path = config_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp", prefix=".cogtrix_mcp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        os.replace(tmp_path, str(config_path))
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Failed to write config file: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/servers",
    summary="List connected MCP servers",
    description=(
        "List all MCP servers defined in the config with their current connection status, "
        "discovered tools, and any connection error."
    ),
    response_model=APIResponse[list[MCPServerOut]],
    responses={
        200: {"description": "MCP server list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_mcp_servers(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[MCPServerOut]]:
    """List all configured MCP servers with runtime status.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    cfg = _get_config(request)
    servers_dict = _get_mcp_servers(cfg)
    mcp_client = _get_mcp_client(request)
    runtime = _runtime_info_map(mcp_client)
    result = [
        _config_entry_to_out(name, entry, runtime.get(name)) for name, entry in servers_dict.items()
    ]
    return APIResponse(data=result)


@router.post(
    "/servers",
    summary="Add a new MCP server",
    description=(
        "Add a new MCP server configuration and attempt to connect immediately. "
        "The server config is written to the active config file. "
        "Admin only."
    ),
    response_model=APIResponse[MCPServerOut],
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "MCP server added and connection attempted."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {"description": "Server name already exists (VALIDATION_ERROR)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
        503: {"description": "Config file not available or I/O error (SERVICE_UNAVAILABLE)."},
    },
)
async def add_mcp_server(
    body: MCPServerAddRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[MCPServerOut]:
    """Add and connect a new MCP server (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, VALIDATION_ERROR, SERVICE_UNAVAILABLE.
    """
    cfg = _get_config(request)
    servers_dict = _get_mcp_servers(cfg)

    if body.name in servers_dict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"MCP server '{body.name}' already exists.",
            },
        )

    entry = _request_to_config_entry(body)

    if cfg is not None:
        cfg.mcp_servers[body.name] = entry

    try:
        await asyncio.to_thread(_persist_mcp_servers, cfg)
    except RuntimeError as exc:
        # Roll back the in-memory change and surface a clear error.
        if cfg is not None:
            cfg.mcp_servers.pop(body.name, None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    mcp_client = _get_mcp_client(request)
    runtime = _runtime_info_map(mcp_client)
    return APIResponse(data=_config_entry_to_out(body.name, entry, runtime.get(body.name)))


@router.get(
    "/servers/{server_name}",
    summary="Get MCP server details",
    description="Return details and tool list for a single MCP server.",
    response_model=APIResponse[MCPServerOut],
    responses={
        200: {"description": "Server details returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Server not found (MCP_SERVER_NOT_FOUND)."},
    },
)
async def get_mcp_server(
    server_name: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[MCPServerOut]:
    """Return details and tool list for a single MCP server.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, MCP_SERVER_NOT_FOUND.
    """
    cfg = _get_config(request)
    servers_dict = _get_mcp_servers(cfg)

    if server_name not in servers_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )

    mcp_client = _get_mcp_client(request)
    runtime = _runtime_info_map(mcp_client)
    return APIResponse(
        data=_config_entry_to_out(server_name, servers_dict[server_name], runtime.get(server_name))
    )


@router.delete(
    "/servers/{server_name}",
    summary="Remove an MCP server",
    description=(
        "Disconnect and remove an MCP server configuration. "
        "Any tools from this server are immediately removed from all active sessions. "
        "Admin only."
    ),
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Server removed."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Server not found (MCP_SERVER_NOT_FOUND)."},
    },
)
async def remove_mcp_server(
    server_name: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> Response:
    """Disconnect and remove an MCP server (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, MCP_SERVER_NOT_FOUND.
    """
    cfg = _get_config(request)
    servers_dict = _get_mcp_servers(cfg)

    if server_name not in servers_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )

    entry = servers_dict.get(server_name)

    if cfg is not None:
        cfg.mcp_servers.pop(server_name, None)

    try:
        await asyncio.to_thread(_persist_mcp_servers, cfg)
    except RuntimeError as exc:
        # Roll back the in-memory change and surface a clear error.
        if entry is not None and cfg is not None:
            cfg.mcp_servers[server_name] = entry
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/servers/{server_name}/restart",
    summary="Restart an MCP server connection",
    description=(
        "Disconnect and reconnect an MCP server. "
        "Tools from this server are temporarily unavailable during the restart. "
        "Admin only. Equivalent to the /mcp restart CLI command."
    ),
    response_model=APIResponse[MCPServerOut],
    responses={
        200: {"description": "Server restarted; updated status returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Server not found (MCP_SERVER_NOT_FOUND)."},
        503: {"description": "Restart failed (MCP_RESTART_FAILED)."},
    },
)
async def restart_mcp_server(
    server_name: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[MCPServerOut]:
    """Restart an MCP server connection (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, MCP_SERVER_NOT_FOUND, MCP_RESTART_FAILED.
    """
    cfg = _get_config(request)
    servers_dict = _get_mcp_servers(cfg)

    if server_name not in servers_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )

    mcp_client = _get_mcp_client(request)
    restart_fn = getattr(mcp_client, "restart_server", None)
    if restart_fn is not None:
        try:
            await asyncio.to_thread(restart_fn, server_name)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "MCP_RESTART_FAILED",
                    "message": f"MCP server restart failed: {exc}",
                },
            ) from exc

    runtime = _runtime_info_map(mcp_client)
    return APIResponse(
        data=_config_entry_to_out(server_name, servers_dict[server_name], runtime.get(server_name))
    )
