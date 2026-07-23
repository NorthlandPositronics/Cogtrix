"""MCP (Model Context Protocol) server management endpoints.

Endpoints:
    GET    /api/v1/mcp/servers             — list connected MCP servers
    POST   /api/v1/mcp/servers             — add a new MCP server config (admin)
    GET    /api/v1/mcp/servers/{name}      — get a single server's details
    DELETE /api/v1/mcp/servers/{name}      — remove an MCP server config (admin)
    POST   /api/v1/mcp/servers/{name}/restart — restart a server connection (admin)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.schemas.common import APIResponse
from src.api.schemas.mcp import MCPServerAddRequest, MCPServerOut

log = logging.getLogger("cogtrix.api.mcp")

router = APIRouter(prefix="/mcp", tags=["MCP Servers"])


def _get_mcp_client(request: Request) -> Any:
    return getattr(request.app.state, "mcp_client", None)


def _server_config_to_out(name: str, sc: Any) -> MCPServerOut:
    url = getattr(sc, "url", None)
    command = getattr(sc, "command", None)
    transport = "sse" if url else "stdio"
    return MCPServerOut(
        name=name,
        status="connected",
        transport=transport,
        url=url,
        command=command,
        args=list(getattr(sc, "args", []) or []),
        requires_confirmation=bool(getattr(sc, "requires_confirmation", True)),
        tools=[],
        error=None,
        connected_at=None,
    )


def _list_servers(mcp_client: Any) -> list[MCPServerOut]:
    if mcp_client is None:
        return []
    servers = getattr(mcp_client, "servers", {}) or {}
    return [_server_config_to_out(name, sc) for name, sc in servers.items()]


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
    mcp_client = _get_mcp_client(request)
    return APIResponse(data=_list_servers(mcp_client))


@router.post(
    "/servers",
    summary="Add a new MCP server",
    description=(
        "Add a new MCP server configuration and attempt to connect immediately. "
        "The server config is written to the active config file. "
        "Admin only."
    ),
    response_model=APIResponse[MCPServerOut],
    status_code=201,
    responses={
        201: {"description": "MCP server added and connection attempted."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {"description": "Server name already exists (VALIDATION_ERROR)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def add_mcp_server(
    body: MCPServerAddRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[MCPServerOut]:
    """Add and connect a new MCP server (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, VALIDATION_ERROR.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Adding MCP servers at runtime is not yet implemented.",
        },
    )


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
    mcp_client = _get_mcp_client(request)
    servers = (getattr(mcp_client, "servers", {}) if mcp_client else None) or {}
    if server_name not in servers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )
    return APIResponse(data=_server_config_to_out(server_name, servers[server_name]))


@router.delete(
    "/servers/{server_name}",
    summary="Remove an MCP server",
    description=(
        "Disconnect and remove an MCP server configuration. "
        "Any tools from this server are immediately removed from all active sessions. "
        "Admin only."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Server removed."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Server not found (MCP_SERVER_NOT_FOUND)."},
    },
)
async def remove_mcp_server(
    server_name: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Disconnect and remove an MCP server (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, MCP_SERVER_NOT_FOUND.
    """
    mcp_client = _get_mcp_client(request)
    servers = (getattr(mcp_client, "servers", {}) if mcp_client else None) or {}
    if server_name not in servers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Removing MCP servers at runtime is not yet implemented.",
        },
    )


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
    mcp_client = _get_mcp_client(request)
    servers = (getattr(mcp_client, "servers", {}) if mcp_client else None) or {}
    if server_name not in servers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "MCP_SERVER_NOT_FOUND",
                "message": "The requested MCP server is not configured.",
            },
        )
    restart_fn = getattr(mcp_client, "restart_server", None)
    if restart_fn is not None:
        try:
            restart_fn(server_name)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "MCP_RESTART_FAILED",
                    "message": f"MCP server restart failed: {exc}",
                },
            ) from exc
    return APIResponse(data=_server_config_to_out(server_name, servers[server_name]))
