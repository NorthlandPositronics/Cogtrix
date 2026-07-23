"""Tool management endpoints.

Tools are capabilities the agent can invoke.  They are either active
(in the session's active tool set), on-demand (loadable via request_tools),
or disabled (blocked for the session).

Endpoints:
    GET    /api/v1/tools                        — list all available tools in the registry
    GET    /api/v1/tools/{name}                 — get tool details including parameter schema
    GET    /api/v1/sessions/{id}/tools          — get tool status for a specific session
    PATCH  /api/v1/sessions/{id}/tools          — load/unload/enable/disable/approve tools
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, verify_session_owner
from src.api.db.engine import get_db
from src.api.pagination import decode_cursor, encode_cursor
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.tool import ToolActionRequest, ToolOut, ToolParameterSchema, ToolSummary
from src.api.session_bridge import _API_DENIED_DANGEROUS_TOOLS

log = logging.getLogger("cogtrix.api.tools")

router = APIRouter(tags=["Tools"])


def _get_registry(request: Request) -> Any:
    return getattr(request.app.state, "tool_registry", None)


def _get_session_registry(request: Request) -> Any:
    return getattr(request.app.state, "session_registry", None)


def _extract_short_description(description: str) -> str:
    return (description.split("\n")[0] if description else "")[:120]


def _extract_parameters(tool: Any) -> list[ToolParameterSchema]:
    """Extract parameter schema from a StructuredTool's args_schema."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return []
    try:
        schema = args_schema.model_json_schema()
        props = schema.get("properties", {})
        required_fields = set(schema.get("required", []))
        params: list[ToolParameterSchema] = []
        for param_name, param_schema in props.items():
            param_type = param_schema.get("type", "string")
            if isinstance(param_type, list):
                # nullable: ["string", "null"] -> "string"
                param_type = next((t for t in param_type if t != "null"), "string")
            params.append(
                ToolParameterSchema(
                    name=param_name,
                    type=param_type,
                    description=param_schema.get("description", ""),
                    required=param_name in required_fields,
                    default=param_schema.get("default", None),
                    enum=param_schema.get("enum", None),
                )
            )
        return params
    except Exception as exc:
        log.debug("Could not extract parameters from tool schema: %s", exc)
        return []


def _tool_to_summary(
    name: str,
    tool: Any,
    registry: Any,
    tool_status: str = "on_demand",
) -> ToolSummary:
    desc = getattr(tool, "description", "") or ""
    return ToolSummary(
        name=name,
        short_description=_extract_short_description(desc),
        status=tool_status,  # type: ignore[arg-type]
        requires_confirmation=registry.requires_confirmation(name),
        is_mcp=registry.is_mcp_tool(name),
    )


def _tool_to_out(
    name: str,
    tool: Any,
    registry: Any,
    tool_status: str = "on_demand",
) -> ToolOut:
    desc = getattr(tool, "description", "") or ""
    is_mcp = registry.is_mcp_tool(name)
    mcp_server = registry.get_tool_server(name)
    module: str | None = None
    if is_mcp:
        module = mcp_server
    return ToolOut(
        name=name,
        description=desc,
        short_description=_extract_short_description(desc),
        status=tool_status,  # type: ignore[arg-type]
        requires_confirmation=registry.requires_confirmation(name),
        parameters=_extract_parameters(tool),
        module=module,
        is_mcp=is_mcp,
        mcp_server=mcp_server,
    )


def _classify_tool_status(name: str, session_state: Any) -> str:
    if session_state.is_denied(name):
        return "disabled"
    if name in getattr(session_state, "pinned_tools", set()):
        return "pinned"
    # Only report "auto_approved" when the tool is actually loaded —
    # an approval on an on-demand tool just means it won't need confirmation
    # when eventually expanded, not that it's active.
    if name in session_state.get_approvals_snapshot() and name in session_state.loaded_tools:
        return "auto_approved"
    if name in session_state.loaded_tools:
        return "active"
    return "on_demand"


@router.get(
    "/tools",
    summary="List all available tools",
    description=(
        "List every tool registered in the global tool registry — both built-in tools "
        "and any tools discovered from connected MCP servers. "
        "Status tags reflect the default state (not session-specific). "
        "Use GET /sessions/{id}/tools to see per-session status."
    ),
    response_model=APIResponse[CursorPage[ToolSummary]],
    responses={
        200: {"description": "Tool list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_tools(
    request: Request,
    cursor: str | None = None,
    limit: int = 100,
    search: str | None = Query(
        default=None, description="Filter by tool name or description (case-insensitive substring)."
    ),
    include_mcp: bool = Query(default=True, description="Include tools from MCP servers."),
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[ToolSummary]]:
    """List all tools in the global registry.

    Query parameters:
        cursor      — pagination cursor.
        limit       — page size (1–500, default 100).
        search      — case-insensitive substring filter on name and description.
        include_mcp — include MCP server tools (default true).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    registry = _get_registry(request)
    if registry is None:
        empty: CursorPage[ToolSummary] = CursorPage(
            items=[], next_cursor=None, has_more=False, total=0
        )
        return APIResponse(data=empty)

    all_tools = list((registry.tools or {}).items())

    # Filter
    if not include_mcp:
        all_tools = [(n, t) for n, t in all_tools if not registry.is_mcp_tool(n)]
    if search:
        q = search.lower()
        all_tools = [
            (n, t)
            for n, t in all_tools
            if q in n.lower() or q in (getattr(t, "description", "") or "").lower()
        ]

    # Decode cursor
    raw_cursor: str | None = None
    if cursor:
        try:
            raw_cursor = decode_cursor(cursor)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_CURSOR",
                    "message": "The pagination cursor is malformed.",
                },
            ) from exc

    # Paginate
    limit = max(1, min(limit, 500))
    start = 0
    if raw_cursor:
        for i, (n, _) in enumerate(all_tools):
            if n == raw_cursor:
                start = i + 1
                break
    page_pairs = all_tools[start : start + limit]
    has_more = (start + limit) < len(all_tools)
    next_cursor = encode_cursor(page_pairs[-1][0]) if has_more and page_pairs else None

    items = [_tool_to_summary(n, t, registry, "on_demand") for n, t in page_pairs]
    page: CursorPage[ToolSummary] = CursorPage(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=len(all_tools),
    )
    return APIResponse(data=page)


@router.get(
    "/tools/{tool_name}",
    summary="Get tool details",
    description="Return full details for a single tool including its parameter schema.",
    response_model=APIResponse[ToolOut],
    responses={
        200: {"description": "Tool details returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Tool not found (TOOL_NOT_FOUND)."},
    },
)
async def get_tool(
    tool_name: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[ToolOut]:
    """Return full details and parameter schema for a single tool.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TOOL_NOT_FOUND.
    """
    registry = _get_registry(request)
    if registry is None or tool_name not in (registry.tools or {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TOOL_NOT_FOUND",
                "message": "The requested tool does not exist in the registry.",
            },
        )
    tool = registry.tools[tool_name]
    return APIResponse(data=_tool_to_out(tool_name, tool, registry, "on_demand"))


@router.get(
    "/sessions/{session_id}/tools",
    summary="Get session tool status",
    description=(
        "List all tools with their status relative to the given session: "
        "active (loaded in agent's tool set), on_demand (loadable), "
        "disabled (blocked), or auto_approved (active + no confirmation)."
    ),
    response_model=APIResponse[list[ToolSummary]],
    responses={
        200: {"description": "Session tool status returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def get_session_tools(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[ToolSummary]]:
    """List all tools with their current status for a specific session.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    await verify_session_owner(session_id, current_user, db, admin_bypass=True)

    registry = _get_registry(request)
    session_registry = _get_session_registry(request)
    if registry is None:
        return APIResponse(data=[])

    live_session = None
    if session_registry is not None:
        live_session = await session_registry.get_or_warm(session_id, db)
    if live_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    ss = live_session.session_state
    items = [
        _tool_to_summary(n, t, registry, _classify_tool_status(n, ss))
        for n, t in (registry.tools or {}).items()
    ]
    return APIResponse(data=items)


@router.patch(
    "/sessions/{session_id}/tools",
    summary="Manage session tool state",
    description=(
        "Load, unload, enable, disable, or auto-approve tools for a session. "
        "Supply only the action fields you want to apply; the rest are ignored. "
        "Changes take effect immediately on the next agent tool call. "
        "Loading an on-demand tool moves it into the active set (equivalent to the agent "
        "calling request_tools). Disabling prevents the agent from loading the tool "
        "even via request_tools."
    ),
    response_model=APIResponse[list[ToolSummary]],
    responses={
        200: {"description": "Tool state updated; updated tool list returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session or tool not found (SESSION_NOT_FOUND, TOOL_NOT_FOUND)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def patch_session_tools(
    session_id: str,
    body: ToolActionRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[ToolSummary]]:
    """Apply tool state changes to a session.

    Auth: bearer token required.
    Error codes:
        UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND,
        TOOL_NOT_FOUND, TOOL_ALREADY_ACTIVE, TOOL_ALREADY_DISABLED,
        TOOL_EXPANSION_FAILED.
    """
    await verify_session_owner(session_id, current_user, db, admin_bypass=True)

    registry = _get_registry(request)
    session_registry = _get_session_registry(request)
    if registry is None:
        return APIResponse(data=[])

    live_session = None
    if session_registry is not None:
        live_session = await session_registry.get_or_warm(session_id, db)
    if live_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    ss = live_session.session_state
    all_tool_names = set((registry.tools or {}).keys())

    def _assert_tool_exists(name: str) -> None:
        if name not in all_tool_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "TOOL_NOT_FOUND",
                    "message": f"Tool '{name}' does not exist in the registry.",
                },
            )

    # #2070/#2050: while api_dangerous_tools is disabled, the tools route must
    # not re-open the denied exec tools — `load` would activate them and
    # `enable` (allow_tool) would delete the warm-time denial. Reject up-front so
    # no partial state is applied. (DedupedToolInvoker also enforces is_denied at
    # execution as a backstop.)
    app_config = getattr(request.app.state, "config", None)
    if not getattr(app_config, "api_dangerous_tools", False):
        _requested = set(body.load or []) | set(body.enable or [])
        _blocked = _requested & set(_API_DENIED_DANGEROUS_TOOLS)
        # #2116: also reject confirmation-gated tools — on no_confirm API
        # sessions they have no confirmation path, so loading/enabling one would
        # let it execute unguarded. Mirrors the warm-time deny in session_bridge.
        _tool_registry = getattr(request.app.state, "tool_registry", None)
        if _tool_registry is not None:
            for _name in _requested:
                try:
                    if _tool_registry.requires_confirmation(_name):
                        _blocked.add(_name)
                except Exception:  # noqa: BLE001 — per-tool lookup must not 500 the route
                    continue
        if _blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "FORBIDDEN",
                    "message": (
                        "Cannot load or enable confirmation-gated/dangerous tools ("
                        + ", ".join(sorted(_blocked))
                        + ") while api_dangerous_tools is disabled."
                    ),
                },
            )

    # Acquire turn_lock before mutating run_config to prevent racing with an
    # in-flight agent turn that reads active_tools_list / available_tools
    # (BUG-196 — consistent with patch_session in sessions.py).
    async with live_session.turn_lock:
        rc = getattr(live_session, "run_config", None)

        if body.load:
            for name in body.load:
                _assert_tool_exists(name)
                ss.loaded_tools.add(name)
                ss.pinned_tools.add(name)
                # Move tool from available to active in run_config so the LLM
                # sees it in its bound schema on the next turn.
                if rc is not None:
                    avail = getattr(rc, "available_tools", None) or {}
                    if name in avail:
                        tool_obj = avail.pop(name)
                        atl = getattr(rc, "active_tools_list", None)
                        if atl is not None:
                            atl.append(tool_obj)

        if body.unload:
            for name in body.unload:
                ss.loaded_tools.discard(name)
                ss.pinned_tools.discard(name)
                # Return tool from active back to available in run_config.
                if rc is not None:
                    atl = getattr(rc, "active_tools_list", None) or []
                    avail = getattr(rc, "available_tools", None)
                    for i, t in enumerate(atl):
                        if getattr(t, "name", None) == name:
                            atl.pop(i)
                            if avail is not None:
                                orig = ss.all_tool_originals.get(name, t)
                                avail[name] = orig
                            break

        if body.enable:
            for name in body.enable:
                ss.allow_tool(name)

        if body.disable:
            for name in body.disable:
                _assert_tool_exists(name)
                ss.deny_tool(name)
                ss.loaded_tools.discard(name)
                ss.pinned_tools.discard(name)
                # Also remove from run_config active tools so the LLM can't invoke it.
                if rc is not None:
                    atl = getattr(rc, "active_tools_list", None) or []
                    avail = getattr(rc, "available_tools", None)
                    for i, t in enumerate(atl):
                        if getattr(t, "name", None) == name:
                            atl.pop(i)
                            if avail is not None:
                                orig = ss.all_tool_originals.get(name, t)
                                avail[name] = orig
                            break

        if body.auto_approve:
            for name in body.auto_approve:
                _assert_tool_exists(name)
                ss.add_approval(name)

        if body.revoke_approval:
            for name in body.revoke_approval:
                ss.revoke_approval(name)

    items = [
        _tool_to_summary(n, t, registry, _classify_tool_status(n, ss))
        for n, t in (registry.tools or {}).items()
    ]
    return APIResponse(data=items)
