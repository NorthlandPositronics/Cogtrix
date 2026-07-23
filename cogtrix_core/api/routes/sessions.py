"""Session management endpoints.

A session groups conversation history, memory state, and tool configuration.
Each user can have multiple concurrent sessions.

Endpoints:
    POST   /api/v1/sessions              — create a new session
    GET    /api/v1/sessions              — list sessions (paginated, cursor-based)
    GET    /api/v1/sessions/{id}         — get session details and current state
    PATCH  /api/v1/sessions/{id}         — update session name or config
    DELETE /api/v1/sessions/{id}         — archive (or permanently delete) a session
    POST   /api/v1/sessions/{id}/restore — unarchive a session
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.auth import TokenData, get_current_user
from cogtrix_core.api.db.engine import get_db
from cogtrix_core.api.db.models import ApiSessionRecord
from cogtrix_core.api.db.repositories.sessions import SessionRepository
from cogtrix_core.api.plan_enforcement import maybe_require_api_call_capacity
from cogtrix_core.api.schemas.common import APIResponse, CursorPage
from cogtrix_core.api.schemas.session import (
    SessionConfig,
    SessionCreateRequest,
    SessionOut,
    SessionPatchRequest,
    TokenCounts,
)
from cogtrix_core.api.session_bridge import warm_session

log = logging.getLogger("cogtrix.api.sessions")

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_cursor(value: str) -> str:
    """Base64-encode an opaque cursor value."""
    return base64.urlsafe_b64encode(value.encode()).decode()


def _decode_cursor(cursor: str) -> str:
    """Decode a base64 cursor; raises HTTPException on malformed input."""
    try:
        return base64.urlsafe_b64decode(cursor.encode()).decode()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
        ) from exc


def _record_to_out(record: ApiSessionRecord, live_session: Any = None) -> SessionOut:
    """Map an ORM record (+ optional live in-memory session) to SessionOut."""
    # Config
    try:
        cfg_dict = json.loads(record.config_json) if record.config_json else {}
    except (json.JSONDecodeError, TypeError):
        cfg_dict = {}
    config = SessionConfig(**{k: v for k, v in cfg_dict.items() if k in SessionConfig.model_fields})

    # Token counts
    if live_session is not None:
        tc_dict = live_session.token_counts
    else:
        try:
            tc_dict = json.loads(record.token_counts_json) if record.token_counts_json else {}
        except (json.JSONDecodeError, TypeError):
            tc_dict = {}
    token_counts = TokenCounts(
        input_tokens=tc_dict.get("input_tokens", 0),
        output_tokens=tc_dict.get("output_tokens", 0),
        context_window=tc_dict.get("context_window", 0),
    )

    # Active tools
    if live_session is not None and live_session.session_state is not None:
        active_tools = sorted(live_session.session_state.loaded_tools)
    else:
        try:
            active_tools = json.loads(record.active_tools_json) if record.active_tools_json else []
        except (json.JSONDecodeError, TypeError):
            active_tools = []

    # Agent state
    agent_state = live_session.agent_state if live_session is not None else (record.state or "idle")

    return SessionOut(
        id=record.id,
        name=record.name,
        owner_id=record.user_id,
        state=agent_state,  # type: ignore[arg-type]
        config=config,
        token_counts=token_counts,
        active_tools=active_tools,
        created_at=_ensure_tz(record.created_at),
        updated_at=_ensure_tz(record.updated_at),
        archived_at=_ensure_tz(record.archived_at) if record.archived_at else None,
        workspace_id=record.workspace_id,
    )


def _ensure_tz(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware (UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _get_registry(request: Request) -> Any:
    """Return the ApiSessionRegistry from app state."""
    return getattr(request.app.state, "session_registry", None)


async def _check_session_access(
    session_id: str,
    current_user: TokenData,
    db: AsyncSession,
    *,
    admin_bypass: bool = True,
) -> ApiSessionRecord:
    """Fetch session and enforce ownership + workspace membership; returns the record on success.

    Admins may access any session when *admin_bypass* is ``True`` (default).
    Regular users may only access their own sessions, and must still be a
    member of the workspace the session belongs to (if any).

    Args:
        session_id: UUID v4 of the session to check.
        current_user: Decoded JWT claims from the request.
        db: The caller's database session (from ``Depends(get_db)``).
        admin_bypass: When ``True``, skip the ownership check for admin callers.

    Raises:
        HTTPException 404 SESSION_NOT_FOUND — session does not exist.
        HTTPException 403 FORBIDDEN — session belongs to a different user
            or caller is no longer a workspace member.
    """
    from sqlalchemy import select as _select

    result = await db.execute(_select(ApiSessionRecord).where(ApiSessionRecord.id == session_id))
    record = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    if not (admin_bypass and current_user.is_admin) and record.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
            },
        )

    # Workspace isolation: non-admins must still be a member of the session's workspace.
    if record.workspace_id is not None and not (admin_bypass and current_user.is_admin):
        from cogtrix_core.api.db.repositories.workspaces import WorkspaceRepository

        ws_repo = WorkspaceRepository(db)
        membership = await ws_repo.get_membership(record.workspace_id, current_user.user_id)
        if membership is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "NOT_A_MEMBER",
                    "message": "You are not a member of this workspace.",
                },
            )

    return record


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    summary="Create a new session",
    description=(
        "Create a new agent session. "
        "The session starts in the 'idle' state with default config values inherited "
        "from the application config unless overridden in the request body. "
        "After creation, connect to the session WebSocket to stream agent output."
    ),
    response_model=APIResponse[SessionOut],
    status_code=201,
    responses={
        201: {"description": "Session created."},
        401: {"description": "Not authenticated (UNAUTHORIZED or TOKEN_EXPIRED)."},
        409: {"description": "Duplicate session name (SESSION_NAME_DUPLICATE)."},
        422: {"description": "Request body validation failed (VALIDATION_ERROR)."},
    },
)
async def create_session(
    request: Request,
    body: SessionCreateRequest | None = Body(default=None),
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _plan: None = Depends(maybe_require_api_call_capacity),
) -> APIResponse[SessionOut]:
    """Create a new agent session owned by the current user.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, VALIDATION_ERROR.

    The request body is optional — POSTing with no body (or ``{}``) is
    equivalent to POSTing ``SessionCreateRequest()`` with all defaults
    populated (auto-generated name, default ``SessionConfig``, empty
    tool lists, no workspace). Frontends that just want "give me a new
    session" can call ``POST /sessions`` without serialising a body.
    See #1882.
    """
    # #1882: every field on SessionCreateRequest has a default — accept
    # missing body as "all defaults". Empty JSON ``{}`` already produced
    # this state; the prior signature was the only thing rejecting it.
    if body is None:
        body = SessionCreateRequest()
    repo = SessionRepository(db)

    # Enforce concurrent session quota
    app_config = getattr(request.app.state, "config", None)
    if app_config is not None:
        from cogtrix_core.api.quota import _quota_config_from_app_config, get_enforcer

        quota_cfg = _quota_config_from_app_config(app_config)
        if quota_cfg.max_concurrent_sessions is not None:
            count = await repo.count_by_user(current_user.user_id)
            get_enforcer(quota_cfg).check_concurrent_sessions(current_user.user_id, count)

    # Enforce session name uniqueness per user
    if await repo.name_exists_for_user(current_user.user_id, body.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "SESSION_NAME_DUPLICATE",
                "message": f"A session named '{body.name}' already exists.",
            },
        )

    config_dict = body.config.model_dump(exclude_none=True)
    config_json = json.dumps(config_dict)

    # If a workspace is requested, verify membership before creating the session.
    if body.workspace_id is not None:
        from cogtrix_core.api.db.repositories.workspaces import WorkspaceRepository

        ws_repo = WorkspaceRepository(db)
        ws = await ws_repo.get_by_id(body.workspace_id)
        if ws is None or not ws.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": "Workspace not found."},
            )
        if not current_user.is_admin:
            membership = await ws_repo.get_membership(body.workspace_id, current_user.user_id)
            if membership is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "code": "NOT_A_MEMBER",
                        "message": "You are not a member of this workspace.",
                    },
                )

    record = await repo.create(
        user_id=current_user.user_id,
        name=body.name,
        config_json=config_json,
        workspace_id=body.workspace_id,
    )
    await db.commit()
    await db.refresh(record)

    # Warm the session into the registry (best-effort — non-blocking)
    tool_registry = getattr(request.app.state, "tool_registry", None)
    registry = _get_registry(request)
    live_session = None
    if registry is not None:
        try:
            live_session = await warm_session(record, request.app.state)
            await registry.put(live_session)
        except Exception as exc:
            log.warning("Could not warm new session %s: %s", record.id, exc)

    # Apply initial_tools and auto_approve_tools from the request.
    # Uses the same mutation pattern as PATCH /sessions/{id}/tools (BUG-196 pattern).
    if live_session is not None and (body.initial_tools or body.auto_approve_tools):
        if tool_registry is not None:
            ss = live_session.session_state
            rc = getattr(live_session, "run_config", None)
            all_tool_names = set((getattr(tool_registry, "tools", None) or {}).keys())
            async with live_session.turn_lock:
                for name in body.initial_tools:
                    if name not in all_tool_names:
                        log.warning(
                            "create_session: initial_tools tool '%s' not found, skipping", name
                        )
                        continue
                    ss.loaded_tools.add(name)
                    ss.pinned_tools.add(name)
                    if rc is not None:
                        avail = getattr(rc, "available_tools", None) or {}
                        if name in avail:
                            tool_obj = avail.pop(name)
                            atl = getattr(rc, "active_tools_list", None)
                            if atl is not None:
                                # Apply safety wrapper if required (for audit trail even in no_confirm mode)
                                if tool_registry.requires_confirmation(name):
                                    from cogtrix_core.agent.safety import create_safe_tool_wrapper

                                    tool_obj = create_safe_tool_wrapper(
                                        tool_obj,
                                        name,
                                        tool_registry,
                                        set(),  # no pre-approved set for initial load
                                        session_state=ss,
                                        tool_trust=rc.tool_trust,
                                    )
                                atl.append(tool_obj)
                for name in body.auto_approve_tools:
                    ss.add_approval(name)

    return APIResponse(data=_record_to_out(record, live_session))


@router.get(
    "",
    summary="List sessions",
    description=(
        "List all sessions owned by the current user, ordered by last activity (newest first). "
        "Uses cursor-based pagination. Pass the returned next_cursor as the cursor query "
        "parameter to retrieve the next page."
    ),
    response_model=APIResponse[CursorPage[SessionOut]],
    responses={
        200: {"description": "Session list returned."},
        401: {"description": "Not authenticated."},
        400: {"description": "Invalid cursor (INVALID_CURSOR)."},
    },
)
async def list_sessions(
    request: Request,
    cursor: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    include_archived: bool = False,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[CursorPage[SessionOut]]:
    """List all sessions owned by the current user (paginated, newest first).

    Query parameters:
        cursor          — opaque pagination cursor from the previous response.
        limit           — page size (1–100, default 20).
        include_archived — when true, include sessions that have been deleted/archived.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    after_id = _decode_cursor(cursor) if cursor else None

    repo = SessionRepository(db)

    if current_user.is_admin:
        rows = await repo.list_all(
            after_id=after_id,
            limit=limit,
            include_archived=include_archived,
        )
    else:
        rows = await repo.list_by_user(
            current_user.user_id,
            after_id=after_id,
            limit=limit,
            include_archived=include_archived,
        )

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    registry = _get_registry(request)
    items = []
    for r in page_rows:
        live = (await registry.get_cached(r.id)) if registry else None
        items.append(_record_to_out(r, live))

    next_cursor = None
    if has_more and page_rows:
        next_cursor = _encode_cursor(page_rows[-1].id)

    page: CursorPage[SessionOut] = CursorPage(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=None,
    )
    return APIResponse(data=page)


@router.get(
    "/{session_id}",
    summary="Get session details",
    description=(
        "Return full details for a single session including current config, "
        "agent state, active tools, and cumulative token counts."
    ),
    response_model=APIResponse[SessionOut],
    responses={
        200: {"description": "Session details returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Session belongs to another user (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def get_session(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SessionOut]:
    """Return full details for a single session.

    Auth: bearer token required. Admins may access any session.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    record = await _check_session_access(session_id, current_user, db, admin_bypass=True)

    registry = _get_registry(request)
    live_session = (await registry.get_cached(session_id)) if registry else None

    return APIResponse(data=_record_to_out(record, live_session))


@router.patch(
    "/{session_id}",
    summary="Update session name or config",
    description=(
        "Partially update a session. All fields are optional. "
        "Config changes (model, provider, memory_mode) take effect on the next agent turn. "
        "Changing provider or model invalidates the LLM bind-tools cache."
    ),
    response_model=APIResponse[SessionOut],
    responses={
        200: {"description": "Session updated."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def patch_session(
    session_id: str,
    body: SessionPatchRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SessionOut]:
    """Partially update session name and/or configuration.

    Auth: bearer token required. Admins may update any session.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND,
                 VALIDATION_ERROR, MODEL_UNAVAILABLE, PROVIDER_UNREACHABLE.
    """
    record = await _check_session_access(session_id, current_user, db, admin_bypass=True)
    repo = SessionRepository(db)

    # Refuse to mutate LLM/config while an agent turn is actively running.
    # The turn's finally block merges its local bound-tools cache back into
    # the persistent cache, which would re-pollute it with stale entries if
    # we allowed the patch to proceed concurrently.
    registry_early = _get_registry(request)
    live_session_early = (await registry_early.get_cached(session_id)) if registry_early else None
    if live_session_early is not None and body.config is not None:
        if live_session_early.turn_task is not None and not live_session_early.turn_task.done():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "TURN_IN_PROGRESS",
                    "message": (
                        "An agent turn is in progress for this session. "
                        "Wait for it to complete before patching the config."
                    ),
                },
            )

    # Build update dict
    updates: dict[str, Any] = {}

    if body.name is not None:
        if await repo.name_exists_for_user(record.user_id, body.name, exclude_id=session_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "SESSION_NAME_DUPLICATE",
                    "message": f"A session named '{body.name}' already exists.",
                },
            )
        updates["name"] = body.name

    if body.config is not None:
        # Merge with existing config
        try:
            existing_config = json.loads(record.config_json) if record.config_json else {}
        except (json.JSONDecodeError, TypeError):
            existing_config = {}
        patch_dict = body.config.model_dump(exclude_none=True)
        merged = {**existing_config, **patch_dict}

        # Validate model alias before touching the DB so an invalid alias produces a
        # clear 422 rather than silently committing an unusable config (P0).
        if body.config.model is not None:
            app_cfg = getattr(request.app.state, "config", None)
            if app_cfg is not None:
                try:
                    app_cfg.resolve_llm_config_for(body.config.model)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail={
                            "code": "MODEL_NOT_FOUND",
                            "message": f"Model alias '{body.config.model}' is not configured: {exc}",
                        },
                    ) from exc

        updates["config_json"] = json.dumps(merged)

    if updates:
        record = await repo.update(session_id, **updates)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "SESSION_NOT_FOUND",
                    "message": "The requested session does not exist.",
                },
            )
        await db.commit()
        await db.refresh(record)

    # If model changed, rebuild the LLM in the live session.
    # Reuse the registry reference and live_session already fetched above for the
    # 409 check — avoids a second lookup that could return a different object after
    # the blocking LLM build (BUG-API-007).
    # Hold turn_lock while mutating live_session.llm / live_session.run_config so
    # a concurrent agent turn cannot observe a partially-updated session state
    # (BUG-FORGE-002).
    live_session = live_session_early

    if live_session is not None and body.config is not None:
        model_changed = body.config.model is not None
        if model_changed:
            try:
                from cogtrix_core.api.session_bridge import (  # noqa: PLC0415
                    _build_llm,
                    _build_run_config,
                )
                from cogtrix_core.orchestration.runner import invalidate_llm_caches

                new_config = json.loads(record.config_json) if record.config_json else {}
                # Build the LLM outside the lock — network I/O must not hold the lock.
                new_llm = await asyncio.to_thread(_build_llm, new_config, request.app.state)
                # Guard: _build_llm swallows errors and returns None on failure.
                # Never assign None to the live session — that would break all future
                # agent turns in this process lifetime.
                if new_llm is None:
                    log.warning(
                        "LLM build returned None for session %s (model=%s) — "
                        "live session retains previous LLM until next warm",
                        session_id,
                        body.config.model,
                    )
                else:
                    # Swap the live session state atomically under turn_lock so an
                    # in-flight agent turn cannot observe a half-updated session.
                    # _build_run_config is called inside the lock so it reads the
                    # just-built LLM and cannot race with an in-flight turn.
                    async with live_session.turn_lock:
                        new_run_config = _build_run_config(
                            new_llm,
                            live_session.session_state,
                            new_config,
                            request.app.state,
                        )
                        live_session.llm = new_llm
                        live_session.config = new_config
                        live_session.run_config = new_run_config
                    invalidate_llm_caches()
                    log.debug("Rebuilt LLM for session %s after provider/model change", session_id)
            except Exception as exc:
                log.warning("Could not rebuild LLM for session %s: %s", session_id, exc)

    return APIResponse(data=_record_to_out(record, live_session))


@router.delete(
    "/{session_id}",
    summary="Terminate and archive (or permanently delete) a session",
    description=(
        "Stop any in-progress agent turn, persist memory to disk, "
        "and archive the session. Archived sessions are excluded from list results "
        "by default but can be retrieved with include_archived=true. "
        "Memory data is retained for 30 days after archival. "
        "Pass permanent=true to irreversibly delete the session and all its messages."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Session archived or permanently deleted."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def delete_session(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    permanent: bool = Query(
        default=False,
        description=(
            "When true, permanently delete the session and all its messages. "
            "Non-recoverable. When false (default), archive the session."
        ),
    ),
) -> APIResponse[None]:
    """Terminate any active agent turn, save memory, and archive (or hard-delete) the session.

    Auth: bearer token required. Admins may delete any session.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    await _check_session_access(session_id, current_user, db, admin_bypass=True)
    repo = SessionRepository(db)

    # Cancel any running turn and save memory
    registry = _get_registry(request)
    if registry is not None:
        live_session = await registry.get_cached(session_id)
        if live_session is not None:
            # Signal cancellation
            if live_session.cancel_event is not None:
                live_session.cancel_event.set()
            if live_session.turn_task is not None and not live_session.turn_task.done():
                # Unblock any agent thread blocked in read_choice() before
                # cancelling the task; otherwise it can stay blocked for up
                # to 5 minutes waiting for a WebSocket confirmation that will
                # never arrive.  Use active_confirmation_ui — the per-turn UI
                # published by turn_runner — not the stale run_config template
                # (BUG-FORGE-001).
                _conf_ui = getattr(live_session, "active_confirmation_ui", None)
                if _conf_ui is not None and hasattr(_conf_ui, "cancel"):
                    _conf_ui.cancel()
                live_session.turn_task.cancel()
                try:
                    await live_session.turn_task
                except (asyncio.CancelledError, Exception):
                    pass
        # Save memory and evict from registry
        await registry.remove(session_id)

    # Archive or hard-delete in DB first — do the reversible DB write before
    # the irreversible side effects (registry eviction, WS close) so that a
    # failed commit leaves the session still active in memory (BUG-247).
    if permanent:
        deleted = await repo.hard_delete(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "SESSION_NOT_FOUND", "message": "Session not found."},
            )
    else:
        await repo.archive(session_id)
    await db.commit()

    # Close any active WebSocket connection for this session.
    try:
        from cogtrix_core.api.ws import manager as _ws_manager

        await _ws_manager.disconnect(session_id)
    except Exception:
        pass

    return APIResponse(data=None)


@router.post(
    "/{session_id}/restore",
    summary="Restore (unarchive) a session",
    description=(
        "Clear the archived_at timestamp on an archived session, making it visible "
        "in the default session listing again. Has no effect on active (non-archived) sessions."
    ),
    response_model=APIResponse[SessionOut],
    responses={
        200: {"description": "Session restored."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def restore_session(
    session_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SessionOut]:
    """Unarchive a session by clearing its archived_at timestamp.

    Auth: bearer token required. Admins may restore any session.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    await _check_session_access(session_id, current_user, db, admin_bypass=True)
    repo = SessionRepository(db)
    await repo.restore(session_id)
    record = await repo.get_by_id(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SESSION_NOT_FOUND", "message": "Session not found."},
        )
    await db.commit()
    return APIResponse(data=_record_to_out(record))
