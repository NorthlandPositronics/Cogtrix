"""Session management endpoints.

A session groups conversation history, memory state, and tool configuration.
Each user can have multiple concurrent sessions.

Endpoints:
    POST   /api/v1/sessions              — create a new session
    GET    /api/v1/sessions              — list sessions (paginated, cursor-based)
    GET    /api/v1/sessions/{id}         — get session details and current state
    PATCH  /api/v1/sessions/{id}         — update session name or config
    DELETE /api/v1/sessions/{id}         — terminate and archive a session
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user
from src.api.db.engine import get_db
from src.api.db.models import ApiSessionRecord
from src.api.db.repositories.sessions import SessionRepository
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.session import (
    SessionConfig,
    SessionCreateRequest,
    SessionOut,
    SessionPatchRequest,
    TokenCounts,
)
from src.api.session_bridge import warm_session

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
) -> ApiSessionRecord:
    """Fetch session and enforce ownership; returns the record on success.

    Admins may access any session.  Regular users may only access their own.

    Raises:
        HTTPException 404 SESSION_NOT_FOUND — session does not exist.
        HTTPException 403 FORBIDDEN — session belongs to a different user.
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

    if not current_user.is_admin and record.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "Authenticated user lacks permission for this action.",
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
        422: {"description": "Request body validation failed (VALIDATION_ERROR)."},
    },
)
async def create_session(
    request: Request,
    body: SessionCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SessionOut]:
    """Create a new agent session owned by the current user.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, VALIDATION_ERROR.
    """
    repo = SessionRepository(db)

    config_dict = body.config.model_dump(exclude_none=True)
    config_json = json.dumps(config_dict)

    record = await repo.create(
        user_id=current_user.user_id,
        name=body.name,
        config_json=config_json,
    )
    await db.commit()
    await db.refresh(record)

    # Warm the session into the registry (best-effort — non-blocking)
    registry = _get_registry(request)
    live_session = None
    if registry is not None:
        try:
            live_session = await warm_session(record, request.app.state)
            await registry.put(live_session)
        except Exception as exc:
            log.warning("Could not warm new session %s: %s", record.id, exc)

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
    limit: int = 20,
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
    limit = max(1, min(limit, 100))
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
    items = [_record_to_out(r, registry.get_cached(r.id) if registry else None) for r in page_rows]

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
    record = await _check_session_access(session_id, current_user, db)

    registry = _get_registry(request)
    live_session = registry.get_cached(session_id) if registry else None

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
    record = await _check_session_access(session_id, current_user, db)
    repo = SessionRepository(db)

    # Build update dict
    updates: dict[str, Any] = {}

    if body.name is not None:
        updates["name"] = body.name

    if body.config is not None:
        # Merge with existing config
        try:
            existing_config = json.loads(record.config_json) if record.config_json else {}
        except (json.JSONDecodeError, TypeError):
            existing_config = {}
        patch_dict = body.config.model_dump(exclude_none=True)
        merged = {**existing_config, **patch_dict}
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

    # If provider/model changed, rebuild the LLM in the live session
    registry = _get_registry(request)
    live_session = registry.get_cached(session_id) if registry else None

    if live_session is not None and body.config is not None:
        provider_changed = body.config.provider is not None
        model_changed = body.config.model is not None
        if provider_changed or model_changed:
            try:
                from src.api.session_bridge import _build_llm, _build_run_config  # noqa: PLC0415
                from src.orchestration.runner import invalidate_llm_caches

                new_config = json.loads(record.config_json) if record.config_json else {}
                # _build_llm may perform network I/O (e.g. Ollama API introspection);
                # run it off the event loop thread to prevent stalls.
                new_llm = await asyncio.to_thread(_build_llm, new_config, request.app.state)
                live_session.llm = new_llm
                live_session.config = new_config
                new_run_config = _build_run_config(
                    new_llm,
                    live_session.session_state,
                    new_config,
                    request.app.state,
                )
                live_session.run_config = new_run_config
                invalidate_llm_caches()
                log.debug("Rebuilt LLM for session %s after provider/model change", session_id)
            except Exception as exc:
                log.warning("Could not rebuild LLM for session %s: %s", session_id, exc)

    return APIResponse(data=_record_to_out(record, live_session))


@router.delete(
    "/{session_id}",
    summary="Terminate and archive a session",
    description=(
        "Stop any in-progress agent turn, persist memory to disk, "
        "and archive the session. Archived sessions are excluded from list results "
        "by default but can be retrieved with include_archived=true. "
        "Memory data is retained for 30 days after archival."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Session archived."},
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
) -> APIResponse[None]:
    """Terminate any active agent turn, save memory, and archive the session.

    Auth: bearer token required. Admins may archive any session.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    await _check_session_access(session_id, current_user, db)
    repo = SessionRepository(db)

    # Cancel any running turn and save memory
    registry = _get_registry(request)
    if registry is not None:
        live_session = registry.get_cached(session_id)
        if live_session is not None:
            # Signal cancellation
            if live_session.cancel_event is not None:
                live_session.cancel_event.set()
            if live_session.turn_task is not None and not live_session.turn_task.done():
                live_session.turn_task.cancel()
                try:
                    await live_session.turn_task
                except (asyncio.CancelledError, Exception):
                    pass
        # Save memory and evict from registry
        await registry.remove(session_id)

    # Close any active WebSocket connection for this session.
    try:
        from src.api.ws import manager as _ws_manager

        await _ws_manager.disconnect(session_id)
    except Exception:
        pass

    # Archive in DB
    await repo.archive(session_id)
    await db.commit()

    return APIResponse(data=None)
