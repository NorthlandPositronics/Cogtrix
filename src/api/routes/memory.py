"""Memory management endpoints.

Endpoints:
    GET    /api/v1/sessions/{id}/memory      — get current memory state and mode
    DELETE /api/v1/sessions/{id}/memory      — clear session memory
    PATCH  /api/v1/sessions/{id}/memory      — switch memory mode
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, verify_session_owner
from src.api.db.engine import get_db
from src.api.schemas.common import APIResponse
from src.api.schemas.memory import MemoryModeSwitchRequest, MemoryStateOut

log = logging.getLogger("cogtrix.api.memory")

router = APIRouter(tags=["Memory"])


def _get_session_registry(request: Request) -> Any:
    return getattr(request.app.state, "session_registry", None)


def _memory_state_out(session_id: str, live_session: Any, mm: Any) -> MemoryStateOut:
    """Build MemoryStateOut from a live session and its memory manager."""
    state_dict: dict[str, Any] = {}
    if mm is not None and hasattr(mm, "to_dict"):
        try:
            state_dict = mm.to_dict() or {}
        except Exception as exc:
            log.debug("memory_manager.to_dict() failed: %s", exc)

    mode = (
        state_dict.get("mode")
        or (live_session.config.get("memory_mode") if live_session.config else None)
        or "conversation"
    )

    summary = state_dict.get("summary") or None
    window_messages = state_dict.get("window_size") or state_dict.get("window_messages") or 0
    summarized_messages = state_dict.get("summarized_messages") or 0
    tokens_used = state_dict.get("tokens_used") or 0
    context_window = (
        live_session.token_counts.get("context_window", 131072)
        if live_session.token_counts
        else 131072
    )
    vector_recall_enabled = bool(state_dict.get("vector_recall_enabled", False))
    mode_meta = state_dict.get("mode_meta") or {}

    return MemoryStateOut(
        session_id=session_id,
        mode=mode,  # type: ignore[arg-type]
        summary=summary,
        window_messages=int(window_messages),
        summarized_messages=int(summarized_messages),
        tokens_used=int(tokens_used),
        context_window=int(context_window),
        vector_recall_enabled=vector_recall_enabled,
        mode_meta=mode_meta,
        updated_at=datetime.now(UTC),
    )


async def _resolve_session(session_id: str, request: Request, db: AsyncSession) -> Any:
    """Resolve and warm a session, raising 404 if not found."""
    session_registry = _get_session_registry(request)
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
    return live_session


@router.get(
    "/sessions/{session_id}/memory",
    summary="Get memory state",
    description=(
        "Return the current memory snapshot for a session: active mode, "
        "the LLM-generated summary of older messages, sliding window size, "
        "token usage, and mode-specific metadata (goals, code tasks, entities, etc.)."
    ),
    response_model=APIResponse[MemoryStateOut],
    responses={
        200: {"description": "Memory state returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def get_memory(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[MemoryStateOut]:
    """Return the current memory snapshot for a session.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND.
    """
    await verify_session_owner(session_id, current_user, db)
    live_session = await _resolve_session(session_id, request, db)
    mm = live_session.memory_manager
    return APIResponse(data=_memory_state_out(session_id, live_session, mm))


@router.delete(
    "/sessions/{session_id}/memory",
    summary="Clear session memory",
    description=(
        "Clear the full conversation memory for a session: wipe the sliding window, "
        "the summary, the vector recall index, and all mode-specific state. "
        "The session config (provider, model, memory mode) is preserved. "
        "Equivalent to the /clear CLI command."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Memory cleared."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def clear_memory(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """Wipe all memory for a session (history, summary, vector index, mode state).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND, MEMORY_CLEAR_FAILED.
    """
    await verify_session_owner(session_id, current_user, db)
    live_session = await _resolve_session(session_id, request, db)
    mm = live_session.memory_manager
    if mm is not None:
        try:

            def _do_clear() -> None:
                if hasattr(mm, "clear"):
                    mm.clear()
                elif hasattr(mm, "_messages"):
                    mm._messages = []
                    if hasattr(mm, "_summary"):
                        mm._summary = ""

            await asyncio.to_thread(_do_clear)
        except Exception as exc:
            log.warning("Memory clear failed for session %s: %s", session_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "MEMORY_CLEAR_FAILED",
                    "message": "Memory could not be cleared.",
                },
            ) from exc
    return APIResponse(data=None)


@router.patch(
    "/sessions/{session_id}/memory",
    summary="Switch memory mode",
    description=(
        "Switch the active memory mode for a session. "
        "Valid modes: 'conversation' (sliding window + summarization of chat history), "
        "'code' (preserves code tasks, file paths, and edit history), "
        "'reasoning' (tracks goals, decisions, and reasoning chains). "
        "Switching mode does not clear existing memory — context is migrated. "
        "Equivalent to the /mode CLI command."
    ),
    response_model=APIResponse[MemoryStateOut],
    responses={
        200: {"description": "Memory mode switched; updated memory state returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def switch_memory_mode(
    session_id: str,
    body: MemoryModeSwitchRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[MemoryStateOut]:
    """Switch the memory mode for a session and return the updated state.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND, VALIDATION_ERROR.
    """
    await verify_session_owner(session_id, current_user, db)
    live_session = await _resolve_session(session_id, request, db)

    target_mode = body.mode
    try:
        import src.memory.modes  # noqa: F401 — register all modes
        from src.memory import JsonFileMemoryStore, MemoryFactory

        if not MemoryFactory.is_registered(target_mode):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": f"Unknown memory mode: {target_mode!r}.",
                },
            )

        old_mm = live_session.memory_manager
        if old_mm is not None:
            try:
                await asyncio.to_thread(old_mm.save)
            except Exception as exc:
                log.warning("Could not save old memory before mode switch: %s", exc)

        def _create_and_load() -> Any:
            app_cfg = getattr(request.app.state, "config", None)
            history_dir = (
                str(Path(app_cfg.data_dir) / "history") if app_cfg is not None else "data/history"
            )
            store = JsonFileMemoryStore(history_dir)
            mm = MemoryFactory.create(target_mode, store, session_id)
            mm.load()
            return mm

        new_mm = await asyncio.to_thread(_create_and_load)
        live_session.memory_manager = new_mm
        if live_session.config is not None:
            live_session.config["memory_mode"] = target_mode

    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Memory mode switch failed for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Could not switch memory mode: {exc}",
            },
        ) from exc

    return APIResponse(
        data=_memory_state_out(session_id, live_session, live_session.memory_manager)
    )
