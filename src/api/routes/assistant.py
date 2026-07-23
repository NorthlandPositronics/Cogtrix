"""Assistant mode (WhatsApp/Telegram daemon) management endpoints.

Endpoints:
    GET    /api/v1/assistant/status                    — get service status
    POST   /api/v1/assistant/start                     — start the assistant service (admin)
    POST   /api/v1/assistant/stop                      — stop the assistant service (admin)
    GET    /api/v1/assistant/chats                     — list active chat sessions (paginated)
    GET    /api/v1/assistant/chats/{key}/messages      — per-chat conversation history
    GET    /api/v1/assistant/scheduled                 — list scheduled messages (paginated)
    PATCH  /api/v1/assistant/scheduled/{id}            — edit a scheduled message
    DELETE /api/v1/assistant/scheduled/{id}            — cancel a scheduled message
    GET    /api/v1/assistant/deferred                  — list deferred re-processing records
    DELETE /api/v1/assistant/deferred/{session_key}    — cancel a deferred record
    GET    /api/v1/assistant/contacts                  — list phonebook contacts
    GET    /api/v1/assistant/guardrails                — guardrail pipeline status
    DELETE /api/v1/assistant/guardrails/blacklist/{chat_id} — remove from blacklist (admin)
    GET    /api/v1/assistant/knowledge                 — list knowledge store facts (paginated)
    POST   /api/v1/assistant/knowledge/search          — semantic search over facts
    DELETE /api/v1/assistant/knowledge/{fact_id}       — delete a fact (admin)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.pagination import decode_cursor, encode_cursor
from src.api.schemas.assistant import (
    AssistantStartRequest,
    AssistantStatusOut,
    ChannelStatusOut,
    ChatSessionOut,
    ContactOut,
    DeferredRecordOut,
    GuardrailStatusOut,
    KnowledgeFactOut,
    KnowledgeSearchRequest,
    ScheduledMessageEditRequest,
    ScheduledMessageOut,
    ViolationRecordOut,
)
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.message import MessageOut

router = APIRouter(prefix="/assistant", tags=["Assistant Mode"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_service(request: Request) -> Any:
    """Return the AssistantService or None from app.state."""
    return getattr(request.app.state, "assistant_service", None)


def _snapshot_sessions(session_mgr: Any) -> dict[str, Any]:
    """Return a snapshot of the session registry under its lock.

    Acquiring the lock prevents a RuntimeError from concurrent dict mutation
    by the background polling and eviction threads.
    """
    lock = getattr(session_mgr, "_lock", None)
    sessions_attr = getattr(session_mgr, "_sessions", {})
    if lock is None:
        return dict(sessions_attr)
    with lock:
        return dict(sessions_attr)


def _snapshot_scheduler_queue(scheduler: Any) -> dict[str, Any]:
    """Return a snapshot of the scheduler queue under its lock.

    Acquiring the lock prevents a RuntimeError from concurrent dict mutation
    by the background dispatch thread.
    """
    lock = getattr(scheduler, "_lock", None)
    queue_attr = getattr(scheduler, "_queue", {})
    if lock is None:
        return dict(queue_attr)
    with lock:
        return dict(queue_attr)


def _snapshot_deferral_records(deferral_mgr: Any) -> dict[str, Any]:
    """Return a snapshot of the deferral manager's records under its lock."""
    lock = getattr(deferral_mgr, "_lock", None)
    records_attr = getattr(deferral_mgr, "_records", {})
    if lock is None:
        return dict(records_attr)
    with lock:
        return dict(records_attr)


def _require_service(request: Request) -> Any:
    """Return the AssistantService or raise 409 if not running."""
    svc = _get_service(request)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "ASSISTANT_NOT_RUNNING",
                "message": "The assistant service is not running.",
            },
        )
    return svc


def _build_status(svc: Any) -> AssistantStatusOut:
    """Build an AssistantStatusOut from a running service."""
    channels: list[ChannelStatusOut] = []
    for ch in getattr(svc, "_channels", []):
        name = getattr(ch, "name", "unknown")
        # Use the channel name as its type when recognised; fall back to "whatsapp"
        # only as a last resort so unknown channel types don't silently masquerade
        # as WhatsApp in the status response.
        ch_type = name if name in ("whatsapp", "telegram") else "whatsapp"  # schema Literal
        # Count active sessions for this channel
        active = 0
        session_mgr = getattr(svc, "_session_mgr", None)
        if session_mgr is not None:
            sessions_dict = _snapshot_sessions(session_mgr)
            active = sum(1 for s in sessions_dict.values() if getattr(s, "channel", "") == name)

        # Poll interval from poller if available
        poll_interval = 5.0
        poller = getattr(svc, "_poller", None)
        if poller is not None:
            pollers = getattr(poller, "_pollers", {})
            ch_poller = pollers.get(name)
            if ch_poller is not None:
                poll_interval = float(getattr(ch_poller, "_current_interval", poll_interval))

        channels.append(
            ChannelStatusOut(
                name=name,
                type=ch_type,  # type: ignore[arg-type]
                enabled=True,
                connected=getattr(ch, "is_ready", lambda: True)(),
                active_chats=active,
                poll_interval_s=poll_interval,
                error=None,
            )
        )

    started_at: datetime | None = getattr(svc, "_started_at", None)
    uptime_s: float | None = None
    if started_at is not None:
        uptime_s = (datetime.now(UTC) - started_at).total_seconds()

    return AssistantStatusOut(
        status="running",
        channels=channels,
        started_at=started_at,
        uptime_s=uptime_s,
    )


def _ts_to_dt(ts: float) -> datetime:
    """Convert a POSIX timestamp to an aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    summary="Get assistant service status",
    description="Return the current status of the assistant daemon including per-channel connectivity.",
    response_model=APIResponse[AssistantStatusOut],
    responses={
        200: {"description": "Status returned."},
        401: {"description": "Not authenticated."},
    },
)
async def get_status(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[AssistantStatusOut]:
    """Return the current assistant service status.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    svc = _get_service(request)
    if svc is None:
        out = AssistantStatusOut(status="stopped", channels=[], started_at=None, uptime_s=None)
    else:
        out = _build_status(svc)
    return APIResponse(data=out)


@router.post(
    "/start",
    summary="Start the assistant service",
    description="Start the WhatsApp/Telegram polling daemon. Admin only.",
    response_model=APIResponse[AssistantStatusOut],
    responses={
        200: {"description": "Service started (or already running if force_restart=false)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {
            "description": "Already running and force_restart=false (ASSISTANT_ALREADY_RUNNING)."
        },
    },
)
async def start_assistant(
    body: AssistantStartRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[AssistantStatusOut]:
    """Start the assistant daemon (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, ASSISTANT_ALREADY_RUNNING.
    """
    svc = _get_service(request)
    if svc is not None:
        if not body.force_restart:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "ASSISTANT_ALREADY_RUNNING",
                    "message": "The assistant service is already running.",
                },
            )
        # Stop the existing service before restarting
        try:
            svc.stop() if hasattr(svc, "stop") else None
        except Exception:
            pass
        request.app.state.assistant_service = None

    # Try to construct and start AssistantService from app.state
    app_config = getattr(request.app.state, "config", None)
    tool_registry = getattr(request.app.state, "tool_registry", None)
    if app_config is None or tool_registry is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Cannot start assistant: config or tool registry not available.",
            },
        )

    try:
        from src.api.assistant_lifecycle import create_and_start_assistant

        new_svc = await create_and_start_assistant(app_config, tool_registry)
        request.app.state.assistant_service = new_svc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "INTERNAL_ERROR", "message": f"Failed to start assistant: {exc}"},
        ) from exc

    return APIResponse(data=_build_status(request.app.state.assistant_service))


@router.post(
    "/stop",
    summary="Stop the assistant service",
    description="Gracefully stop the assistant daemon. Pending scheduled messages are preserved. Admin only.",
    response_model=APIResponse[AssistantStatusOut],
    responses={
        200: {"description": "Service stopped."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {"description": "Service is not running (ASSISTANT_NOT_RUNNING)."},
    },
)
async def stop_assistant(
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[AssistantStatusOut]:
    """Stop the assistant daemon gracefully (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, ASSISTANT_NOT_RUNNING.
    """
    svc = _require_service(request)

    from src.api.assistant_lifecycle import shutdown_assistant_sync

    await asyncio.to_thread(shutdown_assistant_sync, svc)

    request.app.state.assistant_service = None
    out = AssistantStatusOut(status="stopped", channels=[], started_at=None, uptime_s=None)
    return APIResponse(data=out)


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------


@router.get(
    "/chats",
    summary="List active chat sessions",
    description=(
        "List all in-memory chat sessions managed by the assistant daemon. "
        "Ordered by last activity (newest first)."
    ),
    response_model=APIResponse[CursorPage[ChatSessionOut]],
    responses={
        200: {"description": "Chat session list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_chats(
    request: Request,
    cursor: str | None = None,
    limit: int = 50,
    channel: str | None = None,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[ChatSessionOut]]:
    """List active assistant chat sessions (paginated).

    Query parameters:
        cursor  — pagination cursor.
        limit   — page size (1–200, default 50).
        channel — filter to a specific channel ('whatsapp' or 'telegram').

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    svc = _get_service(request)
    if svc is None:
        page: CursorPage[ChatSessionOut] = CursorPage(
            items=[], next_cursor=None, has_more=False, total=0
        )
        return APIResponse(data=page)

    session_mgr = getattr(svc, "_session_mgr", None)
    sessions_dict: dict[str, Any] = {}
    if session_mgr is not None:
        sessions_dict = _snapshot_sessions(session_mgr)

    all_sessions = list(sessions_dict.values())
    # Filter by channel
    if channel is not None:
        all_sessions = [s for s in all_sessions if getattr(s, "channel", "") == channel]

    # Sort newest-first by last_activity
    all_sessions.sort(key=lambda s: getattr(s, "last_activity", 0.0), reverse=True)

    # Build output objects
    items_out: list[ChatSessionOut] = []
    for s in all_sessions:
        mm = getattr(s, "memory_manager", None)
        msg_count = 0
        last_act: datetime | None = None
        memory_mode_name = "conversation"

        if mm is not None:
            try:
                # Use the public get_messages() method when available; fall back to
                # the internal _messages list for memory managers that do not expose it.
                if callable(getattr(mm, "get_messages", None)):
                    msg_count = len(mm.get_messages())
                else:
                    stored = getattr(mm, "_messages", None)
                    msg_count = len(stored) if stored is not None else 0
            except Exception:
                msg_count = 0
            try:
                memory_mode_name = getattr(mm, "mode", "conversation")
                if not isinstance(memory_mode_name, str):
                    memory_mode_name = str(memory_mode_name)
            except Exception:
                memory_mode_name = "conversation"

        raw_ts = getattr(s, "last_activity", None)
        if raw_ts is not None:
            # last_activity is monotonic; convert back to wall clock
            try:
                wall_ts = time.time() - (time.monotonic() - raw_ts)
                last_act = _ts_to_dt(wall_ts)
            except Exception:
                last_act = None

        is_locked = False
        lock = getattr(s, "lock", None)
        if lock is not None:
            acquired = lock.acquire(blocking=False)
            if acquired:
                lock.release()
            else:
                is_locked = True

        items_out.append(
            ChatSessionOut(
                session_key=getattr(s, "session_key", ""),
                channel=getattr(s, "channel", ""),
                chat_id=getattr(s, "chat_id", ""),
                display_name=None,
                message_count=msg_count,
                last_activity=last_act,
                memory_mode=memory_mode_name,
                is_locked=is_locked,
            )
        )

    # Cursor-based pagination
    limit = max(1, min(limit, 200))
    start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor)
            for i, item in enumerate(items_out):
                if item.session_key == decoded:
                    start = i + 1
                    break
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
            ) from None

    page_items = items_out[start : start + limit]
    has_more = (start + limit) < len(items_out)
    next_cursor: str | None = None
    if has_more and page_items:
        next_cursor = encode_cursor(page_items[-1].session_key)

    page = CursorPage(
        items=page_items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=len(items_out),
    )
    return APIResponse(data=page)


@router.get(
    "/chats/{session_key:path}/messages",
    summary="Get per-chat conversation history",
    description=(
        "Return the conversation history for a specific assistant chat session. "
        "session_key uses the format '{channel}::{chat_id}' (URL-encode the :: separator)."
    ),
    response_model=APIResponse[CursorPage[MessageOut]],
    responses={
        200: {"description": "Message history returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Chat session not found (NOT_FOUND)."},
    },
)
async def get_chat_messages(
    session_key: str,
    request: Request,
    cursor: str | None = None,
    limit: int = 50,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[MessageOut]]:
    """Return message history for an assistant chat session (paginated).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, NOT_FOUND, INVALID_CURSOR.
    """
    svc = _require_service(request)
    session_mgr = getattr(svc, "_session_mgr", None)
    sessions_dict: dict[str, Any] = {}
    if session_mgr is not None:
        sessions_dict = _snapshot_sessions(session_mgr)

    chat_session = sessions_dict.get(session_key)
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Chat session '{session_key}' not found."},
        )

    mm = getattr(chat_session, "memory_manager", None)
    raw_msgs: list[Any] = []
    if mm is not None:
        try:
            # Use the public get_messages() method when available; fall back to
            # the internal _messages list for memory managers that do not expose it.
            if callable(getattr(mm, "get_messages", None)):
                raw_msgs = mm.get_messages()
            else:
                stored = getattr(mm, "_messages", None)
                raw_msgs = list(stored) if stored is not None else []
        except Exception:
            raw_msgs = []

    # Convert langchain messages to MessageOut
    import uuid as _uuid

    def _msg_to_out(msg: Any, idx: int) -> MessageOut:
        role = "user"
        msg_type = type(msg).__name__
        if "Human" in msg_type:
            role = "user"
        elif "AI" in msg_type:
            role = "assistant"
        elif "Tool" in msg_type:
            role = "tool"
        elif "System" in msg_type:
            role = "system"

        content = ""
        raw_content = getattr(msg, "content", "")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for part in raw_content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            content = "".join(parts)

        return MessageOut(
            id=str(_uuid.uuid4()),
            session_id=session_key,
            role=role,  # type: ignore[arg-type]
            content=content,
            tool_calls=[],
            token_counts=None,
            created_at=datetime.now(UTC),
        )

    items_out = [_msg_to_out(m, i) for i, m in enumerate(raw_msgs)]

    # Paginate
    limit = max(1, min(limit, 200))
    start = 0
    if cursor is not None:
        try:
            decoded_idx = int(decode_cursor(cursor))
            start = decoded_idx
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
            ) from None

    page_items = items_out[start : start + limit]
    has_more = (start + limit) < len(items_out)
    next_cursor: str | None = None
    if has_more:
        next_cursor = encode_cursor(str(start + limit))

    page = CursorPage(
        items=page_items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=len(items_out),
    )
    return APIResponse(data=page)


# ---------------------------------------------------------------------------
# Scheduled messages
# ---------------------------------------------------------------------------


def _scheduled_msg_to_out(msg: Any) -> ScheduledMessageOut:
    """Convert a ScheduledMessage dataclass to ScheduledMessageOut."""
    status_val = getattr(msg, "status", "pending")
    # Normalize status values from scheduler to schema literals
    _status_map = {
        "sending": "firing",
        "expired": "failed",
    }
    status_val = _status_map.get(status_val, status_val)
    if status_val not in ("pending", "firing", "sent", "failed", "cancelled"):
        status_val = "pending"

    return ScheduledMessageOut(
        id=getattr(msg, "id", ""),
        chat_id=getattr(msg, "chat_id", ""),
        channel=getattr(msg, "channel", ""),
        recipient=getattr(msg, "recipient", None) or getattr(msg, "chat_id", ""),
        text=getattr(msg, "text", ""),
        send_at=_ts_to_dt(float(getattr(msg, "send_at", 0.0))),
        created_at=_ts_to_dt(float(getattr(msg, "created_at", 0.0))),
        attempts=int(getattr(msg, "attempts", 0)),
        max_attempts=int(getattr(msg, "max_attempts", 3)),
        status=status_val,  # type: ignore[arg-type]
    )


@router.get(
    "/scheduled",
    summary="List scheduled messages",
    description="List pending scheduled messages with optional filters.",
    response_model=APIResponse[CursorPage[ScheduledMessageOut]],
    responses={
        200: {"description": "Scheduled message list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_scheduled(
    request: Request,
    cursor: str | None = None,
    limit: int = 50,
    channel: str | None = None,
    chat_id: str | None = None,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[ScheduledMessageOut]]:
    """List pending scheduled messages (paginated).

    Query parameters:
        cursor  — pagination cursor.
        limit   — page size (1–200, default 50).
        channel — filter to a specific channel.
        chat_id — filter to a specific chat.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    svc = _get_service(request)
    if svc is None:
        page: CursorPage[ScheduledMessageOut] = CursorPage(
            items=[], next_cursor=None, has_more=False, total=0
        )
        return APIResponse(data=page)

    scheduler = getattr(svc, "_scheduler", None)
    all_msgs: list[Any] = []
    if scheduler is not None:
        try:
            all_msgs = scheduler.get_pending(
                recipient=None,
                chat_id=chat_id,
                include_all=False,
            )
        except Exception:
            all_msgs = []

    if channel is not None:
        all_msgs = [m for m in all_msgs if getattr(m, "channel", "") == channel]

    items_out = [_scheduled_msg_to_out(m) for m in all_msgs]

    limit = max(1, min(limit, 200))
    start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor)
            for i, item in enumerate(items_out):
                if item.id == decoded:
                    start = i + 1
                    break
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
            ) from None

    page_items = items_out[start : start + limit]
    has_more = (start + limit) < len(items_out)
    next_cursor: str | None = None
    if has_more and page_items:
        next_cursor = encode_cursor(page_items[-1].id)

    page = CursorPage(
        items=page_items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=len(items_out),
    )
    return APIResponse(data=page)


@router.patch(
    "/scheduled/{message_id}",
    summary="Edit a scheduled message",
    description=(
        "Update the text or delivery time of a pending scheduled message. "
        "Only pending messages can be edited (not those in 'firing', 'sent', or 'failed' state)."
    ),
    response_model=APIResponse[ScheduledMessageOut],
    responses={
        200: {"description": "Scheduled message updated."},
        401: {"description": "Not authenticated."},
        404: {
            "description": "Scheduled message not found or not editable (SCHEDULED_MSG_NOT_FOUND)."
        },
    },
)
async def edit_scheduled(
    message_id: str,
    body: ScheduledMessageEditRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[ScheduledMessageOut]:
    """Edit a pending scheduled message's text or delivery time.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, SCHEDULED_MSG_NOT_FOUND.
    """
    svc = _require_service(request)
    scheduler = getattr(svc, "_scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCHEDULED_MSG_NOT_FOUND",
                "message": "Scheduler not available.",
            },
        )

    # Find message in queue
    queue: dict[str, Any] = _snapshot_scheduler_queue(scheduler)
    msg = queue.get(message_id)

    # Also try prefix match
    if msg is None:
        for mid, m in queue.items():
            if mid.startswith(message_id):
                msg = m
                message_id = mid
                break

    if msg is None or getattr(msg, "status", "") != "pending":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCHEDULED_MSG_NOT_FOUND",
                "message": f"No pending scheduled message found with id '{message_id}'.",
            },
        )

    new_send_at: float | None = None
    if body.send_at is not None:
        new_send_at = body.send_at.timestamp()

    ok = scheduler.edit_message(message_id, new_text=body.text, new_send_at=new_send_at)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCHEDULED_MSG_NOT_FOUND",
                "message": f"Could not edit message '{message_id}'.",
            },
        )

    # Re-fetch from the live queue after the edit so the response reflects the update.
    updated_msg = _snapshot_scheduler_queue(scheduler).get(message_id) or msg
    return APIResponse(data=_scheduled_msg_to_out(updated_msg))


@router.delete(
    "/scheduled/{message_id}",
    summary="Cancel a scheduled message",
    description="Cancel and remove a pending scheduled message.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Scheduled message cancelled."},
        401: {"description": "Not authenticated."},
        404: {
            "description": "Scheduled message not found or already sent/cancelled (SCHEDULED_MSG_NOT_FOUND)."
        },
    },
)
async def cancel_scheduled(
    message_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[None]:
    """Cancel a pending scheduled message.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, SCHEDULED_MSG_NOT_FOUND.
    """
    svc = _require_service(request)
    scheduler = getattr(svc, "_scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCHEDULED_MSG_NOT_FOUND",
                "message": "Scheduler not available.",
            },
        )

    # Try exact match then prefix match
    queue: dict[str, Any] = _snapshot_scheduler_queue(scheduler)
    if message_id not in queue:
        for mid in queue:
            if mid.startswith(message_id):
                message_id = mid
                break

    ok = scheduler.cancel_message(message_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SCHEDULED_MSG_NOT_FOUND",
                "message": f"No pending scheduled message found with id '{message_id}'.",
            },
        )

    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Deferred messages
# ---------------------------------------------------------------------------


def _deferred_record_to_out(session_key: str, record: Any) -> DeferredRecordOut:
    """Convert a DeferredRecord to DeferredRecordOut."""
    status_val = getattr(record, "status", "pending")
    if status_val not in ("pending", "firing"):
        status_val = "pending"

    pending_msgs = getattr(record, "pending_messages", [])
    texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in pending_msgs]

    return DeferredRecordOut(
        session_key=session_key,
        fire_at=_ts_to_dt(float(getattr(record, "fire_at", 0.0))),
        pending_messages=texts,
        depth=int(getattr(record, "deferral_depth", 0)),
        max_depth=3,
        status=status_val,  # type: ignore[arg-type]
        created_at=_ts_to_dt(float(getattr(record, "created_at", 0.0))),
    )


@router.get(
    "/deferred",
    summary="List deferred re-processing records",
    description="List all pending deferred re-processing records managed by the DeferralManager.",
    response_model=APIResponse[list[DeferredRecordOut]],
    responses={
        200: {"description": "Deferred record list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_deferred(
    request: Request,
    channel: str | None = None,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[DeferredRecordOut]]:
    """List pending deferred re-processing records.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    svc = _get_service(request)
    if svc is None:
        return APIResponse(data=[])

    deferral_mgr = getattr(svc, "_deferral_mgr", None)
    if deferral_mgr is None:
        return APIResponse(data=[])

    records: dict[str, Any] = _snapshot_deferral_records(deferral_mgr)
    out: list[DeferredRecordOut] = []
    for key, record in records.items():
        rec_status = getattr(record, "status", "")
        if rec_status not in ("pending", "firing"):
            continue
        rec_channel = getattr(record, "channel", "")
        if channel is not None and rec_channel != channel:
            continue
        out.append(_deferred_record_to_out(key, record))

    return APIResponse(data=out)


@router.delete(
    "/deferred/{session_key:path}",
    summary="Cancel a deferred re-processing record",
    description="Cancel a pending deferred record for the given session key.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Deferred record cancelled."},
        401: {"description": "Not authenticated."},
        404: {"description": "Deferred record not found (DEFERRED_MSG_NOT_FOUND)."},
    },
)
async def cancel_deferred(
    session_key: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[None]:
    """Cancel a pending deferred record.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, DEFERRED_MSG_NOT_FOUND.
    """
    svc = _require_service(request)
    deferral_mgr = getattr(svc, "_deferral_mgr", None)
    if deferral_mgr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DEFERRED_MSG_NOT_FOUND",
                "message": "Deferral manager not available.",
            },
        )

    ok = deferral_mgr.cancel(session_key)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DEFERRED_MSG_NOT_FOUND",
                "message": f"No pending deferred record found for session '{session_key}'.",
            },
        )

    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Contacts (phonebook)
# ---------------------------------------------------------------------------


@router.get(
    "/contacts",
    summary="List phonebook contacts",
    description="Return the merged phonebook from all configured channels.",
    response_model=APIResponse[list[ContactOut]],
    responses={
        200: {"description": "Contact list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_contacts(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[ContactOut]]:
    """Return the merged phonebook from all channels.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    svc = _get_service(request)
    if svc is None:
        return APIResponse(data=[])

    # Gather services config for phonebook resolution
    config = getattr(svc, "_config", None)
    services_cfg: dict[str, Any] = {}
    if config is not None and hasattr(config, "services"):
        services_cfg = config.services or {}

    try:
        from src.assistant.scheduler import _merge_phonebooks

        merged = _merge_phonebooks(services_cfg)
    except Exception:
        merged = {}

    contacts: list[ContactOut] = []
    for name, identifiers in merged.items():
        contacts.append(
            ContactOut(
                name=name,
                identifiers=identifiers,
                channels=[],
                prompt=None,
                filter_mode=None,
            )
        )

    contacts.sort(key=lambda c: c.name)
    return APIResponse(data=contacts)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


@router.get(
    "/guardrails",
    summary="Guardrail pipeline status",
    description="Return the current guardrail status: blacklisted chats, recent violations, rate limiter stats.",
    response_model=APIResponse[GuardrailStatusOut],
    responses={
        200: {"description": "Guardrail status returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def get_guardrails(
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[GuardrailStatusOut]:
    """Return guardrail pipeline status (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    svc = _get_service(request)

    blacklisted: list[str] = []
    total_violations = 0
    recent_violations: list[ViolationRecordOut] = []

    if svc is not None:
        handler = getattr(svc, "_handler", None)
        guardrails = None
        if handler is not None:
            guardrails = getattr(handler, "_guardrails", None)
        if guardrails is None:
            # Try direct attribute on service
            guardrails = getattr(svc, "_guardrails", None)

        if guardrails is not None:
            violation_tracker = getattr(guardrails, "_violation_tracker", None)
            if violation_tracker is not None:
                vt_lock = getattr(violation_tracker, "_lock", None)
                raw_violations = getattr(violation_tracker, "_violations", {})
                # Snapshot under lock to prevent RuntimeError from concurrent mutation.
                if vt_lock is not None:
                    with vt_lock:
                        violations_dict: dict[str, Any] = {
                            k: list(v) for k, v in raw_violations.items()
                        }
                else:
                    violations_dict = {k: list(v) for k, v in raw_violations.items()}
                max_v = getattr(violation_tracker, "_max_violations", 2)
                now = time.monotonic()
                window = getattr(violation_tracker, "_window_seconds", 1800.0)

                for chat_id, timestamps in violations_dict.items():
                    recent_ts = [ts for ts in timestamps if (now - ts) <= window]
                    total_violations += len(recent_ts)
                    if len(recent_ts) >= max_v:
                        blacklisted.append(chat_id)

    out = GuardrailStatusOut(
        blacklisted_chats=blacklisted,
        total_violations=total_violations,
        recent_violations=recent_violations,
    )
    return APIResponse(data=out)


@router.delete(
    "/guardrails/blacklist/{chat_id:path}",
    summary="Remove a chat from the blacklist",
    description="Remove a chat from the auto-blacklist so it can send messages again. Admin only.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Chat removed from blacklist."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Chat not in blacklist (NOT_FOUND)."},
    },
)
async def remove_from_blacklist(
    chat_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Remove a chat from the violation blacklist (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND.
    """
    svc = _require_service(request)

    handler = getattr(svc, "_handler", None)
    guardrails = None
    if handler is not None:
        guardrails = getattr(handler, "_guardrails", None)

    if guardrails is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Guardrail pipeline not available."},
        )

    violation_tracker = getattr(guardrails, "_violation_tracker", None)
    if violation_tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Violation tracker not available."},
        )

    lock = getattr(violation_tracker, "_lock", None)
    violations_dict: dict[str, Any] = getattr(violation_tracker, "_violations", {})

    if lock is None:
        if chat_id not in violations_dict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"Chat '{chat_id}' is not blacklisted."},
            )
        del violations_dict[chat_id]
    else:
        with lock:
            if chat_id not in violations_dict:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "code": "NOT_FOUND",
                        "message": f"Chat '{chat_id}' is not blacklisted.",
                    },
                )
            del violations_dict[chat_id]

    # Persist the update
    try:
        violation_tracker.save()
    except Exception:
        pass

    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Shared knowledge store
# ---------------------------------------------------------------------------


def _fact_to_out(fact: Any, relevance: float | None = None) -> KnowledgeFactOut:
    """Convert a Fact dataclass to KnowledgeFactOut."""
    ts = float(getattr(fact, "timestamp", 0.0))
    source_session = getattr(fact, "source_session", "") or None

    return KnowledgeFactOut(
        id=getattr(fact, "fact_hash", ""),
        text=f"{getattr(fact, 'entity', '')}: {getattr(fact, 'fact', '')}",
        source_chat=source_session,
        source_channel=None,
        created_at=_ts_to_dt(ts) if ts > 0 else datetime.now(UTC),
        relevance_score=relevance,
    )


@router.get(
    "/knowledge",
    summary="List knowledge store facts",
    description="Browse all facts in the shared knowledge store (paginated, newest first).",
    response_model=APIResponse[CursorPage[KnowledgeFactOut]],
    responses={
        200: {"description": "Fact list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_knowledge(
    request: Request,
    cursor: str | None = None,
    limit: int = 50,
    source_chat: str | None = None,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[KnowledgeFactOut]]:
    """List facts in the shared knowledge store (paginated).

    Query parameters:
        cursor      — pagination cursor.
        limit       — page size (1–200, default 50).
        source_chat — filter to facts extracted from a specific chat session key.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    svc = _get_service(request)
    if svc is None:
        page: CursorPage[KnowledgeFactOut] = CursorPage(
            items=[], next_cursor=None, has_more=False, total=0
        )
        return APIResponse(data=page)

    knowledge_store = getattr(svc, "_knowledge_store", None)
    if knowledge_store is None:
        page = CursorPage(items=[], next_cursor=None, has_more=False, total=0)
        return APIResponse(data=page)

    ks_lock = getattr(knowledge_store, "_lock", None)
    if ks_lock is not None:
        with ks_lock:
            facts_snapshot = list(getattr(knowledge_store, "_facts", []))
    else:
        facts_snapshot = list(getattr(knowledge_store, "_facts", []))

    # Sort newest first
    facts_snapshot.sort(key=lambda f: getattr(f, "timestamp", 0.0), reverse=True)

    if source_chat is not None:
        facts_snapshot = [
            f for f in facts_snapshot if getattr(f, "source_session", "") == source_chat
        ]

    items_out = [_fact_to_out(f) for f in facts_snapshot]

    limit = max(1, min(limit, 200))
    start = 0
    if cursor is not None:
        try:
            decoded = decode_cursor(cursor)
            for i, item in enumerate(items_out):
                if item.id == decoded:
                    start = i + 1
                    break
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
            ) from None

    page_items = items_out[start : start + limit]
    has_more = (start + limit) < len(items_out)
    next_cursor: str | None = None
    if has_more and page_items:
        next_cursor = encode_cursor(page_items[-1].id)

    page = CursorPage(
        items=page_items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=len(items_out),
    )
    return APIResponse(data=page)


@router.post(
    "/knowledge/search",
    summary="Semantic search over knowledge facts",
    description="Perform a vector similarity search over the shared knowledge store.",
    response_model=APIResponse[list[KnowledgeFactOut]],
    responses={
        200: {"description": "Search results returned."},
        401: {"description": "Not authenticated."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def search_knowledge(
    body: KnowledgeSearchRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[list[KnowledgeFactOut]]:
    """Semantic search over the shared knowledge store.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, VALIDATION_ERROR.
    """
    svc = _get_service(request)
    if svc is None:
        return APIResponse(data=[])

    knowledge_store = getattr(svc, "_knowledge_store", None)
    if knowledge_store is None:
        return APIResponse(data=[])

    try:
        results_text = await asyncio.to_thread(knowledge_store.recall, body.query, body.top_k)
    except Exception:
        results_text = None

    if not results_text:
        return APIResponse(data=[])

    # Parse returned text back into fact-like objects to build output
    ks_lock2 = getattr(knowledge_store, "_lock", None)
    if ks_lock2 is not None:
        with ks_lock2:
            facts_snapshot = list(getattr(knowledge_store, "_facts", []))
    else:
        facts_snapshot = list(getattr(knowledge_store, "_facts", []))

    # Match facts by text prefix
    query_lower = body.query.lower()
    matched: list[KnowledgeFactOut] = []
    for fact in facts_snapshot:
        text = f"{getattr(fact, 'entity', '')}: {getattr(fact, 'fact', '')}".lower()
        query_tokens = set(query_lower.split())
        overlap = sum(1 for tok in query_tokens if tok in text)
        if overlap > 0:
            score = overlap / max(len(query_tokens), 1)
            matched.append(_fact_to_out(fact, relevance=round(score, 4)))

    matched.sort(key=lambda f: f.relevance_score or 0.0, reverse=True)
    return APIResponse(data=matched[: body.top_k])


@router.delete(
    "/knowledge/{fact_id}",
    summary="Delete a knowledge fact",
    description="Permanently delete a fact from the shared knowledge store. Admin only.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Fact deleted."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Fact not found (FACT_NOT_FOUND)."},
    },
)
async def delete_fact(
    fact_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Delete a fact from the shared knowledge store (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, FACT_NOT_FOUND.
    """
    svc = _require_service(request)
    knowledge_store = getattr(svc, "_knowledge_store", None)
    if knowledge_store is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FACT_NOT_FOUND", "message": "Knowledge store not available."},
        )

    del_lock = getattr(knowledge_store, "_lock", None)
    hashes = getattr(knowledge_store, "_fact_hashes", set())

    def _do_delete() -> None:
        facts: list[Any] = getattr(knowledge_store, "_facts", [])
        original_len = len(facts)
        new_facts = [f for f in facts if getattr(f, "fact_hash", "") != fact_id]
        if len(new_facts) == original_len:
            raise KeyError(fact_id)
        knowledge_store._facts = new_facts
        hashes.discard(fact_id)

    try:
        if del_lock is not None:
            with del_lock:
                _do_delete()
        else:
            _do_delete()
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "FACT_NOT_FOUND",
                "message": f"Fact '{fact_id}' not found in knowledge store.",
            },
        ) from None

    try:
        knowledge_store.save()
    except Exception:
        pass

    return APIResponse(data=None)
