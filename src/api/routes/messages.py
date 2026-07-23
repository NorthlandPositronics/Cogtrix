"""Message endpoints and WebSocket session handler.

Messages are the turn-level records within a session.  Sending a message
initiates an agent turn; the response streams over the session WebSocket.

REST Endpoints:
    POST   /api/v1/sessions/{id}/messages        — send a user message, initiate agent turn
    GET    /api/v1/sessions/{id}/messages        — list message history (paginated)
    DELETE /api/v1/sessions/{id}/messages        — clear conversation history

WebSocket Endpoint:
    WS /ws/v1/sessions/{id}                      — stream agent output for the session

The WebSocket handler:
1. Validates the bearer token (header or ?token= query param).
2. Verifies the user owns the session.
3. Registers the connection in the ConnectionManager.
4. Starts the ping/pong keepalive loop (client sends ping every 30 s).
5. Dispatches incoming client messages (user_message, tool_confirm, cancel).
6. Forwards server messages from the agent runner to the client.
7. Closes the connection on 'done' or on idle timeout (90 s of silence).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, _decode_jwt, get_current_user, verify_session_owner
from src.api.db.engine import AsyncSessionLocal, get_db
from src.api.db.repositories.messages import MessageRepository
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.message import ClearHistoryRequest, MessageOut, SendMessageRequest
from src.api.turn_runner import run_message_turn
from src.api.ws import ClientMessage, manager

log = logging.getLogger("cogtrix.api.messages")

router = APIRouter(tags=["Messages"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_to_out(msg: Any) -> MessageOut:
    """Convert a Message ORM instance to a MessageOut schema."""
    content = ""
    try:
        data = json.loads(msg.content_json) if msg.content_json else {}
        content = data.get("text", "") or data.get("content", "") or str(data)
    except Exception:
        content = str(msg.content_json or "")

    tool_calls = []
    if msg.tool_calls_json:
        try:
            tool_calls = json.loads(msg.tool_calls_json) or []
        except Exception:
            pass

    return MessageOut(
        id=msg.id,
        session_id=msg.session_id,
        role=msg.role,
        content=content,
        tool_calls=tool_calls,
        token_counts=None,
        created_at=msg.created_at,
    )


async def _get_session_or_404(session_id: str, request: Request, db: AsyncSession) -> Any:
    """Return the in-memory ApiSession or raise 404."""
    registry = getattr(request.app.state, "session_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "INTERNAL_ERROR", "message": "Session registry not available."},
        )
    sess = await registry.get_cached(session_id)
    if sess is None:
        sess = await registry.get_or_warm(session_id, db)
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )
    return sess


# ---------------------------------------------------------------------------
# REST endpoints (nested under /sessions)
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/messages",
    summary="Send a user message and initiate an agent turn",
    description=(
        "Submit a user message to the session. "
        "The message is persisted immediately and an agent turn is queued. "
        "Connect to the session WebSocket (WS /ws/v1/sessions/{id}) to receive "
        "streaming token output while the agent processes the request. "
        "The REST response returns the persisted user message — not the agent reply."
    ),
    response_model=APIResponse[MessageOut],
    status_code=202,
    responses={
        202: {"description": "Message accepted; agent turn queued. Stream output via WebSocket."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
        409: {"description": "Agent turn already in progress for this session."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[MessageOut]:
    """Persist a user message and queue an agent turn.

    The agent runs asynchronously.  The caller must connect to the session
    WebSocket to receive the streaming response.

    Auth: bearer token required.
    Error codes:
        UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND,
        VALIDATION_ERROR.
    """
    await verify_session_owner(session_id, current_user, db)
    sess = await _get_session_or_404(session_id, request, db)

    # Atomically check-and-set turn_task under turn_lock to prevent a race
    # where two concurrent requests both see turn_task as None and both create tasks.
    async with sess.turn_lock:
        if sess.turn_task is not None and not sess.turn_task.done():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "TURN_IN_PROGRESS",
                    "message": "An agent turn is already in progress for this session.",
                },
            )

        # Persist the user message.
        msg_repo = MessageRepository(db)
        user_msg = await msg_repo.create(
            session_id=session_id,
            role="user",
            content_json=json.dumps({"text": body.content}),
        )
        await db.commit()

        # Launch the agent turn as a background task.
        async def _run() -> None:
            async with AsyncSessionLocal() as turn_db:
                await run_message_turn(
                    session=sess,
                    text=body.content,
                    mode=body.mode,
                    db=turn_db,
                    app_state=request.app.state,
                )

        sess.turn_task = asyncio.create_task(_run(), name=f"turn-{session_id}")

    return APIResponse(data=_message_to_out(user_msg))


@router.get(
    "/sessions/{session_id}/messages",
    summary="List message history",
    description=(
        "List the conversation history for a session ordered chronologically (oldest first). "
        "Uses cursor-based pagination. AI messages include embedded tool_calls records. "
        "Pass next_cursor from the response as the cursor parameter for the next page."
    ),
    response_model=APIResponse[CursorPage[MessageOut]],
    responses={
        200: {"description": "Message history returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
        400: {"description": "Invalid cursor (INVALID_CURSOR)."},
    },
)
async def list_messages(
    session_id: str,
    request: Request,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[CursorPage[MessageOut]]:
    """List the conversation history for a session (paginated, oldest first).

    Query parameters:
        cursor — opaque pagination cursor from the previous response.
        limit  — page size (1–200, default 50).

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND, INVALID_CURSOR.
    """
    await verify_session_owner(session_id, current_user, db)

    # Validate session exists (lightweight; no warm needed for read).
    from src.api.db.repositories.sessions import SessionRepository

    sess_repo = SessionRepository(db)
    record = await sess_repo.get_by_id(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    # The cursor is the message UUID of the last seen item (raw, not base64-encoded).
    after_id: str | None = cursor if cursor is not None else None

    msg_repo = MessageRepository(db)
    rows = await msg_repo.list_by_session(session_id, after_id=after_id, limit=limit)

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    next_cursor = page_rows[-1].id if has_more and page_rows else None

    return APIResponse(
        data=CursorPage(
            items=[_message_to_out(r) for r in page_rows],
            next_cursor=next_cursor,
            has_more=has_more,
            total=None,
        )
    )


@router.delete(
    "/sessions/{session_id}/messages",
    summary="Clear conversation history",
    description=(
        "Clear the conversation history for a session. "
        "Optionally keep the last N messages with keep_last. "
        "Memory summaries and vector index are also reset unless keep_last > 0."
    ),
    response_model=APIResponse[None],
    responses={
        200: {"description": "History cleared."},
        401: {"description": "Not authenticated."},
        403: {"description": "Forbidden (FORBIDDEN)."},
        404: {"description": "Session not found (SESSION_NOT_FOUND)."},
    },
)
async def clear_history(
    session_id: str,
    request: Request,
    body: ClearHistoryRequest | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[None]:
    """Clear or trim conversation history for a session.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, SESSION_NOT_FOUND, MEMORY_CLEAR_FAILED.
    """
    await verify_session_owner(session_id, current_user, db)

    from src.api.db.repositories.sessions import SessionRepository

    sess_repo = SessionRepository(db)
    record = await sess_repo.get_by_id(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "The requested session does not exist.",
            },
        )

    keep_last = (body.keep_last if body is not None else None) or 0

    msg_repo = MessageRepository(db)

    if keep_last > 0:
        # Bulk delete: fetch only message IDs (not full rows), then issue a
        # single DELETE … WHERE id IN (…) rather than a per-row loop.  This
        # avoids loading large message payloads into memory for large sessions.
        from sqlalchemy import delete as _delete
        from sqlalchemy import select as _select

        from src.api.db.models import Message

        id_rows = await db.execute(
            _select(Message.id)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        all_ids = [r[0] for r in id_rows.all()]
        ids_to_delete = all_ids[: max(0, len(all_ids) - keep_last)]
        if ids_to_delete:
            await db.execute(_delete(Message).where(Message.id.in_(ids_to_delete)))
            await db.flush()
    else:
        await msg_repo.delete_by_session(session_id)

    await db.commit()

    # Clear or trim in-memory memory manager if session is warm.
    # clear() joins the background summarization thread and unlinks files — run in thread.
    registry = getattr(request.app.state, "session_registry", None)
    if registry is not None:
        sess = await registry.get_cached(session_id)
        if sess is not None and sess.memory_manager is not None:
            mm = sess.memory_manager
            try:
                if keep_last == 0:
                    await asyncio.to_thread(mm.clear)
                elif hasattr(mm, "trim"):
                    await asyncio.to_thread(mm.trim, keep_last)
            except Exception as exc:
                log.warning("Memory clear/trim failed for session %s: %s", session_id, exc)

    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


ws_router = APIRouter(tags=["WebSocket"])


@ws_router.websocket("/ws/v1/sessions/{session_id}")
async def session_websocket(
    session_id: str,
    websocket: WebSocket,
    token: str | None = Query(
        default=None,
        description="JWT bearer token (query param fallback for browser WebSocket API).",
    ),
    last_seq: int | None = Query(
        default=None,
        description="Last sequence number received; triggers replay of buffered messages.",
    ),
) -> None:
    """WebSocket endpoint for streaming agent output.

    Connection lifecycle:
        1. Client connects with Authorization header OR ?token= query param.
        2. Server validates the JWT and session ownership.
           On failure, server sends close code 4001 (unauthorized) or 4003 (forbidden).
        3. Server registers the connection and sends the first agent_state message.
        4. Client and server exchange messages until the turn completes.
        5. Server sends 'done' message and may keep the connection open for the next turn.
        6. Server closes the connection after 90 s of client silence (no ping received).

    Reconnection:
        If the connection is dropped mid-stream, the client should reconnect using
        the last received seq value. The server will resend any buffered messages
        with seq > client_last_seq (buffer is kept for 30 s post-disconnect).
    """
    # 1. Accept the WebSocket connection so we can send close codes.
    await websocket.accept()

    # 2. Validate JWT from header or query param.
    raw_token: str | None = token
    if raw_token is None:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            raw_token = auth_header[7:]

    if raw_token is None:
        await websocket.close(code=4001, reason="Missing bearer token")
        return

    try:
        claims = _decode_jwt(raw_token)
    except HTTPException:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    user_id: str = claims.get("sub", "")
    role: str = claims.get("role", "user")
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token claims")
        return

    current_user = TokenData(user_id=user_id, role=role, raw_claims=claims)

    # 3. Verify session ownership + warm session (single DB session).
    registry = getattr(websocket.app.state, "session_registry", None)
    if registry is None:
        await websocket.close(code=4000, reason="Session registry unavailable")
        return

    async with AsyncSessionLocal() as db:
        try:
            await verify_session_owner(session_id, current_user, db)
        except HTTPException as exc:
            code = 4003 if exc.status_code == status.HTTP_403_FORBIDDEN else 4004
            await websocket.close(code=code, reason="Session access denied")
            return

        sess = await registry.get_or_warm(session_id, db)

    if sess is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    # 5. Register the WebSocket connection.
    # Cancel any drain task from a previous connection for this session so
    # only one task reads from sess.ws_queue at a time.
    if sess.drain_task is not None and not sess.drain_task.done():
        sess.drain_task.cancel()
        try:
            await sess.drain_task
        except (asyncio.CancelledError, Exception):
            pass
    await manager.connect(session_id, websocket)

    # Replay missed messages if client provides last_seq.
    if last_seq is not None:
        await manager.replay_missed(session_id, last_seq)

    # Send current agent state.
    await manager.send(session_id, "agent_state", {"state": sess.agent_state})

    # 6. Spawn drain task: reads from ws_queue and forwards to client.
    async def _drain() -> None:
        while True:
            try:
                item = await sess.ws_queue.get()
                msg_type = item.get("type", "error")
                payload = item.get("payload", {})
                await manager.send(session_id, msg_type, payload)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("WS drain error for session %s: %s", session_id, exc)
                break

    drain_task = asyncio.create_task(_drain(), name=f"ws-drain-{session_id}")
    sess.drain_task = drain_task

    # 7. Receive loop.
    idle_timeout = 90.0  # seconds
    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=idle_timeout)
            except TimeoutError:
                log.info("WebSocket idle timeout for session %s", session_id)
                break
            except WebSocketDisconnect:
                break

            try:
                data = json.loads(raw)
                client_msg = ClientMessage(**data)
            except Exception as exc:
                log.debug("Invalid client message for session %s: %s", session_id, exc)
                continue

            if client_msg.type == "ping":
                await manager.send(session_id, "pong", {})

            elif client_msg.type == "user_message":
                text = client_msg.payload.get("text", "").strip()
                mode = client_msg.payload.get("mode", "normal")
                if not text:
                    continue

                # Atomically check-and-set turn_task under turn_lock to prevent
                # a race where two concurrent WS messages both see turn_task as None.
                async with sess.turn_lock:
                    if sess.turn_task is not None and not sess.turn_task.done():
                        await manager.send(
                            session_id,
                            "error",
                            {
                                "code": "TURN_IN_PROGRESS",
                                "message": "An agent turn is already running.",
                            },
                        )
                        continue

                    # Persist user message to DB.
                    try:
                        async with AsyncSessionLocal() as db:
                            msg_repo = MessageRepository(db)
                            await msg_repo.create(
                                session_id=session_id,
                                role="user",
                                content_json=json.dumps({"text": text}),
                            )
                            await db.commit()
                    except Exception as exc:
                        log.warning("Could not persist WS user message: %s", exc)

                    async def _run_turn(t: str = text, m: str = mode) -> None:
                        async with AsyncSessionLocal() as turn_db:
                            await run_message_turn(
                                session=sess,
                                text=t,
                                mode=m,
                                db=turn_db,
                                app_state=websocket.app.state,
                            )

                    sess.turn_task = asyncio.create_task(_run_turn(), name=f"turn-ws-{session_id}")

            elif client_msg.type == "tool_confirm":
                confirmation_id = client_msg.payload.get("confirmation_id", "")
                action = client_msg.payload.get("action", "deny")

                # Route to the per-turn ApiConfirmationUI published on the session by
                # turn_runner at the start of each agent turn (BUG-FORGE-001).
                # sess.run_config holds the session-level template (no confirmation_ui);
                # the live UI is always on sess.active_confirmation_ui.
                confirmation_ui = getattr(sess, "active_confirmation_ui", None)
                if confirmation_ui is not None and hasattr(confirmation_ui, "resolve"):
                    confirmation_ui.resolve(confirmation_id, action)

            elif client_msg.type == "cancel":
                if sess.turn_task is not None and not sess.turn_task.done():
                    sess.cancel_event.set()
                    # Unblock any pending confirmation so the agent thread
                    # exits promptly instead of waiting up to 5 minutes.
                    _conf_ui = getattr(sess, "active_confirmation_ui", None)
                    if _conf_ui is not None and hasattr(_conf_ui, "cancel"):
                        _conf_ui.cancel()
                    sess.turn_task.cancel()
                    try:
                        await sess.turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    finally:
                        sess.cancel_event.clear()

    except Exception as exc:
        log.debug("WebSocket receive loop error for session %s: %s", session_id, exc)
    finally:
        drain_task.cancel()
        try:
            await drain_task
        except (asyncio.CancelledError, Exception):
            pass
        # Clear the session's drain_task reference if it still points to ours.
        if sess.drain_task is drain_task:
            sess.drain_task = None
        await manager.disconnect(session_id)

        # Auto-delete empty sessions (0 messages) on disconnect.
        try:
            async with AsyncSessionLocal() as cleanup_db:
                from src.api.db.repositories.sessions import SessionRepository

                msg_repo = MessageRepository(cleanup_db)
                rows = await msg_repo.list_by_session(session_id, limit=1)
                if not rows:
                    sess_repo = SessionRepository(cleanup_db)
                    await sess_repo.archive(session_id)
                    await cleanup_db.commit()
                    if registry is not None:
                        await registry.remove(session_id)
                    log.info("Auto-archived empty session %s on disconnect", session_id)
        except Exception as exc:
            log.debug("Auto-delete check failed for session %s: %s", session_id, exc)
