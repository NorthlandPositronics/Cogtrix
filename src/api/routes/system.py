"""System and observability endpoints.

Endpoints:
    GET  /api/v1/system/info         — version, platform, uptime
    POST /api/v1/system/debug        — toggle debug/verbose logging (admin)

WebSocket:
    WS   /ws/v1/logs                 — live structured log stream (admin)
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect

from src.api.auth import TokenData, _decode_jwt, get_current_user, require_admin
from src.api.schemas.common import APIResponse
from src.api.schemas.system import DebugToggleRequest, SystemInfoOut

log = logging.getLogger("cogtrix.api.system")

router = APIRouter(prefix="/system", tags=["System"])
ws_router = APIRouter(tags=["WebSocket"])

# Module-level startup timestamp (set once at import time)
_STARTUP_TIME = time.monotonic()
_STARTUP_DT = datetime.now(UTC)


def _make_system_info(request: Request) -> SystemInfoOut:
    cfg = getattr(request.app.state, "config", None)
    version = "unknown"
    try:
        from src._version import __version__

        version = __version__
    except Exception:
        pass

    uptime = time.monotonic() - _STARTUP_TIME
    return SystemInfoOut(
        version=version,
        api_version="v1",
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        debug=bool(getattr(cfg, "debug", False) if cfg else False),
        verbose=bool(getattr(cfg, "verbose", False) if cfg else False),
        uptime_s=uptime,
        started_at=_STARTUP_DT,
    )


@router.get(
    "/info",
    summary="System information",
    description="Return Cogtrix version, API version, platform details, and server uptime.",
    response_model=APIResponse[SystemInfoOut],
    responses={
        200: {"description": "System info returned."},
        401: {"description": "Not authenticated."},
    },
)
async def system_info(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[SystemInfoOut]:
    """Return server version, platform, and uptime.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    return APIResponse(data=_make_system_info(request))


@router.post(
    "/debug",
    summary="Toggle debug or verbose logging",
    description="Enable or disable debug and verbose logging at runtime. Admin only.",
    response_model=APIResponse[SystemInfoOut],
    responses={
        200: {"description": "Log level updated; system info returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def toggle_debug(
    body: DebugToggleRequest,
    request: Request,
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[SystemInfoOut]:
    """Toggle debug/verbose logging at runtime (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    cfg = getattr(request.app.state, "config", None)
    level = logging.DEBUG if body.debug else logging.INFO
    logging.getLogger("cogtrix").setLevel(level)
    if cfg is not None:
        cfg.debug = body.debug
        if body.verbose is not None:
            cfg.verbose = body.verbose
    return APIResponse(data=_make_system_info(request))


# ---------------------------------------------------------------------------
# WebSocket log stream
# ---------------------------------------------------------------------------


class _WSLogHandler(logging.Handler):
    """Logging handler that forwards records into an asyncio.Queue."""

    def __init__(
        self, queue: asyncio.Queue[Any], min_level: int, loop: asyncio.AbstractEventLoop
    ) -> None:
        super().__init__(min_level)
        self._queue = queue
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = {
                "type": "log_line",
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            }

            def _safe_put() -> None:
                try:
                    self._queue.put_nowait(msg)
                except asyncio.QueueFull:
                    pass  # drop the record rather than crashing the event loop

            self._loop.call_soon_threadsafe(_safe_put)
        except Exception:
            pass


@ws_router.websocket("/ws/v1/logs")
async def log_stream_websocket(
    websocket: WebSocket,
    token: str | None = Query(default=None, description="JWT bearer token."),
    level: str = Query(
        default="INFO", description="Minimum log level to stream: DEBUG, INFO, WARNING, ERROR."
    ),
) -> None:
    """WebSocket endpoint for live structured log streaming.

    Streams log records as they are emitted by the logging subsystem.
    Each message has type='log_line' with the LogLinePayload schema.

    Connection lifecycle:
        1. Client connects with Authorization header or ?token= query parameter.
        2. Server validates the JWT (admin role required).
        3. Server attaches a logging handler that forwards records to the WebSocket.
        4. Client receives log_line messages with level, logger, message, and timestamp.
        5. Client sends ping every 30 s; server responds with pong.
        6. Connection closes when the client disconnects or after 90 s of silence.

    Auth: admin bearer token in Authorization header or ?token= query parameter.
    Close codes:
        4001 — unauthorized (no token or invalid signature).
        4003 — forbidden (not admin).
    """
    # Resolve token from query param or Authorization header
    raw_token = token
    if raw_token is None:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            raw_token = auth_header[7:]

    await websocket.accept()

    if raw_token is None:
        await websocket.close(code=4001)
        return

    try:
        claims = _decode_jwt(raw_token)
    except Exception:
        await websocket.close(code=4001)
        return

    if claims.get("role") != "admin":
        await websocket.close(code=4003)
        return

    # Determine numeric log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)
    loop = asyncio.get_running_loop()
    handler = _WSLogHandler(queue, numeric_level, loop)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    async def _drain() -> None:
        while True:
            msg = await queue.get()
            try:
                await websocket.send_text(json.dumps(msg))
            except Exception:
                break

    drain_task = asyncio.create_task(_drain())
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=90.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except TimeoutError:
                # 90 s silence — close
                break
            except WebSocketDisconnect:
                break
    finally:
        drain_task.cancel()
        root_logger.removeHandler(handler)
        try:
            await websocket.close()
        except Exception:
            pass
