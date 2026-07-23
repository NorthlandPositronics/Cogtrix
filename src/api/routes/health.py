"""Health endpoints — unauthenticated liveness and readiness checks.

These endpoints do not require a bearer token so that load balancers and
orchestrators can probe the service without credentials.

Endpoints:
    GET /api/v1/health          — liveness check (always 200 when the process is alive)
    GET /api/v1/health/ready    — readiness check (200 when all components ready, 503 otherwise)
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response, status

from src.api.schemas.common import APIResponse
from src.api.schemas.system import HealthOut, ReadinessComponentStatus, ReadinessOut

log = logging.getLogger("cogtrix.api.health")

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    summary="Liveness check",
    description=(
        "Returns HTTP 200 as long as the API process is alive. "
        "No authentication required. "
        "Suitable for load balancer health probes."
    ),
    response_model=APIResponse[HealthOut],
    responses={200: {"description": "Process is alive."}},
)
async def liveness() -> APIResponse[HealthOut]:
    """Return HTTP 200 to confirm the process is alive.

    Auth: none.
    Error codes: none (always 200 when reachable).
    """
    return APIResponse(data=HealthOut(timestamp=datetime.now(UTC)))


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Returns HTTP 200 when all critical components are healthy, "
        "HTTP 503 when one or more are not. "
        "Checks: database connectivity. "
        "No authentication required."
    ),
    response_model=APIResponse[ReadinessOut],
    responses={
        200: {"description": "All components ready."},
        503: {"description": "One or more components not ready."},
    },
)
async def readiness(request: Request, response: Response) -> APIResponse[ReadinessOut]:
    """Return readiness status for all critical system components.

    Auth: none.
    Error codes: none (503 is not an error envelope — it carries ReadinessOut with ready=False).
    """
    components: list[ReadinessComponentStatus] = []

    # Check database connectivity
    db_ok = False
    db_latency_ms: int | None = None
    db_detail: str | None = None
    try:
        from sqlalchemy import text

        from src.api.db.engine import engine

        t0 = time.monotonic()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_latency_ms = int((time.monotonic() - t0) * 1000)
        db_ok = True
        db_detail = "ok"
    except Exception as exc:
        db_detail = str(exc)[:256]

    components.append(
        ReadinessComponentStatus(
            name="database",
            ok=db_ok,
            latency_ms=db_latency_ms,
            detail=db_detail,
        )
    )

    # Check tool registry if available on app.state
    tools_ok = True
    tools_detail = "ok"
    try:
        tool_registry = getattr(request.app.state, "tool_registry", None)
        if tool_registry is None:
            tools_ok = False
            tools_detail = "tool_registry not initialized"
    except Exception as exc:
        tools_ok = False
        tools_detail = str(exc)[:256]

    components.append(
        ReadinessComponentStatus(
            name="tool_registry",
            ok=tools_ok,
            latency_ms=None,
            detail=tools_detail,
        )
    )

    all_ready = all(c.ok for c in components)
    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return APIResponse(data=ReadinessOut(ready=all_ready, components=components))
