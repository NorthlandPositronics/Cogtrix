"""Health endpoints — unauthenticated liveness and readiness checks.

These endpoints do not require a bearer token so that load balancers and
orchestrators can probe the service without credentials.

Endpoints:
    GET /api/v1/health          — liveness check (always 200 when the process is alive)
    GET /api/v1/health/ready    — readiness check (200 when database ready, 503 otherwise)
    GET /api/v1/health/ready-full — full readiness check (200 when DB + Redis + tool registry ready, 503 otherwise)

Notes:
- /ready is intended for Kubernetes readiness probe (database is the critical dependency)
- /ready-full is intended for manual ops checks (DB + Redis + tool_registry)
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
        "Returns HTTP 200 when the database is reachable, HTTP 503 otherwise. "
        "Redis and tool registry are NOT checked — the API can serve requests "
        "without them (they will be loaded lazily on first use). "
        "No authentication required. "
        "Use this for Kubernetes readiness probes."
    ),
    response_model=APIResponse[ReadinessOut],
    responses={
        200: {"description": "Database is ready."},
        503: {"description": "Database connection failed."},
    },
)
async def readiness(request: Request, response: Response) -> APIResponse[ReadinessOut]:
    """Return readiness status for the database component only.

    This endpoint is intended for Kubernetes readiness probes. It returns 200
    when the database is reachable, allowing traffic to be routed even if Redis
    or the tool registry are not yet initialized (both happen lazily on first use).

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

    # Redis and tool_registry are intentionally NOT checked in /ready:
    # - Redis connection depends on config and network
    # - Tool registry initialization depends on API keys
    # - The API can serve requests without either (lazy loading)
    # - Kubernetes readiness probes should only depend on critical dependencies

    all_ready = all(c.ok for c in components)
    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return APIResponse(data=ReadinessOut(ready=all_ready, components=components))


@router.get(
    "/ready-full",
    summary="Full readiness check",
    description=(
        "Returns HTTP 200 when database, Redis, and tool registry are all ready, "
        "HTTP 503 otherwise. "
        "Checks: database connectivity, Redis connectivity, tool registry initialization. "
        "No authentication required. "
        "Use this for manual ops checks or debugging."
    ),
    response_model=APIResponse[ReadinessOut],
    responses={
        200: {"description": "All components ready."},
        503: {"description": "One or more components not ready."},
    },
)
async def readiness_full(request: Request, response: Response) -> APIResponse[ReadinessOut]:
    """Return full readiness status including database, Redis, and tool registry.

    This endpoint checks all three components:
    - Database connectivity
    - Redis connectivity (if configured)
    - Tool registry initialization

    Use this for:
    - Manual ops validation
    - Debugging connectivity issues
    - Pre-flight checks before submitting agent jobs

    Note: For Kubernetes readiness probes, use /api/v1/health/ready instead.

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

    # Check Redis connectivity if configured
    redis_ok = True
    redis_latency_ms: int | None = None
    redis_detail: str | None = None
    try:
        from src.api.redis_sessions import _store

        if _store is None or not _store.is_configured:
            redis_detail = "redis not configured"
        else:
            client = _store._client
            if client is None:
                redis_ok = False
                redis_detail = "redis client not initialized"
            else:
                import asyncio

                t0 = time.monotonic()
                try:
                    await asyncio.wait_for(client.ping(), timeout=2.0)
                    redis_latency_ms = int((time.monotonic() - t0) * 1000)
                    redis_ok = True
                    redis_detail = "ok"
                except Exception as exc:
                    redis_detail = str(exc)[:256]
    except Exception as exc:
        redis_ok = False
        redis_detail = str(exc)[:256]

    components.append(
        ReadinessComponentStatus(
            name="redis",
            ok=redis_ok,
            latency_ms=redis_latency_ms,
            detail=redis_detail,
        )
    )

    # Check tool registry if available on app.state
    tool_registry = getattr(request.app.state, "tool_registry", None)
    tools_ok = False
    tools_latency_ms: int | None = None
    tools_detail: str | None = None
    if tool_registry is None:
        tools_detail = "tool_registry not initialized"
    else:
        tools_detail = "ok"
        tools_ok = True
        tools_latency_ms = 0  # Tool registry is already initialized

    components.append(
        ReadinessComponentStatus(
            name="tool_registry",
            ok=tools_ok,
            latency_ms=tools_latency_ms,
            detail=tools_detail,
        )
    )

    all_ready = all(c.ok for c in components)
    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return APIResponse(data=ReadinessOut(ready=all_ready, components=components))
