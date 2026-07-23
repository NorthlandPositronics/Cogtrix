"""Admin endpoints (admin role required).

Endpoints:
    GET /api/v1/admin/orgs   — list all organizations (paginated, filterable)
    GET /api/v1/admin/stats  — global system statistics (admin)
    GET /api/v1/admin/system — superadmin-only global system stats with DB/Redis health
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import (
    TokenData,
    create_impersonation_token,
    get_current_user,
    require_admin,
    require_superadmin,
)
from src.api.db import get_db
from src.api.db.models import AuditLogEntry, ImpersonationSession, UsageRecord
from src.api.db.repositories.organization import OrganizationRepository
from src.api.db.repositories.sessions import SessionRepository
from src.api.db.repositories.usage import UsageRepository
from src.api.db.repositories.users import UserRepository
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.organization import (
    AdminStats,
    ImpersonateRequest,
    ImpersonateResponse,
    OrgAuditLog,
    OrgSummary,
    OrgUsage,
)
from src.api.schemas.system import SystemStats

log = logging.getLogger("cogtrix.api.admin")

router = APIRouter(prefix="/admin", tags=["Admin"])


def _get_mcp_server_count(request: Request) -> int:
    """Return the number of configured MCP servers from app state config."""
    cfg: Any = getattr(request.app.state, "config", None)
    if cfg is None:
        return 0
    servers: dict[str, Any] = dict(getattr(cfg, "mcp_servers", {}) or {})
    return len(servers)


# ---------------------------------------------------------------------------
# Cursor helpers
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


# ---------------------------------------------------------------------------
# Organization list
# ---------------------------------------------------------------------------


@router.get(
    "/orgs",
    summary="List all organizations",
    description=(
        "Return all organizations with cursor-based pagination and optional filters. "
        "Superadmin only."
    ),
    response_model=APIResponse[CursorPage[OrgSummary]],
    responses={
        200: {"description": "Organization list returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Superadmin required (FORBIDDEN)."},
        400: {"description": "Invalid cursor (INVALID_CURSOR)."},
    },
)
async def list_orgs(
    cursor: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    name: str | None = None,
    status: str | None = None,
    plan: str | None = None,
    created_after: str | None = Query(
        default=None,
        description="ISO 8601 datetime filter (e.g. 2026-01-01T00:00:00). Return orgs created on or after this date.",
    ),
    current_user: TokenData = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[CursorPage[OrgSummary]]:
    """List all organizations (paginated, filterable). Superadmin only.

    Query parameters:
        cursor       — opaque pagination cursor from the previous response.
        limit        — page size (1–100, default 20).
        name         — substring filter on organization name (case-insensitive).
        status       — exact filter on organization status (active/inactive/suspended).
        plan         — exact filter on subscription plan.
        created_after — ISO 8601 datetime; return orgs created on or after this date.

    Auth: superadmin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, INVALID_CURSOR.
    """
    after_id = _decode_cursor(cursor) if cursor else None

    created_after_dt: datetime | None = None
    if created_after:
        created_after_dt = datetime.fromisoformat(created_after)

    repo = OrganizationRepository(db)
    rows = await repo.list_orgs(
        after_id=after_id,
        limit=limit,
        name_filter=name,
        status_filter=status,
        plan_filter=plan,
        created_after=created_after_dt,
    )

    has_more = len(rows) > limit
    page_rows = rows[:limit]

    # Bulk-fetch member counts
    org_ids = [r.id for r in page_rows]
    counts = await repo.count_users_per_org(org_ids)

    items: list[OrgSummary] = []
    for r in page_rows:
        items.append(
            OrgSummary(
                id=r.id,
                name=r.name,
                slug=r.slug,
                status=r.status,
                plan=r.plan,
                member_count=counts.get(r.id, 0),
                created_at=r.created_at,
            )
        )

    next_cursor = None
    if has_more and page_rows:
        next_cursor = _encode_cursor(page_rows[-1].id)

    total = await repo.count_orgs(
        name_filter=name,
        status_filter=status,
        plan_filter=plan,
        created_after=created_after_dt,
    )

    page: CursorPage[OrgSummary] = CursorPage(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=total,
    )
    return APIResponse(data=page)


# ---------------------------------------------------------------------------
# Global system stats
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    summary="Global system statistics",
    description=(
        "Return high-level system counters: total organizations, active sessions, "
        "total users, and configured MCP servers. Admin only."
    ),
    response_model=APIResponse[AdminStats],
    responses={
        200: {"description": "Statistics returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def get_stats(
    request: Request,
    current_user: TokenData = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[AdminStats]:
    """Return global system statistics. Admin only.

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    org_repo = OrganizationRepository(db)
    session_repo = SessionRepository(db)
    user_repo = UserRepository(db)

    total_orgs = await org_repo.count_orgs()
    active_sessions = await session_repo.count_all_active()
    total_users = await user_repo.count_all()
    mcp_server_count = _get_mcp_server_count(request)

    stats = AdminStats(
        total_orgs=total_orgs,
        active_sessions=active_sessions,
        total_users=total_users,
        mcp_server_count=mcp_server_count,
    )
    return APIResponse(data=stats)


# ---------------------------------------------------------------------------
# Org-level usage
# ---------------------------------------------------------------------------


@router.get(
    "/orgs/{org_id}/usage",
    summary="Organization usage metrics",
    description=(
        "Return aggregated usage metrics for a single organization. "
        "Optionally filter by date range (from, to). Admin only."
    ),
    response_model=APIResponse[OrgUsage],
    responses={
        200: {"description": "Usage metrics returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def get_org_usage(
    org_id: str,
    from_date: str | None = Query(default=None, alias="from"),
    to_date: str | None = Query(default=None, alias="to"),
    current_user: TokenData = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[OrgUsage]:
    """Return aggregated usage metrics for an organization. Admin only.

    Query parameters:
        from — start date (ISO 8601, e.g. 2026-01-01).
        to   — end date (ISO 8601, inclusive).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    from datetime import datetime as _dt

    from_dt = None
    to_dt = None
    if from_date:
        from_dt = _dt.fromisoformat(from_date)
    if to_date:
        to_dt = _dt.fromisoformat(to_date)

    repo = UsageRepository(db)
    totals = await repo.aggregate_by_org(org_id, from_date=from_dt, to_date=to_dt)

    usage = OrgUsage(
        org_id=org_id,
        from_date=from_date,
        to_date=to_date,
        total_api_calls=totals.get("api_call", 0),
        total_sessions=totals.get("session_created", 0),
        total_users_provisioned=totals.get("user_provisioned", 0),
        total_storage_kb=totals.get("storage_write_kb", 0),
        total_workspaces=totals.get("workspace_created", 0),
    )
    return APIResponse(data=usage)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


async def _log_audit_action(
    db: AsyncSession,
    *,
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    org_id: str | None = None,
    impersonated_by: str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLogEntry:
    """Insert an immutable audit log entry and return it."""
    entry = AuditLogEntry(
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        org_id=org_id,
        impersonated_by=impersonated_by,
        details=json.dumps(details) if details is not None else None,
    )
    db.add(entry)
    await db.flush()
    await db.refresh(entry)
    return entry


# ---------------------------------------------------------------------------
# Impersonation
# ---------------------------------------------------------------------------


@router.post(
    "/orgs/{org_id}/impersonate",
    summary="Start superadmin impersonation session",
    description=(
        "Allows a superadmin to impersonate an organization member for debugging "
        "and support. Returns an impersonation JWT with a 30-minute default expiry. "
        "Cannot be chained."
    ),
    response_model=APIResponse[ImpersonateResponse],
    responses={
        200: {"description": "Impersonation session created."},
        401: {"description": "Not authenticated."},
        403: {"description": "Superadmin required (FORBIDDEN)."},
        404: {"description": "Organization or user not found."},
        409: {"description": "Already impersonating or target user invalid."},
    },
)
async def start_impersonation(
    org_id: str,
    body: ImpersonateRequest,
    current_user: TokenData = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ImpersonateResponse]:
    """Start an impersonation session. Superadmin only.

    Auth: superadmin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND,
                  ALREADY_IMPERSONATING.
    """
    # Prevent chaining: reject if superadmin already has an active session.
    active_result = await db.execute(
        select(ImpersonationSession).where(
            ImpersonationSession.superadmin_id == current_user.user_id,
            ImpersonationSession.ended_at.is_(None),
        )
    )
    if active_result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "ALREADY_IMPERSONATING",
                "message": "An active impersonation session already exists.",
            },
        )

    # Verify organization exists.
    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(org_id)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": "The requested organization does not exist.",
            },
        )

    # Verify target user exists and belongs to the org.
    user_repo = UserRepository(db)
    target_user = await user_repo.get_by_id_and_org(body.user_id, org_id)
    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": "The requested user does not exist in this organization.",
            },
        )

    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=body.duration_minutes)

    imp_session = ImpersonationSession(
        superadmin_id=current_user.user_id,
        impersonated_user_id=target_user.id,
        org_id=org_id,
        reason=body.reason,
        started_at=now,
        expires_at=expires_at,
    )
    db.add(imp_session)
    await db.flush()
    await db.refresh(imp_session)

    # Audit log: impersonation started.
    await _log_audit_action(
        db,
        actor_id=current_user.user_id,
        impersonated_by=None,
        action="impersonation.start",
        resource_type="impersonation_session",
        resource_id=imp_session.id,
        org_id=org_id,
        details={"reason": body.reason, "duration_minutes": body.duration_minutes},
    )
    await db.commit()

    token = create_impersonation_token(
        impersonated_user_id=target_user.id,
        impersonated_role=target_user.role,
        superadmin_id=current_user.user_id,
        session_id=imp_session.id,
        duration_minutes=body.duration_minutes,
    )

    return APIResponse(
        data=ImpersonateResponse(
            impersonation_token=token,
            expires_at=expires_at,
            impersonated_user_id=target_user.id,
            org_id=org_id,
        )
    )


@router.delete(
    "/impersonate",
    summary="Stop active impersonation session",
    description=(
        "Ends the current user's active impersonation session. "
        "Callable by the superadmin who started it or while carrying the "
        "impersonation token."
    ),
    response_model=APIResponse[dict[str, str]],
    responses={
        200: {"description": "Impersonation session ended."},
        401: {"description": "Not authenticated."},
        404: {"description": "No active impersonation session found."},
    },
)
async def stop_impersonation(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[dict[str, str]]:
    """Stop the active impersonation session.

    If the caller is carrying an impersonation token, the session matching
    ``impersonation_session_id`` in the JWT is ended.  Otherwise, any active
    session started by the caller (as superadmin) is ended.

    Auth: any authenticated user (admin or impersonation token).
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, SESSION_NOT_FOUND.
    """
    imp_session_id: str | None = current_user.raw_claims.get("impersonation_session_id")

    if imp_session_id is not None:
        result = await db.execute(
            select(ImpersonationSession).where(ImpersonationSession.id == imp_session_id)
        )
        imp_session = result.scalar_one_or_none()
    else:
        result = await db.execute(
            select(ImpersonationSession).where(
                ImpersonationSession.superadmin_id == current_user.user_id,
                ImpersonationSession.ended_at.is_(None),
            )
        )
        imp_session = result.scalar_one_or_none()

    if imp_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": "No active impersonation session found.",
            },
        )

    imp_session.ended_at = datetime.now(UTC)
    await db.flush()

    # Audit log: impersonation ended.
    if imp_session_id is not None:
        actor_id = current_user.user_id
        impersonated_by = imp_session.superadmin_id
    else:
        actor_id = current_user.user_id
        impersonated_by = None
    await _log_audit_action(
        db,
        actor_id=actor_id,
        impersonated_by=impersonated_by,
        action="impersonation.end",
        resource_type="impersonation_session",
        resource_id=imp_session.id,
        org_id=imp_session.org_id,
    )
    await db.commit()

    return APIResponse(data={"status": "ended"})


# ---------------------------------------------------------------------------
# Org-level audit log (DB-backed)
# ---------------------------------------------------------------------------


@router.get(
    "/orgs/{org_id}/audit",
    summary="Organization audit log",
    description=("Return DB-backed audit log entries for a single organization. Admin only."),
    response_model=APIResponse[OrgAuditLog],
    responses={
        200: {"description": "Audit log returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def get_org_audit(
    org_id: str,
    current_user: TokenData = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[OrgAuditLog]:
    """Return audit log entries for an organization. Admin only.

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    result = await db.execute(
        select(AuditLogEntry)
        .where(AuditLogEntry.org_id == org_id)
        .order_by(AuditLogEntry.created_at.desc())
    )
    rows = result.scalars().all()

    entries: list[dict[str, Any]] = []
    for row in rows:
        detail: dict[str, Any] | None = None
        if row.details is not None:
            try:
                detail = json.loads(row.details)
            except Exception:
                detail = {"raw": row.details}
        entries.append(
            {
                "id": row.id,
                "actor_id": row.actor_id,
                "impersonated_by": row.impersonated_by,
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "details": detail,
                "created_at": row.created_at.isoformat(),
            }
        )

    return APIResponse(
        data=OrgAuditLog(entries=entries, note="Audit log query not yet implemented")
    )


# ---------------------------------------------------------------------------
# In-memory cache for system stats (60 seconds)
# ---------------------------------------------------------------------------


class _SystemStatsCache:
    """Simple in-memory cache with TTL for system stats."""

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._data: SystemStats | None = None
        self._timestamp: float | None = None
        self._ttl = ttl_seconds

    def get(self) -> tuple[SystemStats | None, float | None]:
        """Return cached data and timestamp if not expired."""
        if self._data is None or self._timestamp is None:
            return None, None
        if time.monotonic() - self._timestamp > self._ttl:
            return None, None
        return self._data, self._timestamp

    def set(self, data: SystemStats) -> None:
        """Store data with current timestamp."""
        self._data = data
        self._timestamp = time.monotonic()


_cache = _SystemStatsCache(ttl_seconds=60)
_cache_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Superadmin system statistics
# ---------------------------------------------------------------------------


@router.get(
    "/system",
    summary="Superadmin global system statistics",
    description=(
        "Return comprehensive system statistics including DB pool status, "
        "Redis health, usage metrics, and uptime. Superadmin only."
    ),
    response_model=APIResponse[SystemStats],
    responses={
        200: {"description": "System statistics returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Superadmin required (FORBIDDEN)."},
    },
)
async def get_system_stats(
    request: Request,
    current_user: TokenData = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SystemStats]:
    """Return comprehensive system statistics (superadmin only).

    Includes:
        - Organization and user counts
        - Active sessions
        - Token and API usage (24h)
        - Error rate
        - Database pool status
        - Redis connection status
        - Server uptime and version

    Auth: superadmin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    # Check cache first
    cached, cached_time = _cache.get()
    if cached is not None and cached_time is not None:
        # Recalculate uptime for accurate timing in response
        cached_dict = cached.model_dump()
        # Update uptime to current value
        startup_time = getattr(request.app.state, "startup_time", cached_time)
        current_uptime = time.monotonic() - startup_time
        cached_dict["uptime_s"] = current_uptime
        # Update started_at to maintain consistency
        cached_dict["started_at"] = getattr(request.app.state, "started_at", datetime.now(UTC))
        return APIResponse(data=SystemStats(**cached_dict))

    # Cache miss - fetch fresh data
    async with _cache_lock:
        await _refresh_system_stats_cache(request, db)
    cached, _ = _cache.get()
    if cached is None:
        # Fallback: try one more time in case of race condition
        async with _cache_lock:
            await _refresh_system_stats_cache(request, db)
        cached, _ = _cache.get()

    if cached is None:
        # Still no data - this should never happen, but handle gracefully
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Failed to retrieve system statistics.",
            },
        )

    return APIResponse(data=cached)


async def _refresh_system_stats_cache(request: Request, db: AsyncSession) -> None:
    """Fetch fresh system statistics and update cache."""
    from datetime import timedelta

    from src.api.db.repositories.usage import EVENT_API_CALL

    # Initialize start time from app state if not set
    if not hasattr(request.app.state, "startup_time"):
        request.app.state.startup_time = time.monotonic()
        request.app.state.started_at = datetime.now(UTC)

    org_repo = OrganizationRepository(db)
    session_repo = SessionRepository(db)
    user_repo = UserRepository(db)

    total_orgs = await org_repo.count_orgs()
    active_sessions = await session_repo.count_all_active()
    total_users = await user_repo.count_all()

    # Compute API usage for last 24h
    now = datetime.now(UTC)
    day_ago = now - timedelta(hours=24)

    # Count API calls in last 24h using func.sum on quantity
    api_result = await db.execute(
        select(func.coalesce(func.sum(UsageRecord.quantity), 0)).where(
            UsageRecord.event_type == EVENT_API_CALL,
            UsageRecord.recorded_at >= day_ago,
        )
    )
    api_requests_24h = api_result.scalar_one()

    # Estimate token usage from API calls (simplified: assume avg tokens per call)
    # In production, you'd track tokens separately in usage records
    # For now, estimate based on API calls
    # TODO: add token usage tracking as a separate event type
    estimated_token_usage = api_requests_24h * 1000  # rough estimate

    # Count errors (events that indicate errors)
    # We don't have a dedicated error event type yet, so we'll use 0 for now
    # In production, track error events in the usage table
    error_count = 0
    error_rate: float | None = None
    if api_requests_24h > 0:
        error_rate = error_count / api_requests_24h

    # DB pool status
    db_pool_status = "healthy"
    db_pool_size = 0
    db_pool_max = 0
    try:
        # Try to get pool info from engine
        engine = db.get_bind()
        if hasattr(engine, "pool"):
            pool = engine.pool
            db_pool_size = pool.size()
            # Use configured pool size (_pool.maxsize for QueuePool) rather than
            # checked-out count (pool.size()) to avoid fluctuating max under load.
            configured_size = getattr(getattr(pool, "_pool", None), "maxsize", None)
            if configured_size is not None:
                db_pool_max = configured_size + pool.maxoverflow
            else:
                db_pool_max = pool.maxoverflow + pool.size()
            # Check if pool is healthy by running a simple query on a real model
            from src.api.db.models import User

            await db.execute(select(func.count()).select_from(User.__table__))
            db_pool_status = "healthy"
        else:
            db_pool_status = "warning"
    except Exception as exc:
        log.warning("DB pool health check failed: %s", exc)
        db_pool_status = "critical"

    # Redis connection status
    redis_connected = False
    redis_latency_ms = None
    try:
        # Check if Redis is configured and connected
        redis_client = getattr(request.app.state, "redis", None)
        if redis_client is not None:
            # Try a simple ping
            await redis_client.ping()
            redis_connected = True
            # Note: aioredis doesn't have a direct ping method in all versions
            # This is a placeholder for future implementation
    except Exception as exc:
        log.warning("Redis health check failed: %s", exc)
        redis_connected = False

    # Version info
    version = "unknown"
    try:
        from src._version import get_version_string

        version = get_version_string()
    except Exception as exc:
        log.warning("Failed to get version: %s", exc)

    # Uptime and start time
    uptime = time.monotonic() - request.app.state.startup_time
    started_at = request.app.state.started_at

    stats = SystemStats(
        total_orgs=total_orgs,
        total_users=total_users,
        active_sessions=active_sessions,
        estimated_token_usage_24h=estimated_token_usage,
        api_requests_24h=api_requests_24h,
        error_rate_24h=error_rate,
        db_pool_status=db_pool_status,
        db_pool_size=db_pool_size,
        db_pool_max=db_pool_max,
        redis_connected=redis_connected,
        redis_latency_ms=redis_latency_ms,
        uptime_s=uptime,
        version=version,
        started_at=started_at,
    )

    _cache.set(stats)
