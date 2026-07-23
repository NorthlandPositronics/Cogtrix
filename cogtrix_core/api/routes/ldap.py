"""LDAP/AD sync routes (Enterprise Phase 1 — task 1.2.3).

Endpoints:
    GET  /api/v1/ldap/status   — check LDAP configuration and connectivity
    POST /api/v1/ldap/sync     — trigger a user sync run

Admin-only. Requires the ``[ldap]`` optional extra for actual sync;
status returns 503 LDAP_NOT_INSTALLED when ldap3 is absent.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.auth import TokenData, require_admin
from cogtrix_core.api.db.engine import get_db
from cogtrix_core.api.ldap.config import get_ldap_config, is_ldap_configured
from cogtrix_core.api.schemas.common import APIResponse

log = logging.getLogger("cogtrix.api.ldap")

router = APIRouter(prefix="/ldap", tags=["LDAP / AD Sync"])


@router.get(
    "/status",
    response_model=APIResponse[dict],
    summary="LDAP configuration status",
)
async def ldap_status(_: TokenData = Depends(require_admin)) -> APIResponse[dict]:
    """Return LDAP configuration presence and library availability."""
    try:
        import ldap3  # type: ignore[import]  # noqa: F401

        ldap3_available = True
    except ImportError:
        ldap3_available = False

    config = get_ldap_config()
    return APIResponse(
        data={
            "configured": is_ldap_configured(),
            "ldap3_installed": ldap3_available,
            "server_url": config.server_url if config else None,
            "search_base": config.search_base if config else None,
        }
    )


@router.post(
    "/sync",
    response_model=APIResponse[dict],
    summary="Trigger LDAP user sync",
    status_code=200,
)
async def ldap_sync(
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Connect to the LDAP server and provision/update users in Cogtrix."""
    config = get_ldap_config()
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "LDAP_NOT_CONFIGURED", "message": "LDAP sync is not configured."},
        )

    try:
        from cogtrix_core.api.ldap.sync import sync_users
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "LDAP_NOT_INSTALLED",
                "message": "LDAP sync requires pip install cogtrix[ldap].",
            },
        ) from None

    result = await sync_users(config, db)
    return APIResponse(
        data={
            "added": result.added,
            "updated": result.updated,
            "skipped": result.skipped,
            "total_processed": result.total_processed,
            "errors": result.errors,
            "success": result.success,
        }
    )
