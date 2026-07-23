"""JIT provisioning admin routes (Enterprise Phase 1 — task 1.2.5).

Endpoints:
    GET  /api/v1/jit/status   — current JIT configuration summary
    POST /api/v1/jit/test     — dry-run: check whether an email would be allowed
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from cogtrix_core.api.auth import TokenData, require_admin
from cogtrix_core.api.jit.config import JITConfig, get_jit_config
from cogtrix_core.api.schemas.common import APIResponse

log = logging.getLogger("cogtrix.api.jit")

router = APIRouter(prefix="/jit", tags=["JIT Provisioning"])


@router.get(
    "/status",
    response_model=APIResponse[dict],
    summary="JIT provisioning status",
)
async def jit_status(_: TokenData = Depends(require_admin)) -> APIResponse[dict]:
    """Return the active JIT provisioning configuration summary."""
    config: JITConfig | None = get_jit_config()
    if config is None:
        return APIResponse(data={"enabled": False, "configured": False})
    return APIResponse(
        data={
            "configured": True,
            "enabled": config.enabled,
            "allowed_domains": config.allowed_domains,
            "default_role": config.default_role,
            "org_id": config.org_id,
            "auto_team_id": config.auto_team_id,
            "max_users": config.max_users,
            "deactivate_unknown": config.deactivate_unknown,
        }
    )


from pydantic import BaseModel  # noqa: E402 — placed after router to keep imports grouped


class JITEmailCheck(BaseModel):
    email: str


@router.post(
    "/test",
    response_model=APIResponse[dict],
    summary="Test whether an email would be JIT-provisioned",
)
async def jit_test(
    body: JITEmailCheck,
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Dry-run: check whether a given email would pass JIT domain validation."""
    config = get_jit_config()
    if config is None or not config.enabled:
        return APIResponse(data={"allowed": False, "reason": "JIT provisioning is not enabled."})
    allowed = config.is_domain_allowed(body.email)
    return APIResponse(
        data={
            "allowed": allowed,
            "email": body.email,
            "reason": "domain allowed" if allowed else "domain not in allowlist",
        }
    )
