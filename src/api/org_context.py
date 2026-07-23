"""Organization context resolution for multi-tenant requests (Enterprise Phase 1 — task 1.1.3).

Provides two FastAPI dependencies:

    get_org_context     — resolves the caller's org (nullable; free-tier users have none)
    require_org_context — like get_org_context but raises 403 when no org is assigned

Usage in a route::

    @router.get("/things")
    async def list_things(ctx: OrgContext = Depends(require_org_context)):
        return repo.list_by_org(ctx.org_id)

Why dependencies (not ASGI middleware)?
- Need async DB access for the user→org lookup
- FastAPI dependencies are composable, cacheable per-request, and trivially testable
- Runs only on routes that opt in — no overhead on unauthenticated or health endpoints
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user
from src.api.db.engine import get_db
from src.api.db.models import Organization, User
from src.api.db.repositories.users import UserRepository

log = logging.getLogger("cogtrix.api.org_context")


@dataclass
class OrgContext:
    """Resolved organization context attached to an authenticated request.

    Attributes:
        user_id: UUID of the authenticated user.
        role:    JWT role string (``'admin'`` or ``'user'``).
        org_id:  UUID of the user's organization, or ``None`` for unassigned users.
        org:     Full ``Organization`` ORM object, or ``None`` when ``org_id`` is ``None``.
    """

    user_id: str
    role: str = field(default="user")
    org_id: str | None = field(default=None)
    org: Organization | None = field(default=None, compare=False, repr=False)

    @property
    def has_org(self) -> bool:
        """True when the user is assigned to an organization."""
        return self.org_id is not None

    @property
    def is_admin(self) -> bool:
        """True when the user holds the ``admin`` or ``superadmin`` role."""
        return self.role in ("admin", "superadmin")


async def get_org_context(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OrgContext:
    """FastAPI dependency: resolve the organization for the current user.

    Looks up the authenticated user in the DB to retrieve their ``org_id``.
    If ``org_id`` is set, also eager-loads the ``Organization`` row so callers
    can access org metadata (plan, settings) without an extra query.

    Returns:
        ``OrgContext`` with ``org_id=None`` and ``org=None`` for users that
        have not yet been assigned to an organization.

    Raises:
        HTTPException 401 — propagated from ``get_current_user``.
    """
    user_repo = UserRepository(db)
    user: User | None = await user_repo.get_by_id(current_user.user_id)

    if user is None or user.org_id is None:
        return OrgContext(user_id=current_user.user_id, role=current_user.role)

    # Load the Organization row for org metadata access downstream.
    from sqlalchemy import select

    result = await db.execute(select(Organization).where(Organization.id == user.org_id))
    org: Organization | None = result.scalar_one_or_none()

    return OrgContext(
        user_id=current_user.user_id,
        role=current_user.role,
        org_id=user.org_id,
        org=org,
    )


async def require_org_context(
    ctx: OrgContext = Depends(get_org_context),
) -> OrgContext:
    """FastAPI dependency: require the user to be assigned to an organization.

    Use this on enterprise-only endpoints where org-less users must be blocked.

    Raises:
        HTTPException 403 ORG_REQUIRED — user has no organization assigned.
    """
    if not ctx.has_org:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "ORG_REQUIRED",
                "message": "This endpoint requires the user to belong to an organization.",
            },
        )
    return ctx


def assert_same_org(
    ctx: OrgContext,
    resource_org_id: str | None,
    *,
    admin_bypass: bool = True,
) -> None:
    """Enforce cross-org isolation: raise 403 if *ctx* cannot access *resource_org_id*.

    Rules:
    - If the caller has no org (``ctx.org_id is None``), they may only access
      unscoped resources (``resource_org_id is None``).  Access to any org-owned
      resource is denied.
    - If the caller has an org, they may access resources from the **same** org.
      Access to a different org's resources is denied.
    - When *admin_bypass* is ``True`` (default), callers whose JWT role is
      ``"admin"`` are exempt — they can access resources across all orgs.
      Pass ``admin_bypass=False`` for endpoints that must enforce isolation even
      for admins (e.g. billing endpoints where cross-org access is never valid).

    Args:
        ctx:              Resolved ``OrgContext`` for the current request.
        resource_org_id:  ``org_id`` of the resource being accessed, or ``None``
                          for unscoped resources.
        admin_bypass:     When ``True``, skip the check for admin callers.

    Raises:
        HTTPException 403 CROSS_ORG_ACCESS — caller's org does not match the
            resource's org.
    """
    # Admins skip the isolation check by default.
    if admin_bypass and ctx.is_admin:
        return

    if ctx.org_id != resource_org_id:
        log.warning(
            "cross-org access attempt: caller_org=%s resource_org=%s user=%s",
            ctx.org_id,
            resource_org_id,
            ctx.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "CROSS_ORG_ACCESS",
                "message": "Access to a resource outside your organization is not permitted.",
            },
        )
