"""JIT user provisioning logic (Enterprise Phase 1 — task 1.2.5).

Provides ``provision_jit_user()`` — the single entry-point called by SAML ACS,
OIDC callbacks, and any future identity provider integrations.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import User
from src.api.jit.config import JITConfig

log = logging.getLogger("cogtrix.api.jit")


class JITDomainDenied(Exception):
    """Raised when the identity's email domain is not in the allowlist."""


class JITCapacityExceeded(Exception):
    """Raised when the org has reached its JIT user limit."""


async def provision_jit_user(
    *,
    email: str,
    username: str,
    config: JITConfig,
    db: AsyncSession,
) -> tuple[User, str]:
    """Find or create a Cogtrix user from a JIT identity.

    Enforces all policies from *config*:
    - Domain allowlist check
    - Capacity limit check
    - Org assignment
    - Auto-team membership

    Args:
        email:    Verified email address from the identity provider.
        username: Preferred username (sanitised before use).
        config:   Active ``JITConfig``.
        db:       Async SQLAlchemy session.

    Returns:
        ``(User, access_token)`` — the Cogtrix user and a signed JWT.

    Raises:
        HTTPException 403 — domain not allowed or capacity exceeded.
        HTTPException 503 — JIT not enabled.
    """
    from src.api.auth import create_access_token, hash_password
    from src.api.db.repositories.organization import OrganizationRepository
    from src.api.db.repositories.teams import TeamRepository
    from src.api.db.repositories.users import UserRepository

    if not config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "JIT_DISABLED", "message": "JIT provisioning is not enabled."},
        )

    if not config.is_domain_allowed(email):
        log.warning("JIT: rejected domain for email=%s", email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "JIT_DOMAIN_DENIED",
                "message": "Your email domain is not authorised for SSO access.",
            },
        )

    user_repo = UserRepository(db)
    org_repo = OrganizationRepository(db)

    # Resolve org.
    if config.org_id:
        org_id: str | None = config.org_id
    else:
        default_org = await org_repo.ensure_default_org()
        await db.commit()
        org_id = default_org.id

    # Capacity check.
    if config.max_users > 0 and org_id:
        count = await org_repo.count_users(org_id)
        if count >= config.max_users:
            log.warning("JIT: org %s at capacity (%d/%d)", org_id, count, config.max_users)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "JIT_CAPACITY_EXCEEDED",
                    "message": "This organisation has reached its maximum user capacity.",
                },
            )

    # Find existing user scoped to this org, or check for conflicts.
    existing = await user_repo.get_by_email(email, org_id=org_id)
    if existing is not None:
        token = create_access_token(user_id=existing.id, role=existing.role)
        log.info("JIT: existing user %s authenticated", existing.id)
        return existing, token

    conflict = await user_repo.get_by_email(email)
    if conflict is not None:
        if conflict.org_id is not None:
            # User belongs to a specific different org — cross-org takeover attempt.
            # Return 422 (not 409) with an opaque message to prevent enumeration.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "USER_ACCOUNT_CONFLICT", "message": "User account conflict."},
            )
        # User exists but has no org — assign to this org and authenticate.
        await user_repo.assign_org(conflict.id, org_id)
        await db.commit()
        token = create_access_token(user_id=conflict.id, role=conflict.role)
        log.info("JIT: unassigned user %s assigned to org %s", conflict.id, org_id)
        return conflict, token

    # Sanitise username: strip @ domain, deduplicate if needed.
    safe_username = username.split("@")[0] if "@" in username else username
    if await user_repo.get_by_username(safe_username) is not None:
        safe_username = f"{safe_username}_{uuid.uuid4().hex[:6]}"

    new_user = await user_repo.create(
        user_id=str(uuid.uuid4()),
        username=safe_username,
        email=email,
        password_hash=hash_password(str(uuid.uuid4())),  # unusable password — SSO only
        role=config.default_role,
        org_id=org_id,
    )
    await db.commit()
    log.info("JIT: provisioned new user %s (org=%s)", new_user.id, org_id)

    # Auto-add to team if configured.
    if config.auto_team_id:
        team_repo = TeamRepository(db)
        team = await team_repo.get_by_id(config.auto_team_id)
        if team is not None:
            if team.org_id != org_id:
                log.warning(
                    "JIT: security alert — auto_team_id %s belongs to org %s, " "not user's org %s",
                    config.auto_team_id,
                    team.org_id,
                    org_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "code": "JIT_TEAM_ORG_MISMATCH",
                        "message": "Team configuration error.",
                    },
                )
            existing_m = await team_repo.get_membership(config.auto_team_id, new_user.id)
            if existing_m is None:
                await team_repo.add_member(
                    membership_id=str(uuid.uuid4()),
                    team_id=config.auto_team_id,
                    user_id=new_user.id,
                    role="member",
                )
                await db.commit()
                log.info("JIT: added %s to team %s", new_user.id, config.auto_team_id)

    token = create_access_token(user_id=new_user.id, role=new_user.role)
    return new_user, token
