"""SCIM 2.0 provisioning endpoints (Enterprise Phase 1 — task 1.2.2).

Implements RFC 7644 protocol for automated user provisioning.  Org-scoped:
all operations are restricted to users belonging to the authenticated
caller's organization.

Endpoints (all under /scim/v2/):
    GET    /ServiceProviderConfig       — server capabilities
    GET    /Users                       — list users (filter + pagination)
    POST   /Users                       — create user
    GET    /Users/{user_id}             — get user
    PUT    /Users/{user_id}             — full replace
    PATCH  /Users/{user_id}             — partial update (RFC 7644 §3.5.2)
    DELETE /Users/{user_id}             — deactivate (soft-delete via is_active)

Authentication: Admin bearer token (same JWT as the rest of the API).
Content-Type:   application/scim+json
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, require_admin
from src.api.db.engine import get_db
from src.api.db.models import User
from src.api.db.repositories.users import UserRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.plan_enforcement import PlanLimitSnapshot, require_user_capacity
from src.api.saml.config import get_saml_config
from src.api.scim.mapping import parse_scim_filter, user_to_scim
from src.api.scim.schemas import (
    SCIMError,
    SCIMListResponse,
    SCIMPatch,
    SCIMServiceProviderConfig,
    SCIMUserCreate,
    SCIMUserReplace,
)

log = logging.getLogger("cogtrix.api.scim")

# ---------------------------------------------------------------------------
# PATCH helpers
# ---------------------------------------------------------------------------

_VALID_PATCH_PATHS = {"username", "emails", "active", ""}


class _SCIMPatchPathError(ValueError):
    """Raised when a SCIM PATCH operation references an unsupported path."""


router = APIRouter(prefix="/scim/v2", tags=["SCIM 2.0"])

_SCIM_CONTENT_TYPE = "application/scim+json"


def _scim_response(body: Any, status_code: int = 200) -> JSONResponse:
    content = body.model_dump(mode="json", exclude_none=True)
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type=_SCIM_CONTENT_TYPE,
    )


def _scim_error(status_code: int, detail: str, scim_type: str | None = None) -> JSONResponse:
    err = SCIMError(status=str(status_code), detail=detail, scimType=scim_type)
    return JSONResponse(
        content=err.model_dump(mode="json", exclude_none=True),
        status_code=status_code,
        media_type=_SCIM_CONTENT_TYPE,
    )


def _scim_uniqueness_error() -> JSONResponse:
    return _scim_error(409, "User already exists.", scim_type="uniqueness")


def _scim_cross_org_error() -> JSONResponse:
    """Opaque error for cross-org conflicts — same body as 409, different status."""
    return _scim_error(422, "User already exists.", scim_type="uniqueness")


def _parse_scim_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    return None


def _base_url() -> str | None:
    config = get_saml_config()
    if config is None:
        return None
    scim_base_url = getattr(config, "scim_base_url", None)
    if not scim_base_url:
        return None
    return scim_base_url.rstrip("/")


# ---------------------------------------------------------------------------
# ServiceProviderConfig
# ---------------------------------------------------------------------------


@router.get("/ServiceProviderConfig", summary="SCIM server capabilities")
async def scim_service_provider_config(
    _request: Request,
    _: TokenData = Depends(require_admin),
) -> JSONResponse:
    cfg = SCIMServiceProviderConfig()
    return _scim_response(cfg)


# ---------------------------------------------------------------------------
# Users — List + Create
# ---------------------------------------------------------------------------


@router.get("/Users", summary="List SCIM users")
async def scim_list_users(
    filter: str | None = Query(default=None, description="SCIM filter expression."),
    startIndex: int = Query(default=1, ge=1),
    count: int = Query(default=100, ge=1, le=200),
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> JSONResponse:
    """Return a paginated, optionally filtered list of users in the caller's org."""
    base = _base_url()
    if base is None:
        return _scim_error(
            503,
            "SCIM base URL is not configured.",
            scim_type="invalidValue",
        )

    stmt = select(User).where(User.org_id == ctx.org_id)

    flt = parse_scim_filter(filter)
    if filter and flt is None:
        return _scim_error(400, "Invalid SCIM filter.", scim_type="invalidFilter")
    if flt:
        clauses = flt if isinstance(flt, list) else [flt]
        for clause in clauses:
            attr, op, value = clause["attr"], clause["op"], clause["value"]
            if attr in ("username",) and op == "eq":
                stmt = stmt.where(User.username == value)
            elif attr in ("emails.value", "emails") and op == "eq":
                stmt = stmt.where(func.lower(User.email) == value.lower())
            elif attr == "active" and op == "eq":
                active = _parse_scim_bool(value)
                if active is None:
                    return _scim_error(400, "Invalid SCIM filter.", scim_type="invalidFilter")
                stmt = stmt.where(User.is_active.is_(active))
            else:
                return _scim_error(400, "Invalid SCIM filter.", scim_type="invalidFilter")

    # Count total before pagination.
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Apply pagination.
    stmt = stmt.order_by(User.created_at).offset(startIndex - 1).limit(count)
    users = list((await db.execute(stmt)).scalars().all())

    resources = [user_to_scim(u, base) for u in users]
    resp = SCIMListResponse(
        totalResults=total,
        startIndex=startIndex,
        itemsPerPage=len(resources),
        Resources=resources,
    )
    return _scim_response(resp)


@router.post("/Users", summary="Create SCIM user", status_code=201)
async def scim_create_user(
    body: SCIMUserCreate,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
    capacity: PlanLimitSnapshot = Depends(require_user_capacity),
) -> JSONResponse:
    """Provision a new user from a SCIM request.

    ``capacity`` is injected by FastAPI for plan enforcement side-effects only.
    """
    del capacity  # noqa: F841 — plan enforcement side-effect only
    base = _base_url()
    if base is None:
        return _scim_error(
            503,
            "SCIM base URL is not configured.",
            scim_type="invalidValue",
        )
    repo = UserRepository(db)

    # Same-org conflict → 409 (normal SCIM behaviour).
    existing = await repo.get_by_username(body.userName, org_id=ctx.org_id)
    if existing is not None:
        return _scim_uniqueness_error()

    email = body.emails[0].value if body.emails else f"{body.userName}@scim.local"
    existing_email = await repo.get_by_email(email, org_id=ctx.org_id)
    if existing_email is not None:
        return _scim_uniqueness_error()

    # Cross-org conflict → 422 with identical opaque body (prevents enumeration).
    cross = await repo.get_by_username(body.userName)
    if cross is not None:
        return _scim_cross_org_error()
    cross_email = await repo.get_by_email(email)
    if cross_email is not None:
        return _scim_cross_org_error()

    from src.api.auth import hash_password

    new_user = await repo.create(
        user_id=str(uuid.uuid4()),
        username=body.userName,
        email=email,
        password_hash=hash_password(str(uuid.uuid4())),  # unusable password — SSO only
        role="user",
        org_id=ctx.org_id,
    )
    new_user.is_active = body.active
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return _scim_uniqueness_error()
    log.info("SCIM: provisioned user %s (org=%s)", new_user.id, ctx.org_id)
    scim_user = user_to_scim(new_user, base)
    resp = _scim_response(scim_user, 201)
    resp.headers["Location"] = (
        scim_user.meta.location if scim_user.meta is not None else None
    ) or ""
    return resp


# ---------------------------------------------------------------------------
# Users — Get / Replace / Patch / Delete by ID
# ---------------------------------------------------------------------------


async def _get_user_or_404(
    user_id: str,
    ctx: OrgContext,
    db: AsyncSession,
) -> User:
    """Load a user by ID, enforcing org-scope. Returns User or raises 404."""
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User not found."},
        )
    if user.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User not found."},
        )
    return user


@router.get("/Users/{user_id}", summary="Get SCIM user")
async def scim_get_user(
    user_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> JSONResponse:
    base = _base_url()
    if base is None:
        return _scim_error(
            503,
            "SCIM base URL is not configured.",
            scim_type="invalidValue",
        )
    user = await _get_user_or_404(user_id, ctx, db)
    return _scim_response(user_to_scim(user, base))


@router.put("/Users/{user_id}", summary="Replace SCIM user")
async def scim_replace_user(
    user_id: str,
    body: SCIMUserReplace,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> JSONResponse:
    """Full replacement of a SCIM user (PUT). userName changes are allowed."""
    base = _base_url()
    if base is None:
        return _scim_error(
            503,
            "SCIM base URL is not configured.",
            scim_type="invalidValue",
        )
    user = await _get_user_or_404(user_id, ctx, db)
    repo = UserRepository(db)

    # Check userName uniqueness if it changed.
    if body.userName != user.username:
        conflict = await repo.get_by_username(body.userName, org_id=ctx.org_id)
        if conflict is not None:
            return _scim_uniqueness_error()
        cross = await repo.get_by_username(body.userName)
        if cross is not None:
            return _scim_cross_org_error()

    email = body.emails[0].value if body.emails else user.email
    if email.lower() != user.email.lower():
        conflict = await repo.get_by_email(email, org_id=ctx.org_id)
        if conflict is not None:
            return _scim_uniqueness_error()
        cross = await repo.get_by_email(email)
        if cross is not None:
            return _scim_cross_org_error()

    # Apply updates via direct ORM mutation (no dedicated replace method needed).
    user.username = body.userName
    user.email = email.lower()
    user.is_active = body.active
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return _scim_uniqueness_error()
    await db.refresh(user)
    log.info("SCIM: replaced user %s", user_id)
    return _scim_response(user_to_scim(user, base))


@router.patch("/Users/{user_id}", summary="Update SCIM user")
async def scim_patch_user(
    user_id: str,
    body: SCIMPatch,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> JSONResponse:
    """Partial update via SCIM PATCH operations (RFC 7644 §3.5.2)."""
    base = _base_url()
    if base is None:
        return _scim_error(
            503,
            "SCIM base URL is not configured.",
            scim_type="invalidValue",
        )
    user = await _get_user_or_404(user_id, ctx, db)
    repo = UserRepository(db)

    # Compute intended new values before mutating the ORM object so we can
    # check uniqueness without triggering SQLAlchemy autoflush (BUG-121).
    try:
        target_username = user.username
        target_email = user.email
        for op in body.Operations:
            target_username, target_email = _compute_patch_target(target_username, target_email, op)
    except _SCIMPatchPathError as exc:
        return _scim_error(400, str(exc), scim_type="invalidPath")
    except ValueError as exc:
        return _scim_error(400, str(exc), scim_type="invalidValue")

    if target_username != user.username:
        conflict = await repo.get_by_username(target_username, org_id=ctx.org_id)
        if conflict is not None and conflict.id != user.id:
            return _scim_uniqueness_error()
        cross = await repo.get_by_username(target_username)
        if cross is not None and cross.id != user.id:
            return _scim_cross_org_error()
    if target_email.lower() != user.email.lower():
        conflict = await repo.get_by_email(target_email, org_id=ctx.org_id)
        if conflict is not None and conflict.id != user.id:
            return _scim_uniqueness_error()
        cross = await repo.get_by_email(target_email)
        if cross is not None and cross.id != user.id:
            return _scim_cross_org_error()

    # Now that uniqueness is verified, apply mutations.
    try:
        for op in body.Operations:
            _apply_patch_op(user, op)
    except _SCIMPatchPathError as exc:
        return _scim_error(400, str(exc), scim_type="invalidPath")
    except ValueError as exc:
        return _scim_error(400, str(exc), scim_type="invalidValue")

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return _scim_uniqueness_error()
    await db.refresh(user)
    log.info("SCIM: patched user %s (%d ops)", user_id, len(body.Operations))
    return _scim_response(user_to_scim(user, base))


def _compute_patch_target(username: str, email: str, op: Any) -> tuple[str, str]:
    """Return the (username, email) that would result from applying *op*.

    This is a pure function — it does not mutate any ORM object — so it can be
    used to preview the outcome of a patch before checking constraints.
    """
    path = (op.path or "").lower()
    value = op.value

    # Validate path regardless of op type (RFC 7644 compliance).
    if path not in _VALID_PATCH_PATHS:
        raise _SCIMPatchPathError(f"Invalid PATCH path: {op.path!r}")

    # For empty path, validate that the value dict only contains allowed keys.
    if not path and isinstance(value, dict):
        for key in value:
            if key.lower() not in _VALID_PATCH_PATHS:
                raise _SCIMPatchPathError(f"Invalid PATCH path: {key!r}")

    if op.op not in ("add", "replace"):
        return username, email

    if path in ("username",) or (not path and isinstance(value, dict) and "userName" in value):
        if isinstance(value, str):
            return value, email
        elif isinstance(value, dict):
            new_name = value.get("userName", username)
            if not isinstance(new_name, str):
                raise ValueError("Invalid SCIM userName value.")
            return new_name, email
        else:
            raise ValueError("Invalid SCIM userName value.")
    elif path == "emails" or (not path and isinstance(value, dict) and "emails" in value):
        if isinstance(value, list):
            emails = value
        elif isinstance(value, dict):
            emails = value.get("emails", [])
        else:
            raise ValueError("Invalid SCIM emails value.")
        if not isinstance(emails, list):
            raise ValueError("Invalid SCIM emails value.")
        if emails:
            new_email = _normalize_scim_email_value(
                emails[0].get("value", email) if isinstance(emails[0], dict) else emails[0]
            )
            return username, new_email
    elif path == "active" or (not path and isinstance(value, dict) and "active" in value):
        active_value = value if path == "active" else value.get("active")
        if _parse_scim_bool(active_value) is None:
            raise ValueError("Invalid SCIM active value.")

    return username, email


def _apply_patch_op(user: User, op: Any) -> None:
    """Apply a single SCIM PATCH operation to the user ORM object."""
    path = (op.path or "").lower()
    value = op.value

    # Validate path regardless of op type (RFC 7644 compliance).
    if path not in _VALID_PATCH_PATHS:
        raise _SCIMPatchPathError(f"Invalid PATCH path: {op.path!r}")

    # For empty path, validate that the value dict only contains allowed keys.
    if not path and isinstance(value, dict):
        for key in value:
            if key.lower() not in _VALID_PATCH_PATHS:
                raise _SCIMPatchPathError(f"Invalid PATCH path: {key!r}")

    if op.op in ("add", "replace"):
        if path in ("username",) or (not path and isinstance(value, dict) and "userName" in value):
            if isinstance(value, str):
                user.username = value
            elif isinstance(value, dict):
                new_name = value.get("userName", user.username)
                if not isinstance(new_name, str):
                    raise ValueError("Invalid SCIM userName value.")
                user.username = new_name
            else:
                raise ValueError("Invalid SCIM userName value.")
        elif path == "emails" or (not path and isinstance(value, dict) and "emails" in value):
            if isinstance(value, list):
                emails = value
            elif isinstance(value, dict):
                emails = value.get("emails", [])
            else:
                raise ValueError("Invalid SCIM emails value.")
            if not isinstance(emails, list):
                raise ValueError("Invalid SCIM emails value.")
            if emails:
                user.email = _normalize_scim_email_value(
                    emails[0].get("value", user.email) if isinstance(emails[0], dict) else emails[0]
                )
        elif path == "active" or (not path and isinstance(value, dict) and "active" in value):
            active_value = value if path == "active" else value.get("active")
            parsed = _parse_scim_bool(active_value)
            if parsed is None:
                raise ValueError("Invalid SCIM active value.")
            user.is_active = parsed
    # "remove" ops are no-ops for the fields we support


def _normalize_scim_email_value(value: Any) -> str:
    """Return a normalized SCIM email value or raise for invalid input."""
    if not isinstance(value, str):
        raise ValueError("Invalid SCIM email value.")
    return value.lower()


@router.delete("/Users/{user_id}", summary="Delete SCIM user", status_code=204)
async def scim_delete_user(
    user_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> Response:
    """Delete a user via SCIM by deactivating the account."""
    user = await _get_user_or_404(user_id, ctx, db)
    repo = UserRepository(db)
    await repo.set_active(user.id, False)
    await db.commit()
    log.info("SCIM: deleted user %s", user_id)
    return Response(status_code=204)
