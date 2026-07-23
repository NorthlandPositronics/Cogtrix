"""User management endpoints (admin only).

Endpoints:
    GET    /api/v1/users              — list all users
    POST   /api/v1/users              — create a user
    PATCH  /api/v1/users/{user_id}   — update user role
    DELETE /api/v1/users/{user_id}   — delete a user
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, hash_password
from src.api.db import get_db
from src.api.db.models import User
from src.api.db.repositories.users import UserRepository
from src.api.org_context import OrgContext, get_org_context
from src.api.quota import _quota_config_from_app_config, get_user_quota_status
from src.api.schemas.common import APIResponse
from src.api.schemas.user import UserCreateRequest, UserOut, UserUpdateRequest
from src.audit import record_user_action
from src.auth.middleware import require
from src.auth.permissions import Permission

router = APIRouter(prefix="/users", tags=["Users"])


def _user_to_out(user: object) -> UserOut:
    return UserOut(
        id=getattr(user, "id", ""),
        username=getattr(user, "username", ""),
        email=getattr(user, "email", ""),
        role=getattr(user, "role", "user"),
        created_at=getattr(user, "created_at", None),
    )


@router.get(
    "/me/quota",
    summary="Get my quota status",
    description="Return quota limits and current usage for the authenticated user.",
    response_model=APIResponse[dict],
    responses={
        200: {"description": "Quota status returned."},
        401: {"description": "Not authenticated."},
    },
)
async def get_my_quota(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
) -> APIResponse[dict]:
    """Return quota limits and usage for the current user.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED.
    """
    app_config = getattr(request.app.state, "config", None)
    quota_cfg = _quota_config_from_app_config(app_config) if app_config is not None else None
    from src.api.quota import QuotaConfig

    status_data = get_user_quota_status(
        current_user.user_id, quota_cfg if quota_cfg is not None else QuotaConfig()
    )
    return APIResponse(data=status_data)


@router.get(
    "",
    summary="List all users",
    description="Return all registered user accounts. Admin only.",
    response_model=APIResponse[list[UserOut]],
    responses={
        200: {"description": "User list returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
    },
)
async def list_users(
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require(Permission.USERS_MANAGE)),
) -> APIResponse[list[UserOut]]:
    """List all registered users in the caller's organization.

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    repo = UserRepository(db)
    if ctx.has_org:
        users = await repo.list_by_org(ctx.org_id)
    else:
        # Back-compat: admin without org sees only unscoped users.
        result = await db.execute(
            select(User).where(User.org_id.is_(None)).order_by(User.created_at)
        )
        users = list(result.scalars().all())
    return APIResponse(data=[_user_to_out(u) for u in users])


@router.post(
    "",
    summary="Create a user",
    description="Create a new user account with the specified role. Admin only.",
    response_model=APIResponse[UserOut],
    status_code=201,
    responses={
        201: {"description": "User created."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        409: {"description": "Username or email already exists (CONFLICT)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def create_user(
    body: UserCreateRequest,
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require(Permission.USERS_CREATE)),
) -> APIResponse[UserOut]:
    """Create a new user account in the caller's organization (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, CONFLICT, VALIDATION_ERROR.
    """
    repo = UserRepository(db)
    try:
        user = await repo.create(
            user_id=str(uuid.uuid4()),
            username=body.username,
            email=str(body.email),
            password_hash=hash_password(body.password),
            role=body.role,
            org_id=ctx.org_id,
        )
        await db.commit()
        await db.refresh(user)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "CONFLICT",
                "message": "A user with that username or email already exists.",
            },
        ) from exc
    return APIResponse(data=_user_to_out(user))


@router.patch(
    "/{user_id}",
    summary="Update user role",
    description="Update the role of an existing user account. Admin only.",
    response_model=APIResponse[UserOut],
    responses={
        200: {"description": "User updated."},
        400: {"description": "Cannot demote own account (BAD_REQUEST)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "User not found (NOT_FOUND)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def update_user(
    user_id: str,
    body: UserUpdateRequest,
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require(Permission.USERS_UPDATE)),
) -> APIResponse[UserOut]:
    """Update user role (admin only, same org).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, VALIDATION_ERROR.
    """
    repo = UserRepository(db)

    if body.role is None:
        user = await repo.get_by_id(user_id)
        if user is None or user.org_id != ctx.org_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
            )
        return APIResponse(data=_user_to_out(user))

    if body.role == "user" and user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Cannot demote your own account."},
        )

    user = await repo.get_by_id(user_id)
    if user is None or user.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )

    previous_role = user.role
    user = await repo.update_role(user_id, body.role)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )
    await db.commit()
    await db.refresh(user)

    record_user_action(
        "update_user_role",
        actor=current_user.user_id,
        status="ok",
        detail={
            "org_id": ctx.org_id,
            "user_id": user_id,
            "previous_role": previous_role,
            "new_role": body.role,
        },
    )

    return APIResponse(data=_user_to_out(user))


@router.delete(
    "/{user_id}",
    summary="Delete a user",
    description="Permanently delete a user account. Admin only.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "User deleted."},
        400: {"description": "Cannot delete own account (BAD_REQUEST)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "User not found (NOT_FOUND)."},
    },
)
async def delete_user(
    user_id: str,
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require(Permission.USERS_DELETE)),
) -> APIResponse[None]:
    """Delete a user account (admin only, same org).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, BAD_REQUEST, NOT_FOUND.
    """
    if user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Cannot delete your own account."},
        )
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user is None or user.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )
    deleted = await repo.delete(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )
    await db.commit()
    return APIResponse(data=None)
