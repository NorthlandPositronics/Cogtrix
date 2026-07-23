"""User management endpoints (admin only).

Endpoints:
    GET    /api/v1/users              — list all users
    POST   /api/v1/users              — create a user
    PATCH  /api/v1/users/{user_id}   — update user role
    DELETE /api/v1/users/{user_id}   — delete a user
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, hash_password, require_admin
from src.api.db import get_db
from src.api.db.repositories.users import UserRepository
from src.api.schemas.common import APIResponse
from src.api.schemas.user import UserCreateRequest, UserOut, UserUpdateRequest

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
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[list[UserOut]]:
    """List all registered users.

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN.
    """
    repo = UserRepository(db)
    users = await repo.list_all()
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
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[UserOut]:
    """Create a new user account (admin only).

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
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[UserOut]:
    """Update user role (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND, VALIDATION_ERROR.
    """
    if body.role is None:
        repo = UserRepository(db)
        user = await repo.get_by_id(user_id)
        if user is None:
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

    repo = UserRepository(db)
    user = await repo.update_role(user_id, body.role)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )
    await db.commit()
    await db.refresh(user)
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
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    """Delete a user account (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, BAD_REQUEST, NOT_FOUND.
    """
    if user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Cannot delete your own account."},
        )
    repo = UserRepository(db)
    deleted = await repo.delete(user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"User '{user_id}' not found."},
        )
    await db.commit()
    return APIResponse(data=None)
