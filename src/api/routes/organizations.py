"""Organization member management endpoints (admin only).

Endpoints:
    PUT /organizations/{org_id}/members/{user_id}/role — update member role
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, require_admin
from src.api.db import get_db
from src.api.db.repositories.users import UserRepository
from src.api.org_context import OrgContext, get_org_context
from src.api.schemas.common import APIResponse
from src.api.schemas.user import UserOut, UserUpdateRequest
from src.audit import record_user_action

router = APIRouter(prefix="/organizations", tags=["Organizations"])


def _user_to_out(user: object) -> UserOut:
    return UserOut(
        id=getattr(user, "id", ""),
        username=getattr(user, "username", ""),
        email=getattr(user, "email", ""),
        role=getattr(user, "role", "user"),
        created_at=getattr(user, "created_at", None),
    )


@router.put(
    "/{org_id}/members/{user_id}/role",
    summary="Update member role",
    description="Update the role of a member in the specified organization. Admin only.",
    response_model=APIResponse[UserOut],
    responses={
        200: {"description": "Member role updated."},
        400: {"description": "Cannot demote own account (BAD_REQUEST)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "User not found in organization (NOT_FOUND)."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def update_member_role(
    org_id: str,
    user_id: str,
    body: UserUpdateRequest,
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_admin),
) -> APIResponse[UserOut]:
    """Update a member's role in the specified organization (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, NOT_FOUND,
                 BAD_REQUEST, VALIDATION_ERROR.
    """
    repo = UserRepository(db)

    if body.role is None:
        user = await repo.get_by_id_and_org(user_id, org_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "NOT_FOUND",
                    "message": f"User '{user_id}' not found in organization '{org_id}'.",
                },
            )
        return APIResponse(data=_user_to_out(user))

    if body.role == "user" and user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "Cannot demote your own account."},
        )

    user = await repo.get_by_id_and_org(user_id, org_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"User '{user_id}' not found in organization '{org_id}'.",
            },
        )

    previous_role = user.role
    user = await repo.update_role(user_id, body.role)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NOT_FOUND",
                "message": f"User '{user_id}' not found in organization '{org_id}'.",
            },
        )
    await db.commit()
    await db.refresh(user)

    record_user_action(
        "update_member_role",
        actor=current_user.user_id,
        status="ok",
        detail={
            "org_id": org_id,
            "user_id": user_id,
            "previous_role": previous_role,
            "new_role": body.role,
        },
    )

    return APIResponse(data=_user_to_out(user))
