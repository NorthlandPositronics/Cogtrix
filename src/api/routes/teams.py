"""Team management routes (Enterprise Phase 1 — task 1.2.4).

Endpoints:
    GET    /api/v1/teams                         — list teams in caller's org
    POST   /api/v1/teams                         — create team
    GET    /api/v1/teams/{team_id}               — get team
    PATCH  /api/v1/teams/{team_id}               — update team
    DELETE /api/v1/teams/{team_id}               — delete team
    GET    /api/v1/teams/{team_id}/members       — list members
    POST   /api/v1/teams/{team_id}/members       — add member
    DELETE /api/v1/teams/{team_id}/members/{uid} — remove member

All endpoints are org-scoped: callers can only manage teams in their own org.
Admin role required.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, require_admin
from src.api.db.engine import get_db
from src.api.db.repositories.teams import TeamRepository
from src.api.db.repositories.users import UserRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.schemas.common import APIResponse
from src.api.schemas.team import AddMemberRequest, MemberOut, TeamCreate, TeamOut, TeamUpdate

log = logging.getLogger("cogtrix.api.teams")

router = APIRouter(prefix="/teams", tags=["Teams"])


def _team_to_out(team, member_count: int = 0) -> TeamOut:
    return TeamOut(
        id=team.id,
        org_id=team.org_id,
        name=team.name,
        description=team.description,
        member_count=member_count,
        created_at=team.created_at,
    )


async def _get_team_in_org(team_id: str, ctx: OrgContext, repo: TeamRepository) -> object:
    """Load a team, enforcing org-scope. Raises 404 if missing or wrong org."""
    team = await repo.get_by_id(team_id)
    if team is None or ctx.org_id is None or team.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Team not found."},
        )
    return team


# ---------------------------------------------------------------------------
# Teams CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=APIResponse[list[TeamOut]], summary="List teams")
async def list_teams(
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[list[TeamOut]]:
    """Return all teams in the caller's organization."""
    repo = TeamRepository(db)
    teams = await repo.list_by_org(ctx.org_id)  # type: ignore[arg-type]
    result = []
    for t in teams:
        count = await repo.count_members(t.id)
        result.append(_team_to_out(t, count))
    return APIResponse(data=result)


@router.post("", response_model=APIResponse[TeamOut], status_code=201, summary="Create team")
async def create_team(
    body: TeamCreate,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[TeamOut]:
    """Create a new team in the caller's org."""
    repo = TeamRepository(db)
    existing = await repo.get_by_name_and_org(body.name, ctx.org_id)  # type: ignore[arg-type]
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Team '{body.name}' already exists."},
        )
    try:
        team = await repo.create(
            team_id=str(uuid.uuid4()),
            org_id=ctx.org_id,  # type: ignore[arg-type]
            name=body.name,
            description=body.description,
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Team '{body.name}' already exists."},
        ) from None
    log.info("Created team %s in org %s", team.id, ctx.org_id)
    return APIResponse(data=_team_to_out(team))


@router.get("/{team_id}", response_model=APIResponse[TeamOut], summary="Get team")
async def get_team(
    team_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[TeamOut]:
    repo = TeamRepository(db)
    team = await _get_team_in_org(team_id, ctx, repo)
    count = await repo.count_members(team.id)  # type: ignore[union-attr]
    return APIResponse(data=_team_to_out(team, count))


@router.patch("/{team_id}", response_model=APIResponse[TeamOut], summary="Update team")
async def update_team(
    team_id: str,
    body: TeamUpdate,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[TeamOut]:
    repo = TeamRepository(db)
    await _get_team_in_org(team_id, ctx, repo)
    if body.name is not None:
        conflict = await repo.get_by_name_and_org(body.name, ctx.org_id)  # type: ignore[arg-type]
        if conflict is not None and conflict.id != team_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "CONFLICT", "message": f"Team name '{body.name}' already taken."},
            )
    try:
        team = await repo.update(team_id, name=body.name, description=body.description)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Team name '{body.name}' already taken."},
        ) from None
    count = await repo.count_members(team_id)
    return APIResponse(data=_team_to_out(team, count))  # type: ignore[arg-type]


@router.delete("/{team_id}", response_model=APIResponse[None], summary="Delete team")
async def delete_team(
    team_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    repo = TeamRepository(db)
    await _get_team_in_org(team_id, ctx, repo)
    await repo.delete(team_id)
    await db.commit()
    log.info("Deleted team %s", team_id)
    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


@router.get(
    "/{team_id}/members",
    response_model=APIResponse[list[MemberOut]],
    summary="List team members",
)
async def list_members(
    team_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[list[MemberOut]]:
    repo = TeamRepository(db)
    await _get_team_in_org(team_id, ctx, repo)
    memberships = await repo.list_memberships(team_id)
    user_repo = UserRepository(db)
    members = []
    for m in memberships:
        user = await user_repo.get_by_id(m.user_id)
        if user:
            members.append(
                MemberOut(
                    user_id=user.id,
                    username=user.username,
                    email=user.email,
                    role=m.role,
                    joined_at=m.joined_at,
                )
            )
    return APIResponse(data=members)


@router.post(
    "/{team_id}/members",
    response_model=APIResponse[MemberOut],
    status_code=201,
    summary="Add team member",
)
async def add_member(
    team_id: str,
    body: AddMemberRequest,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[MemberOut]:
    repo = TeamRepository(db)
    await _get_team_in_org(team_id, ctx, repo)

    # Verify user exists and belongs to the same org.
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(body.user_id)
    if user is None or (ctx.org_id is not None and user.org_id != ctx.org_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User not found in this org."},
        )

    existing = await repo.get_membership(team_id, body.user_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": "User is already a member of this team."},
        )

    membership = await repo.add_member(
        membership_id=str(uuid.uuid4()),
        team_id=team_id,
        user_id=body.user_id,
        role=body.role,
    )
    await db.commit()
    return APIResponse(
        data=MemberOut(
            user_id=user.id,
            username=user.username,
            email=user.email,
            role=membership.role,
            joined_at=membership.joined_at,
        )
    )


@router.delete(
    "/{team_id}/members/{user_id}",
    response_model=APIResponse[None],
    summary="Remove team member",
)
async def remove_member(
    team_id: str,
    user_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    repo = TeamRepository(db)
    await _get_team_in_org(team_id, ctx, repo)
    removed = await repo.remove_member(team_id, user_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User is not a member of this team."},
        )
    await db.commit()
    return APIResponse(data=None)
