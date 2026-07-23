"""Workspace management routes (Enterprise Phase 1 — tasks 1.3.1 + 1.3.2).

Endpoints under /api/v1/workspaces — admin-only, org-scoped.
Config endpoints (1.3.2):
    GET  /workspaces/{id}/config   — read typed workspace config
    PATCH /workspaces/{id}/config  — update workspace config fields
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, require_admin
from src.api.db.engine import get_db
from src.api.db.repositories.users import UserRepository
from src.api.db.repositories.workspaces import WorkspaceRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.plan_enforcement import PlanLimitSnapshot, require_workspace_capacity
from src.api.schemas.common import APIResponse
from src.api.schemas.workspace import (
    AddWorkspaceMemberRequest,
    WorkspaceCreate,
    WorkspaceMemberOut,
    WorkspaceOut,
    WorkspaceUpdate,
)

log = logging.getLogger("cogtrix.api.workspaces")

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


def _ws_to_out(ws, member_count: int = 0) -> WorkspaceOut:
    return WorkspaceOut(
        id=ws.id,
        org_id=ws.org_id,
        name=ws.name,
        description=ws.description,
        settings=ws.settings,
        member_count=member_count,
        is_active=ws.is_active,
        created_at=ws.created_at,
    )


async def _get_ws_in_org(workspace_id: str, ctx: OrgContext, repo: WorkspaceRepository):
    ws = await repo.get_by_id(workspace_id)
    if ws is None or (ctx.org_id is not None and ws.org_id != ctx.org_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Workspace not found."},
        )
    return ws


@router.get("", response_model=APIResponse[list[WorkspaceOut]], summary="List workspaces")
async def list_workspaces(
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[list[WorkspaceOut]]:
    repo = WorkspaceRepository(db)
    workspaces = await repo.list_by_org(ctx.org_id)  # type: ignore[arg-type]
    result = []
    for ws in workspaces:
        count = await repo.count_members(ws.id)
        result.append(_ws_to_out(ws, count))
    return APIResponse(data=result)


@router.post(
    "", response_model=APIResponse[WorkspaceOut], status_code=201, summary="Create workspace"
)
async def create_workspace(
    body: WorkspaceCreate,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
    capacity: PlanLimitSnapshot = Depends(require_workspace_capacity),
) -> APIResponse[WorkspaceOut]:
    del capacity  # noqa: F841 — plan enforcement side-effect only
    repo = WorkspaceRepository(db)
    if await repo.get_by_name_and_org(body.name, ctx.org_id) is not None:  # type: ignore[arg-type]
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": f"Workspace '{body.name}' already exists."},
        )
    ws = await repo.create(
        workspace_id=str(uuid.uuid4()),
        org_id=ctx.org_id,  # type: ignore[arg-type]
        name=body.name,
        description=body.description,
        settings=json.dumps(body.settings) if body.settings else None,
    )
    await db.commit()
    return APIResponse(data=_ws_to_out(ws))


@router.get("/{workspace_id}", response_model=APIResponse[WorkspaceOut], summary="Get workspace")
async def get_workspace(
    workspace_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[WorkspaceOut]:
    repo = WorkspaceRepository(db)
    ws = await _get_ws_in_org(workspace_id, ctx, repo)
    count = await repo.count_members(ws.id)
    return APIResponse(data=_ws_to_out(ws, count))


@router.patch(
    "/{workspace_id}", response_model=APIResponse[WorkspaceOut], summary="Update workspace"
)
async def update_workspace(
    workspace_id: str,
    body: WorkspaceUpdate,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[WorkspaceOut]:
    repo = WorkspaceRepository(db)
    await _get_ws_in_org(workspace_id, ctx, repo)
    if body.name is not None:
        conflict = await repo.get_by_name_and_org(body.name, ctx.org_id)  # type: ignore[arg-type]
        if conflict is not None and conflict.id != workspace_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "CONFLICT", "message": f"Workspace '{body.name}' already taken."},
            )
    ws = await repo.update(
        workspace_id,
        name=body.name,
        description=body.description,
        settings=json.dumps(body.settings) if body.settings is not None else None,
        is_active=body.is_active,
    )
    await db.commit()
    count = await repo.count_members(workspace_id)
    return APIResponse(data=_ws_to_out(ws, count))  # type: ignore[arg-type]


@router.delete("/{workspace_id}", response_model=APIResponse[None], summary="Delete workspace")
async def delete_workspace(
    workspace_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    repo = WorkspaceRepository(db)
    await _get_ws_in_org(workspace_id, ctx, repo)
    await repo.delete(workspace_id)
    await db.commit()
    return APIResponse(data=None)


@router.get(
    "/{workspace_id}/members",
    response_model=APIResponse[list[WorkspaceMemberOut]],
    summary="List workspace members",
)
async def list_ws_members(
    workspace_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[list[WorkspaceMemberOut]]:
    repo = WorkspaceRepository(db)
    await _get_ws_in_org(workspace_id, ctx, repo)
    memberships = await repo.list_memberships(workspace_id)
    user_repo = UserRepository(db)
    members = []
    for m in memberships:
        user = await user_repo.get_by_id(m.user_id)
        if user:
            members.append(
                WorkspaceMemberOut(
                    user_id=user.id,
                    username=user.username,
                    email=user.email,
                    role=m.role,
                    joined_at=m.joined_at,
                )
            )
    return APIResponse(data=members)


@router.post(
    "/{workspace_id}/members",
    response_model=APIResponse[WorkspaceMemberOut],
    status_code=201,
    summary="Add workspace member",
)
async def add_ws_member(
    workspace_id: str,
    body: AddWorkspaceMemberRequest,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[WorkspaceMemberOut]:
    repo = WorkspaceRepository(db)
    await _get_ws_in_org(workspace_id, ctx, repo)
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(body.user_id)
    if user is None or (ctx.org_id is not None and user.org_id != ctx.org_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User not found in this org."},
        )
    if await repo.get_membership(workspace_id, body.user_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": "User is already a member of this workspace."},
        )
    m = await repo.add_member(
        membership_id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        user_id=body.user_id,
        role=body.role,
    )
    await db.commit()
    return APIResponse(
        data=WorkspaceMemberOut(
            user_id=user.id,
            username=user.username,
            email=user.email,
            role=m.role,
            joined_at=m.joined_at,
        )
    )


@router.delete(
    "/{workspace_id}/members/{user_id}",
    response_model=APIResponse[None],
    summary="Remove workspace member",
)
async def remove_ws_member(
    workspace_id: str,
    user_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[None]:
    repo = WorkspaceRepository(db)
    await _get_ws_in_org(workspace_id, ctx, repo)
    if not await repo.remove_member(workspace_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "User is not a member of this workspace."},
        )
    await db.commit()
    return APIResponse(data=None)


# ---------------------------------------------------------------------------
# Workspace-scoped config (task 1.3.2)
# ---------------------------------------------------------------------------


@router.get(
    "/{workspace_id}/config",
    response_model=APIResponse[dict],
    summary="Get workspace config",
)
async def get_ws_config(
    workspace_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Return the typed config overlay for a workspace."""
    from src.api.workspace_context import WorkspaceConfig

    repo = WorkspaceRepository(db)
    ws = await _get_ws_in_org(workspace_id, ctx, repo)
    cfg = WorkspaceConfig.from_json(ws.settings)  # type: ignore[union-attr]
    return APIResponse(data=cfg.to_dict())


@router.patch(
    "/{workspace_id}/config",
    response_model=APIResponse[dict],
    summary="Update workspace config",
)
async def update_ws_config(
    workspace_id: str,
    body: dict,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Merge *body* into the workspace config overlay (partial update).

    Keys set to ``null`` are cleared; other keys are written or updated.
    """
    from src.api.workspace_context import WorkspaceConfig

    repo = WorkspaceRepository(db)
    ws = await _get_ws_in_org(workspace_id, ctx, repo)
    existing = WorkspaceConfig.from_json(ws.settings)  # type: ignore[union-attr]
    current = existing.to_dict()
    for key, value in body.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    new_settings = json.dumps(current) if current else None
    await repo.update(workspace_id, settings=new_settings)
    await db.commit()
    merged_cfg = WorkspaceConfig.from_json(new_settings)
    return APIResponse(data=merged_cfg.to_dict())
