"""Cross-workspace agent communication routes (Enterprise Phase 1 — task 1.3.3).

Endpoints:
    POST   /api/v1/cross-workspace/messages          — send message
    GET    /api/v1/cross-workspace/inbox/{ws_id}     — read inbox
    DELETE /api/v1/cross-workspace/inbox/{ws_id}/{msg_id} — delete message
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user
from src.api.cross_workspace import (
    CrossWorkspaceMessage,
    CrossWorkspacePolicy,
    delete_message,
    read_inbox,
    write_to_inbox,
)
from src.api.db.engine import get_db
from src.api.db.repositories.workspaces import WorkspaceRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.schemas.common import APIResponse

log = logging.getLogger("cogtrix.api.cross_workspace")

router = APIRouter(prefix="/cross-workspace", tags=["Cross-Workspace"])

# Module-level policy instance (can be replaced via configure_cross_workspace_policy).
_policy = CrossWorkspacePolicy(enabled=True)


def configure_cross_workspace_policy(policy: CrossWorkspacePolicy) -> None:
    """Replace the active cross-workspace communication policy."""
    global _policy
    _policy = policy


def get_policy() -> CrossWorkspacePolicy:
    return _policy


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SendMessageRequest(BaseModel):
    from_workspace_id: str = Field(..., description="Source workspace UUID.")
    to_workspace_id: str = Field(..., description="Destination workspace UUID.")
    subject: str = Field(..., min_length=1, max_length=128)
    body: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/messages",
    response_model=APIResponse[dict],
    status_code=201,
    summary="Send cross-workspace message",
)
async def send_message(
    body: SendMessageRequest,
    ctx: OrgContext = Depends(require_org_context),
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:
    """Send a structured message from one workspace to another (same org only)."""
    if not _policy.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "CROSS_WS_DISABLED", "message": "Cross-workspace comms are disabled."},
        )

    repo = WorkspaceRepository(db)
    from_ws = await repo.get_by_id(body.from_workspace_id)
    to_ws = await repo.get_by_id(body.to_workspace_id)

    if from_ws is None or not from_ws.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Source workspace not found."},
        )
    if to_ws is None or not to_ws.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Destination workspace not found."},
        )

    # Same-org enforcement.
    if from_ws.org_id != to_ws.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "CROSS_ORG_BLOCKED",
                "message": "Cross-org communication is not permitted.",
            },
        )

    # Both workspaces must belong to the caller's org.
    if from_ws.org_id != ctx.org_id or to_ws.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "CROSS_ORG_BLOCKED",
                "message": "Workspace does not belong to your org.",
            },
        )

    sender_membership = await repo.get_membership(body.from_workspace_id, current_user.user_id)
    if sender_membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "You must belong to the source workspace to send messages.",
            },
        )

    # Policy pair check.
    if not _policy.is_allowed(body.from_workspace_id, body.to_workspace_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "POLICY_DENIED",
                "message": "Communication between these workspaces is not allowed.",
            },
        )

    msg = CrossWorkspaceMessage(
        from_workspace_id=body.from_workspace_id,
        to_workspace_id=body.to_workspace_id,
        sender_user_id=current_user.user_id,
        subject=body.subject,
        body=body.body,
    )
    write_to_inbox(msg)
    return APIResponse(data=msg.to_dict())


@router.get(
    "/inbox/{workspace_id}",
    response_model=APIResponse[list[dict]],
    summary="Read workspace inbox",
)
async def get_inbox(
    workspace_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[dict]]:
    """Return messages in the target workspace's inbox (newest first)."""
    repo = WorkspaceRepository(db)
    ws = await repo.get_by_id(workspace_id)
    if ws is None or ws.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Workspace not found."},
        )
    messages = read_inbox(workspace_id, limit=limit)
    return APIResponse(data=messages)


@router.delete(
    "/inbox/{workspace_id}/{message_id}",
    response_model=APIResponse[None],
    summary="Delete inbox message",
)
async def remove_message(
    workspace_id: str,
    message_id: str,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """Delete a message from the workspace inbox."""
    repo = WorkspaceRepository(db)
    ws = await repo.get_by_id(workspace_id)
    if ws is None or ws.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Workspace not found."},
        )
    deleted = delete_message(workspace_id, message_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Message not found."},
        )
    return APIResponse(data=None)
