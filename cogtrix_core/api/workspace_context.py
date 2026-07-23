"""Workspace context resolution (Enterprise Phase 1 — task 1.3.2).

Provides ``get_workspace_context`` — a FastAPI dependency that resolves
the active workspace for a request, with its typed config overlay.
Config precedence: global defaults → org settings → workspace settings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.auth import TokenData, get_current_user
from cogtrix_core.api.db.engine import get_db
from cogtrix_core.api.db.repositories.workspaces import WorkspaceRepository
from cogtrix_core.api.org_context import OrgContext, get_org_context

log = logging.getLogger("cogtrix.api.workspace_context")

# ---------------------------------------------------------------------------
# WorkspaceConfig — typed overlay for per-workspace settings
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceConfig:
    """Typed configuration overlay for a workspace.

    Fields are all optional — absent fields fall back to org or global defaults.

    Attributes:
        model_override:       LLM model alias for sessions in this workspace.
        system_prompt:        System prompt prepended to all sessions.
        tool_policy:          ``"all"`` | ``"none"`` | comma-separated tool names.
        max_context_tokens:   Context window override.
        rate_limit_multiplier: Multiply the org-level rate limit by this factor.
        extra:                Arbitrary additional settings (extension point).
    """

    model_override: str | None = field(default=None)
    system_prompt: str | None = field(default=None)
    tool_policy: str | None = field(default=None)
    max_context_tokens: int | None = field(default=None)
    rate_limit_multiplier: float = field(default=1.0)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str | None) -> WorkspaceConfig:
        """Parse a ``WorkspaceConfig`` from the workspace's settings JSON blob."""
        if not raw:
            return cls()
        try:
            data: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return cls()
        return cls(
            model_override=data.get("model_override"),
            system_prompt=data.get("system_prompt"),
            tool_policy=data.get("tool_policy"),
            max_context_tokens=data.get("max_context_tokens"),
            rate_limit_multiplier=float(data.get("rate_limit_multiplier", 1.0)),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "model_override",
                    "system_prompt",
                    "tool_policy",
                    "max_context_tokens",
                    "rate_limit_multiplier",
                }
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for storage in the settings JSON blob."""
        d: dict[str, Any] = {}
        if self.model_override is not None:
            d["model_override"] = self.model_override
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.tool_policy is not None:
            d["tool_policy"] = self.tool_policy
        if self.max_context_tokens is not None:
            d["max_context_tokens"] = self.max_context_tokens
        if self.rate_limit_multiplier != 1.0:
            d["rate_limit_multiplier"] = self.rate_limit_multiplier
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# WorkspaceContext
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceContext:
    """Resolved workspace context for the current request.

    Attributes:
        user_id:    UUID of the authenticated user.
        org_id:     UUID of the user's organization (or None).
        workspace_id: UUID of the resolved workspace (or None).
        config:     Typed workspace config overlay.
    """

    user_id: str
    org_id: str | None = field(default=None)
    workspace_id: str | None = field(default=None)
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)

    @property
    def has_workspace(self) -> bool:
        return self.workspace_id is not None


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def get_workspace_context(
    workspace_id: str | None = Query(
        default=None,
        alias="workspace_id",
        description="Active workspace UUID.  If omitted, no workspace context is applied.",
    ),
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(get_org_context),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceContext:
    """FastAPI dependency: resolve the active workspace for the current request.

    The caller passes ``?workspace_id=<uuid>`` to select a workspace.  The
    workspace must exist, belong to the caller's org, and the caller must be
    a member (or an admin).

    Returns a ``WorkspaceContext`` with ``workspace_id=None`` when no
    workspace is requested.
    """
    if workspace_id is None:
        return WorkspaceContext(user_id=current_user.user_id, org_id=ctx.org_id)

    repo = WorkspaceRepository(db)
    ws = await repo.get_by_id(workspace_id)

    if ws is None or not ws.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Workspace not found."},
        )

    if ctx.org_id is not None and ws.org_id != ctx.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "CROSS_ORG_ACCESS",
                "message": "Workspace does not belong to your org.",
            },
        )

    # Non-admins must be a workspace member.
    if not ctx.is_admin:
        membership = await repo.get_membership(workspace_id, current_user.user_id)
        if membership is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "NOT_A_MEMBER",
                    "message": "You are not a member of this workspace.",
                },
            )

    config = WorkspaceConfig.from_json(ws.settings)
    log.debug("Workspace context resolved: ws=%s model=%s", workspace_id, config.model_override)
    return WorkspaceContext(
        user_id=current_user.user_id,
        org_id=ctx.org_id,
        workspace_id=workspace_id,
        config=config,
    )
