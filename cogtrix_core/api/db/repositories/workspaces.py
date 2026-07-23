"""Workspace repository — CRUD and membership operations."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cogtrix_core.api.db.models import User, Workspace, WorkspaceMembership


class WorkspaceRepository:
    """Data access layer for Workspace and WorkspaceMembership records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        workspace_id: str,
        org_id: str,
        name: str,
        description: str | None = None,
        settings: str | None = None,
    ) -> Workspace:
        ws = Workspace(
            id=workspace_id,
            org_id=org_id,
            name=name,
            description=description,
            settings=settings,
        )
        self._db.add(ws)
        await self._db.flush()
        await self._db.refresh(ws)
        return ws

    async def get_by_id(self, workspace_id: str) -> Workspace | None:
        result = await self._db.execute(select(Workspace).where(Workspace.id == workspace_id))
        return result.scalar_one_or_none()

    async def get_by_name_and_org(self, name: str, org_id: str) -> Workspace | None:
        result = await self._db.execute(
            select(Workspace).where(Workspace.name == name, Workspace.org_id == org_id)
        )
        return result.scalar_one_or_none()

    async def list_by_org(self, org_id: str, *, include_inactive: bool = False) -> list[Workspace]:
        stmt = select(Workspace).where(Workspace.org_id == org_id)
        if not include_inactive:
            stmt = stmt.where(Workspace.is_active.is_(True))
        stmt = stmt.order_by(Workspace.created_at)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        workspace_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        settings: str | None = None,
        is_active: bool | None = None,
    ) -> Workspace | None:
        ws = await self.get_by_id(workspace_id)
        if ws is None:
            return None
        if name is not None:
            ws.name = name
        if description is not None:
            ws.description = description
        if settings is not None:
            ws.settings = settings
        if is_active is not None:
            ws.is_active = is_active
        await self._db.flush()
        await self._db.refresh(ws)
        return ws

    async def delete(self, workspace_id: str) -> bool:
        result = await self._db.execute(delete(Workspace).where(Workspace.id == workspace_id))
        return result.rowcount > 0

    async def add_member(
        self,
        *,
        membership_id: str,
        workspace_id: str,
        user_id: str,
        role: str = "member",
    ) -> WorkspaceMembership:
        m = WorkspaceMembership(
            id=membership_id, workspace_id=workspace_id, user_id=user_id, role=role
        )
        self._db.add(m)
        await self._db.flush()
        await self._db.refresh(m)
        return m

    async def get_membership(self, workspace_id: str, user_id: str) -> WorkspaceMembership | None:
        result = await self._db.execute(
            select(WorkspaceMembership).where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_memberships(self, workspace_id: str) -> list[WorkspaceMembership]:
        result = await self._db.execute(
            select(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == workspace_id)
            .order_by(WorkspaceMembership.joined_at)
        )
        return list(result.scalars().all())

    async def list_members(self, workspace_id: str) -> list[User]:
        result = await self._db.execute(
            select(User)
            .join(WorkspaceMembership, WorkspaceMembership.user_id == User.id)
            .where(WorkspaceMembership.workspace_id == workspace_id)
            .order_by(WorkspaceMembership.joined_at)
        )
        return list(result.scalars().all())

    async def remove_member(self, workspace_id: str, user_id: str) -> bool:
        result = await self._db.execute(
            delete(WorkspaceMembership).where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user_id,
            )
        )
        return result.rowcount > 0

    async def count_members(self, workspace_id: str) -> int:
        from sqlalchemy import func

        result = await self._db.execute(
            select(func.count())
            .select_from(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == workspace_id)
        )
        return result.scalar_one()

    async def list_for_user(self, user_id: str, org_id: str) -> list[Workspace]:
        """Return all active workspaces the user is a member of (within the org)."""
        result = await self._db.execute(
            select(Workspace)
            .join(WorkspaceMembership, WorkspaceMembership.workspace_id == Workspace.id)
            .where(
                WorkspaceMembership.user_id == user_id,
                Workspace.org_id == org_id,
                Workspace.is_active.is_(True),
            )
            .order_by(Workspace.created_at)
        )
        return list(result.scalars().all())
