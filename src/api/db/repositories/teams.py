"""Team repository — CRUD and membership operations for Team and TeamMembership models."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.db.models import Team, TeamMembership, User


class TeamRepository:
    """Data access layer for Team and TeamMembership records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        team_id: str,
        org_id: str,
        name: str,
        description: str | None = None,
    ) -> Team:
        """Insert a new Team row and return it."""
        team = Team(id=team_id, org_id=org_id, name=name, description=description)
        self._db.add(team)
        await self._db.flush()
        await self._db.refresh(team)
        return team

    async def get_by_id(self, team_id: str) -> Team | None:
        """Return the team with the given UUID, or None."""
        result = await self._db.execute(select(Team).where(Team.id == team_id))
        return result.scalar_one_or_none()

    async def get_by_name_and_org(self, name: str, org_id: str) -> Team | None:
        """Return the team with the given name within an org, or None."""
        result = await self._db.execute(
            select(Team).where(Team.name == name, Team.org_id == org_id)
        )
        return result.scalar_one_or_none()

    async def list_by_org(self, org_id: str) -> list[Team]:
        """Return all teams belonging to the given org."""
        result = await self._db.execute(
            select(Team).where(Team.org_id == org_id).order_by(Team.created_at)
        )
        return list(result.scalars().all())

    async def update(
        self,
        team_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Team | None:
        """Partially update a team; return the updated row or None."""
        team = await self.get_by_id(team_id)
        if team is None:
            return None
        if name is not None:
            team.name = name
        if description is not None:
            team.description = description
        await self._db.flush()
        await self._db.refresh(team)
        return team

    async def delete(self, team_id: str) -> bool:
        """Delete the team and all its memberships; return True if deleted."""
        stmt = delete(Team).where(Team.id == team_id)
        result = await self._db.execute(stmt)
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Memberships
    # ------------------------------------------------------------------

    async def add_member(
        self,
        *,
        membership_id: str,
        team_id: str,
        user_id: str,
        role: str = "member",
    ) -> TeamMembership:
        """Add a user to a team; return the new membership."""
        membership = TeamMembership(id=membership_id, team_id=team_id, user_id=user_id, role=role)
        self._db.add(membership)
        await self._db.flush()
        await self._db.refresh(membership)
        return membership

    async def get_membership(self, team_id: str, user_id: str) -> TeamMembership | None:
        """Return the membership for a specific team/user pair, or None."""
        result = await self._db.execute(
            select(TeamMembership).where(
                TeamMembership.team_id == team_id, TeamMembership.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def list_members(self, team_id: str) -> list[User]:
        """Return all users that are members of the given team."""
        result = await self._db.execute(
            select(User)
            .join(TeamMembership, TeamMembership.user_id == User.id)
            .where(TeamMembership.team_id == team_id)
            .order_by(TeamMembership.joined_at)
        )
        return list(result.scalars().all())

    async def list_memberships(self, team_id: str) -> list[TeamMembership]:
        """Return all membership records for a team (includes role info)."""
        result = await self._db.execute(
            select(TeamMembership)
            .where(TeamMembership.team_id == team_id)
            .order_by(TeamMembership.joined_at)
        )
        return list(result.scalars().all())

    async def remove_member(self, team_id: str, user_id: str) -> bool:
        """Remove a user from a team; return True if a row was deleted."""
        stmt = delete(TeamMembership).where(
            TeamMembership.team_id == team_id, TeamMembership.user_id == user_id
        )
        result = await self._db.execute(stmt)
        return result.rowcount > 0

    async def count_members(self, team_id: str) -> int:
        """Return the number of members in a team."""
        from sqlalchemy import func

        result = await self._db.execute(
            select(func.count())
            .select_from(TeamMembership)
            .where(TeamMembership.team_id == team_id)
        )
        return result.scalar_one()
