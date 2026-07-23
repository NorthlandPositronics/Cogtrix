"""Tests for src/api/schemas/team.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.team import (
    AddMemberRequest,
    MemberOut,
    TeamCreate,
    TeamOut,
    TeamUpdate,
)


class TestTeamOut:
    """TeamOut schema construction and validation."""

    def test_team_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        team = TeamOut(
            id="team-123",
            org_id="org-456",
            name="Engineering",
            description="The engineering team.",
            member_count=5,
            created_at=now,
        )
        assert team.name == "Engineering"
        assert team.member_count == 5
        assert team.created_at == now

    def test_team_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        team = TeamOut(
            id="team-123",
            org_id="org-456",
            name="Engineering",
            created_at=naive,
        )
        assert team.created_at.tzinfo is not None

    def test_team_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            TeamOut(
                id="team-123",
                org_id="org-456",
                # name missing
                created_at=datetime.now(UTC),
            )


class TestTeamCreate:
    """TeamCreate schema construction and validation."""

    def test_team_create_valid(self) -> None:
        """Valid input constructs without error."""
        req = TeamCreate(name="Engineering", description="The engineering team.")
        assert req.name == "Engineering"
        assert req.description == "The engineering team."

    def test_team_create_no_description(self) -> None:
        """Description is optional."""
        req = TeamCreate(name="Engineering")
        assert req.description is None

    def test_team_create_empty_name(self) -> None:
        """Empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            TeamCreate(name="")

    def test_team_create_name_too_long(self) -> None:
        """Name over 128 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            TeamCreate(name="x" * 129)

    def test_team_create_description_too_long(self) -> None:
        """Description over 512 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            TeamCreate(name="Engineering", description="x" * 513)


class TestTeamUpdate:
    """TeamUpdate schema construction and validation."""

    def test_team_update_all_fields(self) -> None:
        """All optional fields can be set."""
        req = TeamUpdate(name="New Name", description="New desc.")
        assert req.name == "New Name"
        assert req.description == "New desc."

    def test_team_update_empty(self) -> None:
        """All fields can be omitted."""
        req = TeamUpdate()
        assert req.name is None
        assert req.description is None

    def test_team_update_empty_name(self) -> None:
        """Empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            TeamUpdate(name="")


class TestMemberOut:
    """MemberOut schema construction and validation."""

    def test_member_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        member = MemberOut(
            user_id="user-123",
            username="alice",
            email="alice@example.com",
            role="member",
            joined_at=now,
        )
        assert member.username == "alice"
        assert member.role == "member"

    def test_member_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        member = MemberOut(
            user_id="user-123",
            username="alice",
            email="alice@example.com",
            role="admin",
            joined_at=naive,
        )
        assert member.joined_at.tzinfo is not None


class TestAddMemberRequest:
    """AddMemberRequest schema construction and validation."""

    def test_add_member_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = AddMemberRequest(user_id="user-123", role="admin")
        assert req.user_id == "user-123"
        assert req.role == "admin"

    def test_add_member_request_default_role(self) -> None:
        """Default role is 'member'."""
        req = AddMemberRequest(user_id="user-123")
        assert req.role == "member"

    def test_add_member_request_invalid_role(self) -> None:
        """Invalid role raises ValidationError."""
        with pytest.raises(ValidationError, match="admin.*member"):
            AddMemberRequest(user_id="user-123", role="owner")

    def test_add_member_request_missing_user_id(self) -> None:
        """Missing user_id raises ValidationError."""
        with pytest.raises(ValidationError):
            AddMemberRequest(role="member")
