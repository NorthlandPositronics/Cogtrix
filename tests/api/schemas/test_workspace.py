"""Tests for cogtrix_core/api/schemas/workspace.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.workspace import (
    AddWorkspaceMemberRequest,
    WorkspaceCreate,
    WorkspaceMemberOut,
    WorkspaceOut,
    WorkspaceUpdate,
)


class TestWorkspaceOut:
    """WorkspaceOut schema construction and validation."""

    def test_workspace_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        ws = WorkspaceOut(
            id="ws-123",
            org_id="org-456",
            name="Marketing",
            description="Marketing workspace.",
            settings={"theme": "dark"},
            member_count=3,
            is_active=True,
            created_at=now,
        )
        assert ws.name == "Marketing"
        assert ws.settings == {"theme": "dark"}

    def test_workspace_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        ws = WorkspaceOut(
            id="ws-123",
            org_id="org-456",
            name="Marketing",
            is_active=True,
            created_at=naive,
        )
        assert ws.created_at.tzinfo is not None

    def test_workspace_out_settings_from_json_string(self) -> None:
        """Settings parsed from JSON string."""
        ws = WorkspaceOut(
            id="ws-123",
            org_id="org-456",
            name="Marketing",
            settings='{"theme": "dark"}',
            is_active=True,
            created_at=datetime.now(UTC),
        )
        assert ws.settings == {"theme": "dark"}

    def test_workspace_out_settings_empty_string(self) -> None:
        """Empty string settings become None."""
        ws = WorkspaceOut(
            id="ws-123",
            org_id="org-456",
            name="Marketing",
            settings="",
            is_active=True,
            created_at=datetime.now(UTC),
        )
        assert ws.settings is None

    def test_workspace_out_settings_none(self) -> None:
        """None settings stay None."""
        ws = WorkspaceOut(
            id="ws-123",
            org_id="org-456",
            name="Marketing",
            settings=None,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        assert ws.settings is None

    def test_workspace_out_invalid_json_settings(self) -> None:
        """Invalid JSON string raises ValidationError."""
        with pytest.raises(ValidationError):
            WorkspaceOut(
                id="ws-123",
                org_id="org-456",
                name="Marketing",
                settings="not-json",
                is_active=True,
                created_at=datetime.now(UTC),
            )


class TestWorkspaceCreate:
    """WorkspaceCreate schema construction and validation."""

    def test_workspace_create_valid(self) -> None:
        """Valid input constructs without error."""
        req = WorkspaceCreate(name="Marketing", description="Marketing workspace.")
        assert req.name == "Marketing"

    def test_workspace_create_empty_name(self) -> None:
        """Empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            WorkspaceCreate(name="")

    def test_workspace_create_name_too_long(self) -> None:
        """Name over 128 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            WorkspaceCreate(name="x" * 129)

    def test_workspace_create_description_too_long(self) -> None:
        """Description over 512 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            WorkspaceCreate(name="Marketing", description="x" * 513)


class TestWorkspaceUpdate:
    """WorkspaceUpdate schema construction and validation."""

    def test_workspace_update_all_fields(self) -> None:
        """All optional fields can be set."""
        req = WorkspaceUpdate(
            name="New Name", description="New desc.", settings={"key": "val"}, is_active=False
        )
        assert req.name == "New Name"
        assert req.is_active is False

    def test_workspace_update_empty(self) -> None:
        """All fields can be omitted."""
        req = WorkspaceUpdate()
        assert req.name is None
        assert req.is_active is None


class TestWorkspaceMemberOut:
    """WorkspaceMemberOut schema construction and validation."""

    def test_workspace_member_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        member = WorkspaceMemberOut(
            user_id="user-123",
            username="alice",
            email="alice@example.com",
            role="admin",
            joined_at=now,
        )
        assert member.username == "alice"

    def test_workspace_member_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        member = WorkspaceMemberOut(
            user_id="user-123",
            username="alice",
            email="alice@example.com",
            role="member",
            joined_at=naive,
        )
        assert member.joined_at.tzinfo is not None


class TestAddWorkspaceMemberRequest:
    """AddWorkspaceMemberRequest schema construction and validation."""

    def test_add_workspace_member_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = AddWorkspaceMemberRequest(user_id="user-123", role="admin")
        assert req.role == "admin"

    def test_add_workspace_member_request_default_role(self) -> None:
        """Default role is 'member'."""
        req = AddWorkspaceMemberRequest(user_id="user-123")
        assert req.role == "member"

    def test_add_workspace_member_request_invalid_role(self) -> None:
        """Invalid role raises ValidationError."""
        with pytest.raises(ValidationError, match="member.*admin"):
            AddWorkspaceMemberRequest(user_id="user-123", role="owner")
