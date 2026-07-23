"""Tests for the permission model — resource × action matrix (issue #594)."""

from __future__ import annotations

import pytest

from src.auth.permissions import (
    ROLE_PERMISSIONS,
    Action,
    Permission,
    Resource,
    _perm,
    can,
    get_permissions,
    get_roles,
    has_permission,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestResourceEnum:
    def test_all_resources_defined(self) -> None:
        expected = {
            "sessions",
            "agents",
            "tools",
            "config",
            "users",
            "orgs",
            "billing",
            "audit",
            "assistant",
        }
        assert set(Resource) == expected

    def test_resource_values_are_strings(self) -> None:
        for r in Resource:
            assert isinstance(r.value, str)
            assert r.value == r.name.lower()


class TestActionEnum:
    def test_all_actions_defined(self) -> None:
        expected = {"create", "read", "update", "delete", "execute", "manage"}
        assert set(Action) == expected

    def test_action_values_are_strings(self) -> None:
        for a in Action:
            assert isinstance(a.value, str)
            assert a.value == a.name.lower()


# ---------------------------------------------------------------------------
# Permission constants
# ---------------------------------------------------------------------------


class TestPermissionConstants:
    def test_sessions_create_format(self) -> None:
        assert Permission.SESSIONS_CREATE == "sessions.create"

    def test_agents_read_format(self) -> None:
        assert Permission.AGENTS_READ == "agents.read"

    def test_billing_delete_format(self) -> None:
        assert Permission.BILLING_DELETE == "billing.delete"

    def test_all_constants_unique(self) -> None:
        values = [v for k, v in vars(Permission).items() if not k.startswith("_")]
        assert len(values) == len(set(values)), "duplicate permission constant values"

    def test_permission_count(self) -> None:
        """9 resources × 6 actions = 54 constants."""
        values = [v for k, v in vars(Permission).items() if not k.startswith("_")]
        assert len(values) == 54


class TestPermHelper:
    def test_format(self) -> None:
        assert _perm(Resource.SESSIONS, Action.CREATE) == "sessions.create"

    def test_all_resources_and_actions_produce_unique_strings(self) -> None:
        perms = {_perm(r, a) for r in Resource for a in Action}
        assert len(perms) == len(Resource) * len(Action)


# ---------------------------------------------------------------------------
# ROLE_PERMISSIONS matrix
# ---------------------------------------------------------------------------


_ALL_ROLES = ["superadmin", "admin", "member", "viewer", "readonly"]


class TestRolePermissionsMatrix:
    def test_exactly_five_roles_defined(self) -> None:
        assert set(ROLE_PERMISSIONS.keys()) == set(_ALL_ROLES)

    @pytest.mark.parametrize("role", _ALL_ROLES)
    def test_every_role_has_permissions(self, role: str) -> None:
        assert len(ROLE_PERMISSIONS[role]) > 0, f"{role} has no permissions"


class TestSuperadmin:
    def test_has_all_permissions(self) -> None:
        all_constants = {v for k, v in vars(Permission).items() if not k.startswith("_")}
        assert ROLE_PERMISSIONS["superadmin"] == all_constants

    def test_can_delete_billing(self) -> None:
        assert Permission.BILLING_DELETE in ROLE_PERMISSIONS["superadmin"]


class TestAdmin:
    def test_cannot_delete_billing(self) -> None:
        assert Permission.BILLING_DELETE not in ROLE_PERMISSIONS["admin"]

    def test_has_sessions_manage(self) -> None:
        assert Permission.SESSIONS_MANAGE in ROLE_PERMISSIONS["admin"]

    def test_has_users_manage(self) -> None:
        assert Permission.USERS_MANAGE in ROLE_PERMISSIONS["admin"]

    def test_has_all_other_permissions(self) -> None:
        all_constants = {v for k, v in vars(Permission).items() if not k.startswith("_")}
        expected = all_constants - {Permission.BILLING_DELETE}
        assert ROLE_PERMISSIONS["admin"] == expected


class TestMember:
    def test_can_create_sessions(self) -> None:
        assert Permission.SESSIONS_CREATE in ROLE_PERMISSIONS["member"]

    def test_can_execute_tools(self) -> None:
        assert Permission.TOOLS_EXECUTE in ROLE_PERMISSIONS["member"]

    def test_cannot_manage_users(self) -> None:
        assert Permission.USERS_MANAGE not in ROLE_PERMISSIONS["member"]

    def test_cannot_read_billing(self) -> None:
        assert Permission.BILLING_READ not in ROLE_PERMISSIONS["member"]

    def test_cannot_access_audit(self) -> None:
        assert Permission.AUDIT_READ not in ROLE_PERMISSIONS["member"]

    def test_can_read_sessions(self) -> None:
        assert Permission.SESSIONS_READ in ROLE_PERMISSIONS["member"]

    def test_can_delete_own_agents(self) -> None:
        assert Permission.AGENTS_DELETE in ROLE_PERMISSIONS["member"]


class TestViewer:
    def test_has_read_only_permissions(self) -> None:
        for perm in ROLE_PERMISSIONS["viewer"]:
            assert perm.endswith(".read"), f"viewer has non-read permission: {perm}"

    def test_cannot_create(self) -> None:
        assert Permission.SESSIONS_CREATE not in ROLE_PERMISSIONS["viewer"]

    def test_cannot_delete(self) -> None:
        assert Permission.AGENTS_DELETE not in ROLE_PERMISSIONS["viewer"]

    def test_reads_all_resources(self) -> None:
        for resource in Resource:
            assert (
                _perm(resource, Action.READ) in ROLE_PERMISSIONS["viewer"]
            ), f"viewer missing read on {resource.value}"


class TestReadonly:
    def test_no_billing_read(self) -> None:
        assert Permission.BILLING_READ not in ROLE_PERMISSIONS["readonly"]

    def test_no_audit_read(self) -> None:
        assert Permission.AUDIT_READ not in ROLE_PERMISSIONS["readonly"]

    def test_has_sessions_read(self) -> None:
        assert Permission.SESSIONS_READ in ROLE_PERMISSIONS["readonly"]

    def test_has_config_read(self) -> None:
        assert Permission.CONFIG_READ in ROLE_PERMISSIONS["readonly"]

    def test_no_write_permissions(self) -> None:
        for perm in ROLE_PERMISSIONS["readonly"]:
            assert perm.endswith(".read"), f"readonly has non-read permission: {perm}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHasPermission:
    def test_superadmin_has_any(self) -> None:
        assert has_permission("superadmin", Permission.SESSIONS_CREATE) is True
        assert has_permission("superadmin", Permission.BILLING_DELETE) is True

    def test_unknown_role_denied(self) -> None:
        assert has_permission("bogus", Permission.SESSIONS_READ) is False

    def test_none_role_denied(self) -> None:
        assert has_permission("", Permission.SESSIONS_READ) is False

    def test_member_denied_billing(self) -> None:
        assert has_permission("member", Permission.BILLING_READ) is False

    def test_admin_denied_billing_delete(self) -> None:
        assert has_permission("admin", Permission.BILLING_DELETE) is False


class TestCan:
    def test_admin_can_read_sessions(self) -> None:
        assert can("admin", Resource.SESSIONS, Action.READ) is True

    def test_member_cannot_manage_users(self) -> None:
        assert can("member", Resource.USERS, Action.MANAGE) is False

    def test_viewer_can_read_config(self) -> None:
        assert can("viewer", Resource.CONFIG, Action.READ) is True

    def test_viewer_cannot_create_sessions(self) -> None:
        assert can("viewer", Resource.SESSIONS, Action.CREATE) is False


class TestGetPermissions:
    def test_returns_copy(self) -> None:
        p1 = get_permissions("viewer")
        p2 = get_permissions("viewer")
        p1.add("fake.perm")
        assert "fake.perm" not in p2

    def test_unknown_role_returns_empty(self) -> None:
        assert get_permissions("nonexistent") == set()


class TestGetRoles:
    def test_returns_all_five(self) -> None:
        roles = get_roles()
        assert len(roles) == 5
        assert "superadmin" in roles
        assert "readonly" in roles
