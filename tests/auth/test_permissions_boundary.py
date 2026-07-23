"""Boundary and integration tests for the RBAC permission model (issue #598).

Covers: parametrized cross-role checks, org isolation simulation,
route guard integration, and permission matrix exploration.
"""

from __future__ import annotations

import pytest

from src.auth.permissions import (
    ROLE_PERMISSIONS,
    Action,
    Permission,
    Resource,
    can,
    get_permissions,
    has_permission,
)
from tests.auth.conftest import (
    ALL_ACTIONS,
    ALL_PERMISSIONS,
    ALL_RESOURCES,
    ALL_ROLES,
    NON_ADMIN_ROLES,
    READ_ONLY_ROLES,
    RouteGuard,
    SimulatedOrg,
    SimulatedUser,
    get_permission_diff,
    get_role_permission_summary,
)

# ---------------------------------------------------------------------------
# Parametrized: all roles × all resources × all actions
# ---------------------------------------------------------------------------


class TestFullMatrix:
    """Exhaustive permission check across all 5 roles × 9 resources × 6 actions."""

    def test_full_matrix_no_crashes(self) -> None:
        """Every role/resource/action combination must return bool without crashing."""
        for role in ALL_ROLES:
            for resource in ALL_RESOURCES:
                for action in ALL_ACTIONS:
                    result = can(role, resource, action)
                    assert isinstance(
                        result, bool
                    ), f"can({role}, {resource}, {action}) returned {type(result)}"

    def test_full_matrix_has_permission_no_crashes(self) -> None:
        """Every role/permission combination must return bool without crashing."""
        for role in ALL_ROLES:
            for perm in ALL_PERMISSIONS:
                result = has_permission(role, perm)
                assert isinstance(result, bool)

    def test_can_and_has_permission_are_consistent(self) -> None:
        """can() must be consistent with has_permission() for the same resource.action."""
        for role in ALL_ROLES:
            for resource in ALL_RESOURCES:
                for action in ALL_ACTIONS:
                    perm_str = f"{resource.value}.{action.value}"
                    assert can(role, resource, action) == has_permission(role, perm_str), (
                        f"Inconsistency: can({role}, {resource}, {action}) != "
                        f"has_permission({role}, {perm_str})"
                    )

    def test_superadmin_has_everything(self) -> None:
        """Superadmin must pass every can() and has_permission() check."""
        for resource in ALL_RESOURCES:
            for action in ALL_ACTIONS:
                assert can("superadmin", resource, action) is True
        for perm in ALL_PERMISSIONS:
            assert has_permission("superadmin", perm) is True

    def test_readonly_read_boundaries(self) -> None:
        """Readonly must only have read access to 7 resources (no billing, no audit)."""
        readonly_perms = get_permissions("readonly")
        for resource in ALL_RESOURCES:
            for action in ALL_ACTIONS:
                perm_str = f"{resource.value}.{action.value}"
                if action == Action.READ and resource.value not in ("billing", "audit"):
                    assert perm_str in readonly_perms, f"readonly should have {perm_str}"
                else:
                    assert perm_str not in readonly_perms, f"readonly should NOT have {perm_str}"


# ---------------------------------------------------------------------------
# Permission counts per role
# ---------------------------------------------------------------------------


class TestPermissionCounts:
    """Verify exact permission counts for each role."""

    @pytest.mark.parametrize(
        "role,expected_count",
        [
            ("superadmin", 54),
            ("admin", 53),
            ("member", 20),
            ("viewer", 9),
            ("readonly", 7),
        ],
    )
    def test_role_permission_count(self, role: str, expected_count: int) -> None:
        assert (
            len(ROLE_PERMISSIONS[role]) == expected_count
        ), f"{role}: expected {expected_count}, got {len(ROLE_PERMISSIONS[role])}"


# ---------------------------------------------------------------------------
# Cross-role permission diffs
# ---------------------------------------------------------------------------


class TestPermissionDiffs:
    """Verify permission differences between adjacent roles."""

    def test_admin_vs_member_diff(self) -> None:
        diff = get_permission_diff("admin", "member")
        admin_only = diff["only_admin"]
        member_only = diff["only_member"]
        # Admin should have strictly more permissions than member
        assert len(admin_only) > 0, "admin should have extra permissions"
        assert len(member_only) == 0, f"member has extra permissions: {member_only}"

    def test_member_vs_viewer_diff(self) -> None:
        diff = get_permission_diff("member", "viewer")
        assert len(diff["only_member"]) > 0
        assert Permission.BILLING_READ in diff["only_viewer"]
        assert Permission.AUDIT_READ in diff["only_viewer"]
        # viewer has billing.read and audit.read which member lacks (by design)
        assert (
            len(diff["only_viewer"]) == 2
        ), f"viewer has permissions member does not: {diff['only_viewer']}"

    def test_viewer_vs_readonly_diff(self) -> None:
        diff = get_permission_diff("viewer", "readonly")
        viewer_only = diff["only_viewer"]
        # Viewer should have billing.read and audit.read that readonly lacks
        assert Permission.BILLING_READ in viewer_only
        assert Permission.AUDIT_READ in viewer_only

    def test_superadmin_vs_admin_diff(self) -> None:
        diff = get_permission_diff("superadmin", "admin")
        assert diff["only_superadmin"] == {Permission.BILLING_DELETE}
        assert len(diff["only_admin"]) == 0


# ---------------------------------------------------------------------------
# Route guard integration tests
# ---------------------------------------------------------------------------


class TestRouteGuard:
    """Integration tests for the simulated route guard (stand-in for middleware)."""

    def test_guard_allows_authorized_access(self, admin) -> None:
        """Admin should pass require() for admin-level resources."""
        RouteGuard.require(admin, Resource.SESSIONS, Action.CREATE)

    def test_guard_blocks_unauthorized_access(self, member) -> None:
        """Member should be blocked from admin-only operations."""
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require(member, Resource.USERS, Action.MANAGE)

    def test_guard_require_admin_allows_superadmin(self, superadmin) -> None:
        RouteGuard.require_admin(superadmin)

    def test_guard_require_admin_allows_admin(self, admin) -> None:
        RouteGuard.require_admin(admin)

    def test_guard_require_admin_blocks_member(self, member) -> None:
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require_admin(member)

    def test_guard_require_admin_blocks_viewer(self, viewer) -> None:
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require_admin(viewer)

    def test_guard_require_admin_blocks_readonly(self, readonly) -> None:
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require_admin(readonly)

    def test_guard_forbidden_message_contains_context(self, member) -> None:
        """Error message should include user ID and role for auditability."""
        try:
            RouteGuard.require(member, Resource.USERS, Action.MANAGE)
        except RouteGuard.Forbidden as exc:
            msg = str(exc)
            assert member.user_id in msg
            assert member.role in msg
            assert "users.manage" in msg

    # Parametrized: every non-admin role blocked from admin endpoints
    @pytest.mark.parametrize("role", NON_ADMIN_ROLES)
    def test_non_admin_blocked_from_user_management(self, role) -> None:
        user = SimulatedUser(user_id=f"u-{role}", role=role)
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require(user, Resource.USERS, Action.MANAGE)

    @pytest.mark.parametrize("role", NON_ADMIN_ROLES)
    def test_non_admin_blocked_from_config_management(self, role) -> None:
        user = SimulatedUser(user_id=f"u-{role}", role=role)
        with pytest.raises(RouteGuard.Forbidden):
            RouteGuard.require(user, Resource.CONFIG, Action.UPDATE)

    @pytest.mark.parametrize("role", READ_ONLY_ROLES)
    def test_readonly_roles_blocked_from_write_operations(self, role) -> None:
        user = SimulatedUser(user_id=f"u-{role}", role=role)
        for resource in [Resource.SESSIONS, Resource.AGENTS, Resource.TOOLS]:
            with pytest.raises(RouteGuard.Forbidden):
                RouteGuard.require(user, resource, Action.CREATE)


# ---------------------------------------------------------------------------
# Cross-org isolation tests
# ---------------------------------------------------------------------------


class TestCrossOrgIsolation:
    """Verify that role permissions are scoped within organizations."""

    def test_two_orgs_have_distinct_users(self, two_orgs) -> None:
        org_a, org_b = two_orgs
        assert org_a.org_id != org_b.org_id
        # Users in org A should have org A's ID
        for user in org_a.users:
            assert user.org_id == org_a.org_id
        # Users in org B should have org B's ID
        for user in org_b.users:
            assert user.org_id == org_b.org_id

    def test_same_role_different_orgs(self, two_orgs) -> None:
        """Users with the same role in different orgs should have identical permissions."""
        org_a, org_b = two_orgs
        admin_a = next(u for u in org_a.users if u.role == "admin")
        admin_b = next(u for u in org_b.users if u.role == "admin")
        assert admin_a.permissions == admin_b.permissions
        assert admin_a.is_admin is True
        assert admin_b.is_admin is True

    def test_cross_org_permission_consistency(self, two_orgs) -> None:
        """Permission checks should be role-based, not org-based."""
        org_a, org_b = two_orgs
        member_a = next(u for u in org_a.users if u.role == "member")
        member_b = next(u for u in org_b.users if u.role == "member")
        # Both members should have the same permission profile
        assert member_a.can(Resource.SESSIONS, Action.CREATE) == member_b.can(
            Resource.SESSIONS, Action.CREATE
        )
        assert member_a.can(Resource.USERS, Action.MANAGE) == member_b.can(
            Resource.USERS, Action.MANAGE
        )

    def test_org_context_does_not_leak_permissions(self) -> None:
        """A user in one org must not gain permissions from another org."""
        org_a = SimulatedOrg(org_id="org-a", name="Alpha")
        org_b = SimulatedOrg(org_id="org-b", name="Beta")
        user_a = org_a.add_user("member")
        user_b = org_b.add_user("admin")
        # user_a (member in org_a) should NOT have admin permissions
        assert not user_a.can(Resource.USERS, Action.MANAGE)
        # user_b (admin in org_b) should have admin permissions
        assert user_b.can(Resource.USERS, Action.MANAGE)


# ---------------------------------------------------------------------------
# Permission Matrix audit report generation
# ---------------------------------------------------------------------------


class TestAuditReport:
    """Generate and validate RBAC audit reports."""

    def test_role_summary_is_complete(self) -> None:
        """get_role_permission_summary must cover all 5 roles."""
        summary = get_role_permission_summary()
        assert set(summary.keys()) == set(ALL_ROLES)

    def test_role_summary_has_required_fields(self) -> None:
        summary = get_role_permission_summary()
        required_fields = {"total_permissions", "resources", "actions_per_resource", "permissions"}
        for role, data in summary.items():
            for field in required_fields:
                assert field in data, f"{role} missing field: {field}"

    def test_role_summary_permissions_are_sorted(self) -> None:
        summary = get_role_permission_summary()
        for role, data in summary.items():
            assert data["permissions"] == sorted(
                data["permissions"]
            ), f"{role} permissions are not sorted"
            assert data["resources"] == sorted(
                data["resources"]
            ), f"{role} resources are not sorted"

    def test_superadmin_summary_matches_all_resources(self) -> None:
        summary = get_role_permission_summary()
        su = summary["superadmin"]
        assert su["total_permissions"] == 54
        assert set(su["resources"]) == {r.value for r in Resource}

    def test_viewer_summary_is_read_only(self) -> None:
        summary = get_role_permission_summary()
        v = summary["viewer"]
        for resource, actions in v["actions_per_resource"].items():
            assert actions == ["read"], f"viewer on {resource} has non-read actions: {actions}"


# ---------------------------------------------------------------------------
# SimulatedUser fixture integration tests
# ---------------------------------------------------------------------------


class TestSimulatedUserFixtures:
    """Verify that test fixtures are correctly configured."""

    def test_all_users_have_distinct_roles(self, all_users) -> None:
        roles = [u.role for u in all_users]
        assert roles == ALL_ROLES

    def test_admin_users_have_is_admin_true(self, admin, superadmin) -> None:
        assert admin.is_admin is True
        assert superadmin.is_admin is True

    def test_non_admin_users_have_is_admin_false(self, member, viewer, readonly) -> None:
        assert member.is_admin is False
        assert viewer.is_admin is False
        assert readonly.is_admin is False

    def test_superadmin_permissions_are_complete(self, superadmin) -> None:
        assert len(superadmin.permissions) == 54

    def test_member_has_expected_execute_permissions(self, member) -> None:
        assert member.has(Permission.SESSIONS_EXECUTE) is True
        assert member.has(Permission.AGENTS_EXECUTE) is True
        assert member.has(Permission.TOOLS_EXECUTE) is True
        assert member.has(Permission.ASSISTANT_EXECUTE) is True
        assert member.has(Permission.CONFIG_EXECUTE) is False

    def test_viewer_has_no_write_permissions(self, viewer) -> None:
        for perm in viewer.permissions:
            assert perm.endswith(".read"), f"viewer has non-read: {perm}"


# ---------------------------------------------------------------------------
# Edge cases from harness
# ---------------------------------------------------------------------------


class TestHarnessEdgeCases:
    """Edge-case tests for the test harness itself."""

    def test_unknown_role_user_has_no_permissions(self) -> None:
        user = SimulatedUser(user_id="bogus", role="nonexistent")
        assert len(user.permissions) == 0
        assert user.is_admin is False
        assert user.can(Resource.SESSIONS, Action.READ) is False
        assert user.has(Permission.SESSIONS_READ) is False

    def test_empty_role_user_has_no_permissions(self) -> None:
        user = SimulatedUser(user_id="empty", role="")
        assert len(user.permissions) == 0
        assert user.is_admin is False

    def test_permission_diff_reflexive_is_empty(self) -> None:
        """A role diffed against itself should have no differences."""
        for role in ALL_ROLES:
            diff = get_permission_diff(role, role)
            assert diff[f"only_{role}"] == set()
            assert diff["shared"] == get_permissions(role)


# ---------------------------------------------------------------------------
# Parametrized can() tests for key operations
# ---------------------------------------------------------------------------


class TestParametrizedCan:
    """Parametrized can() checks for frequently audited operations."""

    @pytest.mark.parametrize(
        "role,resource,action,expected",
        [
            # Session operations
            ("superadmin", Resource.SESSIONS, Action.CREATE, True),
            ("admin", Resource.SESSIONS, Action.CREATE, True),
            ("member", Resource.SESSIONS, Action.CREATE, True),
            ("viewer", Resource.SESSIONS, Action.CREATE, False),
            ("readonly", Resource.SESSIONS, Action.CREATE, False),
            # User management
            ("superadmin", Resource.USERS, Action.MANAGE, True),
            ("admin", Resource.USERS, Action.MANAGE, True),
            ("member", Resource.USERS, Action.MANAGE, False),
            ("viewer", Resource.USERS, Action.MANAGE, False),
            ("readonly", Resource.USERS, Action.MANAGE, False),
            # Billing operations
            ("superadmin", Resource.BILLING, Action.DELETE, True),
            ("admin", Resource.BILLING, Action.DELETE, False),
            ("member", Resource.BILLING, Action.DELETE, False),
            ("viewer", Resource.BILLING, Action.DELETE, False),
            ("readonly", Resource.BILLING, Action.DELETE, False),
            # Tool execution
            ("superadmin", Resource.TOOLS, Action.EXECUTE, True),
            ("admin", Resource.TOOLS, Action.EXECUTE, True),
            ("member", Resource.TOOLS, Action.EXECUTE, True),
            ("viewer", Resource.TOOLS, Action.EXECUTE, False),
            ("readonly", Resource.TOOLS, Action.EXECUTE, False),
            # Agent management
            ("superadmin", Resource.AGENTS, Action.MANAGE, True),
            ("admin", Resource.AGENTS, Action.MANAGE, True),
            ("member", Resource.AGENTS, Action.MANAGE, False),
            ("viewer", Resource.AGENTS, Action.MANAGE, False),
            ("readonly", Resource.AGENTS, Action.MANAGE, False),
        ],
    )
    def test_can_parametrized(self, role, resource, action, expected) -> None:
        assert (
            can(role, resource, action) is expected
        ), f"can({role}, {resource.value}, {action.value}) should be {expected}"
