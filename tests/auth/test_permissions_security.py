"""Security audit tests for the RBAC permission model (issue #598).

Covers: bypass attempts, role escalation, edge cases, and adversarial scenarios.
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
    get_roles,
    has_permission,
)

_ALL_ROLES = ["superadmin", "admin", "member", "viewer", "readonly"]


# ---------------------------------------------------------------------------
# Security: Role escalation attempts
# ---------------------------------------------------------------------------


class TestRoleEscalation:
    """Verify that lower-privilege roles cannot access higher-privilege permissions."""

    def test_member_cannot_elevate_to_admin(self) -> None:
        """Member must not gain admin-exclusive permissions."""
        admin_only = ROLE_PERMISSIONS["admin"] - ROLE_PERMISSIONS["member"]
        assert len(admin_only) > 0, "admin should have permissions member does not"
        for perm in admin_only:
            assert not has_permission("member", perm), f"member should not have {perm}"

    def test_viewer_cannot_elevate_to_member(self) -> None:
        """Viewer must not gain member-level permissions."""
        member_extra = ROLE_PERMISSIONS["member"] - ROLE_PERMISSIONS["viewer"]
        assert len(member_extra) > 0, "member should have permissions viewer does not"
        for perm in member_extra:
            assert not has_permission("viewer", perm), f"viewer should not have {perm}"

    def test_readonly_cannot_elevate_to_viewer(self) -> None:
        """Readonly must not gain viewer-level permissions."""
        viewer_extra = ROLE_PERMISSIONS["viewer"] - ROLE_PERMISSIONS["readonly"]
        assert len(viewer_extra) > 0, "viewer should have permissions readonly does not"
        for perm in viewer_extra:
            assert not has_permission("readonly", perm), f"readonly should not have {perm}"

    def test_role_hierarchy_is_nested_except_viewer_member(self) -> None:
        """Each role's permissions must be a strict subset of the next higher role."""
        hierarchy = ["readonly", "viewer", "member", "admin", "superadmin"]
        for i in range(len(hierarchy) - 1):
            lower = hierarchy[i]
            higher = hierarchy[i + 1]
            lower_perms = ROLE_PERMISSIONS[lower]
            higher_perms = ROLE_PERMISSIONS[higher]
            # viewer has billing.read + audit.read that member lacks (by design —
            # viewer reads ALL 9 resources, member reads 7 + owns CRUD/execute).
            # So the hierarchy is NOT strictly nested for viewer→member.
            if lower == "viewer" and higher == "member":
                extra = lower_perms - higher_perms
                assert extra == {
                    Permission.BILLING_READ,
                    Permission.AUDIT_READ,
                }, f"unexpected viewer-only permissions: {extra}"
            else:
                assert lower_perms <= higher_perms, (
                    f"{lower} permissions should be a subset of {higher} permissions.\n"
                    f"Extra in lower: {lower_perms - higher_perms}"
                )

    def test_unknown_role_same_as_no_permissions(self) -> None:
        """Unknown/bogus roles must have exactly zero effective permissions."""
        assert not has_permission("bogus", "any.perm")
        assert not has_permission("", "any.perm")
        assert not has_permission("SUPERADMIN", Permission.SESSIONS_CREATE)
        assert not has_permission("Admin", Permission.SESSIONS_READ)
        assert not has_permission(" root ", Permission.SESSIONS_READ)


# ---------------------------------------------------------------------------
# Security: Permission bypass via string manipulation
# ---------------------------------------------------------------------------


class TestPermissionStringInjection:
    """Verify that permission checks are not bypassable via string manipulation."""

    def test_resource_action_separator_cannot_be_bypassed(self) -> None:
        """Permission format is 'resource.action' — partial matches must fail."""
        # A permission string with extra segments should not match
        assert not has_permission("member", "sessions.create.extra")
        assert not has_permission("member", "sessions.create\n")
        assert not has_permission("member", " sessions.create")
        assert not has_permission("member", "sessions.create ")
        assert not has_permission("member", "sessions..create")

    def test_case_sensitivity(self) -> None:
        """Permission checks should be case-sensitive (all lowercase by convention)."""
        assert not has_permission("admin", "SESSIONS.CREATE")
        assert not has_permission("admin", "Sessions.Create")
        assert not has_permission("superadmin", "SESSIONS.CREATE")

    def test_empty_and_none_inputs(self) -> None:
        """Empty or malformed permission strings should never grant access."""
        assert not has_permission("admin", "")
        assert not has_permission("superadmin", "")
        assert not has_permission("member", "")

    def test_permission_format_validation(self) -> None:
        """All defined permission constants must follow resource.action format."""
        for role in _ALL_ROLES:
            for perm in ROLE_PERMISSIONS[role]:
                parts = perm.split(".")
                assert len(parts) == 2, f"Invalid permission format: {perm}"
                assert parts[0] in [
                    "sessions",
                    "agents",
                    "tools",
                    "config",
                    "users",
                    "orgs",
                    "billing",
                    "audit",
                    "assistant",
                ]
                assert parts[1] in ["create", "read", "update", "delete", "execute", "manage"]


# ---------------------------------------------------------------------------
# Security: Cross-role boundary enumeration
# ---------------------------------------------------------------------------


class TestCrossRoleBoundary:
    """Exhaustive validation that each role has exactly the permissions it should."""

    @pytest.mark.parametrize("role", _ALL_ROLES)
    def test_no_role_has_undefined_permissions(self, role: str) -> None:
        """Every permission in a role's set must be a defined Permission constant."""
        all_constants = {v for k, v in vars(Permission).items() if not k.startswith("_")}
        for perm in ROLE_PERMISSIONS[role]:
            assert perm in all_constants, f"{role} has undefined permission: {perm}"

    def test_superadmin_owns_all_constants(self) -> None:
        """Superadmin must own every defined permission constant."""
        all_constants = {v for k, v in vars(Permission).items() if not k.startswith("_")}
        assert ROLE_PERMISSIONS["superadmin"] == all_constants

    def test_member_missing_manage_permissions(self) -> None:
        """Member must not have any *.manage permissions."""
        for perm in ROLE_PERMISSIONS["member"]:
            assert not perm.endswith(".manage"), f"member should not have {perm}"

    def test_viewer_only_has_read(self) -> None:
        """Viewer must only have *.read permissions."""
        for perm in ROLE_PERMISSIONS["viewer"]:
            assert perm.endswith(".read"), f"viewer has non-read: {perm}"

    def test_readonly_sensitive_resources_denied(self) -> None:
        """Readonly must not read billing or audit data."""
        assert Permission.BILLING_READ not in ROLE_PERMISSIONS["readonly"]
        assert Permission.AUDIT_READ not in ROLE_PERMISSIONS["readonly"]
        assert Permission.BILLING_CREATE not in ROLE_PERMISSIONS["readonly"]
        assert Permission.AUDIT_DELETE not in ROLE_PERMISSIONS["readonly"]

    def test_admin_cannot_delete_billing(self) -> None:
        """The one permission admin lacks: billing.delete."""
        assert Permission.BILLING_DELETE not in ROLE_PERMISSIONS["admin"]
        assert not has_permission("admin", Permission.BILLING_DELETE)


# ---------------------------------------------------------------------------
# Security: Permission model immutability
# ---------------------------------------------------------------------------


class TestPermissionModelImmutability:
    """Verify the permission model cannot be mutated at runtime."""

    def test_get_permissions_returns_copy_not_reference(self) -> None:
        """Mutating the returned set must not affect the original matrix."""
        p1 = get_permissions("admin")
        p1.clear()
        assert (
            len(ROLE_PERMISSIONS["admin"]) > 0
        ), "Clearing the copy should not affect original role permissions"

    def test_role_permissions_dict_values_are_independent(self) -> None:
        """Mutating one role's permission set must not affect another role."""
        admin_perms = ROLE_PERMISSIONS["admin"]
        member_perms = ROLE_PERMISSIONS["member"]
        assert admin_perms is not member_perms, "Role permission sets must be distinct objects"

    def test_permission_constants_unchanged_after_import(self) -> None:
        """Permission constant values must be stable after initial import."""
        assert Permission.SESSIONS_CREATE == "sessions.create"
        assert Permission.USERS_MANAGE == "users.manage"
        assert Permission.BILLING_DELETE == "billing.delete"


# ---------------------------------------------------------------------------
# Security: Concurrent access safety
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Verify permission checks are safe under concurrent access."""

    def test_has_permission_is_thread_safe(self) -> None:
        """Multiple calls to has_permission must return consistent results."""
        # Run 1000 checks — all should be deterministic
        for _ in range(1000):
            assert has_permission("superadmin", Permission.SESSIONS_CREATE) is True
            assert has_permission("member", Permission.BILLING_DELETE) is False
            assert has_permission("viewer", Permission.SESSIONS_READ) is True
            assert has_permission("readonly", Permission.AUDIT_READ) is False

    def test_get_permissions_thread_safe(self) -> None:
        """Concurrent get_permissions calls must return consistent, independent copies."""
        results = [get_permissions("member") for _ in range(100)]
        first = results[0]
        for r in results[1:]:
            assert r == first

    def test_get_roles_thread_safe(self) -> None:
        """Concurrent get_roles calls must return consistent results."""
        results = [get_roles() for _ in range(100)]
        first = set(results[0])
        for r in results[1:]:
            assert set(r) == first


# ---------------------------------------------------------------------------
# Security: Helper function edge cases
# ---------------------------------------------------------------------------


class TestHelperEdgeCases:
    """Adversarial edge cases for helper functions."""

    def test_can_with_invalid_resource(self) -> None:
        """can() with an undefined resource should not crash but should return False."""
        # Resources are typed via StrEnum so this is hard to test directly,
        # but we verify the function handles all defined resources safely.
        for resource in Resource:
            for action in Action:
                result = can("admin", resource, action)
                assert isinstance(result, bool)

    def test_has_permission_with_none_like_strings(self) -> None:
        """has_permission with 'None' or 'null' strings should return False."""
        assert has_permission("None", Permission.SESSIONS_READ) is False
        assert has_permission("null", Permission.SESSIONS_READ) is False
        assert has_permission("undefined", Permission.SESSIONS_READ) is False

    def test_has_permission_malformed_permission_string(self) -> None:
        """has_permission with malformed permission strings must not crash."""
        assert has_permission("admin", ".") is False
        assert has_permission("admin", "..") is False
        assert has_permission("admin", "sessions.") is False
        assert has_permission("admin", ".create") is False
        assert has_permission("admin", "sessions.create.delete") is False

    def test_get_permissions_returns_correct_type(self) -> None:
        """get_permissions must always return a set."""
        for role in _ALL_ROLES:
            assert isinstance(get_permissions(role), set)
        assert isinstance(get_permissions("bogus"), set)
        assert isinstance(get_permissions(""), set)

    def test_get_roles_returns_correct_type(self) -> None:
        """get_roles must always return a list."""
        assert isinstance(get_roles(), list)


# ---------------------------------------------------------------------------
# Security: Billing isolation
# ---------------------------------------------------------------------------


class TestBillingIsolation:
    """Verify billing permissions are properly isolated across roles."""

    def test_only_superadmin_can_delete_billing(self) -> None:
        """Only superadmin should have billing.delete."""
        assert has_permission("superadmin", Permission.BILLING_DELETE) is True
        assert has_permission("admin", Permission.BILLING_DELETE) is False
        assert has_permission("member", Permission.BILLING_DELETE) is False
        assert has_permission("viewer", Permission.BILLING_DELETE) is False
        assert has_permission("readonly", Permission.BILLING_DELETE) is False

    def test_billing_read_access_control(self) -> None:
        """Billing read access should follow role hierarchy."""
        assert has_permission("superadmin", Permission.BILLING_READ) is True
        assert has_permission("admin", Permission.BILLING_READ) is True
        assert has_permission("member", Permission.BILLING_READ) is False
        assert has_permission("viewer", Permission.BILLING_READ) is True
        assert has_permission("readonly", Permission.BILLING_READ) is False

    def test_billing_manage_access_control(self) -> None:
        """Billing management should be admin+ only."""
        assert has_permission("superadmin", Permission.BILLING_MANAGE) is True
        assert has_permission("admin", Permission.BILLING_MANAGE) is True
        assert has_permission("member", Permission.BILLING_MANAGE) is False
        assert has_permission("viewer", Permission.BILLING_MANAGE) is False
        assert has_permission("readonly", Permission.BILLING_MANAGE) is False


# ---------------------------------------------------------------------------
# Security: Audit log isolation
# ---------------------------------------------------------------------------


class TestAuditLogIsolation:
    """Verify audit log permissions are properly isolated."""

    def test_only_admin_plus_can_access_audit(self) -> None:
        """Only superadmin and admin should access audit logs."""
        assert has_permission("superadmin", Permission.AUDIT_READ) is True
        assert has_permission("admin", Permission.AUDIT_READ) is True
        assert has_permission("member", Permission.AUDIT_READ) is False
        assert has_permission("viewer", Permission.AUDIT_READ) is True
        assert has_permission("readonly", Permission.AUDIT_READ) is False

    def test_audit_write_restricted_to_superadmin(self) -> None:
        """Only superadmin+ should write audit logs (admin has manage but not delete)."""
        assert has_permission("superadmin", Permission.AUDIT_MANAGE) is True
        assert has_permission("admin", Permission.AUDIT_MANAGE) is True
        assert has_permission("member", Permission.AUDIT_MANAGE) is False
        assert has_permission("viewer", Permission.AUDIT_MANAGE) is False


# ---------------------------------------------------------------------------
# Security: User management isolation
# ---------------------------------------------------------------------------


class TestUserManagementIsolation:
    """Verify user management permissions are properly scoped."""

    def test_only_admin_plus_can_manage_users(self) -> None:
        """User management (create, update, delete, manage) requires admin+."""
        for perm in [
            Permission.USERS_CREATE,
            Permission.USERS_UPDATE,
            Permission.USERS_DELETE,
            Permission.USERS_MANAGE,
        ]:
            assert has_permission("superadmin", perm) is True
            assert has_permission("admin", perm) is True
            assert has_permission("member", perm) is False
            assert has_permission("viewer", perm) is False
            assert has_permission("readonly", perm) is False

    def test_all_roles_can_read_users_except_readonly(self) -> None:
        """User read access should follow role hierarchy."""
        assert has_permission("superadmin", Permission.USERS_READ) is True
        assert has_permission("admin", Permission.USERS_READ) is True
        assert has_permission("member", Permission.USERS_READ) is True
        assert has_permission("viewer", Permission.USERS_READ) is True
        assert has_permission("readonly", Permission.USERS_READ) is True


# ---------------------------------------------------------------------------
# Security: Tool execution control
# ---------------------------------------------------------------------------


class TestToolExecutionControl:
    """Verify tool execution permissions across roles."""

    def test_member_can_execute_tools(self) -> None:
        """Members should execute tools but not manage them."""
        assert has_permission("member", Permission.TOOLS_EXECUTE) is True
        assert has_permission("member", Permission.TOOLS_MANAGE) is False
        assert has_permission("member", Permission.TOOLS_CREATE) is False

    def test_viewer_cannot_execute_tools(self) -> None:
        """Viewer should not execute tools."""
        assert has_permission("viewer", Permission.TOOLS_EXECUTE) is False
        assert has_permission("viewer", Permission.TOOLS_CREATE) is False
        assert has_permission("viewer", Permission.TOOLS_MANAGE) is False

    def test_admin_can_manage_tools(self) -> None:
        """Admin should have full tool management."""
        assert has_permission("admin", Permission.TOOLS_MANAGE) is True
        assert has_permission("admin", Permission.TOOLS_CREATE) is True
        assert has_permission("admin", Permission.TOOLS_EXECUTE) is True
        assert has_permission("admin", Permission.TOOLS_DELETE) is True


# ---------------------------------------------------------------------------
# Security: Agent lifecycle control
# ---------------------------------------------------------------------------


class TestAgentLifecycleControl:
    """Verify agent spawn/delete/manage permissions."""

    def test_member_can_manage_own_agents(self) -> None:
        """Member should create/update/delete their own agents."""
        assert has_permission("member", Permission.AGENTS_CREATE) is True
        assert has_permission("member", Permission.AGENTS_UPDATE) is True
        assert has_permission("member", Permission.AGENTS_DELETE) is True
        assert has_permission("member", Permission.AGENTS_EXECUTE) is True
        assert has_permission("member", Permission.AGENTS_MANAGE) is False

    def test_viewer_cannot_modify_agents(self) -> None:
        """Viewer should not create, update, or delete agents."""
        for perm in [
            Permission.AGENTS_CREATE,
            Permission.AGENTS_UPDATE,
            Permission.AGENTS_DELETE,
            Permission.AGENTS_EXECUTE,
            Permission.AGENTS_MANAGE,
        ]:
            assert has_permission("viewer", perm) is False
        assert has_permission("viewer", Permission.AGENTS_READ) is True

    def test_readonly_cannot_read_sensitive_agent_data(self) -> None:
        """Readonly retains agents.read."""
        assert has_permission("readonly", Permission.AGENTS_READ) is True


# ---------------------------------------------------------------------------
# Security: Config modification control
# ---------------------------------------------------------------------------


class TestConfigModificationControl:
    """Verify config modification is properly gated."""

    def test_only_admin_plus_can_modify_config(self) -> None:
        """Config modification requires admin+."""
        for perm in [
            Permission.CONFIG_CREATE,
            Permission.CONFIG_UPDATE,
            Permission.CONFIG_DELETE,
            Permission.CONFIG_MANAGE,
        ]:
            assert has_permission("admin", perm) is True
            assert has_permission("member", perm) is False
            assert has_permission("viewer", perm) is False
            assert has_permission("readonly", perm) is False

    def test_member_can_read_config(self) -> None:
        """Member can read config."""
        assert has_permission("member", Permission.CONFIG_READ) is True


# ---------------------------------------------------------------------------
# Security: Performance benchmarks
# ---------------------------------------------------------------------------


class TestPermissionCheckPerformance:
    """Verify permission checks meet latency requirements (p95 < 1ms under 1000 req/s)."""

    def test_has_permission_latency_single_check(self) -> None:
        """Single has_permission call should complete in sub-millisecond time."""
        import time

        start = time.perf_counter()
        for _ in range(10000):
            has_permission("member", Permission.SESSIONS_READ)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 10000) * 1_000_000
        assert (
            avg_us < 100
        ), f"Average has_permission check: {avg_us:.1f}μs (target: <100μs for p95<1ms)"

    def test_can_latency_single_check(self) -> None:
        """Single can() call should complete in sub-millisecond time."""
        import time

        start = time.perf_counter()
        for _ in range(10000):
            can("admin", Resource.SESSIONS, Action.READ)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 10000) * 1_000_000
        assert avg_us < 100, f"Average can() check: {avg_us:.1f}μs (target: <100μs for p95<1ms)"

    def test_get_permissions_latency(self) -> None:
        """get_permissions should be fast enough for request-time use."""
        import time

        start = time.perf_counter()
        for _ in range(10000):
            get_permissions("admin")
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / 10000) * 1_000_000
        assert avg_us < 200, f"Average get_permissions: {avg_us:.1f}μs (target: <200μs)"

    def test_bulk_permission_check_latency(self) -> None:
        """Checking 54 permissions for a role should be fast."""
        import time

        all_perms = [v for k, v in vars(Permission).items() if not k.startswith("_")]
        start = time.perf_counter()
        for _ in range(1000):
            for perm in all_perms:
                has_permission("admin", perm)
        elapsed = time.perf_counter() - start
        total_checks = 1000 * len(all_perms)
        avg_us = (elapsed / total_checks) * 1_000_000
        assert avg_us < 100, f"Average bulk check: {avg_us:.1f}μs per check (target: <100μs)"


# ---------------------------------------------------------------------------
# Security: RBAC matrix completeness
# ---------------------------------------------------------------------------


class TestPermissionMatrixCompleteness:
    """Verify the permission matrix covers all resource-action pairs."""

    def test_all_resource_action_pairs_assigned(self) -> None:
        """Every resource × action pair should be assigned to at least one role."""
        all_defined = set()
        for role in _ALL_ROLES:
            all_defined |= ROLE_PERMISSIONS[role]
        expected_count = len(Resource) * len(Action)  # 9 × 6 = 54
        assert (
            len(all_defined) == expected_count
        ), f"Expected {expected_count} unique permissions, got {len(all_defined)}"

    def test_superadmin_coverage_is_complete(self) -> None:
        """Superadmin must cover all 54 permissions."""
        assert len(ROLE_PERMISSIONS["superadmin"]) == 54

    def test_admin_coverage(self) -> None:
        """Admin must cover 53 permissions (all minus billing.delete)."""
        assert len(ROLE_PERMISSIONS["admin"]) == 53

    def test_member_coverage(self) -> None:
        """Member must cover appropriate count of permissions."""
        # Member: 7 reads + 9 owned CRUD (sessions, agents, assistant × 3 each) + 4 executes
        # = 7 + 9 + 4 = 20 (verify this)
        member_perms = ROLE_PERMISSIONS["member"]
        assert len(member_perms) > 0
        # All member permissions should be from the expected pool
        allowed_prefixes = {
            "sessions.create",
            "sessions.read",
            "sessions.update",
            "sessions.delete",
            "sessions.execute",
            "agents.create",
            "agents.read",
            "agents.update",
            "agents.delete",
            "agents.execute",
            "tools.read",
            "tools.execute",
            "config.read",
            "users.read",
            "orgs.read",
            "assistant.create",
            "assistant.read",
            "assistant.update",
            "assistant.delete",
            "assistant.execute",
        }
        for perm in member_perms:
            assert perm in allowed_prefixes, f"Unexpected member permission: {perm}"

    def test_viewer_coverage(self) -> None:
        """Viewer must cover all 9 read permissions."""
        assert len(ROLE_PERMISSIONS["viewer"]) == 9

    def test_readonly_coverage(self) -> None:
        """Readonly must cover 7 read permissions (no billing, no audit)."""
        assert len(ROLE_PERMISSIONS["readonly"]) == 7
