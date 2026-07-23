"""RBAC integration test harness — fixtures, role simulators, and API mock helpers.

Provides the foundation for integration tests when RBAC enforcement middleware
is implemented (issue #596). All helpers work against the permission model
defined in cogtrix_core/auth/permissions.py (PR #658 / issue #594).

Issue: #598 — RBAC integration tests + audit (Phase 2.2.7)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cogtrix_core.auth.permissions import (
    Action,
    Permission,
    Resource,
    can,
    get_permissions,
    get_roles,
    has_permission,
)

# ---------------------------------------------------------------------------
# Test role enumeration (matches ROLE_PERMISSIONS keys)
# ---------------------------------------------------------------------------

ALL_ROLES = ["superadmin", "admin", "member", "viewer", "readonly"]
ADMIN_ROLES = ["superadmin", "admin"]
NON_ADMIN_ROLES = ["member", "viewer", "readonly"]
READ_ONLY_ROLES = ["viewer", "readonly"]

ALL_RESOURCES = list(Resource)
ALL_ACTIONS = list(Action)
ALL_PERMISSIONS = [v for k, v in vars(Permission).items() if not k.startswith("_")]


# ---------------------------------------------------------------------------
# Simulated user fixture (stand-in for future API user model)
# ---------------------------------------------------------------------------


@dataclass
class SimulatedUser:
    """Simulates a user with a role for permission testing.

    When RBAC middleware is implemented, this will be replaced by the actual
    JWT TokenData / OrgContext dependency chain.
    """

    user_id: str
    role: str
    org_id: str = "org-test-001"

    def can(self, resource: Resource, action: Action) -> bool:
        return can(self.role, resource, action)

    def has(self, permission: str) -> bool:
        return has_permission(self.role, permission)

    @property
    def is_admin(self) -> bool:
        return self.role in ADMIN_ROLES

    @property
    def permissions(self) -> set[str]:
        return get_permissions(self.role)


# ---------------------------------------------------------------------------
# Simulated API route guard (stand-in for future middleware)
# ---------------------------------------------------------------------------


class RouteGuard:
    """Simulates an RBAC-enforced route guard for integration testing.

    Usage pattern (mimics future FastAPI dependency):
        guard = RouteGuard()
        guard.require(user, Resource.SESSIONS, Action.CREATE)
    """

    class Forbidden(Exception):
        pass

    @staticmethod
    def require(user: SimulatedUser, resource: Resource, action: Action) -> None:
        """Raise Forbidden if user lacks permission."""
        if not user.can(resource, action):
            raise RouteGuard.Forbidden(
                f"User {user.user_id} (role={user.role}) lacks " f"{resource.value}.{action.value}"
            )

    @staticmethod
    def require_admin(user: SimulatedUser) -> None:
        """Raise Forbidden if user is not admin+."""
        if not user.is_admin:
            raise RouteGuard.Forbidden(f"User {user.user_id} (role={user.role}) is not admin")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def superadmin() -> SimulatedUser:
    return SimulatedUser(user_id="su-001", role="superadmin")


@pytest.fixture
def admin() -> SimulatedUser:
    return SimulatedUser(user_id="admin-001", role="admin")


@pytest.fixture
def member() -> SimulatedUser:
    return SimulatedUser(user_id="member-001", role="member")


@pytest.fixture
def viewer() -> SimulatedUser:
    return SimulatedUser(user_id="viewer-001", role="viewer")


@pytest.fixture
def readonly() -> SimulatedUser:
    return SimulatedUser(user_id="ro-001", role="readonly")


@pytest.fixture
def all_users(superadmin, admin, member, viewer, readonly) -> list[SimulatedUser]:
    return [superadmin, admin, member, viewer, readonly]


@pytest.fixture
def users_by_role() -> dict[str, SimulatedUser]:
    """Build users for every defined role, including unknown."""
    roles = get_roles()
    result = {}
    for i, role in enumerate(roles):
        result[role] = SimulatedUser(user_id=f"user-{i:03d}", role=role)
    result["bogus"] = SimulatedUser(user_id="bogus-000", role="bogus")
    return result


# ---------------------------------------------------------------------------
# Cross-org isolation fixtures
# ---------------------------------------------------------------------------


@dataclass
class SimulatedOrg:
    """Simulates an organization for cross-org isolation testing."""

    org_id: str
    name: str
    users: list[SimulatedUser] = field(default_factory=list)

    def add_user(self, role: str) -> SimulatedUser:
        user = SimulatedUser(
            user_id=f"user-{self.org_id}-{len(self.users):03d}",
            role=role,
            org_id=self.org_id,
        )
        self.users.append(user)
        return user


@pytest.fixture
def two_orgs() -> tuple[SimulatedOrg, SimulatedOrg]:
    """Two isolated organizations with users in each."""
    org_a = SimulatedOrg(org_id="org-alpha", name="Alpha Corp")
    org_b = SimulatedOrg(org_id="org-beta", name="Beta Inc")

    org_a.add_user("admin")
    org_a.add_user("member")
    org_a.add_user("viewer")

    org_b.add_user("admin")
    org_b.add_user("member")
    org_b.add_user("readonly")

    return org_a, org_b


# ---------------------------------------------------------------------------
# Permission matrix exploration helpers
# ---------------------------------------------------------------------------


def get_permission_diff(role_a: str, role_b: str) -> dict[str, set[str]]:
    """Return the symmetric difference of permissions between two roles."""
    a_perms = get_permissions(role_a)
    b_perms = get_permissions(role_b)
    return {
        f"only_{role_a}": a_perms - b_perms,
        f"only_{role_b}": b_perms - a_perms,
        "shared": a_perms & b_perms,
    }


def get_role_permission_summary() -> dict[str, dict[str, Any]]:
    """Return a structured summary of all role permissions."""
    summary = {}
    for role in get_roles():
        perms = get_permissions(role)
        resources_with_access = set()
        actions_per_resource: dict[str, set[str]] = {}
        for perm in perms:
            resource, action = perm.split(".")
            resources_with_access.add(resource)
            actions_per_resource.setdefault(resource, set()).add(action)

        summary[role] = {
            "total_permissions": len(perms),
            "resources": sorted(resources_with_access),
            "actions_per_resource": {r: sorted(a) for r, a in sorted(actions_per_resource.items())},
            "permissions": sorted(perms),
        }
    return summary


# ---------------------------------------------------------------------------
# API endpoint → permission mapping (for future middleware tests)
# ---------------------------------------------------------------------------

# Mapping of hypothetical API endpoints to required permissions.
# This will be used when enforcement middleware exists.
# Format: (method, path_pattern) → (resource, action)

ENDPOINT_PERMISSION_MAP: dict[tuple[str, str], tuple[Resource, Action]] = {
    # Session endpoints
    ("POST", "/api/v1/sessions"): (Resource.SESSIONS, Action.CREATE),
    ("GET", "/api/v1/sessions"): (Resource.SESSIONS, Action.READ),
    ("GET", "/api/v1/sessions/{session_id}"): (Resource.SESSIONS, Action.READ),
    ("DELETE", "/api/v1/sessions/{session_id}"): (Resource.SESSIONS, Action.DELETE),
    # Agent endpoints
    ("POST", "/api/v1/agents"): (Resource.AGENTS, Action.CREATE),
    ("GET", "/api/v1/agents"): (Resource.AGENTS, Action.READ),
    ("DELETE", "/api/v1/agents/{agent_id}"): (Resource.AGENTS, Action.DELETE),
    # Tool endpoints
    ("GET", "/api/v1/tools"): (Resource.TOOLS, Action.READ),
    ("POST", "/api/v1/tools"): (Resource.TOOLS, Action.CREATE),
    # Config endpoints
    ("GET", "/api/v1/config"): (Resource.CONFIG, Action.READ),
    ("PATCH", "/api/v1/config"): (Resource.CONFIG, Action.UPDATE),
    # User endpoints
    ("GET", "/api/v1/users"): (Resource.USERS, Action.READ),
    ("POST", "/api/v1/users"): (Resource.USERS, Action.CREATE),
    ("DELETE", "/api/v1/users/{user_id}"): (Resource.USERS, Action.DELETE),
    # Org endpoints
    ("GET", "/api/v1/orgs"): (Resource.ORGS, Action.READ),
    ("PATCH", "/api/v1/orgs/{org_id}"): (Resource.ORGS, Action.UPDATE),
    # Billing endpoints
    ("GET", "/api/v1/billing"): (Resource.BILLING, Action.READ),
    ("POST", "/api/v1/billing/checkout"): (Resource.BILLING, Action.CREATE),
    # Audit endpoints
    ("GET", "/api/v1/admin/audit"): (Resource.AUDIT, Action.READ),
    # Assistant endpoints
    ("POST", "/api/v1/assistant/start"): (Resource.ASSISTANT, Action.EXECUTE),
    ("GET", "/api/v1/assistant/status"): (Resource.ASSISTANT, Action.READ),
}


def get_required_permission(method: str, path: str) -> tuple[Resource, Action] | None:
    """Look up the required permission for an endpoint.

    Returns None if the endpoint is not in the mapping (e.g., open endpoints).
    """
    return ENDPOINT_PERMISSION_MAP.get((method, path))
