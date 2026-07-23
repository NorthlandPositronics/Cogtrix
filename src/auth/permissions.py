"""Permission model — resource × action matrix for RBAC (Phase 2.2).

Defines the Resource and Action enums, permission constants, a role-to-permission
mapping (``ROLE_PERMISSIONS``), and helper functions for permission checks.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Resource and Action enums
# ---------------------------------------------------------------------------


class Resource(StrEnum):
    """Resources that can be acted upon."""

    SESSIONS = "sessions"
    AGENTS = "agents"
    TOOLS = "tools"
    CONFIG = "config"
    USERS = "users"
    ORGS = "orgs"
    BILLING = "billing"
    AUDIT = "audit"
    ASSISTANT = "assistant"


class Action(StrEnum):
    """Actions that can be performed on resources."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"


# ---------------------------------------------------------------------------
# Permission helper
# ---------------------------------------------------------------------------


def _perm(resource: Resource, action: Action) -> str:
    """Build a ``resource.action`` permission string."""
    return f"{resource.value}.{action.value}"


# ---------------------------------------------------------------------------
# Permission constants
# ---------------------------------------------------------------------------


class Permission:
    """Permission constants as ``resource.action`` strings."""

    # Sessions
    SESSIONS_CREATE = _perm(Resource.SESSIONS, Action.CREATE)
    SESSIONS_READ = _perm(Resource.SESSIONS, Action.READ)
    SESSIONS_UPDATE = _perm(Resource.SESSIONS, Action.UPDATE)
    SESSIONS_DELETE = _perm(Resource.SESSIONS, Action.DELETE)
    SESSIONS_EXECUTE = _perm(Resource.SESSIONS, Action.EXECUTE)
    SESSIONS_MANAGE = _perm(Resource.SESSIONS, Action.MANAGE)

    # Agents
    AGENTS_CREATE = _perm(Resource.AGENTS, Action.CREATE)
    AGENTS_READ = _perm(Resource.AGENTS, Action.READ)
    AGENTS_UPDATE = _perm(Resource.AGENTS, Action.UPDATE)
    AGENTS_DELETE = _perm(Resource.AGENTS, Action.DELETE)
    AGENTS_EXECUTE = _perm(Resource.AGENTS, Action.EXECUTE)
    AGENTS_MANAGE = _perm(Resource.AGENTS, Action.MANAGE)

    # Tools
    TOOLS_CREATE = _perm(Resource.TOOLS, Action.CREATE)
    TOOLS_READ = _perm(Resource.TOOLS, Action.READ)
    TOOLS_UPDATE = _perm(Resource.TOOLS, Action.UPDATE)
    TOOLS_DELETE = _perm(Resource.TOOLS, Action.DELETE)
    TOOLS_EXECUTE = _perm(Resource.TOOLS, Action.EXECUTE)
    TOOLS_MANAGE = _perm(Resource.TOOLS, Action.MANAGE)

    # Config
    CONFIG_CREATE = _perm(Resource.CONFIG, Action.CREATE)
    CONFIG_READ = _perm(Resource.CONFIG, Action.READ)
    CONFIG_UPDATE = _perm(Resource.CONFIG, Action.UPDATE)
    CONFIG_DELETE = _perm(Resource.CONFIG, Action.DELETE)
    CONFIG_EXECUTE = _perm(Resource.CONFIG, Action.EXECUTE)
    CONFIG_MANAGE = _perm(Resource.CONFIG, Action.MANAGE)

    # Users
    USERS_CREATE = _perm(Resource.USERS, Action.CREATE)
    USERS_READ = _perm(Resource.USERS, Action.READ)
    USERS_UPDATE = _perm(Resource.USERS, Action.UPDATE)
    USERS_DELETE = _perm(Resource.USERS, Action.DELETE)
    USERS_EXECUTE = _perm(Resource.USERS, Action.EXECUTE)
    USERS_MANAGE = _perm(Resource.USERS, Action.MANAGE)

    # Orgs
    ORGS_CREATE = _perm(Resource.ORGS, Action.CREATE)
    ORGS_READ = _perm(Resource.ORGS, Action.READ)
    ORGS_UPDATE = _perm(Resource.ORGS, Action.UPDATE)
    ORGS_DELETE = _perm(Resource.ORGS, Action.DELETE)
    ORGS_EXECUTE = _perm(Resource.ORGS, Action.EXECUTE)
    ORGS_MANAGE = _perm(Resource.ORGS, Action.MANAGE)

    # Billing
    BILLING_CREATE = _perm(Resource.BILLING, Action.CREATE)
    BILLING_READ = _perm(Resource.BILLING, Action.READ)
    BILLING_UPDATE = _perm(Resource.BILLING, Action.UPDATE)
    BILLING_DELETE = _perm(Resource.BILLING, Action.DELETE)
    BILLING_EXECUTE = _perm(Resource.BILLING, Action.EXECUTE)
    BILLING_MANAGE = _perm(Resource.BILLING, Action.MANAGE)

    # Audit
    AUDIT_CREATE = _perm(Resource.AUDIT, Action.CREATE)
    AUDIT_READ = _perm(Resource.AUDIT, Action.READ)
    AUDIT_UPDATE = _perm(Resource.AUDIT, Action.UPDATE)
    AUDIT_DELETE = _perm(Resource.AUDIT, Action.DELETE)
    AUDIT_EXECUTE = _perm(Resource.AUDIT, Action.EXECUTE)
    AUDIT_MANAGE = _perm(Resource.AUDIT, Action.MANAGE)

    # Assistant
    ASSISTANT_CREATE = _perm(Resource.ASSISTANT, Action.CREATE)
    ASSISTANT_READ = _perm(Resource.ASSISTANT, Action.READ)
    ASSISTANT_UPDATE = _perm(Resource.ASSISTANT, Action.UPDATE)
    ASSISTANT_DELETE = _perm(Resource.ASSISTANT, Action.DELETE)
    ASSISTANT_EXECUTE = _perm(Resource.ASSISTANT, Action.EXECUTE)
    ASSISTANT_MANAGE = _perm(Resource.ASSISTANT, Action.MANAGE)


# ---------------------------------------------------------------------------
# Aggregate permission sets
# ---------------------------------------------------------------------------


_ALL_PERMISSIONS: set[str] = {
    Permission.SESSIONS_CREATE,
    Permission.SESSIONS_READ,
    Permission.SESSIONS_UPDATE,
    Permission.SESSIONS_DELETE,
    Permission.SESSIONS_EXECUTE,
    Permission.SESSIONS_MANAGE,
    Permission.AGENTS_CREATE,
    Permission.AGENTS_READ,
    Permission.AGENTS_UPDATE,
    Permission.AGENTS_DELETE,
    Permission.AGENTS_EXECUTE,
    Permission.AGENTS_MANAGE,
    Permission.TOOLS_CREATE,
    Permission.TOOLS_READ,
    Permission.TOOLS_UPDATE,
    Permission.TOOLS_DELETE,
    Permission.TOOLS_EXECUTE,
    Permission.TOOLS_MANAGE,
    Permission.CONFIG_CREATE,
    Permission.CONFIG_READ,
    Permission.CONFIG_UPDATE,
    Permission.CONFIG_DELETE,
    Permission.CONFIG_EXECUTE,
    Permission.CONFIG_MANAGE,
    Permission.USERS_CREATE,
    Permission.USERS_READ,
    Permission.USERS_UPDATE,
    Permission.USERS_DELETE,
    Permission.USERS_EXECUTE,
    Permission.USERS_MANAGE,
    Permission.ORGS_CREATE,
    Permission.ORGS_READ,
    Permission.ORGS_UPDATE,
    Permission.ORGS_DELETE,
    Permission.ORGS_EXECUTE,
    Permission.ORGS_MANAGE,
    Permission.BILLING_CREATE,
    Permission.BILLING_READ,
    Permission.BILLING_UPDATE,
    Permission.BILLING_DELETE,
    Permission.BILLING_EXECUTE,
    Permission.BILLING_MANAGE,
    Permission.AUDIT_CREATE,
    Permission.AUDIT_READ,
    Permission.AUDIT_UPDATE,
    Permission.AUDIT_DELETE,
    Permission.AUDIT_EXECUTE,
    Permission.AUDIT_MANAGE,
    Permission.ASSISTANT_CREATE,
    Permission.ASSISTANT_READ,
    Permission.ASSISTANT_UPDATE,
    Permission.ASSISTANT_DELETE,
    Permission.ASSISTANT_EXECUTE,
    Permission.ASSISTANT_MANAGE,
}

_READ_ALL: set[str] = {
    Permission.SESSIONS_READ,
    Permission.AGENTS_READ,
    Permission.TOOLS_READ,
    Permission.CONFIG_READ,
    Permission.USERS_READ,
    Permission.ORGS_READ,
    Permission.BILLING_READ,
    Permission.AUDIT_READ,
    Permission.ASSISTANT_READ,
}

_MEMBER_READ: set[str] = {
    Permission.SESSIONS_READ,
    Permission.AGENTS_READ,
    Permission.TOOLS_READ,
    Permission.CONFIG_READ,
    Permission.USERS_READ,
    Permission.ORGS_READ,
    Permission.ASSISTANT_READ,
}

_READ_RESTRICTED: set[str] = _MEMBER_READ

_MEMBER_OWNED_CRUD: set[str] = {
    Permission.SESSIONS_CREATE,
    Permission.SESSIONS_UPDATE,
    Permission.SESSIONS_DELETE,
    Permission.AGENTS_CREATE,
    Permission.AGENTS_UPDATE,
    Permission.AGENTS_DELETE,
    Permission.ASSISTANT_CREATE,
    Permission.ASSISTANT_UPDATE,
    Permission.ASSISTANT_DELETE,
}

_MEMBER_EXECUTE: set[str] = {
    Permission.SESSIONS_EXECUTE,
    Permission.AGENTS_EXECUTE,
    Permission.TOOLS_EXECUTE,
    Permission.ASSISTANT_EXECUTE,
}


# ---------------------------------------------------------------------------
# ROLE_PERMISSIONS — the authoritative role-to-permission matrix
# ---------------------------------------------------------------------------


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "superadmin": _ALL_PERMISSIONS,
    "admin": _ALL_PERMISSIONS - {Permission.BILLING_DELETE},
    "member": _MEMBER_READ | _MEMBER_OWNED_CRUD | _MEMBER_EXECUTE,
    "viewer": _READ_ALL,
    "readonly": _READ_RESTRICTED,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def has_permission(role: str, permission: str) -> bool:
    """Return ``True`` when *role* holds *permission*.

    Unknown roles always return ``False``.
    """
    perms = ROLE_PERMISSIONS.get(role)
    if perms is None:
        return False
    return permission in perms


def can(role: str, resource: Resource, action: Action) -> bool:
    """Return ``True`` when *role* can perform *action* on *resource*."""
    return has_permission(role, _perm(resource, action))


def get_permissions(role: str) -> set[str]:
    """Return a copy of the permissions granted to *role*.

    Returns an empty set for unknown roles.
    """
    return ROLE_PERMISSIONS.get(role, set()).copy()


def get_roles() -> list[str]:
    """Return the list of defined role names."""
    return list(ROLE_PERMISSIONS.keys())
