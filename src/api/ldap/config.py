"""LDAP/AD directory sync configuration (Enterprise Phase 1 — task 1.2.3).

Requires the ``[ldap]`` optional extra::

    pip install cogtrix[ldap]

No system packages required — ``ldap3`` is a pure-Python implementation.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

log = logging.getLogger("cogtrix.api.ldap")

_lock = threading.Lock()
_ldap_config: LDAPConfig | None = None

# Valid objectClass values for user and group filters.
_ALLOWED_USER_OBJECT_CLASSES = {"person", "user", "inetOrgPerson"}
_ALLOWED_GROUP_OBJECT_CLASSES = {"group", "groupOfNames"}

# Simple LDAP filter validator — rejects OR (|), wildcard objectClass,
# and anything that does not start with a recognized (objectClass=…) pattern.
_ldap_filter_re = re.compile(
    r"^\((?:objectClass=(person|user|inetOrgPerson|group|groupOfNames)"
    r"|&\(objectClass=(person|user|inetOrgPerson|group|groupOfNames)\).+)\)$"
)


def _validate_ldap_filter(filter_str: str, allowed_classes: set[str]) -> None:
    """Raise ValueError if *filter_str* is not a safe, allow-listed LDAP filter.

    Rules:
    - Must start with ``(`` and end with ``)``.
    - Must not contain ``|`` (OR operator).
    - Must not use ``*`` as an objectClass value.
    - Must match ``(objectClass=<allowed>)`` or
      ``(&(objectClass=<allowed>)(...))``.
    """
    if not filter_str or filter_str[0] != "(" or filter_str[-1] != ")":
        raise ValueError(f"LDAP filter must be a parenthesized expression: {filter_str!r}")
    if "|" in filter_str:
        raise ValueError(f"LDAP filter contains disallowed OR operator '|': {filter_str!r}")
    if "(objectClass=*)" in filter_str or "(objectClass=* )" in filter_str:
        raise ValueError(f"LDAP filter uses wildcard objectClass: {filter_str!r}")

    # Extract objectClass value from simple or AND-wrapped filter.
    oc_match = re.search(r"objectClass=([^)]+)", filter_str)
    if not oc_match:
        raise ValueError(f"LDAP filter must target objectClass: {filter_str!r}")
    oc_value = oc_match.group(1).strip()
    if oc_value not in allowed_classes:
        raise ValueError(
            f"LDAP filter objectClass '{oc_value}' not in allowlist "
            f"{sorted(allowed_classes)}: {filter_str!r}"
        )

    # Ensure the overall shape is valid.
    if not _ldap_filter_re.match(filter_str):
        raise ValueError(f"LDAP filter has invalid structure: {filter_str!r}")


@dataclass
class LDAPConfig:
    """Configuration for LDAP/Active Directory user synchronisation.

    Attributes:
        server_url:     LDAP server URL, e.g. ``ldaps://ldap.example.com:636``
                        or ``ldap://192.168.1.10:389``.
        bind_dn:        Bind distinguished name for the service account,
                        e.g. ``cn=svc-cogtrix,ou=service,dc=example,dc=com``.
        bind_password:  Password for the bind account.
        search_base:    Base DN for user searches,
                        e.g. ``ou=users,dc=example,dc=com``.
        search_filter:  LDAP filter for selecting sync targets.
                        Default: ``(objectClass=person)``.
        attribute_map:  Maps LDAP attribute names → Cogtrix field names.
                        Required keys: ``username``, ``email``.
                        Optional: ``display_name``.
        use_ssl:        Enforce TLS/SSL (recommended for production).
        ldap_tls_skip_verify: Disable certificate verification for local/dev
                              LDAP servers only.
        org_id:         Cogtrix org that synced users are assigned to.
        default_role:   Role assigned to provisioned users.
        page_size:      Number of results per LDAP paged-results request.
    """

    server_url: str
    bind_dn: str
    bind_password: str
    search_base: str
    search_filter: str = field(default="(objectClass=person)")
    attribute_map: dict[str, str] = field(
        default_factory=lambda: {
            "username": "sAMAccountName",
            "email": "mail",
            "display_name": "displayName",
        }
    )
    use_ssl: bool = field(default=True)
    ldap_tls_skip_verify: bool = field(default=False)
    org_id: str | None = field(default=None)
    default_role: str = field(default="user")
    page_size: int = field(default=200)
    group_role_map: dict[str, str] = field(default_factory=dict)
    group_role_default: str | None = field(default=None)


def configure_ldap(config: LDAPConfig) -> None:
    """Register the LDAP configuration. Thread-safe; replaces any prior config.

    Validates *search_filter* before accepting the config.
    """
    _validate_ldap_filter(config.search_filter, _ALLOWED_USER_OBJECT_CLASSES)
    global _ldap_config
    with _lock:
        _ldap_config = config
    log.info("LDAP configured: server=%s base=%s", config.server_url, config.search_base)


def get_ldap_config() -> LDAPConfig | None:
    """Return the active LDAP configuration, or ``None`` if not configured."""
    with _lock:
        return _ldap_config


def is_ldap_configured() -> bool:
    """Return True when an LDAP configuration has been registered."""
    with _lock:
        return _ldap_config is not None
