"""LDAP/AD user sync logic (Enterprise Phase 1 — task 1.2.3).

Connects to the configured LDAP server, searches for user entries, and
provisions them in Cogtrix via the UserRepository.

All heavy I/O runs in a thread-pool via ``asyncio.to_thread`` so the event
loop is never blocked.
"""

from __future__ import annotations

import logging
import ssl
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import certifi
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.ldap.config import (
    _ALLOWED_GROUP_OBJECT_CLASSES,
    _ALLOWED_USER_OBJECT_CLASSES,
    LDAPConfig,
    _validate_ldap_filter,
)

if TYPE_CHECKING:
    from ldap3 import Connection  # type: ignore[import]

log = logging.getLogger("cogtrix.api.ldap")


@dataclass
class LDAPSyncResult:
    """Summary of a completed LDAP sync run."""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.added + self.updated + self.skipped

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def _require_ldap3() -> tuple:
    """Import ldap3 classes, raising ImportError with actionable message if absent."""
    try:
        from ldap3 import (  # type: ignore[import]
            ALL_ATTRIBUTES,
            SUBTREE,  # type: ignore[import]
            Connection,
            Server,
            Tls,
        )

        return Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE
    except ImportError as exc:
        raise ImportError(
            "The [ldap] optional extra is required for LDAP/AD sync. "
            "Install with: pip install cogtrix[ldap]"
        ) from exc


def _build_tls(config: LDAPConfig):
    """Return the ldap3 ``Tls`` object and related classes for the config."""
    Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE = _require_ldap3()
    if not config.use_ssl:
        return Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE, None

    validate = ssl.CERT_NONE if config.ldap_tls_skip_verify else ssl.CERT_REQUIRED
    ca_certs_file = None if config.ldap_tls_skip_verify else certifi.where()
    tls = Tls(
        validate=validate,
        ca_certs_file=ca_certs_file,
        version=ssl.PROTOCOL_TLS_CLIENT,
    )
    return Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE, tls


def _fetch_ldap_users(
    config: LDAPConfig, *, conn: Connection | None = None
) -> list[dict[str, str]]:
    """Connect to LDAP and return a list of user attribute dicts.

    This is a synchronous function — call it via ``asyncio.to_thread``.

    Args:
        config: Active LDAP configuration.
        conn:    Optional existing ``ldap3.Connection``.  When provided the
                 caller is responsible for lifecycle (pool borrow, manual
                 creation, etc.).  When omitted a one-shot connection is
                 created and closed automatically.

    Returns:
        List of dicts with keys from ``config.attribute_map`` values.
        Each dict is guaranteed to have at least ``username`` and ``email``.

    Raises:
        ImportError: when ldap3 is not installed.
        RuntimeError: when the LDAP bind or search fails.
    """
    _validate_ldap_filter(config.search_filter, _ALLOWED_USER_OBJECT_CLASSES)

    Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE = _require_ldap3()

    attr_map = config.attribute_map
    ldap_attrs = list(attr_map.values())

    one_shot = conn is None
    if one_shot:
        Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE, tls = _build_tls(config)
        server = Server(config.server_url, use_ssl=config.use_ssl, tls=tls, get_info=None)
        conn = Connection(
            server, user=config.bind_dn, password=config.bind_password, auto_bind=True
        )
        if not conn.bind():
            raise RuntimeError(f"LDAP bind failed: {conn.result}")
    else:
        # Re-bind in case the pooled connection timed out on the server side.
        if conn.closed:
            conn.open()
            conn.bind()

    results: list[dict[str, str]] = []
    page_cookie: bytes | None = None

    while True:
        conn.search(
            search_base=config.search_base,
            search_filter=config.search_filter,
            search_scope=SUBTREE,
            attributes=ldap_attrs,
            paged_size=config.page_size,
            paged_cookie=page_cookie,
        )

        for entry in conn.entries:
            user_data: dict[str, str] = {}
            for cogtrix_field, ldap_attr in attr_map.items():
                raw = entry[ldap_attr].value if ldap_attr in entry else None
                user_data[cogtrix_field] = str(raw) if raw else ""
            user_data["dn"] = str(entry.entry_dn) if hasattr(entry, "entry_dn") else ""
            if user_data.get("username") and user_data.get("email"):
                results.append(user_data)

        # Follow paged results cookie.
        cookie = (
            conn.result.get("controls", {})
            .get("1.2.840.113556.1.4.319", {})
            .get("value", {})
            .get("cookie")
        )
        if not cookie:
            break
        page_cookie = cookie

    if one_shot:
        try:
            conn.unbind()
        except Exception:
            pass
    log.info("LDAP: fetched %d users from %s", len(results), config.server_url)
    return results


def _resolve_role_from_groups(
    config: LDAPConfig, group_entries: list[dict[str, str]]
) -> str | None:
    """Map LDAP group DNs to a Cogtrix role using ``config.group_role_map``.

    Walks the map in insertion order — first match wins.  Falls back to
    ``config.group_role_default`` when no group matches.  Returns ``None``
    when the map is empty so callers can skip the update.
    """
    if not config.group_role_map:
        return None
    group_dns = {g["dn"] for g in group_entries}
    for group_dn, role in config.group_role_map.items():
        if group_dn in group_dns:
            return role
    return config.group_role_default


async def sync_users(config: LDAPConfig, db: AsyncSession) -> LDAPSyncResult:
    """Synchronise LDAP users into Cogtrix.

    Fetches user entries from the LDAP server, then for each entry:
    - Creates a new Cogtrix user if one with that username does not exist.
    - Updates the email if the user exists but the email differs.
    - Assigns all provisioned users to ``config.org_id`` (or default org).

    Args:
        config: Active LDAP configuration.
        db:     Async SQLAlchemy session.

    Returns:
        ``LDAPSyncResult`` with counts and any per-entry errors.
    """
    import asyncio

    from src.api.auth import hash_password
    from src.api.db.repositories.organization import OrganizationRepository
    from src.api.db.repositories.users import UserRepository

    result = LDAPSyncResult()

    # Resolve org.
    org_repo = OrganizationRepository(db)
    if config.org_id:
        org_id: str | None = config.org_id
    else:
        default_org = await org_repo.ensure_default_org()
        await db.commit()
        org_id = default_org.id

    # Fetch LDAP entries in a thread to avoid blocking the event loop.
    # Use the connection pool when ldap3 is available for re-use.
    try:
        from src.api.ldap.pool import get_pool

        pool = get_pool(config)
        with pool.borrow() as conn:
            entries = await asyncio.to_thread(_fetch_ldap_users, config, conn=conn)
    except ImportError:
        raise
    except Exception as exc:
        result.errors.append(f"LDAP fetch error: {exc}")
        log.error("LDAP sync fetch failed: %s", exc)
        return result

    user_repo = UserRepository(db)

    for entry in entries:
        username = entry["username"]
        email = entry["email"]
        user = None
        try:
            existing = await user_repo.get_by_username(username, org_id=org_id)
            if existing is None:
                conflict = await user_repo.get_by_username(username)
                if conflict is not None and conflict.org_id is not None:
                    # User belongs to a specific different org — skip.
                    result.errors.append(f"Username {username!r} exists in another org — skipping")
                    result.skipped += 1
                    continue
                elif conflict is not None and conflict.org_id is None:
                    # User exists but unassigned — reassign to this org.
                    await user_repo.assign_org(conflict.id, org_id)
                    result.updated += 1
                    user = conflict
                else:
                    user = await user_repo.create(
                        user_id=str(uuid.uuid4()),
                        username=username,
                        email=email,
                        password_hash=hash_password(str(uuid.uuid4())),
                        role=config.default_role,
                        org_id=org_id,
                    )
                    result.added += 1
            else:
                if existing.email.lower() != email.lower():
                    existing.email = email.lower()
                result.updated += 1
                user = existing

            # Group-to-role mapping (Phase 2.1.4)
            if user is not None and config.group_role_map:
                try:
                    user_groups = await search_groups_async(config, user_dn=entry.get("dn", ""))
                except Exception as exc:
                    result.errors.append(f"Failed to fetch groups for {username}: {exc}")
                    log.warning("LDAP group search failed for %s: %s", username, exc)
                    user_groups = []

                resolved_role = _resolve_role_from_groups(config, user_groups)
                if resolved_role is not None and user.role != resolved_role:
                    await user_repo.update_role(user.id, resolved_role)
        except Exception as exc:
            result.errors.append(f"Failed to provision {username}: {exc}")
            result.skipped += 1
            log.warning("LDAP sync: skipping %s — %s", username, exc)

    await db.commit()
    log.info(
        "LDAP sync complete: added=%d updated=%d skipped=%d errors=%d",
        result.added,
        result.updated,
        result.skipped,
        len(result.errors),
    )
    return result


# ---------------------------------------------------------------------------
# Group search (AD/LDAP) — used for group-to-role mapping (Phase 2.1.5)
# ---------------------------------------------------------------------------


def search_groups(
    config: LDAPConfig,
    *,
    group_filter: str = "(objectClass=group)",
    user_dn: str | None = None,
    conn: Connection | None = None,
) -> list[dict[str, str]]:
    """Search LDAP/AD for group entries.

    Args:
        config:       LDAP configuration.
        group_filter: LDAP filter for selecting groups.
                      Default ``(objectClass=group)`` works for Active
                      Directory; OpenLDAP may need ``(objectClass=groupOfNames)``.
        user_dn:      If provided, return only groups that list this DN as a
                      member (filter: ``(&(objectClass=group)(member=USER_DN))``).
        conn:         Optional existing connection (pool borrow).

    Returns:
        List of group dicts with keys: ``dn``, ``name``, ``description``.
    """
    from ldap3 import SUBTREE  # type: ignore[import]
    from ldap3.utils.conv import escape_filter_chars  # type: ignore[import]

    if user_dn is None:
        _validate_ldap_filter(group_filter, _ALLOWED_GROUP_OBJECT_CLASSES)

    one_shot = conn is None
    if one_shot:
        Server, Connection, Tls, ALL_ATTRIBUTES, SUBTREE, tls = _build_tls(config)
        server = Server(config.server_url, use_ssl=config.use_ssl, tls=tls, get_info=None)
        conn = Connection(
            server, user=config.bind_dn, password=config.bind_password, auto_bind=True
        )
        if not conn.bind():
            raise RuntimeError(f"LDAP bind failed: {conn.result}")

    if user_dn:
        escaped = escape_filter_chars(user_dn)
        filter_str = f"(&(objectClass=group)(member={escaped}))"
    else:
        filter_str = group_filter

    results: list[dict[str, str]] = []
    page_cookie: bytes | None = None

    while True:
        conn.search(
            search_base=config.search_base,
            search_filter=filter_str,
            search_scope=SUBTREE,
            attributes=["cn", "description", "distinguishedName"],
            paged_size=config.page_size,
            paged_cookie=page_cookie,
        )

        for entry in conn.entries:
            dn = str(entry.entry_dn) if hasattr(entry, "entry_dn") else ""
            name = str(entry.cn.value) if "cn" in entry else ""
            description = str(entry.description.value) if "description" in entry else ""
            results.append(
                {
                    "dn": dn,
                    "name": name,
                    "description": description,
                }
            )

        cookie = (
            conn.result.get("controls", {})
            .get("1.2.840.113556.1.4.319", {})
            .get("value", {})
            .get("cookie")
        )
        if not cookie:
            break
        page_cookie = cookie

    if one_shot:
        try:
            conn.unbind()
        except Exception:
            pass

    log.info("LDAP: fetched %d groups from %s", len(results), config.server_url)
    return results


async def search_groups_async(
    config: LDAPConfig,
    group_filter: str = "(objectClass=group)",
    user_dn: str | None = None,
) -> list[dict[str, str]]:
    """Async wrapper around ``search_groups`` using the connection pool."""
    import asyncio

    from src.api.ldap.pool import get_pool

    if user_dn is None:
        _validate_ldap_filter(group_filter, _ALLOWED_GROUP_OBJECT_CLASSES)

    pool = get_pool(config)
    with pool.borrow() as conn:
        return await asyncio.to_thread(
            search_groups, config, group_filter=group_filter, user_dn=user_dn, conn=conn
        )
