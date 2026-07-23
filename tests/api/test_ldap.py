"""Tests for LDAP/AD user sync — cross-org isolation (Enterprise Phase 1 — task 1.2.3).

Coverage:
  - sync_users provisions a new user and assigns it to LDAPConfig.org_id.
  - sync_users updates the email when an existing username has a different email.
  - sync_users reassigns a user to the configured org when the user is in a
    different org (documents the current cross-org reassignment contract).
  - sync_users with no org_id falls back to the default org.
  - sync_users returns an LDAPSyncResult with correct added/updated counts.
  - _fetch_ldap_users raises ImportError with an actionable message when ldap3
    is not installed.
  - LDAPSyncResult.total_processed and .success properties.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch  # type: ignore[attr-defined]

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ldap3")

from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.ldap.config import (  # noqa: E402
    _ALLOWED_GROUP_OBJECT_CLASSES,
    _ALLOWED_USER_OBJECT_CLASSES,
    LDAPConfig,
    _validate_ldap_filter,
)
from cogtrix_core.api.ldap.sync import LDAPSyncResult, _fetch_ldap_users, sync_users  # noqa: E402

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


def _make_ldap_config(org_id: str | None = None) -> LDAPConfig:
    return LDAPConfig(
        server_url="ldap://ldap.example.com:389",
        bind_dn="cn=svc,dc=example,dc=com",
        bind_password="secret",
        search_base="ou=users,dc=example,dc=com",
        org_id=org_id,
        use_ssl=False,
    )


# ---------------------------------------------------------------------------
# LDAPSyncResult unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestLDAPSyncResult:
    def test_total_processed_sums_all_buckets(self):
        result = LDAPSyncResult(added=3, updated=2, skipped=1)
        assert result.total_processed == 6

    def test_success_true_when_no_errors(self):
        result = LDAPSyncResult(added=1)
        assert result.success is True

    def test_success_false_when_errors_present(self):
        result = LDAPSyncResult(errors=["something went wrong"])
        assert result.success is False

    def test_empty_result_is_successful(self):
        result = LDAPSyncResult()
        assert result.total_processed == 0
        assert result.success is True


# ---------------------------------------------------------------------------
# _fetch_ldap_users — ImportError path (no ldap3 installed)
# ---------------------------------------------------------------------------


class TestFetchLDAPUsersImportError:
    def test_raises_import_error_with_actionable_message(self, monkeypatch):
        """_fetch_ldap_users must raise ImportError when ldap3 is absent."""
        import builtins

        real_import = builtins.__import__

        def _block_ldap3(name, *args, **kwargs):
            if name == "ldap3" or name.startswith("ldap3."):
                raise ImportError("No module named 'ldap3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_ldap3)

        config = _make_ldap_config()
        with pytest.raises(ImportError, match="cogtrix\\[ldap\\]"):
            _fetch_ldap_users(config)


# ---------------------------------------------------------------------------
# sync_users — uses mocked _fetch_ldap_users to avoid network I/O
# ---------------------------------------------------------------------------


def _patch_fetch(entries: list[dict]):
    """Return a context manager that patches asyncio.to_thread and get_pool
    so sync_users runs without real LDAP I/O.

    sync_users imports asyncio inside the function body, so we patch via
    the top-level asyncio module rather than the sync module's attribute.
    """
    fake_conn = MagicMock()
    fake_conn.closed = False
    fake_pool = MagicMock()
    fake_pool.borrow.return_value.__enter__ = MagicMock(return_value=fake_conn)
    fake_pool.borrow.return_value.__exit__ = MagicMock(return_value=False)

    stack = ExitStack()
    stack.enter_context(patch("cogtrix_core.api.ldap.pool.get_pool", return_value=fake_pool))
    stack.enter_context(patch("asyncio.to_thread", AsyncMock(return_value=entries)))
    return stack


class TestSyncUsersNew:
    """sync_users provisions new users from LDAP entries."""

    def test_new_user_added_to_configured_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Org", slug="ldap-org")
                await session.commit()

                entries = [{"username": "ldapuser1", "email": "ldapuser1@example.com"}]
                config = _make_ldap_config(org_id=org_id)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                assert result.added == 1
                assert result.updated == 0
                assert result.success is True

                user_repo = UserRepository(session)
                user = await user_repo.get_by_username("ldapuser1")
                assert user is not None
                assert user.org_id == org_id

        asyncio.run(_run())

    def test_new_user_without_org_uses_default_org(self, sf):
        async def _run():
            async with sf() as session:
                entries = [{"username": "ldapdefault", "email": "ldapdefault@example.com"}]
                config = _make_ldap_config(org_id=None)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                assert result.added == 1
                user_repo = UserRepository(session)
                user = await user_repo.get_by_username("ldapdefault")
                assert user is not None
                assert user.org_id is not None

        asyncio.run(_run())

    def test_multiple_entries_all_added(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Multi", slug="ldap-multi")
                await session.commit()

                entries = [
                    {"username": "multi1", "email": "multi1@example.com"},
                    {"username": "multi2", "email": "multi2@example.com"},
                    {"username": "multi3", "email": "multi3@example.com"},
                ]
                config = _make_ldap_config(org_id=org_id)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                assert result.added == 3
                assert result.success is True

        asyncio.run(_run())


class TestSyncUsersExisting:
    """sync_users update behaviour for users that already exist."""

    def test_existing_user_email_updated(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Update", slug="ldap-update")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="ldapupdate",
                    email="old@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                entries = [{"username": "ldapupdate", "email": "new@example.com"}]
                config = _make_ldap_config(org_id=org_id)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                assert result.updated == 1
                assert result.added == 0

                user = await user_repo.get_by_username("ldapupdate")
                assert user is not None
                assert user.email == "new@example.com"

        asyncio.run(_run())

    def test_existing_user_same_email_counted_as_updated(self, sf):
        """User exists with the same email — still counted as updated (no-op mutation path)."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP NoOp", slug="ldap-noop")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="ldapnoop",
                    email="same@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                entries = [{"username": "ldapnoop", "email": "same@example.com"}]
                config = _make_ldap_config(org_id=org_id)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                assert result.updated == 1
                assert result.added == 0
                assert result.success is True

        asyncio.run(_run())

    def test_existing_user_different_org_is_skipped(self, sf):
        """Cross-org conflict: LDAP entry whose username exists in another org is skipped."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_a_id = _uid()
                org_b_id = _uid()
                await org_repo.create(org_id=org_a_id, name="LDAP OrgA", slug="ldap-org-a")
                await org_repo.create(org_id=org_b_id, name="LDAP OrgB", slug="ldap-org-b")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="ldapcrossorg",
                    email="ldapcross@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_a_id,
                )
                await session.commit()

                entries = [{"username": "ldapcrossorg", "email": "ldapcross@example.com"}]
                config = _make_ldap_config(org_id=org_b_id)

                with _patch_fetch(entries):
                    result = await sync_users(config, session)

                # Entry is skipped, not updated.
                assert result.updated == 0
                assert result.skipped == 1
                assert any("another org" in e for e in result.errors)

                # Verify the user was NOT moved to org B.
                untouched = await user_repo.get_by_username("ldapcrossorg")
                assert untouched is not None
                assert untouched.org_id == org_a_id

        asyncio.run(_run())


class TestSyncUsersFetchError:
    """sync_users gracefully handles LDAP fetch failures."""

    def test_fetch_exception_recorded_in_errors(self, sf):
        async def _run():
            async with sf() as session:
                config = _make_ldap_config()

                exc = RuntimeError("LDAP connection refused")
                with _patch_fetch([]):
                    # Override the asyncio.to_thread mock with a side-effect.
                    with patch(
                        "asyncio.to_thread",
                        AsyncMock(side_effect=exc),
                    ):
                        result = await sync_users(config, session)

                assert result.added == 0
                assert len(result.errors) == 1
                assert "LDAP fetch error" in result.errors[0]
                assert result.success is False

        asyncio.run(_run())


class TestSyncUsersGroupRoleMapping:
    """Group-to-role mapping during LDAP sync (Phase 2.1.4)."""

    def test_mapped_group_assigns_role(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Role", slug="ldap-role")
                await session.commit()

                entries = [
                    {
                        "username": "adminuser",
                        "email": "admin@example.com",
                        "dn": "CN=Admin User,OU=users,DC=example,DC=com",
                    }
                ]
                config = _make_ldap_config(org_id=org_id)
                config.group_role_map = {
                    "CN=Admins,OU=groups,DC=example,DC=com": "admin",
                }

                with _patch_fetch(entries):
                    with patch(
                        "cogtrix_core.api.ldap.sync.search_groups_async",
                        AsyncMock(
                            return_value=[
                                {
                                    "dn": "CN=Admins,OU=groups,DC=example,DC=com",
                                    "name": "Admins",
                                    "description": "",
                                }
                            ]
                        ),
                    ):
                        result = await sync_users(config, session)

                assert result.added == 1
                user = await user_repo.get_by_username("adminuser")
                assert user is not None
                assert user.role == "admin"

        asyncio.run(_run())

    def test_first_match_wins_for_multiple_groups(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Multi", slug="ldap-multi")
                await session.commit()

                entries = [
                    {
                        "username": "multirole",
                        "email": "multi@example.com",
                        "dn": "CN=Multi,OU=users,DC=example,DC=com",
                    }
                ]
                config = _make_ldap_config(org_id=org_id)
                config.group_role_map = {
                    "CN=Admins,OU=groups,DC=example,DC=com": "admin",
                    "CN=Users,OU=groups,DC=example,DC=com": "user",
                }

                with _patch_fetch(entries):
                    with patch(
                        "cogtrix_core.api.ldap.sync.search_groups_async",
                        AsyncMock(
                            return_value=[
                                {
                                    "dn": "CN=Users,OU=groups,DC=example,DC=com",
                                    "name": "Users",
                                    "description": "",
                                },
                                {
                                    "dn": "CN=Admins,OU=groups,DC=example,DC=com",
                                    "name": "Admins",
                                    "description": "",
                                },
                            ]
                        ),
                    ):
                        result = await sync_users(config, session)

                assert result.added == 1
                user = await user_repo.get_by_username("multirole")
                assert user is not None
                assert user.role == "admin"  # first match in config order wins

        asyncio.run(_run())

    def test_no_mapped_group_uses_default(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP Def", slug="ldap-def")
                await session.commit()

                entries = [
                    {
                        "username": "defaultuser",
                        "email": "default@example.com",
                        "dn": "CN=Default,OU=users,DC=example,DC=com",
                    }
                ]
                config = _make_ldap_config(org_id=org_id)
                config.group_role_map = {
                    "CN=Admins,OU=groups,DC=example,DC=com": "admin",
                }
                config.group_role_default = "user"

                with _patch_fetch(entries):
                    with patch(
                        "cogtrix_core.api.ldap.sync.search_groups_async",
                        AsyncMock(return_value=[]),
                    ):
                        result = await sync_users(config, session)

                assert result.added == 1
                user = await user_repo.get_by_username("defaultuser")
                assert user is not None
                assert user.role == "user"

        asyncio.run(_run())

    def test_no_group_role_map_leaves_role_unchanged(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="LDAP NoMap", slug="ldap-nomap")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="nomapuser",
                    email="nomap@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="viewer",
                    org_id=org_id,
                )
                await session.commit()

                entries = [
                    {
                        "username": "nomapuser",
                        "email": "nomap@example.com",
                        "dn": "CN=NoMap,OU=users,DC=example,DC=com",
                    }
                ]
                config = _make_ldap_config(org_id=org_id)
                # group_role_map is empty by default

                with _patch_fetch(entries):
                    with patch("cogtrix_core.api.ldap.sync.search_groups_async") as mock_search:
                        result = await sync_users(config, session)

                assert result.updated == 1
                user = await user_repo.get_by_username("nomapuser")
                assert user is not None
                assert user.role == "viewer"
                mock_search.assert_not_called()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# LDAP filter validation (security — issue #429)
# ---------------------------------------------------------------------------


class TestValidateLDAPFilter:
    """_validate_ldap_filter rejects overly broad or malformed filters."""

    # -- valid user filters --

    def test_valid_person_filter(self):
        _validate_ldap_filter("(objectClass=person)", _ALLOWED_USER_OBJECT_CLASSES)

    def test_valid_user_filter(self):
        _validate_ldap_filter("(objectClass=user)", _ALLOWED_USER_OBJECT_CLASSES)

    def test_valid_inetorgperson_filter(self):
        _validate_ldap_filter("(objectClass=inetOrgPerson)", _ALLOWED_USER_OBJECT_CLASSES)

    def test_valid_and_wrapped_person_filter(self):
        _validate_ldap_filter("(&(objectClass=person)(cn=foo))", _ALLOWED_USER_OBJECT_CLASSES)

    def test_valid_and_wrapped_user_filter(self):
        _validate_ldap_filter(
            "(&(objectClass=user)(department=engineering))",
            _ALLOWED_USER_OBJECT_CLASSES,
        )

    # -- valid group filters --

    def test_valid_group_filter(self):
        _validate_ldap_filter("(objectClass=group)", _ALLOWED_GROUP_OBJECT_CLASSES)

    def test_valid_groupofnames_filter(self):
        _validate_ldap_filter("(objectClass=groupOfNames)", _ALLOWED_GROUP_OBJECT_CLASSES)

    def test_valid_and_wrapped_group_filter(self):
        _validate_ldap_filter(
            "(&(objectClass=group)(cn=admins))",
            _ALLOWED_GROUP_OBJECT_CLASSES,
        )

    # -- invalid filters --

    def test_rejects_unparenthesized_filter(self):
        with pytest.raises(ValueError, match="parenthesized"):
            _validate_ldap_filter("objectClass=person", _ALLOWED_USER_OBJECT_CLASSES)

    def test_rejects_or_operator(self):
        with pytest.raises(ValueError, match="OR operator"):
            _validate_ldap_filter(
                "(|(objectClass=person)(objectClass=user))",
                _ALLOWED_USER_OBJECT_CLASSES,
            )

    def test_rejects_wildcard_objectclass(self):
        with pytest.raises(ValueError, match="wildcard objectClass"):
            _validate_ldap_filter("(objectClass=*)", _ALLOWED_USER_OBJECT_CLASSES)

    def test_rejects_unknown_objectclass(self):
        with pytest.raises(ValueError, match="not in allowlist"):
            _validate_ldap_filter(
                "(objectClass=organizationalUnit)",
                _ALLOWED_USER_OBJECT_CLASSES,
            )

    def test_rejects_and_without_objectclass(self):
        with pytest.raises(ValueError, match="must target objectClass"):
            _validate_ldap_filter("(&(cn=foo)(mail=bar))", _ALLOWED_USER_OBJECT_CLASSES)

    def test_rejects_invalid_structure(self):
        with pytest.raises(ValueError, match="invalid structure"):
            _validate_ldap_filter(
                "(objectClass=person)(cn=foo)",
                _ALLOWED_USER_OBJECT_CLASSES,
            )


class TestConfigureLdapValidatesFilter:
    """configure_ldap rejects configs with unsafe search_filter values."""

    def test_rejects_broad_filter_at_configure_time(self):
        from cogtrix_core.api.ldap.config import configure_ldap

        bad_config = LDAPConfig(
            server_url="ldap://example.com:389",
            bind_dn="cn=svc,dc=example,dc=com",
            bind_password="secret",
            search_base="ou=users,dc=example,dc=com",
            search_filter="(|(objectClass=*))",
            use_ssl=False,
        )
        with pytest.raises(ValueError, match="OR operator"):
            configure_ldap(bad_config)

    def test_accepts_safe_filter_at_configure_time(self):
        from cogtrix_core.api.ldap.config import configure_ldap, get_ldap_config

        good_config = LDAPConfig(
            server_url="ldap://example.com:389",
            bind_dn="cn=svc,dc=example,dc=com",
            bind_password="secret",
            search_base="ou=users,dc=example,dc=com",
            search_filter="(&(objectClass=person)(department=engineering))",
            use_ssl=False,
        )
        configure_ldap(good_config)
        assert get_ldap_config() is not None
