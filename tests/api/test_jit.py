"""Tests for JIT user provisioning — cross-org isolation (Enterprise Phase 1 — task 1.2.5).

Coverage:
  - New user is provisioned and assigned to the org declared in JITConfig.org_id.
  - New user without a config org_id is assigned to the default org.
  - Existing user in the *same* org: no reassignment, token is issued.
  - Existing user in a *different* org: assign_org is called (cross-org
    reassignment is currently performed — tests document the contract).
  - Domain not in allowlist: 403 JIT_DOMAIN_DENIED.
  - JIT disabled: 503 JIT_DISABLED.
  - Capacity exceeded: 403 JIT_CAPACITY_EXCEEDED.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.jit.config import JITConfig  # noqa: E402
from cogtrix_core.api.jit.provisioning import provision_jit_user  # noqa: E402

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tests — new user provisioning
# ---------------------------------------------------------------------------


class TestProvisionJITUserNew:
    """provision_jit_user creates a new account when no email match exists."""

    def test_new_user_assigned_to_config_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="JIT Org", slug="jit-org")
                await session.commit()

                config = JITConfig(enabled=True, org_id=org_id)
                user, token = await provision_jit_user(
                    email="jituser@example.com",
                    username="jituser",
                    config=config,
                    db=session,
                )

                assert user.email == "jituser@example.com"
                assert user.org_id == org_id
                assert isinstance(token, str) and token != ""

        asyncio.run(_run())

    def test_new_user_without_config_org_uses_default_org(self, sf):
        async def _run():
            async with sf() as session:
                config = JITConfig(enabled=True, org_id=None)
                user, token = await provision_jit_user(
                    email="jitdefault@example.com",
                    username="jitdefault",
                    config=config,
                    db=session,
                )

                assert user.org_id is not None
                assert isinstance(token, str) and token != ""

        asyncio.run(_run())

    def test_email_at_sign_username_is_sanitised(self, sf):
        """Username passed as an email address must be stripped to its local part."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="JIT Sanitise", slug="jit-sanitise")
                await session.commit()

                config = JITConfig(enabled=True, org_id=org_id)
                user, token = await provision_jit_user(
                    email="sanitise@example.com",
                    username="sanitise@example.com",
                    config=config,
                    db=session,
                )

                # Username must not contain the @ domain part.
                assert "@" not in user.username
                assert isinstance(token, str) and token != ""

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests — existing user
# ---------------------------------------------------------------------------


class TestProvisionJITUserExisting:
    """provision_jit_user behaviour when a user with the same email already exists."""

    def test_existing_user_same_org_no_reassignment(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="JIT Same", slug="jit-same")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="jitexisting",
                    email="jitexisting@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                config = JITConfig(enabled=True, org_id=org_id)
                user, token = await provision_jit_user(
                    email="jitexisting@example.com",
                    username="jitexisting",
                    config=config,
                    db=session,
                )

                assert user.org_id == org_id
                assert token != ""

        asyncio.run(_run())

    def test_existing_user_different_org_raises_422(self, sf):
        """Cross-org conflict returns 422 (not 409) to prevent user enumeration."""

        async def _run():
            from fastapi import HTTPException

            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_a_id = _uid()
                org_b_id = _uid()
                await org_repo.create(org_id=org_a_id, name="JIT Org A", slug="jit-org-a")
                await org_repo.create(org_id=org_b_id, name="JIT Org B", slug="jit-org-b")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="jitcrossorg",
                    email="jitcross@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_a_id,
                )
                await session.commit()

                # Config points to org B — user is in org A.
                config = JITConfig(enabled=True, org_id=org_b_id)

                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="jitcross@example.com",
                        username="jitcrossorg",
                        config=config,
                        db=session,
                    )

                # 422 (not 409) prevents an attacker from confirming the user exists.
                assert exc_info.value.status_code == 422

                # Verify the user was NOT moved to org B.
                untouched = await user_repo.get_by_email("jitcross@example.com")
                assert untouched is not None
                assert untouched.org_id == org_a_id

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests — policy enforcement
# ---------------------------------------------------------------------------


class TestProvisionJITUserPolicies:
    """Domain allowlist, disabled flag, and capacity limit enforcement."""

    def test_domain_denied_raises_403(self, sf):
        async def _run():
            async with sf() as session:
                config = JITConfig(
                    enabled=True,
                    allowed_domains=["allowed.com"],
                    org_id=None,
                )
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="blocked@blocked.com",
                        username="blocked",
                        config=config,
                        db=session,
                    )
                assert exc_info.value.status_code == 403
                assert exc_info.value.detail["code"] == "JIT_DOMAIN_DENIED"

        asyncio.run(_run())

    def test_jit_disabled_raises_503(self, sf):
        async def _run():
            async with sf() as session:
                config = JITConfig(enabled=False)
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="any@example.com",
                        username="any",
                        config=config,
                        db=session,
                    )
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["code"] == "JIT_DISABLED"

        asyncio.run(_run())

    def test_capacity_exceeded_raises_403(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="JIT Cap", slug="jit-cap")
                await session.commit()

                # Fill the org to max_users=1.
                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="jitcap1",
                    email="jitcap1@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                config = JITConfig(enabled=True, org_id=org_id, max_users=1)
                with pytest.raises(HTTPException) as exc_info:
                    await provision_jit_user(
                        email="jitcap2@example.com",
                        username="jitcap2",
                        config=config,
                        db=session,
                    )
                assert exc_info.value.status_code == 403
                assert exc_info.value.detail["code"] == "JIT_CAPACITY_EXCEEDED"

        asyncio.run(_run())

    def test_allowed_domain_passes(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="JIT Allow", slug="jit-allow")
                await session.commit()

                config = JITConfig(
                    enabled=True,
                    allowed_domains=["allowed.com"],
                    org_id=org_id,
                )
                user, token = await provision_jit_user(
                    email="hello@allowed.com",
                    username="hello",
                    config=config,
                    db=session,
                )
                assert user.email == "hello@allowed.com"
                assert token != ""

        asyncio.run(_run())
