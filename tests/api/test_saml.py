"""Tests for SAML ACS user provisioning — cross-org isolation (Enterprise Phase 1 — task 1.2.1).

Coverage:
  - New user is created and assigned to the org declared in SAMLConfig.org_id.
  - Existing user in the *same* org: no reassignment occurs, token is issued.
  - Existing user in a *different* org: assign_org is called (cross-org reassignment
    is currently performed — these tests document the contract so that any future
    silent change is caught by CI).
  - When SAMLConfig.org_id is None, the default org is used for provisioning.
  - _provision_user returns (user, token) tuple with a non-empty JWT string.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")

from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.routes.saml import _provision_user  # noqa: E402
from cogtrix_core.api.saml.config import SAMLConfig, SAMLIdPConfig  # noqa: E402
from cogtrix_core.api.saml.provider import SAMLAssertion  # noqa: E402

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


def _make_idp() -> SAMLIdPConfig:
    return SAMLIdPConfig(
        entity_id="https://idp.example.com",
        sso_url="https://idp.example.com/sso",
        certificate="FAKECERT",
    )


def _make_config(org_id: str | None = None) -> SAMLConfig:
    return SAMLConfig(
        sp_entity_id="https://sp.example.com",
        sp_acs_url="https://sp.example.com/saml/acs",
        idp=_make_idp(),
        org_id=org_id,
    )


def _make_assertion(
    email: str = "alice@example.com",
    username: str = "alice",
    name_id: str | None = None,
) -> SAMLAssertion:
    return SAMLAssertion(
        name_id=name_id or email,
        email=email,
        username=username,
        attributes={},
        session_index=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProvisionUserNewUser:
    """_provision_user creates a new account when no email match exists."""

    def test_new_user_assigned_to_config_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="SAML Org", slug="saml-org")
                await session.commit()

                config = _make_config(org_id=org_id)
                assertion = _make_assertion(email="newuser@example.com", username="newuser")

                user, token = await _provision_user(session, assertion, config)

                assert user.email == "newuser@example.com"
                assert user.org_id == org_id
                assert token != ""

        asyncio.run(_run())

    def test_new_user_without_config_org_uses_default_org(self, sf):
        async def _run():
            async with sf() as session:
                config = _make_config(org_id=None)
                assertion = _make_assertion(email="defaultorg@example.com", username="defaultorg")

                user, token = await _provision_user(session, assertion, config)

                # Must be assigned to some org (the default org was auto-created).
                assert user.org_id is not None
                assert token != ""

        asyncio.run(_run())

    def test_returned_token_is_string(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Token Org", slug="token-org")
                await session.commit()

                config = _make_config(org_id=org_id)
                assertion = _make_assertion(email="tokentest@example.com", username="tokentest")

                user, token = await _provision_user(session, assertion, config)
                assert user.email == "tokentest@example.com"
                assert isinstance(token, str)
                assert len(token) > 0

        asyncio.run(_run())


class TestProvisionUserExistingUser:
    """_provision_user behaviour when a user with the same email already exists."""

    def test_existing_user_same_org_no_reassignment(self, sf):
        """User already in the correct org: assign_org must not mutate org_id."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Same Org", slug="same-org")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="existingsame",
                    email="existingsame@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                config = _make_config(org_id=org_id)
                assertion = _make_assertion(
                    email="existingsame@example.com", username="existingsame"
                )

                user, token = await _provision_user(session, assertion, config)

                assert user.email == "existingsame@example.com"
                assert user.org_id == org_id
                assert token != ""

        asyncio.run(_run())


class TestProvisionUserUsernameCollisions:
    """_provision_user retries username collisions and fails cleanly after exhaustion."""

    def test_username_collision_retries_with_fresh_suffix(self, sf, monkeypatch):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Collision Org", slug="collision-org")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="takenname",
                    email="taken@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                username_gen = Mock(side_effect=["takenname", "takenname_2"])
                monkeypatch.setattr("cogtrix_core.api.routes.saml._unique_username", username_gen)

                config = _make_config(org_id=org_id)
                assertion = _make_assertion(
                    email="collision@example.com",
                    username="takenname",
                )

                user, token = await _provision_user(session, assertion, config)

                assert user.email == "collision@example.com"
                assert user.username == "takenname_2"
                assert token != ""
                assert username_gen.call_count == 2

        asyncio.run(_run())

    def test_username_collision_exhausts_retries_with_409(self, sf, monkeypatch):
        async def _run():
            from fastapi import HTTPException

            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Retry Org", slug="retry-org")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="takenname",
                    email="taken@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                username_gen = Mock(return_value="takenname")
                monkeypatch.setattr("cogtrix_core.api.routes.saml._unique_username", username_gen)

                config = _make_config(org_id=org_id)
                assertion = _make_assertion(
                    email="collision@example.com",
                    username="takenname",
                )

                with pytest.raises(HTTPException) as exc_info:
                    await _provision_user(session, assertion, config)

                assert exc_info.value.status_code == 409
                assert exc_info.value.detail["code"] == "VALIDATION_ERROR"
                assert username_gen.call_count == 3
                assert await user_repo.get_by_email("collision@example.com") is None

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
                await org_repo.create(org_id=org_a_id, name="Org A", slug="org-a-saml")
                await org_repo.create(org_id=org_b_id, name="Org B", slug="org-b-saml")
                await session.commit()

                from cogtrix_core.api.auth import hash_password

                await user_repo.create(
                    user_id=_uid(),
                    username="crossorguser",
                    email="crossorg@example.com",
                    password_hash=hash_password("irrelevant"),
                    role="user",
                    org_id=org_a_id,
                )
                await session.commit()

                # Config points to org B — user is in org A.
                config = _make_config(org_id=org_b_id)
                assertion = _make_assertion(email="crossorg@example.com", username="crossorguser")

                with pytest.raises(HTTPException) as exc_info:
                    await _provision_user(session, assertion, config)

                # 422 (not 409) prevents an attacker from confirming the user exists.
                assert exc_info.value.status_code == 422

                # Verify the user was NOT moved to org B.
                untouched = await user_repo.get_by_email("crossorg@example.com")
                assert untouched is not None
                assert untouched.org_id == org_a_id

        asyncio.run(_run())
