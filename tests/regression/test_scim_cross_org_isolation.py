"""Regression tests for SCIM cross-organization isolation (ISSUE #578).

Tests ensure SCIM operations cannot affect users in other organizations.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from cogtrix_core.api.app import create_app  # noqa: E402
from cogtrix_core.api.auth import create_access_token, hash_password  # noqa: E402
from cogtrix_core.api.db.engine import Base, get_db  # noqa: E402
from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402
from cogtrix_core.api.saml.config import SAMLConfig, SAMLIdPConfig, configure_saml  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    """Create an admin auth header for the given user ID."""
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


def _user_header(user_id: str) -> dict:
    """Create a regular user auth header for the given user ID."""
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def reset_saml_config():
    """Reset global SAML config before and after each test."""
    import cogtrix_core.api.saml.config as _cfg

    _cfg._saml_config = None
    yield
    _cfg._saml_config = None


@pytest.fixture()
def scim_test_client():
    """Return a TestClient with in-memory SQLite DB and pre-seeded orgs."""
    # Configure SAML with SCIM base URL to enable SCIM endpoints
    saml_config = SAMLConfig(
        sp_entity_id="http://cogtrix.test/sp",
        sp_acs_url="http://cogtrix.test/saml/acs",
        idp=SAMLIdPConfig(
            entity_id="http://cogtrix.test/idp",
            sso_url="http://cogtrix.test/idp/sso",
            certificate="dummy-cert-for-testing",
        ),
        scim_base_url="http://cogtrix.test",
    )
    configure_saml(saml_config)

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with factory() as session:
            # Create two organizations
            org_repo = OrganizationRepository(session)
            org1 = await org_repo.create(org_id="org_1", name="Org 1", slug="org-1")
            org2 = await org_repo.create(org_id="org_2", name="Org 2", slug="org-2")
            await session.commit()

            # Create users in both orgs
            user1 = await _create_user(session, "user1", org1.id, "user1@org1.test")
            user2 = await _create_user(session, "user2", org2.id, "user2@org2.test")
            await session.commit()

            return org1.id, org2.id, user1.id, user2.id

    org1_id, org2_id, user1_id, user2_id = asyncio.run(_setup())
    admin_header = _admin_header(user1_id)

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, org1_id, org2_id, user1_id, user2_id, admin_header

    asyncio.run(engine.dispose())


async def _create_user(db_session, username: str, org_id: str, email: str | None = None):
    """Helper to create a user in a specific org for testing."""
    repo = UserRepository(db_session)
    return await repo.create(
        user_id=_uid(),
        username=username,
        email=email or f"{username}@test.com",
        password_hash=hash_password("dummy"),
        role="user",
        org_id=org_id,
    )


class TestScimCrossOrgIsolation:
    """Regression tests for cross-org isolation in SCIM endpoints."""

    def test_scim_list_users_only_returns_org_users(self, scim_test_client):
        """Regression: SCIM list users must only return users from the caller's org."""
        client, _, _, user1_id, _, admin_header = scim_test_client

        resp = client.get("/scim/v2/Users", headers=admin_header)
        assert resp.status_code == 200
        data = resp.json()
        resources = data.get("Resources", [])

        # Should only see user1 from org1
        assert len(resources) == 1
        assert resources[0]["userName"] == "user1"

    def test_scim_create_user_cannot_create_duplicate_username_in_other_org(self, scim_test_client):
        """Regression: SCIM create must not allow duplicate username across orgs (422 opaque error)."""
        client, _, _, user1_id, user2_id, admin_header = scim_test_client

        # Try to create a user with username that exists in org2
        payload = {
            "userName": "user2",
            "emails": [{"value": "different@org1.test"}],
            "active": True,
        }

        resp = client.post("/scim/v2/Users", json=payload, headers=admin_header)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
        # Error body should be opaque (same as same-org conflict)
        data = resp.json()
        assert data["detail"] == "User already exists."

    def test_scim_create_user_cannot_create_duplicate_email_in_other_org(self, scim_test_client):
        """Regression: SCIM create must not allow duplicate email across orgs (422 opaque error)."""
        client, _, _, user1_id, user2_id, admin_header = scim_test_client

        # Try to create a user with email that exists in org2
        payload = {
            "userName": "newuser1",
            "emails": [{"value": "user2@org2.test"}],
            "active": True,
        }

        resp = client.post("/scim/v2/Users", json=payload, headers=admin_header)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["detail"] == "User already exists."

    def test_scim_get_user_only_returns_org_user(self, scim_test_client):
        """Regression: SCIM get user must only return users from the caller's org."""
        client, _, _, _, user2_id, admin_header = scim_test_client

        # Try to get user from org2 (should return 404, not expose existence)
        resp = client.get(f"/scim/v2/Users/{user2_id}", headers=admin_header)
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    def test_scim_get_user_returns_org_user(self, scim_test_client):
        """Regression: SCIM get user should work for valid user in caller's org."""
        client, _, _, user1_id, _, admin_header = scim_test_client

        resp = client.get(f"/scim/v2/Users/{user1_id}", headers=admin_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["userName"] == "user1"

    def test_scim_put_user_only_modifies_org_user(self, scim_test_client):
        """Regression: SCIM put must only modify users from the caller's org."""
        client, _, _, _, user2_id, admin_header = scim_test_client

        # Try to modify user from org2 (should return 404, not expose existence)
        payload = {
            "userName": "user2_modified",
            "emails": [{"value": "modified@org2.test"}],
            "active": False,
        }
        resp = client.put(f"/scim/v2/Users/{user2_id}", json=payload, headers=admin_header)
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    def test_scim_patch_user_only_modifies_org_user(self, scim_test_client):
        """Regression: SCIM patch must only modify users from the caller's org."""
        client, _, _, _, user2_id, admin_header = scim_test_client

        # Try to modify user from org2 (should return 404, not expose existence)
        payload = {
            "Operations": [{"op": "replace", "path": "userName", "value": "user2_modified"}],
        }
        resp = client.patch(f"/scim/v2/Users/{user2_id}", json=payload, headers=admin_header)
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

    def test_scim_delete_user_only_deletes_org_user(self, scim_test_client):
        """Regression: SCIM delete must only delete users from the caller's org."""
        client, _, _, _, user2_id, admin_header = scim_test_client

        # Try to delete user from org2 (should return 404, not expose existence)
        resp = client.delete(f"/scim/v2/Users/{user2_id}", headers=admin_header)
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
        assert resp.status_code != 204, "Should not succeed in deleting other org's user"

    def test_scim_create_user_can_create_in_own_org(self, scim_test_client):
        """Regression: SCIM create should work normally for org-local operations."""
        client, _, _, user1_id, _, admin_header = scim_test_client

        payload = {
            "userName": "newuser",
            "emails": [{"value": "newuser@org1.test"}],
            "active": True,
        }

        resp = client.post("/scim/v2/Users", json=payload, headers=admin_header)
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["userName"] == "newuser"

    def test_scim_create_user_fails_on_same_org_duplicate(self, scim_test_client):
        """Regression: SCIM create should fail on same-org duplicate with 409."""
        client, _, _, user1_id, _, admin_header = scim_test_client

        # Try to create user with existing username in same org
        payload = {
            "userName": "user1",
            "emails": [{"value": "different@org1.test"}],
            "active": True,
        }

        resp = client.post("/scim/v2/Users", json=payload, headers=admin_header)
        assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["detail"] == "User already exists."
