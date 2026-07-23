"""Tests for SCIM 2.0 provisioning API (Enterprise Phase 1 — task 1.2.2).

Covers:
  - SCIMUser / SCIMListResponse / SCIMError schema serialisation
  - parse_scim_filter
  - user_to_scim mapping
  - GET /scim/v2/ServiceProviderConfig
  - GET /scim/v2/Users (list, filter, pagination)
  - POST /scim/v2/Users (create, duplicate userName/email)
  - GET /scim/v2/Users/{id} (get, 404, cross-org isolation)
  - PUT /scim/v2/Users/{id} (replace)
  - PATCH /scim/v2/Users/{id} (partial update)
  - DELETE /scim/v2/Users/{id} (delete, 404)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import patch  # noqa: E402

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from src.api.auth import create_access_token  # noqa: E402
from src.api.db.engine import Base, get_db  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402
from src.api.saml.config import (  # noqa: E402
    SAMLConfig,
    SAMLIdPConfig,
    configure_saml,
)
from src.api.scim.mapping import parse_scim_filter, user_to_scim  # noqa: E402
from src.api.scim.schemas import SCIMListResponse, SCIMUser  # noqa: E402


def _uid() -> str:
    return str(uuid.uuid4())


def _admin_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="admin")
    return {"Authorization": f"Bearer {token}"}


def _user_header(user_id: str) -> dict:
    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        token = create_access_token(user_id=user_id, role="user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def reset_saml_config():
    """Reset global SAML config before and after each test."""
    import src.api.saml.config as _cfg

    _cfg._saml_config = None
    yield
    _cfg._saml_config = None


def _make_saml_config(*, scim_base_url: str = "https://scim.example.com") -> SAMLConfig:
    return SAMLConfig(
        sp_entity_id="https://sp.example.com/saml/metadata",
        sp_acs_url="https://sp.example.com/api/v1/saml/acs",
        idp=SAMLIdPConfig(
            entity_id="https://idp.example.com",
            sso_url="https://idp.example.com/sso",
            certificate="FAKECERT",
        ),
        scim_base_url=scim_base_url,
    )


# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------


class TestSCIMSchemas:
    def test_scim_user_default_schemas(self):
        u = SCIMUser(userName="alice")
        assert "urn:ietf:params:scim:schemas:core:2.0:User" in u.schemas

    def test_scim_list_response_default_schemas(self):
        lr = SCIMListResponse(totalResults=0, itemsPerPage=0)
        assert "urn:ietf:params:scim:api:messages:2.0:ListResponse" in lr.schemas

    def test_scim_user_serialise_excludes_none(self):
        u = SCIMUser(userName="alice")
        d = u.model_dump(mode="json", exclude_none=True)
        assert "id" not in d
        assert "userName" in d


class TestParseSCIMFilter:
    def test_username_eq(self):
        f = parse_scim_filter('userName eq "alice"')
        assert f == {"attr": "username", "op": "eq", "value": "alice"}

    def test_email_eq(self):
        f = parse_scim_filter('emails.value eq "alice@example.com"')
        assert f is not None
        assert f["attr"] == "emails.value"
        assert f["value"] == "alice@example.com"

    def test_case_insensitive_op(self):
        f = parse_scim_filter('userName EQ "alice"')
        assert f is not None
        assert f["op"] == "eq"

    def test_none_returns_none(self):
        assert parse_scim_filter(None) is None

    def test_empty_returns_none(self):
        assert parse_scim_filter("") is None

    def test_invalid_returns_none(self):
        assert parse_scim_filter("not a valid filter") is None

    def test_compound_and_filter(self):
        f = parse_scim_filter('userName eq "alice" and active eq true')
        assert isinstance(f, list)
        assert len(f) == 2
        assert f[0] == {"attr": "username", "op": "eq", "value": "alice"}
        assert f[1] == {"attr": "active", "op": "eq", "value": "true"}

    def test_compound_and_filter_three_clauses(self):
        f = parse_scim_filter(
            'userName eq "alice" and active eq true and emails.value eq "a@b.com"'
        )
        assert isinstance(f, list)
        assert len(f) == 3

    def test_compound_and_with_quoted_and(self):
        # "and" inside a quoted value should not split
        f = parse_scim_filter('userName eq "alice and bob" and active eq true')
        assert isinstance(f, list)
        assert len(f) == 2
        assert f[0]["value"] == "alice and bob"
        assert f[1]["value"] == "true"

    def test_compound_or_not_supported(self):
        # "or" is not supported — should return None
        assert parse_scim_filter('userName eq "alice" or active eq true') is None


class TestUserToSCIM:
    def test_basic_mapping(self, tmp_path):
        from datetime import UTC, datetime

        from src.api.db.models import User

        user = User()
        user.id = _uid()
        user.username = "alice"
        user.email = "alice@example.com"
        user.role = "user"
        user.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        user.org_id = None

        scim_user = user_to_scim(user, "https://example.com")
        assert scim_user.id == user.id
        assert scim_user.userName == "alice"
        assert scim_user.emails[0].value == "alice@example.com"
        assert scim_user.emails[0].primary is True
        assert scim_user.meta is not None
        assert scim_user.meta.resourceType == "User"
        assert "/scim/v2/Users/" in scim_user.meta.location


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scim_setup():
    """Return (TestClient, org_id, admin_user_id)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    org_id = _uid()
    admin_id = _uid()

    async def _seed():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as session:
            org_repo = OrganizationRepository(session)
            user_repo = UserRepository(session)
            await org_repo.create(org_id=org_id, name="Test Org", slug="test-org")
            await user_repo.create(
                user_id=admin_id,
                username="admin",
                email="admin@example.com",
                password_hash="h",
                role="admin",
                org_id=org_id,
            )
            await session.commit()

    asyncio.run(_seed())

    from src.api.app import create_app

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": _TEST_JWT_SECRET}):
        app = create_app()
        configure_saml(_make_saml_config())

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
            yield client, org_id, admin_id

    asyncio.run(engine.dispose())


# ---------------------------------------------------------------------------
# ServiceProviderConfig
# ---------------------------------------------------------------------------


class TestSCIMServiceProviderConfig:
    def test_returns_200_for_admin(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get("/scim/v2/ServiceProviderConfig", headers=_admin_header(admin_id))
        assert r.status_code == 200
        data = r.json()
        assert "urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig" in data["schemas"]
        assert data["patch"]["supported"] is True

    def test_requires_admin(self, scim_setup):
        client, _, admin_id = scim_setup
        non_admin_id = _uid()
        r = client.get("/scim/v2/ServiceProviderConfig", headers=_user_header(non_admin_id))
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Users CRUD
# ---------------------------------------------------------------------------


class TestSCIMListUsers:
    def test_list_empty(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get("/scim/v2/Users", headers=_admin_header(admin_id))
        assert r.status_code == 200
        data = r.json()
        assert "urn:ietf:params:scim:api:messages:2.0:ListResponse" in data["schemas"]
        # admin user is in this org — at least 1 result
        assert data["totalResults"] >= 1

    def test_filter_by_username(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(
            '/scim/v2/Users?filter=userName eq "admin"',
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["totalResults"] == 1
        assert r.json()["Resources"][0]["userName"] == "admin"

    def test_pagination(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get("/scim/v2/Users?startIndex=1&count=1", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["itemsPerPage"] == 1

    def test_compound_filter_and(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(
            '/scim/v2/Users?filter=userName eq "admin" and active eq true',
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["totalResults"] == 1
        assert r.json()["Resources"][0]["userName"] == "admin"
        assert r.json()["Resources"][0]["active"] is True

    def test_compound_filter_and_no_match(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(
            '/scim/v2/Users?filter=userName eq "admin" and active eq false',
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 200
        assert r.json()["totalResults"] == 0

    def test_invalid_filter_returns_400(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(
            "/scim/v2/Users?filter=not a valid filter",
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidFilter"

    def test_or_filter_returns_400(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(
            '/scim/v2/Users?filter=userName eq "admin" or active eq true',
            headers=_admin_header(admin_id),
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidFilter"


class TestSCIMCreateUser:
    def test_create_user_uses_configured_scim_base_url(self, scim_setup):
        client, _, admin_id = scim_setup

        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "baseurluser",
            "emails": [{"value": "baseurluser@example.com", "primary": True}],
        }

        headers = _admin_header(admin_id) | {"Host": "attacker.evil.com"}
        r = client.post("/scim/v2/Users", json=payload, headers=headers)

        assert r.status_code == 201
        data = r.json()
        assert data["meta"]["location"] == f"https://scim.example.com/scim/v2/Users/{data['id']}"
        assert r.headers["Location"] == data["meta"]["location"]
        assert "attacker.evil.com" not in data["meta"]["location"]

    def test_create_user_checks_uniqueness_per_org_first(self, scim_setup):
        """Same-org lookup is scoped by org_id; global lookup follows for cross-org guard."""
        client, org_id, admin_id = scim_setup

        calls = {"username": [], "email": []}

        async def _username_lookup(self, username, org_id=None):
            calls["username"].append(org_id)
            return None

        async def _email_lookup(self, email, org_id=None):
            calls["email"].append(org_id)
            return None

        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "orgscoped",
            "emails": [{"value": "orgscoped@example.com", "primary": True}],
        }

        with (
            patch.object(UserRepository, "get_by_username", _username_lookup),
            patch.object(UserRepository, "get_by_email", _email_lookup),
        ):
            r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))

        assert r.status_code == 201
        # First lookup is same-org scoped; second is global.
        assert calls["username"][0] == org_id
        assert calls["email"][0] == org_id
        assert calls["username"][1] is None
        assert calls["email"][1] is None

    def test_create_user(self, scim_setup):
        client, _, admin_id = scim_setup
        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "newuser",
            "emails": [{"value": "newuser@example.com", "primary": True}],
        }
        r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))
        assert r.status_code == 201
        data = r.json()
        assert data["userName"] == "newuser"
        assert data["id"] is not None

    def test_duplicate_username_returns_409(self, scim_setup):
        client, _, admin_id = scim_setup
        payload = {"userName": "admin", "emails": [{"value": "dup@example.com", "primary": True}]}
        r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))
        assert r.status_code == 409

    def test_duplicate_email_returns_409(self, scim_setup):
        client, _, admin_id = scim_setup
        payload = {
            "userName": "uniqueuser",
            "emails": [{"value": "admin@example.com", "primary": True}],
        }
        r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))
        assert r.status_code == 409

    def test_cross_org_username_returns_422_opaque_body(self, scim_setup):
        """Cross-org conflict must return 422 (not 409) with identical opaque body."""
        client, org_id, admin_id = scim_setup

        class _FakeUser:
            id = "other-user-id"

        # Same-org lookup returns None; global lookup returns a conflict.
        async def _username_lookup(self, username, org_id=None):
            return None if org_id else _FakeUser()

        async def _email_lookup(self, email, org_id=None):
            return None

        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "crossuser",
            "emails": [{"value": "new@example.com", "primary": True}],
        }

        with (
            patch.object(UserRepository, "get_by_username", _username_lookup),
            patch.object(UserRepository, "get_by_email", _email_lookup),
        ):
            r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))

        assert r.status_code == 422
        body = r.json()
        assert body["scimType"] == "uniqueness"
        assert "already exists" in body["detail"].lower()

    def test_cross_org_email_returns_422_opaque_body(self, scim_setup):
        """Cross-org email conflict must return 422 (not 409) with identical opaque body."""
        client, org_id, admin_id = scim_setup

        class _FakeUser:
            id = "other-user-id"

        async def _username_lookup(self, username, org_id=None):
            return None

        async def _email_lookup(self, email, org_id=None):
            return None if org_id else _FakeUser()

        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "newuser2",
            "emails": [{"value": "cross2@example.com", "primary": True}],
        }

        with (
            patch.object(UserRepository, "get_by_username", _username_lookup),
            patch.object(UserRepository, "get_by_email", _email_lookup),
        ):
            r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))

        assert r.status_code == 422
        body = r.json()
        assert body["scimType"] == "uniqueness"


class TestSCIMGetUser:
    def test_get_existing_user(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(f"/scim/v2/Users/{admin_id}", headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["id"] == admin_id

    def test_get_missing_user_returns_404(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.get(f"/scim/v2/Users/{_uid()}", headers=_admin_header(admin_id))
        assert r.status_code == 404


class TestSCIMReplaceUser:
    def test_replace_username(self, scim_setup):
        client, _, admin_id = scim_setup
        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "admin-renamed",
            "emails": [{"value": "admin@example.com", "primary": True}],
        }
        r = client.put(f"/scim/v2/Users/{admin_id}", json=payload, headers=_admin_header(admin_id))
        assert r.status_code == 200
        assert r.json()["userName"] == "admin-renamed"

    def test_replace_username_checks_uniqueness_globally(self, scim_setup):
        client, _, admin_id = scim_setup

        seen = {"username_org_id": None, "email_org_id": None}

        async def _username_lookup(self, username, org_id=None):
            seen["username_org_id"] = org_id
            return None

        async def _email_lookup(self, email, org_id=None):
            seen["email_org_id"] = org_id
            return None

        payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "admin-renamed-global",
            "emails": [{"value": "admin@example.com", "primary": True}],
        }

        with (
            patch.object(UserRepository, "get_by_username", _username_lookup),
            patch.object(UserRepository, "get_by_email", _email_lookup),
        ):
            r = client.put(
                f"/scim/v2/Users/{admin_id}", json=payload, headers=_admin_header(admin_id)
            )

        assert r.status_code == 200
        assert seen["username_org_id"] is None
        assert seen["email_org_id"] is None


class TestSCIMPatchUser:
    def test_patch_username(self, scim_setup):
        client, _, admin_id = scim_setup
        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "userName", "value": "admin-patched"}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 200
        assert r.json()["userName"] == "admin-patched"

    def test_patch_username_checks_uniqueness_globally(self, scim_setup):
        client, _, admin_id = scim_setup

        seen = {"username_org_id": None}

        async def _username_lookup(self, username, org_id=None):
            seen["username_org_id"] = org_id
            return None

        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "userName", "value": "admin-patched-global"}],
        }

        with patch.object(UserRepository, "get_by_username", _username_lookup):
            r = client.patch(
                f"/scim/v2/Users/{admin_id}", json=payload, headers=_admin_header(admin_id)
            )

        assert r.status_code == 200
        assert seen["username_org_id"] is None

    def test_patch_active_false_deactivates_user(self, scim_setup):
        client, _, admin_id = scim_setup
        create_payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "patch-active",
            "emails": [{"value": "patch-active@example.com", "primary": True}],
        }
        create_resp = client.post(
            "/scim/v2/Users", json=create_payload, headers=_admin_header(admin_id)
        )
        assert create_resp.status_code == 201
        user_id = create_resp.json()["id"]

        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "active", "value": False}],
        }
        r = client.patch(
            f"/scim/v2/Users/{user_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 200
        assert r.json()["active"] is False

    def test_patch_username_duplicate_returns_409(self, scim_setup):
        """BUG-121: PATCH userName to an existing username must return 409, not 500."""
        client, _, admin_id = scim_setup
        # Create a second user to collide with.
        create_payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "existing-user",
            "emails": [{"value": "existing@example.com", "primary": True}],
        }
        create_resp = client.post(
            "/scim/v2/Users", json=create_payload, headers=_admin_header(admin_id)
        )
        assert create_resp.status_code == 201

        # Try to patch admin's username to the existing one.
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "userName", "value": "existing-user"}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 409
        assert r.json()["scimType"] == "uniqueness"

    def test_patch_email_duplicate_returns_409(self, scim_setup):
        """BUG-121: PATCH email to an existing email must return 409, not 500."""
        client, _, admin_id = scim_setup
        # Create a second user to collide with.
        create_payload = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
            "userName": "existing-user2",
            "emails": [{"value": "existing2@example.com", "primary": True}],
        }
        create_resp = client.post(
            "/scim/v2/Users", json=create_payload, headers=_admin_header(admin_id)
        )
        assert create_resp.status_code == 201

        # Try to patch admin's email to the existing one.
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {"op": "replace", "path": "emails", "value": [{"value": "existing2@example.com"}]}
            ],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 409
        assert r.json()["scimType"] == "uniqueness"

    def test_patch_email_null_returns_400(self, scim_setup):
        """Malformed PATCH email values must return 400 invalidValue, not 500."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "emails", "value": [{"value": None}]}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidValue"

    def test_patch_invalid_path_returns_400(self, scim_setup):
        """RFC 7644: unsupported PATCH paths must return 400 invalidPath."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "displayName", "value": "foo"}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidPath"

    def test_patch_empty_path_with_invalid_key_returns_400(self, scim_setup):
        """Empty path with unsupported key in value dict must return 400 invalidPath."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "value": {"displayName": "foo"}}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidPath"

    def test_patch_remove_invalid_path_returns_400(self, scim_setup):
        """RFC 7644: remove on unsupported path must return 400 invalidPath."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "remove", "path": "displayName"}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidPath"

    def test_patch_username_non_string_value_returns_400(self, scim_setup):
        """PATCH userName with non-string value must return 400 invalidValue."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "userName", "value": 123}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidValue"

    def test_patch_active_invalid_value_returns_400(self, scim_setup):
        """PATCH active with unparseable value must return 400 invalidValue."""
        client, _, admin_id = scim_setup
        patch_payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "active", "value": "maybe"}],
        }
        r = client.patch(
            f"/scim/v2/Users/{admin_id}", json=patch_payload, headers=_admin_header(admin_id)
        )
        assert r.status_code == 400
        assert r.json()["scimType"] == "invalidValue"


class TestSCIMDeleteUser:
    def test_delete_user(self, scim_setup):
        client, _, admin_id = scim_setup
        # Create a user to delete.
        payload = {
            "userName": "todelete",
            "emails": [{"value": "todelete@example.com"}],
        }
        r = client.post("/scim/v2/Users", json=payload, headers=_admin_header(admin_id))
        assert r.status_code == 201
        new_id = r.json()["id"]

        r = client.delete(f"/scim/v2/Users/{new_id}", headers=_admin_header(admin_id))
        assert r.status_code == 204

        get_resp = client.get(f"/scim/v2/Users/{new_id}", headers=_admin_header(admin_id))
        assert get_resp.status_code == 200
        assert get_resp.json()["active"] is False

    def test_delete_missing_user_returns_404(self, scim_setup):
        client, _, admin_id = scim_setup
        r = client.delete(f"/scim/v2/Users/{_uid()}", headers=_admin_header(admin_id))
        assert r.status_code == 404
