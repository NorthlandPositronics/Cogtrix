"""Regression tests for #321 — admin user CRUD org-scoping.

Verifies that an admin in Org A cannot enumerate, modify, or delete users
belonging to Org B.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("fastapi")

_TEST_JWT_SECRET = "testsecret_mustbe32chars_minimum00"
os.environ.setdefault("COGTRIX_JWT_SECRET", _TEST_JWT_SECRET)
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from cogtrix_core.api.db.engine import Base  # noqa: E402
from cogtrix_core.api.db.models import Organization, User  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402


@pytest.fixture()
def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def _setup():
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())
    yield eng
    asyncio.run(eng.dispose())


@pytest.fixture()
def sf(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


def _uid() -> str:
    return str(uuid.uuid4())


class TestCrossOrgUserCrud:
    """End-to-end org-scoping regression for #321."""

    def _make_app(self, org_a_id: str, org_b_id: str, admin_a_id: str, admin_b_id: str):
        from fastapi import FastAPI

        from cogtrix_core.api.auth import TokenData, get_current_user
        from cogtrix_core.api.db.engine import get_db
        from cogtrix_core.api.org_context import OrgContext, get_org_context
        from cogtrix_core.api.routes import users as users_module

        app = FastAPI()

        async def _override_db():
            async with self._sf() as session:
                yield session

        app.dependency_overrides[get_db] = _override_db

        # Build override for admin A ( Org A )
        def _make_current_user(user_id: str, role: str = "admin"):
            return TokenData(user_id=user_id, role=role, raw_claims={"sub": user_id, "role": role})

        def _make_org_context(user_id: str, org_id: str):
            return OrgContext(user_id=user_id, role="admin", org_id=org_id, org=None)

        # Store state for test to switch between admin A and admin B
        self._current_user_id = admin_a_id
        self._current_org_id = org_a_id

        def _override_current_user():
            return _make_current_user(self._current_user_id)

        def _override_org_context():
            return _make_org_context(self._current_user_id, self._current_org_id)

        app.dependency_overrides[get_current_user] = _override_current_user
        app.dependency_overrides[get_org_context] = _override_org_context
        app.include_router(users_module.router)
        return app

    def _switch_user(self, user_id: str, org_id: str):
        self._current_user_id = user_id
        self._current_org_id = org_id

    def test_list_users_scoped_to_org(self, sf):
        self._sf = sf

        async def _run():
            async with sf() as session:
                # Create two orgs
                org_a = Organization(id=_uid(), name="Org A", slug="org-a")
                org_b = Organization(id=_uid(), name="Org B", slug="org-b")
                session.add_all([org_a, org_b])
                await session.commit()

                # Create admin A in org A
                admin_a_id = _uid()
                user_a = User(
                    id=admin_a_id,
                    username="admin_a",
                    email="admin_a@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_a.id,
                )
                # Create user in org A
                user_a2 = User(
                    id=_uid(),
                    username="user_a2",
                    email="user_a2@example.com",
                    password_hash="x",
                    role="user",
                    org_id=org_a.id,
                )
                # Create admin B in org B
                admin_b_id = _uid()
                user_b = User(
                    id=admin_b_id,
                    username="admin_b",
                    email="admin_b@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_b.id,
                )
                # Create user in org B
                user_b2 = User(
                    id=_uid(),
                    username="user_b2",
                    email="user_b2@example.com",
                    password_hash="x",
                    role="user",
                    org_id=org_b.id,
                )
                session.add_all([user_a, user_a2, user_b, user_b2])
                await session.commit()

                # Build app with admin A context
                app = self._make_app(org_a.id, org_b.id, admin_a_id, admin_b_id)
                client = TestClient(app, raise_server_exceptions=False)

                # Admin A lists users — should see only org A users
                resp = client.get("/users")
                assert resp.status_code == 200
                data = resp.json()["data"]
                usernames = {u["username"] for u in data}
                assert "admin_a" in usernames
                assert "user_a2" in usernames
                assert "admin_b" not in usernames
                assert "user_b2" not in usernames

        asyncio.run(_run())

    def test_update_user_cross_org_returns_404(self, sf):
        self._sf = sf

        async def _run():
            async with sf() as session:
                org_a = Organization(id=_uid(), name="Org A", slug="org-a")
                org_b = Organization(id=_uid(), name="Org B", slug="org-b")
                session.add_all([org_a, org_b])
                await session.commit()

                admin_a_id = _uid()
                user_a = User(
                    id=admin_a_id,
                    username="admin_a",
                    email="admin_a@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_a.id,
                )
                user_b_id = _uid()
                user_b = User(
                    id=user_b_id,
                    username="user_b",
                    email="user_b@example.com",
                    password_hash="x",
                    role="user",
                    org_id=org_b.id,
                )
                session.add_all([user_a, user_b])
                await session.commit()

                app = self._make_app(org_a.id, org_b.id, admin_a_id, _uid())
                client = TestClient(app, raise_server_exceptions=False)

                # Admin A tries to update user B (cross-org) → 404
                resp = client.patch(
                    f"/users/{user_b_id}",
                    json={"role": "admin"},
                )
                assert resp.status_code == 404
                assert resp.json()["detail"]["code"] == "NOT_FOUND"

        asyncio.run(_run())

    def test_delete_user_cross_org_returns_404(self, sf):
        self._sf = sf

        async def _run():
            async with sf() as session:
                org_a = Organization(id=_uid(), name="Org A", slug="org-a")
                org_b = Organization(id=_uid(), name="Org B", slug="org-b")
                session.add_all([org_a, org_b])
                await session.commit()

                admin_a_id = _uid()
                user_a = User(
                    id=admin_a_id,
                    username="admin_a",
                    email="admin_a@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_a.id,
                )
                user_b_id = _uid()
                user_b = User(
                    id=user_b_id,
                    username="user_b",
                    email="user_b@example.com",
                    password_hash="x",
                    role="user",
                    org_id=org_b.id,
                )
                session.add_all([user_a, user_b])
                await session.commit()

                app = self._make_app(org_a.id, org_b.id, admin_a_id, _uid())
                client = TestClient(app, raise_server_exceptions=False)

                # Admin A tries to delete user B (cross-org) → 404
                resp = client.delete(f"/users/{user_b_id}")
                assert resp.status_code == 404
                assert resp.json()["detail"]["code"] == "NOT_FOUND"

        asyncio.run(_run())

    def test_create_user_sets_org_id(self, sf):
        self._sf = sf

        async def _run():
            async with sf() as session:
                org_a = Organization(id=_uid(), name="Org A", slug="org-a")
                session.add(org_a)
                await session.commit()

                admin_a_id = _uid()
                user_a = User(
                    id=admin_a_id,
                    username="admin_a",
                    email="admin_a@example.com",
                    password_hash="x",
                    role="admin",
                    org_id=org_a.id,
                )
                session.add(user_a)
                await session.commit()

                app = self._make_app(org_a.id, _uid(), admin_a_id, _uid())
                client = TestClient(app, raise_server_exceptions=False)

                resp = client.post(
                    "/users",
                    json={
                        "username": "new_user",
                        "email": "new@example.com",
                        "password": "TestPass1!",
                        "role": "user",
                    },
                )
                assert resp.status_code == 201
                new_user_id = resp.json()["data"]["id"]

                # Verify org_id was set
                repo = UserRepository(session)
                new_user = await repo.get_by_id(new_user_id)
                assert new_user is not None
                assert new_user.org_id == org_a.id

        asyncio.run(_run())
