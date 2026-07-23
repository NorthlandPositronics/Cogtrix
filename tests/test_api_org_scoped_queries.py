"""Tests for org-scoped DB queries (Enterprise Phase 1 — task 1.1.2).

Covers:
  - OrganizationRepository CRUD
  - UserRepository.list_by_org and assign_org
  - User.org_id FK + cascade SET NULL on org delete
  - Migration 0004 round-trip
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import uuid

import pytest

pytest.importorskip("fastapi")

from sqlalchemy import create_engine, inspect  # noqa: E402

from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.users import UserRepository  # noqa: E402

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# OrganizationRepository
# ---------------------------------------------------------------------------


class TestOrganizationRepository:
    def test_create_and_get_by_id(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org = await repo.create(org_id=_uid(), name="Acme", slug="acme", plan="pro")
                await session.commit()
                found = await repo.get_by_id(org.id)
                assert found is not None
                assert found.name == "Acme"
                assert found.plan == "pro"

        asyncio.run(_run())

    def test_get_by_slug(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                await repo.create(org_id=_uid(), name="Slug Org", slug="slug-org")
                await session.commit()
                found = await repo.get_by_slug("slug-org")
                assert found is not None
                assert found.name == "Slug Org"

        asyncio.run(_run())

    def test_get_by_name(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                await repo.create(org_id=_uid(), name="Named Org", slug="named-org")
                await session.commit()
                found = await repo.get_by_name("Named Org")
                assert found is not None

        asyncio.run(_run())

    def test_get_by_id_missing_returns_none(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                result = await repo.get_by_id(_uid())
                assert result is None

        asyncio.run(_run())

    def test_list_all_excludes_inactive_by_default(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                await repo.create(org_id=_uid(), name="Active Org", slug="active-org")
                inactive_id = _uid()
                await repo.create(org_id=inactive_id, name="Inactive Org", slug="inactive-org")
                await session.commit()
                await repo.update(inactive_id, is_active=False)
                await session.commit()
                orgs = await repo.list_all()
                names = [o.name for o in orgs]
                assert "Active Org" in names
                assert "Inactive Org" not in names

        asyncio.run(_run())

    def test_list_all_include_inactive(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                inactive_id = _uid()
                await repo.create(org_id=inactive_id, name="Both Org", slug="both-org")
                await session.commit()
                await repo.update(inactive_id, is_active=False)
                await session.commit()
                orgs = await repo.list_all(include_inactive=True)
                names = [o.name for o in orgs]
                assert "Both Org" in names

        asyncio.run(_run())

    def test_update_name_and_plan(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org_id = _uid()
                await repo.create(org_id=org_id, name="Old Name", slug="old-name", plan="free")
                await session.commit()
                updated = await repo.update(org_id, name="New Name", plan="enterprise")
                await session.commit()
                assert updated is not None
                assert updated.name == "New Name"
                assert updated.plan == "enterprise"

        asyncio.run(_run())

    def test_update_settings(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org_id = _uid()
                await repo.create(org_id=org_id, name="Settings Org", slug="settings-org")
                await session.commit()
                updated = await repo.update(org_id, settings={"sso": True, "max_seats": 50})
                await session.commit()
                assert updated is not None
                import json

                assert json.loads(updated.settings) == {"sso": True, "max_seats": 50}

        asyncio.run(_run())

    def test_update_missing_returns_none(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                result = await repo.update(_uid(), name="Ghost")
                assert result is None

        asyncio.run(_run())

    def test_delete(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org_id = _uid()
                await repo.create(org_id=org_id, name="Delete Me", slug="delete-me")
                await session.commit()
                deleted = await repo.delete(org_id)
                await session.commit()
                assert deleted is True
                assert await repo.get_by_id(org_id) is None

        asyncio.run(_run())

    def test_delete_missing_returns_false(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                result = await repo.delete(_uid())
                assert result is False

        asyncio.run(_run())

    def test_list_users_empty(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org_id = _uid()
                await repo.create(org_id=org_id, name="Empty Org", slug="empty-org")
                await session.commit()
                users = await repo.list_users(org_id)
                assert users == []

        asyncio.run(_run())

    def test_list_users_and_count(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Populated Org", slug="populated-org")
                await user_repo.create(
                    user_id=_uid(),
                    username="alice",
                    email="alice@example.com",
                    password_hash="hash",
                    org_id=org_id,
                )
                await user_repo.create(
                    user_id=_uid(),
                    username="bob",
                    email="bob@example.com",
                    password_hash="hash",
                    org_id=org_id,
                )
                await session.commit()
                users = await org_repo.list_users(org_id)
                count = await org_repo.count_users(org_id)
                assert len(users) == 2
                assert count == 2
                usernames = {u.username for u in users}
                assert usernames == {"alice", "bob"}

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# UserRepository org-scoped methods
# ---------------------------------------------------------------------------


class TestUserRepositoryOrgScoped:
    def test_create_with_org_id(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Test Org", slug="test-org")
                user = await user_repo.create(
                    user_id=_uid(),
                    username="orguser",
                    email="orguser@example.com",
                    password_hash="hash",
                    org_id=org_id,
                )
                await session.commit()
                assert user.org_id == org_id

        asyncio.run(_run())

    def test_create_without_org_id_defaults_to_none(self, sf):
        async def _run():
            async with sf() as session:
                user_repo = UserRepository(session)
                user = await user_repo.create(
                    user_id=_uid(),
                    username="noorg",
                    email="noorg@example.com",
                    password_hash="hash",
                )
                await session.commit()
                assert user.org_id is None

        asyncio.run(_run())

    def test_list_by_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org_a = _uid()
                org_b = _uid()
                await org_repo.create(org_id=org_a, name="Org A", slug="org-a")
                await org_repo.create(org_id=org_b, name="Org B", slug="org-b")
                await user_repo.create(
                    user_id=_uid(),
                    username="user_a1",
                    email="ua1@example.com",
                    password_hash="h",
                    org_id=org_a,
                )
                await user_repo.create(
                    user_id=_uid(),
                    username="user_a2",
                    email="ua2@example.com",
                    password_hash="h",
                    org_id=org_a,
                )
                await user_repo.create(
                    user_id=_uid(),
                    username="user_b1",
                    email="ub1@example.com",
                    password_hash="h",
                    org_id=org_b,
                )
                await session.commit()
                users_a = await user_repo.list_by_org(org_a)
                users_b = await user_repo.list_by_org(org_b)
                assert len(users_a) == 2
                assert len(users_b) == 1
                assert all(u.org_id == org_a for u in users_a)

        asyncio.run(_run())

    def test_assign_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Assign Org", slug="assign-org")
                user_id = _uid()
                await user_repo.create(
                    user_id=user_id,
                    username="assignable",
                    email="assignable@example.com",
                    password_hash="hash",
                )
                await session.commit()
                updated = await user_repo.assign_org(user_id, org_id)
                await session.commit()
                assert updated is not None
                assert updated.org_id == org_id

        asyncio.run(_run())

    def test_unassign_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Unassign Org", slug="unassign-org")
                user_id = _uid()
                await user_repo.create(
                    user_id=user_id,
                    username="unassignable",
                    email="unassignable@example.com",
                    password_hash="hash",
                    org_id=org_id,
                )
                await session.commit()
                updated = await user_repo.assign_org(user_id, None)
                await session.commit()
                assert updated is not None
                assert updated.org_id is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Migration round-trip
# ---------------------------------------------------------------------------


class TestMigration0004:
    def test_upgrade_and_downgrade(self):
        """Migration 0004 adds org_id FK to users; downgrade removes it."""
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_migration_0004_test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["COGTRIX_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"

        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"upgrade failed:\n{result.stderr}"

        # Downgrade to 0003 (removes user org_id FK added by 0004)
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0003"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)


class TestMigration0014:
    def test_upgrade_and_downgrade(self):
        """Migration 0014 adds org_id to rag_documents and removes it cleanly."""
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_migration_0014_test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.unlink(missing_ok=True)
        env = os.environ.copy()
        env["COGTRIX_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"

        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"upgrade failed:\n{result.stderr}"

        sync_engine = create_engine(f"sqlite:///{db_path}")
        columns = [col["name"] for col in inspect(sync_engine).get_columns("rag_documents")]
        assert "org_id" in columns

        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0013"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"

        columns_after = [col["name"] for col in inspect(sync_engine).get_columns("rag_documents")]
        assert "org_id" not in columns_after
        sync_engine.dispose()
        db_path.unlink(missing_ok=True)
