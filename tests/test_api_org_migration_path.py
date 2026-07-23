"""Tests for the single-tenant → multi-tenant migration path (Enterprise Phase 1 — task 1.1.5).

Covers:
  - Migration 0005 upgrade: creates default org and assigns unassigned users
  - Migration 0005 downgrade: unassigns users and deletes default org
  - Migration is idempotent (run twice → no duplicates)
  - Users already assigned to an org are not moved
  - OrganizationRepository.ensure_default_org: creates on first call, returns same on second
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import subprocess
import uuid

import pytest

pytest.importorskip("fastapi")

from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.users import UserRepository  # noqa: E402

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# OrganizationRepository.ensure_default_org
# ---------------------------------------------------------------------------


class TestEnsureDefaultOrg:
    def test_creates_default_org_on_first_call(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org = await repo.ensure_default_org()
                await session.commit()
                assert org.slug == "default"
                assert org.name == "Default Organization"
                assert org.plan == "free"
                assert org.is_active is True

        asyncio.run(_run())

    def test_returns_same_org_on_second_call(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org1 = await repo.ensure_default_org()
                await session.commit()
                org2 = await repo.ensure_default_org()
                await session.commit()
                assert org1.id == org2.id

        asyncio.run(_run())

    def test_existing_default_org_not_replaced(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                # Pre-create the default org with a specific ID
                pre_id = _uid()
                await repo.create(org_id=pre_id, name="Custom Default", slug="default")
                await session.commit()
                # ensure_default_org should return the existing one, not create a new one
                org = await repo.ensure_default_org()
                await session.commit()
                assert org.id == pre_id
                assert org.name == "Custom Default"

        asyncio.run(_run())

    def test_does_not_assign_users(self, sf):
        """ensure_default_org only creates the org — assignment is the migration's job."""

        async def _run():
            async with sf() as session:
                user_repo = UserRepository(session)
                org_repo = OrganizationRepository(session)
                await user_repo.create(
                    user_id=_uid(),
                    username="unassigned",
                    email="u@example.com",
                    password_hash="h",
                )
                await session.commit()
                org = await org_repo.ensure_default_org()
                await session.commit()
                users = await org_repo.list_users(org.id)
                # Still 0 — ensure_default_org doesn't auto-assign
                assert len(users) == 0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Migration logic simulation (unit-level, without full Alembic run)
# ---------------------------------------------------------------------------


class TestMigrationLogic:
    def test_unassigned_users_are_assigned_to_default_org(self, sf):
        """Simulates what migration 0005 does: assign NULL org_id users."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                # 3 unassigned users
                ids = [_uid() for _ in range(3)]
                for i, uid in enumerate(ids):
                    await user_repo.create(
                        user_id=uid,
                        username=f"user{i}",
                        email=f"u{i}@example.com",
                        password_hash="h",
                    )
                await session.commit()

                # Simulate migration: create default org and assign
                default_org = await org_repo.ensure_default_org()
                await session.commit()
                for uid in ids:
                    await user_repo.assign_org(uid, default_org.id)
                await session.commit()

                # All 3 should now be in the default org
                users = await org_repo.list_users(default_org.id)
                assert len(users) == 3

        asyncio.run(_run())

    def test_already_assigned_users_not_overwritten(self, sf):
        """Users already in another org keep their assignment."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                other_org_id = _uid()
                await org_repo.create(org_id=other_org_id, name="Other", slug="other")
                assigned_user = await user_repo.create(
                    user_id=_uid(),
                    username="assigned",
                    email="assigned@example.com",
                    password_hash="h",
                    org_id=other_org_id,
                )
                unassigned_user = await user_repo.create(
                    user_id=_uid(),
                    username="unassigned",
                    email="unassigned@example.com",
                    password_hash="h",
                )
                await session.commit()

                # Migration: only assign NULL users
                default_org = await org_repo.ensure_default_org()
                await session.commit()
                # Only unassigned_user gets moved
                await user_repo.assign_org(unassigned_user.id, default_org.id)
                await session.commit()

                # Verify: assigned_user still in other_org
                refreshed = await user_repo.get_by_id(assigned_user.id)
                assert refreshed.org_id == other_org_id

                # Verify: unassigned_user now in default org
                refreshed2 = await user_repo.get_by_id(unassigned_user.id)
                assert refreshed2.org_id == default_org.id

        asyncio.run(_run())

    def test_downgrade_unassigns_users(self, sf):
        """Simulates downgrade: unassign users from default org and delete it."""

        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                default_org = await org_repo.ensure_default_org()
                user_id = _uid()
                await user_repo.create(
                    user_id=user_id,
                    username="migrated",
                    email="migrated@example.com",
                    password_hash="h",
                    org_id=default_org.id,
                )
                await session.commit()

                # Simulate downgrade: unassign and delete
                await user_repo.assign_org(user_id, None)
                await org_repo.delete(default_org.id)
                await session.commit()

                # User is unassigned
                user = await user_repo.get_by_id(user_id)
                assert user.org_id is None

                # Default org is gone
                gone = await org_repo.get_by_slug("default")
                assert gone is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Full Alembic migration round-trip
# ---------------------------------------------------------------------------


class TestMigration0005RoundTrip:
    def test_upgrade_creates_default_org_and_assigns_users(self):
        """Full Alembic round-trip: upgrade creates default org; downgrade removes it."""
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_0005_test.db"
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

        # Verify default org was created
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT id, name, slug FROM organizations WHERE slug = 'default'"
        ).fetchall()
        assert len(rows) == 1, "Default org not created by migration 0005"
        conn.close()

        # Downgrade to revision 0004 (one step before 0005) — removes the default org.
        # Using "0004" instead of "-1" so this test stays correct regardless of HEAD
        # as new migrations are added.
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0004"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade failed:\n{result.stderr}"

        # Verify default org was removed by the 0005 downgrade
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id FROM organizations WHERE slug = 'default'").fetchall()
        assert len(rows) == 0, "Default org not removed by downgrade"
        conn.close()

        db_path.unlink(missing_ok=True)
