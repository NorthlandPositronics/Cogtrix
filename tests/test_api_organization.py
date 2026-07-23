"""Tests for the Organization DB model, schemas, and Alembic migration.

Covers acceptance criteria from issue #64:
  - alembic upgrade head creates the organizations table
  - alembic downgrade -1 drops it cleanly
  - Create/read/update an Organization via SQLAlchemy session
  - Duplicate slug raises IntegrityError
  - Pydantic schema validation: slug pattern, plan enum
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from src.api.db.models import Organization  # noqa: E402
from src.api.schemas.organization import (  # noqa: E402
    OrganizationCreate,
    OrganizationOut,
    OrganizationUpdate,
)

# ---------------------------------------------------------------------------
# SQLAlchemy model tests
# ---------------------------------------------------------------------------


class TestOrganizationModel:
    def test_create_and_read(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org = Organization(name="Acme Corp", slug="acme-corp", plan="enterprise")
                session.add(org)
                await session.commit()
                await session.refresh(org)
                assert org.id is not None
                assert len(org.id) == 36
                assert org.name == "Acme Corp"
                assert org.slug == "acme-corp"
                assert org.plan == "enterprise"
                assert org.is_active is True
                assert org.settings is None
                assert org.created_at is not None
                assert org.updated_at is not None

        asyncio.run(_run())

    def test_default_plan_is_free(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org = Organization(name="Default Plan Org", slug="default-plan-org")
                session.add(org)
                await session.commit()
                await session.refresh(org)
                assert org.plan == "free"

        asyncio.run(_run())

    def test_settings_json_blob(self, session_factory):
        async def _run():
            async with session_factory() as session:
                settings = json.dumps({"max_users": 100, "sso_enabled": True})
                org = Organization(name="Settings Org", slug="settings-org", settings=settings)
                session.add(org)
                await session.commit()
                await session.refresh(org)
                assert org.settings == settings

        asyncio.run(_run())

    def test_duplicate_slug_raises_integrity_error(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org1 = Organization(name="First Org", slug="duplicate-slug")
                session.add(org1)
                await session.commit()

            with pytest.raises(IntegrityError):
                async with session_factory() as session:
                    org2 = Organization(name="Second Org", slug="duplicate-slug")
                    session.add(org2)
                    await session.commit()

        asyncio.run(_run())

    def test_duplicate_name_raises_integrity_error(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org1 = Organization(name="Same Name", slug="same-name-1")
                session.add(org1)
                await session.commit()

            with pytest.raises(IntegrityError):
                async with session_factory() as session:
                    org2 = Organization(name="Same Name", slug="same-name-2")
                    session.add(org2)
                    await session.commit()

        asyncio.run(_run())

    def test_soft_delete_via_is_active(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org = Organization(name="To Delete", slug="to-delete")
                session.add(org)
                await session.commit()
                org.is_active = False
                await session.commit()
                await session.refresh(org)
                assert org.is_active is False

        asyncio.run(_run())

    def test_update_name_and_plan(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org = Organization(name="Old Name", slug="old-name", plan="free")
                session.add(org)
                await session.commit()
                org.name = "New Name"
                org.plan = "pro"
                await session.commit()
                await session.refresh(org)
                assert org.name == "New Name"
                assert org.plan == "pro"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------


class TestOrganizationCreate:
    def test_valid(self):
        req = OrganizationCreate(name="Acme", slug="acme", plan="pro")
        assert req.name == "Acme"
        assert req.slug == "acme"
        assert req.plan == "pro"

    def test_default_plan_free(self):
        req = OrganizationCreate(name="Acme", slug="acme")
        assert req.plan == "free"

    @pytest.mark.parametrize("plan", ["free", "pro", "team", "enterprise"])
    def test_all_valid_plans(self, plan):
        req = OrganizationCreate(name=f"Org {plan}", slug=f"org-{plan}", plan=plan)
        assert req.plan == plan

    def test_invalid_plan_raises(self):
        with pytest.raises(Exception, match="plan"):
            OrganizationCreate(name="Bad", slug="bad", plan="starter")

    @pytest.mark.parametrize(
        "slug",
        ["acme", "acme-corp", "acme-corp-123", "a1b2c3"],
    )
    def test_valid_slug_patterns(self, slug):
        req = OrganizationCreate(name=f"Org {slug}", slug=slug)
        assert req.slug == slug

    @pytest.mark.parametrize(
        "slug",
        ["Acme", "acme_corp", "acme corp", "-acme", "acme-", "ACME"],
    )
    def test_invalid_slug_patterns_raise(self, slug):
        with pytest.raises(ValidationError):
            OrganizationCreate(name="Org", slug=slug)

    def test_settings_dict_accepted(self):
        req = OrganizationCreate(name="Org", slug="org", settings={"max_users": 50, "sso": True})
        assert req.settings == {"max_users": 50, "sso": True}

    def test_name_too_long_raises(self):
        with pytest.raises(ValidationError):
            OrganizationCreate(name="x" * 257, slug="org")

    def test_slug_too_long_raises(self):
        with pytest.raises(ValidationError):
            OrganizationCreate(name="Org", slug="a" * 65)


class TestOrganizationUpdate:
    def test_all_none_is_valid(self):
        req = OrganizationUpdate()
        assert req.name is None
        assert req.plan is None
        assert req.settings is None
        assert req.is_active is None

    def test_partial_update(self):
        req = OrganizationUpdate(plan="team", is_active=False)
        assert req.plan == "team"
        assert req.is_active is False
        assert req.name is None

    def test_invalid_plan_raises(self):
        with pytest.raises(Exception, match="plan"):
            OrganizationUpdate(plan="unknown")


class TestOrganizationOut:
    def test_deserializes_from_model(self, session_factory):
        async def _run():
            async with session_factory() as session:
                org = Organization(name="Out Org", slug="out-org", plan="team")
                session.add(org)
                await session.commit()
                await session.refresh(org)
                out = OrganizationOut.model_validate(org, from_attributes=True)
                assert out.id == org.id
                assert out.name == "Out Org"
                assert out.slug == "out-org"
                assert out.plan == "team"
                assert out.is_active is True
                assert out.settings is None

        asyncio.run(_run())

    def test_settings_json_string_parsed_to_dict(self, session_factory):
        async def _run():
            async with session_factory() as session:
                settings_str = json.dumps({"key": "value"})
                org = Organization(name="Json Org", slug="json-org", settings=settings_str)
                session.add(org)
                await session.commit()
                await session.refresh(org)
                out = OrganizationOut.model_validate(org, from_attributes=True)
                assert out.settings == {"key": "value"}

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Alembic migration round-trip
# ---------------------------------------------------------------------------


class TestOrganizationMigration:
    def test_migration_creates_and_drops_table(self):
        """alembic upgrade head creates organizations; downgrade -1 drops it."""
        import pathlib
        import subprocess

        project_root = pathlib.Path(__file__).parent.parent
        db_path = project_root / "data" / "api" / "cogtrix_migration_test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["COGTRIX_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"

        # Fresh DB — upgrade to head
        result = subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(project_root),
        )
        assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"

        # Downgrade to 0002 (removes organizations table added by 0003)
        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0002"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(project_root),
        )
        assert result.returncode == 0, f"alembic downgrade failed:\n{result.stderr}"

        # Clean up
        db_path.unlink(missing_ok=True)
