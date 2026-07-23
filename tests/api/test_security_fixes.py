"""Tests for API security fixes (plan enforcement, org scoping, membership guard,
Stripe unsigned flag hardening).

Coverage:
  - PlanLimitSnapshot.within_limit and capacity boolean properties
  - get_plan_limit_snapshot: resolves limits from plan row and falls back to unlimited
  - require_workspace_capacity: raises HTTP 402 when workspace quota reached
  - require_user_capacity: raises HTTP 402 when user seat quota reached
  - create_workspace route: plan enforcement dependency is wired (capacity param present)
  - scim_create_user route: plan enforcement dependency is wired (capacity param present)
  - scim_create_user: Location header uses safe optional access (no AttributeError when meta is None)
  - SCIMUser with meta=None: user_to_scim always sets meta; safe access guard is correct
  - Stripe webhook: 503 returned when STRIPE_WEBHOOK_SECRET absent and STRIPE_ALLOW_UNSIGNED unset
"""

from __future__ import annotations

import asyncio
import inspect
import uuid

import pytest

pytest.importorskip("fastapi")

from cogtrix_core.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from cogtrix_core.api.db.repositories.plans import PlanRepository  # noqa: E402
from cogtrix_core.api.plan_enforcement import (  # noqa: E402
    PlanLimitSnapshot,
    get_plan_limit_snapshot,
    require_user_capacity,
    require_workspace_capacity,
)


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PlanLimitSnapshot unit tests
# ---------------------------------------------------------------------------


class TestPlanLimitSnapshot:
    def _snap(self, **overrides) -> PlanLimitSnapshot:
        defaults = dict(
            plan_slug="free",
            max_users=5,
            max_workspaces=3,
            max_api_calls_per_month=1000,
            max_storage_gb=10,
            current_users=0,
            current_workspaces=0,
            current_api_calls=0,
        )
        defaults.update(overrides)
        return PlanLimitSnapshot(**defaults)

    def test_within_limit_zero_means_unlimited(self):
        snap = self._snap(max_users=0, current_users=9999)
        assert snap.within_limit(0, 9999) is True

    def test_within_limit_returns_true_when_below(self):
        assert (
            PlanLimitSnapshot(
                plan_slug="pro",
                max_users=10,
                max_workspaces=0,
                max_api_calls_per_month=0,
                max_storage_gb=0,
                current_users=9,
                current_workspaces=0,
                current_api_calls=0,
            ).can_add_user
            is True
        )

    def test_within_limit_returns_false_when_at_cap(self):
        snap = self._snap(max_users=5, current_users=5)
        assert snap.can_add_user is False

    def test_can_add_workspace_false_at_cap(self):
        snap = self._snap(max_workspaces=2, current_workspaces=2)
        assert snap.can_add_workspace is False

    def test_can_add_workspace_true_below_cap(self):
        snap = self._snap(max_workspaces=2, current_workspaces=1)
        assert snap.can_add_workspace is True

    def test_can_make_api_call_unlimited_when_zero(self):
        snap = self._snap(max_api_calls_per_month=0, current_api_calls=999_999)
        assert snap.can_make_api_call is True

    def test_can_make_api_call_false_when_exhausted(self):
        snap = self._snap(max_api_calls_per_month=100, current_api_calls=100)
        assert snap.can_make_api_call is False


# ---------------------------------------------------------------------------
# get_plan_limit_snapshot integration tests
# ---------------------------------------------------------------------------


class TestGetPlanLimitSnapshot:
    def test_returns_unlimited_limits_when_no_plan_assigned(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="No Plan Org", slug="no-plan-org")
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                assert snap.max_users == 0
                assert snap.max_workspaces == 0
                assert snap.plan_slug == "free"

        asyncio.run(_run())

    def test_loads_limits_from_plan_row(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)

                org_id = _uid()
                plan_id = _uid()
                limits = {"max_users": 10, "max_workspaces": 5, "max_api_calls_per_month": 500}
                await plan_repo.create(
                    plan_id=plan_id,
                    name="Starter",
                    slug="starter",
                    limits=limits,
                )
                org = await org_repo.create(org_id=org_id, name="Starter Org", slug="starter-org")
                org.plan_id = plan_id
                org.plan = "starter"
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                assert snap.max_users == 10
                assert snap.max_workspaces == 5
                assert snap.max_api_calls_per_month == 500
                assert snap.plan_slug == "starter"

        asyncio.run(_run())

    def test_current_users_count_is_accurate(self, sf):
        async def _run():
            async with sf() as session:
                from cogtrix_core.api.auth import hash_password
                from cogtrix_core.api.db.repositories.users import UserRepository

                org_repo = OrganizationRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                await org_repo.create(org_id=org_id, name="User Count Org", slug="user-count-org")
                for i in range(3):
                    await user_repo.create(
                        user_id=_uid(),
                        username=f"u{i}_{uuid.uuid4().hex[:4]}",
                        email=f"u{i}_{uuid.uuid4().hex[:4]}@example.com",
                        password_hash=hash_password("pw"),
                        role="user",
                        org_id=org_id,
                    )
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                assert snap.current_users == 3

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# require_workspace_capacity / require_user_capacity dependency tests
# ---------------------------------------------------------------------------


class TestRequireWorkspaceCapacity:
    def test_raises_402_when_workspace_quota_reached(self, sf):
        from fastapi import HTTPException

        async def _run():
            async with sf() as session:
                from cogtrix_core.api.db.repositories.workspaces import WorkspaceRepository

                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)
                ws_repo = WorkspaceRepository(session)

                org_id = _uid()
                plan_id = _uid()
                await plan_repo.create(
                    plan_id=plan_id,
                    name="Tiny",
                    slug="tiny",
                    limits={"max_workspaces": 1},
                )
                org = await org_repo.create(org_id=org_id, name="Tiny Org", slug="tiny-org")
                org.plan_id = plan_id
                org.plan = "tiny"
                # Create 1 workspace to fill the quota.
                await ws_repo.create(
                    workspace_id=_uid(),
                    org_id=org_id,
                    name="ws-one",
                )
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                assert snap.can_add_workspace is False

                with pytest.raises(HTTPException) as exc_info:
                    await require_workspace_capacity(snap)
                assert exc_info.value.status_code == 402
                assert exc_info.value.detail["code"] == "QUOTA_EXCEEDED"
                assert exc_info.value.detail["resource"] == "workspaces"

        asyncio.run(_run())

    def test_passes_when_within_quota(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)

                org_id = _uid()
                plan_id = _uid()
                await plan_repo.create(
                    plan_id=plan_id,
                    name="Medium",
                    slug="medium",
                    limits={"max_workspaces": 10},
                )
                org = await org_repo.create(org_id=org_id, name="Medium Org", slug="medium-org")
                org.plan_id = plan_id
                org.plan = "medium"
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                result = await require_workspace_capacity(snap)
                assert result is snap

        asyncio.run(_run())


class TestRequireUserCapacity:
    def test_raises_402_when_user_quota_reached(self, sf):
        from fastapi import HTTPException

        async def _run():
            async with sf() as session:
                from cogtrix_core.api.auth import hash_password
                from cogtrix_core.api.db.repositories.users import UserRepository

                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)
                user_repo = UserRepository(session)

                org_id = _uid()
                plan_id = _uid()
                await plan_repo.create(
                    plan_id=plan_id,
                    name="Solo",
                    slug="solo",
                    limits={"max_users": 1},
                )
                org = await org_repo.create(org_id=org_id, name="Solo Org", slug="solo-org")
                org.plan_id = plan_id
                org.plan = "solo"
                await user_repo.create(
                    user_id=_uid(),
                    username=f"only_{uuid.uuid4().hex[:4]}",
                    email=f"only_{uuid.uuid4().hex[:4]}@example.com",
                    password_hash=hash_password("pw"),
                    role="user",
                    org_id=org_id,
                )
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                assert snap.can_add_user is False

                with pytest.raises(HTTPException) as exc_info:
                    await require_user_capacity(snap)
                assert exc_info.value.status_code == 402
                assert exc_info.value.detail["code"] == "QUOTA_EXCEEDED"
                assert exc_info.value.detail["resource"] == "users"

        asyncio.run(_run())

    def test_passes_for_unlimited_plan(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Unl Org", slug="unl-org")
                await session.commit()

                snap = await get_plan_limit_snapshot(org_id, session)
                # max_users == 0 means unlimited
                assert snap.max_users == 0
                result = await require_user_capacity(snap)
                assert result is snap

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Route dependency wiring — verify capacity params are present
# ---------------------------------------------------------------------------


class TestCapacityDependencyWiring:
    def test_create_workspace_has_capacity_param(self):
        from cogtrix_core.api.routes.workspaces import create_workspace

        sig = inspect.signature(create_workspace)
        assert (
            "capacity" in sig.parameters
        ), "create_workspace must declare a 'capacity' parameter for plan enforcement"
        param = sig.parameters["capacity"]
        assert param.annotation is PlanLimitSnapshot or str(param.annotation) in (
            "PlanLimitSnapshot",
            "cogtrix_core.api.plan_enforcement.PlanLimitSnapshot",
        )

    def test_scim_create_user_has_capacity_param(self):
        from cogtrix_core.api.routes.scim import scim_create_user

        sig = inspect.signature(scim_create_user)
        assert (
            "capacity" in sig.parameters
        ), "scim_create_user must declare a 'capacity' parameter for plan enforcement"
        param = sig.parameters["capacity"]
        assert param.annotation is PlanLimitSnapshot or str(param.annotation) in (
            "PlanLimitSnapshot",
            "cogtrix_core.api.plan_enforcement.PlanLimitSnapshot",
        )


# ---------------------------------------------------------------------------
# SCIM Location header — safe optional meta access
# ---------------------------------------------------------------------------


class TestSCIMLocationHeaderSafety:
    def test_user_to_scim_always_sets_meta(self):
        """user_to_scim always populates meta, confirming normal path is safe."""
        from cogtrix_core.api.db.models import User
        from cogtrix_core.api.scim.mapping import user_to_scim

        user = User(
            id=_uid(),
            username="alice",
            email="alice@example.com",
            password_hash="x",
            role="user",
        )
        scim_user = user_to_scim(user, "https://api.example.com")
        assert scim_user.meta is not None
        assert scim_user.meta.location is not None
        assert "/scim/v2/Users/" in scim_user.meta.location

    def test_location_header_expression_safe_when_meta_is_none(self):
        """Verify the guard expression in scim_create_user does not raise when meta is None."""
        from cogtrix_core.api.scim.schemas import SCIMUser

        # Construct a SCIMUser with meta=None to simulate the guarded path.
        scim_user = SCIMUser(userName="bob", meta=None)
        # Replicate the exact expression used in scim_create_user.
        location_value = (scim_user.meta.location if scim_user.meta is not None else None) or ""
        assert location_value == ""

    def test_location_header_expression_uses_meta_when_present(self):
        from cogtrix_core.api.scim.schemas import SCIMMeta, SCIMUser

        scim_user = SCIMUser(
            userName="carol",
            meta=SCIMMeta(
                resourceType="User", location="https://api.example.com/scim/v2/Users/123"
            ),
        )
        location_value = (scim_user.meta.location if scim_user.meta is not None else None) or ""
        assert location_value == "https://api.example.com/scim/v2/Users/123"


# ---------------------------------------------------------------------------
# Stripe unsigned flag hardening (regression for issue #119)
# ---------------------------------------------------------------------------


class TestStripeUnsignedFlagHardening:
    """Webhook endpoint must return 503 when STRIPE_WEBHOOK_SECRET is absent
    and STRIPE_ALLOW_UNSIGNED is not set to '1'."""

    def _make_webhook_app(self, sf):
        from fastapi import FastAPI

        from cogtrix_core.api.routes import billing as billing_module

        app = FastAPI()

        async def _override_db():
            async with sf() as session:
                yield session

        from cogtrix_core.api.db.engine import get_db

        app.dependency_overrides[get_db] = _override_db
        app.include_router(billing_module.router)
        return app

    def test_no_secret_no_allow_flag_returns_503(self, sf, monkeypatch):
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("STRIPE_ALLOW_UNSIGNED", raising=False)
        import json as _json

        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = _json.dumps({"type": "ping", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 503
        assert "STRIPE_WEBHOOK_SECRET" in resp.json()["error"]

    def test_allow_unsigned_flag_permits_unsigned_request(self, sf, monkeypatch):
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        import json as _json

        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = _json.dumps({"type": "unknown_event", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["received"] is True
        assert body["handled"] is False

    def test_allow_unsigned_flag_value_zero_is_rejected(self, sf, monkeypatch):
        """STRIPE_ALLOW_UNSIGNED='0' must not bypass the secret check."""
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "0")
        import json as _json

        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = _json.dumps({"type": "ping", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 503

    def test_allow_unsigned_in_production_returns_503(self, sf, monkeypatch):
        """STRIPE_ALLOW_UNSIGNED=1 must be rejected in production (COGTRIX_ENV=production)."""
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        monkeypatch.setenv("COGTRIX_ENV", "production")
        import json as _json

        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = _json.dumps({"type": "ping", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 503

    def test_allow_unsigned_in_staging_is_permitted(self, sf, monkeypatch):
        """STRIPE_ALLOW_UNSIGNED=1 is permitted in non-production environments."""
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        monkeypatch.setenv("COGTRIX_ENV", "staging")
        import json as _json

        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = _json.dumps({"type": "unknown_event", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 200


class TestStripeUnsignedStartupAssertion:
    """Startup must reject STRIPE_ALLOW_UNSIGNED=1 in production."""

    def test_startup_raises_in_production_with_allow_unsigned(self, monkeypatch):
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        monkeypatch.setenv("COGTRIX_ENV", "production")

        from unittest.mock import MagicMock

        from cogtrix_core.api.app import lifespan

        mock_app = MagicMock()

        async def _enter():
            async with lifespan(mock_app):  # type: ignore[arg-type]
                pass

        with pytest.raises(RuntimeError, match="STRIPE_ALLOW_UNSIGNED"):
            import asyncio

            asyncio.run(_enter())

    def test_startup_succeeds_in_staging_with_allow_unsigned(self, monkeypatch):
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        monkeypatch.setenv("COGTRIX_ENV", "staging")

        from unittest.mock import MagicMock

        from cogtrix_core.api.app import lifespan

        mock_app = MagicMock()

        async def _enter():
            async with lifespan(mock_app):  # type: ignore[arg-type]
                pass

        import asyncio

        # Should NOT raise — staging is allowed.
        asyncio.run(_enter())
