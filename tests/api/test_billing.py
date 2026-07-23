"""Tests for Stripe billing integration (Enterprise Phase 1 — task 1.4.3).

Coverage:
  - stripe_client.get_stripe_client raises RuntimeError when key is absent
  - Organization model has stripe_* columns
  - Webhook: checkout.session.completed updates org fields and assigns plan
  - Webhook: customer.subscription.updated syncs status
  - Webhook: customer.subscription.deleted reverts to free plan and marks canceled
  - Webhook: unknown event returns 200 with handled=False
  - Webhook: missing signature header returns 200 with received=False
  - GET /billing/subscription returns correct envelope
  - POST /billing/portal raises 400 when no stripe_customer_id
  - Alembic migration 0011 applies and rolls back cleanly
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import uuid

import pytest

pytest.importorskip("fastapi")
# Ensure stripe is importable; skip entire module if not installed.
stripe_mod = pytest.importorskip("stripe", reason="stripe package not installed")

from src.api.db.models import Organization, ProcessedStripeEvent  # noqa: E402
from src.api.db.repositories.organization import OrganizationRepository  # noqa: E402
from src.api.db.repositories.plans import PlanRepository  # noqa: E402
from src.api.routes.billing import (  # noqa: E402
    _dispatch_event,
    _handle_checkout_completed,
    _handle_subscription_deleted,
    _handle_subscription_updated,
)

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# stripe_client unit tests
# ---------------------------------------------------------------------------


class TestGetStripeClient:
    def test_raises_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        from src.api.stripe_client import get_stripe_client

        with pytest.raises(RuntimeError, match="STRIPE_SECRET_KEY not configured"):
            get_stripe_client()

    def test_returns_stripe_module_with_key_set(self, monkeypatch):
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        from src.api.stripe_client import get_stripe_client

        result = get_stripe_client()
        import stripe as _stripe

        assert result is _stripe
        assert _stripe.api_key == "sk_test_dummy"


# ---------------------------------------------------------------------------
# Organization model — stripe columns
# ---------------------------------------------------------------------------


class TestOrganizationStripeColumns:
    def test_stripe_fields_default_to_none(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org = await repo.create(org_id=_uid(), name="Acme Stripe", slug="acme-stripe")
                await session.commit()
                assert org.stripe_customer_id is None
                assert org.stripe_subscription_id is None
                assert org.stripe_subscription_status is None

        asyncio.run(_run())

    def test_stripe_fields_can_be_set(self, sf):
        async def _run():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org = await repo.create(org_id=_uid(), name="Paid Org", slug="paid-org")
                org.stripe_customer_id = "cus_test123"
                org.stripe_subscription_id = "sub_test456"
                org.stripe_subscription_status = "active"
                await session.commit()
                await session.refresh(org)
                assert org.stripe_customer_id == "cus_test123"
                assert org.stripe_subscription_id == "sub_test456"
                assert org.stripe_subscription_status == "active"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Webhook handler unit tests (call handlers directly — no HTTP layer needed)
# ---------------------------------------------------------------------------


class TestCheckoutSessionCompleted:
    def test_sets_stripe_ids_and_active_status(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Checkout Org", slug="checkout-org")
                await session.commit()

                session_obj = {
                    "metadata": {"org_id": org_id, "plan_slug": "pro"},
                    "customer": "cus_abc",
                    "subscription": "sub_abc",
                }
                await _handle_checkout_completed(session_obj, session)

                org = await org_repo.get_by_id(org_id)
                assert org is not None
                assert org.stripe_customer_id == "cus_abc"
                assert org.stripe_subscription_id == "sub_abc"
                assert org.stripe_subscription_status == "active"

        asyncio.run(_run())

    def test_assigns_plan_when_slug_matches(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)

                org_id = _uid()
                plan_id = _uid()
                await org_repo.create(org_id=org_id, name="Plan Org", slug="plan-org")
                await plan_repo.create(
                    plan_id=plan_id,
                    name="Pro",
                    slug="pro",
                    limits={},
                )
                await session.commit()

                session_obj = {
                    "metadata": {"org_id": org_id, "plan_slug": "pro"},
                    "customer": "cus_xyz",
                    "subscription": "sub_xyz",
                }
                await _handle_checkout_completed(session_obj, session)

                org = await org_repo.get_by_id(org_id)
                assert org is not None
                assert org.plan == "pro"
                assert org.plan_id == plan_id

        asyncio.run(_run())

    def test_missing_org_id_logs_and_returns_gracefully(self, sf):
        """Handler must not raise when org_id is absent from metadata."""

        async def _run():
            async with sf() as session:
                session_obj = {
                    "metadata": {},
                    "customer": "cus_x",
                    "subscription": "sub_x",
                }
                # Should not raise
                await _handle_checkout_completed(session_obj, session)

        asyncio.run(_run())

    def test_unknown_org_id_logs_and_returns_gracefully(self, sf):
        async def _run():
            async with sf() as session:
                session_obj = {
                    "metadata": {"org_id": _uid(), "plan_slug": "pro"},
                    "customer": "cus_y",
                    "subscription": "sub_y",
                }
                await _handle_checkout_completed(session_obj, session)
                # No exception — silently swallowed

        asyncio.run(_run())


class TestSubscriptionUpdated:
    def test_updates_status_for_matching_org(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                org = await org_repo.create(org_id=org_id, name="Update Org", slug="update-org")
                org.stripe_subscription_id = "sub_update123"
                org.stripe_subscription_status = "active"
                await session.commit()

                sub_obj = {"id": "sub_update123", "status": "past_due"}
                await _handle_subscription_updated(sub_obj, session)

                await session.refresh(org)
                assert org.stripe_subscription_status == "past_due"

        asyncio.run(_run())

    def test_missing_subscription_id_is_handled(self, sf):
        async def _run():
            async with sf() as session:
                await _handle_subscription_updated({}, session)
                # No exception raised

        asyncio.run(_run())


class TestSubscriptionDeleted:
    def test_marks_canceled_and_reverts_to_free(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                plan_repo = PlanRepository(session)

                free_plan_id = _uid()
                pro_plan_id = _uid()
                await plan_repo.create(plan_id=free_plan_id, name="Free", slug="free", limits={})
                await plan_repo.create(plan_id=pro_plan_id, name="Pro", slug="pro", limits={})

                org_id = _uid()
                org = await org_repo.create(org_id=org_id, name="Cancel Org", slug="cancel-org")
                org.stripe_subscription_id = "sub_del123"
                org.stripe_subscription_status = "active"
                org.plan = "pro"
                org.plan_id = pro_plan_id
                await session.commit()

                sub_obj = {"id": "sub_del123"}
                await _handle_subscription_deleted(sub_obj, session)

                await session.refresh(org)
                assert org.stripe_subscription_status == "canceled"
                assert org.plan == "free"
                assert org.plan_id == free_plan_id

        asyncio.run(_run())

    def test_missing_subscription_id_is_handled(self, sf):
        async def _run():
            async with sf() as session:
                await _handle_subscription_deleted({}, session)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# HTTP-layer tests via TestClient
# ---------------------------------------------------------------------------


def _make_app_for_billing(org: Organization, user_id: str, role: str = "user"):
    """Build a minimal FastAPI test app that injects a fixed OrgContext."""
    from fastapi import FastAPI

    from src.api.org_context import OrgContext
    from src.api.routes import billing as billing_module

    app = FastAPI()

    def _override_org_context():
        return OrgContext(
            user_id=user_id,
            role=role,
            org_id=org.id,
            org=org,
        )

    def _override_current_user():
        from src.api.auth import TokenData

        return TokenData(user_id=user_id, role=role, raw_claims={"sub": user_id, "role": role})

    from src.api.auth import get_current_user
    from src.api.org_context import require_org_context

    app.dependency_overrides[require_org_context] = _override_org_context
    app.dependency_overrides[get_current_user] = _override_current_user
    app.include_router(billing_module.router)
    return app


class TestSubscriptionEndpoint:
    def test_returns_plan_and_stripe_ids(self):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="Sub Org",
            slug="sub-org",
            plan="pro",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_abc",
            stripe_subscription_status="active",
        )
        app = _make_app_for_billing(org, _uid())
        client = TestClient(app)
        resp = client.get("/billing/subscription")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["plan"] == "pro"
        assert data["status"] == "active"
        assert data["stripe_customer_id"] == "cus_abc"
        assert data["stripe_subscription_id"] == "sub_abc"


class TestPortalEndpoint:
    def test_raises_400_when_no_customer(self):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="No Cust Org",
            slug="no-cust-org",
            plan="free",
            stripe_customer_id=None,
        )
        app = _make_app_for_billing(org, _uid(), role="admin")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/billing/portal")
        assert resp.status_code == 400
        body = resp.json()
        # The detail dict is returned either wrapped in an APIResponse envelope
        # (when the global handler is registered) or as Starlette's raw detail dict.
        detail = body.get("error") or body.get("detail") or {}
        code = detail.get("code") if isinstance(detail, dict) else None
        assert code == "NO_STRIPE_CUSTOMER"

    def test_non_admin_gets_403(self):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="User Org",
            slug="user-org",
            plan="free",
            stripe_customer_id="cus_abc",
        )
        app = _make_app_for_billing(org, _uid(), role="user")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/billing/portal")
        assert resp.status_code == 403


class TestCheckoutEndpoint:
    def test_non_admin_gets_403(self):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="User Org",
            slug="user-org",
            plan="free",
            stripe_customer_id="cus_abc",
        )
        app = _make_app_for_billing(org, _uid(), role="user")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/billing/checkout",
            json={
                "plan_slug": "pro",
                "success_url": "https://example.com/s",
                "cancel_url": "https://example.com/c",
            },
        )
        assert resp.status_code == 403

    def test_invalid_plan_returns_400(self, monkeypatch):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="Admin Org",
            slug="admin-org",
            plan="free",
            stripe_customer_id="cus_abc",
        )
        app = _make_app_for_billing(org, _uid(), role="admin")
        client = TestClient(app, raise_server_exceptions=False)

        async def _fake_get_by_slug(_self, slug: str):
            return None

        monkeypatch.setattr(
            "src.api.db.repositories.plans.PlanRepository.get_by_slug",
            _fake_get_by_slug,
        )

        resp = client.post(
            "/billing/checkout",
            json={
                "plan_slug": "nonexistent",
                "success_url": "https://example.com/s",
                "cancel_url": "https://example.com/c",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "INVALID_PLAN"

    def test_valid_plan_creates_checkout(self, monkeypatch):
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="Admin Org",
            slug="admin-org",
            plan="free",
            stripe_customer_id="cus_abc",
        )
        app = _make_app_for_billing(org, _uid(), role="admin")
        client = TestClient(app, raise_server_exceptions=False)

        calls = []

        class FakePlan:
            is_active = True
            stripe_price_id = "price_test"

        async def _fake_get_by_slug(_self, slug: str):
            return FakePlan()

        monkeypatch.setattr(
            "src.api.db.repositories.plans.PlanRepository.get_by_slug",
            _fake_get_by_slug,
        )

        class FakeStripe:
            class checkout:
                class Session:
                    @staticmethod
                    def create(**kwargs):
                        calls.append(kwargs)
                        return {"url": "https://checkout.stripe.com/test", "id": "cs_test"}

        def _fake_get_stripe_client():
            return FakeStripe()

        monkeypatch.setattr(
            "src.api.routes.billing.get_stripe_client",
            _fake_get_stripe_client,
        )

        resp = client.post(
            "/billing/checkout",
            json={
                "plan_slug": "pro",
                "success_url": "https://example.com/s",
                "cancel_url": "https://example.com/c",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["checkout_url"] == "https://checkout.stripe.com/test"
        # Verify the plan's stripe_price_id was used
        assert calls[0]["line_items"] == [{"price": "price_test", "quantity": 1}]

    def test_creates_stripe_customer_when_none(self, monkeypatch):
        """When stripe_customer_id is None the handler must create a Stripe customer."""
        from fastapi.testclient import TestClient

        org = Organization(
            id=_uid(),
            name="Admin Org",
            slug="admin-org",
            plan="free",
            stripe_customer_id=None,
        )
        app = _make_app_for_billing(org, _uid(), role="admin")
        client = TestClient(app, raise_server_exceptions=False)

        class FakePlan:
            is_active = True
            stripe_price_id = "price_test"

        async def _fake_get_by_slug(_self, slug: str):
            return FakePlan()

        monkeypatch.setattr(
            "src.api.db.repositories.plans.PlanRepository.get_by_slug",
            _fake_get_by_slug,
        )

        stripe_calls = []

        class FakeStripe:
            class Customer:
                @staticmethod
                def create(**kwargs):
                    stripe_calls.append(("Customer.create", kwargs))
                    return {"id": "cus_new"}

            class checkout:
                class Session:
                    @staticmethod
                    def create(**kwargs):
                        stripe_calls.append(("Session.create", kwargs))
                        return {"url": "https://checkout.stripe.com/test", "id": "cs_test"}

        monkeypatch.setattr(
            "src.api.routes.billing.get_stripe_client",
            lambda: FakeStripe(),
        )

        resp = client.post(
            "/billing/checkout",
            json={
                "plan_slug": "pro",
                "success_url": "https://example.com/s",
                "cancel_url": "https://example.com/c",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["checkout_url"] == "https://checkout.stripe.com/test"
        # Verify customer was created because stripe_customer_id was None
        assert any(c[0] == "Customer.create" for c in stripe_calls)
        session_call = next(c for c in stripe_calls if c[0] == "Session.create")
        assert session_call[1]["customer"] == "cus_new"


class TestCheckoutRaceCondition:
    """Regression tests for Issue #1118 — duplicate Stripe customer creation race."""

    def test_refresh_called_with_for_update_when_no_customer(self, monkeypatch):
        """When stripe_customer_id is None on a persistent org, db.refresh must
        be called with with_for_update=True to serialize concurrent checkout."""
        from unittest.mock import AsyncMock, MagicMock

        org_id = _uid()
        org = Organization(
            id=org_id,
            name="Race Org",
            slug="race-org",
            plan="free",
            stripe_customer_id=None,
        )

        # Mock inspect to claim the org is persistent (production path)
        mock_inspect = MagicMock()
        mock_inspect.return_value.persistent = True
        monkeypatch.setattr(
            "src.api.routes.billing.sa_inspect",
            mock_inspect,
        )

        mock_db = AsyncMock()
        mock_db.refresh = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.org = org
        mock_ctx.org_id = org_id

        mock_plan = MagicMock()
        mock_plan.is_active = True
        mock_plan.stripe_price_id = "price_test"

        async def _fake_plan_get_by_slug(_self, slug: str):
            return mock_plan

        monkeypatch.setattr(
            "src.api.db.repositories.plans.PlanRepository.get_by_slug",
            _fake_plan_get_by_slug,
        )

        class FakeStripe:
            class Customer:
                @staticmethod
                def create(**kwargs):
                    return {"id": "cus_race"}

            class checkout:
                class Session:
                    @staticmethod
                    def create(**kwargs):
                        return {"url": "https://checkout.test", "id": "cs_race"}

        monkeypatch.setattr(
            "src.api.routes.billing.get_stripe_client",
            lambda: FakeStripe(),
        )

        from src.api.routes.billing import create_checkout_session

        body = MagicMock()
        body.plan_slug = "pro"
        body.success_url = "https://example.com/s"
        body.cancel_url = "https://example.com/c"

        async def _run():
            result = await create_checkout_session(body, mock_ctx, mock_db, MagicMock())
            # Verify refresh was called twice:
            # 1) with_for_update=True (race guard)
            # 2) plain refresh after commit
            assert mock_db.refresh.await_count == 2
            mock_db.refresh.assert_any_await(org, with_for_update=True)
            mock_db.refresh.assert_any_await(org)
            assert result.data["checkout_url"] == "https://checkout.test"

        asyncio.run(_run())


class TestWebhookEndpoint:
    """Test the webhook HTTP endpoint with no signature verification (no secret set)."""

    def _make_webhook_app(self, sf):
        """Build a dedicated webhook test app wired to the in-memory session."""
        from fastapi import FastAPI

        from src.api.routes import billing as billing_module

        app = FastAPI()

        async def _override_db():
            async with sf() as session:
                yield session

        from src.api.db.engine import get_db

        app.dependency_overrides[get_db] = _override_db
        app.include_router(billing_module.router)
        return app

    def test_webhook_secret_unset_returns_503_without_allow_unsigned(self, sf, monkeypatch):
        """Issue #119: webhook must return 503 when STRIPE_WEBHOOK_SECRET is absent
        and STRIPE_ALLOW_UNSIGNED is not explicitly set to '1'."""
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("STRIPE_ALLOW_UNSIGNED", raising=False)
        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = json.dumps({"type": "payment_intent.created", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 503
        assert "STRIPE_WEBHOOK_SECRET" in resp.json()["error"]

    def test_unknown_event_returns_200_not_handled(self, sf, monkeypatch):
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = json.dumps({"type": "payment_intent.created", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["received"] is True
        assert body["handled"] is False

    def test_valid_checkout_event_returns_200_handled(self, sf, monkeypatch):
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("STRIPE_ALLOW_UNSIGNED", "1")
        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)

        async def _create_org():
            async with sf() as session:
                repo = OrganizationRepository(session)
                org_id = _uid()
                await repo.create(org_id=org_id, name="WH Org", slug="wh-org")
                await session.commit()
                return org_id

        org_id = asyncio.run(_create_org())

        payload = json.dumps(
            {
                "type": "checkout.session.completed",
                "data": {
                    "object": {
                        "metadata": {"org_id": org_id, "plan_slug": "pro"},
                        "customer": "cus_wh",
                        "subscription": "sub_wh",
                    }
                },
            }
        )
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["received"] is True
        assert body["handled"] is True

    def test_missing_signature_with_webhook_secret_returns_200(self, sf, monkeypatch):
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testsecret")
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
        from fastapi.testclient import TestClient

        app = self._make_webhook_app(sf)
        client = TestClient(app)
        payload = json.dumps({"type": "ping", "data": {"object": {}}})
        resp = client.post("/billing/webhook", content=payload.encode())
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["received"] is False
        assert body["reason"] == "missing_signature"


# ---------------------------------------------------------------------------
# Alembic migration 0011 round-trip
# ---------------------------------------------------------------------------


class TestMigration0011:
    def test_upgrade_and_downgrade(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_migration_0011_test.db"
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

        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0010"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade to 0010 failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Stripe webhook idempotency guard
# ---------------------------------------------------------------------------


class TestStripeIdempotency:
    """_dispatch_event records event IDs and skips duplicate deliveries."""

    def _checkout_event(self, org_id: str, event_id: str = "evt_test_001") -> dict:
        return {
            "id": event_id,
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "metadata": {"org_id": org_id},
                    "customer": "cus_idem",
                    "subscription": "sub_idem",
                }
            },
        }

    def test_first_delivery_records_event_and_applies_mutation(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Idem Org 1", slug="idem-org-1")
                await session.commit()

                event = self._checkout_event(org_id, "evt_idem_001")
                await _dispatch_event(event, "checkout.session.completed", session)

                # Event recorded
                record = await session.get(ProcessedStripeEvent, "evt_idem_001")
                assert record is not None

                # Mutation applied
                org = await org_repo.get_by_id(org_id)
                assert org is not None
                assert org.stripe_subscription_status == "active"

        asyncio.run(_run())

    def test_duplicate_delivery_is_skipped(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Idem Org 2", slug="idem-org-2")
                await session.commit()

                event = self._checkout_event(org_id, "evt_idem_002")

                # First delivery
                await _dispatch_event(event, "checkout.session.completed", session)

                # Reset status to confirm second delivery does not re-apply
                org = await org_repo.get_by_id(org_id)
                assert org is not None
                org.stripe_subscription_status = "canceled"
                await session.commit()

                # Second delivery (duplicate)
                await _dispatch_event(event, "checkout.session.completed", session)

                # Status must remain "canceled" — duplicate was skipped
                org = await org_repo.get_by_id(org_id)
                assert org is not None
                assert org.stripe_subscription_status == "canceled"

        asyncio.run(_run())

    def test_event_without_id_still_dispatches(self, sf):
        async def _run():
            async with sf() as session:
                org_repo = OrganizationRepository(session)
                org_id = _uid()
                await org_repo.create(org_id=org_id, name="Idem Org 3", slug="idem-org-3")
                await session.commit()

                # Event with no id field — guard skips recording but handler still runs
                event = {
                    "type": "checkout.session.completed",
                    "data": {
                        "object": {
                            "metadata": {"org_id": org_id},
                            "customer": "cus_noid",
                            "subscription": "sub_noid",
                        }
                    },
                }
                await _dispatch_event(event, "checkout.session.completed", session)

                org = await org_repo.get_by_id(org_id)
                assert org is not None
                assert org.stripe_subscription_status == "active"

        asyncio.run(_run())

    def test_processed_stripe_event_model_table_name(self):
        assert ProcessedStripeEvent.__tablename__ == "processed_stripe_events"

    def test_migration_0013_applies_and_rolls_back(self):
        db_path = _PROJECT_ROOT / "data" / "api" / "cogtrix_migration_0013_test.db"
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

        result = subprocess.run(
            ["uv", "run", "alembic", "downgrade", "0012"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0, f"downgrade to 0012 failed:\n{result.stderr}"
        db_path.unlink(missing_ok=True)
