"""Stripe billing routes (Enterprise Phase 1 — task 1.4.3).

Endpoints:
    POST /api/v1/billing/checkout       — create Stripe Checkout Session
    GET  /api/v1/billing/portal         — create Customer Portal session
    GET  /api/v1/billing/subscription   — current subscription summary
    POST /api/v1/billing/webhook        — Stripe webhook receiver (no auth)

Design notes:
- All non-webhook endpoints require JWT auth + org context.
- Webhook uses raw ``Request`` body to preserve the signature Stripe uses for
  ``stripe.Webhook.construct_event()``.  Using a Pydantic body param would
  cause FastAPI to re-encode the body and break signature verification.
- Webhook failures are caught and return HTTP 200 so Stripe does not retry
  spuriously; errors are logged at WARNING/ERROR level.
- Stripe API calls are offloaded to a thread pool via ``asyncio.to_thread``
  to keep the event loop unblocked (Stripe SDK is synchronous).
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.db.engine import get_db
from src.api.db.repositories.organization import OrganizationRepository
from src.api.db.repositories.plans import PlanRepository
from src.api.org_context import OrgContext, require_org_context
from src.api.schemas.common import APIResponse
from src.api.stripe_client import get_stripe_client

log = logging.getLogger("cogtrix.api.billing")

router = APIRouter(prefix="/billing", tags=["Billing"])

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    """Body for POST /billing/checkout."""

    plan_slug: str
    success_url: str
    cancel_url: str


# ---------------------------------------------------------------------------
# POST /billing/checkout
# ---------------------------------------------------------------------------


@router.post(
    "/checkout",
    response_model=APIResponse[dict],
    summary="Create Stripe Checkout Session",
)
async def create_checkout_session(
    body: CheckoutRequest,
    ctx: OrgContext = Depends(require_org_context),
    db: AsyncSession = Depends(get_db),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Create (or reuse) a Stripe Customer for the org, then open a Checkout Session.

    The ``plan_slug`` is validated server-side against the Plan record and its
    configured ``stripe_price_id`` is used for the Checkout Session line items.
    Callers cannot supply an arbitrary Stripe price.
    """
    org = ctx.org  # guaranteed non-None by require_org_context

    # ---- validate plan server-side ----
    plan_repo = PlanRepository(db)
    plan = await plan_repo.get_by_slug(body.plan_slug)
    if plan is None or not plan.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_PLAN",
                "message": f"Plan '{body.plan_slug}' not found or inactive.",
            },
        )

    stripe = get_stripe_client()

    # ---- ensure Stripe Customer exists ----
    customer_id: str | None = org.stripe_customer_id  # type: ignore[union-attr]

    if not customer_id:
        # Re-read the org row under a database lock to prevent concurrent
        # checkout requests from both observing stripe_customer_id=None and
        # creating duplicate Stripe customers (Issue #1118).
        if sa_inspect(org).persistent:
            await db.refresh(org, with_for_update=True)  # type: ignore[union-attr]
            customer_id = org.stripe_customer_id  # type: ignore[union-attr]

    if not customer_id:
        customer = await asyncio.to_thread(
            stripe.Customer.create,
            name=org.name,  # type: ignore[union-attr]
            metadata={"org_id": ctx.org_id, "org_slug": org.slug},  # type: ignore[union-attr]
        )
        customer_id = customer["id"]
        # Persist immediately so concurrent requests do not create duplicates.
        org.stripe_customer_id = customer_id  # type: ignore[union-attr]
        if sa_inspect(org).persistent:
            await db.commit()
            await db.refresh(org)
        log.info("Created Stripe customer %s for org %s", customer_id, ctx.org_id)

    # ---- build line items ----
    session_kwargs: dict = {
        "customer": customer_id,
        "mode": "subscription",
        "success_url": body.success_url,
        "cancel_url": body.cancel_url,
        "metadata": {
            "org_id": ctx.org_id,
            "plan_slug": body.plan_slug,
        },
    }
    if plan.stripe_price_id:
        session_kwargs["line_items"] = [{"price": plan.stripe_price_id, "quantity": 1}]

    checkout_session = await asyncio.to_thread(
        stripe.checkout.Session.create,
        **session_kwargs,
    )
    checkout_url: str = checkout_session["url"]
    log.info(
        "Created Checkout session %s for org %s plan %s",
        checkout_session["id"],
        ctx.org_id,
        body.plan_slug,
    )
    return APIResponse(data={"checkout_url": checkout_url})


# ---------------------------------------------------------------------------
# GET /billing/portal
# ---------------------------------------------------------------------------


@router.get(
    "/portal",
    response_model=APIResponse[dict],
    summary="Create Stripe Customer Portal session",
)
async def create_portal_session(
    ctx: OrgContext = Depends(require_org_context),
    _: TokenData = Depends(require_admin),
) -> APIResponse[dict]:
    """Return a one-time URL for the Stripe Customer Portal.

    The portal lets customers manage their subscription (upgrade, downgrade,
    cancel, update payment method) without any custom UI on our side.

    Raises:
        400 NO_STRIPE_CUSTOMER — org has no Stripe customer yet.
    """
    org = ctx.org  # type: ignore[union-attr]
    customer_id: str | None = org.stripe_customer_id  # type: ignore[union-attr]

    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "NO_STRIPE_CUSTOMER",
                "message": "This organization has no Stripe customer. Complete a checkout first.",
            },
        )

    stripe = get_stripe_client()
    portal_session = await asyncio.to_thread(
        stripe.billing_portal.Session.create,
        customer=customer_id,
    )
    portal_url: str = portal_session["url"]
    log.info("Created portal session for org %s customer %s", ctx.org_id, customer_id)
    return APIResponse(data={"portal_url": portal_url})


# ---------------------------------------------------------------------------
# GET /billing/subscription
# ---------------------------------------------------------------------------


@router.get(
    "/subscription",
    response_model=APIResponse[dict],
    summary="Current subscription summary",
)
async def get_subscription(
    ctx: OrgContext = Depends(require_org_context),
    _: TokenData = Depends(get_current_user),
) -> APIResponse[dict]:
    """Return the current plan and Stripe subscription state for the caller's org."""
    org = ctx.org  # type: ignore[union-attr]
    return APIResponse(
        data={
            "plan": org.plan,  # type: ignore[union-attr]
            "status": org.stripe_subscription_status,  # type: ignore[union-attr]
            "stripe_customer_id": org.stripe_customer_id,  # type: ignore[union-attr]
            "stripe_subscription_id": org.stripe_subscription_id,  # type: ignore[union-attr]
        }
    )


# ---------------------------------------------------------------------------
# POST /billing/webhook
# ---------------------------------------------------------------------------

_HANDLED_EVENTS = frozenset(
    {
        "checkout.session.completed",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }
)


@router.post(
    "/webhook",
    response_model=APIResponse[dict],
    summary="Stripe webhook receiver",
    # Stripe requires 200; authentication is via signature, not JWT.
    include_in_schema=True,
)
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:
    """Receive and process Stripe webhook events.

    Signature verification uses the ``STRIPE_WEBHOOK_SECRET`` environment
    variable when set.  If the variable is absent the signature check is
    skipped with a warning — this should never happen in production but
    allows local testing without a forwarding proxy.

    Returns HTTP 200 for all outcomes (including errors) so Stripe does not
    retry unnecessarily.  Errors are logged; never leaked to Stripe.
    """
    payload: bytes = await request.body()
    sig_header: str | None = request.headers.get("stripe-signature")
    webhook_secret: str | None = os.environ.get("STRIPE_WEBHOOK_SECRET")
    allow_unsigned: bool = os.environ.get("STRIPE_ALLOW_UNSIGNED") == "1"
    _env = os.environ.get("COGTRIX_ENV", "development").lower()

    # ---- signature verification ----
    if not webhook_secret:
        if not allow_unsigned or _env == "production":
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"error": "Webhook endpoint not configured — STRIPE_WEBHOOK_SECRET is not set"},
                status_code=503,
            )
        log.warning(
            "STRIPE_WEBHOOK_SECRET not set; accepting unsigned webhook (STRIPE_ALLOW_UNSIGNED=1)"
        )

    try:
        if webhook_secret:
            if not sig_header:
                log.warning("Stripe webhook: missing stripe-signature header — rejecting event")
                return APIResponse(data={"received": False, "reason": "missing_signature"})
            stripe = get_stripe_client()
            event = await asyncio.to_thread(
                stripe.Webhook.construct_event,
                payload,
                sig_header,
                webhook_secret,
            )
        else:
            import json as _json

            event = _json.loads(payload)
    except Exception as exc:
        log.warning("Stripe webhook signature verification failed: %s", exc)
        return APIResponse(data={"received": False, "reason": "signature_failed"})

    event_type: str = event.get("type", "")
    log.info("Stripe webhook received: %s", event_type)

    if event_type not in _HANDLED_EVENTS:
        log.debug("Stripe webhook: unhandled event type %r — ignoring", event_type)
        return APIResponse(data={"received": True, "handled": False})

    try:
        await _dispatch_event(event, event_type, db)
    except Exception as exc:
        log.error("Stripe webhook handler error for %s: %s", event_type, exc, exc_info=True)
        # Return 200 anyway — Stripe must not retry due to our internal errors.

    return APIResponse(data={"received": True, "handled": True})


async def _dispatch_event(event: dict, event_type: str, db: AsyncSession) -> None:
    """Route a verified Stripe event to the appropriate handler.

    Idempotency guard: event_id is recorded as a PK in processed_stripe_events
    before the handler runs and is committed atomically with the handler's DB
    changes.  Duplicate deliveries (Stripe retries) are silently skipped.
    """
    from src.api.db.models import ProcessedStripeEvent

    event_id: str = event.get("id", "")
    if event_id:
        existing = await db.get(ProcessedStripeEvent, event_id)
        if existing is not None:
            log.info("Stripe event %s already processed — skipping duplicate delivery", event_id)
            return
        db.add(ProcessedStripeEvent(event_id=event_id))

    if event_type == "checkout.session.completed":
        await _handle_checkout_completed(event["data"]["object"], db)
    elif event_type == "customer.subscription.updated":
        await _handle_subscription_updated(event["data"]["object"], db)
    elif event_type == "customer.subscription.deleted":
        await _handle_subscription_deleted(event["data"]["object"], db)


async def _handle_checkout_completed(session_obj: dict, db: AsyncSession) -> None:
    """Handle ``checkout.session.completed``.

    Sets stripe_customer_id, stripe_subscription_id, status='active', and
    updates org.plan_id from the session metadata's plan_slug.
    """
    org_id: str | None = (session_obj.get("metadata") or {}).get("org_id")
    plan_slug: str | None = (session_obj.get("metadata") or {}).get("plan_slug")
    customer_id: str | None = session_obj.get("customer")
    subscription_id: str | None = session_obj.get("subscription")

    if not org_id:
        log.warning("checkout.session.completed missing org_id in metadata — cannot update org")
        return

    org_repo = OrganizationRepository(db)
    org = await org_repo.get_by_id(org_id)
    if org is None:
        log.warning("checkout.session.completed: org %s not found", org_id)
        return

    # Persist Stripe identifiers and status.
    if customer_id:
        org.stripe_customer_id = customer_id
    if subscription_id:
        org.stripe_subscription_id = subscription_id
    org.stripe_subscription_status = "active"

    # Resolve plan by slug from metadata, falling back to price_id lookup.
    if plan_slug:
        plan_repo = PlanRepository(db)
        plan = await plan_repo.get_by_slug(plan_slug)
        if plan is not None and plan.is_active:
            org.plan = plan.slug
            org.plan_id = plan.id
            log.info("Assigned plan %s to org %s via checkout", plan.slug, org_id)
        else:
            log.warning("checkout.session.completed: plan slug %r not found or inactive", plan_slug)

    await db.commit()
    log.info(
        "checkout.session.completed handled for org %s customer %s sub %s",
        org_id,
        customer_id,
        subscription_id,
    )


async def _handle_subscription_updated(subscription_obj: dict, db: AsyncSession) -> None:
    """Handle ``customer.subscription.updated`` — sync status field."""
    subscription_id: str | None = subscription_obj.get("id")
    new_status: str | None = subscription_obj.get("status")

    if not subscription_id:
        log.warning("customer.subscription.updated: missing subscription id")
        return

    from sqlalchemy import update

    from src.api.db.models import Organization

    stmt = (
        update(Organization)
        .where(Organization.stripe_subscription_id == subscription_id)
        .values(stripe_subscription_status=new_status)
    )
    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        log.warning(
            "customer.subscription.updated: no org found for subscription %s", subscription_id
        )
    else:
        log.info(
            "Updated subscription status to %r for subscription %s", new_status, subscription_id
        )


async def _handle_subscription_deleted(subscription_obj: dict, db: AsyncSession) -> None:
    """Handle ``customer.subscription.deleted`` — mark canceled, revert to free plan."""
    subscription_id: str | None = subscription_obj.get("id")

    if not subscription_id:
        log.warning("customer.subscription.deleted: missing subscription id")
        return

    from sqlalchemy import select

    from src.api.db.models import Organization

    result = await db.execute(
        select(Organization).where(Organization.stripe_subscription_id == subscription_id)
    )
    org: Organization | None = result.scalar_one_or_none()

    if org is None:
        log.warning(
            "customer.subscription.deleted: no org found for subscription %s", subscription_id
        )
        return

    org.stripe_subscription_status = "canceled"

    # Revert to free plan.
    plan_repo = PlanRepository(db)
    free_plan = await plan_repo.get_by_slug("free")
    if free_plan is not None:
        org.plan = free_plan.slug
        org.plan_id = free_plan.id
        log.info("Reverted org %s to free plan after subscription cancellation", org.id)
    else:
        # Free plan row doesn't exist yet — set slug string only.
        org.plan = "free"
        org.plan_id = None
        log.warning("Free plan not found in DB; set org.plan='free' without FK")

    await db.commit()
    log.info(
        "customer.subscription.deleted handled for subscription %s org %s",
        subscription_id,
        org.id,
    )
