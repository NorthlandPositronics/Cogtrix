"""Add Stripe billing columns to organizations.

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-27

Adds three nullable columns to the ``organizations`` table to support
Stripe billing integration (Enterprise Phase 1 — task 1.4.3):

- ``stripe_customer_id``       — Stripe Customer object ID (cus_…)
- ``stripe_subscription_id``   — Stripe Subscription object ID (sub_…)
- ``stripe_subscription_status`` — latest subscription status string
  (e.g. ``"active"``, ``"canceled"``, ``"past_due"``)

All three columns are nullable: existing rows are unaffected and orgs
that have not yet gone through a Checkout flow will have NULL values.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("stripe_customer_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("stripe_subscription_id", sa.String(64), nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("stripe_subscription_status", sa.String(32), nullable=True),
    )
    op.create_index(
        "ix_organizations_stripe_customer_id",
        "organizations",
        ["stripe_customer_id"],
    )
    op.create_index(
        "ix_organizations_stripe_subscription_id",
        "organizations",
        ["stripe_subscription_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_organizations_stripe_subscription_id", table_name="organizations")
    op.drop_index("ix_organizations_stripe_customer_id", table_name="organizations")
    op.drop_column("organizations", "stripe_subscription_status")
    op.drop_column("organizations", "stripe_subscription_id")
    op.drop_column("organizations", "stripe_customer_id")
