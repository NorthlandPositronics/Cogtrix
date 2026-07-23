"""Add plans table and plan_id FK to organizations.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-27

Adds the Plan model and seeds the four default plans (free, pro, team,
enterprise).  Organizations gain a nullable plan_id FK that references
the active plan record.  The existing plan string field is kept for
backwards compatibility.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_DEFAULT_PLANS = [
    {
        "slug": "free",
        "name": "Free",
        "description": "Single user, no team features.",
        "price_monthly_cents": 0,
        "price_annual_cents": 0,
        "limits": json.dumps(
            {
                "max_users": 1,
                "max_workspaces": 1,
                "max_api_calls_per_month": 1000,
                "max_storage_gb": 1,
            }
        ),
        "is_public": True,
    },
    {
        "slug": "pro",
        "name": "Pro",
        "description": "Small team, extended limits.",
        "price_monthly_cents": 2900,
        "price_annual_cents": 29000,
        "limits": json.dumps(
            {
                "max_users": 10,
                "max_workspaces": 5,
                "max_api_calls_per_month": 50000,
                "max_storage_gb": 20,
            }
        ),
        "is_public": True,
    },
    {
        "slug": "team",
        "name": "Team",
        "description": "Growing teams with SSO and SCIM.",
        "price_monthly_cents": 9900,
        "price_annual_cents": 99000,
        "limits": json.dumps(
            {
                "max_users": 50,
                "max_workspaces": 20,
                "max_api_calls_per_month": 500000,
                "max_storage_gb": 100,
            }
        ),
        "is_public": True,
    },
    {
        "slug": "enterprise",
        "name": "Enterprise",
        "description": "Unlimited users, SAML, custom SLAs.",
        "price_monthly_cents": 0,  # custom pricing
        "price_annual_cents": 0,
        "limits": json.dumps(
            {"max_users": 0, "max_workspaces": 0, "max_api_calls_per_month": 0, "max_storage_gb": 0}
        ),
        "is_public": True,
    },
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def upgrade() -> None:
    # 1. Create plans table.
    op.create_table(
        "plans",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("slug", sa.String(32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("price_monthly_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("price_annual_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("limits", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_plans_slug"),
    )
    op.create_index("ix_plans_slug", "plans", ["slug"])

    # 2. Seed default plans.
    conn = op.get_bind()
    now = _now_iso()
    for p in _DEFAULT_PLANS:
        conn.execute(
            sa.text(
                "INSERT INTO plans "
                "(id, name, slug, description, price_monthly_cents, price_annual_cents, "
                "limits, is_active, is_public, created_at, updated_at) "
                "VALUES (:id, :name, :slug, :desc, :pm, :pa, :limits, :active, :public, :now, :now)"
            ),
            {
                "id": str(uuid.uuid4()),
                "name": p["name"],
                "slug": p["slug"],
                "desc": p["description"],
                "pm": p["price_monthly_cents"],
                "pa": p["price_annual_cents"],
                "limits": p["limits"],
                "active": 1,
                "public": 1,
                "now": now,
            },
        )

    # 3. Add plan_id FK to organizations.
    with op.batch_alter_table("organizations") as batch_op:
        batch_op.add_column(sa.Column("plan_id", sa.String(36), nullable=True))
        batch_op.create_index("ix_organizations_plan_id", ["plan_id"])
        batch_op.create_foreign_key(
            "fk_organizations_plan_id",
            "plans",
            ["plan_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("organizations") as batch_op:
        batch_op.drop_constraint("fk_organizations_plan_id", type_="foreignkey")
        batch_op.drop_index("ix_organizations_plan_id")
        batch_op.drop_column("plan_id")

    op.drop_index("ix_plans_slug", table_name="plans")
    op.drop_table("plans")
