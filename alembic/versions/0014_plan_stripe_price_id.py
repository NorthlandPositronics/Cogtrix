"""Add stripe_price_id to plans table.

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-30 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column(
        "plans",
        sa.Column("stripe_price_id", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("plans", "stripe_price_id")
