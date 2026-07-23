"""Add usage_records table for metering.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-27

Adds UsageRecord model (Enterprise Phase 1 — task 1.4.2).
Each row records a metered usage event for an organization.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "usage_records",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("workspace_id", sa.String(36), nullable=True),
        sa.Column("user_id", sa.String(36), nullable=True),
        sa.Column("period_year", sa.Integer(), nullable=False),
        sa.Column("period_month", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_usage_records_org_id", "usage_records", ["org_id"])
    op.create_index("ix_usage_records_event_type", "usage_records", ["event_type"])
    op.create_index("ix_usage_records_period_year", "usage_records", ["period_year"])
    op.create_index("ix_usage_records_recorded_at", "usage_records", ["recorded_at"])
    # Composite index for the most common query: org + period
    op.create_index(
        "ix_usage_records_org_period",
        "usage_records",
        ["org_id", "period_year", "period_month"],
    )


def downgrade() -> None:
    op.drop_index("ix_usage_records_org_period", table_name="usage_records")
    op.drop_index("ix_usage_records_recorded_at", table_name="usage_records")
    op.drop_index("ix_usage_records_period_year", table_name="usage_records")
    op.drop_index("ix_usage_records_event_type", table_name="usage_records")
    op.drop_index("ix_usage_records_org_id", table_name="usage_records")
    op.drop_table("usage_records")
