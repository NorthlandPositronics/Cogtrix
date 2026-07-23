"""Add missing performance indexes for enterprise queries.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-27

Adds indexes identified during holistic audit:
- ix_usage_records_org_event_period: composite covering the hot
  count_for_period() query (org_id, event_type, period_year, period_month).
- ix_usage_records_workspace_id / ix_usage_records_user_id: per-user and
  per-workspace usage lookups used in billing detail views.
- ix_workspaces_is_active: all workspace list queries filter on is_active.
"""

from __future__ import annotations

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # Composite index covering the plan-enforcement hot path:
    # UsageRepository.count_for_period filters on all four columns.
    op.create_index(
        "ix_usage_records_org_event_period",
        "usage_records",
        ["org_id", "event_type", "period_year", "period_month"],
    )
    # Per-user and per-workspace usage drill-down queries.
    op.create_index(
        "ix_usage_records_workspace_id",
        "usage_records",
        ["workspace_id"],
    )
    op.create_index(
        "ix_usage_records_user_id",
        "usage_records",
        ["user_id"],
    )
    # Active-workspace filter appears in every workspace list query.
    op.create_index(
        "ix_workspaces_is_active",
        "workspaces",
        ["is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_workspaces_is_active", table_name="workspaces")
    op.drop_index("ix_usage_records_user_id", table_name="usage_records")
    op.drop_index("ix_usage_records_workspace_id", table_name="usage_records")
    op.drop_index("ix_usage_records_org_event_period", table_name="usage_records")
