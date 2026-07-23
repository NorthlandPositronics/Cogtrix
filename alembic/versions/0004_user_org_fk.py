"""Add org_id FK to users table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-27

Links existing users to an Organization (Enterprise Phase 1 — task 1.1.2).
The column is nullable: existing single-tenant users have no org assignment
until the migration path (task 1.1.5) moves them to a default org.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("org_id", sa.String(36), nullable=True))
        batch_op.create_index("ix_users_org_id", ["org_id"])
        batch_op.create_foreign_key(
            "fk_users_org_id",
            "organizations",
            ["org_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("fk_users_org_id", type_="foreignkey")
        batch_op.drop_index("ix_users_org_id")
        batch_op.drop_column("org_id")
