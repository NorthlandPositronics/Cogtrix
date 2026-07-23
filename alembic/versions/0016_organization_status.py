"""Add status column to organizations.

Revision ID: 0016
Revises: 0015
Create Date: 2026-05-01 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column("organizations", sa.Column("status", sa.String(16), nullable=True))
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE organizations SET status = CASE WHEN is_active = 1 THEN 'active' ELSE 'inactive' END"
        )
    )
    with op.batch_alter_table("organizations") as batch_op:
        batch_op.alter_column("status", existing_type=sa.String(16), nullable=False)
        batch_op.create_index("ix_organizations_status", ["status"])


def downgrade() -> None:
    with op.batch_alter_table("organizations") as batch_op:
        batch_op.drop_index("ix_organizations_status")
        batch_op.drop_column("status")
