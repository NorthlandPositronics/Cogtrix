"""Add unique constraint on teams(org_id, name).

Revision ID: 0017
Revises: 0016
Create Date: 2026-05-08

Eliminates TOCTOU race in create_team by enforcing name uniqueness
at the database level.
"""

from __future__ import annotations

from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    with op.batch_alter_table("teams", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_teams_org_name", ["org_id", "name"])


def downgrade() -> None:
    with op.batch_alter_table("teams", schema=None) as batch_op:
        batch_op.drop_constraint("uq_teams_org_name", type_="unique")
