"""Add workspace_id to api_sessions.

Revision ID: 0018
Revises: 0017
Create Date: 2026-05-20

PR #1542 (a3dbd64, "fix(api): enforce workspace isolation on session
and message endpoints") added the ``workspace_id`` column to the
``ApiSessionRecord`` SQLAlchemy model and to
``SessionRepository.create()`` — but did not ship the alembic
migration that creates the column in the database. Fresh installs
(including the docker smoke test in ``ci-docker.yml``) therefore
fail at ``POST /api/v1/sessions`` with ``no such column:
api_sessions.workspace_id`` on SQLite (and the equivalent on
Postgres), producing the "Session creation failed" error the smoke
suite has been reporting since #1542 merged.

The column is nullable in the model, so no backfill is required —
existing rows simply get ``NULL`` for ``workspace_id``. The FK
mirrors the model declaration: ``ON DELETE SET NULL`` so that
deleting a workspace doesn't cascade-delete user sessions.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    with op.batch_alter_table("api_sessions") as batch_op:
        batch_op.add_column(sa.Column("workspace_id", sa.String(36), nullable=True))
        batch_op.create_foreign_key(
            "fk_api_sessions_workspace_id",
            "workspaces",
            ["workspace_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_api_sessions_workspace_id",
            ["workspace_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("api_sessions") as batch_op:
        batch_op.drop_index("ix_api_sessions_workspace_id")
        batch_op.drop_constraint("fk_api_sessions_workspace_id", type_="foreignkey")
        batch_op.drop_column("workspace_id")
