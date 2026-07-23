"""Fix boolean server_defaults for PostgreSQL compatibility.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-29

On SQLite, the initial schema (0001) used ``server_default=sa.text("0")`` for
Boolean columns.  That syntax is invalid on PostgreSQL, which requires the
dialect-agnostic ``false()`` expression.

This migration corrects those columns on **existing SQLite databases** that
were created before 0001 was fixed.  On PostgreSQL the table was always
created with the correct default (from 0001 onward), so the migration is a
no-op on that dialect.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite requires batch mode for column alterations (no ALTER COLUMN support).
        # Fix revoked column in refresh_tokens (was server_default=text("0"))
        with op.batch_alter_table("refresh_tokens") as batch_op:
            batch_op.alter_column(
                "revoked",
                server_default=sa.false(),
                existing_type=sa.Boolean(),
            )
        # Fix revoked column in api_keys (was server_default=text("0"))
        with op.batch_alter_table("api_keys") as batch_op:
            batch_op.alter_column(
                "revoked",
                server_default=sa.false(),
                existing_type=sa.Boolean(),
            )
    # PostgreSQL: no-op — correct server_default was used in 0001 onward


def downgrade() -> None:
    pass  # no rollback needed — leaving correct defaults in place is safe
