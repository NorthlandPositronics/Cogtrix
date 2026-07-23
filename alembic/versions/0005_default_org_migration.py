"""Assign existing single-tenant users to a default organization.

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-27

Migration path for task 1.1.5 (Enterprise Phase 1).

Before this migration all users have org_id = NULL because they were
created in the single-tenant era.  This migration:

  1. Creates a single ``Organization`` row with slug ``'default'``
     (skipped if it already exists — idempotent).
  2. Assigns every user whose ``org_id IS NULL`` to that organization.

The default org acts as a compatibility wrapper for existing deployments
so that enterprise routes with ``require_org_context`` work without
forcing every deployment to create orgs manually.

Downgrade removes the assignment and deletes the default org.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

DEFAULT_ORG_SLUG = "default"
DEFAULT_ORG_NAME = "Default Organization"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Create the default org if it doesn't exist (idempotent).
    result = conn.execute(
        sa.text("SELECT id FROM organizations WHERE slug = :slug"),
        {"slug": DEFAULT_ORG_SLUG},
    )
    row = result.fetchone()

    if row is None:
        default_org_id = str(uuid.uuid4())
        now = _now_iso()
        conn.execute(
            sa.text(
                "INSERT INTO organizations "
                "(id, name, slug, plan, settings, created_at, updated_at, is_active) "
                "VALUES (:id, :name, :slug, :plan, :settings, :created_at, :updated_at, :is_active)"
            ),
            {
                "id": default_org_id,
                "name": DEFAULT_ORG_NAME,
                "slug": DEFAULT_ORG_SLUG,
                "plan": "free",
                "settings": None,
                "created_at": now,
                "updated_at": now,
                "is_active": 1,  # SQLite stores booleans as integers
            },
        )
    else:
        default_org_id = row[0]

    # 2. Assign all unassigned users to the default org.
    conn.execute(
        sa.text("UPDATE users SET org_id = :org_id WHERE org_id IS NULL"),
        {"org_id": default_org_id},
    )


def downgrade() -> None:
    conn = op.get_bind()

    # Find the default org.
    result = conn.execute(
        sa.text("SELECT id FROM organizations WHERE slug = :slug"),
        {"slug": DEFAULT_ORG_SLUG},
    )
    row = result.fetchone()
    if row is None:
        return  # nothing to undo

    default_org_id = row[0]

    # Unassign users that were assigned by this migration.
    conn.execute(
        sa.text("UPDATE users SET org_id = NULL WHERE org_id = :org_id"),
        {"org_id": default_org_id},
    )

    # Delete the default org.
    conn.execute(
        sa.text("DELETE FROM organizations WHERE id = :id"),
        {"id": default_org_id},
    )
