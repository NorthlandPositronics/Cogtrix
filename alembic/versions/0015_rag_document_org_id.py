"""Scope RAG documents to organizations.

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-30 00:00:00.000000
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

DEFAULT_ORG_SLUG = "default"
DEFAULT_ORG_NAME = "Default Organization"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_default_org_id() -> str:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT id FROM organizations WHERE slug = :slug"),
        {"slug": DEFAULT_ORG_SLUG},
    )
    row = result.fetchone()
    if row is not None:
        return row[0]

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
            "is_active": 1,
        },
    )
    return default_org_id


def upgrade() -> None:
    conn = op.get_bind()
    default_org_id = _ensure_default_org_id()

    op.add_column("rag_documents", sa.Column("org_id", sa.String(36), nullable=True))

    conn.execute(
        sa.text("UPDATE rag_documents SET org_id = :org_id WHERE org_id IS NULL"),
        {"org_id": default_org_id},
    )

    with op.batch_alter_table("rag_documents") as batch_op:
        batch_op.alter_column("org_id", existing_type=sa.String(36), nullable=False)
        batch_op.create_index("ix_rag_documents_org_id", ["org_id"])
        batch_op.create_foreign_key(
            "fk_rag_documents_org_id",
            "organizations",
            ["org_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("rag_documents") as batch_op:
        batch_op.drop_constraint("fk_rag_documents_org_id", type_="foreignkey")
        batch_op.drop_index("ix_rag_documents_org_id")
        batch_op.drop_column("org_id")
