"""新增独立音频和图片对象、运行表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_audio_image_media"
down_revision: str | Sequence[str] | None = "0003_document_text_only"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _scope_columns() -> tuple[sa.Column[object], ...]:
    return (
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("application_id", sa.String(128), nullable=False),
        sa.Column("knowledge_base_id", sa.String(128), nullable=False),
    )


def _timestamps() -> tuple[sa.Column[object], ...]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def upgrade() -> None:
    for kind in ("audio", "image"):
        op.create_table(
            f"{kind}_object",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            *_scope_columns(),
            sa.Column("object_ref", sa.String(128), nullable=False),
            sa.Column("original_filename", sa.String(512), nullable=False),
            sa.Column("declared_mime", sa.String(128), nullable=False),
            sa.Column("detected_mime", sa.String(128), nullable=False),
            sa.Column("size_bytes", sa.BigInteger(), nullable=False),
            sa.Column("sha256", sa.String(64), nullable=False),
            sa.Column("relative_path", sa.String(1024), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("scan_details", sa.JSON(), nullable=False),
            *_timestamps(),
            sa.UniqueConstraint(
                "tenant_id", "application_id", "knowledge_base_id", "object_ref",
                name=f"uq_{kind}_object_scope_ref",
            ),
        )
        op.create_index(
            f"ix_{kind}_object_scope_sha",
            f"{kind}_object",
            ["tenant_id", "application_id", "knowledge_base_id", "sha256"],
        )
        op.create_table(
            f"{kind}_understanding_run",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            *_scope_columns(),
            sa.Column("run_id", sa.String(128), nullable=False),
            sa.Column("object_ref", sa.String(128), nullable=False),
            sa.Column("idempotency_key", sa.String(128), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("current_stage", sa.String(64), nullable=False),
            sa.Column("warning_codes", sa.JSON(), nullable=False),
            sa.Column("error_code", sa.String(128)),
            sa.Column("config_snapshot", sa.JSON(), nullable=False),
            sa.Column("document_relative_path", sa.String(1024)),
            sa.Column("document_sha256", sa.String(64)),
            sa.Column("document_size_bytes", sa.BigInteger()),
            *_timestamps(),
            sa.UniqueConstraint(
                "tenant_id", "application_id", "knowledge_base_id", "run_id",
                name=f"uq_{kind}_run_scope_id",
            ),
            sa.UniqueConstraint(
                "tenant_id", "application_id", "knowledge_base_id", "idempotency_key",
                name=f"uq_{kind}_run_scope_idempotency",
            ),
        )


def downgrade() -> None:
    for kind in ("image", "audio"):
        op.drop_table(f"{kind}_understanding_run")
        op.drop_index(f"ix_{kind}_object_scope_sha", table_name=f"{kind}_object")
        op.drop_table(f"{kind}_object")
