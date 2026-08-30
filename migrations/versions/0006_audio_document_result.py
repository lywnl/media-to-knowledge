"""为音频结果增加独立资产、章节和摘要表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_audio_document_result"
down_revision: str | Sequence[str] | None = "0005_media_artifact_pointers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _scope() -> tuple[sa.Column[object], ...]:
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
    op.create_table(
        "audio_asset",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *_scope(),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("object_ref", sa.String(128), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_audio_asset_scope_run",
        ),
    )
    op.create_table(
        "audio_segment",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *_scope(),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("segment_id", sa.String(128), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id", "segment_id",
            name="uq_audio_segment_scope_run_id",
        ),
    )
    op.create_index(
        "ix_audio_segment_scope_run_time",
        "audio_segment",
        ["tenant_id", "application_id", "knowledge_base_id", "run_id", "start_ms"],
    )
    op.create_table(
        "audio_summary",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *_scope(),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        *_timestamps(),
        sa.UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_audio_summary_scope_run",
        ),
    )


def downgrade() -> None:
    op.drop_table("audio_summary")
    op.drop_index("ix_audio_segment_scope_run_time", table_name="audio_segment")
    op.drop_table("audio_segment")
    op.drop_table("audio_asset")
