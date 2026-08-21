"""建立视频理解 Demo 的持久化表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_video_demo"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCOPE_COLUMNS = (
    sa.Column("tenant_id", sa.String(128), nullable=False),
    sa.Column("application_id", sa.String(128), nullable=False),
    sa.Column("knowledge_base_id", sa.String(128), nullable=False),
)
TIMESTAMP_COLUMNS = (
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)


def upgrade() -> None:
    _create_video_object()
    _create_video_asset()
    _create_video_run()
    _create_job()
    _create_video_segment()
    _create_video_summary()


def downgrade() -> None:
    op.drop_table("video_summary")
    op.drop_index("ix_video_segment_scope_run_time", table_name="video_segment")
    op.drop_table("video_segment")
    op.drop_index("ix_job_claim", table_name="job")
    op.drop_table("job")
    op.drop_table("video_understanding_run")
    op.drop_table("video_asset")
    op.drop_index("ix_video_object_scope_sha", table_name="video_object")
    op.drop_table("video_object")


def _create_video_object() -> None:
    op.create_table(
        "video_object",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("object_ref", sa.String(128), nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("declared_mime", sa.String(128), nullable=False),
        sa.Column("detected_mime", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.String(1024), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("scan_details", sa.JSON(), nullable=False),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
            name="uq_video_object_scope_ref",
        ),
    )
    op.create_index(
        "ix_video_object_scope_sha",
        "video_object",
        ["tenant_id", "application_id", "knowledge_base_id", "sha256"],
    )


def _create_video_asset() -> None:
    op.create_table(
        "video_asset",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("object_ref", sa.String(128), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("manifest_relative_path", sa.String(1024), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
            "source_sha256",
            name="uq_video_asset_scope_object_sha",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "asset_id",
            name="uq_video_asset_scope_id",
        ),
    )


def _create_video_run() -> None:
    op.create_table(
        "video_understanding_run",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("object_ref", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("current_stage", sa.String(64), nullable=False),
        sa.Column("warning_codes", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(128)),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("artifact_manifest_relative_path", sa.String(1024)),
        sa.Column("artifact_manifest_sha256", sa.String(64)),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            name="uq_video_run_scope_id",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_video_run_scope_idempotency",
        ),
    )


def _create_job() -> None:
    op.create_table(
        "job",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("job_id", sa.String(128), nullable=False),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(128)),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "job_id",
            name="uq_job_scope_id",
        ),
    )
    op.create_index("ix_job_claim", "job", ["status", "next_attempt_at", "lease_expires_at"])


def _create_video_segment() -> None:
    op.create_table(
        "video_segment",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("segment_id", sa.String(128), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("retrieval_hash", sa.String(64), nullable=False),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "segment_id",
            name="uq_video_segment_scope_run_id",
        ),
    )
    op.create_index(
        "ix_video_segment_scope_run_time",
        "video_segment",
        ["tenant_id", "application_id", "knowledge_base_id", "run_id", "start_ms"],
    )


def _create_video_summary() -> None:
    op.create_table(
        "video_summary",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        *SCOPE_COLUMNS,
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("retrieval_hash", sa.String(64), nullable=False),
        *TIMESTAMP_COLUMNS,
        sa.UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            name="uq_video_summary_scope_run",
        ),
    )
