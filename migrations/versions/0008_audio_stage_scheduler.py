"""增加可恢复的音频转写与 LLM 阶段状态表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_audio_stage_scheduler"
down_revision: str | Sequence[str] | None = "0007_video_stage_scheduler"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audio_pipeline_stage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("application_id", sa.String(128), nullable=False),
        sa.Column("knowledge_base_id", sa.String(128), nullable=False),
        sa.Column("run_id", sa.String(128), nullable=False),
        sa.Column("stage_name", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, default=0),
        sa.Column("max_attempts", sa.Integer(), nullable=False, default=3),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("checkpoint_relative_path", sa.String(1024)),
        sa.Column("checkpoint_sha256", sa.String(64)),
        sa.Column("error_code", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id", "stage_name",
            name="uq_audio_stage_scope_run_name",
        ),
    )
    op.create_index(
        "ix_audio_stage_recovery",
        "audio_pipeline_stage",
        ["stage_name", "status", "next_attempt_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_audio_stage_recovery", table_name="audio_pipeline_stage")
    op.drop_table("audio_pipeline_stage")
