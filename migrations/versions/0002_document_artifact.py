"""为 3.0 知识文档增加独立 Markdown 制品指针。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_document_artifact"
down_revision: str | Sequence[str] | None = "0001_video_demo"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch 模式会在 SQLite 上安全重建表，同时保留 0001 数据与约束。
    with op.batch_alter_table("video_understanding_run") as batch:
        batch.add_column(sa.Column("document_relative_path", sa.String(1024)))
        batch.add_column(sa.Column("document_sha256", sa.String(64)))
        batch.add_column(sa.Column("document_size_bytes", sa.BigInteger()))


def downgrade() -> None:
    with op.batch_alter_table("video_understanding_run") as batch:
        batch.drop_column("document_size_bytes")
        batch.drop_column("document_sha256")
        batch.drop_column("document_relative_path")
