"""删除独立 retrieval_text 投影，Markdown 作为唯一视频文本产物。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_document_text_only"
down_revision: str | Sequence[str] | None = "0002_document_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
_EMPTY_TEXT_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def upgrade() -> None:
    with op.batch_alter_table("video_segment") as batch:
        batch.drop_column("retrieval_hash")
        batch.drop_column("retrieval_text")
    with op.batch_alter_table("video_summary") as batch:
        batch.drop_column("retrieval_hash")
        batch.drop_column("retrieval_text")


def downgrade() -> None:
    with op.batch_alter_table("video_segment") as batch:
        batch.add_column(sa.Column("retrieval_text", sa.Text(), nullable=False, server_default=""))
        batch.add_column(
            sa.Column(
                "retrieval_hash",
                sa.String(64),
                nullable=False,
                server_default=_EMPTY_TEXT_SHA256,
            )
        )
    with op.batch_alter_table("video_summary") as batch:
        batch.add_column(sa.Column("retrieval_text", sa.Text(), nullable=False, server_default=""))
        batch.add_column(
            sa.Column(
                "retrieval_hash",
                sa.String(64),
                nullable=False,
                server_default=_EMPTY_TEXT_SHA256,
            )
        )
    with op.batch_alter_table("video_segment") as batch:
        batch.alter_column("retrieval_text", server_default=None)
        batch.alter_column("retrieval_hash", server_default=None)
    with op.batch_alter_table("video_summary") as batch:
        batch.alter_column("retrieval_text", server_default=None)
        batch.alter_column("retrieval_hash", server_default=None)
