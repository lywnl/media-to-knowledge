"""为音频和图片运行增加结果制品指针。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_media_artifact_pointers"
down_revision: str | Sequence[str] | None = "0004_audio_image_media"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("audio_understanding_run", "image_understanding_run"):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("artifact_relative_path", sa.String(1024)))
            batch.add_column(sa.Column("artifact_sha256", sa.String(64)))


def downgrade() -> None:
    for table in ("image_understanding_run", "audio_understanding_run"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("artifact_sha256")
            batch.drop_column("artifact_relative_path")
