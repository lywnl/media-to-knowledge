from __future__ import annotations

from pydantic import Field

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.document import DocumentGenerationConfig, TranscriptSource


class DocumentWritingContext(FrozenModel):
    """章节写作和全局编辑共享的本地事实闭包。"""

    run_id: StableId
    asset_sha256: Sha256
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    transcript_source: TranscriptSource
    document_config: DocumentGenerationConfig
