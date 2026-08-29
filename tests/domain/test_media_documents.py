from __future__ import annotations

import pytest

from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.image_document import (
    ImageContentBlock,
    ImageDocument,
    ImageSourceEvidence,
    ImageUnderstandingResult,
)


def _audio_chapter(chapter_id: str, start_ms: int, end_ms: int) -> AudioChapter:
    return AudioChapter(
        chapter_id=chapter_id,
        start_ms=start_ms,
        end_ms=end_ms,
        title="章节",
        title_evidence_refs=("asr_001",),
        summary_zh="摘要",
        summary_evidence_refs=("asr_001",),
        body_blocks=(),
        claims=(),
        evidence_refs=("asr_001",),
        transcript_source="ASR",
    )


def test_audio_result_requires_contiguous_timeline() -> None:
    with pytest.raises(ValueError, match="连续"):
        AudioUnderstandingResult(
            run_id="run_audio_001",
            asset_sha256="a" * 64,
            summary=AudioDocumentSummary(title="音频", duration_ms=2_000, overview_zh="概览"),
            chapters=(
                _audio_chapter("chapter_001", 0, 900),
                _audio_chapter("chapter_002", 1_000, 2_000),
            ),
        )


def test_image_result_keeps_only_image_evidence_closure() -> None:
    source = ImageSourceEvidence(
        evidence_id="image_source_001",
        relative_path="images/source.jpg",
        mime_type="image/jpeg",
        sha256="a" * 64,
        width=100,
        height=80,
        size_bytes=100,
    )
    document = ImageDocument(
        title="图片",
        overview_zh="图片概览",
        content_blocks=(
            ImageContentBlock(
                content_type="DESCRIPTION",
                text="图片内容",
                evidence_refs=(source.evidence_id,),
            ),
        ),
        claims=(),
        evidence_refs=(source.evidence_id,),
    )
    result = ImageUnderstandingResult(
        run_id="run_image_001",
        asset_sha256="b" * 64,
        document=document,
        source=source,
    )
    assert result.document.evidence_refs == ("image_source_001",)
