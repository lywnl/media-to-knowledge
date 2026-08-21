from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.result import (
    SummaryChapter,
    VideoSegment,
    VideoSummary,
    VideoUnderstandingResult,
    validate_evidence_references,
)
from video_demo.errors import ErrorCode, VideoDemoError


def _evidence(
    evidence_id: str = "ev_asr_001",
    start_ms: int = 100,
    end_ms: int = 400,
) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _segment(
    evidence_refs: tuple[str, ...] = ("ev_asr_001",),
    start_ms: int = 0,
    end_ms: int = 500,
) -> VideoSegment:
    retrieval_text = "标题：问候"
    return VideoSegment(
        segment_id="seg_001",
        start_ms=start_ms,
        end_ms=end_ms,
        title="问候",
        summary_zh="讲者向观众问好。",
        speakers=("SPEAKER_01",),
        languages=("en",),
        topics=("问候",),
        entities=(),
        actions=("问好",),
        keywords=("问候",),
        original_keywords=("Hello",),
        evidence_refs=evidence_refs,
        retrieval_text=retrieval_text,
        retrieval_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
    )


def _result(segment: VideoSegment | None = None) -> VideoUnderstandingResult:
    actual_segment = segment or _segment()
    retrieval_text = "标题：测试视频"
    summary = VideoSummary(
        title="测试视频",
        duration_ms=500,
        summary_zh="视频包含一段问候。",
        chapters=(
            SummaryChapter(
                title="问候",
                start_ms=actual_segment.start_ms,
                end_ms=actual_segment.end_ms,
                segment_ids=(actual_segment.segment_id,),
            ),
        ),
        languages=("en",),
        speakers=("SPEAKER_01",),
        topics=("问候",),
        entities=(),
        actions=("问好",),
        keywords=("问候",),
        original_keywords=("Hello",),
        retrieval_text=retrieval_text,
        retrieval_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
    )
    return VideoUnderstandingResult(
        schema_version="1.0.0",
        run_id="run_001",
        asset_sha256="c" * 64,
        segments=(actual_segment,),
        summary=summary,
    )


def test_result_accepts_known_evidence_inside_segment_range() -> None:
    validate_evidence_references(_result(), (_evidence(),))


def test_result_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_evidence(), _evidence()))

    assert raised.value.code == ErrorCode.DUPLICATE_EVIDENCE_ID


def test_result_rejects_unknown_evidence_reference() -> None:
    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(_segment(("ev_missing",))), (_evidence(),))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


def test_result_rejects_evidence_outside_segment_range() -> None:
    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_evidence(start_ms=450, end_ms=600),))

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_SEGMENT


def test_result_rejects_summary_chapter_with_unknown_segment() -> None:
    result = _result()
    invalid_summary = result.summary.model_copy(
        update={
            "chapters": (
                SummaryChapter(
                    title="伪造章节",
                    start_ms=0,
                    end_ms=500,
                    segment_ids=("seg_missing",),
                ),
            ),
        },
    )
    invalid_result = result.model_copy(update={"summary": invalid_summary})

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(invalid_result, (_evidence(),))

    assert raised.value.code == ErrorCode.UNKNOWN_SEGMENT_REFERENCE


def test_result_rejects_segment_or_chapter_outside_video_duration() -> None:
    segment = _segment(end_ms=600)

    with pytest.raises(ValidationError):
        _result(segment)


def test_result_rejects_chapter_that_does_not_cover_referenced_segment() -> None:
    result = _result()
    invalid_summary = result.summary.model_copy(
        update={
            "chapters": (
                SummaryChapter(
                    title="过窄章节",
                    start_ms=100,
                    end_ms=400,
                    segment_ids=(result.segments[0].segment_id,),
                ),
            ),
        },
    )

    with pytest.raises(ValidationError):
        VideoUnderstandingResult.model_validate(
            {
                **result.model_dump(),
                "summary": invalid_summary.model_dump(),
            },
        )
