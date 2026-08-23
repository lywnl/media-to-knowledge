from __future__ import annotations

import pytest

from video_demo.domain.evidence import (
    BoundingBox,
    OcrEvidence,
    OcrLine,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import SegmentUnderstanding
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import (
    BoundaryPoint,
    WindowUnderstanding,
    _project_evidence,
    merge_segment_understandings,
    snap_boundary,
)


def _understanding(
    *,
    evidence_ref: str,
    summary: str,
    title: str = "产品演示",
) -> SegmentUnderstanding:
    return SegmentUnderstanding(
        title=title,
        summary_zh=summary,
        speakers=("SPEAKER_01",),
        languages=("zh",),
        topics=("视频检索",),
        entities=("Demo",),
        actions=("演示",),
        keywords=("视频理解",),
        original_keywords=("retrieval",),
        evidence_refs=(evidence_ref,),
    )


def _boundaries(middle_source: str = "silence") -> tuple[BoundaryPoint, ...]:
    return (
        BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
        BoundaryPoint(timestamp_ms=500, sources=(middle_source,)),
        BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
    )


def test_boundary_snapping_can_only_return_existing_candidate() -> None:
    candidates = _boundaries()

    assert snap_boundary(480, candidates, max_distance_ms=50) == 500
    with pytest.raises(ValueError, match="没有可吸附的候选边界"):
        snap_boundary(700, candidates, max_distance_ms=50)


def test_adjacent_similar_windows_merge_after_snapping_to_candidates() -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_001",
            start_ms=20,
            end_ms=480,
            understanding=_understanding(
                evidence_ref="asr_001",
                summary="介绍视频理解能力。",
            ),
        ),
        WindowUnderstanding(
            window_id="window_002",
            start_ms=520,
            end_ms=980,
            understanding=_understanding(
                evidence_ref="asr_002",
                summary="演示检索文本生成。",
            ),
        ),
    )

    segments = merge_segment_understandings(
        windows,
        boundaries=_boundaries(),
        max_snap_distance_ms=50,
    )

    assert len(segments) == 1
    assert (segments[0].start_ms, segments[0].end_ms) == (0, 1_000)
    assert segments[0].evidence_refs == ("asr_001", "asr_002")
    assert segments[0].summary_zh == "介绍视频理解能力。演示检索文本生成。"


def test_merge_removes_duplicate_summary_sentences_but_keeps_distinct_facts() -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_001",
            start_ms=0,
            end_ms=500,
            understanding=_understanding(
                evidence_ref="asr_001",
                summary="介绍视频理解能力。视频支持字幕检索。",
            ),
        ),
        WindowUnderstanding(
            window_id="window_002",
            start_ms=500,
            end_ms=1_000,
            understanding=_understanding(
                evidence_ref="asr_002",
                summary="介绍视频理解能力。视频支持字幕检索。第二部分展示 OCR。",
            ),
        ),
    )

    segments = merge_segment_understandings(windows, boundaries=_boundaries())

    assert len(segments) == 1
    assert segments[0].summary_zh == (
        "介绍视频理解能力。视频支持字幕检索。第二部分展示 OCR。"
    )


def test_merge_projects_subtitle_asr_ocr_and_visual_facts_to_segment() -> None:
    subtitle = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=500,
        text="字幕原文",
        language="zh",
        stream_index=0,
    )
    asr = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=500,
        text="ASR 原文",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    ocr = OcrEvidence(
        evidence_id="ocr_001",
        start_ms=0,
        end_ms=500,
        keyframe_id="keyframe_001",
        timestamp_ms=100,
        language="zh",
        lines=(
            OcrLine(
                text="画面文字",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.9,
            ),
        ),
        provider_request_id="ocr-request-001",
    )
    understanding = _understanding(
        evidence_ref="subtitle_001",
        summary="片段摘要。",
    ).model_copy(
        update={
            "evidence_refs": ("subtitle_001", "asr_001", "ocr_001"),
            "visual_facts": ("画面显示软件界面。",),
        },
    )

    segment = merge_segment_understandings(
        (
            WindowUnderstanding(
                window_id="window_001",
                start_ms=0,
                end_ms=500,
                understanding=understanding,
            ),
        ),
        boundaries=(
            BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
            BoundaryPoint(timestamp_ms=500, sources=("video_end",)),
        ),
        evidence=(subtitle, asr, ocr),
    )[0]

    assert segment.transcript_source == "SUBTITLE"
    assert segment.transcript_text == "字幕原文"
    assert segment.ocr_text == ("画面文字",)
    assert segment.visual_facts == ("画面显示软件界面。",)


def test_evidence_projection_is_stable_when_reference_order_changes() -> None:
    subtitle_late = SubtitleCue(
        evidence_id="subtitle_002",
        start_ms=300,
        end_ms=500,
        text="后半句",
        language="zh",
        stream_index=0,
    )
    subtitle_early = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=200,
        text="前半句",
        language="zh",
        stream_index=0,
    )
    ocr_late = OcrEvidence(
        evidence_id="ocr_002",
        start_ms=300,
        end_ms=500,
        keyframe_id="keyframe_002",
        timestamp_ms=400,
        language="zh",
        lines=(
            OcrLine(
                text="重复文字",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.9,
            ),
        ),
        provider_request_id="ocr-request-002",
    )
    ocr_early = ocr_late.model_copy(
        update={
            "evidence_id": "ocr_001",
            "start_ms": 0,
            "end_ms": 200,
            "keyframe_id": "keyframe_001",
            "timestamp_ms": 100,
            "provider_request_id": "ocr-request-001",
        },
    )
    understanding = _understanding(
        evidence_ref="subtitle_002",
        summary="片段摘要。",
    ).model_copy(
        update={
            "evidence_refs": (
                "ocr_002",
                "subtitle_002",
                "ocr_001",
                "subtitle_001",
            ),
        },
    )
    window = WindowUnderstanding(
        window_id="window_001",
        start_ms=0,
        end_ms=500,
        understanding=understanding,
    )

    segment = merge_segment_understandings(
        (window,),
        boundaries=(
            BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
            BoundaryPoint(timestamp_ms=500, sources=("video_end",)),
        ),
        evidence=(subtitle_late, subtitle_early, ocr_late, ocr_early),
    )[0]

    assert segment.transcript_text == "前半句 后半句"
    assert segment.ocr_text == ("重复文字",)


def test_evidence_projection_removes_overlapping_duplicate_transcript_and_normalizes_ocr() -> None:
    first = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=500,
        text="重复  口播",
        language="zh",
        stream_index=0,
    )
    overlapping = first.model_copy(
        update={
            "evidence_id": "subtitle_002",
            "text": "重复口播",
            "start_ms": 200,
            "end_ms": 700,
        },
    )
    separated = first.model_copy(
        update={
            "evidence_id": "subtitle_003",
            "start_ms": 700,
            "end_ms": 1_000,
        },
    )
    ocr = OcrEvidence(
        evidence_id="ocr_001",
        start_ms=0,
        end_ms=1_000,
        keyframe_id="keyframe_001",
        timestamp_ms=100,
        language="zh",
        lines=(
            OcrLine(
                text="  重复   文字 ",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.9,
            ),
            OcrLine(
                text="重复 文字",
                bounding_box=BoundingBox(x=10, y=0, width=10, height=10),
                confidence=0.9,
            ),
        ),
        provider_request_id="ocr-request-001",
    )

    transcript, ocr_text, source = _project_evidence(
        ("subtitle_001", "subtitle_002", "subtitle_003", "ocr_001", "ocr_001"),
        {item.evidence_id: item for item in (first, overlapping, separated, ocr)},
    )

    assert transcript == "重复 口播 重复口播 重复 口播"
    assert ocr_text == ("重复 文字",)
    assert source == "SUBTITLE"


def test_merged_semantic_text_respects_domain_limits_and_keeps_all_evidence_refs() -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_001",
            start_ms=0,
            end_ms=500,
            understanding=_understanding(
                evidence_ref="asr_001",
                title="甲" * 160,
                summary="甲" * 3_000,
            ),
        ),
        WindowUnderstanding(
            window_id="window_002",
            start_ms=500,
            end_ms=1_000,
            understanding=_understanding(
                evidence_ref="asr_002",
                title="乙" * 160,
                summary="乙" * 3_000,
            ),
        ),
    )

    segments = merge_segment_understandings(windows, boundaries=_boundaries())

    assert len(segments) == 1
    assert len(segments[0].title) == 200
    assert len(segments[0].summary_zh) == 4_000
    assert segments[0].evidence_refs == ("asr_001", "asr_002")


@pytest.mark.parametrize("hard_source", ["scene_hard", "ocr_change"])
def test_hard_scene_or_ocr_boundary_prevents_semantic_merge(hard_source: str) -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_001",
            start_ms=0,
            end_ms=500,
            understanding=_understanding(evidence_ref="asr_001", summary="第一部分。"),
        ),
        WindowUnderstanding(
            window_id="window_002",
            start_ms=500,
            end_ms=1_000,
            understanding=_understanding(evidence_ref="asr_002", summary="第二部分。"),
        ),
    )

    segments = merge_segment_understandings(windows, boundaries=_boundaries(hard_source))

    assert len(segments) == 2


def test_partial_qwen_failure_preserves_successful_window() -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_failed",
            start_ms=0,
            end_ms=500,
            failure_code="DEPENDENCY_TEMPORARY_FAILURE",
        ),
        WindowUnderstanding(
            window_id="window_success",
            start_ms=500,
            end_ms=1_000,
            understanding=_understanding(evidence_ref="asr_002", summary="成功片段。"),
        ),
    )

    segments = merge_segment_understandings(windows, boundaries=_boundaries())

    assert len(segments) == 1
    assert segments[0].evidence_refs == ("asr_002",)


def test_all_qwen_windows_failed_closes_result_stage() -> None:
    windows = (
        WindowUnderstanding(
            window_id="window_failed",
            start_ms=0,
            end_ms=1_000,
            failure_code="DEPENDENCY_TEMPORARY_FAILURE",
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        merge_segment_understandings(windows, boundaries=_boundaries())

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
