from __future__ import annotations

import pytest

from video_demo.domain.result import SegmentUnderstanding
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import (
    BoundaryPoint,
    WindowUnderstanding,
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
