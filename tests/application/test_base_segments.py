from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import pytest

from video_demo.application.base_segments import build_base_segments
from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    SpeechBoundaryCandidate,
)
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError

_ASSET_SHA256 = "a" * 64


def _limits(*, max_segments: int = 20_000) -> EvidencePreparationLimits:
    return EvidencePreparationLimits(
        max_transcript_evidence_items=20_000,
        max_transcript_chars=2_000_000,
        max_base_segments=max_segments,
    )


def _speech(evidence_id: str, start_ms: int, end_ms: int) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="语音内容",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _build(
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue] = (),
    boundaries: Sequence[SpeechBoundaryCandidate] = (),
    limits: EvidencePreparationLimits | None = None,
):
    return build_base_segments(
        asset_sha256=_ASSET_SHA256,
        duration_ms=duration_ms,
        transcript_evidence=transcript,
        speech_boundaries=boundaries,
        limits=limits or _limits(),
    )


def test_textless_video_uses_sparse_grid_and_covers_timeline() -> None:
    segments = _build(7_200_000)

    assert len(segments) == 240
    assert segments[0].start_ms == 0
    assert segments[-1].end_ms == 7_200_000
    assert all(left.end_ms == right.start_ms for left, right in pairwise(segments))
    assert all(item.transcript_source == "NONE" for item in segments)
    assert all(not hasattr(item, "scene_refs") for item in segments)


def test_grid_keeps_short_tail() -> None:
    segments = _build(95_000)

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (0, 30_000), (30_000, 60_000), (60_000, 90_000), (90_000, 95_000)
    ]


def test_transcript_evidence_is_assigned_without_visual_fields() -> None:
    transcript = (_speech("asr_001", 0, 1_000), _speech("asr_002", 31_000, 32_000))
    segments = _build(60_000, transcript)
    assert sum(len(item.evidence_refs) for item in segments) == 2
    assert all(item.transcript_source == "ASR" for item in segments)
    assert all(not hasattr(item, "scene_refs") for item in segments)


def test_sentence_boundaries_are_used_only_when_safe() -> None:
    transcript = tuple(_speech(f"asr_{i}", i * 2_000, i * 2_000 + 1_000) for i in range(60))
    boundaries = tuple(
        SpeechBoundaryCandidate(item.end_ms, "sentence_end", 0.8)
        for item in transcript if item.end_ms < 120_000
    )
    segments = _build(120_000, transcript, boundaries)
    assert len(segments) <= 5
    assert sum(len(item.evidence_refs) for item in segments) == len(transcript)


def test_segment_budget_failure_is_explicit() -> None:
    with pytest.raises(VideoDemoError, match="基础片段数量超过上限") as raised:
        _build(120_000, limits=_limits(max_segments=1))
    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
