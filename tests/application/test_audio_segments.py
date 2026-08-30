from __future__ import annotations

from collections.abc import Sequence

from video_demo.application.audio_segments import build_audio_segments
from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    SpeechBoundaryCandidate,
)
from video_demo.domain.evidence import SpeechSegment


def _speech(evidence_id: str, start_ms: int, end_ms: int) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="语音",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _limits() -> EvidencePreparationLimits:
    return EvidencePreparationLimits(
        max_transcript_evidence_items=20_000,
        max_transcript_chars=2_000_000,
        max_scene_boundaries=20_000,
        max_base_segments=20_000,
    )


def test_audio_segments_cover_timeline_without_scene_references() -> None:
    transcript: Sequence[SpeechSegment] = (
        _speech("asr_001", 0, 1_000),
        _speech("asr_002", 31_000, 32_000),
    )

    segments = build_audio_segments(
        "a" * 64,
        60_000,
        transcript,
        (SpeechBoundaryCandidate(30_000, "silence"),),
        _limits(),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (0, 30_000),
        (30_000, 60_000),
    ]
    assert all(item.scene_refs == () for item in segments if hasattr(item, "scene_refs"))
    assert all(not hasattr(item, "scene_refs") for item in segments)
    assert {ref for item in segments for ref in item.evidence_refs} == {
        "asr_001",
        "asr_002",
    }
