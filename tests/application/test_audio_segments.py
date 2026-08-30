from __future__ import annotations

from collections.abc import Sequence

from video_demo.application.audio_contracts import (
    AudioEvidencePreparationLimits,
    AudioSpeechBoundaryCandidate,
)
from video_demo.application.audio_segments import build_audio_segments
from video_demo.domain.evidence import SpeechSegment, SubtitleCue


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


def _limits() -> AudioEvidencePreparationLimits:
    return AudioEvidencePreparationLimits(
        max_transcript_evidence_items=20_000,
        max_transcript_chars=2_000_000,
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
        (AudioSpeechBoundaryCandidate(30_000, "silence"),),
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


def test_audio_segments_keep_dense_asr_boundaries_sparse() -> None:
    transcript = tuple(
        _speech(f"asr_{index:03d}", index * 1_000, (index + 1) * 1_000) for index in range(120)
    )
    boundaries = tuple(
        AudioSpeechBoundaryCandidate((index + 1) * 1_000, "sentence_end", 0.8)
        for index in range(119)
    )

    segments = build_audio_segments(
        "a" * 64,
        120_000,
        transcript,
        boundaries,
        _limits(),
    )

    assert len(segments) <= 6
    assert segments[0].start_ms == 0
    assert segments[-1].end_ms == 120_000
    assert tuple(ref for segment in segments for ref in segment.evidence_refs) == tuple(
        item.evidence_id for item in transcript
    )


def test_audio_segments_preserve_subtitle_boundaries_when_grid_crosses_a_cue() -> None:
    transcript = (
        SubtitleCue(
            evidence_id="subtitle_001",
            start_ms=10_000,
            end_ms=50_000,
            text="一条跨越网格边界的字幕",
            language="zh",
            stream_index=0,
        ),
    )

    segments = build_audio_segments(
        "a" * 64,
        60_000,
        transcript,
        (AudioSpeechBoundaryCandidate(30_000, "silence"),),
        _limits(),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (0, 10_000),
        (10_000, 60_000),
    ]
    assert tuple(ref for segment in segments for ref in segment.evidence_refs) == ("subtitle_001",)
