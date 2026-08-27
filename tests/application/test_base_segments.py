from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import pytest

from video_demo.application.base_segments import build_base_segments
from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    SpeechBoundaryCandidate,
)
from video_demo.domain.evidence import SceneBoundary, SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError

_ASSET_SHA256 = "a" * 64


def _limits(
    *,
    transcript_items: int = 20_000,
    transcript_chars: int = 2_000_000,
    scenes: int = 20_000,
    segments: int = 20_000,
) -> EvidencePreparationLimits:
    return EvidencePreparationLimits(
        max_transcript_evidence_items=transcript_items,
        max_transcript_chars=transcript_chars,
        max_scene_boundaries=scenes,
        max_base_segments=segments,
    )


def _scene(index: int, start_ms: int, end_ms: int) -> SceneBoundary:
    return SceneBoundary(
        evidence_id=f"scene_{index:03d}",
        start_ms=start_ms,
        end_ms=end_ms,
        transition="candidate" if index == 0 else "hard_cut",
        score=0.9,
    )


def _subtitle(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
    text: str = "字幕",
) -> SubtitleCue:
    return SubtitleCue(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="zh",
        stream_index=0,
    )


def _speech(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
    text: str = "语音",
) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _build(
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue] = (),
    scenes: Sequence[SceneBoundary] | None = None,
    boundaries: Sequence[SpeechBoundaryCandidate] = (),
    limits: EvidencePreparationLimits | None = None,
):
    return build_base_segments(
        asset_sha256=_ASSET_SHA256,
        duration_ms=duration_ms,
        transcript_evidence=transcript,
        scenes=scenes or (_scene(0, 0, duration_ms),),
        speech_boundaries=boundaries,
        limits=limits or _limits(),
    )


def test_two_hour_textless_single_scene_uses_grid_and_covers_whole_video() -> None:
    segments = _build(7_200_000)

    assert len(segments) == 240
    assert segments[0].start_ms == 0
    assert segments[-1].end_ms == 7_200_000
    assert all(left.end_ms == right.start_ms for left, right in pairwise(segments))
    assert all(item.duration_ms == 30_000 for item in segments)
    assert all(item.transcript_source == "NONE" for item in segments)


def test_grid_starts_at_thirty_seconds_and_keeps_short_tail() -> None:
    segments = _build(95_000)

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (0, 30_000),
        (30_000, 60_000),
        (60_000, 90_000),
        (90_000, 95_000),
    ]


def test_boundary_inside_subtitle_is_removed_and_evidence_has_unique_owner() -> None:
    cue = _subtitle("subtitle_001", 25_000, 35_000)
    segments = _build(60_000, (cue,))

    owners = [item for item in segments if cue.evidence_id in item.evidence_refs]
    assert [(item.start_ms, item.end_ms) for item in owners] == [(25_000, 35_000)]
    assert sum(item.evidence_refs.count(cue.evidence_id) for item in segments) == 1
    assert all(not (item.end_ms == 30_000 or item.start_ms == 30_000) for item in segments)


def test_sentence_and_silence_candidates_become_safe_boundaries() -> None:
    segments = _build(
        40_000,
        boundaries=(
            SpeechBoundaryCandidate(10_000, "sentence_end"),
            SpeechBoundaryCandidate(20_000, "silence"),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [
        (0, 10_000),
        (10_000, 20_000),
        (20_000, 30_000),
        (30_000, 40_000),
    ]


def test_exactly_five_minute_cue_is_allowed_as_one_segment() -> None:
    cue = _subtitle("subtitle_001", 0, 300_000)

    segments = _build(300_000, (cue,))

    assert len(segments) == 1
    assert segments[0].duration_ms == 300_000


def test_overlapping_transcript_chain_without_safe_cut_fails_closed() -> None:
    transcript = (
        _speech("asr_001", 0, 200_000),
        _speech("asr_002", 150_000, 350_000),
    )

    with pytest.raises(VideoDemoError) as raised:
        _build(350_000, transcript)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


@pytest.mark.parametrize(
    ("transcript", "limit_overrides"),
    [
        ((_subtitle("subtitle_001", 0, 1_000),), {"transcript_items": 1}),
        ((_subtitle("subtitle_001", 0, 1_000, "两个字"),), {"transcript_chars": 2}),
    ],
)
def test_transcript_budget_is_fail_closed(
    transcript: tuple[SubtitleCue, ...],
    limit_overrides: dict[str, int],
) -> None:
    limits = _limits(**limit_overrides)
    duplicated = (
        transcript
        if limits.max_transcript_chars == 2
        else (*transcript, _subtitle("subtitle_002", 1_000, 2_000))
    )

    with pytest.raises(VideoDemoError) as raised:
        _build(5_000, duplicated, limits=limits)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_non_normalized_scene_input_and_final_segment_budget_are_fail_closed() -> None:
    scenes = (_scene(0, 0, 1_000), _scene(1, 1_000, 2_000))

    with pytest.raises(VideoDemoError) as scene_error:
        _build(2_000, scenes=scenes, limits=_limits(scenes=1))
    with pytest.raises(VideoDemoError) as segment_error:
        _build(60_000, limits=_limits(segments=1))

    assert scene_error.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert segment_error.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_duplicate_scene_ids_are_visual_contract_error() -> None:
    scenes = (
        _scene(0, 0, 1_000),
        SceneBoundary(
            evidence_id="scene_000",
            start_ms=1_000,
            end_ms=2_000,
            transition="hard_cut",
            score=0.8,
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        _build(2_000, scenes=scenes)

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID


def test_adjacent_long_cues_can_split_at_shared_safe_endpoint() -> None:
    transcript = (
        _subtitle("subtitle_001", 0, 200_000),
        _subtitle("subtitle_002", 200_000, 400_000),
    )

    segments = _build(400_000, transcript)

    owners = {
        evidence_ref: (segment.start_ms, segment.end_ms)
        for segment in segments
        for evidence_ref in segment.evidence_refs
    }
    assert owners == {
        "subtitle_001": (0, 200_000),
        "subtitle_002": (200_000, 400_000),
    }


def test_bypassing_scene_normalization_with_257_refs_is_visual_contract_error() -> None:
    scenes = tuple(
        _scene(index, index * 1_000, (index + 1) * 1_000)
        for index in range(300)
    )
    cue = _subtitle("subtitle_001", 0, 300_000)

    with pytest.raises(VideoDemoError) as raised:
        _build(300_000, (cue,), scenes=scenes)

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID


def test_long_cue_can_reference_more_than_eight_intersecting_scenes() -> None:
    scenes = tuple(
        _scene(index, index * 30_000, (index + 1) * 30_000)
        for index in range(10)
    )
    cue = _subtitle("subtitle_001", 0, 300_000)

    segments = _build(300_000, (cue,), scenes=scenes)

    assert len(segments) == 1
    assert len(segments[0].scene_refs) == 10


def test_single_segment_transcript_reference_budget_fails_closed() -> None:
    accepted = tuple(
        _subtitle(f"subtitle_{index:03d}", 0, 1_000)
        for index in range(256)
    )
    rejected = (*accepted, _subtitle("subtitle_256", 0, 1_000))

    segments = _build(1_000, accepted)

    assert len(segments) == 1
    assert len(segments[0].evidence_refs) == 256
    with pytest.raises(VideoDemoError) as raised:
        _build(1_000, rejected)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
