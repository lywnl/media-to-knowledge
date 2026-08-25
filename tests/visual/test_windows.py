from __future__ import annotations

import pytest

from video_demo.domain.evidence import SceneBoundary
from video_demo.domain.run import TimeRange
from video_demo.visual.windows import (
    BoundaryCandidate,
    build_provisional_windows,
    build_visual_observation_windows,
    merge_adjacent_windows,
    refine_windows_with_ocr,
    validate_contiguous_windows,
)


def test_weak_scene_alone_does_not_force_semantic_boundary() -> None:
    windows = build_provisional_windows(
        duration_ms=20_000,
        candidates=(BoundaryCandidate(8_000, "scene", score=0.8),),
    )

    assert [(window.start_ms, window.end_ms) for window in windows] == [(0, 20_000)]


def test_sentence_silence_and_language_build_hybrid_windows() -> None:
    windows = build_provisional_windows(
        duration_ms=40_000,
        candidates=(
            BoundaryCandidate(9_800, "sentence_end", score=1.0),
            BoundaryCandidate(10_000, "silence", score=1.0),
            BoundaryCandidate(25_000, "language_change", score=1.0),
            BoundaryCandidate(25_100, "scene", score=0.9),
        ),
    )

    assert [(window.start_ms, window.end_ms) for window in windows] == [
        (0, 10_000),
        (10_000, 25_000),
        (25_000, 40_000),
    ]


def test_maximum_30_second_window_is_a_hard_fallback() -> None:
    windows = build_provisional_windows(duration_ms=65_000, candidates=())

    assert [(window.start_ms, window.end_ms) for window in windows] == [
        (0, 30_000),
        (30_000, 60_000),
        (60_000, 65_000),
    ]


def test_minimum_two_second_window_ignores_too_close_boundary() -> None:
    windows = build_provisional_windows(
        duration_ms=6_000,
        candidates=(
            BoundaryCandidate(1_000, "silence", score=1.0),
            BoundaryCandidate(3_000, "silence", score=1.0),
        ),
    )

    assert [(window.start_ms, window.end_ms) for window in windows] == [
        (0, 3_000),
        (3_000, 6_000),
    ]


def test_ocr_page_change_refines_only_second_stage() -> None:
    provisional = build_provisional_windows(duration_ms=20_000, candidates=())

    refined = refine_windows_with_ocr(
        provisional,
        ocr_changes_ms=(8_000,),
        duration_ms=20_000,
    )

    assert [(window.start_ms, window.end_ms) for window in provisional] == [(0, 20_000)]
    assert [(window.start_ms, window.end_ms) for window in refined] == [
        (0, 8_000),
        (8_000, 20_000),
    ]


def test_merge_adjacent_windows_packs_without_changing_existing_boundaries() -> None:
    merged = merge_adjacent_windows(
        (
            TimeRange(start_ms=0, end_ms=8_000),
            TimeRange(start_ms=8_000, end_ms=19_000),
            TimeRange(start_ms=19_000, end_ms=31_000),
            TimeRange(start_ms=31_000, end_ms=40_000),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in merged] == [
        (0, 19_000),
        (19_000, 40_000),
    ]


def test_merge_adjacent_windows_rejects_gap_or_oversized_source_window() -> None:
    with pytest.raises(ValueError, match="连续"):
        merge_adjacent_windows(
            (
                TimeRange(start_ms=0, end_ms=10_000),
                TimeRange(start_ms=10_001, end_ms=20_000),
            ),
        )

    with pytest.raises(ValueError, match="上限"):
        merge_adjacent_windows((TimeRange(start_ms=0, end_ms=30_001),))


def test_visual_observation_windows_use_fixed_cadence_and_scene_boundaries() -> None:
    windows = build_visual_observation_windows(
        duration_ms=60_000,
        scenes=(
            SceneBoundary(
                evidence_id="scene_0",
                start_ms=0,
                end_ms=22_000,
                transition="candidate",
                score=1.0,
            ),
            SceneBoundary(
                evidence_id="scene_1",
                start_ms=22_000,
                end_ms=60_000,
                transition="hard_cut",
                score=1.0,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in windows] == [
        (0, 22_000),
        (22_000, 30_000),
        (30_000, 45_000),
        (45_000, 60_000),
    ]
    validate_contiguous_windows(windows, duration_ms=60_000)


def test_visual_observation_windows_do_not_create_short_scene_window() -> None:
    windows = build_visual_observation_windows(
        duration_ms=30_000,
        scenes=(
            SceneBoundary(
                evidence_id="scene_0",
                start_ms=0,
                end_ms=8_000,
                transition="candidate",
                score=1.0,
            ),
            SceneBoundary(
                evidence_id="scene_1",
                start_ms=8_000,
                end_ms=30_000,
                transition="hard_cut",
                score=1.0,
            ),
        ),
    )

    assert all(item.duration_ms >= 8_000 for item in windows)
    validate_contiguous_windows(windows, duration_ms=30_000)


def test_visual_observation_windows_merge_short_tail_without_losing_coverage() -> None:
    starts = [0, 4_612, 5_278, 7_671, 17_863, 20_991]
    duration_ms = 23_070
    scenes = tuple(
        SceneBoundary(
            evidence_id=f"scene_{index:02d}",
            start_ms=start_ms,
            end_ms=starts[index + 1] if index + 1 < len(starts) else duration_ms,
            transition="candidate",
            score=1.0,
        )
        for index, start_ms in enumerate(starts)
    )

    windows = build_visual_observation_windows(duration_ms=duration_ms, scenes=scenes)

    assert [(item.start_ms, item.end_ms) for item in windows] == [
        (0, 15_000),
        (15_000, duration_ms),
    ]
    assert all(8_000 <= item.duration_ms <= 30_000 for item in windows)


def test_visual_observation_windows_split_long_gaps_within_limits() -> None:
    duration_ms = 213_570
    windows = build_visual_observation_windows(duration_ms=duration_ms, scenes=())

    assert windows[0].start_ms == 0
    assert windows[-1].end_ms == duration_ms
    assert all(8_000 <= item.duration_ms <= 30_000 for item in windows)
    validate_contiguous_windows(windows, duration_ms=duration_ms)


def test_visual_observation_windows_allow_video_shorter_than_minimum() -> None:
    windows = build_visual_observation_windows(duration_ms=5_000, scenes=())

    assert [(item.start_ms, item.end_ms) for item in windows] == [(0, 5_000)]


def test_validate_contiguous_windows_rejects_gap() -> None:
    with pytest.raises(ValueError, match="空档"):
        validate_contiguous_windows(
            (TimeRange(start_ms=0, end_ms=500), TimeRange(start_ms=501, end_ms=1_000)),
            duration_ms=1_000,
        )
