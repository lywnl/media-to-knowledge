from __future__ import annotations

import pytest

from video_demo.domain.run import TimeRange
from video_demo.visual.windows import (
    BoundaryCandidate,
    build_provisional_windows,
    merge_adjacent_windows,
    refine_windows_with_ocr,
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
