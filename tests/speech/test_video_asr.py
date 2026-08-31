from __future__ import annotations

from itertools import pairwise

import pytest

from video_demo.speech.asr import RawAsrSegment
from video_demo.speech.video_asr import (
    VIDEO_ASR_CHUNK_DURATION_MS,
    build_fixed_asr_windows,
    project_fixed_asr_window,
)


def test_build_fixed_asr_windows_covers_thirty_two_minutes_without_overlap() -> None:
    windows = build_fixed_asr_windows(32 * 60_000)

    assert [
        (window.chunk_index, window.upload_range.start_ms, window.upload_range.end_ms)
        for window in windows
    ] == [
        (0, 0, 600_000),
        (1, 600_000, 1_200_000),
        (2, 1_200_000, 1_800_000),
        (3, 1_800_000, 1_920_000),
    ]
    assert all(window.upload_range == window.owned_range for window in windows)
    assert all(
        previous.owned_range.end_ms == current.owned_range.start_ms
        for previous, current in pairwise(windows)
    )


@pytest.mark.parametrize("duration_ms", (600_000, 39 * 60_000))
def test_build_fixed_asr_windows_uses_ten_minute_chunks(duration_ms: int) -> None:
    windows = build_fixed_asr_windows(duration_ms)

    expected_count = 1 if duration_ms == 600_000 else 4
    assert len(windows) == expected_count
    assert windows[0].upload_range.duration_ms == VIDEO_ASR_CHUNK_DURATION_MS
    assert windows[-1].upload_range.end_ms == duration_ms


def test_project_fixed_asr_window_offsets_provider_timestamps_to_absolute_time() -> None:
    window = build_fixed_asr_windows(1_200_000)[1]

    projection = project_fixed_asr_window(
        window,
        language="en",
        raw_segments=(RawAsrSegment(100, 1_100, "second chunk", 0.9),),
    )

    assert (projection.segments[0].start_ms, projection.segments[0].end_ms) == (
        600_100,
        601_100,
    )
    assert projection.language_span.start_ms == 600_000
    assert projection.language_span.end_ms == 1_200_000


def test_project_fixed_asr_window_clamps_invalid_provider_ranges() -> None:
    window = build_fixed_asr_windows(600_000)[0]

    projection = project_fixed_asr_window(
        window,
        language="zh",
        raw_segments=(
            RawAsrSegment(-100, 100, "clamped", 0.9),
            RawAsrSegment(700_000, 701_000, "dropped", 0.9),
        ),
    )

    assert tuple(item.text for item in projection.segments) == ("clamped",)
    assert "ASR_TIMESTAMP_CLAMPED" in projection.warnings
    assert "ASR_TIMESTAMP_DROPPED" in projection.warnings


def test_build_fixed_asr_windows_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        build_fixed_asr_windows(0)
    with pytest.raises(ValueError):
        build_fixed_asr_windows(1_000, chunk_duration_ms=0)
