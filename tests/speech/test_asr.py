from __future__ import annotations

import tomllib
from itertools import pairwise
from pathlib import Path

import pytest

from video_demo.speech.asr import (
    CloudAsrWindow,
    RawAsrSegment,
    build_cloud_asr_windows,
    build_speech_segments,
    project_cloud_asr_window,
    remove_adjacent_cloud_asr_duplicates,
)
from video_demo.speech.language import LanguageSpan
from video_demo.speech.vad import SpeechInterval


def _speech_interval(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
) -> SpeechInterval:
    return SpeechInterval(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.9,
    )


def test_cloud_asr_windows_keep_short_vad_intervals_independent() -> None:
    speech = (
        _speech_interval("vad_001", 1_000, 10_000),
        _speech_interval("vad_002", 15_000, 30_000),
    )

    windows = build_cloud_asr_windows(
        speech,
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert windows == tuple(
        CloudAsrWindow(
            upload_range=item,
            owned_range=item,
            speech_interval=item,
        )
        for item in speech
    )


def test_cloud_asr_window_does_not_split_exact_ten_minutes() -> None:
    speech = _speech_interval("vad_exact", 12_345, 612_345)

    windows = build_cloud_asr_windows(
        (speech,),
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert windows == (
        CloudAsrWindow(
            upload_range=speech,
            owned_range=speech,
            speech_interval=speech,
        ),
    )


@pytest.mark.parametrize(
    ("duration_ms", "expected_count"),
    [
        (720_000, 2),
        (1_200_000, 3),
        (1_800_000, 4),
        (720_001, 2),
    ],
)
def test_cloud_asr_windows_balance_long_intervals_with_unique_ownership(
    duration_ms: int,
    expected_count: int,
) -> None:
    start_ms = 12_345
    speech = _speech_interval("vad_long", start_ms, start_ms + duration_ms)

    windows = build_cloud_asr_windows(
        (speech,),
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert len(windows) == expected_count
    assert windows[0].owned_range.start_ms == speech.start_ms
    assert windows[-1].owned_range.end_ms == speech.end_ms
    assert all(window.speech_interval == speech for window in windows)
    assert all(window.upload_range.duration_ms <= 600_000 for window in windows)
    assert all(
        previous.owned_range.end_ms == current.owned_range.start_ms
        for previous, current in pairwise(windows)
    )
    assert all(
        previous.upload_range.end_ms - current.upload_range.start_ms == 1_000
        for previous, current in pairwise(windows)
    )
    owned_durations = [window.owned_range.duration_ms for window in windows]
    assert max(owned_durations) - min(owned_durations) <= 1


def test_cloud_asr_windows_keep_114_regular_intervals_as_114_requests() -> None:
    speech = tuple(
        _speech_interval(f"vad_{index:03d}", index * 2_000, index * 2_000 + 1_000)
        for index in range(114)
    )

    windows = build_cloud_asr_windows(
        speech,
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert len(windows) == 114
    assert tuple(window.speech_interval for window in windows) == speech


@pytest.mark.parametrize(
    "speech",
    [
        (
            _speech_interval("vad_later", 2_000, 3_000),
            _speech_interval("vad_earlier", 0, 1_000),
        ),
        (
            _speech_interval("vad_first", 0, 2_000),
            _speech_interval("vad_overlap", 1_999, 3_000),
        ),
    ],
)
def test_cloud_asr_windows_reject_unordered_or_overlapping_intervals(
    speech: tuple[SpeechInterval, ...],
) -> None:
    with pytest.raises(ValueError, match="有序且不能重叠"):
        build_cloud_asr_windows(
            speech,
            max_window_ms=600_000,
            overlap_ms=1_000,
        )


def test_cloud_asr_projection_keeps_only_midpoints_owned_by_the_window() -> None:
    speech = _speech_interval("vad_long", 10_000, 730_000)
    first, second = build_cloud_asr_windows(
        (speech,),
        max_window_ms=600_000,
        overlap_ms=1_000,
    )
    first_result = project_cloud_asr_window(
        first,
        language="en",
        raw_segments=(
            RawAsrSegment(359_000, 360_400, "owned by first", 0.8),
            RawAsrSegment(359_700, 360_500, "owned by second", 0.9),
        ),
    )
    second_result = project_cloud_asr_window(
        second,
        language="en",
        raw_segments=(
            RawAsrSegment(200, 1_000, "owned by second", 0.9),
            RawAsrSegment(0, 700, "owned by first", 0.8),
        ),
    )

    assert tuple(item.text for item in first_result.segments) == ("owned by first",)
    assert tuple(item.text for item in second_result.segments) == ("owned by second",)
    assert (
        first_result.segments[0].start_ms,
        first_result.segments[0].end_ms,
    ) == (369_000, 370_000)
    assert (
        second_result.segments[0].start_ms,
        second_result.segments[0].end_ms,
    ) == (370_000, 370_500)
    assert first_result.warnings == ("ASR_OVERLAP_TIMESTAMP_CLAMPED",)
    assert second_result.warnings == ("ASR_OVERLAP_TIMESTAMP_CLAMPED",)


def test_cloud_asr_projection_clamps_provider_timestamps_to_upload_before_ownership() -> None:
    speech = _speech_interval("vad_short", 10_000, 20_000)
    window = CloudAsrWindow(
        upload_range=speech,
        owned_range=speech,
        speech_interval=speech,
    )

    result = project_cloud_asr_window(
        window,
        language="zh",
        raw_segments=(
            RawAsrSegment(-200, 500, "起点越界", 0.9),
            RawAsrSegment(9_800, 10_200, "终点越界", 0.8),
            RawAsrSegment(10_100, 10_300, "完全在外", 0.7),
        ),
    )

    assert [(item.start_ms, item.end_ms, item.text) for item in result.segments] == [
        (10_000, 10_500, "起点越界"),
        (19_800, 20_000, "终点越界"),
    ]
    assert result.warnings == ("ASR_TIMESTAMP_CLAMPED", "ASR_TIMESTAMP_DROPPED")


def test_cloud_asr_duplicate_removal_uses_only_exact_normalized_text() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_duplicate",
        start_ms=0,
        end_ms=10_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(0, 1_000, "\uff21\uff29   News", 0.7),
            RawAsrSegment(1_000, 2_000, "ai news", 0.9),
            RawAsrSegment(2_000, 3_000, "AI news today", 0.8),
            RawAsrSegment(3_000, 4_000, "news today", 0.6),
        ),
    )

    deduplicated = remove_adjacent_cloud_asr_duplicates(segments)

    assert [(item.text, item.confidence) for item in deduplicated] == [
        ("ai news", 0.9),
        ("AI news today", 0.8),
        ("news today", 0.6),
    ]


def test_speech_extra_declares_compatible_runtime_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    speech_dependencies = project["project"]["optional-dependencies"]["speech"]

    assert speech_dependencies == [
        "silero-vad>=5.1,<7",
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
    ]


def test_build_asr_segments_preserves_original_language_text_and_absolute_time() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=10_000,
        end_ms=20_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(start_ms=0, end_ms=1_500, text=" Hello world ", confidence=0.85),
        ),
    )

    assert len(segments) == 1
    assert segments[0].start_ms == 10_000
    assert segments[0].end_ms == 11_500
    assert segments[0].text == "Hello world"
    assert segments[0].language == "en"
    assert segments[0].is_fully_evaluated_language is True


def test_build_asr_segments_clamps_one_whisper_timestamp_tick_at_slice_end() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=25_890,
        end_ms=30_000,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=3_000,
                end_ms=4_120,
                text="量化到下一个时间刻度",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(28_890, 30_000)]


def test_build_asr_segments_clamps_timestamp_after_unaligned_slice_end() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_001",
        start_ms=0,
        end_ms=8_668,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=6_140,
                end_ms=8_700,
                text="真实失败切片的末段",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(6_140, 8_668)]


def test_build_asr_segments_clamps_short_slice_overrun() -> None:
    """模型在极短切片上多报 92ms 时，应截断而不是使整段 ASR 失败。"""
    language_span = LanguageSpan(
        evidence_id="lid_short_overrun",
        start_ms=12_000,
        end_ms=13_468,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=0,
                end_ms=1_560,
                text="短窗口末尾时间戳略有越界",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(12_000, 13_468)]


def test_build_asr_segments_clamps_larger_short_slice_overrun() -> None:
    """模型在短切片上多报 140ms 时，也应截断到切片末尾。"""
    language_span = LanguageSpan(
        evidence_id="lid_larger_short_overrun",
        start_ms=20_000,
        end_ms=22_300,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=0,
                end_ms=2_440,
                text="短窗口末尾时间戳越界一百四十毫秒",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(20_000, 22_300)]


def test_build_asr_segments_clamps_three_hundred_ms_overrun() -> None:
    """短切片末尾越界接近 300ms 时，仍应安全截断。"""
    language_span = LanguageSpan(
        evidence_id="lid_three_hundred_ms_overrun",
        start_ms=30_000,
        end_ms=32_300,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    segments = build_speech_segments(
        language_span,
        (
            RawAsrSegment(
                start_ms=0,
                end_ms=2_591,
                text="接近三百毫秒的短窗口越界",
                confidence=0.85,
            ),
        ),
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(30_000, 32_300)]


def test_build_asr_segments_clamps_material_timestamp_overrun_and_records_warning() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_material_overrun",
        start_ms=4_642,
        end_ms=7_198,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    warnings: list[str] = []

    segments = build_speech_segments(
        language_span,
        (RawAsrSegment(0, 3_480, "Machine learning is another hot topic.", 0.85),),
        warnings=warnings,
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(4_642, 7_198)]
    assert warnings == ["ASR_TIMESTAMP_CLAMPED"]


def test_build_asr_segments_clamps_negative_start_and_records_warning() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_negative_start",
        start_ms=10_000,
        end_ms=12_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    warnings: list[str] = []

    segments = build_speech_segments(
        language_span,
        (RawAsrSegment(-250, 500, "起点略早", 0.85),),
        warnings=warnings,
    )

    assert [(item.start_ms, item.end_ms) for item in segments] == [(10_000, 10_500)]
    assert warnings == ["ASR_TIMESTAMP_CLAMPED"]


def test_build_asr_segments_drops_segment_without_valid_overlap() -> None:
    language_span = LanguageSpan(
        evidence_id="lid_no_overlap",
        start_ms=10_000,
        end_ms=12_000,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    warnings: list[str] = []

    segments = build_speech_segments(
        language_span,
        (RawAsrSegment(2_000, 2_500, "窗口外文本", 0.85),),
        warnings=warnings,
    )

    assert segments == ()
    assert warnings == ["ASR_TIMESTAMP_DROPPED"]
