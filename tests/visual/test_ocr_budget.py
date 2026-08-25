from __future__ import annotations

from fractions import Fraction

import pytest

from video_demo.domain.evidence import BoundingBox, KeyframeEvidence, OcrLine
from video_demo.visual.ocr_budget import (
    OcrClassification,
    OcrFrameObservation,
    assess_probe_text,
    batch_new_text_count,
    calculate_ocr_budget,
    effective_frame_texts,
    extend_keyframes,
    select_probe_keyframes,
)


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (30_000, (3, 3, 6)),
        (60_000, (3, 5, 7)),
        (180_000, (4, 8, 12)),
        (300_000, (5, 10, 15)),
        (600_000, (6, 14, 21)),
        (1_200_000, (9, 19, 30)),
        (1_800_000, (9, 24, 36)),
        (921_400, (8, 17, 26)),
    ],
)
def test_ocr_budget_scales_original_square_root_tiers_down_by_about_25_percent(
    duration_ms: int,
    expected: tuple[int, int, int],
) -> None:
    budget = calculate_ocr_budget(duration_ms)

    assert (budget.probe, budget.base, budget.hard_limit) == expected


def test_ocr_budget_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="视频时长"):
        calculate_ocr_budget(0)


def test_probe_selection_covers_full_timeline_deterministically() -> None:
    keyframes = tuple(_keyframe(timestamp_ms) for timestamp_ms in range(5_000, 100_000, 10_000))

    selected = select_probe_keyframes(keyframes, duration_ms=100_000, count=3)

    assert [item.timestamp_ms for item in selected] == [15_000, 45_000, 85_000]


def test_probe_selection_never_duplicates_when_candidates_are_fewer_than_budget() -> None:
    keyframes = (_keyframe(20_000), _keyframe(80_000))

    selected = select_probe_keyframes(keyframes, duration_ms=100_000, count=6)

    assert selected == keyframes


def test_probe_selection_deduplicates_identical_images_across_timestamps() -> None:
    first = _keyframe(20_000)
    duplicate = _keyframe(80_000).model_copy(update={"sha256": first.sha256})

    selected = select_probe_keyframes(
        (first, duplicate),
        duration_ms=100_000,
        count=3,
    )

    assert selected == (first,)


def test_keyframe_extension_keeps_probe_prefix_and_fills_largest_time_gaps() -> None:
    keyframes = tuple(_keyframe(timestamp_ms) for timestamp_ms in (0, 20, 40, 60, 80, 99))
    probes = (_keyframe(20), _keyframe(80))

    extended = extend_keyframes(keyframes, probes, count=3)

    assert extended[:2] == probes
    assert [item.timestamp_ms for item in extended[2:]] == [0, 40, 60]
    assert len({item.keyframe_id for item in extended}) == 5


def test_dense_text_requires_all_four_value_metrics() -> None:
    observations = (
        _observation(10_000, "第一页课程内容包含足够多的有效文字并完整说明基础背景知识"),
        _observation(50_000, "第二页展示另一组完全不同的重要知识以及详细业务实现步骤"),
        _observation(90_000, "第三页继续说明新的架构实现与业务价值并给出最终验收结论"),
    )

    assessment = assess_probe_text(
        observations,
        duration_ms=100_000,
        has_subtitle_track=False,
    )

    assert assessment.classification == OcrClassification.DENSE_TEXT
    assert assessment.valid_text_ratio == Fraction(1, 1)
    assert assessment.text_change_ratio == Fraction(1, 1)
    assert assessment.median_effective_chars >= 24
    assert assessment.time_coverage_ratio == Fraction(1, 1)


def test_low_text_stops_below_one_third_effective_frames() -> None:
    observations = (
        _observation(10_000, "短字"),
        _observation(30_000, ""),
        _observation(50_000, "只有这一帧包含足够长且有效的画面文字内容"),
        _observation(70_000, "标点", confidence=0.79),
    )

    assessment = assess_probe_text(
        observations,
        duration_ms=80_000,
        has_subtitle_track=False,
    )

    assert assessment.classification == OcrClassification.LOW_TEXT
    assert assessment.valid_text_ratio == Fraction(1, 4)


def test_normal_text_meets_minimum_value_but_not_dense_conditions() -> None:
    observations = (
        _observation(10_000, "同一段画面文字内容已经达到有效长度标准第一版"),
        _observation(40_000, "同一段画面文字内容已经达到有效长度标准第二版"),
        _observation(70_000, ""),
    )

    assessment = assess_probe_text(
        observations,
        duration_ms=90_000,
        has_subtitle_track=False,
    )

    assert assessment.classification == OcrClassification.NORMAL_TEXT
    assert assessment.valid_text_ratio == Fraction(2, 3)
    assert assessment.text_change_ratio == Fraction(0, 1)


def test_fixed_ui_repeated_in_two_thirds_of_probes_does_not_create_text_value() -> None:
    observations = (
        _observation(10_000, "固定课程平台导航文字", x=20, y=20),
        _observation(40_000, "固定课程平台导航文字", x=20, y=20),
        _observation(70_000, "这一页有独立且足够长度的正文内容信息", x=200, y=200),
    )

    assessment = assess_probe_text(
        observations,
        duration_ms=90_000,
        has_subtitle_track=False,
    )

    assert assessment.classification == OcrClassification.NORMAL_TEXT
    assert assessment.valid_text_ratio == Fraction(1, 3)
    assert len(assessment.fixed_line_keys) == 1
    assert assessment.frame_texts[:2] == ("", "")


def test_bottom_short_text_is_ignored_only_when_valid_subtitle_track_exists() -> None:
    subtitle_like = tuple(
        _observation(timestamp_ms, text, y=800)
        for timestamp_ms, text in zip(
            (10_000, 40_000, 70_000),
            (
                "这是画面底部出现的第一句中文字幕",
                "接下来画面切换为完全不同的第二句字幕",
                "最后在画面底部展示第三句字幕文本内容",
            ),
            strict=True,
        )
    )

    with_subtitle = assess_probe_text(
        subtitle_like,
        duration_ms=90_000,
        has_subtitle_track=True,
    )
    without_subtitle = assess_probe_text(
        subtitle_like,
        duration_ms=90_000,
        has_subtitle_track=False,
    )

    assert with_subtitle.classification == OcrClassification.LOW_TEXT
    assert with_subtitle.valid_text_ratio == Fraction(0, 1)
    assert without_subtitle.valid_text_ratio == Fraction(1, 1)


def test_three_probe_minimum_prevents_dense_classification_when_candidates_are_sparse() -> None:
    assessment = assess_probe_text(
        (
            _observation(10_000, "第一页包含大量完全不同且有价值的文字信息"),
            _observation(80_000, "第二页同样包含另一批重要业务知识内容"),
        ),
        duration_ms=90_000,
        has_subtitle_track=False,
    )

    assert assessment.classification == OcrClassification.INSUFFICIENT_PROBES


def test_batch_must_produce_two_new_pages_to_justify_another_batch() -> None:
    previous = ("已经见过的课程页面文字内容",)
    batch = (
        _observation(20_000, "已经见过的课程页面文字内容"),
        _observation(30_000, "第一张此前没有见过的新页面知识内容"),
        _observation(40_000, "第二张此前没有见过且完全不同的业务信息"),
    )

    count = batch_new_text_count(
        batch,
        previous_texts=previous,
        fixed_line_keys=frozenset(),
        has_subtitle_track=False,
    )

    assert count == 2
    assert effective_frame_texts(
        batch,
        fixed_line_keys=frozenset(),
        has_subtitle_track=False,
    ) == (
        "已经见过的课程页面文字内容",
        "第一张此前没有见过的新页面知识内容",
        "第二张此前没有见过且完全不同的业务信息",
    )


def test_invalid_short_previous_text_does_not_suppress_new_valid_page() -> None:
    count = batch_new_text_count(
        (_observation(20_000, "甲" * 12),),
        previous_texts=("甲" * 11,),
        fixed_line_keys=frozenset(),
        has_subtitle_track=False,
    )

    assert count == 1


def _keyframe(timestamp_ms: int) -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id=f"keyframe_evidence_{timestamp_ms}",
        start_ms=0,
        end_ms=max(1, timestamp_ms + 1),
        keyframe_id=f"keyframe_{timestamp_ms}",
        timestamp_ms=timestamp_ms,
        relative_path=f"visual/keyframes/frame_{timestamp_ms}.jpg",
        mime_type="image/jpeg",
        sha256=f"{timestamp_ms:064x}",
        perceptual_hash=f"{timestamp_ms:016x}",
    )


def _observation(
    timestamp_ms: int,
    text: str,
    *,
    confidence: float = 0.95,
    x: int = 100,
    y: int = 100,
) -> OcrFrameObservation:
    lines = (
        (
            OcrLine(
                text=text,
                bounding_box=BoundingBox(x=x, y=y, width=300, height=40),
                confidence=confidence,
            ),
        )
        if text
        else ()
    )
    return OcrFrameObservation(
        timestamp_ms=timestamp_ms,
        lines=lines,
        image_width=1_000,
        image_height=1_000,
    )
