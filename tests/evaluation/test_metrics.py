from __future__ import annotations

import math

import pytest

from video_demo.evaluation.metrics import (
    boundary_f1,
    character_error_rate,
    exact_item_accuracy,
    fact_scores,
    optimal_interval_error_rate,
    percentile_90,
    runtime_resource_metrics,
    valid_result_rate,
    word_error_rate,
)


def test_character_and_word_error_rates_are_hand_calculable() -> None:
    assert math.isclose(character_error_rate("你好世界", "你好世人"), 0.25)
    assert math.isclose(word_error_rate("hello brave world", "hello world"), 0.5)


def test_percentile_90_uses_nearest_rank() -> None:
    assert percentile_90((1, 2, 3, 4, 5, 6, 7, 8, 9, 100)) == 9


def test_optimal_interval_error_rate_uses_global_assignment() -> None:
    reference = ((0, 100), (100, 200))
    hypothesis = ((0, 200),)

    assert math.isclose(optimal_interval_error_rate(reference, hypothesis), 0.5)


def test_boundary_f1_uses_one_to_one_tolerance_matching() -> None:
    score = boundary_f1(
        reference_ms=(1_000, 2_000),
        hypothesis_ms=(900, 1_100, 2_300),
        tolerance_ms=200,
    )

    assert math.isclose(score, 0.4)


def test_fact_support_and_recall_use_set_semantics() -> None:
    scores = fact_scores(
        predicted=("a", "b", "b"),
        supported=("a",),
        key_facts=("a", "c"),
    )

    assert scores.support_rate == 0.5
    assert scores.key_fact_recall == 0.5


def test_key_fact_recall_requires_the_fact_to_be_predicted() -> None:
    scores = fact_scores(
        predicted=("a",),
        supported=("a", "c"),
        key_facts=("a", "c"),
    )

    assert scores.key_fact_recall == 0.5


def test_exact_item_accuracy_covers_ocr_items() -> None:
    assert exact_item_accuracy(("hello", "world"), ("hello", "word")) == 0.5


def test_exact_item_accuracy_normalizes_unicode_and_penalizes_extra_items() -> None:
    assert exact_item_accuracy(("\uff21", "world"), ("a", "world", "extra")) == 2 / 3


def test_schema_time_valid_rate_counts_all_attempted_results() -> None:
    assert valid_result_rate(valid_count=9, attempted_count=10) == 0.9


def test_runtime_metrics_compute_rtf_and_keep_peak_resources() -> None:
    metrics = runtime_resource_metrics(
        video_duration_ms=60_000,
        elapsed_seconds=120.0,
        peak_rss_bytes=2_000,
        peak_disk_bytes=3_000,
    )

    assert metrics.rtf == 2.0
    assert metrics.peak_rss_bytes == 2_000
    assert metrics.peak_disk_bytes == 3_000


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("video_duration_ms", 0),
        ("video_duration_ms", math.nan),
        ("elapsed_seconds", math.inf),
        ("elapsed_seconds", -1.0),
        ("peak_rss_bytes", -1),
        ("peak_disk_bytes", -1),
        ("peak_disk_bytes", math.nan),
    ),
)
def test_runtime_metrics_reject_invalid_inputs(keyword: str, value: int | float) -> None:
    arguments: dict[str, int | float] = {
        "video_duration_ms": 60_000,
        "elapsed_seconds": 30.0,
        "peak_rss_bytes": 2_000,
        "peak_disk_bytes": 3_000,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError):
        runtime_resource_metrics(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("valid_count", "attempted_count"),
    ((-1, 10), (11, 10), (0, 0)),
)
def test_valid_result_rate_rejects_invalid_counts(
    valid_count: int,
    attempted_count: int,
) -> None:
    with pytest.raises(ValueError):
        valid_result_rate(valid_count=valid_count, attempted_count=attempted_count)


def test_edit_counts_support_micro_averaging_without_averaging_sample_rates() -> None:
    from video_demo.evaluation.metrics import character_edit_counts, word_edit_counts

    short = character_edit_counts("甲丙", "甲乙")
    long = character_edit_counts("甲乙丙丁戊己庚辛", "甲乙丙丁戊己庚辛")
    english = word_edit_counts("one three", "one two three")

    assert (short.errors + long.errors) / (
        short.reference_units + long.reference_units
    ) == 0.1
    assert english.errors == 1
    assert english.reference_units == 3


def test_ocr_nfkc_counts_do_not_apply_asr_casefold_or_remove_spaces() -> None:
    from video_demo.evaluation.metrics import nfkc_character_edit_counts

    assert nfkc_character_edit_counts("A", "\uff21").errors == 0
    assert nfkc_character_edit_counts("a", "A").errors == 1
    assert nfkc_character_edit_counts("AB", "A B").reference_units == 3
