from __future__ import annotations

import math
import tracemalloc
from collections.abc import Sequence

import pytest

from video_demo.evaluation.metrics import (
    boundary_f1,
    character_error_rate,
    diarization_error_rate,
    event_macro_f1,
    exact_item_accuracy,
    fact_scores,
    optimal_interval_error_rate,
    percentile_90,
    runtime_resource_metrics,
    speaker_label_accuracy,
    valid_result_rate,
    word_error_rate,
    word_time_errors_ms,
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


def test_event_macro_f1_averages_each_label_including_misses() -> None:
    score = event_macro_f1(
        reference={"music": ((0, 100),), "applause": ((100, 200),)},
        hypothesis={"music": ((0, 100),), "applause": (), "alarm": ((0, 50),)},
        tolerance_ms=0,
    )

    assert math.isclose(score, 1 / 3)


def test_event_macro_f1_requires_both_boundaries_within_tolerance() -> None:
    score = event_macro_f1(
        reference={"music": ((0, 1_000),)},
        hypothesis={"music": ((0, 1_500),)},
        tolerance_ms=100,
    )

    assert score == 0.0


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


def test_exact_item_accuracy_covers_ocr_and_speaker_labels() -> None:
    assert exact_item_accuracy(("hello", "world"), ("hello", "word")) == 0.5


def test_exact_item_accuracy_normalizes_unicode_and_penalizes_extra_items() -> None:
    assert exact_item_accuracy(("\uff21", "world"), ("a", "world", "extra")) == 2 / 3


def test_speaker_label_accuracy_uses_global_anonymous_mapping() -> None:
    score = speaker_label_accuracy(
        ("speaker-a", "speaker-a", "speaker-b", "speaker-b"),
        ("anonymous-2", "anonymous-2", "anonymous-1", "anonymous-2"),
    )

    assert score == 0.75


def test_diarization_error_rate_uses_global_speaker_mapping() -> None:
    score = diarization_error_rate(
        reference=((0, 1_000, "speaker-a"), (1_000, 2_000, "speaker-b")),
        hypothesis=(
            (0, 1_000, "anonymous-1"),
            (1_000, 1_500, "anonymous-2"),
            (1_500, 2_000, "anonymous-1"),
        ),
    )

    assert score == 0.25


def test_diarization_error_rate_counts_overlap_misses() -> None:
    score = diarization_error_rate(
        reference=((0, 1_000, "speaker-a"), (500, 1_000, "speaker-b")),
        hypothesis=((0, 1_000, "anonymous-1"), (500, 750, "anonymous-2")),
    )

    assert math.isclose(score, 1 / 6)


def test_diarization_error_rate_counts_pure_miss_and_false_alarm() -> None:
    assert diarization_error_rate(
        reference=((0, 1_000, "speaker-a"),),
        hypothesis=((0, 500, "anonymous-1"),),
    ) == 0.5
    assert diarization_error_rate(
        reference=((0, 1_000, "speaker-a"),),
        hypothesis=((0, 1_500, "anonymous-1"),),
    ) == 0.5


def test_diarization_error_rate_rejects_empty_reference() -> None:
    with pytest.raises(ValueError, match="参考说话人"):
        diarization_error_rate(reference=(), hypothesis=())


def test_word_time_errors_use_largest_boundary_difference_per_aligned_word() -> None:
    errors = word_time_errors_ms(
        reference=((0, 500), (600, 1_000)),
        hypothesis=((100, 450), (650, 1_300)),
    )

    assert errors == (100, 300)


def test_word_time_errors_require_equal_non_empty_alignment() -> None:
    with pytest.raises(ValueError, match="一一对齐"):
        word_time_errors_ms(reference=((0, 500),), hypothesis=())


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


def test_interval_metrics_reject_invalid_event_intervals() -> None:
    with pytest.raises(ValueError, match="半开区间"):
        event_macro_f1(
            reference={"music": ((100, 100),)},
            hypothesis={},
            tolerance_ms=0,
        )


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


def test_textual_word_alignment_penalizes_deletions_but_times_only_matching_words() -> None:
    from video_demo.evaluation.metrics import aligned_word_time_errors_ms, word_edit_counts

    errors = aligned_word_time_errors_ms(
        reference=(("one", 0, 400), ("two", 500, 900), ("three", 1_000, 1_400)),
        hypothesis=(("one", 100, 350), ("three", 1_050, 1_700)),
    )

    assert errors == (100, 300)
    assert word_edit_counts("one three", "one two three").errors == 1


def test_diarization_split_uses_reference_concurrency_atoms() -> None:
    from video_demo.evaluation.metrics import diarization_error_rates_by_overlap

    non_overlap, overlap = diarization_error_rates_by_overlap(
        reference=((0, 1_000, "a"), (500, 1_000, "b")),
        hypothesis=((0, 1_000, "x"),),
    )

    assert non_overlap == 0.0
    assert overlap == 0.5


def test_ocr_nfkc_counts_do_not_apply_asr_casefold_or_remove_spaces() -> None:
    from video_demo.evaluation.metrics import nfkc_character_edit_counts

    assert nfkc_character_edit_counts("A", "\uff21").errors == 0
    assert nfkc_character_edit_counts("a", "A").errors == 1
    assert nfkc_character_edit_counts("AB", "A B").reference_units == 3


def test_diarization_counts_use_one_global_mapping_before_partitioning() -> None:
    from video_demo.evaluation.metrics import diarization_counts_by_overlap

    reference = ((0, 3_000, "a"), (2_000, 3_000, "b"))
    hypothesis = (
        (0, 2_000, "x"),
        (2_000, 3_000, "y"),
        (2_000, 3_000, "z"),
    )

    non_overlap, overlap = diarization_counts_by_overlap(
        reference=reference,
        hypothesis=hypothesis,
    )

    assert diarization_error_rate(reference=reference, hypothesis=hypothesis) == 0.25
    assert (non_overlap.error_speaker_ms, non_overlap.reference_speaker_ms) == (0, 2_000)
    assert (overlap.error_speaker_ms, overlap.reference_speaker_ms) == (1_000, 2_000)


def test_diarization_counts_keep_integer_denominators_for_cross_sample_micro_average() -> None:
    from video_demo.evaluation.metrics import diarization_counts_by_overlap

    short, _ = diarization_counts_by_overlap(
        reference=((0, 1_000, "a"),),
        hypothesis=(),
    )
    long, _ = diarization_counts_by_overlap(
        reference=((0, 9_000, "a"),),
        hypothesis=((0, 9_000, "x"),),
    )

    assert (short.error_speaker_ms + long.error_speaker_ms) / (
        short.reference_speaker_ms + long.reference_speaker_ms
    ) == 0.1


def test_diarization_counts_deduplicate_same_prediction_label_inside_atom() -> None:
    from video_demo.evaluation.metrics import diarization_counts_by_overlap

    non_overlap, overlap = diarization_counts_by_overlap(
        reference=((0, 1_000, "a"),),
        hypothesis=((0, 1_000, "x"), (0, 1_000, "x")),
    )

    assert (non_overlap.error_speaker_ms, non_overlap.reference_speaker_ms) == (0, 1_000)
    assert (overlap.error_speaker_ms, overlap.reference_speaker_ms) == (0, 0)


def test_word_alignment_uses_linear_space_for_long_exact_sequence() -> None:
    from video_demo.evaluation.metrics import aligned_word_time_errors_ms

    words = tuple((f"word-{index}", index * 2, index * 2 + 1) for index in range(1_000))
    tracemalloc.start()
    try:
        errors = aligned_word_time_errors_ms(reference=words, hypothesis=words)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert errors == (0,) * 1_000
    assert peak < 8_000_000


def test_linear_word_alignment_preserves_existing_duplicate_word_tie_break() -> None:
    from video_demo.evaluation.metrics import aligned_word_time_errors_ms

    assert aligned_word_time_errors_ms(
        reference=(("a", 100, 200),),
        hypothesis=(("a", 0, 50), ("a", 110, 210)),
    ) == (10,)


class _NoSliceSequence(Sequence[str]):
    def __init__(self, values: tuple[str, ...]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> str:
        if isinstance(index, slice):
            raise AssertionError("词对齐内部不得复制序列切片")
        return self._values[index]


def test_edit_alignment_never_slices_input_sequences() -> None:
    from video_demo.evaluation.metrics import _edit_alignment

    reference = tuple("a" if index == 63 else "r" for index in range(64))
    hypothesis = tuple("a" if index == 991 else "h" for index in range(1_024))

    assert _edit_alignment(
        _NoSliceSequence(reference), _NoSliceSequence(hypothesis)
    ) == _edit_alignment(reference, hypothesis)


def test_word_alignment_uses_linear_space_for_skewed_low_similarity_sequences() -> None:
    from video_demo.evaluation.metrics import aligned_word_time_errors_ms

    reference = tuple((f"r-{index}", index * 2, index * 2 + 1) for index in range(64))
    hypothesis = tuple((f"h-{index}", index * 2, index * 2 + 1) for index in range(1_024))
    tracemalloc.start()
    try:
        errors = aligned_word_time_errors_ms(reference=reference, hypothesis=hypothesis)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert errors == ()
    assert peak < 8_000_000
