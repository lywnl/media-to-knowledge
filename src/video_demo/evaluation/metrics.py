from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

from pydantic import Field

from video_demo.domain.base import FrozenModel

Interval = tuple[int, int]
LabeledInterval = tuple[int, int, str]
TimedText = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class EditCounts:
    errors: int
    reference_units: int


@dataclass(frozen=True, slots=True)
class MatchCounts:
    matches: int
    predicted_units: int
    reference_units: int


@dataclass(frozen=True, slots=True)
class DiarizationCounts:
    error_speaker_ms: int
    reference_speaker_ms: int


@dataclass(frozen=True, slots=True)
class FactScores:
    support_rate: float
    key_fact_recall: float


class RuntimeResourceMetrics(FrozenModel):
    rtf: float = Field(ge=0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(ge=0)
    peak_disk_bytes: int = Field(ge=0)


def character_error_rate(hypothesis: str, reference: str) -> float:
    counts = character_edit_counts(hypothesis, reference)
    return _rate_from_counts(counts)


def word_error_rate(hypothesis: str, reference: str) -> float:
    return _rate_from_counts(word_edit_counts(hypothesis, reference))


def character_edit_counts(hypothesis: str, reference: str) -> EditCounts:
    reference_units = tuple(_normalize_text(reference).replace(" ", ""))
    hypothesis_units = tuple(_normalize_text(hypothesis).replace(" ", ""))
    return EditCounts(
        errors=_edit_distance(hypothesis_units, reference_units),
        reference_units=len(reference_units),
    )


def word_edit_counts(hypothesis: str, reference: str) -> EditCounts:
    reference_units = tuple(_normalize_text(reference).split())
    hypothesis_units = tuple(_normalize_text(hypothesis).split())
    return EditCounts(
        errors=_edit_distance(hypothesis_units, reference_units),
        reference_units=len(reference_units),
    )


def nfkc_character_edit_counts(hypothesis: str, reference: str) -> EditCounts:
    """OCR 字符计数：仅做 NFKC，不折叠大小写，也不删除空白。"""

    reference_units = tuple(unicodedata.normalize("NFKC", reference))
    hypothesis_units = tuple(unicodedata.normalize("NFKC", hypothesis))
    return EditCounts(
        errors=_edit_distance(hypothesis_units, reference_units),
        reference_units=len(reference_units),
    )


def percentile_90(values: Sequence[int | float]) -> float:
    if not values:
        raise ValueError("P90 至少需要一个数值")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("P90 数值必须有限")
    rank = max(1, math.ceil(0.9 * len(ordered)))
    return ordered[rank - 1]


def optimal_interval_error_rate(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> float:
    _validate_intervals(reference)
    _validate_intervals(hypothesis)
    reference_duration = sum(end - start for start, end in reference)
    if reference_duration == 0:
        return 0.0 if not hypothesis else 1.0
    overlaps = [
        [_overlap_ms(reference_interval, hypothesis_interval) for hypothesis_interval in hypothesis]
        for reference_interval in reference
    ]
    matched = _maximum_assignment(overlaps)
    return max(0.0, 1 - matched / reference_duration)


def diarization_error_rate(
    *,
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
) -> float:
    _validate_labeled_intervals(reference)
    _validate_labeled_intervals(hypothesis)
    if not reference:
        raise ValueError("DER 至少需要一个参考说话人区间")

    reference_speakers = sorted({speaker for _start, _end, speaker in reference})
    hypothesis_speakers = sorted({speaker for _start, _end, speaker in hypothesis})
    overlaps = [
        [
            _speaker_overlap_ms(reference, hypothesis, expected, actual)
            for actual in hypothesis_speakers
        ]
        for expected in reference_speakers
    ]
    correctly_attributed_ms = _maximum_assignment(overlaps)
    scored_speaker_ms = _maximum_active_speaker_ms(reference, hypothesis)
    reference_speaker_ms = sum(end - start for start, end, _speaker in reference)
    return (scored_speaker_ms - correctly_attributed_ms) / reference_speaker_ms


def event_macro_f1(
    *,
    reference: Mapping[str, Sequence[Interval]],
    hypothesis: Mapping[str, Sequence[Interval]],
    tolerance_ms: int,
) -> float:
    labels = sorted(set(reference) | set(hypothesis))
    if not labels:
        return 1.0
    return sum(
        match_counts_f1(counts)
        for counts in event_match_counts(
            reference=reference,
            hypothesis=hypothesis,
            tolerance_ms=tolerance_ms,
        ).values()
    ) / len(labels)


def event_match_counts(
    *,
    reference: Mapping[str, Sequence[Interval]],
    hypothesis: Mapping[str, Sequence[Interval]],
    tolerance_ms: int,
) -> dict[str, MatchCounts]:
    labels = sorted(set(reference) | set(hypothesis))
    return {
        label: interval_match_counts(
            reference=reference.get(label, ()),
            hypothesis=hypothesis.get(label, ()),
            tolerance_ms=tolerance_ms,
        )
        for label in labels
    }


def boundary_f1(
    *,
    reference_ms: Sequence[int],
    hypothesis_ms: Sequence[int],
    tolerance_ms: int,
) -> float:
    return match_counts_f1(
        boundary_match_counts(
            reference_ms=reference_ms,
            hypothesis_ms=hypothesis_ms,
            tolerance_ms=tolerance_ms,
        )
    )


def boundary_match_counts(
    *, reference_ms: Sequence[int], hypothesis_ms: Sequence[int], tolerance_ms: int
) -> MatchCounts:
    return MatchCounts(
        matches=_maximum_boundary_matches(reference_ms, hypothesis_ms, tolerance_ms),
        predicted_units=len(hypothesis_ms),
        reference_units=len(reference_ms),
    )


def interval_match_counts(
    *,
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
    tolerance_ms: int,
) -> MatchCounts:
    _validate_intervals(reference)
    _validate_intervals(hypothesis)
    if tolerance_ms < 0:
        raise ValueError("匹配容差不得小于 0")
    matches = _maximum_assignment(
        [
            [
                int(
                    abs(expected_start - actual_start) <= tolerance_ms
                    and abs(expected_end - actual_end) <= tolerance_ms
                )
                for actual_start, actual_end in hypothesis
            ]
            for expected_start, expected_end in reference
        ]
    )
    return MatchCounts(matches, len(hypothesis), len(reference))


def match_counts_f1(counts: MatchCounts) -> float:
    return _f1(counts.matches, counts.predicted_units, counts.reference_units)


def fact_scores(
    *,
    predicted: Sequence[str],
    supported: Sequence[str],
    key_facts: Sequence[str],
) -> FactScores:
    predicted_set = {_normalize_text(value) for value in predicted if _normalize_text(value)}
    supported_set = {_normalize_text(value) for value in supported if _normalize_text(value)}
    key_fact_set = {_normalize_text(value) for value in key_facts if _normalize_text(value)}
    support_rate = (
        len(predicted_set & supported_set) / len(predicted_set) if predicted_set else 1.0
    )
    key_fact_recall = (
        len(key_fact_set & predicted_set & supported_set) / len(key_fact_set)
        if key_fact_set
        else 1.0
    )
    return FactScores(support_rate=support_rate, key_fact_recall=key_fact_recall)


def exact_item_accuracy(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    denominator = max(len(reference), len(hypothesis))
    if denominator == 0:
        return 1.0
    matches = sum(
        _normalize_text(expected) == _normalize_text(actual)
        for expected, actual in zip(reference, hypothesis, strict=False)
    )
    return matches / denominator


def speaker_label_accuracy(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    denominator = max(len(reference), len(hypothesis))
    if denominator == 0:
        return 1.0
    normalized_reference = tuple(_require_label(value) for value in reference)
    normalized_hypothesis = tuple(_require_label(value) for value in hypothesis)
    reference_labels = sorted(set(normalized_reference))
    hypothesis_labels = sorted(set(normalized_hypothesis))
    confusion = [
        [
            sum(
                expected == reference_label and actual == hypothesis_label
                for expected, actual in zip(
                    normalized_reference,
                    normalized_hypothesis,
                    strict=False,
                )
            )
            for hypothesis_label in hypothesis_labels
        ]
        for reference_label in reference_labels
    ]
    return _maximum_assignment(confusion) / denominator


def word_time_errors_ms(
    *,
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
) -> tuple[int, ...]:
    _validate_intervals(reference)
    _validate_intervals(hypothesis)
    if not reference or len(reference) != len(hypothesis):
        raise ValueError("词时间评测要求非空且一一对齐的参考与预测")
    return tuple(
        max(abs(expected_start - actual_start), abs(expected_end - actual_end))
        for (expected_start, expected_end), (actual_start, actual_end) in zip(
            reference,
            hypothesis,
            strict=True,
        )
    )


def aligned_word_time_errors_ms(
    *,
    reference: Sequence[TimedText],
    hypothesis: Sequence[TimedText],
) -> tuple[int, ...]:
    """按文本序列编辑对齐，只返回规范化文字相同的匹配单元时间误差。"""

    _validate_intervals(tuple((start, end) for _text, start, end in reference))
    _validate_intervals(tuple((start, end) for _text, start, end in hypothesis))
    reference_text = tuple(_normalize_text(text) for text, _start, _end in reference)
    hypothesis_text = tuple(_normalize_text(text) for text, _start, _end in hypothesis)
    pairs = _edit_alignment(reference_text, hypothesis_text)
    return tuple(
        max(
            abs(reference[reference_index][1] - hypothesis[hypothesis_index][1]),
            abs(reference[reference_index][2] - hypothesis[hypothesis_index][2]),
        )
        for reference_index, hypothesis_index in pairs
        if reference_text[reference_index] == hypothesis_text[hypothesis_index]
    )


def diarization_error_rates_by_overlap(
    *,
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
) -> tuple[float, float]:
    """按人工参考同时活跃人数，将 DER 拆成非重叠和重叠两个口径。"""

    non_overlap, overlap = diarization_counts_by_overlap(
        reference=reference, hypothesis=hypothesis
    )
    return (
        _diarization_rate_from_counts(non_overlap),
        _diarization_rate_from_counts(overlap),
    )


def diarization_counts_by_overlap(
    *,
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
) -> tuple[DiarizationCounts, DiarizationCounts]:
    """用一次全局 speaker mapping 按参考并发人数累计两个 DER 整数计数。"""

    reference = tuple(dict.fromkeys(reference))
    hypothesis = tuple(dict.fromkeys(hypothesis))
    _validate_labeled_intervals(reference)
    _validate_labeled_intervals(hypothesis)
    if not reference:
        raise ValueError("DER 至少需要一个参考说话人区间")
    reference_speakers = sorted({speaker for _start, _end, speaker in reference})
    hypothesis_speakers = sorted({speaker for _start, _end, speaker in hypothesis})
    overlaps = [
        [
            _speaker_overlap_ms(reference, hypothesis, expected, actual)
            for actual in hypothesis_speakers
        ]
        for expected in reference_speakers
    ]
    mapping = {
        reference_speakers[ref_index]: hypothesis_speakers[hyp_index]
        for ref_index, hyp_index in _maximum_assignment_pairs(overlaps)
        if ref_index < len(reference_speakers) and hyp_index < len(hypothesis_speakers)
    }
    boundaries = sorted(
        {point for start, end, _speaker in (*reference, *hypothesis) for point in (start, end)}
    )
    errors = [0, 0]
    references = [0, 0]
    for start, end in pairwise(boundaries):
        active_reference = {
            speaker for item_start, item_end, speaker in reference if item_start <= start < item_end
        }
        active_hypothesis = {
            speaker
            for item_start, item_end, speaker in hypothesis
            if item_start <= start < item_end
        }
        partition = int(len(active_reference) >= 2)
        correctly_mapped = sum(
            mapping.get(speaker) in active_hypothesis for speaker in active_reference
        )
        duration = end - start
        errors[partition] += duration * (
            max(len(active_reference), len(active_hypothesis)) - correctly_mapped
        )
        references[partition] += duration * len(active_reference)
    return (
        DiarizationCounts(errors[0], references[0]),
        DiarizationCounts(errors[1], references[1]),
    )


def valid_result_rate(*, valid_count: int, attempted_count: int) -> float:
    if attempted_count <= 0:
        raise ValueError("尝试结果数必须大于 0")
    if valid_count < 0 or valid_count > attempted_count:
        raise ValueError("合法结果数必须介于 0 和尝试结果数之间")
    return valid_count / attempted_count


def runtime_resource_metrics(
    *,
    video_duration_ms: int,
    elapsed_seconds: float,
    peak_rss_bytes: int,
    peak_disk_bytes: int,
) -> RuntimeResourceMetrics:
    _require_integer(video_duration_ms, "视频时长", positive=True)
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, Real)
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("处理耗时必须是非负有限数值")
    _require_integer(peak_rss_bytes, "峰值 RSS")
    _require_integer(peak_disk_bytes, "峰值磁盘")
    return RuntimeResourceMetrics(
        rtf=elapsed_seconds / (video_duration_ms / 1_000),
        peak_rss_bytes=peak_rss_bytes,
        peak_disk_bytes=peak_disk_bytes,
    )


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_label(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError("评测标签不得为空")
    return normalized


def _error_rate(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(hypothesis, reference) / len(reference)


def _rate_from_counts(counts: EditCounts) -> float:
    if counts.reference_units == 0:
        return 0.0 if counts.errors == 0 else 1.0
    return counts.errors / counts.reference_units


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                ),
            )
        previous = current
    return previous[-1]


def _edit_alignment(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[tuple[int, int], ...]:
    return _hirschberg(
        reference,
        hypothesis,
        0,
        len(reference),
        0,
        len(hypothesis),
    )


def _hirschberg(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    reference_start: int,
    reference_end: int,
    hypothesis_start: int,
    hypothesis_end: int,
) -> tuple[tuple[int, int], ...]:
    reference_length = reference_end - reference_start
    hypothesis_length = hypothesis_end - hypothesis_start
    if reference_length == 0 or hypothesis_length == 0:
        return ()
    if reference_length == 1 or hypothesis_length == 1:
        return _small_alignment(
            reference,
            hypothesis,
            reference_start,
            reference_end,
            hypothesis_start,
            hypothesis_end,
        )
    reference_middle = reference_start + reference_length // 2
    hypothesis_middle = _canonical_middle_column(
        reference,
        hypothesis,
        reference_start,
        reference_middle,
        reference_end,
        hypothesis_start,
        hypothesis_end,
    )
    return (
        *_hirschberg(
            reference,
            hypothesis,
            reference_start,
            reference_middle,
            hypothesis_start,
            hypothesis_middle,
        ),
        *_hirschberg(
            reference,
            hypothesis,
            reference_middle,
            reference_end,
            hypothesis_middle,
            hypothesis_end,
        ),
    )


def _canonical_middle_column(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    reference_start: int,
    reference_middle: int,
    reference_end: int,
    hypothesis_start: int,
    hypothesis_end: int,
) -> int:
    """线性空间传播规范回溯祖先，保持旧矩阵的对角/删除/插入 tie-break。"""

    hypothesis_length = hypothesis_end - hypothesis_start
    previous = list(range(hypothesis_length + 1))
    for ref_index in range(reference_start, reference_middle):
        relative_ref_index = ref_index - reference_start + 1
        current = [relative_ref_index]
        for relative_hyp_index, hyp_index in enumerate(
            range(hypothesis_start, hypothesis_end), start=1
        ):
            current.append(
                min(
                    current[-1] + 1,
                    previous[relative_hyp_index] + 1,
                    previous[relative_hyp_index - 1]
                    + (reference[ref_index] != hypothesis[hyp_index]),
                )
            )
        previous = current
    previous_origins = list(range(hypothesis_start, hypothesis_end + 1))
    for ref_index in range(reference_middle, reference_end):
        relative_ref_index = ref_index - reference_start + 1
        current = [relative_ref_index]
        current_origins = [previous_origins[0]]
        for relative_hyp_index, hyp_index in enumerate(
            range(hypothesis_start, hypothesis_end), start=1
        ):
            substitution = previous[relative_hyp_index - 1] + (
                reference[ref_index] != hypothesis[hyp_index]
            )
            deletion = previous[relative_hyp_index] + 1
            insertion = current[-1] + 1
            value = min(substitution, deletion, insertion)
            current.append(value)
            if value == substitution:
                current_origins.append(previous_origins[relative_hyp_index - 1])
            elif value == deletion:
                current_origins.append(previous_origins[relative_hyp_index])
            else:
                current_origins.append(current_origins[-1])
        previous = current
        previous_origins = current_origins
    return previous_origins[-1]


def _small_alignment(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    reference_start: int,
    reference_end: int,
    hypothesis_start: int,
    hypothesis_end: int,
) -> tuple[tuple[int, int], ...]:
    if reference_end - reference_start == 1:
        matching = next(
            (
                index
                for index in range(hypothesis_end - 1, hypothesis_start - 1, -1)
                if hypothesis[index] == reference[reference_start]
            ),
            None,
        )
        index = hypothesis_end - 1 if matching is None else matching
        return ((reference_start, index),)
    matching = next(
        (
            index
            for index in range(reference_end - 1, reference_start - 1, -1)
            if reference[index] == hypothesis[hypothesis_start]
        ),
        None,
    )
    index = reference_end - 1 if matching is None else matching
    return ((index, hypothesis_start),)


def _partition_diarization(
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
    *,
    overlap: bool,
) -> tuple[tuple[LabeledInterval, ...], tuple[LabeledInterval, ...]]:
    boundaries = sorted(
        {point for start, end, _speaker in (*reference, *hypothesis) for point in (start, end)}
    )
    selected_reference: list[LabeledInterval] = []
    selected_hypothesis: list[LabeledInterval] = []
    for start, end in pairwise(boundaries):
        reference_count = _active_speaker_count(reference, start)
        if (reference_count >= 2) != overlap:
            continue
        selected_reference.extend(
            (start, end, speaker)
            for item_start, item_end, speaker in reference
            if item_start <= start and end <= item_end
        )
        selected_hypothesis.extend(
            (start, end, speaker)
            for item_start, item_end, speaker in hypothesis
            if item_start <= start and end <= item_end
        )
    return tuple(selected_reference), tuple(selected_hypothesis)


def _diarization_rate_allow_empty(
    reference: Sequence[LabeledInterval], hypothesis: Sequence[LabeledInterval]
) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return diarization_error_rate(reference=reference, hypothesis=hypothesis)


def _diarization_rate_from_counts(counts: DiarizationCounts) -> float:
    if counts.reference_speaker_ms == 0:
        return 0.0 if counts.error_speaker_ms == 0 else 1.0
    return counts.error_speaker_ms / counts.reference_speaker_ms


def _validate_intervals(intervals: Sequence[Interval]) -> None:
    for start, end in intervals:
        _require_integer(start, "区间起点")
        _require_integer(end, "区间终点")
        if end <= start:
            raise ValueError("评测区间必须是非空半开区间")


def _validate_labeled_intervals(intervals: Sequence[LabeledInterval]) -> None:
    _validate_intervals(tuple((start, end) for start, end, _speaker in intervals))
    by_speaker: dict[str, list[Interval]] = {}
    for start, end, speaker in intervals:
        if not speaker:
            raise ValueError("说话人标签不得为空")
        by_speaker.setdefault(speaker, []).append((start, end))
    for speaker_intervals in by_speaker.values():
        ordered = sorted(speaker_intervals)
        if any(left[1] > right[0] for left, right in pairwise(ordered)):
            raise ValueError("同一说话人的评测区间不得重叠")


def _overlap_ms(left: Interval, right: Interval) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _speaker_overlap_ms(
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
    reference_speaker: str,
    hypothesis_speaker: str,
) -> int:
    return sum(
        _overlap_ms((reference_start, reference_end), (hypothesis_start, hypothesis_end))
        for reference_start, reference_end, expected in reference
        for hypothesis_start, hypothesis_end, actual in hypothesis
        if expected == reference_speaker and actual == hypothesis_speaker
    )


def _maximum_active_speaker_ms(
    reference: Sequence[LabeledInterval],
    hypothesis: Sequence[LabeledInterval],
) -> int:
    boundaries = sorted(
        {point for start, end, _speaker in (*reference, *hypothesis) for point in (start, end)}
    )
    total = 0
    for start, end in pairwise(boundaries):
        reference_count = _active_speaker_count(reference, start)
        hypothesis_count = _active_speaker_count(hypothesis, start)
        total += (end - start) * max(reference_count, hypothesis_count)
    return total


def _active_speaker_count(intervals: Sequence[LabeledInterval], timestamp_ms: int) -> int:
    return sum(start <= timestamp_ms < end for start, end, _speaker in intervals)


def _maximum_assignment(overlaps: Sequence[Sequence[int]]) -> int:
    return sum(
        overlaps[row][column]
        for row, column in _maximum_assignment_pairs(overlaps)
        if row < len(overlaps) and column < len(overlaps[row])
    )


def _maximum_assignment_pairs(
    overlaps: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    row_count = len(overlaps)
    column_count = max((len(row) for row in overlaps), default=0)
    if any(len(row) != column_count for row in overlaps):
        raise ValueError("匹配权重矩阵必须为矩形")
    size = max(row_count, column_count)
    if size == 0:
        return ()

    maximum_weight = max((max(row, default=0) for row in overlaps), default=0)
    costs = [
        [
            maximum_weight
            - (overlaps[row][column] if row < row_count and column < column_count else 0)
            for column in range(size)
        ]
        for row in range(size)
    ]
    return _minimum_cost_assignment(costs)


def _minimum_cost_assignment(costs: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    size = len(costs)
    row_potential = [0] * (size + 1)
    column_potential = [0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum_reduced_cost = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced_cost < minimum_reduced_cost[candidate]:
                    minimum_reduced_cost[candidate] = reduced_cost
                    previous_column[candidate] = column
                if minimum_reduced_cost[candidate] < delta:
                    delta = minimum_reduced_cost[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += int(delta)
                    column_potential[candidate] -= int(delta)
                else:
                    minimum_reduced_cost[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = previous_column[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    return tuple((matched_row[column] - 1, column - 1) for column in range(1, size + 1))


def _interval_f1(
    reference: Sequence[Interval],
    hypothesis: Sequence[Interval],
    *,
    tolerance_ms: int,
) -> float:
    return match_counts_f1(
        interval_match_counts(
            reference=reference,
            hypothesis=hypothesis,
            tolerance_ms=tolerance_ms,
        )
    )


def _maximum_boundary_matches(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    tolerance_ms: int,
) -> int:
    if tolerance_ms < 0:
        raise ValueError("匹配容差不得小于 0")
    for value in (*reference, *hypothesis):
        _require_integer(value, "边界时间")
    ordered_reference = sorted(reference)
    ordered_hypothesis = sorted(hypothesis)
    matches = 0
    hypothesis_index = 0
    for expected in ordered_reference:
        while (
            hypothesis_index < len(ordered_hypothesis)
            and ordered_hypothesis[hypothesis_index] < expected - tolerance_ms
        ):
            hypothesis_index += 1
        if (
            hypothesis_index < len(ordered_hypothesis)
            and abs(ordered_hypothesis[hypothesis_index] - expected) <= tolerance_ms
        ):
            matches += 1
            hypothesis_index += 1
    return matches


def _require_integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}必须是整数")
    if value < 0 or (positive and value == 0):
        qualifier = "大于 0" if positive else "不得小于 0"
        raise ValueError(f"{label}{qualifier}")
    return value


def _f1(matches: int, predicted_count: int, reference_count: int) -> float:
    if predicted_count == 0 and reference_count == 0:
        return 1.0
    if matches == 0:
        return 0.0
    precision = matches / predicted_count
    recall = matches / reference_count
    return 2 * precision * recall / (precision + recall)
