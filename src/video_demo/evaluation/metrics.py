from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from pydantic import Field

from video_demo.domain.base import FrozenModel

Interval = tuple[int, int]


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


def _validate_intervals(intervals: Sequence[Interval]) -> None:
    for start, end in intervals:
        _require_integer(start, "区间起点")
        _require_integer(end, "区间终点")
        if end <= start:
            raise ValueError("评测区间必须是非空半开区间")


def _overlap_ms(left: Interval, right: Interval) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


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
