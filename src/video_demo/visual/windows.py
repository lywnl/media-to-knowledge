from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from video_demo.domain.run import TimeRange


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    timestamp_ms: int
    source: str
    score: float


_SOURCE_WEIGHTS = {
    "sentence_end": 1.0,
    "silence": 1.0,
    "language_change": 1.0,
    "scene": 0.45,
    "ocr_change": 1.0,
}
_SOURCE_PRIORITY = {
    "silence": 0,
    "language_change": 1,
    "sentence_end": 2,
    "ocr_change": 3,
    "scene": 4,
}


def build_provisional_windows(
    *,
    duration_ms: int,
    candidates: Sequence[BoundaryCandidate],
    min_window_ms: int = 2_000,
    max_window_ms: int = 30_000,
    cluster_tolerance_ms: int = 300,
    boundary_threshold: float = 0.9,
) -> tuple[TimeRange, ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    clusters = _cluster_candidates(candidates, duration_ms, cluster_tolerance_ms)
    accepted = [
        _representative(cluster)
        for cluster in clusters
        if _cluster_score(cluster) >= boundary_threshold
    ]
    boundaries = _apply_window_limits(
        duration_ms,
        accepted,
        min_window_ms=min_window_ms,
        max_window_ms=max_window_ms,
    )
    return _boundaries_to_windows(duration_ms, boundaries)


def refine_windows_with_ocr(
    provisional: Sequence[TimeRange],
    *,
    ocr_changes_ms: Sequence[int],
    duration_ms: int,
    min_window_ms: int = 2_000,
) -> tuple[TimeRange, ...]:
    if not provisional:
        raise ValueError("provisional windows 不能为空")
    boundaries = {window.end_ms for window in provisional[:-1]}
    for change in ocr_changes_ms:
        if 0 < change < duration_ms:
            boundaries.add(change)
    ordered = _drop_too_close_boundaries(
        sorted(boundaries),
        duration_ms,
        min_window_ms,
    )
    return _boundaries_to_windows(duration_ms, ordered)


def merge_adjacent_windows(
    windows: Sequence[TimeRange],
    *,
    max_window_ms: int = 30_000,
) -> tuple[TimeRange, ...]:
    """只沿已有边界贪心合并，生成连续且受限的全片理解窗口。"""

    if not windows or max_window_ms < 1:
        raise ValueError("窗口和上限必须有效")
    merged: list[TimeRange] = []
    current = windows[0]
    if current.duration_ms > max_window_ms:
        raise ValueError("源窗口超过合并上限")
    for window in windows[1:]:
        if window.start_ms != current.end_ms:
            raise ValueError("源窗口必须连续且有序")
        if window.duration_ms > max_window_ms:
            raise ValueError("源窗口超过合并上限")
        if window.end_ms - current.start_ms <= max_window_ms:
            current = TimeRange(start_ms=current.start_ms, end_ms=window.end_ms)
            continue
        merged.append(current)
        current = window
    merged.append(current)
    return tuple(merged)


def _cluster_candidates(
    candidates: Sequence[BoundaryCandidate],
    duration_ms: int,
    tolerance_ms: int,
) -> list[list[BoundaryCandidate]]:
    ordered = sorted(
        (candidate for candidate in candidates if 0 < candidate.timestamp_ms < duration_ms),
        key=lambda item: item.timestamp_ms,
    )
    clusters: list[list[BoundaryCandidate]] = []
    for candidate in ordered:
        if clusters and candidate.timestamp_ms - clusters[-1][-1].timestamp_ms <= tolerance_ms:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _cluster_score(cluster: Sequence[BoundaryCandidate]) -> float:
    by_source: dict[str, float] = {}
    for candidate in cluster:
        weight = _SOURCE_WEIGHTS.get(candidate.source)
        if weight is None:
            raise ValueError(f"未知边界来源: {candidate.source}")
        by_source[candidate.source] = max(
            by_source.get(candidate.source, 0.0),
            weight * candidate.score,
        )
    return sum(by_source.values())


def _representative(cluster: Sequence[BoundaryCandidate]) -> int:
    ranked = sorted(
        cluster,
        key=lambda item: (
            _SOURCE_PRIORITY[item.source],
            -_SOURCE_WEIGHTS[item.source] * item.score,
            item.timestamp_ms,
        ),
    )
    return ranked[0].timestamp_ms


def _apply_window_limits(
    duration_ms: int,
    accepted: Sequence[int],
    *,
    min_window_ms: int,
    max_window_ms: int,
) -> list[int]:
    boundaries: list[int] = []
    cursor = 0
    for candidate in sorted(set(accepted)):
        while candidate - cursor > max_window_ms:
            fallback = cursor + max_window_ms
            boundaries.append(fallback)
            cursor = fallback
        if candidate - cursor >= min_window_ms and duration_ms - candidate >= min_window_ms:
            boundaries.append(candidate)
            cursor = candidate
    while duration_ms - cursor > max_window_ms:
        fallback = cursor + max_window_ms
        boundaries.append(fallback)
        cursor = fallback
    return _drop_too_close_boundaries(boundaries, duration_ms, min_window_ms)


def _drop_too_close_boundaries(
    boundaries: Sequence[int],
    duration_ms: int,
    min_window_ms: int,
) -> list[int]:
    accepted: list[int] = []
    cursor = 0
    for boundary in sorted(set(boundaries)):
        if boundary - cursor < min_window_ms:
            continue
        if duration_ms - boundary < min_window_ms:
            continue
        accepted.append(boundary)
        cursor = boundary
    return accepted


def _boundaries_to_windows(duration_ms: int, boundaries: Sequence[int]) -> tuple[TimeRange, ...]:
    points = [0, *boundaries, duration_ms]
    return tuple(
        TimeRange(start_ms=start, end_ms=end)
        for start, end in pairwise(points)
    )
