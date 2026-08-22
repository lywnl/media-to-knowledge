from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from statistics import median

from video_demo.domain.evidence import KeyframeEvidence, OcrLine

_MINIMUM_LINE_CONFIDENCE = 0.80
_MINIMUM_EFFECTIVE_CHARS = 12
_DENSE_MEDIAN_CHARS = 24
_TEXT_CHANGE_THRESHOLD = 0.65
_SPACE_AND_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")

FixedLineKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class OcrBudget:
    probe: int
    base: int
    hard_limit: int


class OcrClassification(StrEnum):
    LOW_TEXT = "LOW_TEXT"
    NORMAL_TEXT = "NORMAL_TEXT"
    DENSE_TEXT = "DENSE_TEXT"
    INSUFFICIENT_PROBES = "INSUFFICIENT_PROBES"


@dataclass(frozen=True, slots=True)
class OcrFrameObservation:
    timestamp_ms: int
    lines: tuple[OcrLine, ...]
    image_width: int | None
    image_height: int | None


@dataclass(frozen=True, slots=True)
class OcrTextAssessment:
    classification: OcrClassification
    valid_text_ratio: Fraction
    text_change_ratio: Fraction
    median_effective_chars: float
    time_coverage_ratio: Fraction
    frame_texts: tuple[str, ...]
    fixed_line_keys: frozenset[FixedLineKey]


def calculate_ocr_budget(duration_ms: int) -> OcrBudget:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    minutes = duration_ms / 60_000
    root = math.sqrt(minutes)
    probe = max(3, min(12, math.ceil(2.5 * root)))
    base = max(probe, min(36, math.ceil(5.5 * root)))
    hard_limit = max(base, min(50, math.ceil(8.5 * root)))
    return OcrBudget(probe=probe, base=base, hard_limit=hard_limit)


def select_probe_keyframes(
    keyframes: Sequence[KeyframeEvidence],
    *,
    duration_ms: int,
    count: int,
) -> tuple[KeyframeEvidence, ...]:
    if duration_ms < 1 or count < 1:
        raise ValueError("视频时长和探针数量必须大于 0")
    ordered = _ordered_unique_keyframes(keyframes, duration_ms=duration_ms)
    if len(ordered) <= count:
        return ordered
    remaining = list(ordered)
    selected: list[KeyframeEvidence] = []
    for index in range(count):
        target_numerator = (2 * index + 1) * duration_ms
        candidate = min(
            remaining,
            key=lambda item: (
                abs(2 * count * item.timestamp_ms - target_numerator),
                item.timestamp_ms,
                item.keyframe_id,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return tuple(sorted(selected, key=_keyframe_order))


def extend_keyframes(
    keyframes: Sequence[KeyframeEvidence],
    selected: Sequence[KeyframeEvidence],
    *,
    count: int,
) -> tuple[KeyframeEvidence, ...]:
    if count < 0:
        raise ValueError("新增关键帧数量不得小于 0")
    ordered = _ordered_unique_keyframes(keyframes)
    selected_ids = {item.keyframe_id for item in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("已选关键帧不得重复")
    available_ids = {item.keyframe_id for item in ordered}
    if not selected_ids <= available_ids:
        raise ValueError("已选关键帧必须来自候选集合")
    result = list(selected)
    remaining = [item for item in ordered if item.keyframe_id not in selected_ids]
    for _index in range(min(count, len(remaining))):
        if result:
            candidate = min(
                remaining,
                key=lambda item: (
                    -min(abs(item.timestamp_ms - kept.timestamp_ms) for kept in result),
                    item.timestamp_ms,
                    item.keyframe_id,
                ),
            )
        else:
            candidate = remaining[0]
        result.append(candidate)
        remaining.remove(candidate)
    return tuple(result)


def assess_probe_text(
    observations: Sequence[OcrFrameObservation],
    *,
    duration_ms: int,
    has_subtitle_track: bool,
) -> OcrTextAssessment:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    ordered = _ordered_observations(observations, duration_ms=duration_ms)
    if not ordered:
        return OcrTextAssessment(
            classification=OcrClassification.INSUFFICIENT_PROBES,
            valid_text_ratio=Fraction(0, 1),
            text_change_ratio=Fraction(0, 1),
            median_effective_chars=0.0,
            time_coverage_ratio=Fraction(0, 1),
            frame_texts=(),
            fixed_line_keys=frozenset(),
        )
    prepared = tuple(
        _prepared_lines(item, has_subtitle_track=has_subtitle_track)
        for item in ordered
    )
    fixed_keys = _fixed_line_keys(prepared, len(ordered))
    frame_texts = tuple(_frame_text(lines, fixed_keys) for lines in prepared)
    valid = tuple(text for text in frame_texts if len(text) >= _MINIMUM_EFFECTIVE_CHARS)
    valid_ratio = Fraction(len(valid), len(ordered))
    changes = sum(
        _text_is_new(right, (left,))
        for left, right in pairwise(valid)
    )
    change_ratio = Fraction(changes, len(valid) - 1) if len(valid) > 1 else Fraction(0, 1)
    median_chars = float(median(tuple(map(len, valid)))) if valid else 0.0
    covered_thirds = {
        min(2, item.timestamp_ms * 3 // duration_ms)
        for item, text in zip(ordered, frame_texts, strict=True)
        if len(text) >= _MINIMUM_EFFECTIVE_CHARS
    }
    coverage_ratio = Fraction(len(covered_thirds), 3)
    if len(ordered) < 3:
        classification = OcrClassification.INSUFFICIENT_PROBES
    elif valid_ratio < Fraction(1, 3):
        classification = OcrClassification.LOW_TEXT
    elif (
        valid_ratio >= Fraction(2, 3)
        and change_ratio >= Fraction(1, 2)
        and median_chars >= _DENSE_MEDIAN_CHARS
        and coverage_ratio >= Fraction(2, 3)
    ):
        classification = OcrClassification.DENSE_TEXT
    else:
        classification = OcrClassification.NORMAL_TEXT
    return OcrTextAssessment(
        classification=classification,
        valid_text_ratio=valid_ratio,
        text_change_ratio=change_ratio,
        median_effective_chars=median_chars,
        time_coverage_ratio=coverage_ratio,
        frame_texts=frame_texts,
        fixed_line_keys=fixed_keys,
    )


def batch_new_text_count(
    observations: Sequence[OcrFrameObservation],
    *,
    previous_texts: Sequence[str],
    fixed_line_keys: Set[FixedLineKey],
    has_subtitle_track: bool,
) -> int:
    seen = [
        normalized
        for item in previous_texts
        if len(normalized := _normalize_text(item)) >= _MINIMUM_EFFECTIVE_CHARS
    ]
    new_count = 0
    for observation in sorted(observations, key=lambda item: item.timestamp_ms):
        lines = _prepared_lines(observation, has_subtitle_track=has_subtitle_track)
        text = _frame_text(lines, fixed_line_keys)
        if len(text) < _MINIMUM_EFFECTIVE_CHARS or not _text_is_new(text, seen):
            continue
        new_count += 1
        seen.append(text)
    return new_count


def effective_frame_texts(
    observations: Sequence[OcrFrameObservation],
    *,
    fixed_line_keys: Set[FixedLineKey],
    has_subtitle_track: bool,
) -> tuple[str, ...]:
    return tuple(
        _frame_text(
            _prepared_lines(item, has_subtitle_track=has_subtitle_track),
            fixed_line_keys,
        )
        for item in sorted(observations, key=lambda observation: observation.timestamp_ms)
    )


def _ordered_unique_keyframes(
    keyframes: Sequence[KeyframeEvidence],
    *,
    duration_ms: int | None = None,
) -> tuple[KeyframeEvidence, ...]:
    ordered = tuple(sorted(keyframes, key=_keyframe_order))
    if len({item.keyframe_id for item in ordered}) != len(ordered):
        raise ValueError("候选关键帧不得重复")
    if duration_ms is not None and any(
        not 0 <= item.timestamp_ms < duration_ms for item in ordered
    ):
        raise ValueError("候选关键帧必须位于视频时间轴内")
    unique_by_image: dict[str, KeyframeEvidence] = {}
    for item in ordered:
        unique_by_image.setdefault(item.sha256, item)
    return tuple(unique_by_image.values())


def _keyframe_order(item: KeyframeEvidence) -> tuple[int, str]:
    return item.timestamp_ms, item.keyframe_id


def _ordered_observations(
    observations: Sequence[OcrFrameObservation],
    *,
    duration_ms: int,
) -> tuple[OcrFrameObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda item: item.timestamp_ms))
    if any(not 0 <= item.timestamp_ms < duration_ms for item in ordered):
        raise ValueError("OCR 观察必须位于视频时间轴内")
    return ordered


def _prepared_lines(
    observation: OcrFrameObservation,
    *,
    has_subtitle_track: bool,
) -> tuple[tuple[str, FixedLineKey | None], ...]:
    prepared: list[tuple[str, FixedLineKey | None]] = []
    for line in observation.lines:
        if line.confidence < _MINIMUM_LINE_CONFIDENCE:
            continue
        text = _normalize_text(line.text)
        if not text or _is_subtitle_line(line, observation, text, has_subtitle_track):
            continue
        prepared.append((text, _fixed_line_key(line, observation, text)))
    return tuple(prepared)


def _is_subtitle_line(
    line: OcrLine,
    observation: OcrFrameObservation,
    text: str,
    has_subtitle_track: bool,
) -> bool:
    if not has_subtitle_track or len(text) >= _DENSE_MEDIAN_CHARS:
        return False
    if observation.image_height is None or observation.image_height < 1:
        return False
    center_y = line.bounding_box.y + line.bounding_box.height / 2
    return center_y >= observation.image_height * 0.75


def _fixed_line_key(
    line: OcrLine,
    observation: OcrFrameObservation,
    text: str,
) -> FixedLineKey | None:
    if (
        observation.image_width is None
        or observation.image_width < 1
        or observation.image_height is None
        or observation.image_height < 1
    ):
        return None
    center_x = line.bounding_box.x + line.bounding_box.width / 2
    center_y = line.bounding_box.y + line.bounding_box.height / 2
    column = min(2, max(0, int(center_x * 3 / observation.image_width)))
    row = min(2, max(0, int(center_y * 3 / observation.image_height)))
    return text, row, column


def _fixed_line_keys(
    prepared: Sequence[Sequence[tuple[str, FixedLineKey | None]]],
    probe_count: int,
) -> frozenset[FixedLineKey]:
    counts: Counter[FixedLineKey] = Counter()
    for lines in prepared:
        counts.update({key for _text, key in lines if key is not None})
    minimum = math.ceil(probe_count * 2 / 3)
    return frozenset(key for key, count in counts.items() if count >= minimum)


def _frame_text(
    lines: Sequence[tuple[str, FixedLineKey | None]],
    fixed_line_keys: Set[FixedLineKey],
) -> str:
    return "".join(text for text, key in lines if key not in fixed_line_keys)


def _normalize_text(value: str) -> str:
    return _SPACE_AND_PUNCTUATION.sub("", value.casefold())


def _text_is_new(text: str, previous: Sequence[str]) -> bool:
    return all(
        SequenceMatcher(a=known, b=text, autojunk=False).ratio() < _TEXT_CHANGE_THRESHOLD
        for known in previous
    )


__all__ = [
    "FixedLineKey",
    "OcrBudget",
    "OcrClassification",
    "OcrFrameObservation",
    "OcrTextAssessment",
    "assess_probe_text",
    "batch_new_text_count",
    "calculate_ocr_budget",
    "effective_frame_texts",
    "extend_keyframes",
    "select_probe_keyframes",
]
