from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Self

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, stable_identifier
from video_demo.domain.evidence import OcrEvidence, SpeechSegment, SubtitleCue, TimedEvidence
from video_demo.domain.legacy_result import SegmentUnderstanding, VideoSegment
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.retrieval_text import (
    normalize_retrieval_value,
    render_segment_fields,
)

_HARD_BOUNDARY_SOURCES = frozenset({"scene_hard", "ocr_change"})
_MAX_TITLE_LENGTH = 200
_MAX_SUMMARY_LENGTH = 4_000


class BoundaryPoint(FrozenModel):
    timestamp_ms: int = Field(ge=0)
    sources: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_sources(self) -> Self:
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("边界来源不得重复")
        return self


class WindowUnderstanding(TimeRange):
    window_id: str = Field(min_length=3, max_length=128)
    understanding: SegmentUnderstanding | None = None
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)

    @model_validator(mode="after")
    def require_exactly_one_outcome(self) -> Self:
        if (self.understanding is None) == (self.failure_code is None):
            raise ValueError("窗口必须且只能包含理解结果或失败码")
        return self


def snap_boundary(
    timestamp_ms: int,
    candidates: Sequence[BoundaryPoint],
    *,
    max_distance_ms: int = 300,
) -> int:
    if max_distance_ms < 0:
        raise ValueError("max_distance_ms 不得小于 0")
    if not candidates:
        raise ValueError("候选边界不能为空")
    nearest = min(
        candidates,
        key=lambda item: (abs(item.timestamp_ms - timestamp_ms), item.timestamp_ms),
    )
    if abs(nearest.timestamp_ms - timestamp_ms) > max_distance_ms:
        raise ValueError("没有可吸附的候选边界")
    return nearest.timestamp_ms


def merge_segment_understandings(
    windows: Sequence[WindowUnderstanding],
    *,
    boundaries: Sequence[BoundaryPoint],
    max_snap_distance_ms: int = 300,
    semantic_merge_threshold: float = 0.6,
    evidence: Iterable[TimedEvidence] = (),
    video_title: str = "",
) -> tuple[VideoSegment, ...]:
    if not 0 <= semantic_merge_threshold <= 1:
        raise ValueError("semantic_merge_threshold 必须在 0 到 1 之间")
    boundary_by_time = _canonical_boundaries(boundaries)
    successful = [item for item in windows if item.understanding is not None]
    if not successful:
        raise VideoDemoError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "全部 Qwen 窗口理解失败",
        )
    successful.sort(key=lambda item: (item.start_ms, item.end_ms, item.window_id))
    snapped = [
        _SnappedWindow(
            start_ms=snap_boundary(
                item.start_ms,
                tuple(boundary_by_time.values()),
                max_distance_ms=max_snap_distance_ms,
            ),
            end_ms=snap_boundary(
                item.end_ms,
                tuple(boundary_by_time.values()),
                max_distance_ms=max_snap_distance_ms,
            ),
            understanding=item.understanding,
        )
        for item in successful
        if item.understanding is not None
    ]
    _validate_snapped_windows(snapped)

    merged: list[_SnappedWindow] = []
    for current in snapped:
        if merged and _can_merge(
            merged[-1],
            current,
            boundary_by_time,
            semantic_merge_threshold,
        ):
            merged[-1] = _merge_pair(merged[-1], current)
        else:
            merged.append(current)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    return tuple(
        _build_segment(item, evidence_by_id=evidence_by_id, video_title=video_title)
        for item in merged
    )


class _SnappedWindow(TimeRange):
    understanding: SegmentUnderstanding


def _canonical_boundaries(
    boundaries: Sequence[BoundaryPoint],
) -> dict[int, BoundaryPoint]:
    grouped: dict[int, list[str]] = {}
    for boundary in boundaries:
        grouped.setdefault(boundary.timestamp_ms, []).extend(boundary.sources)
    return {
        timestamp_ms: BoundaryPoint(
            timestamp_ms=timestamp_ms,
            sources=tuple(sorted(set(sources))),
        )
        for timestamp_ms, sources in sorted(grouped.items())
    }


def _validate_snapped_windows(windows: Sequence[_SnappedWindow]) -> None:
    for previous, current in pairwise(windows):
        if previous.end_ms > current.start_ms:
            raise ValueError("吸附后的理解窗口不得重叠")


def _can_merge(
    previous: _SnappedWindow,
    current: _SnappedWindow,
    boundaries: dict[int, BoundaryPoint],
    threshold: float,
) -> bool:
    if previous.end_ms != current.start_ms:
        return False
    boundary = boundaries[current.start_ms]
    if _HARD_BOUNDARY_SOURCES.intersection(boundary.sources):
        return False
    return _semantic_similarity(previous.understanding, current.understanding) >= threshold


def _semantic_similarity(
    left: SegmentUnderstanding,
    right: SegmentUnderstanding,
) -> float:
    comparisons = (
        (left.speakers, right.speakers),
        (left.languages, right.languages),
        (left.topics, right.topics),
        (left.entities, right.entities),
        (left.keywords, right.keywords),
    )
    scores = [_jaccard(first, second) for first, second in comparisons if first or second]
    if normalize_retrieval_value(left.title) == normalize_retrieval_value(right.title):
        scores.append(1.0)
    return sum(scores) / len(scores) if scores else 0.0


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _merge_pair(left: _SnappedWindow, right: _SnappedWindow) -> _SnappedWindow:
    return _SnappedWindow(
        start_ms=left.start_ms,
        end_ms=right.end_ms,
        understanding=SegmentUnderstanding(
            title=_merge_title(left.understanding.title, right.understanding.title),
            summary_zh=_merge_summary(
                left.understanding.summary_zh,
                right.understanding.summary_zh,
            ),
            speakers=_ordered_union(left.understanding.speakers, right.understanding.speakers),
            languages=_ordered_union(left.understanding.languages, right.understanding.languages),
            topics=_ordered_union(left.understanding.topics, right.understanding.topics),
            entities=_ordered_union(left.understanding.entities, right.understanding.entities),
            actions=_ordered_union(left.understanding.actions, right.understanding.actions),
            keywords=_ordered_union(left.understanding.keywords, right.understanding.keywords),
            original_keywords=_ordered_union(
                left.understanding.original_keywords,
                right.understanding.original_keywords,
            ),
            visual_facts=_ordered_union(
                left.understanding.visual_facts,
                right.understanding.visual_facts,
            ),
            evidence_refs=_ordered_union(
                left.understanding.evidence_refs,
                right.understanding.evidence_refs,
            ),
        ),
    )


def _merge_title(left: str, right: str) -> str:
    normalized_left = normalize_retrieval_value(left)
    normalized_right = normalize_retrieval_value(right)
    if normalized_left == normalized_right:
        return normalized_left
    return _truncate_merged_text(
        f"{normalized_left} / {normalized_right}",
        _MAX_TITLE_LENGTH,
    )


def _merge_summary(left: str, right: str) -> str:
    normalized_left = normalize_retrieval_value(left)
    normalized_right = normalize_retrieval_value(right)
    if normalized_left == normalized_right:
        return normalized_left
    sentences: list[str] = []
    for sentence in (*_split_sentences(normalized_left), *_split_sentences(normalized_right)):
        if sentence not in sentences:
            sentences.append(sentence)
    return _truncate_merged_text("".join(sentences), _MAX_SUMMARY_LENGTH)


def _split_sentences(value: str) -> tuple[str, ...]:
    import re

    return tuple(
        part
        for part in re.findall(r"[^。！？!?；;]+[。！？!?；;]?", value)  # noqa: RUF001
        if part
    )


def _truncate_merged_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _ordered_union(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*left, *right)))


def _build_segment(
    window: _SnappedWindow,
    *,
    evidence_by_id: dict[str, TimedEvidence],
    video_title: str,
) -> VideoSegment:
    understanding = window.understanding
    projected = _project_evidence(understanding.evidence_refs, evidence_by_id)
    retrieval_text = render_segment_fields(
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        semantics=understanding,
        video_title=video_title,
        transcript_text=projected[0],
        ocr_text=projected[1],
        visual_facts=understanding.visual_facts,
        transcript_source=projected[2],
    )
    semantic_values = understanding.model_dump(exclude={"visual_facts"})
    return VideoSegment(
        segment_id=stable_identifier(
            "segment",
            {
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
                "evidence_refs": understanding.evidence_refs,
            },
        ),
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        **semantic_values,
        retrieval_text=retrieval_text,
        retrieval_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
        video_title=video_title,
        transcript_text=projected[0],
        ocr_text=projected[1],
        visual_facts=understanding.visual_facts,
        transcript_source=projected[2],
    )


def _project_evidence(
    evidence_refs: Sequence[str],
    evidence_by_id: dict[str, TimedEvidence],
) -> tuple[str, tuple[str, ...], str]:
    items = tuple(
        sorted(
            {
                evidence_by_id[ref]
                for ref in evidence_refs
                if ref in evidence_by_id
            },
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        )
    )
    subtitles = tuple(item for item in items if isinstance(item, SubtitleCue) and item.text.strip())
    speech = tuple(item for item in items if isinstance(item, SpeechSegment) and item.text.strip())
    transcript_items = _remove_overlapping_duplicate_transcripts(subtitles or speech)
    source = "SUBTITLE" if subtitles else "ASR" if speech else "NONE"
    transcript = " ".join(" ".join(item.text.split()) for item in transcript_items)
    ocr_text = tuple(
        " ".join(line.text.split())
        for item in items
        if isinstance(item, OcrEvidence)
        for line in item.lines
        if line.text.strip()
    )
    return transcript, tuple(dict.fromkeys(ocr_text)), source


def _remove_overlapping_duplicate_transcripts(
    items: Sequence[SubtitleCue | SpeechSegment],
) -> tuple[SubtitleCue | SpeechSegment, ...]:
    kept: list[SubtitleCue | SpeechSegment] = []
    for item in items:
        normalized = " ".join(item.text.split())
        if any(
            item.start_ms < previous.end_ms
            and previous.start_ms < item.end_ms
            and normalized == " ".join(previous.text.split())
            for previous in kept
        ):
            continue
        kept.append(item)
    return tuple(kept)
