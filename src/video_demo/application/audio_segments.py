"""从音频转写证据构建连续的音频基础片段。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    SpeechBoundaryCandidate,
)
from video_demo.domain.audio_plan import AudioBaseSegment, AudioTranscriptEvidence
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError

_GRID_INTERVAL_MS = 30_000
_MAX_SEGMENT_DURATION_MS = 300_000
_MAX_SEGMENT_EVIDENCE_REFS = 256
_EMPTY_RANGE_MAX_MS = 10_000


def build_audio_segments(
    asset_sha256: Sha256,
    duration_ms: int,
    transcript_evidence: Sequence[AudioTranscriptEvidence],
    speech_boundaries: Sequence[SpeechBoundaryCandidate],
    limits: EvidencePreparationLimits,
) -> tuple[AudioBaseSegment, ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    transcript = _validate_transcript(transcript_evidence, duration_ms, limits)
    boundaries = _safe_boundaries(duration_ms, transcript, speech_boundaries)
    ranges = _merge_empty_ranges(tuple(pairwise((0, *boundaries, duration_ms))), transcript)
    segments = _assemble_segments(asset_sha256, ranges, transcript)
    if len(segments) > limits.max_base_segments:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频基础片段数量超过上限")
    return segments


def _validate_transcript(
    evidence: Sequence[AudioTranscriptEvidence],
    duration_ms: int,
    limits: EvidencePreparationLimits,
) -> tuple[AudioTranscriptEvidence, ...]:
    if len(evidence) > limits.max_transcript_evidence_items:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "转写证据数量超过上限")
    if sum(len(item.text) for item in evidence) > limits.max_transcript_chars:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "转写证据字符数超过上限")
    ordered = tuple(sorted(evidence, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)))
    ids = tuple(item.evidence_id for item in ordered)
    if len(ids) != len(set(ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "转写证据标识不得重复")
    if any(item.end_ms > duration_ms for item in ordered):
        raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "转写证据超出音频时间轴")
    if ordered and len({type(item) for item in ordered}) != 1:
        raise VideoDemoError(ErrorCode.EVIDENCE_TYPE_MISMATCH, "字幕与 ASR 证据不能混用")
    return ordered


def _safe_boundaries(
    duration_ms: int,
    transcript: Sequence[AudioTranscriptEvidence],
    candidates: Sequence[SpeechBoundaryCandidate],
) -> tuple[int, ...]:
    values = set(range(_GRID_INTERVAL_MS, duration_ms, _GRID_INTERVAL_MS))
    values.update(
        candidate.timestamp_ms
        for candidate in candidates
        if 0 < candidate.timestamp_ms < duration_ms and candidate.score >= 0.8
    )
    safe = tuple(
        value
        for value in sorted(values)
        if not any(item.start_ms < value < item.end_ms for item in transcript)
    )
    boundaries = list(safe)
    while True:
        ranges = tuple(pairwise((0, *boundaries, duration_ms)))
        oversized = next(((start, end) for start, end in ranges if end - start > _MAX_SEGMENT_DURATION_MS), None)
        if oversized is None:
            return tuple(boundaries)
        start_ms, end_ms = oversized
        endpoint = max(
            (
                item.end_ms
                for item in transcript
                if start_ms < item.end_ms <= min(end_ms, start_ms + _MAX_SEGMENT_DURATION_MS)
                and not any(other.start_ms < item.end_ms < other.end_ms for other in transcript)
            ),
            default=None,
        )
        if endpoint is None or endpoint in boundaries:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "缺少可安全切分的音频片段边界")
        boundaries.append(endpoint)
        boundaries.sort()


def _merge_empty_ranges(
    ranges: tuple[tuple[int, int], ...],
    transcript: Sequence[AudioTranscriptEvidence],
) -> tuple[tuple[int, int], ...]:
    if not transcript:
        return ranges
    normalized = list(ranges)
    while len(normalized) > 1:
        index = next(
            (
                position
                for position, (start, end) in enumerate(normalized)
                if position > 0
                and end - start <= _EMPTY_RANGE_MAX_MS
                and not any(start <= item.start_ms and item.end_ms <= end for item in transcript)
            ),
            None,
        )
        if index is None:
            break
        _start, end = normalized.pop(index)
        previous_start, _previous_end = normalized[index - 1]
        normalized[index - 1] = (previous_start, end)
    return tuple(normalized)


def _assemble_segments(
    asset_sha256: Sha256,
    ranges: Sequence[tuple[int, int]],
    transcript: Sequence[AudioTranscriptEvidence],
) -> tuple[AudioBaseSegment, ...]:
    segments: list[AudioBaseSegment] = []
    index = 0
    for start_ms, end_ms in ranges:
        refs: list[str] = []
        while index < len(transcript) and transcript[index].start_ms < end_ms:
            item = transcript[index]
            if item.start_ms < start_ms or item.end_ms > end_ms:
                raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "转写证据跨越音频片段边界")
            refs.append(item.evidence_id)
            index += 1
        if len(refs) > _MAX_SEGMENT_EVIDENCE_REFS:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "单个音频片段转写证据超过 256 条")
        source = "NONE"
        if refs:
            source = "ASR" if isinstance(transcript[0], SpeechSegment) else "SUBTITLE"
        segments.append(
            AudioBaseSegment(
                segment_id=stable_identifier(
                    "audio_segment",
                    {"asset_sha256": asset_sha256, "start_ms": start_ms, "end_ms": end_ms, "evidence_refs": refs},
                ),
                start_ms=start_ms,
                end_ms=end_ms,
                evidence_refs=tuple(refs),
                transcript_source=source,
            ),
        )
    if index != len(transcript):
        raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "存在未归属到音频片段的转写证据")
    return tuple(segments)
