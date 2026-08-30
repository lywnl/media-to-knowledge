"""从音频转写证据构建连续的音频基础片段。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from video_demo.application.audio_contracts import (
    AudioEvidencePreparationLimits,
    AudioSpeechBoundaryCandidate,
)
from video_demo.domain.audio_plan import AudioBaseSegment, AudioTranscriptEvidence
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError

_GRID_INTERVAL_MS = 30_000
_SPARSE_SENTENCE_BOUNDARY_LIMIT = 16
_SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS = 60_000
_SENTENCE_BOUNDARY_GRID_TOLERANCE_MS = 8_000
_MAX_SEGMENT_DURATION_MS = 300_000
_MAX_SEGMENT_EVIDENCE_REFS = 256
_EMPTY_RANGE_MAX_MS = 10_000


def build_audio_segments(
    asset_sha256: Sha256,
    duration_ms: int,
    transcript_evidence: Sequence[AudioTranscriptEvidence],
    speech_boundaries: Sequence[AudioSpeechBoundaryCandidate],
    limits: AudioEvidencePreparationLimits,
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
    limits: AudioEvidencePreparationLimits,
) -> tuple[AudioTranscriptEvidence, ...]:
    if len(evidence) > limits.max_transcript_evidence_items:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "转写证据数量超过上限")
    if sum(len(item.text) for item in evidence) > limits.max_transcript_chars:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "转写证据字符数超过上限")
    ordered = tuple(
        sorted(evidence, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
    )
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
    candidates: Sequence[AudioSpeechBoundaryCandidate],
) -> tuple[int, ...]:
    grid_candidates = set(range(_GRID_INTERVAL_MS, duration_ms, _GRID_INTERVAL_MS))
    candidates_set = {
        *grid_candidates,
        *(
            candidate.timestamp_ms
            for candidate in candidates
            if candidate.source != "sentence_end" or candidate.score >= 0.95
        ),
    }
    sentence_ends = tuple(
        candidate.timestamp_ms
        for candidate in candidates
        if candidate.source == "sentence_end" and candidate.score >= 0.8
    )
    selected_sentence_ends = (
        sentence_ends
        if len(sentence_ends) <= _SPARSE_SENTENCE_BOUNDARY_LIMIT
        else _sparse_sentence_boundaries(sentence_ends, duration_ms)
    )
    candidates_set.difference_update(
        {
            grid
            for grid in grid_candidates
            if any(
                abs(grid - sentence_end) <= _SENTENCE_BOUNDARY_GRID_TOLERANCE_MS
                for sentence_end in selected_sentence_ends
            )
        },
    )
    candidates_set.update(selected_sentence_ends)
    ordered_candidates = tuple(
        value for value in sorted(candidates_set) if 0 < value < duration_ms
    )
    safe = _exclude_transcript_interiors(ordered_candidates, transcript)
    safe = _snap_dense_grid_boundaries(safe, ordered_candidates, transcript)
    safe = _add_required_transcript_boundaries(safe, duration_ms, transcript)
    if any(
        end_ms - start_ms > _MAX_SEGMENT_DURATION_MS
        for start_ms, end_ms in pairwise((0, *safe, duration_ms))
    ):
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "缺少可安全切分的音频片段边界")
    return safe


def _sparse_sentence_boundaries(
    sentence_ends: Sequence[int],
    duration_ms: int,
) -> tuple[int, ...]:
    selected: dict[int, tuple[int, int]] = {}
    for timestamp_ms in sentence_ends:
        if not 0 < timestamp_ms < duration_ms:
            continue
        bucket = timestamp_ms // _SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS
        target = min(duration_ms, (bucket + 1) * _SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS)
        distance = abs(timestamp_ms - target)
        if distance > _SENTENCE_BOUNDARY_GRID_TOLERANCE_MS:
            continue
        previous = selected.get(bucket)
        if previous is None or (distance, timestamp_ms) < previous:
            selected[bucket] = (distance, timestamp_ms)
    return tuple(timestamp for _distance, timestamp in selected.values())


def _exclude_transcript_interiors(
    candidates: Sequence[int],
    transcript: Sequence[AudioTranscriptEvidence],
) -> tuple[int, ...]:
    safe: list[int] = []
    transcript_index = 0
    maximum_open_end = 0
    for candidate in candidates:
        while (
            transcript_index < len(transcript)
            and transcript[transcript_index].start_ms < candidate
        ):
            maximum_open_end = max(maximum_open_end, transcript[transcript_index].end_ms)
            transcript_index += 1
        if maximum_open_end <= candidate:
            safe.append(candidate)
    return tuple(safe)


def _snap_dense_grid_boundaries(
    safe: tuple[int, ...],
    candidates: Sequence[int],
    transcript: Sequence[AudioTranscriptEvidence],
) -> tuple[int, ...]:
    if len(transcript) < 10 or not any(isinstance(item, SpeechSegment) for item in transcript):
        return safe
    if all(item.duration_ms >= 20_000 for item in transcript):
        return safe
    safe_set = set(safe)
    endpoints = tuple(
        sorted(
            {
                item.end_ms
                for item in transcript
                if _is_safe_transcript_endpoint(item.end_ms, transcript)
            },
        ),
    )
    for grid in candidates:
        if grid in safe_set or grid % _GRID_INTERVAL_MS:
            continue
        nearest = min(
            (
                (abs(endpoint - grid), endpoint)
                for endpoint in endpoints
                if abs(endpoint - grid) <= _SENTENCE_BOUNDARY_GRID_TOLERANCE_MS
            ),
            default=None,
        )
        if nearest is not None:
            safe_set.add(nearest[1])
    return tuple(sorted(safe_set))


def _add_required_transcript_boundaries(
    safe: tuple[int, ...],
    duration_ms: int,
    transcript: Sequence[AudioTranscriptEvidence],
) -> tuple[int, ...]:
    boundaries = list(safe)
    while True:
        ranges = tuple(pairwise((0, *boundaries, duration_ms)))
        oversized = next(
            ((start_ms, end_ms) for start_ms, end_ms in ranges
             if end_ms - start_ms > _MAX_SEGMENT_DURATION_MS),
            None,
        )
        if oversized is None:
            return tuple(boundaries)
        start_ms, end_ms = oversized
        endpoint = max(
            (
                item.end_ms
                for item in transcript
                if start_ms < item.end_ms <= min(end_ms, start_ms + _MAX_SEGMENT_DURATION_MS)
                and _is_safe_transcript_endpoint(item.end_ms, transcript)
            ),
            default=None,
        )
        if endpoint is None or endpoint in boundaries:
            return tuple(boundaries)
        boundaries.append(endpoint)
        boundaries.sort()


def _is_safe_transcript_endpoint(
    timestamp_ms: int,
    transcript: Sequence[AudioTranscriptEvidence],
) -> bool:
    return not any(item.start_ms < timestamp_ms < item.end_ms for item in transcript)


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
                    {
                        "asset_sha256": asset_sha256,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "evidence_refs": refs,
                    },
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
