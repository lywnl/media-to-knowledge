from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    SpeechBoundaryCandidate,
)
from video_demo.domain.base import Sha256, stable_identifier
from video_demo.domain.document import TranscriptSource
from video_demo.domain.document_plan import BaseSegment
from video_demo.domain.evidence import SceneBoundary, SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError

_GRID_INTERVAL_MS = 30_000
_MAX_SEGMENT_DURATION_MS = 300_000
_MAX_SEGMENT_EVIDENCE_REFS = 256
_MAX_SEGMENT_SCENE_REFS = 256


def build_base_segments(
    asset_sha256: Sha256,
    duration_ms: int,
    transcript_evidence: Sequence[SpeechSegment | SubtitleCue],
    scenes: Sequence[SceneBoundary],
    speech_boundaries: Sequence[SpeechBoundaryCandidate],
    limits: EvidencePreparationLimits,
) -> tuple[BaseSegment, ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    transcript = _validate_transcript_budget(transcript_evidence, duration_ms, limits)
    ordered_scenes = _validate_scenes(scenes, duration_ms, limits)
    boundaries = _safe_boundaries(
        duration_ms,
        transcript,
        ordered_scenes,
        speech_boundaries,
    )
    ranges = tuple(pairwise((0, *boundaries, duration_ms)))
    segments = _assemble_segments(
        asset_sha256,
        ranges,
        transcript,
        ordered_scenes,
    )
    if len(segments) > limits.max_base_segments:
        raise _budget_error("基础片段数量超过上限")
    return segments


def _validate_transcript_budget(
    evidence: Sequence[SpeechSegment | SubtitleCue],
    duration_ms: int,
    limits: EvidencePreparationLimits,
) -> tuple[SpeechSegment | SubtitleCue, ...]:
    if len(evidence) > limits.max_transcript_evidence_items:
        raise _budget_error("转写证据数量超过上限")
    if sum(len(item.text) for item in evidence) > limits.max_transcript_chars:
        raise _budget_error("转写证据字符数超过上限")
    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    ids = [item.evidence_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "转写证据标识不得重复")
    if any(item.end_ms > duration_ms for item in ordered):
        raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "转写证据超出视频时间轴")
    if ordered and len({type(item) for item in ordered}) != 1:
        raise VideoDemoError(ErrorCode.EVIDENCE_TYPE_MISMATCH, "字幕与 ASR 证据不能混用")
    return ordered


def _validate_scenes(
    scenes: Sequence[SceneBoundary],
    duration_ms: int,
    limits: EvidencePreparationLimits,
) -> tuple[SceneBoundary, ...]:
    ordered = tuple(
        sorted(
            scenes,
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    if len(ordered) > limits.max_scene_boundaries:
        raise VideoDemoError(
            ErrorCode.VISUAL_RESULT_INVALID,
            "基础片段收到超过预算的非规范场景集合",
        )
    scene_ids = tuple(item.evidence_id for item in ordered)
    if len(scene_ids) != len(set(scene_ids)):
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "规范场景标识不得重复")
    if not ordered or ordered[0].start_ms != 0 or ordered[-1].end_ms != duration_ms:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "规范场景未覆盖完整视频")
    if any(left.end_ms != right.start_ms for left, right in pairwise(ordered)):
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "规范场景必须连续且无重叠")
    return ordered


def _safe_boundaries(
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue],
    scenes: Sequence[SceneBoundary],
    speech_boundaries: Sequence[SpeechBoundaryCandidate],
) -> tuple[int, ...]:
    # ASR 通常每 1~3 秒产生一条证据；把每条证据的首尾都作为切点会把
    # 章节规划输入膨胀成数百个基础片段。优先使用稀疏的网格、镜头和
    # 高价值语音边界，只有在这些边界无法满足 5 分钟硬上限时才补充
    # 转写结束点。
    candidates = {
        *range(_GRID_INTERVAL_MS, duration_ms, _GRID_INTERVAL_MS),
        *(scene.start_ms for scene in scenes[1:]),
        *(
            candidate.timestamp_ms
            for candidate in speech_boundaries
            if candidate.source != "sentence_end"
            or candidate.score >= 0.95
        ),
    }
    ordered_candidates = _ordered_candidates(candidates, duration_ms)
    safe = _exclude_evidence_interiors(ordered_candidates, transcript)
    safe = _add_blocking_transcript_boundaries(
        safe,
        ordered_candidates,
        duration_ms,
        transcript,
    )
    safe = _add_required_transcript_boundaries(safe, duration_ms, transcript)
    ranges = pairwise((0, *safe, duration_ms))
    if any(
        end_ms - start_ms > _MAX_SEGMENT_DURATION_MS
        for start_ms, end_ms in ranges
    ):
        raise _budget_error("缺少可安全切分的基础片段边界")
    return safe


def _ordered_candidates(candidates: set[int], duration_ms: int) -> tuple[int, ...]:
    return tuple(
        timestamp
        for timestamp in sorted(candidates)
        if 0 < timestamp < duration_ms
    )


def _add_required_transcript_boundaries(
    safe: tuple[int, ...],
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[int, ...]:
    boundaries = list(safe)
    while True:
        ranges = tuple(pairwise((0, *boundaries, duration_ms)))
        oversized = next(
            (
                (start_ms, end_ms)
                for start_ms, end_ms in ranges
                if end_ms - start_ms > _MAX_SEGMENT_DURATION_MS
            ),
            None,
        )
        if oversized is None:
            return tuple(boundaries)
        start_ms, end_ms = oversized
        endpoint = next(
            (
                item.end_ms
                for item in transcript
                if start_ms < item.end_ms <= min(end_ms, start_ms + _MAX_SEGMENT_DURATION_MS)
                and _is_safe_transcript_endpoint(item.end_ms, transcript)
            ),
            None,
        )
        if endpoint is None or endpoint in boundaries:
            return tuple(boundaries)
        boundaries.append(endpoint)
        boundaries.sort()


def _add_blocking_transcript_boundaries(
    safe: tuple[int, ...],
    candidates: Sequence[int],
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[int, ...]:
    boundaries = set(safe)
    for item in transcript:
        if not any(item.start_ms < candidate < item.end_ms for candidate in candidates):
            continue
        for endpoint in (item.start_ms, item.end_ms):
            if (
                0 < endpoint < duration_ms
                and _is_safe_transcript_endpoint(endpoint, transcript)
            ):
                boundaries.add(endpoint)
    return tuple(sorted(boundaries))


def _is_safe_transcript_endpoint(
    timestamp_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> bool:
    return not any(item.start_ms < timestamp_ms < item.end_ms for item in transcript)


def _exclude_evidence_interiors(
    candidates: Sequence[int],
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[int, ...]:
    safe: list[int] = []
    transcript_index = 0
    maximum_open_end = 0
    for candidate in candidates:
        while (
            transcript_index < len(transcript)
            and transcript[transcript_index].start_ms < candidate
        ):
            maximum_open_end = max(
                maximum_open_end,
                transcript[transcript_index].end_ms,
            )
            transcript_index += 1
        if maximum_open_end <= candidate:
            safe.append(candidate)
    return tuple(safe)


def _assemble_segments(
    asset_sha256: str,
    ranges: Sequence[tuple[int, int]],
    transcript: Sequence[SpeechSegment | SubtitleCue],
    scenes: Sequence[SceneBoundary],
) -> tuple[BaseSegment, ...]:
    segments: list[BaseSegment] = []
    transcript_index = 0
    scene_index = 0
    for start_ms, end_ms in ranges:
        evidence_refs, transcript_index = _take_transcript_refs(
            transcript,
            transcript_index,
            start_ms,
            end_ms,
        )
        while scene_index < len(scenes) and scenes[scene_index].end_ms <= start_ms:
            scene_index += 1
        scene_refs = _overlapping_scene_refs(scenes, scene_index, start_ms, end_ms)
        segments.append(
            _build_segment(
                asset_sha256,
                start_ms,
                end_ms,
                evidence_refs,
                scene_refs,
                _transcript_source(transcript, evidence_refs),
            ),
        )
    if transcript_index != len(transcript):
        raise VideoDemoError(
            ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
            "存在未归属到基础片段的转写证据",
        )
    return tuple(segments)


def _take_transcript_refs(
    transcript: Sequence[SpeechSegment | SubtitleCue],
    index: int,
    start_ms: int,
    end_ms: int,
) -> tuple[tuple[str, ...], int]:
    evidence_refs: list[str] = []
    while index < len(transcript) and transcript[index].start_ms < end_ms:
        item = transcript[index]
        if item.start_ms < start_ms or item.end_ms > end_ms:
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
                "转写证据跨越了基础片段安全边界",
            )
        evidence_refs.append(item.evidence_id)
        index += 1
    return tuple(evidence_refs), index


def _overlapping_scene_refs(
    scenes: Sequence[SceneBoundary],
    index: int,
    start_ms: int,
    end_ms: int,
) -> tuple[str, ...]:
    refs: list[str] = []
    while index < len(scenes) and scenes[index].start_ms < end_ms:
        if start_ms < scenes[index].end_ms:
            refs.append(scenes[index].evidence_id)
        index += 1
    return tuple(refs)


def _build_segment(
    asset_sha256: str,
    start_ms: int,
    end_ms: int,
    evidence_refs: tuple[str, ...],
    scene_refs: tuple[str, ...],
    transcript_source: TranscriptSource,
) -> BaseSegment:
    if len(evidence_refs) > _MAX_SEGMENT_EVIDENCE_REFS:
        raise _budget_error("单个基础片段的转写证据引用超过 256 条")
    if len(scene_refs) > _MAX_SEGMENT_SCENE_REFS:
        raise VideoDemoError(
            ErrorCode.VISUAL_RESULT_INVALID,
            "基础片段引用了超过 256 个非规范场景",
        )
    return BaseSegment(
        segment_id=stable_identifier(
            "base_segment",
            {
                "asset_sha256": asset_sha256,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "evidence_refs": list(evidence_refs),
            },
        ),
        start_ms=start_ms,
        end_ms=end_ms,
        evidence_refs=evidence_refs,
        scene_refs=scene_refs,
        transcript_source=transcript_source,
    )


def _transcript_source(
    transcript: Sequence[SpeechSegment | SubtitleCue],
    evidence_refs: tuple[str, ...],
) -> TranscriptSource:
    if not evidence_refs:
        return "NONE"
    return "ASR" if isinstance(transcript[0], SpeechSegment) else "SUBTITLE"


def _budget_error(message: str) -> VideoDemoError:
    return VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, message)
