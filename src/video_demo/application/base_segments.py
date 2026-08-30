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
_SPARSE_SENTENCE_BOUNDARY_LIMIT = 16
_SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS = 60_000
_SENTENCE_BOUNDARY_GRID_TOLERANCE_MS = 8_000
_EMPTY_TRANSCRIPT_RANGE_MAX_MS = 10_000
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
    *,
    allow_empty_scenes: bool = False,
) -> tuple[BaseSegment, ...]:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    transcript = _validate_transcript_budget(transcript_evidence, duration_ms, limits)
    ordered_scenes = _validate_scenes(
        scenes,
        duration_ms,
        limits,
        allow_empty=allow_empty_scenes,
    )
    boundaries = _safe_boundaries(
        duration_ms,
        transcript,
        ordered_scenes,
        speech_boundaries,
    )
    ranges = _merge_empty_transcript_ranges(
        tuple(pairwise((0, *boundaries, duration_ms))),
        transcript,
    )
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
    *,
    allow_empty: bool = False,
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
    if not ordered and allow_empty:
        return ordered
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
    # 场景检测边界用于给每个基础片段建立 scene_refs，不再直接变成切点。
    # 视频编辑中的短镜头可能只有 1~2 秒；把每个镜头都切成基础片段会让
    # 章节规划输入按镜头数量膨胀，迫使模型重复调用，而不会改善章节语义。
    # 网格边界提供稳定的稀疏覆盖，章节抽帧阶段仍会读取完整场景索引。
    grid_candidates = set(range(_GRID_INTERVAL_MS, duration_ms, _GRID_INTERVAL_MS))
    candidates = {
        *grid_candidates,
        *(
            candidate.timestamp_ms
            for candidate in speech_boundaries
            if candidate.source != "sentence_end" or candidate.score >= 0.95
        ),
    }
    sentence_ends = tuple(
        candidate.timestamp_ms
        for candidate in speech_boundaries
        if candidate.source == "sentence_end" and candidate.score >= 0.8
    )
    if len(sentence_ends) <= _SPARSE_SENTENCE_BOUNDARY_LIMIT:
        selected_sentence_ends = sentence_ends
    else:
        selected_sentence_ends = _sparse_sentence_boundaries(sentence_ends, duration_ms)
    candidates.difference_update(
        {
            grid
            for grid in grid_candidates
            if any(
                abs(grid - sentence_end) <= _SENTENCE_BOUNDARY_GRID_TOLERANCE_MS
                for sentence_end in selected_sentence_ends
            )
        },
    )
    candidates.update(selected_sentence_ends)
    ordered_candidates = _ordered_candidates(candidates, duration_ms)
    safe = _exclude_evidence_interiors(ordered_candidates, transcript)
    safe = _snap_dense_grid_boundaries(safe, ordered_candidates, transcript)
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


def _merge_empty_transcript_ranges(
    ranges: tuple[tuple[int, int], ...],
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[tuple[int, int], ...]:
    """把 ASR/字幕之间的纯静音区间并入相邻证据片段，避免空规划单元。"""

    if not transcript:
        return ranges
    normalized = list(ranges)
    while len(normalized) > 1:
        empty_index = next(
            (
                index
                for index, (start_ms, end_ms) in enumerate(normalized, start=0)
                if end_ms - start_ms <= _EMPTY_TRANSCRIPT_RANGE_MAX_MS
                if index > 0
                if not any(
                    start_ms <= item.start_ms and item.end_ms <= end_ms
                    for item in transcript
                )
            ),
            None,
        )
        if empty_index is None:
            break
        _start_ms, end_ms = normalized.pop(empty_index)
        previous_start_ms, _previous_end_ms = normalized[empty_index - 1]
        normalized[empty_index - 1] = (previous_start_ms, end_ms)
    return tuple(normalized)


def _snap_dense_grid_boundaries(
    safe: tuple[int, ...],
    candidates: Sequence[int],
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[int, ...]:
    """为被连续 ASR 覆盖的网格补一个最近安全句尾。"""

    if len(transcript) < 10 or not any(
        isinstance(item, SpeechSegment) for item in transcript
    ):
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


def _sparse_sentence_boundaries(
    sentence_ends: Sequence[int],
    duration_ms: int,
) -> tuple[int, ...]:
    """在密集句尾中每个 30 秒网格最多保留一个最近边界。

    这样既保留长视频的语义切分机会，又避免把连续 ASR 句尾退化成逐句基础片段。
    误差超过容忍范围的句尾不强行加入，交给网格和静音边界处理。
    """

    selected: dict[int, tuple[int, int]] = {}
    for timestamp_ms in sentence_ends:
        if not 0 < timestamp_ms < duration_ms:
            continue
        bucket = timestamp_ms // _SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS
        target = min(
            duration_ms,
            (bucket + 1) * _SENTENCE_BOUNDARY_SAMPLE_INTERVAL_MS,
        )
        distance = abs(timestamp_ms - target)
        if distance > _SENTENCE_BOUNDARY_GRID_TOLERANCE_MS:
            continue
        previous = selected.get(bucket)
        if previous is None or (distance, timestamp_ms) < previous:
            selected[bucket] = (distance, timestamp_ms)
    return tuple(timestamp for _distance, timestamp in selected.values())


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
        # 在安全上限内取最晚的证据结束点，尽量把密集 ASR 合并进同一片段。
        # 取最早结束点会把连续的 1~3 秒 ASR 退化成“一条证据一个片段”，
        # 既增加章节规划批次，也没有带来更可靠的时间边界。
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


def _add_blocking_transcript_boundaries(
    safe: tuple[int, ...],
    candidates: Sequence[int],
    duration_ms: int,
    transcript: Sequence[SpeechSegment | SubtitleCue],
) -> tuple[int, ...]:
    boundaries = set(safe)
    for item in transcript:
        # ASR 通常是 1~3 秒的密集短片段；如果网格/镜头边界恰好穿过它，
        # 为每条 ASR 补首尾会重新退化成“一条 ASR 一个基础片段”。ASR
        # 只需由 `_exclude_evidence_interiors()` 保证不被切穿，边界继续
        # 使用下一个安全候选。字幕 cue 更稀疏，仍保留精确起止边界，
        # 维持字幕时间轴的可追溯性和既有契约。
        if not isinstance(item, SubtitleCue):
            continue
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
