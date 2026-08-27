from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import pairwise
from threading import Lock
from typing import Literal

from pydantic import Field, StrictInt, field_validator

from video_demo.application.base_segments import (
    build_base_segments as build_base_segments,
)
from video_demo.domain.base import FrozenModel, Sha256, stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_artifact import MAX_METRIC_VALUE
from video_demo.domain.document_plan import (
    BaseSegment,
    ChapterDraft,
    ChapterPlan,
    VisualSearchTarget,
    VisualTargetDraft,
)
from video_demo.domain.evidence import SceneBoundary, SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanningResponse,
    ChapterPlanRepairRequest,
    DocumentTextPort,
    InvalidModelResponse,
    ModelResponseValidationError,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import prompt_for_planning
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_MIN_CHAPTER_DURATION_MS = 60_000
_MAX_CHAPTER_DURATION_MS = 300_000
_MAX_TARGET_ANCHOR_SPAN_MS = 30_000
_DEFAULT_MAX_PLANNING_BATCHES = 64
_DEFAULT_PLANNING_CONCURRENCY = 2
_GRANULARITY_TARGET_DURATION_MS = {
    "fine": 120_000,
    "standard": 180_000,
    "coarse": 300_000,
}
_CHAPTER_PLANNING_METRIC_NAMES = frozenset(
    {
        "chapter_planner_logical_calls",
        "chapter_planner_provider_attempts",
        "chapter_planner_structure_repairs",
        "chapter_planner_cache_hits",
        "chapter_planner_fallback_chapters",
    },
)
_LOGGER = logging.getLogger(__name__)


class ChapterPlanningBatch(FrozenModel):
    plans: tuple[ChapterPlan, ...] = Field(min_length=1, max_length=240)
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) - _CHAPTER_PLANNING_METRIC_NAMES:
            raise ValueError("章节规划指标包含未知白名单键")
        if any(
            type(metric) is not int or not 0 <= metric <= MAX_METRIC_VALUE
            for metric in value.values()
        ):
            raise ValueError("章节规划指标必须是 0 到 2^63-1 的严格整数")
        return value


class _PlanningCounters:
    def __init__(self) -> None:
        self._lock = Lock()
        self.logical_calls = 0
        self.provider_attempts = 0
        self.structure_repairs = 0
        self.cache_hits = 0
        self.fallback_chapters = 0

    def provider_attempt(self) -> None:
        with self._lock:
            self.provider_attempts += 1

    def cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def structure_repair(self) -> None:
        with self._lock:
            self.structure_repairs += 1

    def as_metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "chapter_planner_logical_calls": self.logical_calls,
                "chapter_planner_provider_attempts": self.provider_attempts,
                "chapter_planner_structure_repairs": self.structure_repairs,
                "chapter_planner_cache_hits": self.cache_hits,
                "chapter_planner_fallback_chapters": self.fallback_chapters,
            }


class ChapterPlanner:
    """把模型草稿收敛为程序拥有时间、ID 与引用的连续章节计划。"""

    def __init__(
        self,
        text_port: DocumentTextPort,
        identity: ModelInvocationIdentity,
        *,
        max_input_chars: int,
        max_input_bytes: int,
        max_chapters: int,
        invocation_wait_timeout_seconds: float,
        max_planning_batches: int = _DEFAULT_MAX_PLANNING_BATCHES,
        concurrency: int = _DEFAULT_PLANNING_CONCURRENCY,
    ) -> None:
        if min(
            max_input_chars,
            max_input_bytes,
            max_chapters,
            max_planning_batches,
            concurrency,
        ) < 1:
            raise ValueError("章节规划预算必须大于 0")
        if (
            max_chapters > 240
            or max_planning_batches > _DEFAULT_MAX_PLANNING_BATCHES
            or concurrency > _DEFAULT_PLANNING_CONCURRENCY
        ):
            raise ValueError("章节规划预算超过硬契约上限")
        if invocation_wait_timeout_seconds <= 0:
            raise ValueError("模型调用锁等待时间必须大于 0")
        self._text_port = text_port
        self._identity = identity
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._max_chapters = max_chapters
        self._max_planning_batches = max_planning_batches
        self._concurrency = concurrency
        self._invocation_wait_timeout_seconds = invocation_wait_timeout_seconds

    def plan(
        self,
        *,
        cache: DocumentModelCache,
        asset_sha256: Sha256,
        title_hint: str,
        duration_ms: int,
        segments: tuple[BaseSegment, ...],
        transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
        scenes: tuple[SceneBoundary, ...],
        document_config: DocumentGenerationConfig,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterPlanningBatch:
        ordered_segments = _validate_planning_inputs(
            duration_ms,
            segments,
            transcript_evidence,
            scenes,
        )
        _validate_chapter_count_feasibility(ordered_segments, self._max_chapters)
        counters = _PlanningCounters()
        used_fallback = False
        fallback_segment_ids: set[str] = set()
        if transcript_evidence:
            drafts: list[ChapterDraft] = []
            requests = self._planning_requests(
                title_hint,
                duration_ms,
                ordered_segments,
                transcript_evidence,
                document_config,
            )
            results = self._plan_batches(
                requests,
                cache,
                ordered_segments,
                counters,
                is_cancel_requested,
            )
            for request, response in results:
                if response is None:
                    drafts.extend(_rule_drafts(
                        request.segments,
                        request.transcript_evidence,
                        request.document_config,
                    ))
                    fallback_segment_ids.update(segment.segment_id for segment in request.segments)
                    used_fallback = True
                else:
                    drafts.extend(response.chapter_drafts)
        else:
            drafts = list(_rule_drafts(ordered_segments, (), document_config))

        normalized = _normalize_draft_count(
            _merge_short_drafts(
                tuple(drafts),
                ordered_segments,
                minimum_duration_ms=_minimum_chapter_duration_ms(document_config),
            ),
            ordered_segments,
            self._max_chapters,
        )
        counters.fallback_chapters = sum(
            bool(set(draft.segment_refs) & fallback_segment_ids)
            for draft in normalized
        )
        plans = _materialize_plans(
            asset_sha256,
            normalized,
            ordered_segments,
            transcript_evidence,
            scenes,
        )
        warnings = tuple(
            f"CHAPTER_PLANNING_FALLBACK:{plan.chapter_id}"
            for plan in plans
            if set(plan.segment_refs) & fallback_segment_ids
        )
        return ChapterPlanningBatch(
            plans=plans,
            warnings=warnings,
            status="PARTIAL_SUCCEEDED" if used_fallback else "SUCCEEDED",
            metrics=counters.as_metrics(),
        )

    def _plan_batches(
        self,
        requests: tuple[ChapterPlanningRequest, ...],
        cache: DocumentModelCache,
        all_segments: tuple[BaseSegment, ...],
        counters: _PlanningCounters,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[tuple[ChapterPlanningRequest, ChapterPlanningResponse | None], ...]:
        results: list[
            tuple[ChapterPlanningRequest, ChapterPlanningResponse | None] | None
        ] = [None] * len(requests)
        executor = ThreadPoolExecutor(
            max_workers=self._concurrency,
            thread_name_prefix="chapter-planning",
        )
        pending: dict[Future[ChapterPlanningResponse | None], tuple[int, float]] = {}
        next_index = 0
        try:
            while next_index < len(requests) or pending:
                if is_cancel_requested():
                    for future in pending:
                        future.cancel()
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_index < len(requests) and len(pending) < self._concurrency:
                    counters.logical_calls += 1
                    future = executor.submit(
                        self._logical_call,
                        cache,
                        requests[next_index],
                        all_segments,
                        counters,
                        is_cancel_requested,
                    )
                    pending[future] = (next_index, time.monotonic())
                    next_index += 1
                completed, _ = wait(tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, started_at = pending.pop(future)
                    response = future.result()
                    request = requests[index]
                    data = prompt_for_planning(request)[2]
                    _LOGGER.info(
                        "章节规划批次完成 batch=%d/%d chars=%d bytes=%d elapsed_ms=%d status=%s",
                        index + 1,
                        len(requests),
                        len(data),
                        len(data.encode("utf-8")),
                        max(0, round((time.monotonic() - started_at) * 1_000)),
                        "SUCCEEDED" if response is not None else "FALLBACK",
                    )
                    results[index] = (request, response)
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return tuple(result for result in results if result is not None)

    def _planning_requests(
        self,
        title_hint: str,
        duration_ms: int,
        segments: tuple[BaseSegment, ...],
        transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
        document_config: DocumentGenerationConfig,
    ) -> tuple[ChapterPlanningRequest, ...]:
        evidence_by_id = {item.evidence_id: item for item in transcript_evidence}
        requests: list[ChapterPlanningRequest] = []
        batch: list[BaseSegment] = []
        for segment in segments:
            candidate = (*batch, segment)
            request = _planning_request(
                title_hint,
                duration_ms,
                candidate,
                evidence_by_id,
                document_config,
            )
            if _request_fits(request, self._max_input_chars, self._max_input_bytes):
                batch.append(segment)
                continue
            if not batch:
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "单个基础片段超过文本模型输入预算",
                )
            if len(requests) >= self._max_planning_batches:
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "章节规划批次数超过上限",
                )
            requests.append(
                _planning_request(
                    title_hint,
                    duration_ms,
                    tuple(batch),
                    evidence_by_id,
                    document_config,
                ),
            )
            batch = [segment]
            single = _planning_request(
                title_hint,
                duration_ms,
                (segment,),
                evidence_by_id,
                document_config,
            )
            if not _request_fits(single, self._max_input_chars, self._max_input_bytes):
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "单个基础片段超过文本模型输入预算",
                )
        if batch:
            if len(requests) >= self._max_planning_batches:
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "章节规划批次数超过上限",
                )
            requests.append(
                _planning_request(
                    title_hint,
                    duration_ms,
                    tuple(batch),
                    evidence_by_id,
                    document_config,
                ),
            )
        return tuple(requests)

    def _logical_call(
        self,
        cache: DocumentModelCache,
        request: ChapterPlanningRequest,
        all_segments: tuple[BaseSegment, ...],
        counters: _PlanningCounters,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterPlanningResponse | None:
        def validate(response: ChapterPlanningResponse) -> None:
            _validate_planning_response(response, request, all_segments)
        cached = cache.get(
            self._identity,
            request,
            ChapterPlanningResponse,
            validate,
        )
        if cached is not None:
            counters.cache_hit()
            return cached.response
        with cache.invocation_lock(
            self._identity,
            request,
            wait_timeout_seconds=self._invocation_wait_timeout_seconds,
            is_cancel_requested=is_cancel_requested,
        ):
            cached = cache.get(
                self._identity,
                request,
                ChapterPlanningResponse,
                validate,
            )
            if cached is not None:
                counters.cache_hit()
                return cached.response
            invalid_response: InvalidModelResponse | None = None
            try:
                response = self._text_port.plan_chapters(
                    request,
                    on_provider_attempt=counters.provider_attempt,
                )
            except ModelResponseValidationError as error:
                invalid_response = error.invalid_response
            except VideoDemoError as error:
                if _is_fallback_error(error):
                    return None
                raise
            else:
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    invalid_response = _invalid_local_response(response, error)
            if invalid_response is None:
                successful_path: Literal["MAIN", "REPAIR"] = "MAIN"
            else:
                try:
                    response = self._repair(request, invalid_response, counters)
                except ModelResponseValidationError:
                    return None
                except VideoDemoError as error:
                    if _is_fallback_error(error):
                        return None
                    raise
                try:
                    validate(response)
                except (ValueError, TypeError):
                    return None
                successful_path = "REPAIR"
            return cache.put(
                self._identity,
                request,
                response,
                successful_path=successful_path,
                validate=validate,
            ).response

    def _repair(
        self,
        request: ChapterPlanningRequest,
        invalid_response: object,
        counters: _PlanningCounters,
    ) -> ChapterPlanningResponse:
        if not isinstance(invalid_response, InvalidModelResponse):
            raise TypeError("规划修复上下文类型非法")
        counters.structure_repair()
        return self._text_port.repair_chapter_plan(
            ChapterPlanRepairRequest(
                request=request,
                invalid_response=invalid_response,
                allowed_segment_ids=tuple(item.segment_id for item in request.segments),
                allowed_transcript_ids=tuple(
                    item.evidence_id for item in request.transcript_evidence
                ),
                prompt_version="chapter-planner-repair-v1",
            ),
            on_provider_attempt=counters.provider_attempt,
        )


def _validate_planning_inputs(
    duration_ms: int,
    segments: tuple[BaseSegment, ...],
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
    scenes: tuple[SceneBoundary, ...],
) -> tuple[BaseSegment, ...]:
    if not segments or segments[0].start_ms != 0 or segments[-1].end_ms != duration_ms:
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "基础片段未覆盖完整视频")
    if any(left.end_ms != right.start_ms for left, right in pairwise(segments)):
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "基础片段必须连续且有序")
    segment_ids = tuple(item.segment_id for item in segments)
    if len(segment_ids) != len(set(segment_ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_SEGMENT_ID, "基础片段标识不得重复")
    transcript_ids = tuple(item.evidence_id for item in transcript_evidence)
    if len(transcript_ids) != len(set(transcript_ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "转写证据标识不得重复")
    referenced_ids = tuple(ref for segment in segments for ref in segment.evidence_refs)
    if referenced_ids != transcript_ids:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
            "基础片段必须按顺序完整引用转写证据",
        )
    if any(item.end_ms > duration_ms for item in transcript_evidence):
        raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "转写证据超出视频范围")
    transcript_by_id = {item.evidence_id: item for item in transcript_evidence}
    for segment in segments:
        if any(
            transcript_by_id[ref].start_ms < segment.start_ms
            or transcript_by_id[ref].end_ms > segment.end_ms
            for ref in segment.evidence_refs
        ):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
                "转写证据不在引用它的基础片段内",
            )
    if scenes:
        if scenes[0].start_ms != 0 or scenes[-1].end_ms != duration_ms:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景未覆盖完整视频")
        if any(left.end_ms != right.start_ms for left, right in pairwise(scenes)):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景必须连续且有序")
        scene_ids = tuple(item.evidence_id for item in scenes)
        if len(scene_ids) != len(set(scene_ids)):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景标识不得重复")
    return segments


def _planning_request(
    title_hint: str,
    duration_ms: int,
    segments: tuple[BaseSegment, ...],
    evidence_by_id: dict[str, SpeechSegment | SubtitleCue],
    document_config: DocumentGenerationConfig,
) -> ChapterPlanningRequest:
    evidence = tuple(
        evidence_by_id[ref]
        for segment in segments
        for ref in segment.evidence_refs
    )
    return ChapterPlanningRequest(
        title_hint=title_hint,
        duration_ms=duration_ms,
        segments=segments,
        transcript_evidence=evidence,
        document_config=document_config,
        prompt_version="chapter-planner-v1",
    )


def _validate_chapter_count_feasibility(
    segments: tuple[BaseSegment, ...],
    maximum: int,
) -> None:
    if any(segment.duration_ms > _MAX_CHAPTER_DURATION_MS for segment in segments):
        raise VideoDemoError(
            ErrorCode.INPUT_BUDGET_EXCEEDED,
            "存在无法放入 5 分钟章节的基础片段",
        )
    minimum_chapters = 1
    chapter_start_ms = segments[0].start_ms
    for segment in segments:
        if segment.end_ms - chapter_start_ms <= _MAX_CHAPTER_DURATION_MS:
            continue
        minimum_chapters += 1
        chapter_start_ms = segment.start_ms
    if minimum_chapters > maximum:
        raise VideoDemoError(
            ErrorCode.INPUT_BUDGET_EXCEEDED,
            "视频时长与安全片段边界无法满足章节数量上限",
        )


def _request_fits(
    request: ChapterPlanningRequest,
    max_chars: int,
    max_bytes: int,
) -> bool:
    data = prompt_for_planning(request)[2]
    return len(data) <= max_chars and len(data.encode("utf-8")) <= max_bytes


def _validate_planning_response(
    response: ChapterPlanningResponse,
    request: ChapterPlanningRequest,
    all_segments: tuple[BaseSegment, ...],
) -> None:
    requested_ids = tuple(item.segment_id for item in request.segments)
    actual_ids = tuple(
        ref
        for draft in response.chapter_drafts
        for ref in draft.segment_refs
    )
    if actual_ids != requested_ids:
        raise ValueError("章节草稿必须按顺序完整分区当前批次基础片段")
    segment_by_id = {item.segment_id: item for item in all_segments}
    allowed_evidence = {
        item.evidence_id for item in request.transcript_evidence
    }
    evidence_by_id = {
        item.evidence_id: item for item in request.transcript_evidence
    }
    for draft in response.chapter_drafts:
        chapter_evidence = {
            ref
            for segment_ref in draft.segment_refs
            for ref in segment_by_id[segment_ref].evidence_refs
        }
        start_ms = segment_by_id[draft.segment_refs[0]].start_ms
        end_ms = segment_by_id[draft.segment_refs[-1]].end_ms
        if end_ms - start_ms > _MAX_CHAPTER_DURATION_MS:
            raise ValueError("章节草稿超过 5 分钟")
        semantic_count = len(draft.semantic_targets)
        maximum = 4 if draft.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2
        if semantic_count > maximum:
            raise ValueError("章节草稿语义目标超过视觉模式上限")
        if draft.visual_mode == "NONE" and semantic_count:
            raise ValueError("visual_mode=NONE 不得包含语义目标")
        if draft.visual_mode in {"COMPARISON", "MULTI_STEP"} and semantic_count < 2:
            raise ValueError("复杂视觉模式至少需要两个语义目标")
        anchor_groups: list[set[str]] = []
        for target in draft.semantic_targets:
            anchor_refs = target.anchor_evidence_refs
            anchors = set(anchor_refs)
            if len(anchor_refs) != len(anchors):
                raise ValueError("语义目标锚点不得重复")
            if not anchors <= allowed_evidence or not anchors <= chapter_evidence:
                raise ValueError("语义目标锚点必须属于当前章节")
            anchor_evidence = tuple(evidence_by_id[ref] for ref in anchor_refs)
            if anchor_evidence != tuple(
                sorted(
                    anchor_evidence,
                    key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
                ),
            ):
                raise ValueError("语义目标锚点必须按时间排序")
            if (
                anchor_evidence[-1].end_ms - anchor_evidence[0].start_ms
                > _MAX_TARGET_ANCHOR_SPAN_MS
            ):
                raise ValueError("单个语义目标的锚点跨度不得超过 30 秒")
            anchor_groups.append(anchors)
        if draft.visual_mode in {"COMPARISON", "MULTI_STEP"} and any(
            group & other
            for index, group in enumerate(anchor_groups)
            for other in anchor_groups[index + 1 :]
        ):
            raise ValueError("复杂视觉模式的锚点组必须不重叠")


def _rule_drafts(
    segments: tuple[BaseSegment, ...],
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
    document_config: DocumentGenerationConfig,
) -> tuple[ChapterDraft, ...]:
    evidence_by_id = {item.evidence_id: item for item in transcript_evidence}
    target_duration_ms = _GRANULARITY_TARGET_DURATION_MS[
        document_config.chapter_granularity
    ]
    minimum_duration_ms = _minimum_chapter_duration_ms(document_config)
    groups = _group_segments_near_target(segments, target_duration_ms)
    if groups:
        current = groups[-1]
        if (
            len(groups) > 1
            and current[-1].end_ms - current[0].start_ms < minimum_duration_ms
            and current[-1].end_ms - groups[-2][0].start_ms <= _MAX_CHAPTER_DURATION_MS
        ):
            groups[-2].extend(groups.pop())
    drafts: list[ChapterDraft] = []
    for index, group in enumerate(groups, start=1):
        first_text = next(
            (
                evidence_by_id[ref].text
                for segment in group
                for ref in segment.evidence_refs
                if ref in evidence_by_id
            ),
            "",
        )
        drafts.append(
            ChapterDraft(
                segment_refs=tuple(item.segment_id for item in group),
                title_hint=(first_text[:200] or f"章节 {index}"),
                visual_mode="NONE" if transcript_evidence else "SINGLE",
                semantic_targets=(),
            ),
        )
    return tuple(drafts)


def _group_segments_near_target(
    segments: tuple[BaseSegment, ...],
    target_duration_ms: int,
) -> list[list[BaseSegment]]:
    groups: list[list[BaseSegment]] = []
    start = 0
    while start < len(segments):
        best_end = start + 1
        best_distance = abs(segments[start].duration_ms - target_duration_ms)
        cursor = start + 1
        while cursor < len(segments):
            duration_ms = segments[cursor].end_ms - segments[start].start_ms
            if duration_ms > _MAX_CHAPTER_DURATION_MS:
                break
            distance = abs(duration_ms - target_duration_ms)
            if distance < best_distance:
                best_end = cursor + 1
                best_distance = distance
            if duration_ms >= target_duration_ms:
                break
            cursor += 1
        groups.append(list(segments[start:best_end]))
        start = best_end
    return groups


def _normalize_draft_count(
    drafts: tuple[ChapterDraft, ...],
    segments: tuple[BaseSegment, ...],
    maximum: int,
) -> tuple[ChapterDraft, ...]:
    if len(drafts) <= maximum:
        return drafts
    segment_by_id = {item.segment_id: item for item in segments}
    normalized = list(drafts)
    while len(normalized) > maximum:
        merge_at = next(
            (
                index
                for index in range(len(normalized) - 1)
                if _drafts_can_merge(
                    normalized[index],
                    normalized[index + 1],
                    segment_by_id,
                )
            ),
            None,
        )
        if merge_at is None:
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "章节数量超过上限且无法安全合并",
            )
        normalized[merge_at : merge_at + 2] = [
            _merge_drafts(normalized[merge_at], normalized[merge_at + 1]),
        ]
    return tuple(normalized)


def _merge_short_drafts(
    drafts: tuple[ChapterDraft, ...],
    segments: tuple[BaseSegment, ...],
    *,
    minimum_duration_ms: int,
) -> tuple[ChapterDraft, ...]:
    segment_by_id = {item.segment_id: item for item in segments}
    normalized = list(drafts)
    index = 0
    while index < len(normalized):
        if _draft_duration_ms(normalized[index], segment_by_id) >= minimum_duration_ms:
            index += 1
            continue
        if index > 0 and _drafts_can_merge(
            normalized[index - 1],
            normalized[index],
            segment_by_id,
        ):
            normalized[index - 1 : index + 1] = [
                _merge_drafts(normalized[index - 1], normalized[index]),
            ]
            index = max(0, index - 1)
            continue
        if index + 1 < len(normalized) and _drafts_can_merge(
            normalized[index],
            normalized[index + 1],
            segment_by_id,
        ):
            normalized[index : index + 2] = [
                _merge_drafts(normalized[index], normalized[index + 1]),
            ]
            continue
        index += 1
    return tuple(normalized)


def _minimum_chapter_duration_ms(
    document_config: DocumentGenerationConfig,
) -> int:
    if document_config.chapter_granularity == "coarse":
        return 120_000
    return _MIN_CHAPTER_DURATION_MS


def _draft_duration_ms(
    draft: ChapterDraft,
    segment_by_id: dict[str, BaseSegment],
) -> int:
    return (
        segment_by_id[draft.segment_refs[-1]].end_ms
        - segment_by_id[draft.segment_refs[0]].start_ms
    )


def _drafts_can_merge(
    left: ChapterDraft,
    right: ChapterDraft,
    segment_by_id: dict[str, BaseSegment],
) -> bool:
    duration_ms = (
        segment_by_id[right.segment_refs[-1]].end_ms
        - segment_by_id[left.segment_refs[0]].start_ms
    )
    targets = (*left.semantic_targets, *right.semantic_targets)
    both_have_targets = bool(left.semantic_targets and right.semantic_targets)
    compatible_single_targets = (
        left.visual_mode == right.visual_mode == "SINGLE" and len(targets) <= 2
    )
    return (
        duration_ms <= _MAX_CHAPTER_DURATION_MS
        and len(targets) <= 4
        and (not both_have_targets or compatible_single_targets)
    )


def _merge_drafts(left: ChapterDraft, right: ChapterDraft) -> ChapterDraft:
    targets = (*left.semantic_targets, *right.semantic_targets)
    visual_mode: Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]
    if not targets:
        visual_mode = "SINGLE" if "SINGLE" in {left.visual_mode, right.visual_mode} else "NONE"
    elif left.semantic_targets and right.semantic_targets:
        visual_mode = "SINGLE"
    else:
        target_owner = left if left.semantic_targets else right
        visual_mode = target_owner.visual_mode
    return ChapterDraft(
        segment_refs=(*left.segment_refs, *right.segment_refs),
        title_hint=left.title_hint,
        visual_mode=visual_mode,
        semantic_targets=targets,
    )


def _materialize_plans(
    asset_sha256: str,
    drafts: tuple[ChapterDraft, ...],
    segments: tuple[BaseSegment, ...],
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
    scenes: tuple[SceneBoundary, ...],
) -> tuple[ChapterPlan, ...]:
    segment_by_id = {item.segment_id: item for item in segments}
    transcript_ids = {item.evidence_id for item in transcript_evidence}
    plans: list[ChapterPlan] = []
    for draft in drafts:
        chapter_segments = tuple(segment_by_id[ref] for ref in draft.segment_refs)
        start_ms = chapter_segments[0].start_ms
        end_ms = chapter_segments[-1].end_ms
        chapter_id = stable_identifier(
            "chapter",
            {
                "asset_sha256": asset_sha256,
                "segment_refs": list(draft.segment_refs),
            },
        )
        semantic_targets = tuple(
            _semantic_target(
                asset_sha256,
                chapter_id,
                ordinal,
                target,
                transcript_ids,
            )
            for ordinal, target in enumerate(draft.semantic_targets)
        )
        base_target = _base_coverage_target(
            asset_sha256,
            chapter_id,
            start_ms,
            end_ms,
            scenes,
        )
        plans.append(
            ChapterPlan(
                chapter_id=chapter_id,
                start_ms=start_ms,
                end_ms=end_ms,
                segment_refs=draft.segment_refs,
                title_hint=draft.title_hint,
                visual_mode=draft.visual_mode,
                semantic_targets=semantic_targets,
                base_coverage_targets=(base_target,),
            ),
        )
    _validate_plan_timeline(tuple(plans), segments)
    return tuple(plans)


def _semantic_target(
    asset_sha256: str,
    chapter_id: str,
    ordinal: int,
    draft: VisualTargetDraft,
    transcript_ids: set[str],
) -> VisualSearchTarget:
    if any(ref not in transcript_ids for ref in draft.anchor_evidence_refs):
        raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "语义目标引用未知转写")
    payload = draft.model_dump(mode="json")
    return VisualSearchTarget(
        target_id=stable_identifier(
            "visual_target",
            {
                "asset_sha256": asset_sha256,
                "chapter_id": chapter_id,
                "purpose": "SEMANTIC",
                "ordinal": ordinal,
                "target": payload,
            },
        ),
        purpose="SEMANTIC",
        query_zh=draft.query_zh,
        anchor_evidence_refs=draft.anchor_evidence_refs,
    )


def _base_coverage_target(
    asset_sha256: str,
    chapter_id: str,
    start_ms: int,
    end_ms: int,
    scenes: tuple[SceneBoundary, ...],
) -> VisualSearchTarget:
    overlapping = tuple(
        item.evidence_id
        for item in scenes
        if start_ms < item.end_ms and item.start_ms < end_ms
    )
    scene_refs = (
        overlapping
        if len(overlapping) <= 2
        else (overlapping[0], overlapping[-1])
    )
    sample_timestamps = () if scene_refs else (start_ms + (end_ms - start_ms) // 2,)
    payload = {
        "scene_refs": list(scene_refs),
        "sample_timestamps_ms": list(sample_timestamps),
    }
    return VisualSearchTarget(
        target_id=stable_identifier(
            "visual_target",
            {
                "asset_sha256": asset_sha256,
                "chapter_id": chapter_id,
                "purpose": "BASE_COVERAGE",
                "ordinal": 0,
                "target": payload,
            },
        ),
        purpose="BASE_COVERAGE",
        query_zh="检查本章代表性画面是否提供语音之外的信息",
        scene_refs=scene_refs,
        sample_timestamps_ms=sample_timestamps,
    )


def _validate_plan_timeline(
    plans: tuple[ChapterPlan, ...],
    segments: tuple[BaseSegment, ...],
) -> None:
    if not plans or plans[0].start_ms != 0 or plans[-1].end_ms != segments[-1].end_ms:
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "章节未覆盖完整视频")
    if any(left.end_ms != right.start_ms for left, right in pairwise(plans)):
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "章节时间轴不连续")
    actual_refs = tuple(ref for plan in plans for ref in plan.segment_refs)
    expected_refs = tuple(item.segment_id for item in segments)
    if actual_refs != expected_refs:
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "章节未完整分区基础片段")


def _invalid_local_response(
    response: ChapterPlanningResponse,
    error: BaseException,
) -> InvalidModelResponse:
    serialized = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return invalid_model_response(
        serialized,
        (str(error)[:500] or "chapter_planning:invalid",),
        parsed_json=response.model_dump(mode="json"),
    )


def _is_fallback_error(error: BaseException) -> bool:
    return isinstance(error, VideoDemoError) and error.code in {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
    }
