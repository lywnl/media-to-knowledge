"""音频独立章节规划：只处理音频片段和转写证据。"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import pairwise
from threading import Lock
from typing import Literal

from pydantic import Field, StrictInt, field_validator

from video_demo.domain.audio_plan import (
    AudioBaseSegment,
    AudioChapterDraft,
    AudioChapterPlan,
    AudioDocumentConfig,
    AudioTranscriptEvidence,
)
from video_demo.domain.base import FrozenModel, Sha256, stable_identifier
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.audio_document_port import (
    AudioChapterBoundaryCoordinationRequest,
    AudioChapterBoundaryCoordinationResponse,
    AudioChapterBoundaryInput,
    AudioChapterPlanningRequest,
    AudioChapterPlanningResponse,
    AudioChapterPlanRepairRequest,
    AudioInvalidModelResponse,
    AudioModelResponseValidationError,
    AudioTextPort,
    audio_invalid_model_response,
)
from video_demo.integrations.audio_document_prompts import prompt_for_audio_planning
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_LOGGER = logging.getLogger(__name__)

_MAX_BATCH_SEGMENTS = 24
_MAX_CHAPTER_DURATION_MS = 300_000
_MIN_CHAPTER_DURATION_MS = 60_000
_TARGET_DURATION_MS = {"fine": 120_000, "standard": 180_000, "coarse": 300_000}
_METRICS = frozenset(
    {
        "audio_planner_logical_calls",
        "audio_planner_provider_attempts",
        "audio_planner_repairs",
        "audio_planner_cache_hits",
        "audio_planner_fallback_chapters",
    }
)


@dataclass(frozen=True, slots=True)
class _PlanningCallResult:
    response: AudioChapterPlanningResponse | None
    repair_used: bool = False
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class _PlanningBatchResult:
    request: AudioChapterPlanningRequest
    response: AudioChapterPlanningResponse | None
    repair_used: bool = False
    fallback: bool = False


class AudioChapterPlanningBatch(FrozenModel):
    plans: tuple[AudioChapterPlan, ...] = Field(min_length=1, max_length=240)
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != _METRICS:
            raise ValueError("音频章节规划指标不完整")
        if any(type(item) is not int or item < 0 for item in value.values()):
            raise ValueError("音频章节规划指标必须是非负整数")
        return value


class _Counters:
    def __init__(self) -> None:
        self._lock = Lock()
        self.logical = 0
        self.attempts = 0
        self.repairs = 0
        self.cache_hits = 0
        self.fallback_chapters = 0

    def attempt(self) -> None:
        with self._lock:
            self.attempts += 1

    def repair(self) -> None:
        with self._lock:
            self.repairs += 1

    def cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1


class AudioChapterPlanner:
    def __init__(
        self,
        text_port: AudioTextPort,
        identity: ModelInvocationIdentity,
        *,
        max_input_chars: int,
        max_input_bytes: int,
        max_chapters: int,
        invocation_wait_timeout_seconds: float,
        max_planning_batches: int = 64,
        concurrency: int = 2,
        boundary_identity: ModelInvocationIdentity | None = None,
    ) -> None:
        if (
            min(max_input_chars, max_input_bytes, max_chapters, max_planning_batches, concurrency)
            < 1
        ):
            raise ValueError("音频规划预算必须大于 0")
        if concurrency > 2 or max_chapters > 240 or max_planning_batches > 64:
            raise ValueError("音频规划预算超过硬上限")
        if invocation_wait_timeout_seconds <= 0:
            raise ValueError("模型调用锁等待时间必须大于 0")
        self._port = text_port
        self._identity = identity
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._max_chapters = max_chapters
        self._max_batches = max_planning_batches
        self._concurrency = concurrency
        self._wait_timeout = invocation_wait_timeout_seconds
        self._boundary_identity = boundary_identity or identity.model_copy(
            update={
                "logical_operation": "audio_chapter_boundary_coordination",
                "main_response_schema_name": "audio_chapter_boundary_coordination_v1",
                "main_prompt_version": "audio-chapter-boundary-coordinator-v1",
                "repair_response_schema_name": "audio_chapter_boundary_coordination_v1",
                "repair_prompt_version": "audio-chapter-boundary-coordinator-v1",
            },
        )

    def plan(
        self,
        *,
        cache: DocumentModelCache,
        asset_sha256: Sha256,
        title_hint: str,
        duration_ms: int,
        segments: tuple[AudioBaseSegment, ...],
        transcript_evidence: tuple[AudioTranscriptEvidence, ...],
        document_config: AudioDocumentConfig,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioChapterPlanningBatch:
        _validate_inputs(duration_ms, segments, transcript_evidence)
        _validate_chapter_count_feasibility(segments, self._max_chapters)
        requests = _requests(
            title_hint,
            duration_ms,
            segments,
            transcript_evidence,
            document_config,
            self._max_input_chars,
            self._max_input_bytes,
            max_batches=self._max_batches,
        )
        if len(requests) > self._max_batches:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频章节规划批次数超过上限")
        counters = _Counters()
        results = self._plan_batches(
            requests,
            cache,
            counters,
            is_cancel_requested,
        )
        drafts: list[AudioChapterDraft] = []
        batch_drafts: list[list[AudioChapterDraft]] = []
        fallback_segment_ids: set[str] = set()
        used_fallback = False
        for index, request in enumerate(requests):
            result = results[index]
            assert result is not None
            if result.response is None:
                local_drafts = list(
                    _rule_drafts(request.segments, request.transcript_evidence, document_config)
                )
                fallback_segment_ids.update(item.segment_id for item in request.segments)
                used_fallback = True
            else:
                local_drafts = list(result.response.chapter_drafts)
            batch_drafts.append(local_drafts)
            drafts.extend(local_drafts)
        self._coordinate_suspicious_boundaries(
            batch_drafts,
            tuple(result for result in results if result is not None),
            transcript_evidence,
            segments,
            document_config,
            counters,
            cache,
            is_cancel_requested,
        )
        drafts = [draft for batch in batch_drafts for draft in batch]
        normalized = _normalize_draft_count(
            _merge_short_drafts(tuple(drafts), segments, document_config.chapter_granularity),
            segments,
            self._max_chapters,
        )
        if len(normalized) > self._max_chapters:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频章节数量超过上限")
        plans = _materialize(asset_sha256, normalized, segments)
        counters.fallback_chapters = sum(
            bool(set(draft.segment_refs) & fallback_segment_ids)
            for draft in normalized
        )
        warning = tuple(
            f"AUDIO_CHAPTER_PLANNING_FALLBACK:{plan.chapter_id}"
            for plan in plans
            if set(plan.segment_refs) & fallback_segment_ids
        )
        return AudioChapterPlanningBatch(
            plans=plans,
            warnings=warning,
            status="PARTIAL_SUCCEEDED" if used_fallback else "SUCCEEDED",
            metrics={
                "audio_planner_logical_calls": counters.logical,
                "audio_planner_provider_attempts": counters.attempts,
                "audio_planner_repairs": counters.repairs,
                "audio_planner_cache_hits": counters.cache_hits,
                "audio_planner_fallback_chapters": counters.fallback_chapters,
            },
        )

    def _plan_batches(
        self,
        requests: tuple[AudioChapterPlanningRequest, ...],
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[_PlanningBatchResult, ...]:
        results: list[_PlanningBatchResult | None] = [None] * len(requests)
        executor = ThreadPoolExecutor(
            max_workers=self._concurrency,
            thread_name_prefix="audio-planning",
        )
        pending: dict[Future[_PlanningCallResult], tuple[int, float]] = {}
        next_index = 0
        try:
            while next_index < len(requests) or pending:
                if is_cancel_requested():
                    for future in pending:
                        future.cancel()
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_index < len(requests) and len(pending) < self._concurrency:
                    counters.logical += 1
                    request = requests[next_index]
                    future = executor.submit(
                        self._call,
                        cache,
                        request,
                        counters,
                        is_cancel_requested,
                    )
                    _log_planning_batch_start(request, next_index, len(requests))
                    pending[future] = (next_index, time.monotonic())
                    next_index += 1
                done, _ = wait(tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    index, started_at = pending.pop(future)
                    call_result = future.result()
                    request = requests[index]
                    results[index] = _PlanningBatchResult(
                        request=request,
                        response=call_result.response,
                        repair_used=call_result.repair_used,
                        fallback=call_result.fallback,
                    )
                    _log_planning_batch_finished(
                        request,
                        index,
                        len(requests),
                        call_result.response is not None,
                        started_at,
                    )
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return tuple(result for result in results if result is not None)

    def _coordinate_suspicious_boundaries(
        self,
        batch_drafts: list[list[AudioChapterDraft]],
        batch_results: tuple[_PlanningBatchResult, ...],
        transcript_evidence: tuple[AudioTranscriptEvidence, ...],
        segments: tuple[AudioBaseSegment, ...],
        document_config: AudioDocumentConfig,
        counters: _Counters,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> None:
        if len(batch_drafts) < 2:
            return
        suspicious = tuple(
            index
            for index in range(len(batch_drafts) - 1)
            if batch_drafts[index]
            and batch_drafts[index + 1]
            and _audio_boundary_is_suspicious(
                index,
                batch_drafts,
                batch_results,
                segments,
                document_config,
            )
        )
        if not suspicious:
            _LOGGER.info("音频章节边界协调跳过 reason=NORMAL_BOUNDARIES")
            return
        request = _audio_boundary_request(
            suspicious,
            batch_drafts,
            batch_results,
            transcript_evidence,
            segments,
        )
        if request is None:
            _LOGGER.info("音频章节边界协调跳过 reason=REQUEST_SIZE_LIMIT")
            return
        _LOGGER.info(
            "音频章节边界协调开始 boundaries=%d suspicious=%d",
            len(batch_drafts) - 1,
            len(suspicious),
        )
        response = self._coordinate_boundaries(
            request,
            cache,
            counters,
            is_cancel_requested,
        )
        if response is None:
            _LOGGER.info("音频章节边界协调跳过 reason=COORDINATOR_FAILED")
            return
        _apply_audio_boundary_decisions(response, suspicious, batch_drafts, segments)

    def _coordinate_boundaries(
        self,
        request: AudioChapterBoundaryCoordinationRequest,
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioChapterBoundaryCoordinationResponse | None:
        def validate(response: AudioChapterBoundaryCoordinationResponse) -> None:
            allowed = {item.boundary_index for item in request.boundaries}
            if any(item.boundary_index not in allowed for item in response.decisions):
                raise ValueError("音频边界协调返回未知 boundary_index")

        cached = cache.get(
            self._boundary_identity,
            request,
            AudioChapterBoundaryCoordinationResponse,
            validate,
        )
        if cached is not None:
            counters.cache_hit()
            return cached.response
        try:
            with cache.invocation_lock(
                self._boundary_identity,
                request,
                wait_timeout_seconds=self._wait_timeout,
                is_cancel_requested=is_cancel_requested,
            ):
                cached = cache.get(
                    self._boundary_identity,
                    request,
                    AudioChapterBoundaryCoordinationResponse,
                    validate,
                )
                if cached is not None:
                    counters.cache_hit()
                    return cached.response
                response = self._port.coordinate_chapter_boundaries(
                    request,
                    on_provider_attempt=counters.attempt,
                )
                validate(response)
                return cache.put(
                    self._boundary_identity,
                    request,
                    response,
                    successful_path="MAIN",
                    validate=validate,
                ).response
        except (AudioModelResponseValidationError, ValueError, VideoDemoError) as error:
            if isinstance(error, VideoDemoError) and error.code == ErrorCode.JOB_CANCELLED:
                raise
            _LOGGER.info("音频章节边界协调跳过 reason=%s", type(error).__name__)
            return None

    def _call(
        self,
        cache: DocumentModelCache,
        request: AudioChapterPlanningRequest,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> _PlanningCallResult:
        def validate(response: AudioChapterPlanningResponse) -> None:
            _validate_response(response, request)

        cached = cache.get(self._identity, request, AudioChapterPlanningResponse, validate)
        if cached is not None:
            counters.cache_hit()
            return _PlanningCallResult(
                cached.response,
                repair_used=cached.successful_path == "REPAIR",
            )
        with cache.invocation_lock(
            self._identity,
            request,
            wait_timeout_seconds=self._wait_timeout,
            is_cancel_requested=is_cancel_requested,
        ):
                cached = cache.get(
                    self._identity,
                    request,
                    AudioChapterPlanningResponse,
                    validate,
                )
                if cached is not None:
                    counters.cache_hit()
                    return _PlanningCallResult(
                        cached.response,
                        repair_used=cached.successful_path == "REPAIR",
                    )
                repaired = False
                invalid: AudioInvalidModelResponse | None = None
                try:
                    response = self._port.plan_chapters(
                        request,
                        on_provider_attempt=counters.attempt,
                    )
                except AudioModelResponseValidationError as error:
                    if error.code == ErrorCode.JOB_CANCELLED:
                        raise
                    invalid = error.invalid_response
                    _log_planning_validation_failure(request, invalid.validation_errors)
                except VideoDemoError as error:
                    if error.code == ErrorCode.JOB_CANCELLED:
                        raise
                    if error.code in {
                        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                    }:
                        return _PlanningCallResult(None, fallback=True)
                    raise
                else:
                    try:
                        validate(response)
                    except (ValueError, TypeError) as error:
                        invalid = _invalid_planning_local_error(response, error)
                        _log_planning_validation_failure(request, invalid.validation_errors)
                if invalid is not None:
                    counters.repair()
                    try:
                        response = self._port.repair_chapter_plan(
                            _plan_repair_request(request, invalid),
                            on_provider_attempt=counters.attempt,
                        )
                    except AudioModelResponseValidationError:
                        return _PlanningCallResult(None, fallback=True)
                    except VideoDemoError as error:
                        if error.code == ErrorCode.JOB_CANCELLED:
                            raise
                        if error.code in {
                            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                        }:
                            return _PlanningCallResult(None, fallback=True)
                        raise
                    else:
                        try:
                            validate(response)
                        except (ValueError, TypeError):
                            return _PlanningCallResult(None, fallback=True)
                        repaired = True
                stored = cache.put(
                    self._identity,
                    request,
                    response,
                    successful_path="REPAIR" if repaired else "MAIN",
                    validate=validate,
                )
                return _PlanningCallResult(stored.response, repair_used=repaired)


def _audio_boundary_is_suspicious(
    index: int,
    batch_drafts: list[list[AudioChapterDraft]],
    batch_results: tuple[_PlanningBatchResult, ...],
    segments: tuple[AudioBaseSegment, ...],
    document_config: AudioDocumentConfig,
) -> bool:
    left = batch_drafts[index][-1]
    right = batch_drafts[index + 1][0]
    left_title = _audio_draft_title_key(left.title_hint)
    right_title = _audio_draft_title_key(right.title_hint)
    same_title = bool(left_title and right_title) and (
        left_title == right_title
        or left_title in right_title
        or right_title in left_title
    )
    minimum = _audio_minimum_chapter_duration_ms(document_config)
    return (
        same_title
        or _audio_boundary_draft_duration_ms(left, segments) < minimum
        or _audio_boundary_draft_duration_ms(right, segments) < minimum
        or batch_results[index].repair_used
        or batch_results[index + 1].repair_used
        or batch_results[index].fallback
        or batch_results[index + 1].fallback
    )


def _audio_draft_title_key(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", value).casefold()
        if char.isalnum()
    )


def _audio_boundary_draft_duration_ms(
    draft: AudioChapterDraft,
    segments: tuple[AudioBaseSegment, ...],
) -> int:
    segment_by_id = {item.segment_id: item for item in segments}
    return (
        segment_by_id[draft.segment_refs[-1]].end_ms
        - segment_by_id[draft.segment_refs[0]].start_ms
    )


def _audio_boundary_request(
    suspicious: tuple[int, ...],
    batch_drafts: list[list[AudioChapterDraft]],
    batch_results: tuple[_PlanningBatchResult, ...],
    transcript_evidence: tuple[AudioTranscriptEvidence, ...],
    segments: tuple[AudioBaseSegment, ...],
) -> AudioChapterBoundaryCoordinationRequest | None:
    evidence_by_id = {item.evidence_id: item for item in transcript_evidence}
    segments_by_id = {
        item.segment_id: item
        for result in batch_results
        for item in result.request.segments
    }
    boundaries: list[AudioChapterBoundaryInput] = []
    for index in suspicious:
        if len(boundaries) >= 63:
            _LOGGER.warning(
                "音频章节边界协调跳过部分边界 boundary_index=%d reason=REQUEST_SIZE_LIMIT",
                index,
            )
            break
        left = batch_drafts[index][-1]
        right = batch_drafts[index + 1][0]
        left_evidence_ids = {
            evidence_id
            for segment_id in left.segment_refs
            for evidence_id in segments_by_id[segment_id].evidence_refs
        }
        right_evidence_ids = {
            evidence_id
            for segment_id in right.segment_refs
            for evidence_id in segments_by_id[segment_id].evidence_refs
        }
        left_evidence = tuple(
            _truncate_audio_boundary_evidence(evidence_by_id[item.evidence_id])
            for item in transcript_evidence
            if item.evidence_id in left_evidence_ids
        )[-2:]
        right_evidence = tuple(
            _truncate_audio_boundary_evidence(evidence_by_id[item.evidence_id])
            for item in transcript_evidence
            if item.evidence_id in right_evidence_ids
        )[:2]
        boundary = AudioChapterBoundaryInput(
            boundary_index=index,
            left_title_hint=left.title_hint,
            right_title_hint=right.title_hint,
            left_duration_ms=_audio_boundary_draft_duration_ms(left, segments),
            right_duration_ms=_audio_boundary_draft_duration_ms(right, segments),
            left_tail_evidence=left_evidence,
            right_head_evidence=right_evidence,
        )
        try:
            AudioChapterBoundaryCoordinationRequest(
                boundaries=tuple((*boundaries, boundary)),
                prompt_version="audio-chapter-boundary-coordinator-v1",
            )
        except ValueError as error:
            if "64 KiB" not in str(error):
                raise
            _LOGGER.warning(
                "音频章节边界协调跳过部分边界 boundary_index=%d reason=REQUEST_SIZE_LIMIT",
                index,
            )
            break
        boundaries.append(boundary)
    if not boundaries:
        return None
    return AudioChapterBoundaryCoordinationRequest(
        boundaries=tuple(boundaries),
        prompt_version="audio-chapter-boundary-coordinator-v1",
    )


def _truncate_audio_boundary_evidence(
    evidence: AudioTranscriptEvidence,
) -> AudioTranscriptEvidence:
    if len(evidence.text) <= 240:
        return evidence
    return evidence.model_copy(update={"text": evidence.text[:240]})


def _apply_audio_boundary_decisions(
    response: AudioChapterBoundaryCoordinationResponse,
    suspicious: tuple[int, ...],
    batch_drafts: list[list[AudioChapterDraft]],
    segments: tuple[AudioBaseSegment, ...],
) -> None:
    segment_by_id = {item.segment_id: item for item in segments}
    consumed: set[int] = set()
    suspicious_set = set(suspicious)
    for decision in sorted(response.decisions, key=lambda item: item.boundary_index):
        index = decision.boundary_index
        if index not in suspicious_set or decision.decision != "MERGE":
            continue
        if not batch_drafts[index] or not batch_drafts[index + 1]:
            _LOGGER.info(
                "音频章节边界合并拒绝 boundary_index=%d reason=CHAPTER_ALREADY_CONSUMED",
                index,
            )
            continue
        left = batch_drafts[index][-1]
        right = batch_drafts[index + 1][0]
        if id(left) in consumed or id(right) in consumed:
            _LOGGER.info(
                "音频章节边界合并拒绝 boundary_index=%d reason=CHAPTER_ALREADY_CONSUMED",
                index,
            )
            continue
        if segment_by_id[left.segment_refs[-1]].end_ms != segment_by_id[
            right.segment_refs[0]
        ].start_ms:
            _LOGGER.info(
                "音频章节边界合并拒绝 boundary_index=%d reason=NON_CONTIGUOUS",
                index,
            )
            continue
        if not _audio_drafts_can_merge(left, right, segment_by_id):
            _LOGGER.info(
                "音频章节边界合并拒绝 boundary_index=%d reason=MERGE_CONSTRAINT",
                index,
            )
            continue
        merged = _merge_audio_drafts(left, right)
        if decision.merged_title_hint:
            merged = merged.model_copy(update={"title_hint": decision.merged_title_hint})
        batch_drafts[index][-1] = merged
        del batch_drafts[index + 1][0]
        consumed.update({id(left), id(right)})
        _LOGGER.info("音频章节边界协调完成 status=MERGED boundary_index=%d", index)


def _audio_drafts_can_merge(
    left: AudioChapterDraft,
    right: AudioChapterDraft,
    segment_by_id: dict[str, AudioBaseSegment],
) -> bool:
    return (
        segment_by_id[right.segment_refs[-1]].end_ms
        - segment_by_id[left.segment_refs[0]].start_ms
        <= _MAX_CHAPTER_DURATION_MS
    )


def _merge_audio_drafts(
    left: AudioChapterDraft,
    right: AudioChapterDraft,
) -> AudioChapterDraft:
    return AudioChapterDraft(
        segment_refs=(*left.segment_refs, *right.segment_refs),
        title_hint=left.title_hint,
    )


def _audio_minimum_chapter_duration_ms(config: AudioDocumentConfig) -> int:
    return 120_000 if config.chapter_granularity == "coarse" else _MIN_CHAPTER_DURATION_MS


def _plan_repair_request(
    request: AudioChapterPlanningRequest,
    invalid: AudioInvalidModelResponse,
) -> AudioChapterPlanRepairRequest:
    return AudioChapterPlanRepairRequest(
        request=request,
        invalid_response=invalid,
        allowed_segment_ids=tuple(item.segment_id for item in request.segments),
        prompt_version="audio-chapter-planner-repair-v1",
    )


def _invalid_planning_local_error(
    response: AudioChapterPlanningResponse | None,
    error: BaseException | None = None,
) -> AudioInvalidModelResponse:
    reason = _planning_validation_reason(error) if error is not None else "audio_response:invalid"
    if response is None:
        return AudioInvalidModelResponse(
            content_sha256="f" * 64,
            validation_errors=(reason,),
        )
    parsed = response.model_dump(mode="json")
    raw = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return audio_invalid_model_response(raw, (reason,), parsed_json=parsed)


def _planning_validation_reason(error: BaseException) -> str:
    message = str(error)
    if "越界" in message:
        return "segment_range:out_of_bounds"
    if "不连续" in message:
        return "segment_range:not_contiguous"
    if "为空" in message:
        return "segment_range:empty"
    if "重叠" in message or "重复" in message:
        return "segment_range:overlap"
    if "完整覆盖" in message or "完整分区" in message:
        return "segment_range:incomplete_coverage"
    return "audio_response:invalid"


def _log_planning_batch_start(
    request: AudioChapterPlanningRequest,
    index: int,
    total: int,
) -> None:
    data = prompt_for_audio_planning(request)[2]
    _LOGGER.info(
        "音频章节规划批次开始 batch=%d/%d segments=%d segment_start_ms=%d "
        "segment_end_ms=%d chars=%d bytes=%d",
        index + 1,
        total,
        len(request.segments),
        request.segments[0].start_ms,
        request.segments[-1].end_ms,
        len(data),
        len(data.encode("utf-8")),
    )


def _log_planning_validation_failure(
    request: AudioChapterPlanningRequest,
    validation_errors: tuple[str, ...],
) -> None:
    _LOGGER.warning(
        "音频章节规划响应校验失败 segment_start_ms=%d segment_end_ms=%d validation_errors=%s",
        request.segments[0].start_ms,
        request.segments[-1].end_ms,
        ",".join(validation_errors[:8]),
    )


def _log_planning_batch_finished(
    request: AudioChapterPlanningRequest,
    index: int,
    total: int,
    succeeded: bool,
    started_at: float,
) -> None:
    data = prompt_for_audio_planning(request)[2]
    _LOGGER.info(
        "音频章节规划批次完成 batch=%d/%d segments=%d segment_start_ms=%d "
        "segment_end_ms=%d chars=%d bytes=%d elapsed_ms=%d status=%s",
        index + 1,
        total,
        len(request.segments),
        request.segments[0].start_ms,
        request.segments[-1].end_ms,
        len(data),
        len(data.encode("utf-8")),
        max(0, round((time.monotonic() - started_at) * 1_000)),
        "SUCCEEDED" if succeeded else "FALLBACK",
    )


def _validate_inputs(
    duration_ms: int,
    segments: tuple[AudioBaseSegment, ...],
    transcript: tuple[AudioTranscriptEvidence, ...],
) -> None:
    if not segments or segments[0].start_ms != 0 or segments[-1].end_ms != duration_ms:
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "音频片段未覆盖完整时长")
    if any(left.end_ms != right.start_ms for left, right in pairwise(segments)):
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "音频片段必须连续")
    segment_ids = tuple(item.segment_id for item in segments)
    if len(segment_ids) != len(set(segment_ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_SEGMENT_ID, "音频片段标识不得重复")
    ids = tuple(item.evidence_id for item in transcript)
    if len(ids) != len(set(ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "音频转写证据标识不得重复")
    if tuple(ref for item in segments for ref in item.evidence_refs) != ids:
        raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "音频片段未按顺序完整引用转写")
    evidence_by_id = {item.evidence_id: item for item in transcript}
    if any(item.end_ms > duration_ms for item in transcript):
        raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_SEGMENT, "音频转写证据超出音频范围")
    for segment in segments:
        if any(
            evidence_by_id[ref].start_ms < segment.start_ms
            or evidence_by_id[ref].end_ms > segment.end_ms
            for ref in segment.evidence_refs
        ):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
                "音频转写证据不在引用它的音频片段内",
            )


def _requests(
    title: str,
    duration: int,
    segments: tuple[AudioBaseSegment, ...],
    transcript: tuple[AudioTranscriptEvidence, ...],
    config: AudioDocumentConfig,
    max_chars: int,
    max_bytes: int,
    *,
    max_batches: int | None = None,
) -> tuple[AudioChapterPlanningRequest, ...]:
    by_id = {item.evidence_id: item for item in transcript}
    requests: list[AudioChapterPlanningRequest] = []
    batch: list[AudioBaseSegment] = []
    for segment in segments:
        if len(batch) >= _MAX_BATCH_SEGMENTS:
            if max_batches is not None and len(requests) >= max_batches:
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "音频章节规划批次数超过上限",
                )
            requests.append(
                _build_request(title, duration, tuple(batch), by_id, config),
            )
            batch = []
        candidate = (*batch, segment)
        request = _build_request(title, duration, candidate, by_id, config)
        if _request_fits(request, max_chars, max_bytes):
            batch.append(segment)
            continue
        if not batch:
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "单个音频基础片段超过章节规划输入预算",
            )
        if max_batches is not None and len(requests) >= max_batches:
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "音频章节规划批次数超过上限",
            )
        requests.append(
            _build_request(title, duration, tuple(batch), by_id, config),
        )
        batch = [segment]
        single = _build_request(title, duration, (segment,), by_id, config)
        if not _request_fits(single, max_chars, max_bytes):
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "单个音频基础片段超过章节规划输入预算",
            )
    if batch:
        if max_batches is not None and len(requests) >= max_batches:
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "音频章节规划批次数超过上限",
            )
        requests.append(
            _build_request(title, duration, tuple(batch), by_id, config),
        )
    return tuple(requests)


def _validate_chapter_count_feasibility(
    segments: tuple[AudioBaseSegment, ...],
    maximum: int,
) -> None:
    if any(segment.duration_ms > _MAX_CHAPTER_DURATION_MS for segment in segments):
        raise VideoDemoError(
            ErrorCode.INPUT_BUDGET_EXCEEDED,
            "存在无法放入 5 分钟章节的音频基础片段",
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
            "音频时长与安全片段边界无法满足章节数量上限",
        )


def _build_request(
    title: str,
    duration: int,
    segments: tuple[AudioBaseSegment, ...],
    evidence_by_id: dict[str, AudioTranscriptEvidence],
    config: AudioDocumentConfig,
) -> AudioChapterPlanningRequest:
    evidence = tuple(evidence_by_id[ref] for item in segments for ref in item.evidence_refs)
    return AudioChapterPlanningRequest(
        title_hint=title,
        duration_ms=duration,
        segments=segments,
        transcript_evidence=evidence,
        document_config=config,
        prompt_version="audio-chapter-planner-v1",
    )


def _request_fits(
    request: AudioChapterPlanningRequest,
    max_chars: int,
    max_bytes: int,
) -> bool:
    data = prompt_for_audio_planning(request)[2]
    return len(data) <= max_chars and len(data.encode("utf-8")) <= max_bytes


def _rule_drafts(
    segments: tuple[AudioBaseSegment, ...],
    transcript: tuple[AudioTranscriptEvidence, ...],
    config: AudioDocumentConfig,
) -> tuple[AudioChapterDraft, ...]:
    groups = _group_segments_near_target(
        segments,
        _TARGET_DURATION_MS[config.chapter_granularity],
    )
    minimum = 120_000 if config.chapter_granularity == "coarse" else _MIN_CHAPTER_DURATION_MS
    if len(groups) > 1 and (
        groups[-1][-1].end_ms - groups[-1][0].start_ms < minimum
        and groups[-1][-1].end_ms - groups[-2][0].start_ms <= _MAX_CHAPTER_DURATION_MS
    ):
        groups[-2].extend(groups.pop())
    evidence_by_id = {item.evidence_id: item for item in transcript}
    return tuple(
        AudioChapterDraft(
            segment_refs=tuple(item.segment_id for item in group),
            title_hint=next(
                (
                    evidence_by_id[ref].text[:200]
                    for item in group
                    for ref in item.evidence_refs
                    if ref in evidence_by_id
                ),
                f"章节 {index}",
            ),
        )
        for index, group in enumerate(groups, 1)
    )


def _group_segments_near_target(
    segments: tuple[AudioBaseSegment, ...],
    target_duration_ms: int,
) -> list[list[AudioBaseSegment]]:
    groups: list[list[AudioBaseSegment]] = []
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


def _merge_short_drafts(
    drafts: tuple[AudioChapterDraft, ...], segments: tuple[AudioBaseSegment, ...], granularity: str
) -> tuple[AudioChapterDraft, ...]:
    by_id = {item.segment_id: item for item in segments}
    minimum = 120_000 if granularity == "coarse" else _MIN_CHAPTER_DURATION_MS
    normalized = list(drafts)
    index = 0
    while index < len(normalized):
        current = normalized[index]
        duration = by_id[current.segment_refs[-1]].end_ms - by_id[current.segment_refs[0]].start_ms
        if duration >= minimum:
            index += 1
            continue
        if index > 0 and _audio_drafts_can_merge(
            normalized[index - 1],
            current,
            by_id,
        ):
            normalized[index - 1 : index + 1] = [
                _merge_audio_drafts(normalized[index - 1], current),
            ]
            index = max(0, index - 1)
            continue
        if index + 1 < len(normalized) and _audio_drafts_can_merge(
            current,
            normalized[index + 1],
            by_id,
        ):
            normalized[index : index + 2] = [
                _merge_audio_drafts(current, normalized[index + 1]),
            ]
            continue
        index += 1
    return tuple(normalized)


def _normalize_draft_count(
    drafts: tuple[AudioChapterDraft, ...],
    segments: tuple[AudioBaseSegment, ...],
    maximum: int,
) -> tuple[AudioChapterDraft, ...]:
    if len(drafts) <= maximum:
        return drafts
    by_id = {item.segment_id: item for item in segments}
    normalized = list(drafts)
    while len(normalized) > maximum:
        merge_at = next(
            (
                index
                for index in range(len(normalized) - 1)
                if _draft_duration(normalized[index], normalized[index + 1], by_id)
                <= _MAX_CHAPTER_DURATION_MS
            ),
            None,
        )
        if merge_at is None:
            raise VideoDemoError(
                ErrorCode.INPUT_BUDGET_EXCEEDED,
                "音频章节数量超过上限且无法安全合并",
            )
        left, right = normalized[merge_at : merge_at + 2]
        normalized[merge_at : merge_at + 2] = [
            AudioChapterDraft(
                segment_refs=(*left.segment_refs, *right.segment_refs),
                title_hint=left.title_hint,
            ),
        ]
    return tuple(normalized)


def _draft_duration(
    left: AudioChapterDraft,
    right: AudioChapterDraft,
    by_id: dict[str, AudioBaseSegment],
) -> int:
    return by_id[right.segment_refs[-1]].end_ms - by_id[left.segment_refs[0]].start_ms


def _materialize(
    asset_sha256: Sha256,
    drafts: tuple[AudioChapterDraft, ...],
    segments: tuple[AudioBaseSegment, ...],
) -> tuple[AudioChapterPlan, ...]:
    by_id = {item.segment_id: item for item in segments}
    plans = tuple(
        AudioChapterPlan(
            chapter_id=stable_identifier(
                "audio_chapter",
                {"asset_sha256": asset_sha256, "segment_refs": list(draft.segment_refs)},
            ),
            start_ms=by_id[draft.segment_refs[0]].start_ms,
            end_ms=by_id[draft.segment_refs[-1]].end_ms,
            segment_refs=draft.segment_refs,
            title_hint=draft.title_hint,
        )
        for draft in drafts
    )
    if (
        plans[0].start_ms != 0
        or plans[-1].end_ms != segments[-1].end_ms
        or any(left.end_ms != right.start_ms for left, right in pairwise(plans))
    ):
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "音频章节计划必须连续覆盖")
    actual_refs = tuple(ref for plan in plans for ref in plan.segment_refs)
    expected_refs = tuple(item.segment_id for item in segments)
    if actual_refs != expected_refs:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_SEGMENT_REFERENCE,
            "音频章节计划必须完整覆盖音频片段",
        )
    return plans


def _validate_response(
    response: AudioChapterPlanningResponse, request: AudioChapterPlanningRequest
) -> None:
    actual = tuple(ref for draft in response.chapter_drafts for ref in draft.segment_refs)
    expected = tuple(item.segment_id for item in request.segments)
    if actual != expected:
        raise ValueError("音频章节草稿必须完整覆盖当前批次片段")
    segment_by_id = {item.segment_id: item for item in request.segments}
    if any(
        segment_by_id[draft.segment_refs[-1]].end_ms
        - segment_by_id[draft.segment_refs[0]].start_ms
        > _MAX_CHAPTER_DURATION_MS
        for draft in response.chapter_drafts
    ):
        raise ValueError("音频章节草稿超过 5 分钟")
