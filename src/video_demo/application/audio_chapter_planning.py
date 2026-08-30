"""音频独立章节规划：只处理音频片段和转写证据。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
    ) -> None:
        if (
            min(max_input_chars, max_input_bytes, max_chapters, max_planning_batches, concurrency)
            < 1
        ):
            raise ValueError("音频规划预算必须大于 0")
        if concurrency > 2 or max_chapters > 240 or max_planning_batches > 64:
            raise ValueError("音频规划预算超过硬上限")
        self._port = text_port
        self._identity = identity
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._max_chapters = max_chapters
        self._max_batches = max_planning_batches
        self._concurrency = concurrency
        self._wait_timeout = invocation_wait_timeout_seconds

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
        requests = _requests(
            title_hint,
            duration_ms,
            segments,
            transcript_evidence,
            document_config,
            self._max_input_chars,
            self._max_input_bytes,
        )
        if len(requests) > self._max_batches:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频章节规划批次数超过上限")
        counters = _Counters()
        results: list[AudioChapterPlanningResponse | None] = [None] * len(requests)
        fallback: set[int] = set()
        with ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="audio-planning"
        ) as executor:
            futures: dict[Future[AudioChapterPlanningResponse | None], tuple[int, float]] = {}
            next_index = 0
            while next_index < len(requests) or futures:
                if is_cancel_requested():
                    for future in futures:
                        future.cancel()
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_index < len(requests) and len(futures) < self._concurrency:
                    counters.logical += 1
                    future = executor.submit(
                        self._call,
                        cache,
                        requests[next_index],
                        counters,
                        is_cancel_requested,
                    )
                    request = requests[next_index]
                    _log_planning_batch_start(request, next_index, len(requests))
                    futures[future] = (next_index, time.monotonic())
                    next_index += 1
                done, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    index, started_at = futures.pop(future)
                    result = future.result()
                    results[index] = result
                    if result is None:
                        fallback.add(index)
                    _log_planning_batch_finished(
                        requests[index],
                        index,
                        len(requests),
                        result is not None,
                        started_at,
                    )
        drafts: list[AudioChapterDraft] = []
        for index, request in enumerate(requests):
            response = results[index]
            drafts.extend(
                response.chapter_drafts
                if response is not None
                else _rule_drafts(request.segments, request.transcript_evidence, document_config)
            )
        normalized = _normalize_draft_count(
            _merge_short_drafts(tuple(drafts), segments, document_config.chapter_granularity),
            segments,
            self._max_chapters,
        )
        if len(normalized) > self._max_chapters:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频章节数量超过上限")
        plans = _materialize(asset_sha256, normalized, segments)
        warning = tuple(
            f"AUDIO_CHAPTER_PLANNING_FALLBACK:{plan.chapter_id}"
            for plan in plans
            if any(
                index in fallback
                for index, request in enumerate(requests)
                if set(plan.segment_refs) & {item.segment_id for item in request.segments}
            )
        )
        return AudioChapterPlanningBatch(
            plans=plans,
            warnings=warning,
            status="PARTIAL_SUCCEEDED" if fallback else "SUCCEEDED",
            metrics={
                "audio_planner_logical_calls": counters.logical,
                "audio_planner_provider_attempts": counters.attempts,
                "audio_planner_repairs": counters.repairs,
                "audio_planner_cache_hits": counters.cache_hits,
                "audio_planner_fallback_chapters": len(warning),
            },
        )

    def _call(
        self,
        cache: DocumentModelCache,
        request: AudioChapterPlanningRequest,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioChapterPlanningResponse | None:
        def validate(response: AudioChapterPlanningResponse) -> None:
            _validate_response(response, request)

        cached = cache.get(self._identity, request, AudioChapterPlanningResponse, validate)
        if cached is not None:
            counters.cache_hit()
            return cached.response
        try:
            with cache.invocation_lock(
                self._identity,
                request,
                wait_timeout_seconds=self._wait_timeout,
                is_cancel_requested=is_cancel_requested,
            ):
                cached = cache.get(self._identity, request, AudioChapterPlanningResponse, validate)
                if cached is not None:
                    counters.cache_hit()
                    return cached.response
                repaired = False
                invalid: AudioInvalidModelResponse | None = None
                try:
                    response = self._port.plan_chapters(
                        request, on_provider_attempt=counters.attempt
                    )
                    validate(response)
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
                    }:
                        return None
                    if error.code == ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE:
                        return None
                    if error.code != ErrorCode.TEXT_LLM_RESPONSE_INVALID:
                        raise
                    invalid = _invalid_planning_provider_error(error)
                    _log_planning_validation_failure(request, invalid.validation_errors)
                except (ValueError, TypeError) as error:
                    invalid = _invalid_planning_local_error(error)
                    _log_planning_validation_failure(request, invalid.validation_errors)
                if invalid is not None:
                    counters.repair()
                    try:
                        response = self._port.repair_chapter_plan(
                            _plan_repair_request(request, invalid),
                            on_provider_attempt=counters.attempt,
                        )
                        validate(response)
                        repaired = True
                    except AudioModelResponseValidationError:
                        return None
                    except VideoDemoError as repair_error:
                        if repair_error.code == ErrorCode.JOB_CANCELLED:
                            raise
                        if repair_error.code in {
                            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                            ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE,
                            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                        }:
                            return None
                        raise
                    except (ValueError, TypeError):
                        return None
                return cache.put(
                    self._identity,
                    request,
                    response,
                    successful_path="REPAIR" if repaired else "MAIN",
                    validate=validate,
                ).response
        except VideoDemoError as error:
            if error.code in {
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE,
            }:
                return None
            raise


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


def _invalid_planning_provider_error(error: VideoDemoError) -> AudioInvalidModelResponse:
    return audio_invalid_model_response(
        error.code.value.encode("utf-8"),
        ("provider_response:invalid", f"provider_code:{error.code.value}"),
    )


def _invalid_planning_local_error(error: BaseException | None = None) -> AudioInvalidModelResponse:
    reason = _planning_validation_reason(error) if error is not None else "audio_response:invalid"
    return AudioInvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=(reason,),
    )


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
    ids = tuple(item.evidence_id for item in transcript)
    if tuple(ref for item in segments for ref in item.evidence_refs) != ids:
        raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "音频片段未按顺序完整引用转写")


def _requests(
    title: str,
    duration: int,
    segments: tuple[AudioBaseSegment, ...],
    transcript: tuple[AudioTranscriptEvidence, ...],
    config: AudioDocumentConfig,
    max_chars: int,
    max_bytes: int,
) -> tuple[AudioChapterPlanningRequest, ...]:
    by_id = {item.evidence_id: item for item in transcript}
    requests: list[AudioChapterPlanningRequest] = []
    batch: list[AudioBaseSegment] = []
    for segment in segments:
        if len(batch) >= _MAX_BATCH_SEGMENTS:
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
        requests.append(
            _build_request(title, duration, tuple(batch), by_id, config),
        )
    return tuple(requests)


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
        if index + 1 < len(normalized):
            merged = (*current.segment_refs, *normalized[index + 1].segment_refs)
            if by_id[merged[-1]].end_ms - by_id[merged[0]].start_ms <= _MAX_CHAPTER_DURATION_MS:
                normalized[index : index + 2] = [
                    AudioChapterDraft(segment_refs=merged, title_hint=current.title_hint)
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
    return plans


def _validate_response(
    response: AudioChapterPlanningResponse, request: AudioChapterPlanningRequest
) -> None:
    actual = tuple(ref for draft in response.chapter_drafts for ref in draft.segment_refs)
    expected = tuple(item.segment_id for item in request.segments)
    if actual != expected:
        raise ValueError("音频章节草稿必须完整覆盖当前批次片段")
