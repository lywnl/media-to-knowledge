"""音频章节写作内核。

该模块只消费音频章节计划和转写证据，直接生成音频结果，不经过其他媒体结果。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import pairwise
from threading import Event, Lock
from typing import Literal

from pydantic import BaseModel, Field

from video_demo.application.audio_rendering import (
    RenderedAudioDocument,
    clean_audio_text,
    render_audio_markdown,
)
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.audio_plan import (
    AudioBulletListBlock,
    AudioChapterPlan,
    AudioDocumentConfig,
    AudioGroundedClaim,
    AudioParagraphBlock,
    AudioQuoteBlock,
    AudioTranscriptEvidence,
)
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.audio_document_port import (
    AudioChapterWritingRepairRequest,
    AudioChapterWritingRequest,
    AudioChapterWritingResponse,
    AudioDocumentTextPort,
    AudioGlobalChapterInput,
    AudioGlobalWritingRepairRequest,
    AudioGlobalWritingRequest,
    AudioGlobalWritingResponse,
    AudioInvalidModelResponse,
    AudioModelResponseValidationError,
    allowed_audio_evidence_ids,
    audio_invalid_model_response,
)
from video_demo.integrations.audio_document_prompts import (
    prompt_for_audio_global,
    prompt_for_audio_global_repair,
    prompt_for_audio_writing,
    prompt_for_audio_writing_repair,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_DETAIL_BODY_LIMITS = {"concise": 800, "standard": 2_000, "detailed": 4_000}
_DETAIL_SUMMARY_LIMITS = {"concise": 160, "standard": 300, "detailed": 500}
_MAX_INPUT_CHARS = 60_000
_MAX_INPUT_BYTES = 1_048_576
_PLACEHOLDER = "本时段未提取到可验证语义内容"
_LOGGER = logging.getLogger(__name__)


class _Counters:
    def __init__(self) -> None:
        self._lock = Lock()
        self.chapter_logical = 0
        self.chapter_attempts = 0
        self.chapter_repairs = 0
        self.chapter_cache_hits = 0
        self.chapter_fallbacks = 0
        self.global_logical = 0
        self.global_attempts = 0
        self.global_repairs = 0
        self.global_cache_hits = 0
        self.global_fallbacks = 0

    def chapter_attempt(self) -> None:
        with self._lock:
            self.chapter_attempts += 1

    def global_attempt(self) -> None:
        with self._lock:
            self.global_attempts += 1

    def chapter_repair(self) -> None:
        with self._lock:
            self.chapter_repairs += 1

    def global_repair(self) -> None:
        with self._lock:
            self.global_repairs += 1

    def chapter_cache_hit(self) -> None:
        with self._lock:
            self.chapter_cache_hits += 1

    def global_cache_hit(self) -> None:
        with self._lock:
            self.global_cache_hits += 1

    def chapter_fallback(self) -> None:
        with self._lock:
            self.chapter_fallbacks += 1

    def global_fallback(self) -> None:
        with self._lock:
            self.global_fallbacks += 1

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "audio_chapter_writer_logical_calls": self.chapter_logical,
                "audio_chapter_writer_provider_attempts": self.chapter_attempts,
                "audio_chapter_writer_structure_repairs": self.chapter_repairs,
                "audio_chapter_writer_cache_hits": self.chapter_cache_hits,
                "audio_chapter_writer_fallback_chapters": self.chapter_fallbacks,
                "audio_global_editor_logical_calls": self.global_logical,
                "audio_global_editor_provider_attempts": self.global_attempts,
                "audio_global_editor_structure_repairs": self.global_repairs,
                "audio_global_editor_cache_hits": self.global_cache_hits,
                "audio_global_editor_fallbacks": self.global_fallbacks,
            }


class _CacheCommitGate:
    """把取消检查与不可撤销的缓存写入收敛到一个线性化边界。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._finished = Event()
        self._state: Literal["PRE_COMMIT", "COMMITTING", "FINISHED", "CANCELLED"] = "PRE_COMMIT"

    def begin_commit(self, is_cancel_requested: Callable[[], bool]) -> None:
        with self._lock:
            if self._state == "CANCELLED" or is_cancel_requested():
                self._state = "CANCELLED"
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
            if self._state != "PRE_COMMIT":
                raise RuntimeError("音频缓存提交门状态非法")
            self._state = "COMMITTING"

    def finish_commit(self) -> None:
        with self._lock:
            if self._state != "COMMITTING":
                raise RuntimeError("音频缓存提交门状态非法")
            self._state = "FINISHED"
            self._finished.set()

    def cancel(self) -> Event | None:
        with self._lock:
            if self._state == "PRE_COMMIT":
                self._state = "CANCELLED"
                return None
            if self._state == "COMMITTING":
                return self._finished
            return None


class AudioWritingContext(FrozenModel):
    run_id: StableId
    asset_sha256: Sha256
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    transcript_source: Literal["SUBTITLE", "ASR", "NONE"]
    document_config: AudioDocumentConfig


@dataclass(frozen=True, slots=True)
class AudioWrittenDocument:
    result: AudioUnderstandingResult
    document: RenderedAudioDocument
    warnings: tuple[str, ...]
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    metrics: dict[str, int]


@dataclass(frozen=True, slots=True)
class _ChapterOutcome:
    chapter: AudioChapter
    degraded: bool
    provider_attempts: int
    structure_repairs: int
    cache_hits: int


class AudioDocumentWriter:
    def __init__(
        self,
        text_port: AudioDocumentTextPort,
        *,
        chapter_identity: ModelInvocationIdentity,
        global_identity: ModelInvocationIdentity,
        concurrency: int,
        max_input_chars: int,
        max_input_bytes: int,
        invocation_wait_timeout_seconds: float,
    ) -> None:
        if not 1 <= concurrency <= 2:
            raise ValueError("音频章节写作并发必须位于 1~2")
        if min(max_input_chars, max_input_bytes) < 1 or invocation_wait_timeout_seconds <= 0:
            raise ValueError("音频写作预算和锁等待时间必须大于 0")
        if max_input_chars > _MAX_INPUT_CHARS or max_input_bytes > _MAX_INPUT_BYTES:
            raise ValueError("音频文本输入预算超过硬上限")
        self._port = text_port
        self._chapter_identity = chapter_identity
        self._global_identity = global_identity
        self._concurrency = concurrency
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._wait_timeout = invocation_wait_timeout_seconds

    def write(
        self,
        context: AudioWritingContext,
        plans: tuple[AudioChapterPlan, ...],
        transcript_evidence: tuple[AudioTranscriptEvidence, ...],
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioWrittenDocument:
        _validate_inputs(context, plans, transcript_evidence)
        requests = tuple(
            _chapter_request(
                context,
                plan,
                transcript_evidence,
                self._max_input_chars,
                self._max_input_bytes,
            )
            for plan in plans
        )
        counters = _Counters()
        chapters: list[AudioChapter | None] = [None] * len(requests)
        warnings: list[str] = []
        pending: list[tuple[int, AudioChapterWritingRequest]] = []
        for index, request in enumerate(requests):
            if request is None:
                chapters[index] = _empty_audio_chapter(plans[index])
                continue
            counters.chapter_logical += 1
            pending.append((index, request))

        executor = ThreadPoolExecutor(
            max_workers=self._concurrency,
            thread_name_prefix="audio-writing",
        )
        futures: dict[Future[_ChapterOutcome], tuple[int, _CacheCommitGate]] = {}
        next_index = 0
        try:
            while next_index < len(pending) or futures:
                if is_cancel_requested():
                    _cancel_before_or_wait_for_commits(gate for _, gate in futures.values())
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_index < len(pending) and len(futures) < self._concurrency:
                    index, request = pending[next_index]
                    gate = _CacheCommitGate()
                    future = executor.submit(
                        self._write_one,
                        request,
                        context.transcript_source,
                        cache,
                        is_cancel_requested,
                        gate,
                    )
                    futures[future] = (index, gate)
                    next_index += 1
                done, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    index, _gate = futures.pop(future)
                    outcome = future.result()
                    chapters[index] = outcome.chapter
                    if outcome.degraded:
                        warnings.append(
                            f"AUDIO_CHAPTER_WRITING_FALLBACK:{outcome.chapter.chapter_id}",
                        )
                    if outcome.degraded:
                        counters.chapter_fallback()
                    counters.chapter_attempts += outcome.provider_attempts
                    counters.chapter_repairs += outcome.structure_repairs
                    counters.chapter_cache_hits += outcome.cache_hits
        except BaseException:
            _cancel_before_or_wait_for_commits(gate for _, gate in futures.values())
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        complete = tuple(item for item in chapters if item is not None)
        overview, global_degraded = self._write_global(
            context,
            complete,
            cache,
            counters,
            is_cancel_requested,
        )
        result = AudioUnderstandingResult(
            run_id=context.run_id,
            asset_sha256=context.asset_sha256,
            summary=AudioDocumentSummary(
                title=context.document_config.document_title or context.title_hint,
                duration_ms=context.duration_ms,
                overview_zh=overview,
            ),
            chapters=complete,
        )
        rendered = render_audio_markdown(result)
        if global_degraded:
            warnings.append("AUDIO_GLOBAL_WRITING_FALLBACK")
        unique_warnings = tuple(dict.fromkeys(warnings))
        return AudioWrittenDocument(
            result=result,
            document=rendered,
            warnings=unique_warnings,
            status="PARTIAL_SUCCEEDED" if unique_warnings else "SUCCEEDED",
            metrics=counters.metrics(),
        )

    def _write_one(
        self,
        request: AudioChapterWritingRequest,
        source: str,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
        commit_gate: _CacheCommitGate,
    ) -> _ChapterOutcome:
        started_at = time.monotonic()
        _log_audio_chapter_start(request)
        counters = _Counters()

        def validate(response: AudioChapterWritingResponse) -> None:
            _validate_response(response, request)

        try:
            cached = cache.get(
                self._chapter_identity,
                request,
                AudioChapterWritingResponse,
                validate,
            )
            if cached is not None:
                counters.chapter_cache_hit()
                outcome = _ChapterOutcome(
                    _materialize_chapter(request.chapter, cached.response, source),
                    False,
                    0,
                    0,
                    1,
                )
                _log_audio_chapter_finished(request, outcome, started_at)
                return outcome
            with cache.invocation_lock(
                self._chapter_identity,
                request,
                wait_timeout_seconds=self._wait_timeout,
                is_cancel_requested=is_cancel_requested,
            ):
                cached = cache.get(
                    self._chapter_identity, request, AudioChapterWritingResponse, validate
                )
                if cached is not None:
                    counters.chapter_cache_hit()
                    outcome = _ChapterOutcome(
                        _materialize_chapter(request.chapter, cached.response, source),
                        False,
                        0,
                        0,
                        1,
                    )
                    _log_audio_chapter_finished(request, outcome, started_at)
                    return outcome
                invalid: AudioInvalidModelResponse | None = None
                try:
                    response = self._port.write_chapter(
                        request,
                        on_provider_attempt=_cancellable_attempt_callback(
                            counters.chapter_attempt,
                            is_cancel_requested,
                        ),
                    )
                except AudioModelResponseValidationError as error:
                    _raise_if_cancelled(is_cancel_requested)
                    invalid = error.invalid_response
                    _log_audio_validation_failure(request, "main", error, counters, started_at)
                except VideoDemoError as error:
                    _raise_if_cancelled(is_cancel_requested)
                    if error.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID:
                        invalid = _invalid_audio_provider_error(error)
                        _log_audio_validation_failure(
                            request, "main", invalid, counters, started_at
                        )
                    elif _is_fallback_error(error):
                        _log_audio_degradation(
                            request,
                            "main",
                            "DEPENDENCY_OR_PROVIDER_FAILURE",
                            error,
                            counters,
                            started_at,
                        )
                        outcome = _fallback_outcome(request, source, counters)
                        _log_audio_chapter_finished(request, outcome, started_at)
                        return outcome
                    else:
                        raise
                else:
                    _raise_if_cancelled(is_cancel_requested)
                    try:
                        validate(response)
                    except (ValueError, TypeError) as error:
                        invalid = _invalid_audio_local_response(response, error)
                        _log_audio_validation_failure(
                            request, "main", invalid, counters, started_at
                        )
                if invalid is None:
                    path: Literal["MAIN", "REPAIR"] = "MAIN"
                else:
                    counters.chapter_repair()
                    try:
                        response = self._port.repair_chapter_writing(
                            AudioChapterWritingRepairRequest(
                                request=request,
                                invalid_response=invalid,
                                allowed_evidence_ids=allowed_audio_evidence_ids(request),
                                prompt_version="audio-chapter-writer-repair-v1",
                            ),
                            on_provider_attempt=_cancellable_attempt_callback(
                                counters.chapter_attempt,
                                is_cancel_requested,
                            ),
                        )
                    except AudioModelResponseValidationError as error:
                        _raise_if_cancelled(is_cancel_requested)
                        _log_audio_validation_failure(
                            request, "repair", error, counters, started_at
                        )
                        _log_audio_degradation(
                            request,
                            "repair",
                            "MODEL_RESPONSE_INVALID_AFTER_REPAIR",
                            error,
                            counters,
                            started_at,
                        )
                        outcome = _fallback_outcome(request, source, counters)
                        _log_audio_chapter_finished(request, outcome, started_at)
                        return outcome
                    except VideoDemoError as error:
                        _raise_if_cancelled(is_cancel_requested)
                        if _is_fallback_error(error):
                            _log_audio_degradation(
                                request,
                                "repair",
                                "DEPENDENCY_OR_PROVIDER_FAILURE_AFTER_REPAIR",
                                error,
                                counters,
                                started_at,
                            )
                            outcome = _fallback_outcome(request, source, counters)
                            _log_audio_chapter_finished(request, outcome, started_at)
                            return outcome
                        raise
                    try:
                        validate(response)
                    except (ValueError, TypeError) as error:
                        invalid = _invalid_audio_local_response(response, error)
                        _log_audio_validation_failure(
                            request, "repair", invalid, counters, started_at
                        )
                        _log_audio_degradation(
                            request,
                            "repair",
                            "MODEL_RESPONSE_INVALID_AFTER_REPAIR",
                            VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "响应校验失败"),
                            counters,
                            started_at,
                        )
                        outcome = _fallback_outcome(request, source, counters)
                        _log_audio_chapter_finished(request, outcome, started_at)
                        return outcome
                    path = "REPAIR"
                commit_gate.begin_commit(is_cancel_requested)
                try:
                    stored = cache.put(
                        self._chapter_identity,
                        request,
                        response,
                        successful_path=path,
                        validate=validate,
                    ).response
                finally:
                    commit_gate.finish_commit()
                outcome = _ChapterOutcome(
                    _materialize_chapter(request.chapter, stored, source),
                    False,
                    counters.chapter_attempts,
                    counters.chapter_repairs,
                    counters.chapter_cache_hits,
                )
                _log_audio_chapter_finished(request, outcome, started_at)
                return outcome
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_CANCELLED:
                raise
            if _is_fallback_error(error):
                outcome = _fallback_outcome(request, source, counters)
                _log_audio_degradation(
                    request, "main", "DEPENDENCY_OR_PROVIDER_FAILURE", error, counters, started_at
                )
                _log_audio_chapter_finished(request, outcome, started_at)
                return outcome
            raise
        except (ValueError, TypeError) as error:
            invalid = _invalid_audio_local_response_from_error(error)
            _log_audio_degradation(
                request,
                "main",
                "MODEL_RESPONSE_INVALID_AFTER_REPAIR",
                VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, str(error)),
                counters,
                started_at,
            )
            outcome = _fallback_outcome(request, source, counters)
            _log_audio_chapter_finished(request, outcome, started_at)
            return outcome

    def _write_global(
        self,
        context: AudioWritingContext,
        chapters: tuple[AudioChapter, ...],
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[str, bool]:
        request = _global_request(
            context,
            chapters,
            self._max_input_chars,
            self._max_input_bytes,
        )
        counters.global_logical += 1
        started_at = time.monotonic()
        _log_audio_global_start(request)
        gate = _CacheCommitGate()
        try:
            cached = cache.get(
                self._global_identity,
                request,
                AudioGlobalWritingResponse,
                _validate_global_response,
            )
            if cached is not None:
                counters.global_cache_hit()
                return cached.response.overview_zh.strip() or _fallback_overview(chapters), False
            with cache.invocation_lock(
                self._global_identity,
                request,
                wait_timeout_seconds=self._wait_timeout,
                is_cancel_requested=is_cancel_requested,
            ):
                cached = cache.get(
                    self._global_identity,
                    request,
                    AudioGlobalWritingResponse,
                    _validate_global_response,
                )
                if cached is not None:
                    counters.global_cache_hit()
                    return cached.response.overview_zh.strip() or _fallback_overview(
                        chapters
                    ), False
                invalid: AudioInvalidModelResponse | None = None
                try:
                    response = self._port.organize_document(
                        request,
                        on_provider_attempt=_cancellable_attempt_callback(
                            counters.global_attempt,
                            is_cancel_requested,
                        ),
                    )
                except AudioModelResponseValidationError as error:
                    invalid = error.invalid_response
                    _log_audio_global_validation("main", error, counters, started_at)
                except VideoDemoError as error:
                    if error.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID:
                        invalid = _invalid_audio_provider_error(error)
                        _log_audio_global_validation(
                            "main", invalid, counters, started_at
                        )
                    elif error.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE:
                        counters.global_fallback()
                        _log_audio_global_degradation(
                            "main", "DEPENDENCY_OR_PROVIDER_FAILURE", error, counters, started_at
                        )
                        return _fallback_overview(chapters), True
                    else:
                        raise
                else:
                    try:
                        _validate_global_response(response)
                    except (ValueError, TypeError) as error:
                        invalid = _invalid_audio_local_response(response, error)
                        _log_audio_global_validation("main", invalid, counters, started_at)
                if invalid is None:
                    path: Literal["MAIN", "REPAIR"] = "MAIN"
                else:
                    counters.global_repair()
                    try:
                        response = self._port.repair_global_writing(
                            AudioGlobalWritingRepairRequest(
                                request=request,
                                invalid_response=invalid,
                                allowed_chapter_ids=tuple(
                                    item.chapter_id for item in request.chapters
                                ),
                                prompt_version="audio-global-editor-repair-v1",
                            ),
                            on_provider_attempt=_cancellable_attempt_callback(
                                counters.global_attempt,
                                is_cancel_requested,
                            ),
                        )
                        _validate_global_response(response)
                    except AudioModelResponseValidationError as error:
                        counters.global_fallback()
                        _log_audio_global_validation("repair", error, counters, started_at)
                        _log_audio_global_degradation(
                            "repair",
                            "MODEL_RESPONSE_INVALID_AFTER_REPAIR",
                            error,
                            counters,
                            started_at,
                        )
                        return _fallback_overview(chapters), True
                    except VideoDemoError as error:
                        if _is_fallback_error(error):
                            counters.global_fallback()
                            _log_audio_global_degradation(
                                "repair",
                                "DEPENDENCY_OR_PROVIDER_FAILURE_AFTER_REPAIR",
                                error,
                                counters,
                                started_at,
                            )
                            return _fallback_overview(chapters), True
                        raise
                    except (ValueError, TypeError) as error:
                        counters.global_fallback()
                        _log_audio_global_validation(
                            "repair",
                            _invalid_audio_local_response(response, error),
                            counters,
                            started_at,
                        )
                        _log_audio_global_degradation(
                            "repair",
                            "MODEL_RESPONSE_INVALID_AFTER_REPAIR",
                            VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, str(error)),
                            counters,
                            started_at,
                        )
                        return _fallback_overview(chapters), True
                    path = "REPAIR"
                gate.begin_commit(is_cancel_requested)
                try:
                    response = cache.put(
                        self._global_identity,
                        request,
                        response,
                        successful_path=path,
                        validate=_validate_global_response,
                    ).response
                finally:
                    gate.finish_commit()
                _LOGGER.info(
                    "音频全局文章写作完成 status=SUCCEEDED elapsed_ms=%d",
                    max(0, round((time.monotonic() - started_at) * 1_000)),
                )
                return response.overview_zh.strip() or _fallback_overview(chapters), False
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_CANCELLED:
                raise
            if _is_fallback_error(error):
                counters.global_fallback()
                _log_audio_global_degradation(
                    "main", "DEPENDENCY_OR_PROVIDER_FAILURE", error, counters, started_at
                )
                return _fallback_overview(chapters), True
            raise


def _validate_inputs(
    context: AudioWritingContext,
    plans: tuple[AudioChapterPlan, ...],
    evidence: tuple[AudioTranscriptEvidence, ...],
) -> None:
    if not plans or plans[0].start_ms != 0 or plans[-1].end_ms != context.duration_ms:
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "音频章节计划未覆盖完整时长")
    if any(left.end_ms != right.start_ms for left, right in pairwise(plans)):
        raise VideoDemoError(ErrorCode.UNKNOWN_SEGMENT_REFERENCE, "音频章节计划必须连续")
    ids = tuple(item.evidence_id for item in evidence)
    if len(ids) != len(set(ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "音频证据标识不得重复")


def _chapter_request(
    context: AudioWritingContext,
    plan: AudioChapterPlan,
    transcript: tuple[AudioTranscriptEvidence, ...],
    max_chars: int,
    max_bytes: int,
) -> AudioChapterWritingRequest | None:
    chapter_evidence = tuple(
        sorted(
            (item for item in transcript if plan.contains(item)),
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    if not chapter_evidence:
        return None
    while True:
        request = AudioChapterWritingRequest(
            run_id=context.run_id,
            asset_sha256=context.asset_sha256,
            title_hint=plan.title_hint,
            duration_ms=context.duration_ms,
            transcript_source=context.transcript_source,
            document_config=context.document_config,
            chapter=plan,
            transcript_evidence=chapter_evidence,
            prompt_version="audio-chapter-writer-v1",
        )
        if _chapter_request_fits(request, max_chars, max_bytes):
            return request
        if len(chapter_evidence) <= 1:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频单章最小输入超过预算")
        chapter_evidence = chapter_evidence[:-1]


def _chapter_request_fits(
    request: AudioChapterWritingRequest,
    max_chars: int,
    max_bytes: int,
) -> bool:
    invalid = AudioInvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=tuple(f"{index:02d}:response:invalid" for index in range(32)),
        safe_json_excerpt="x" * 8_000,
    )
    repair = AudioChapterWritingRepairRequest(
        request=request,
        invalid_response=invalid,
        allowed_evidence_ids=allowed_audio_evidence_ids(request),
        prompt_version="audio-chapter-writer-repair-v1",
    )
    return _prompts_fit(
        (prompt_for_audio_writing(request)[2], prompt_for_audio_writing_repair(repair)[2]),
        max_chars,
        max_bytes,
    )


def _global_request(
    context: AudioWritingContext,
    chapters: tuple[AudioChapter, ...],
    max_chars: int,
    max_bytes: int,
) -> AudioGlobalWritingRequest:
    request = AudioGlobalWritingRequest(
        title_hint=context.title_hint,
        duration_ms=context.duration_ms,
        chapters=tuple(
            AudioGlobalChapterInput(
                chapter_id=chapter.chapter_id,
                start_ms=chapter.start_ms,
                end_ms=chapter.end_ms,
                title=chapter.title,
                summary_zh=chapter.summary_zh,
            )
            for chapter in chapters
        ),
        prompt_version="audio-global-editor-v1",
    )
    if _global_request_fits(request, max_chars, max_bytes):
        return request
    minimum = _resized_global_request(context, chapters, scale=0)
    if not _global_request_fits(minimum, max_chars, max_bytes):
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "音频全局编辑最小输入超过预算")
    lower, upper = 0, 1_000_000
    best = minimum
    while lower <= upper:
        scale = (lower + upper) // 2
        candidate = _resized_global_request(context, chapters, scale=scale)
        if _global_request_fits(candidate, max_chars, max_bytes):
            best = candidate
            lower = scale + 1
        else:
            upper = scale - 1
    return best


def _resized_global_request(
    context: AudioWritingContext,
    chapters: tuple[AudioChapter, ...],
    *,
    scale: int,
) -> AudioGlobalWritingRequest:
    return AudioGlobalWritingRequest(
        title_hint=context.title_hint,
        duration_ms=context.duration_ms,
        chapters=tuple(
            AudioGlobalChapterInput(
                chapter_id=chapter.chapter_id,
                start_ms=chapter.start_ms,
                end_ms=chapter.end_ms,
                title=chapter.title,
                summary_zh=chapter.summary_zh[
                    : max(1, len(chapter.summary_zh) * scale // 1_000_000)
                ],
            )
            for chapter in chapters
        ),
        prompt_version="audio-global-editor-v1",
    )


def _global_request_fits(
    request: AudioGlobalWritingRequest,
    max_chars: int,
    max_bytes: int,
) -> bool:
    invalid = AudioInvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=tuple(f"{index:02d}:response:invalid" for index in range(32)),
        safe_json_excerpt="x" * 8_000,
    )
    repair = AudioGlobalWritingRepairRequest(
        request=request,
        invalid_response=invalid,
        allowed_chapter_ids=tuple(item.chapter_id for item in request.chapters),
        prompt_version="audio-global-editor-repair-v1",
    )
    return _prompts_fit(
        (prompt_for_audio_global(request)[2], prompt_for_audio_global_repair(repair)[2]),
        max_chars,
        max_bytes,
    )


def _prompts_fit(prompts: Iterable[str], max_chars: int, max_bytes: int) -> bool:
    return all(
        len(prompt) <= max_chars and len(prompt.encode("utf-8")) <= max_bytes for prompt in prompts
    )


def _validate_response(
    response: AudioChapterWritingResponse, request: AudioChapterWritingRequest | None = None
) -> None:
    if request is None:
        return
    allowed = {item.evidence_id for item in request.transcript_evidence}
    refs = (
        *response.title_evidence_refs,
        *response.summary_evidence_refs,
        *(ref for block in response.body_blocks for ref in block.evidence_refs),
        *(ref for claim in response.claims for ref in claim.evidence_refs),
    )
    unknown_refs = tuple(sorted(set(refs) - allowed))
    if unknown_refs:
        safe_refs = ",".join(_safe_reference_label(ref) for ref in unknown_refs[:8])
        raise ValueError(f"evidence_refs:unknown:{safe_refs or 'omitted'}")
    if (
        response.body_blocks
        and _response_body_characters(response)
        > _DETAIL_BODY_LIMITS[request.document_config.detail_level]
    ):
        raise ValueError("音频章节正文超过配置字符预算")
    if len(response.summary_zh) > _DETAIL_SUMMARY_LIMITS[request.document_config.detail_level]:
        raise ValueError("音频章节摘要超过配置字符预算")


def _response_body_characters(response: AudioChapterWritingResponse) -> int:
    body = 0
    for block in response.body_blocks:
        if isinstance(block, (AudioParagraphBlock, AudioQuoteBlock)):
            body += len(block.text)
        elif isinstance(block, AudioBulletListBlock):
            body += sum(len(item) for item in block.items)
        else:
            body += len(str(block))
    return body + sum(len(claim.text) for claim in response.claims)


def _validate_global_response(response: AudioGlobalWritingResponse) -> None:
    if not response.overview_zh.strip():
        raise ValueError("音频核心概览不能为空")


def _is_response_invalid(error: VideoDemoError) -> bool:
    return error.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID


def _materialize_chapter(
    plan: AudioChapterPlan, response: AudioChapterWritingResponse, source: str
) -> AudioChapter:
    response = _normalize_audio_response(response)
    response = _ensure_audio_chapter_claims(response)
    refs = tuple(
        dict.fromkeys(
            (
                *response.title_evidence_refs,
                *response.summary_evidence_refs,
                *(ref for item in response.body_blocks for ref in item.evidence_refs),
                *(ref for item in response.claims for ref in item.evidence_refs),
            ),
        ),
    )
    return AudioChapter(
        chapter_id=plan.chapter_id,
        start_ms=plan.start_ms,
        end_ms=plan.end_ms,
        title=response.title,
        title_evidence_refs=response.title_evidence_refs,
        summary_zh=response.summary_zh,
        summary_evidence_refs=response.summary_evidence_refs,
        body_blocks=response.body_blocks,
        claims=response.claims,
        evidence_refs=refs,
        transcript_source=source,
        content_status="GROUNDED" if refs else "NO_SEMANTIC_EVIDENCE",
    )


def _fallback_chapter(
    plan: AudioChapterPlan, evidence: tuple[AudioTranscriptEvidence, ...], source: str
) -> AudioChapter:
    refs = tuple(item.evidence_id for item in evidence)
    body_limit = _DETAIL_BODY_LIMITS["standard"]
    blocks: list[AudioParagraphBlock] = []
    for start in range(0, len(evidence), 32):
        remaining = body_limit - sum(len(item.text) for item in blocks)
        if remaining < 1:
            break
        batch = evidence[start : start + 32]
        text = " ".join(item.text for item in batch)[:remaining]
        if text:
            blocks.append(
                AudioParagraphBlock(
                    text=clean_audio_text(text),
                    evidence_refs=tuple(item.evidence_id for item in batch),
                ),
            )
    summary = (_first_audio_summary_sentence(blocks[0].text) if blocks else plan.title_hint)[
        : _DETAIL_SUMMARY_LIMITS["standard"]
    ]
    response = AudioChapterWritingResponse(
        title=plan.title_hint,
        title_evidence_refs=refs[:1],
        summary_zh=summary,
        summary_evidence_refs=refs[:1],
        body_blocks=tuple(blocks),
        claims=(),
    )
    return _materialize_chapter(plan, response, source)


def _normalize_audio_response(
    response: AudioChapterWritingResponse,
) -> AudioChapterWritingResponse:
    blocks = tuple(_normalize_audio_block(block) for block in response.body_blocks)
    claims = tuple(
        claim.model_copy(update={"text": clean_audio_text(claim.text)}) for claim in response.claims
    )
    return response.model_copy(
        update={
            "title": clean_audio_text(response.title),
            "summary_zh": clean_audio_text(response.summary_zh),
            "body_blocks": blocks,
            "claims": claims,
        },
    )


def _normalize_audio_block(block: object) -> object:
    if isinstance(block, (AudioParagraphBlock, AudioQuoteBlock)):
        return block.model_copy(update={"text": clean_audio_text(block.text)})
    if isinstance(block, AudioBulletListBlock):
        return block.model_copy(
            update={"items": tuple(clean_audio_text(item) for item in block.items)},
        )
    return block


def _ensure_audio_chapter_claims(
    response: AudioChapterWritingResponse,
) -> AudioChapterWritingResponse:
    """模型漏返结论时，沿用章节摘要补出最小可验证结论。"""

    if response.claims or not response.summary_zh.strip():
        return response
    evidence_refs = response.summary_evidence_refs or response.title_evidence_refs
    if not evidence_refs:
        return response
    return response.model_copy(
        update={
            "claims": (
                AudioGroundedClaim(
                    text=_first_audio_summary_sentence(response.summary_zh),
                    evidence_refs=evidence_refs,
                    certainty=0.7,
                ),
            ),
        },
    )


def _first_audio_summary_sentence(summary: str) -> str:
    for index, character in enumerate(summary):
        if character in "。!?\uff01\uff1f":
            return summary[: index + 1]
    return summary[:2_000]


def _fallback_overview(chapters: tuple[AudioChapter, ...]) -> str:
    return (
        "；".join(item.summary_zh for item in chapters if item.summary_zh)[:8_000]
        or "未提取到可验证语义内容。"
    )


def _fallback_outcome(
    request: AudioChapterWritingRequest,
    source: str,
    counters: _Counters | None = None,
) -> _ChapterOutcome:
    provider_attempts = counters.chapter_attempts if counters is not None else 0
    structure_repairs = counters.chapter_repairs if counters is not None else 0
    cache_hits = counters.chapter_cache_hits if counters is not None else 0
    return _ChapterOutcome(
        _fallback_chapter(request.chapter, request.transcript_evidence, source),
        True,
        provider_attempts,
        structure_repairs,
        cache_hits,
    )


def _invalid_audio_local_response(
    response: BaseModel,
    error: BaseException,
) -> AudioInvalidModelResponse:
    payload = response.model_dump(mode="json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return audio_invalid_model_response(
        raw, (str(error)[:500] or "audio_response:invalid",), parsed_json=payload
    )


def _invalid_audio_local_response_from_error(error: BaseException) -> AudioInvalidModelResponse:
    return audio_invalid_model_response(
        b"audio-local-response-error",
        (str(error)[:500] or "audio_response:invalid",),
    )


def _invalid_audio_provider_error(error: VideoDemoError) -> AudioInvalidModelResponse:
    """把 Provider 直接返回的结构错误转换为可修复的安全上下文。"""

    return audio_invalid_model_response(
        error.code.value.encode("utf-8"),
        ("provider_response:invalid", f"provider_code:{error.code.value}"),
    )


def _safe_reference_label(value: object) -> str:
    text = "".join(char for char in str(value) if char.isalnum() or char in "_-.")
    return text[:80]


def _log_audio_chapter_start(request: AudioChapterWritingRequest) -> None:
    prompt = prompt_for_audio_writing(request)[2]
    _LOGGER.info(
        "音频章节写作开始 chapter_id=%s start_ms=%d end_ms=%d evidence_count=%d "
        "input_chars=%d input_bytes=%d",
        request.chapter.chapter_id,
        request.chapter.start_ms,
        request.chapter.end_ms,
        len(request.transcript_evidence),
        len(prompt),
        len(prompt.encode("utf-8")),
    )


def _log_audio_chapter_finished(
    request: AudioChapterWritingRequest,
    outcome: _ChapterOutcome,
    started_at: float,
) -> None:
    _LOGGER.info(
        "音频章节写作完成 chapter_id=%s status=%s elapsed_ms=%d "
        "provider_attempts=%d repairs=%d cache_hits=%d body_blocks=%d claims=%d",
        request.chapter.chapter_id,
        "FALLBACK" if outcome.degraded else "SUCCEEDED",
        max(0, round((time.monotonic() - started_at) * 1_000)),
        outcome.provider_attempts,
        outcome.structure_repairs,
        outcome.cache_hits,
        len(outcome.chapter.body_blocks),
        len(outcome.chapter.claims),
    )


def _log_audio_validation_failure(
    request: AudioChapterWritingRequest,
    phase: str,
    error: AudioModelResponseValidationError | AudioInvalidModelResponse,
    counters: _Counters,
    started_at: float,
) -> None:
    invalid = (
        error.invalid_response if isinstance(error, AudioModelResponseValidationError) else error
    )
    _LOGGER.warning(
        "音频章节写作响应校验失败 chapter_id=%s phase=%s elapsed_ms=%d "
        "provider_attempts=%d validation_errors=%s",
        request.chapter.chapter_id,
        phase,
        max(0, round((time.monotonic() - started_at) * 1_000)),
        counters.chapter_attempts,
        ",".join(invalid.validation_errors[:8]),
    )


def _log_audio_degradation(
    request: AudioChapterWritingRequest,
    phase: str,
    reason: str,
    error: VideoDemoError,
    counters: _Counters,
    started_at: float,
) -> None:
    _LOGGER.warning(
        "音频章节写作降级 chapter_id=%s phase=%s reason=%s code=%s "
        "elapsed_ms=%d provider_attempts=%d",
        request.chapter.chapter_id,
        phase,
        reason,
        error.code.value,
        max(0, round((time.monotonic() - started_at) * 1_000)),
        counters.chapter_attempts,
    )


def _log_audio_global_start(request: AudioGlobalWritingRequest) -> None:
    prompt = prompt_for_audio_global(request)[2]
    _LOGGER.info(
        "音频全局文章写作开始 chapter_count=%d input_chars=%d input_bytes=%d",
        len(request.chapters),
        len(prompt),
        len(prompt.encode("utf-8")),
    )


def _log_audio_global_validation(
    phase: str,
    error: AudioModelResponseValidationError | AudioInvalidModelResponse,
    counters: _Counters,
    started_at: float,
) -> None:
    invalid = (
        error.invalid_response if isinstance(error, AudioModelResponseValidationError) else error
    )
    _LOGGER.warning(
        "音频全局文章响应校验失败 phase=%s elapsed_ms=%d provider_attempts=%d validation_errors=%s",
        phase,
        max(0, round((time.monotonic() - started_at) * 1_000)),
        counters.global_attempts,
        ",".join(invalid.validation_errors[:8]),
    )


def _log_audio_global_degradation(
    phase: str,
    reason: str,
    error: VideoDemoError,
    counters: _Counters,
    started_at: float,
) -> None:
    _LOGGER.warning(
        "音频全局文章写作降级 phase=%s reason=%s code=%s elapsed_ms=%d provider_attempts=%d",
        phase,
        reason,
        error.code.value,
        max(0, round((time.monotonic() - started_at) * 1_000)),
        counters.global_attempts,
    )


def _is_fallback_error(error: VideoDemoError) -> bool:
    return error.code in {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
    }


def _raise_if_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
    if is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


def _cancellable_attempt_callback(
    count_attempt: Callable[[], None],
    is_cancel_requested: Callable[[], bool],
) -> Callable[[], None]:
    def before_provider_attempt() -> None:
        _raise_if_cancelled(is_cancel_requested)
        count_attempt()

    return before_provider_attempt


def _cancel_before_or_wait_for_commits(gates: Iterable[_CacheCommitGate]) -> None:
    events = tuple(event for gate in gates if (event := gate.cancel()) is not None)
    for event in events:
        event.wait()


def _empty_audio_chapter(plan: AudioChapterPlan) -> AudioChapter:
    return AudioChapter(
        chapter_id=plan.chapter_id,
        start_ms=plan.start_ms,
        end_ms=plan.end_ms,
        title=_PLACEHOLDER,
        title_evidence_refs=(),
        summary_zh=_PLACEHOLDER,
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        evidence_refs=(),
        transcript_source="NONE",
        content_status="NO_SEMANTIC_EVIDENCE",
    )
