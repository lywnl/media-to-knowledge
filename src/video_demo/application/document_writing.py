from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import pairwise
from threading import Event, Lock
from typing import Literal, cast

from pydantic import StrictInt, field_validator

from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.base import FrozenModel
from video_demo.domain.document import (
    BulletListBlock,
    ChapterBodyBlock,
    CodeBlock,
    DocumentGenerationMetadata,
    FormulaBlock,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    QuoteBlock,
    SemanticChapter,
    TableBlock,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    VisualBlock,
    validate_evidence_references,
)
from video_demo.domain.document_artifact import MAX_METRIC_VALUE
from video_demo.domain.document_plan import ChapterPlan
from video_demo.domain.evidence import (
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterWritingRepairRequest,
    ChapterWritingRequest,
    ChapterWritingResponse,
    DocumentTextPort,
    GlobalChapterInput,
    GlobalWritingRepairRequest,
    GlobalWritingRequest,
    GlobalWritingResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
    allowed_global_chapter_ids,
    allowed_writing_evidence_ids,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import (
    prompt_for_global_editing,
    prompt_for_global_repair,
    prompt_for_writing,
    prompt_for_writing_repair,
)
from video_demo.integrations.document_writing_normalization import (
    normalize_optional_visual_blocks,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_DETAIL_BODY_LIMITS = {"concise": 800, "standard": 2_000, "detailed": 4_000}
_DETAIL_SUMMARY_LIMITS = {"concise": 160, "standard": 300, "detailed": 500}
_WRITING_METRICS = frozenset(
    {
        "chapter_writer_logical_calls",
        "chapter_writer_provider_attempts",
        "chapter_writer_structure_repairs",
        "chapter_writer_cache_hits",
        "chapter_writer_fallback_chapters",
        "global_editor_logical_calls",
        "global_editor_provider_attempts",
        "global_editor_structure_repairs",
        "global_editor_cache_hits",
        "global_editor_fallbacks",
    },
)
_PLACEHOLDER = "本时段未提取到可验证语义内容"
_MAX_INPUT_CHARS = 60_000
_MAX_INPUT_BYTES = 1_048_576
_LOGGER = logging.getLogger(__name__)


class WrittenDocument(FrozenModel):
    result: VideoUnderstandingResult
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) - _WRITING_METRICS:
            raise ValueError("文档写作指标包含未知白名单键")
        if any(
            type(item) is not int or not 0 <= item <= MAX_METRIC_VALUE
            for item in value.values()
        ):
            raise ValueError("文档写作指标必须是非负严格整数")
        return value


class _Counters:
    def __init__(self) -> None:
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
        self.chapter_attempts += 1

    def global_attempt(self) -> None:
        self.global_attempts += 1

    def metrics(self) -> dict[str, int]:
        return {
            "chapter_writer_logical_calls": self.chapter_logical,
            "chapter_writer_provider_attempts": self.chapter_attempts,
            "chapter_writer_structure_repairs": self.chapter_repairs,
            "chapter_writer_cache_hits": self.chapter_cache_hits,
            "chapter_writer_fallback_chapters": self.chapter_fallbacks,
            "global_editor_logical_calls": self.global_logical,
            "global_editor_provider_attempts": self.global_attempts,
            "global_editor_structure_repairs": self.global_repairs,
            "global_editor_cache_hits": self.global_cache_hits,
            "global_editor_fallbacks": self.global_fallbacks,
        }


class _CacheCommitGate:
    """让取消与不可撤销的缓存提交形成单一线性化边界。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._commit_finished = Event()
        self._state: Literal["PRE_COMMIT", "COMMITTING", "FINISHED", "CANCELLED"] = (
            "PRE_COMMIT"
        )

    def begin_commit(self, is_cancel_requested: Callable[[], bool]) -> None:
        with self._lock:
            if self._state == "CANCELLED" or is_cancel_requested():
                self._state = "CANCELLED"
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
            if self._state != "PRE_COMMIT":
                raise RuntimeError("缓存提交门状态非法")
            self._state = "COMMITTING"

    def finish_commit(self) -> None:
        with self._lock:
            if self._state != "COMMITTING":
                raise RuntimeError("缓存提交门状态非法")
            self._state = "FINISHED"
            self._commit_finished.set()

    def cancel(self) -> Event | None:
        """取消先到则阻止提交；提交先到则返回必须等待的完成事件。"""

        with self._lock:
            if self._state == "PRE_COMMIT":
                self._state = "CANCELLED"
                return None
            if self._state == "COMMITTING":
                return self._commit_finished
            return None


@dataclass(frozen=True, slots=True)
class _ChapterOutcome:
    chapter: SemanticChapter
    degraded: bool
    provider_attempts: int
    structure_repairs: int
    cache_hits: int


class DocumentWriter:
    """把有界模型草稿收敛为程序拥有引用、顺序与 Markdown 产物的 4.1 结果。"""

    def __init__(
        self,
        text_port: DocumentTextPort,
        *,
        text_model_id: str,
        vlm_model_id: str,
        prompt_versions: PromptVersions,
        chapter_identity: ModelInvocationIdentity,
        global_identity: ModelInvocationIdentity,
        chapter_writer_concurrency: int,
        max_input_chars: int,
        max_input_bytes: int,
        invocation_wait_timeout_seconds: float,
    ) -> None:
        if not text_model_id.strip() or not vlm_model_id.strip():
            raise ValueError("写作模型身份不能为空")
        if not 1 <= chapter_writer_concurrency <= 2:
            raise ValueError("章节写作并发必须位于 1~2")
        if max_input_chars < 1 or max_input_bytes < 1:
            raise ValueError("文本输入预算必须大于 0")
        if max_input_chars > _MAX_INPUT_CHARS:
            raise ValueError("文本输入字符预算不得超过 60000")
        if max_input_bytes > _MAX_INPUT_BYTES:
            raise ValueError("文本输入字节预算不得超过 1048576")
        if invocation_wait_timeout_seconds <= 0:
            raise ValueError("模型调用锁等待时间必须大于 0")
        if chapter_identity.main_prompt_version != prompt_versions.chapter_writer:
            raise ValueError("章节写作缓存身份与 Prompt 版本不一致")
        if chapter_identity.logical_operation != "chapter_writing":
            raise ValueError("章节写作缓存身份与逻辑操作不一致")
        if chapter_identity.repair_prompt_version != prompt_versions.chapter_writer_repair:
            raise ValueError("章节写作缓存身份与修复 Prompt 版本不一致")
        if chapter_identity.model_id != text_model_id:
            raise ValueError("章节写作缓存身份与文本模型不一致")
        if chapter_identity.main_response_schema_name != "chapter_writing_v2":
            raise ValueError("章节写作缓存身份与响应 Schema 不一致")
        if chapter_identity.repair_response_schema_name != "chapter_writing_repair_v2":
            raise ValueError("章节写作缓存身份与修复 Schema 不一致")
        if global_identity.main_prompt_version != prompt_versions.global_editor:
            raise ValueError("全局编辑缓存身份与 Prompt 版本不一致")
        if global_identity.logical_operation != "global_editing":
            raise ValueError("全局编辑缓存身份与逻辑操作不一致")
        if global_identity.repair_prompt_version != prompt_versions.global_editor_repair:
            raise ValueError("全局编辑缓存身份与修复 Prompt 版本不一致")
        if global_identity.model_id != text_model_id:
            raise ValueError("全局编辑缓存身份与文本模型不一致")
        if global_identity.main_response_schema_name != "global_writing_v1":
            raise ValueError("全局编辑缓存身份与主响应 Schema 不一致")
        if global_identity.repair_response_schema_name != "global_writing_repair_v1":
            raise ValueError("全局编辑缓存身份与修复响应 Schema 不一致")
        self._text_port = text_port
        self._text_model_id = text_model_id
        self._vlm_model_id = vlm_model_id
        self._prompt_versions = prompt_versions
        self._chapter_identity = chapter_identity
        self._global_identity = global_identity
        self._concurrency = chapter_writer_concurrency
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._wait_timeout_seconds = invocation_wait_timeout_seconds

    def write(
        self,
        context: DocumentWritingContext,
        plans: tuple[ChapterPlan, ...],
        transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
        visual_evidence: tuple[VisualObservationEvidence, ...],
        keyframe_evidence: tuple[KeyframeEvidence, ...],
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> WrittenDocument:
        _validate_inputs(context, plans, transcript_evidence, visual_evidence, keyframe_evidence)
        _raise_if_cancelled(is_cancel_requested)
        counters = _Counters()
        packages = tuple(
            (
                plan,
                _chapter_request(
                    context,
                    plan,
                    transcript_evidence,
                    visual_evidence,
                    self._prompt_versions.chapter_writer,
                    self._max_input_chars,
                    self._max_input_bytes,
                ),
            )
            for plan in plans
        )
        outcomes = self._write_chapters(
            packages,
            keyframe_evidence,
            cache,
            counters,
            is_cancel_requested,
        )
        _raise_if_cancelled(is_cancel_requested)
        chapters = tuple(item.chapter for item in outcomes)
        chapter_degraded = tuple(item.chapter.chapter_id for item in outcomes if item.degraded)
        global_response, global_degraded = self._write_global(
            context,
            chapters,
            cache,
            counters,
            is_cancel_requested,
        )
        _raise_if_cancelled(is_cancel_requested)
        result = _materialize_result(
            context,
            chapters,
            global_response,
            text_model_id=self._text_model_id,
            vlm_model_id=self._vlm_model_id,
            prompt_versions=self._prompt_versions,
        )
        validate_evidence_references(
            result,
            (*transcript_evidence, *keyframe_evidence, *visual_evidence),
        )
        warnings = tuple(
            f"CHAPTER_WRITING_DEGRADED:{chapter_id}" for chapter_id in chapter_degraded
        )
        if global_degraded:
            warnings += ("GLOBAL_WRITING_DEGRADED",)
        _raise_if_cancelled(is_cancel_requested)
        return WrittenDocument(
            result=result,
            warnings=warnings,
            status="PARTIAL_SUCCEEDED" if warnings else "SUCCEEDED",
            metrics=counters.metrics(),
        )

    def _write_chapters(
        self,
        packages: tuple[tuple[ChapterPlan, ChapterWritingRequest | None], ...],
        keyframes: tuple[KeyframeEvidence, ...],
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[_ChapterOutcome, ...]:
        outcomes: list[_ChapterOutcome | None] = [None] * len(packages)
        pending_indexes: list[int] = []
        for index, (plan, request) in enumerate(packages):
            if request is None:
                outcomes[index] = _ChapterOutcome(_empty_chapter(plan), False, 0, 0, 0)
                continue
            counters.chapter_logical += 1
            concrete_request = request

            def validate(
                response: ChapterWritingResponse,
                request_to_validate: ChapterWritingRequest = concrete_request,
            ) -> None:
                response = normalize_optional_visual_blocks(
                    response,
                    request_to_validate.visual_observations,
                )
                _validate_chapter_response(response, request_to_validate)

            cached = cache.get(
                self._chapter_identity,
                concrete_request,
                ChapterWritingResponse,
                validate,
            )
            if cached is not None:
                normalized = normalize_optional_visual_blocks(
                    cached.response,
                    concrete_request.visual_observations,
                )
                normalized = _normalize_response_blocks(normalized, concrete_request)
                outcomes[index] = _ChapterOutcome(
                    _materialize_chapter(concrete_request, normalized, keyframes),
                    False,
                    0,
                    0,
                    1,
                )
                counters.chapter_cache_hits += 1
            else:
                pending_indexes.append(index)
        if not pending_indexes:
            return cast(tuple[_ChapterOutcome, ...], tuple(outcomes))
        executor = ThreadPoolExecutor(max_workers=self._concurrency)
        in_flight: dict[Future[_ChapterOutcome], tuple[int, _CacheCommitGate]] = {}
        next_position = 0
        try:
            while next_position < len(pending_indexes) or in_flight:
                if is_cancel_requested():
                    _cancel_before_or_wait_for_commits(
                        gate for _, gate in in_flight.values()
                    )
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_position < len(pending_indexes) and len(in_flight) < self._concurrency:
                    index = pending_indexes[next_position]
                    request = packages[index][1]
                    assert request is not None
                    commit_gate = _CacheCommitGate()
                    future = executor.submit(
                        self._write_one,
                        request,
                        keyframes,
                        cache,
                        is_cancel_requested,
                        commit_gate,
                    )
                    in_flight[future] = (index, commit_gate)
                    next_position += 1
                completed, _ = wait(
                    tuple(in_flight),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    continue
                for future in completed:
                    index, _ = in_flight.pop(future)
                    outcome = future.result()
                    outcomes[index] = outcome
                    counters.chapter_attempts += outcome.provider_attempts
                    counters.chapter_repairs += outcome.structure_repairs
                    counters.chapter_cache_hits += outcome.cache_hits
                    counters.chapter_fallbacks += int(outcome.degraded)
        except BaseException as error:
            _cancel_before_or_wait_for_commits(
                gate for _, gate in in_flight.values()
            )
            for future in in_flight:
                future.cancel()
            if isinstance(error, VideoDemoError) and error.code == ErrorCode.JOB_CANCELLED:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            _done, unfinished = wait(
                tuple(in_flight),
                timeout=self._wait_timeout_seconds,
            )
            if unfinished:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return cast(tuple[_ChapterOutcome, ...], tuple(outcomes))

    def _write_one(
        self,
        request: ChapterWritingRequest,
        keyframes: tuple[KeyframeEvidence, ...],
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
        commit_gate: _CacheCommitGate,
    ) -> _ChapterOutcome:
        local = _Counters()
        response = self._chapter_logical_call(
            request,
            cache,
            local,
            is_cancel_requested,
            commit_gate,
        )
        degraded = response is None
        if response is None:
            if not _fallback_header_evidence_refs(request):
                return _ChapterOutcome(
                    _empty_chapter(request.chapter),
                    True,
                    local.chapter_attempts,
                    local.chapter_repairs,
                    local.chapter_cache_hits,
                )
            response = _fallback_chapter_response(request)
        response = normalize_optional_visual_blocks(
            response,
            request.visual_observations,
        )
        response = _normalize_response_blocks(response, request)
        chapter = _materialize_chapter(request, response, keyframes)
        return _ChapterOutcome(
            chapter,
            degraded,
            local.chapter_attempts,
            local.chapter_repairs,
            local.chapter_cache_hits,
        )

    def _chapter_logical_call(
        self,
        request: ChapterWritingRequest,
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
        commit_gate: _CacheCommitGate,
    ) -> ChapterWritingResponse | None:
        def validate(response: ChapterWritingResponse) -> None:
            response = normalize_optional_visual_blocks(
                response,
                request.visual_observations,
            )
            _validate_chapter_response(response, request)

        cached = cache.get(self._chapter_identity, request, ChapterWritingResponse, validate)
        if cached is not None:
            counters.chapter_cache_hits += 1
            return normalize_optional_visual_blocks(
                cached.response,
                request.visual_observations,
            )
        with cache.invocation_lock(
            self._chapter_identity,
            request,
            wait_timeout_seconds=self._wait_timeout_seconds,
            is_cancel_requested=is_cancel_requested,
        ):
            cached = cache.get(self._chapter_identity, request, ChapterWritingResponse, validate)
            if cached is not None:
                counters.chapter_cache_hits += 1
                return normalize_optional_visual_blocks(
                    cached.response,
                    request.visual_observations,
                )
            invalid: InvalidModelResponse | None = None
            try:
                response = self._text_port.write_chapter(
                    request,
                    on_provider_attempt=_cancellable_attempt_callback(
                        counters.chapter_attempt,
                        is_cancel_requested,
                    ),
                )
                response = normalize_optional_visual_blocks(
                    response,
                    request.visual_observations,
                )
            except ModelResponseValidationError as error:
                _raise_if_cancelled(is_cancel_requested)
                invalid = error.invalid_response
                _log_chapter_validation_failure(
                    chapter_id=request.chapter.chapter_id,
                    phase="main",
                    code=error.code,
                    invalid=invalid,
                    provider_attempts=counters.chapter_attempts,
                )
            except VideoDemoError as error:
                _raise_if_cancelled(is_cancel_requested)
                if _is_fallback_error(error):
                    return None
                raise
            else:
                _raise_if_cancelled(is_cancel_requested)
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    invalid = _invalid_local_response(response, error)
                    _log_chapter_validation_failure(
                        chapter_id=request.chapter.chapter_id,
                        phase="main",
                        code=ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                        invalid=invalid,
                        provider_attempts=counters.chapter_attempts,
                    )
            if invalid is None:
                path: Literal["MAIN", "REPAIR"] = "MAIN"
            else:
                _raise_if_cancelled(is_cancel_requested)
                counters.chapter_repairs += 1
                try:
                    response = self._text_port.repair_chapter_writing(
                        ChapterWritingRepairRequest(
                            request=request,
                            invalid_response=invalid,
                            allowed_evidence_ids=allowed_writing_evidence_ids(request),
                            prompt_version=self._prompt_versions.chapter_writer_repair,
                        ),
                        on_provider_attempt=_cancellable_attempt_callback(
                            counters.chapter_attempt,
                            is_cancel_requested,
                        ),
                    )
                    response = normalize_optional_visual_blocks(
                        response,
                        request.visual_observations,
                    )
                except ModelResponseValidationError as error:
                    _raise_if_cancelled(is_cancel_requested)
                    _log_chapter_validation_failure(
                        chapter_id=request.chapter.chapter_id,
                        phase="repair",
                        code=error.code,
                        invalid=error.invalid_response,
                        provider_attempts=counters.chapter_attempts,
                    )
                    return None
                except VideoDemoError as error:
                    _raise_if_cancelled(is_cancel_requested)
                    if _is_fallback_error(error):
                        return None
                    raise
                _raise_if_cancelled(is_cancel_requested)
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    _log_chapter_validation_failure(
                        chapter_id=request.chapter.chapter_id,
                        phase="repair",
                        code=ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                        invalid=_invalid_local_response(response, error),
                        provider_attempts=counters.chapter_attempts,
                    )
                    return None
                path = "REPAIR"
            commit_gate.begin_commit(is_cancel_requested)
            try:
                return cache.put(
                    self._chapter_identity,
                    request,
                    response,
                    successful_path=path,
                    validate=validate,
                ).response
            finally:
                commit_gate.finish_commit()

    def _write_global(
        self,
        context: DocumentWritingContext,
        chapters: tuple[SemanticChapter, ...],
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[GlobalWritingResponse, bool]:
        if not any(chapter.content_status == "GROUNDED" for chapter in chapters):
            return _fallback_global_response(chapters), False
        request = _global_request(
            context,
            chapters,
            self._prompt_versions.global_editor,
            self._max_input_chars,
            self._max_input_bytes,
        )
        counters.global_logical += 1

        commit_gate = _CacheCommitGate()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._global_logical_call,
            request,
            chapters,
            cache,
            counters,
            is_cancel_requested,
            commit_gate,
        )
        try:
            while True:
                if is_cancel_requested():
                    _cancel_before_or_wait_for_commits((commit_gate,))
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                completed, _ = wait(
                    (future,),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if completed:
                    result = future.result()
                    executor.shutdown(wait=True)
                    return result
        except BaseException:
            future.cancel()
            executor.shutdown(wait=future.done(), cancel_futures=True)
            raise

    def _global_logical_call(
        self,
        request: GlobalWritingRequest,
        chapters: tuple[SemanticChapter, ...],
        cache: DocumentModelCache,
        counters: _Counters,
        is_cancel_requested: Callable[[], bool],
        commit_gate: _CacheCommitGate,
    ) -> tuple[GlobalWritingResponse, bool]:
        def validate(response: GlobalWritingResponse) -> None:
            _validate_global_response(response, request, chapters)

        _raise_if_cancelled(is_cancel_requested)
        cached = cache.get(self._global_identity, request, GlobalWritingResponse, validate)
        if cached is not None:
            _raise_if_cancelled(is_cancel_requested)
            counters.global_cache_hits += 1
            return cached.response, False
        with cache.invocation_lock(
            self._global_identity,
            request,
            wait_timeout_seconds=self._wait_timeout_seconds,
            is_cancel_requested=is_cancel_requested,
        ):
            cached = cache.get(self._global_identity, request, GlobalWritingResponse, validate)
            if cached is not None:
                _raise_if_cancelled(is_cancel_requested)
                counters.global_cache_hits += 1
                return cached.response, False
            _raise_if_cancelled(is_cancel_requested)
            invalid: InvalidModelResponse | None = None
            try:
                response = self._text_port.organize_document(
                    request,
                    on_provider_attempt=_cancellable_attempt_callback(
                        counters.global_attempt,
                        is_cancel_requested,
                    ),
                )
            except ModelResponseValidationError as error:
                _raise_if_cancelled(is_cancel_requested)
                invalid = error.invalid_response
            except VideoDemoError as error:
                _raise_if_cancelled(is_cancel_requested)
                if _is_fallback_error(error):
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                raise
            else:
                _raise_if_cancelled(is_cancel_requested)
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    invalid = _invalid_local_response(response, error)
            if invalid is None:
                path: Literal["MAIN", "REPAIR"] = "MAIN"
            else:
                counters.global_repairs += 1
                _raise_if_cancelled(is_cancel_requested)
                try:
                    response = self._text_port.repair_global_writing(
                        GlobalWritingRepairRequest(
                            request=request,
                            invalid_response=invalid,
                            allowed_chapter_ids=allowed_global_chapter_ids(request),
                            prompt_version=self._prompt_versions.global_editor_repair,
                        ),
                        on_provider_attempt=_cancellable_attempt_callback(
                            counters.global_attempt,
                            is_cancel_requested,
                        ),
                    )
                except ModelResponseValidationError:
                    _raise_if_cancelled(is_cancel_requested)
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                except VideoDemoError as error:
                    _raise_if_cancelled(is_cancel_requested)
                    if _is_fallback_error(error):
                        counters.global_fallbacks += 1
                        return _fallback_global_response(chapters), True
                    raise
                _raise_if_cancelled(is_cancel_requested)
                try:
                    validate(response)
                except (ValueError, TypeError):
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                path = "REPAIR"
            commit_gate.begin_commit(is_cancel_requested)
            try:
                response = cache.put(
                    self._global_identity,
                    request,
                    response,
                    successful_path=path,
                    validate=validate,
                ).response
            finally:
                commit_gate.finish_commit()
            return response, False


def _validate_inputs(
    context: DocumentWritingContext,
    plans: tuple[ChapterPlan, ...],
    transcript: tuple[SpeechSegment | SubtitleCue, ...],
    observations: tuple[VisualObservationEvidence, ...],
    keyframes: tuple[KeyframeEvidence, ...],
) -> None:
    if not plans or plans[0].start_ms != 0 or plans[-1].end_ms != context.duration_ms:
        raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划未覆盖完整视频")
    if any(left.end_ms != right.start_ms for left, right in pairwise(plans)):
        raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划必须连续有序")
    all_ids = tuple(
        item.evidence_id for item in (*transcript, *keyframes, *observations)
    )
    if len(all_ids) != len(set(all_ids)):
        raise VideoDemoError(ErrorCode.DUPLICATE_EVIDENCE_ID, "写作输入证据 ID 不得重复")
    plan_by_id = {plan.chapter_id: plan for plan in plans}
    if len(plan_by_id) != len(plans):
        raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划 ID 不得重复")
    for transcript_item in transcript:
        owners = tuple(plan for plan in plans if plan.contains(transcript_item))
        if transcript_item.end_ms > context.duration_ms or len(owners) != 1:
            raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_CHAPTER, "转写证据超出视频范围")
    for keyframe in keyframes:
        if sum(plan.contains(keyframe) for plan in plans) != 1:
            raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_CHAPTER, "关键帧不属于唯一章节")
    for observation in observations:
        plan = plan_by_id.get(observation.chapter_id)
        if plan is None or not plan.contains(observation):
            raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_CHAPTER, "视觉观察不属于声明章节")
    keyframe_by_id = {item.evidence_id: item for item in keyframes}
    transcript_by_id = {item.evidence_id: item for item in transcript}
    referenced_keyframes: set[str] = set()
    for observation in observations:
        plan = plan_by_id[observation.chapter_id]
        if not set(observation.keyframe_refs).issubset(keyframe_by_id):
            raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "视觉观察缺少已晋升关键帧")
        if not set(observation.transcript_evidence_refs).issubset(transcript_by_id):
            raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "视觉观察缺少转写证据")
        referenced_keyframes.update(observation.keyframe_refs)
        if any(
            not plan.contains(keyframe_by_id[ref])
            or not observation.contains(keyframe_by_id[ref])
            for ref in observation.keyframe_refs
        ):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                "视觉观察的关键帧不属于观察及其声明章节",
            )
        if any(
            not plan.contains(transcript_by_id[ref])
            for ref in observation.transcript_evidence_refs
        ):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                "视觉观察的转写证据不属于声明章节",
            )
    orphan_keyframes = set(keyframe_by_id) - referenced_keyframes
    if orphan_keyframes:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
            "写作输入不得包含未被视觉观察引用的孤立关键帧",
        )


def _chapter_request(
    context: DocumentWritingContext,
    plan: ChapterPlan,
    transcript: tuple[SpeechSegment | SubtitleCue, ...],
    observations: tuple[VisualObservationEvidence, ...],
    prompt_version: Literal["chapter-writer-v1"],
    max_chars: int,
    max_bytes: int,
) -> ChapterWritingRequest | None:
    chapter_transcript = tuple(
        sorted(
            (item for item in transcript if plan.contains(item)),
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    chapter_observations = tuple(
        sorted(
            (item for item in observations if item.chapter_id == plan.chapter_id),
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    if not chapter_transcript and not chapter_observations:
        return None
    while True:
        request = ChapterWritingRequest(
            context=context,
            chapter=plan,
            transcript_evidence=chapter_transcript,
            visual_observations=chapter_observations,
            prompt_version=prompt_version,
        )
        if _chapter_request_fits(request, max_chars, max_bytes):
            return request
        # 视觉观察是前序 VLM 已经付费验证的结果，必须优先保留；输入超限时
        # 先裁剪冗余 ASR，再在确实存在多个观察时裁剪视觉观察。否则写作请求
        # 会丢掉唯一的视觉证据，最终 Markdown 无法生成关键帧引用。
        if (chapter_observations and chapter_transcript) or len(chapter_transcript) > 1:
            chapter_transcript = chapter_transcript[:-1]
        elif len(chapter_observations) > 1:
            chapter_observations = chapter_observations[:-1]
        else:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "单章最小证据包超过输入预算")


def _chapter_request_fits(request: ChapterWritingRequest, max_chars: int, max_bytes: int) -> bool:
    repairs = tuple(
        ChapterWritingRepairRequest(
            request=request,
            invalid_response=invalid,
            allowed_evidence_ids=allowed_writing_evidence_ids(request),
            prompt_version="chapter-writer-repair-v1",
        )
        for invalid in _worst_invalid_responses()
    )
    return _prompts_fit(
        (
            prompt_for_writing(request)[2],
            *(prompt_for_writing_repair(repair)[2] for repair in repairs),
        ),
        max_chars,
        max_bytes,
    )


def _global_request(
    context: DocumentWritingContext,
    chapters: tuple[SemanticChapter, ...],
    prompt_version: Literal["global-editor-v1"],
    max_chars: int,
    max_bytes: int,
) -> GlobalWritingRequest:
    request = GlobalWritingRequest(
        context=context,
        chapters=_global_chapter_inputs(chapters),
        prompt_version=prompt_version,
    )
    if _global_request_fits(request, max_chars, max_bytes):
        return request
    minimum = _resized_global_request(context, chapters, prompt_version, scale=0)
    if not _global_request_fits(minimum, max_chars, max_bytes):
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "全局编辑最小输入超过预算")
    lower = 0
    upper = 1_000_000
    best = minimum
    while lower <= upper:
        scale = (lower + upper) // 2
        candidate = _resized_global_request(context, chapters, prompt_version, scale=scale)
        if _global_request_fits(candidate, max_chars, max_bytes):
            best = candidate
            lower = scale + 1
        else:
            upper = scale - 1
    return best


def _resized_global_request(
    context: DocumentWritingContext,
    chapters: tuple[SemanticChapter, ...],
    prompt_version: Literal["global-editor-v1"],
    *,
    scale: int,
) -> GlobalWritingRequest:
    resized = tuple(
        GlobalChapterInput(
            chapter_id=chapter.chapter_id,
            start_ms=chapter.start_ms,
            end_ms=chapter.end_ms,
            title=chapter.title,
            summary_zh=chapter.summary_zh[: max(1, len(chapter.summary_zh) * scale // 1_000_000)],
            content_status=chapter.content_status,
        )
        for chapter in chapters
    )
    return GlobalWritingRequest(
        context=context,
        chapters=resized,
        prompt_version=prompt_version,
    )


def _global_chapter_inputs(
    chapters: tuple[SemanticChapter, ...],
) -> tuple[GlobalChapterInput, ...]:
    return tuple(
        GlobalChapterInput(
            chapter_id=chapter.chapter_id,
            start_ms=chapter.start_ms,
            end_ms=chapter.end_ms,
            title=chapter.title,
            summary_zh=chapter.summary_zh,
            content_status=chapter.content_status,
        )
        for chapter in chapters
    )


def _global_request_fits(request: GlobalWritingRequest, max_chars: int, max_bytes: int) -> bool:
    repairs = tuple(
        GlobalWritingRepairRequest(
            request=request,
            invalid_response=invalid,
            allowed_chapter_ids=allowed_global_chapter_ids(request),
            prompt_version="global-editor-repair-v1",
        )
        for invalid in _worst_invalid_responses()
    )
    return _prompts_fit(
        (
            prompt_for_global_editing(request)[2],
            *(prompt_for_global_repair(repair)[2] for repair in repairs),
        ),
        max_chars,
        max_bytes,
    )


def _prompts_fit(prompts: Iterable[str], max_chars: int, max_bytes: int) -> bool:
    return all(
        len(item) <= max_chars and len(item.encode("utf-8")) <= max_bytes
        for item in prompts
    )


def _worst_invalid_responses() -> tuple[InvalidModelResponse, InvalidModelResponse]:
    return (_worst_invalid_response("\\"), _worst_invalid_response("\U00010000"))


def _worst_invalid_response(character: str = "\\") -> InvalidModelResponse:
    return InvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=tuple(
            f"{index:02d}:" + character * 497 for index in range(32)
        ),
        safe_json_excerpt=character * 8_000,
    )


def _validate_chapter_response(
    response: ChapterWritingResponse,
    request: ChapterWritingRequest,
) -> None:
    allowed = set(allowed_writing_evidence_ids(request))
    observations = {item.evidence_id: item for item in request.visual_observations}
    _validate_attributed_text_refs(
        response.title_evidence_refs,
        "章节标题",
        allowed,
        observations,
        request,
    )
    _validate_attributed_text_refs(
        response.summary_evidence_refs,
        "章节摘要",
        allowed,
        observations,
        request,
    )
    for block in response.body_blocks:
        if not block.evidence_refs:
            raise ValueError("正文块至少需要一个本章证据引用")
        if not set(block.evidence_refs).issubset(allowed):
            raise ValueError("正文块引用了证据包之外的 ID")
        if isinstance(block, VisualBlock):
            if block.visual_observation_ref not in observations:
                raise ValueError("VISUAL block 必须引用本章视觉观察")
            if block.evidence_refs != (block.visual_observation_ref,):
                raise ValueError("VISUAL block 只能绑定其视觉观察")
            observation = observations[block.visual_observation_ref]
            allowed_content_ids = {
                *(item.visual_content_id for item in observation.content_blocks),
                *(item.visual_fact_id for item in observation.visual_facts),
            }
            if (
                bool(allowed_content_ids) != bool(block.visual_content_refs)
                or not set(block.visual_content_refs).issubset(allowed_content_ids)
            ):
                raise ValueError("VISUAL block 引用了未知或跨观察子内容")
            _validate_visual_reference_policy(
                observation,
                request,
                is_visual_block=True,
            )
        else:
            for ref in block.evidence_refs:
                referenced_observation = observations.get(ref)
                if referenced_observation is not None:
                    _validate_visual_reference_policy(
                        referenced_observation,
                        request,
                        is_visual_block=False,
                    )
        if (
            isinstance(block, QuoteBlock)
            and not request.context.document_config.include_verbatim_quotes
        ):
            raise ValueError("当前配置禁止逐字引用")
    for claim in response.claims:
        if not set(claim.evidence_refs).issubset(allowed):
            raise ValueError("Claim 引用了证据包之外的 ID")
        for ref in claim.evidence_refs:
            referenced_observation = observations.get(ref)
            if referenced_observation is not None:
                _validate_visual_reference_policy(
                    referenced_observation,
                    request,
                    is_visual_block=False,
                )
    detail = request.context.document_config.detail_level
    if len(response.summary_zh) > _DETAIL_SUMMARY_LIMITS[detail]:
        raise ValueError("章节摘要超过配置字符预算")
    if _response_body_characters(response) > _DETAIL_BODY_LIMITS[detail]:
        raise ValueError("章节正文超过配置字符预算")
    normalized = _normalize_response_blocks(response, request)
    if _response_body_characters(normalized) > _DETAIL_BODY_LIMITS[detail]:
        raise ValueError("章节归一化正文超过配置字符预算")
    source_refs = _selected_keyframes(
        normalized.body_blocks,
        request.visual_observations,
        (),
        validate_keyframe_membership=False,
    )
    maximum = min(
        request.context.document_config.max_visuals_per_chapter,
        3 if request.chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
    )
    if len(source_refs) > maximum:
        raise ValueError("正文使用的关键帧超过章节图片预算")


def _validate_attributed_text_refs(
    evidence_refs: tuple[str, ...],
    field: str,
    allowed: set[str],
    observations: dict[str, VisualObservationEvidence],
    request: ChapterWritingRequest,
) -> None:
    if not set(evidence_refs).issubset(allowed):
        raise ValueError(f"{field}引用了证据包之外的 ID")
    for ref in evidence_refs:
        observation = observations.get(ref)
        if observation is not None:
            _validate_visual_reference_policy(
                observation,
                request,
                is_visual_block=False,
            )


def _validate_visual_reference_policy(
    observation: VisualObservationEvidence,
    request: ChapterWritingRequest,
    *,
    is_visual_block: bool,
) -> None:
    relation = observation.relation_to_transcript
    if relation == "DUPLICATE":
        if is_visual_block:
            return
        raise ValueError("DUPLICATE 视觉观察不得由普通正文或 Claim 重复表达")
    if relation == "CONFLICTING" and not is_visual_block:
        raise ValueError("CONFLICTING 视觉观察只能通过类型化视觉块表达")


def _response_body_characters(response: ChapterWritingResponse) -> int:
    return sum(_block_character_count(block) for block in response.body_blocks) + sum(
        len(claim.text) for claim in response.claims
    )


def _block_character_count(block: ChapterBodyBlock) -> int:
    if isinstance(block, (ParagraphBlock, QuoteBlock)):
        return len(block.text)
    if isinstance(block, BulletListBlock):
        return sum(len(item) for item in block.items)
    if isinstance(block, CodeBlock):
        return len(block.code)
    if isinstance(block, TableBlock):
        return sum(map(len, block.columns)) + sum(len(cell) for row in block.rows for cell in row)
    if isinstance(block, FormulaBlock):
        return len(block.latex) + len(block.explanation)
    if isinstance(block, VisualBlock):
        return len(block.caption)
    raise TypeError("未知章节正文块")


def _normalize_response_blocks(
    response: ChapterWritingResponse,
    request: ChapterWritingRequest,
) -> ChapterWritingResponse:
    observations = {item.evidence_id: item for item in request.visual_observations}
    blocks: list[ChapterBodyBlock] = []
    for block in response.body_blocks:
        if isinstance(block, QuoteBlock) and not _quote_matches(
            block,
            request.transcript_evidence,
        ):
            blocks.append(ParagraphBlock(text=block.text, evidence_refs=block.evidence_refs))
            continue
        if isinstance(block, VisualBlock):
            observation = observations[block.visual_observation_ref]
            if observation.relation_to_transcript == "DUPLICATE":
                continue
        blocks.append(block)
    blocks = _append_omitted_visual_blocks(tuple(blocks), request)
    return response.model_copy(update={"body_blocks": tuple(blocks)})


def _append_omitted_visual_blocks(
    blocks: tuple[ChapterBodyBlock, ...],
    request: ChapterWritingRequest,
) -> list[ChapterBodyBlock]:
    """模型漏写视觉块时，确定性保留已验证的最少关键画面。"""

    mutable_blocks = list(blocks)
    observations = {
        item.evidence_id: item for item in request.visual_observations
    }
    rendered_observations = {
        block.visual_observation_ref
        for block in mutable_blocks
        if isinstance(block, VisualBlock)
    }
    maximum = min(
        request.context.document_config.max_visuals_per_chapter,
        3 if request.chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
    )
    used_sources = set(
        ref
        for block in mutable_blocks
        if isinstance(block, VisualBlock)
        for ref in _visual_block_sources(block, observations)
    )
    body_limit = _DETAIL_BODY_LIMITS[request.context.document_config.detail_level]
    body_size = sum(_block_character_count(block) for block in mutable_blocks)
    for observation in request.visual_observations:
        if observation.evidence_id in rendered_observations:
            continue
        if not _visual_observation_is_renderable(observation, request):
            continue
        selection = _fallback_visual_selection(
            observation,
            used_sources=used_sources,
            maximum=maximum,
        )
        if selection is None:
            continue
        content_refs, sources = selection
        visual = VisualBlock(
            visual_observation_ref=observation.evidence_id,
            visual_content_refs=content_refs,
            caption=observation.caption,
            evidence_refs=(observation.evidence_id,),
        )
        if body_size + _block_character_count(visual) > body_limit:
            continue
        mutable_blocks.append(visual)
        body_size += _block_character_count(visual)
        used_sources.update(sources)
        if len(used_sources) >= maximum:
            break
    return mutable_blocks


def _visual_block_sources(
    block: VisualBlock,
    observations: dict[str, VisualObservationEvidence],
) -> tuple[str, ...]:
    observation = observations[block.visual_observation_ref]
    content_by_id = {
        item.visual_content_id: item.source_keyframe_refs
        for item in observation.content_blocks
    }
    content_by_id.update(
        {
            item.visual_fact_id: item.source_keyframe_refs
            for item in observation.visual_facts
        },
    )
    return tuple(
        dict.fromkeys(
            ref
            for content_ref in block.visual_content_refs
            for ref in content_by_id[content_ref]
        )
    ) or observation.keyframe_refs


def _visual_observation_is_renderable(
    observation: VisualObservationEvidence,
    request: ChapterWritingRequest,
) -> bool:
    # 冲突观察必须由模型显式组织为 VISUAL block，避免程序自动追加后
    # 把冲突信息误包装成正文；支持/补充观察则可安全确定性展示。
    return observation.relation_to_transcript not in {"DUPLICATE", "CONFLICTING"}


def _quote_matches(
    block: QuoteBlock,
    transcript: tuple[SpeechSegment | SubtitleCue, ...],
) -> bool:
    def normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        return "".join(
            character
            for character in normalized
            if not _is_ignorable_quote_char(character)
        )

    indexes = tuple(
        index
        for index, item in enumerate(transcript)
        if item.evidence_id in block.evidence_refs
    )
    if not indexes or indexes != tuple(range(indexes[0], indexes[-1] + 1)):
        return False
    referenced_text = " ".join(transcript[index].text for index in indexes)
    normalized_quote = normalize(block.text)
    return bool(normalized_quote) and normalized_quote in normalize(referenced_text)


def _is_ignorable_quote_char(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


def _materialize_chapter(
    request: ChapterWritingRequest,
    response: ChapterWritingResponse,
    keyframes: tuple[KeyframeEvidence, ...],
) -> SemanticChapter:
    response = _ensure_chapter_claims(response)
    selected = _selected_keyframes(
        response.body_blocks,
        request.visual_observations,
        keyframes,
    )
    evidence_refs = tuple(
        dict.fromkeys(
            (
                *response.title_evidence_refs,
                *response.summary_evidence_refs,
                *(item.evidence_id for item in request.transcript_evidence),
                *(item.evidence_id for item in request.visual_observations),
                *selected,
            ),
        ),
    )
    provisional = SemanticChapter(
        chapter_id=request.chapter.chapter_id,
        start_ms=request.chapter.start_ms,
        end_ms=request.chapter.end_ms,
        title=response.title,
        title_evidence_refs=response.title_evidence_refs,
        summary_zh=response.summary_zh,
        summary_evidence_refs=response.summary_evidence_refs,
        body_blocks=response.body_blocks,
        claims=response.claims,
        evidence_refs=evidence_refs,
        selected_keyframe_refs=selected,
        transcript_source=(
            request.context.transcript_source if request.transcript_evidence else "NONE"
        ),
    )
    return provisional


def _ensure_chapter_claims(response: ChapterWritingResponse) -> ChapterWritingResponse:
    """模型漏返结论时，复用已校验的章节摘要补出最小可检索结论。"""

    if response.claims or not response.summary_zh.strip():
        return response
    evidence_refs = response.summary_evidence_refs or response.title_evidence_refs
    if not evidence_refs:
        return response
    claim_text = _first_summary_sentence(response.summary_zh)
    return response.model_copy(
        update={
            "claims": (
                GroundedClaim(
                    text=claim_text,
                    evidence_refs=evidence_refs,
                    certainty=0.7,
                ),
            ),
        },
    )


def _first_summary_sentence(summary: str) -> str:
    for index, character in enumerate(summary):
        if character in "。!?\uff01\uff1f":
            return summary[: index + 1]
    return summary[:2_000]


def _selected_keyframes(
    blocks: tuple[ChapterBodyBlock, ...],
    observations: tuple[VisualObservationEvidence, ...],
    keyframes: tuple[KeyframeEvidence, ...],
    *,
    validate_keyframe_membership: bool = True,
) -> tuple[str, ...]:
    observation_by_id = {item.evidence_id: item for item in observations}
    allowed_keyframes = {item.evidence_id for item in keyframes}
    selected: list[str] = []
    for block in blocks:
        if not isinstance(block, VisualBlock):
            continue
        observation = observation_by_id[block.visual_observation_ref]
        content_by_id = {
            item.visual_content_id: item.source_keyframe_refs
            for item in observation.content_blocks
        }
        content_by_id.update(
            {
                item.visual_fact_id: item.source_keyframe_refs
                for item in observation.visual_facts
            },
        )
        sources = tuple(
            ref
            for content_ref in block.visual_content_refs
            for ref in content_by_id[content_ref]
        ) or observation.keyframe_refs
        sources = tuple(dict.fromkeys(sources))
        if not _frames_have_relation_path(observation, sources):
            raise ValueError("多图视觉正文缺少所选帧之间的对应关系")
        for ref in sources:
            if validate_keyframe_membership and ref not in allowed_keyframes:
                raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "展示图未晋升")
            if ref not in selected:
                selected.append(ref)
    return tuple(selected)


def _frames_have_relation_path(
    observation: VisualObservationEvidence,
    sources: tuple[str, ...],
) -> bool:
    selected = set(sources)
    if len(selected) <= 1:
        return True
    relation_edges = {
        frozenset((relation.from_keyframe_ref, relation.to_keyframe_ref))
        for relation in observation.frame_relations
        if (
            relation.from_keyframe_ref in selected
            and relation.to_keyframe_ref in selected
        )
    }
    if len(selected) == 2:
        return frozenset(selected) in relation_edges
    adjacency: dict[str, set[str]] = {source: set() for source in selected}
    for relation in observation.frame_relations:
        if (
            relation.from_keyframe_ref in selected
            and relation.to_keyframe_ref in selected
        ):
            adjacency[relation.from_keyframe_ref].add(relation.to_keyframe_ref)
            adjacency[relation.to_keyframe_ref].add(relation.from_keyframe_ref)
    visited: set[str] = set()
    stack = [sources[0]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    return visited == selected


def _empty_chapter(plan: ChapterPlan) -> SemanticChapter:
    return SemanticChapter(
        chapter_id=plan.chapter_id,
        start_ms=plan.start_ms,
        end_ms=plan.end_ms,
        title=_PLACEHOLDER,
        title_evidence_refs=(),
        summary_zh=_PLACEHOLDER,
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        selected_keyframe_refs=(),
        transcript_source="NONE",
    )


def _fallback_chapter_response(request: ChapterWritingRequest) -> ChapterWritingResponse:
    blocks: list[ChapterBodyBlock] = []
    body_limit = _DETAIL_BODY_LIMITS[request.context.document_config.detail_level]
    for start in range(0, len(request.transcript_evidence), 32):
        batch = request.transcript_evidence[start : start + 32]
        remaining = body_limit - sum(_block_character_count(block) for block in blocks)
        if remaining < 1:
            break
        text = " ".join(item.text for item in batch)[:remaining]
        if text:
            blocks.append(
                ParagraphBlock(
                    text=text,
                    evidence_refs=tuple(item.evidence_id for item in batch),
                ),
            )
    maximum = min(
        request.context.document_config.max_visuals_per_chapter,
        3 if request.chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
    )
    used_sources: set[str] = set()
    for observation in request.visual_observations:
        if observation.relation_to_transcript == "DUPLICATE":
            continue
        selection = _fallback_visual_selection(
            observation,
            used_sources=used_sources,
            maximum=maximum,
        )
        if selection is None:
            continue
        content_refs, current_sources = selection
        visual = VisualBlock(
            visual_observation_ref=observation.evidence_id,
            visual_content_refs=content_refs,
            caption=observation.caption,
            evidence_refs=(observation.evidence_id,),
        )
        remaining = body_limit - sum(_block_character_count(block) for block in blocks)
        if len(visual.caption) > remaining:
            continue
        blocks.append(visual)
        used_sources.update(current_sources)
    return ChapterWritingResponse(
        title=request.chapter.title_hint,
        title_evidence_refs=_fallback_header_evidence_refs(request),
        summary_zh=(
            blocks[0].text
            if blocks and isinstance(blocks[0], ParagraphBlock)
            else request.chapter.title_hint
        )[: _DETAIL_SUMMARY_LIMITS[request.context.document_config.detail_level]],
        summary_evidence_refs=_fallback_header_evidence_refs(request),
        body_blocks=tuple(blocks),
        claims=(),
    )


def _fallback_header_evidence_refs(request: ChapterWritingRequest) -> tuple[str, ...]:
    if request.transcript_evidence:
        return (request.transcript_evidence[0].evidence_id,)
    for observation in request.visual_observations:
        if observation.relation_to_transcript in {"DUPLICATE", "CONFLICTING"}:
            continue
        return (observation.evidence_id,)
    return ()


def _fallback_visual_selection(
    observation: VisualObservationEvidence,
    *,
    used_sources: set[str],
    maximum: int,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    candidates = tuple(
        (
            item.visual_content_id,
            tuple(dict.fromkeys(item.source_keyframe_refs)),
        )
        for item in observation.content_blocks
    ) + tuple(
        (
            item.visual_fact_id,
            tuple(dict.fromkeys(item.source_keyframe_refs)),
        )
        for item in observation.visual_facts
    )
    if not candidates:
        sources = tuple(dict.fromkeys(observation.keyframe_refs))
        if (
            len(used_sources | set(sources)) <= maximum
            and _frames_have_relation_path(observation, sources)
        ):
            return (), sources
        return None
    for content_id, sources in candidates:
        if len(used_sources | set(sources)) > maximum:
            continue
        if _frames_have_relation_path(observation, sources):
            return (content_id,), sources
    return None


def _validate_global_response(
    response: GlobalWritingResponse,
    request: GlobalWritingRequest,
    chapters: tuple[SemanticChapter, ...],
) -> None:
    if not response.overview_zh.strip():
        raise ValueError("全局核心概览不能为空")
    expected = set(allowed_global_chapter_ids(request))
    if not expected:
        raise ValueError("全局编辑至少需要一个章节")


def _fallback_global_response(chapters: tuple[SemanticChapter, ...]) -> GlobalWritingResponse:
    grounded = tuple(chapter for chapter in chapters if chapter.content_status == "GROUNDED")
    if not grounded:
        overview = _PLACEHOLDER
    else:
        overview = "；".join(chapter.summary_zh for chapter in grounded)[:8_000]
    return GlobalWritingResponse(overview_zh=overview)


def _materialize_result(
    context: DocumentWritingContext,
    chapters: tuple[SemanticChapter, ...],
    global_response: GlobalWritingResponse,
    *,
    text_model_id: str,
    vlm_model_id: str,
    prompt_versions: PromptVersions,
) -> VideoUnderstandingResult:
    overview = global_response.overview_zh.strip() or _fallback_global_overview(chapters)
    provisional_summary = VideoDocumentSummary(
        title=context.document_config.document_title or context.title_hint,
        duration_ms=context.duration_ms,
        overview_zh=overview,
    )
    summary = provisional_summary
    return VideoUnderstandingResult(
        run_id=context.run_id,
        asset_sha256=context.asset_sha256,
        summary=summary,
        chapters=chapters,
        generation=DocumentGenerationMetadata(
            document_config=context.document_config,
            text_model_id=text_model_id,
            vlm_model_id=vlm_model_id,
            prompt_versions=prompt_versions,
        ),
    )


def _fallback_global_overview(chapters: tuple[SemanticChapter, ...]) -> str:
    grounded_summaries = tuple(
        chapter.summary_zh.strip()
        for chapter in chapters
        if chapter.content_status == "GROUNDED" and chapter.summary_zh.strip()
    )
    return "；".join(grounded_summaries)[:8_000] or _PLACEHOLDER


def _invalid_local_response(response: FrozenModel, error: BaseException) -> InvalidModelResponse:
    payload = response.model_dump(mode="json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return invalid_model_response(
        raw,
        (str(error)[:500] or "document_writing:invalid",),
        parsed_json=payload,
    )


def _log_chapter_validation_failure(
    *,
    chapter_id: str,
    phase: Literal["main", "repair"],
    code: ErrorCode,
    invalid: InvalidModelResponse,
    provider_attempts: int,
) -> None:
    _LOGGER.warning(
        "章节写作响应校验失败 chapter_id=%s phase=%s code=%s "
        "provider_attempts=%d validation_errors=%s",
        chapter_id,
        phase,
        code,
        provider_attempts,
        ",".join(invalid.validation_errors[:8]),
    )


def _is_fallback_error(error: BaseException) -> bool:
    return isinstance(error, VideoDemoError) and error.code in {
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
    """先阻止尚未开始的提交，再等待已经跨过提交边界的提交完成。"""

    commit_events = tuple(
        event
        for gate in gates
        if (event := gate.cancel()) is not None
    )
    for event in commit_events:
        event.wait()
