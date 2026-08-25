from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import pairwise
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
    ParagraphBlock,
    PromptVersions,
    QuoteBlock,
    SectionDraft,
    SemanticChapter,
    SemanticSection,
    SummaryPoint,
    TableBlock,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    VisualBlock,
    section_id_for,
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
from video_demo.fusion.retrieval_text import (
    render_document_chapter_retrieval_text,
    render_document_summary_retrieval_text,
)
from video_demo.integrations.document_port import (
    ChapterWritingRepairRequest,
    ChapterWritingRequest,
    ChapterWritingResponse,
    DocumentTextPort,
    GlobalChapterGroup,
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
_MAX_GLOBAL_GROUP_CHAPTERS = 20
_MAX_GLOBAL_DIGEST_CHARS = 4_000
_MAX_INPUT_CHARS = 60_000
_MAX_INPUT_BYTES = 1_048_576


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


@dataclass(frozen=True, slots=True)
class _ChapterOutcome:
    chapter: SemanticChapter
    degraded: bool
    provider_attempts: int
    structure_repairs: int
    cache_hits: int


class DocumentWriter:
    """把有界模型草稿收敛为程序拥有引用、顺序与检索投影的 3.0 结果。"""

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
        if chapter_identity.repair_prompt_version != prompt_versions.chapter_writer_repair:
            raise ValueError("章节写作缓存身份与修复 Prompt 版本不一致")
        if chapter_identity.model_id != text_model_id:
            raise ValueError("章节写作缓存身份与文本模型不一致")
        if global_identity.main_prompt_version != prompt_versions.global_editor:
            raise ValueError("全局编辑缓存身份与 Prompt 版本不一致")
        if global_identity.repair_prompt_version != prompt_versions.global_editor_repair:
            raise ValueError("全局编辑缓存身份与修复 Prompt 版本不一致")
        if global_identity.model_id != text_model_id:
            raise ValueError("全局编辑缓存身份与文本模型不一致")
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
        chapters = tuple(item.chapter for item in outcomes)
        chapter_degraded = tuple(item.chapter.chapter_id for item in outcomes if item.degraded)
        global_response, global_degraded = self._write_global(
            context,
            chapters,
            cache,
            counters,
            is_cancel_requested,
        )
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
                _validate_chapter_response(response, request_to_validate)

            cached = cache.get(
                self._chapter_identity,
                concrete_request,
                ChapterWritingResponse,
                validate,
            )
            if cached is not None:
                normalized = _normalize_response_blocks(cached.response, concrete_request)
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
        in_flight: dict[Future[_ChapterOutcome], int] = {}
        next_position = 0
        try:
            while next_position < len(pending_indexes) or in_flight:
                if is_cancel_requested():
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                while next_position < len(pending_indexes) and len(in_flight) < self._concurrency:
                    index = pending_indexes[next_position]
                    request = packages[index][1]
                    assert request is not None
                    future = executor.submit(
                        self._write_one,
                        request,
                        keyframes,
                        cache,
                        is_cancel_requested,
                    )
                    in_flight[future] = index
                    next_position += 1
                completed, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in completed:
                    index = in_flight.pop(future)
                    outcome = future.result()
                    outcomes[index] = outcome
                    counters.chapter_attempts += outcome.provider_attempts
                    counters.chapter_repairs += outcome.structure_repairs
                    counters.chapter_cache_hits += outcome.cache_hits
                    counters.chapter_fallbacks += int(outcome.degraded)
        finally:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return cast(tuple[_ChapterOutcome, ...], tuple(outcomes))

    def _write_one(
        self,
        request: ChapterWritingRequest,
        keyframes: tuple[KeyframeEvidence, ...],
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> _ChapterOutcome:
        local = _Counters()
        response = self._chapter_logical_call(
            request,
            cache,
            local,
            is_cancel_requested,
        )
        degraded = response is None
        if response is None:
            response = _fallback_chapter_response(request)
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
    ) -> ChapterWritingResponse | None:
        def validate(response: ChapterWritingResponse) -> None:
            _validate_chapter_response(response, request)

        cached = cache.get(self._chapter_identity, request, ChapterWritingResponse, validate)
        if cached is not None:
            counters.chapter_cache_hits += 1
            return cached.response
        with cache.invocation_lock(
            self._chapter_identity,
            request,
            wait_timeout_seconds=self._wait_timeout_seconds,
            is_cancel_requested=is_cancel_requested,
        ):
            cached = cache.get(self._chapter_identity, request, ChapterWritingResponse, validate)
            if cached is not None:
                counters.chapter_cache_hits += 1
                return cached.response
            invalid: InvalidModelResponse | None = None
            try:
                response = self._text_port.write_chapter(
                    request,
                    on_provider_attempt=counters.chapter_attempt,
                )
            except ModelResponseValidationError as error:
                invalid = error.invalid_response
            except VideoDemoError as error:
                if _is_fallback_error(error):
                    return None
                raise
            else:
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    invalid = _invalid_local_response(response, error)
            if invalid is None:
                path: Literal["MAIN", "REPAIR"] = "MAIN"
            else:
                counters.chapter_repairs += 1
                try:
                    response = self._text_port.repair_chapter_writing(
                        ChapterWritingRepairRequest(
                            request=request,
                            invalid_response=invalid,
                            allowed_evidence_ids=allowed_writing_evidence_ids(request),
                            prompt_version=self._prompt_versions.chapter_writer_repair,
                        ),
                        on_provider_attempt=counters.chapter_attempt,
                    )
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
                path = "REPAIR"
            return cache.put(
                self._chapter_identity,
                request,
                response,
                successful_path=path,
                validate=validate,
            ).response

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
        def validate(response: GlobalWritingResponse) -> None:
            _validate_global_response(response, request, chapters)

        cached = cache.get(self._global_identity, request, GlobalWritingResponse, validate)
        if cached is not None:
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
                counters.global_cache_hits += 1
                return cached.response, False
            invalid: InvalidModelResponse | None = None
            try:
                response = self._text_port.organize_document(
                    request,
                    on_provider_attempt=counters.global_attempt,
                )
            except ModelResponseValidationError as error:
                invalid = error.invalid_response
            except VideoDemoError as error:
                if _is_fallback_error(error):
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                raise
            else:
                try:
                    validate(response)
                except (ValueError, TypeError) as error:
                    invalid = _invalid_local_response(response, error)
            if invalid is None:
                path: Literal["MAIN", "REPAIR"] = "MAIN"
            else:
                counters.global_repairs += 1
                try:
                    response = self._text_port.repair_global_writing(
                        GlobalWritingRepairRequest(
                            request=request,
                            invalid_response=invalid,
                            allowed_chapter_ids=allowed_global_chapter_ids(request),
                            prompt_version=self._prompt_versions.global_editor_repair,
                        ),
                        on_provider_attempt=counters.global_attempt,
                    )
                except ModelResponseValidationError:
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                except VideoDemoError as error:
                    if _is_fallback_error(error):
                        counters.global_fallbacks += 1
                        return _fallback_global_response(chapters), True
                    raise
                try:
                    validate(response)
                except (ValueError, TypeError):
                    counters.global_fallbacks += 1
                    return _fallback_global_response(chapters), True
                path = "REPAIR"
            response = cache.put(
                self._global_identity,
                request,
                response,
                successful_path=path,
                validate=validate,
            ).response
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
        if chapter_observations:
            chapter_observations = chapter_observations[:-1]
        elif len(chapter_transcript) > 1:
            chapter_transcript = chapter_transcript[:-1]
        else:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "单章最小证据包超过输入预算")


def _chapter_request_fits(request: ChapterWritingRequest, max_chars: int, max_bytes: int) -> bool:
    repair = ChapterWritingRepairRequest(
        request=request,
        invalid_response=_worst_invalid_response(),
        allowed_evidence_ids=allowed_writing_evidence_ids(request),
        prompt_version="chapter-writer-repair-v1",
    )
    return _prompts_fit(
        (prompt_for_writing(request)[2], prompt_for_writing_repair(repair)[2]),
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
    groups = list(_global_groups(chapters))
    request = GlobalWritingRequest(
        context=context,
        groups=tuple(groups),
        prompt_version=prompt_version,
    )
    if _global_request_fits(request, max_chars, max_bytes):
        return request
    original_lengths = [len(group.digest_zh) for group in groups]
    for numerator in range(99, 0, -1):
        resized: list[GlobalChapterGroup] = []
        for group, original_length in zip(groups, original_lengths, strict=True):
            if not group.grounded_chapter_refs:
                resized.append(group)
                continue
            length = max(1, original_length * numerator // 100)
            resized.append(group.model_copy(update={"digest_zh": group.digest_zh[:length]}))
        request = GlobalWritingRequest(
            context=context,
            groups=tuple(resized),
            prompt_version=prompt_version,
        )
        if _global_request_fits(request, max_chars, max_bytes):
            return request
    raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "全局编辑最小输入超过预算")


def _global_groups(chapters: tuple[SemanticChapter, ...]) -> tuple[GlobalChapterGroup, ...]:
    groups: list[GlobalChapterGroup] = []
    for start in range(0, len(chapters), _MAX_GLOBAL_GROUP_CHAPTERS):
        batch = chapters[start : start + _MAX_GLOBAL_GROUP_CHAPTERS]
        grounded = tuple(
            chapter for chapter in batch if chapter.content_status == "GROUNDED"
        )
        digest = "\n".join(
            f"{chapter.title}|{chapter.summary_zh}|{chapter.retrieval_text}"
            for chapter in grounded
        )[:_MAX_GLOBAL_DIGEST_CHARS]
        groups.append(
            GlobalChapterGroup(
                start_ms=batch[0].start_ms,
                end_ms=batch[-1].end_ms,
                chapter_refs=tuple(chapter.chapter_id for chapter in batch),
                grounded_chapter_refs=tuple(chapter.chapter_id for chapter in grounded),
                digest_zh=digest,
            ),
        )
    return tuple(groups)


def _global_request_fits(request: GlobalWritingRequest, max_chars: int, max_bytes: int) -> bool:
    repair = GlobalWritingRepairRequest(
        request=request,
        invalid_response=_worst_invalid_response(),
        allowed_chapter_ids=allowed_global_chapter_ids(request),
        prompt_version="global-editor-repair-v1",
    )
    return _prompts_fit(
        (prompt_for_global_editing(request)[2], prompt_for_global_repair(repair)[2]),
        max_chars,
        max_bytes,
    )


def _prompts_fit(prompts: Iterable[str], max_chars: int, max_bytes: int) -> bool:
    return all(
        len(item) <= max_chars and len(item.encode("utf-8")) <= max_bytes
        for item in prompts
    )


def _worst_invalid_response() -> InvalidModelResponse:
    return InvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=tuple(f"field_{index:02d}:" + "x" * 488 for index in range(32)),
        safe_json_excerpt="x" * 8_000,
    )


def _validate_chapter_response(
    response: ChapterWritingResponse,
    request: ChapterWritingRequest,
) -> None:
    allowed = set(allowed_writing_evidence_ids(request))
    observations = {item.evidence_id: item for item in request.visual_observations}
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
        if (
            isinstance(block, QuoteBlock)
            and not request.context.document_config.include_verbatim_quotes
        ):
            raise ValueError("当前配置禁止逐字引用")
    for claim in response.claims:
        if not set(claim.evidence_refs).issubset(allowed):
            raise ValueError("Claim 引用了证据包之外的 ID")
    detail = request.context.document_config.detail_level
    if len(response.summary_zh) > _DETAIL_SUMMARY_LIMITS[detail]:
        raise ValueError("章节摘要超过配置字符预算")
    if _response_body_characters(response) > _DETAIL_BODY_LIMITS[detail]:
        raise ValueError("章节正文超过配置字符预算")
    normalized = _normalize_response_blocks(response, request)
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
            if observation.relation_to_transcript == "CONFLICTING":
                if request.context.document_config.uncertainty_policy == "conservative":
                    caption = _conservative_conflict_caption(observation)
                else:
                    caption = _conflict_aware_caption(observation)
                blocks.append(
                    block.model_copy(update={"caption": caption}),
                )
                continue
        blocks.append(block)
    return response.model_copy(update={"body_blocks": tuple(blocks)})


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
    selected = _selected_keyframes(
        response.body_blocks,
        request.visual_observations,
        keyframes,
    )
    evidence_refs = tuple(
        dict.fromkeys(
            (
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
        summary_zh=response.summary_zh,
        body_blocks=response.body_blocks,
        claims=response.claims,
        evidence_refs=evidence_refs,
        selected_keyframe_refs=selected,
        transcript_source=(
            request.context.transcript_source if request.transcript_evidence else "NONE"
        ),
        retrieval_text="",
        retrieval_hash=hashlib.sha256(b"").hexdigest(),
    )
    retrieval = render_document_chapter_retrieval_text(provisional, request.visual_observations)
    return provisional.model_copy(
        update={
            "retrieval_text": retrieval,
            "retrieval_hash": hashlib.sha256(retrieval.encode("utf-8")).hexdigest(),
        },
    )


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
        if len(set(sources)) > 1 and not observation.frame_relations:
            raise ValueError("多图视觉正文必须包含对应帧关系")
        for ref in sources:
            if validate_keyframe_membership and ref not in allowed_keyframes:
                raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "展示图未晋升")
            if ref not in selected:
                selected.append(ref)
    return tuple(selected)


def _empty_chapter(plan: ChapterPlan) -> SemanticChapter:
    return SemanticChapter(
        chapter_id=plan.chapter_id,
        start_ms=plan.start_ms,
        end_ms=plan.end_ms,
        title=_PLACEHOLDER,
        summary_zh=_PLACEHOLDER,
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        selected_keyframe_refs=(),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash=hashlib.sha256(b"").hexdigest(),
    )


def _fallback_chapter_response(request: ChapterWritingRequest) -> ChapterWritingResponse:
    blocks: list[ChapterBodyBlock] = []
    if request.transcript_evidence:
        blocks.append(
            ParagraphBlock(
                text=" ".join(item.text for item in request.transcript_evidence)[
                    : _DETAIL_BODY_LIMITS[request.context.document_config.detail_level]
                ],
                evidence_refs=tuple(item.evidence_id for item in request.transcript_evidence),
            ),
        )
    for observation in request.visual_observations:
        if observation.relation_to_transcript == "DUPLICATE":
            continue
        current_sources = _observation_source_refs(observation)
        maximum = min(
            request.context.document_config.max_visuals_per_chapter,
            3 if request.chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
        )
        used_sources = {
            ref
            for block in blocks
            if isinstance(block, VisualBlock)
            for ref in _observation_source_refs(
                next(
                    item
                    for item in request.visual_observations
                    if item.evidence_id == block.visual_observation_ref
                ),
            )
        }
        if len(used_sources | set(current_sources)) > maximum:
            continue
        visual = VisualBlock(
            visual_observation_ref=observation.evidence_id,
            visual_content_refs=tuple(
                (
                    *(item.visual_content_id for item in observation.content_blocks),
                    *(item.visual_fact_id for item in observation.visual_facts),
                ),
            ),
            caption=(
                _conservative_conflict_caption(observation)
                if observation.relation_to_transcript == "CONFLICTING"
                and request.context.document_config.uncertainty_policy == "conservative"
                else _conflict_aware_caption(observation)
            ),
            evidence_refs=(observation.evidence_id,),
        )
        remaining = _DETAIL_BODY_LIMITS[
            request.context.document_config.detail_level
        ] - sum(_block_character_count(block) for block in blocks)
        if remaining < 1:
            continue
        blocks.append(visual.model_copy(update={"caption": visual.caption[:remaining]}))
    return ChapterWritingResponse(
        title=request.chapter.title_hint,
        summary_zh=(
            blocks[0].text
            if blocks and isinstance(blocks[0], ParagraphBlock)
            else request.chapter.title_hint
        )[: _DETAIL_SUMMARY_LIMITS[request.context.document_config.detail_level]],
        body_blocks=tuple(blocks),
        claims=(),
    )


def _conflict_aware_caption(observation: VisualObservationEvidence) -> str:
    if observation.relation_to_transcript != "CONFLICTING":
        return observation.caption
    uncertainty = "；".join(observation.uncertainties)
    return f"{observation.caption} (音画信息存在冲突：{uncertainty})"


def _conservative_conflict_caption(observation: VisualObservationEvidence) -> str:
    uncertainty = "；".join(observation.uncertainties)
    return f"画面信息与转写存在冲突，未采纳为确定事实：{uncertainty}"


def _observation_source_refs(observation: VisualObservationEvidence) -> tuple[str, ...]:
    sources = tuple(
        ref for content in observation.content_blocks for ref in content.source_keyframe_refs
    ) + tuple(ref for fact in observation.visual_facts for ref in fact.source_keyframe_refs)
    return tuple(dict.fromkeys(sources or observation.keyframe_refs))


def _validate_global_response(
    response: GlobalWritingResponse,
    request: GlobalWritingRequest,
    chapters: tuple[SemanticChapter, ...],
) -> None:
    expected = allowed_global_chapter_ids(request)
    actual = tuple(ref for section in response.sections for ref in section.chapter_refs)
    if actual != expected:
        raise ValueError("Section 必须按顺序完整覆盖所有章节一次")
    grounded = {chapter.chapter_id for chapter in chapters if chapter.content_status == "GROUNDED"}
    if any(ref not in grounded for point in response.key_points for ref in point.chapter_refs):
        raise ValueError("全局关键点只能引用事实章节")
    placeholder_ids = {
        chapter.chapter_id
        for chapter in chapters
        if chapter.content_status == "NO_SEMANTIC_EVIDENCE"
    }
    for section in response.sections:
        if placeholder_ids.intersection(section.chapter_refs) and "信息不足" not in (
            section.title + section.summary_zh
        ):
            raise ValueError("覆盖占位章的 Section 必须明确该时段信息不足")


def _fallback_global_response(chapters: tuple[SemanticChapter, ...]) -> GlobalWritingResponse:
    grounded = tuple(chapter for chapter in chapters if chapter.content_status == "GROUNDED")
    if not grounded:
        overview = _PLACEHOLDER
        points: tuple[SummaryPoint, ...] = ()
    else:
        overview = "；".join(chapter.summary_zh for chapter in grounded)[:8_000]
        points = tuple(
            SummaryPoint(text=chapter.summary_zh, chapter_refs=(chapter.chapter_id,))
            for chapter in grounded[:64]
            if chapter.summary_zh
        )
    sections = tuple(
        SectionDraft(
            title=chapter.title if chapter.content_status == "GROUNDED" else "信息不足时段",
            summary_zh=chapter.summary_zh if chapter.content_status == "GROUNDED" else "",
            chapter_refs=(chapter.chapter_id,),
        )
        for chapter in chapters
    )
    return GlobalWritingResponse(overview_zh=overview, key_points=points, sections=sections)


def _materialize_result(
    context: DocumentWritingContext,
    chapters: tuple[SemanticChapter, ...],
    global_response: GlobalWritingResponse,
    *,
    text_model_id: str,
    vlm_model_id: str,
    prompt_versions: PromptVersions,
) -> VideoUnderstandingResult:
    sections = tuple(
        SemanticSection(
            section_id=section_id_for(context.asset_sha256, draft.chapter_refs),
            title=draft.title,
            summary_zh=draft.summary_zh,
            chapter_refs=draft.chapter_refs,
        )
        for draft in global_response.sections
    )
    provisional_summary = VideoDocumentSummary(
        title=context.document_config.document_title or context.title_hint,
        duration_ms=context.duration_ms,
        overview_zh=global_response.overview_zh,
        key_points=global_response.key_points,
        retrieval_text="",
        retrieval_hash=hashlib.sha256(b"").hexdigest(),
    )
    retrieval = (
        render_document_summary_retrieval_text(provisional_summary)
        if any(chapter.content_status == "GROUNDED" for chapter in chapters)
        else ""
    )
    summary = provisional_summary.model_copy(
        update={
            "retrieval_text": retrieval,
            "retrieval_hash": hashlib.sha256(retrieval.encode("utf-8")).hexdigest(),
        },
    )
    return VideoUnderstandingResult(
        run_id=context.run_id,
        asset_sha256=context.asset_sha256,
        summary=summary,
        sections=sections,
        chapters=chapters,
        generation=DocumentGenerationMetadata(
            document_config=context.document_config,
            text_model_id=text_model_id,
            vlm_model_id=vlm_model_id,
            prompt_versions=prompt_versions,
        ),
    )


def _invalid_local_response(response: FrozenModel, error: BaseException) -> InvalidModelResponse:
    payload = response.model_dump(mode="json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return invalid_model_response(
        raw,
        (str(error)[:500] or "document_writing:invalid",),
        parsed_json=payload,
    )


def _is_fallback_error(error: BaseException) -> bool:
    return isinstance(error, VideoDemoError) and error.code in {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
    }
