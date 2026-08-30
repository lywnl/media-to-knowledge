"""音频章节写作内核。

该模块只消费音频章节计划和转写证据，直接生成音频结果，不经过其他媒体结果。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from pydantic import Field

from video_demo.application.audio_rendering import RenderedAudioDocument
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.domain.audio_plan import (
    AudioChapterPlan,
    AudioDocumentConfig,
    AudioGroundedClaim,
    AudioParagraphBlock,
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
    AudioGlobalWritingRequest,
    AudioGlobalWritingResponse,
    AudioInvalidModelResponse,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity


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
        evidence_by_chapter = tuple(
            tuple(item for item in transcript_evidence if plan.contains(item)) for plan in plans
        )
        chapters: list[AudioChapter | None] = [None] * len(plans)
        warnings: list[str] = []
        with ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="audio-writing"
        ) as executor:
            futures: dict[Future[tuple[AudioChapter, bool]], int] = {
                executor.submit(
                    self._write_one,
                    context,
                    plan,
                    evidence,
                    cache,
                    is_cancel_requested,
                ): index
                for index, (plan, evidence) in enumerate(
                    zip(plans, evidence_by_chapter, strict=True)
                )
            }
            while futures:
                if is_cancel_requested():
                    for future in futures:
                        future.cancel()
                    raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
                done, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    chapter, degraded = future.result()
                    chapters[index] = chapter
                    if degraded:
                        warnings.append(f"AUDIO_CHAPTER_WRITING_FALLBACK:{chapter.chapter_id}")
        complete = tuple(item for item in chapters if item is not None)
        overview = self._write_global(context, complete, cache, is_cancel_requested)
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
        from video_demo.application.audio_rendering import render_audio_markdown

        rendered = render_audio_markdown(result)
        unique_warnings = tuple(dict.fromkeys(warnings))
        return AudioWrittenDocument(
            result=result,
            document=rendered,
            warnings=unique_warnings,
            status="PARTIAL_SUCCEEDED" if unique_warnings else "SUCCEEDED",
        )

    def _write_one(
        self,
        context: AudioWritingContext,
        plan: AudioChapterPlan,
        evidence: tuple[AudioTranscriptEvidence, ...],
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[AudioChapter, bool]:
        request = AudioChapterWritingRequest(
            run_id=context.run_id,
            asset_sha256=context.asset_sha256,
            title_hint=plan.title_hint,
            duration_ms=context.duration_ms,
            transcript_source=context.transcript_source,
            document_config=context.document_config,
            chapter=plan,
            transcript_evidence=evidence,
            prompt_version="audio-chapter-writer-v1",
        )
        try:
            cached = cache.get(
                self._chapter_identity,
                request,
                AudioChapterWritingResponse,
                lambda item: _validate_response(item, request),
            )
            if cached is not None:
                return _materialize_chapter(plan, cached.response, context.transcript_source), False
            with cache.invocation_lock(
                self._chapter_identity,
                request,
                wait_timeout_seconds=self._wait_timeout,
                is_cancel_requested=is_cancel_requested,
            ):
                response = self._port.write_chapter(request, on_provider_attempt=None)
                try:
                    _validate_response(response, request)
                except (ValueError, TypeError):
                    response = self._port.repair_chapter_writing(
                        AudioChapterWritingRepairRequest(
                            request=request,
                            invalid_response=AudioInvalidModelResponse(
                                content_sha256="f" * 64,
                                validation_errors=("audio_response:invalid",),
                            ),
                            allowed_evidence_ids=tuple(item.evidence_id for item in evidence),
                            prompt_version="audio-chapter-writer-repair-v1",
                        ),
                        on_provider_attempt=None,
                    )
                    _validate_response(response, request)
                cache.put(
                    self._chapter_identity,
                    request,
                    response,
                    successful_path="MAIN",
                    validate=lambda item: _validate_response(item, request),
                )
                return _materialize_chapter(plan, response, context.transcript_source), False
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_CANCELLED:
                raise
            return _fallback_chapter(plan, evidence, context.transcript_source), True
        except (ValueError, TypeError):
            return _fallback_chapter(plan, evidence, context.transcript_source), True

    def _write_global(
        self,
        context: AudioWritingContext,
        chapters: tuple[AudioChapter, ...],
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> str:
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
        try:
            cached = cache.get(
                self._global_identity,
                request,
                AudioGlobalWritingResponse,
                lambda item: _validate_global_response(item),
            )
            if cached is not None:
                return cached.response.overview_zh.strip() or _fallback_overview(chapters)
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
                    lambda item: _validate_global_response(item),
                )
                if cached is not None:
                    return cached.response.overview_zh.strip() or _fallback_overview(chapters)
                response = self._port.organize_document(request, on_provider_attempt=None)
                _validate_global_response(response)
                cache.put(
                    self._global_identity,
                    request,
                    response,
                    successful_path="MAIN",
                    validate=lambda item: _validate_global_response(item),
                )
            return response.overview_zh.strip() or _fallback_overview(chapters)
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_CANCELLED:
                raise
            return _fallback_overview(chapters)
        except (ValueError, TypeError, AttributeError):
            return _fallback_overview(chapters)


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
    if not set(refs) <= allowed:
        raise ValueError("音频写作响应引用未知证据")


def _validate_global_response(response: AudioGlobalWritingResponse) -> None:
    if not response.overview_zh.strip():
        raise ValueError("音频核心概览不能为空")


def _materialize_chapter(
    plan: AudioChapterPlan, response: AudioChapterWritingResponse, source: str
) -> AudioChapter:
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
    text = " ".join(item.text for item in evidence)[:4_000] or plan.title_hint
    paragraph = AudioParagraphBlock(text=text, evidence_refs=refs[:32]) if refs else None
    return AudioChapter(
        chapter_id=plan.chapter_id,
        start_ms=plan.start_ms,
        end_ms=plan.end_ms,
        title=plan.title_hint,
        title_evidence_refs=refs[:1],
        summary_zh=text[:4_000],
        summary_evidence_refs=refs[:1],
        body_blocks=(paragraph,) if paragraph else (),
        claims=(AudioGroundedClaim(text=text[:2_000], evidence_refs=refs[:1], certainty=0.6),)
        if refs
        else (),
        evidence_refs=refs,
        transcript_source=source if refs else "NONE",
        content_status="GROUNDED" if refs else "NO_SEMANTIC_EVIDENCE",
    )


def _fallback_overview(chapters: tuple[AudioChapter, ...]) -> str:
    return (
        "；".join(item.summary_zh for item in chapters if item.summary_zh)[:8_000]
        or "未提取到可验证语义内容。"
    )
