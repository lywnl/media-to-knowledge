from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field, StrictInt, field_validator, model_validator

from video_demo.application.chapter_frames import ChapterFrameSearchBatch, ChapterFrameStatus
from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_artifact import MAX_METRIC_VALUE
from video_demo.domain.document_plan import ChapterFrameSet, ChapterPlan, FrameCandidateArtifact
from video_demo.domain.evidence import (
    ChapterVisualObservation,
    GroundedVisualFact,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    VisualCodeContent,
    VisualCodeContentDraft,
    VisualDiagramContent,
    VisualDiagramContentDraft,
    VisualFactDraft,
    VisualFieldChange,
    VisualFormulaContent,
    VisualFormulaContentDraft,
    VisualFrameRelation,
    VisualFrameRelationDraft,
    VisualObservationEvidence,
    VisualStateContent,
    VisualStateContentDraft,
    VisualTableContent,
    VisualTableContentDraft,
    VisualTextContent,
    VisualTextContentDraft,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterVisionPort,
    ChapterVisionRepairRequest,
    ChapterVisionRequest,
    ChapterVisionResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import (
    prompt_for_vision,
    prompt_for_vision_repair,
    vision_payload_size_upper_bound,
)
from video_demo.integrations.document_validation import validate_chapter_vision_response
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity
from video_demo.visual.candidate_artifacts import (
    CandidateDirectoryLease,
    read_verified_candidate_jpeg,
    validate_candidate_descriptor,
)
from video_demo.visual.keyframe_artifacts import KeyframeArtifactSession

TranscriptEvidence: TypeAlias = SpeechSegment | SubtitleCue
ChapterVisionStatus: TypeAlias = Literal[
    "SUCCEEDED",
    "DEGRADED",
    "NO_VALUE",
    "NO_CANDIDATE",
    "DISABLED",
]

_METRIC_NAMES = frozenset(
    {
        "vlm_logical_analyses",
        "vlm_provider_attempts",
        "vlm_structure_repairs",
        "vlm_cache_hits",
        "vlm_no_value_chapters",
        "vlm_fallback_chapters",
        "visual_published_budget_degraded_chapters",
    },
)
_FALLBACK_CODES = frozenset(
    {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.QWEN_RESPONSE_INVALID,
    },
)
_RELATION_VALUE = {
    "DUPLICATE": 0,
    "SUPPORTING": 1,
    "INDEPENDENT": 2,
    "COMPLEMENTARY": 3,
    "CONFLICTING": 4,
}
_LOGGER = logging.getLogger(__name__)


class ChapterVisionBatch(FrozenModel):
    observations: tuple[VisualObservationEvidence, ...]
    evidence: tuple[KeyframeEvidence | VisualObservationEvidence, ...]
    keyframe_evidence: tuple[KeyframeEvidence, ...]
    chapter_status: tuple[tuple[StableId, ChapterVisionStatus], ...]
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) - _METRIC_NAMES:
            raise ValueError("章节视觉指标包含未知白名单键")
        if any(
            type(metric) is not int or not 0 <= metric <= MAX_METRIC_VALUE
            for metric in value.values()
        ):
            raise ValueError("章节视觉指标必须是非负严格整数")
        return value

    @model_validator(mode="after")
    def validate_batch(self) -> ChapterVisionBatch:
        chapter_ids = tuple(chapter_id for chapter_id, _status in self.chapter_status)
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("章节视觉状态 ID 不得重复")
        expected_evidence = (*self.keyframe_evidence, *self.observations)
        if self.evidence != expected_evidence:
            raise ValueError("视觉证据必须由关键帧和观察按固定顺序组成")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("视觉证据 ID 不得重复")
        keyframe_ids = {item.evidence_id for item in self.keyframe_evidence}
        referenced = {ref for item in self.observations for ref in item.keyframe_refs}
        if keyframe_ids != referenced:
            raise ValueError("关键帧与视觉观察引用闭包不一致")
        partial = any(status == "DEGRADED" for _chapter_id, status in self.chapter_status)
        if partial != (self.status == "PARTIAL_SUCCEEDED"):
            raise ValueError("视觉批次状态与章节降级状态不一致")
        return self


class ChapterVisionFrameCacheInput(FrozenModel):
    frame_id: StableId
    timestamp_ms: int = Field(ge=0)
    sha256: Sha256
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)


class ChapterVisionCacheInput(FrozenModel):
    chapter: ChapterPlan
    frames: tuple[ChapterVisionFrameCacheInput, ...] = Field(min_length=1, max_length=6)
    transcript_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=20_000)
    document_config: DocumentGenerationConfig


class _ChapterCounters:
    def __init__(self, *, logical_analyses: int) -> None:
        self.logical_analyses = logical_analyses
        self.provider_attempts = 0
        self.structure_repairs = 0
        self.cache_hits = 0
        self.no_value = 0
        self.fallback = 0

    def provider_attempt(self) -> None:
        self.provider_attempts += 1


@dataclass(frozen=True, slots=True)
class _ChapterAnalysis:
    chapter: ChapterPlan
    frames: tuple[FrameCandidateArtifact, ...]
    response: ChapterVisionResponse | None
    status: ChapterVisionStatus
    warning: str | None
    counters: _ChapterCounters


@dataclass(frozen=True, slots=True)
class _MaterializedObservation:
    chapter: ChapterPlan
    draft: ChapterVisualObservation
    frames: tuple[FrameCandidateArtifact, ...]


class _DeferredExecutorCleanup(Exception):
    def __init__(self, error: BaseException, executor: ThreadPoolExecutor) -> None:
        super().__init__(str(error))
        self.error = error
        self.executor = executor


class _DeferredCleanupHandoff:
    """在外部调用前启动，确保取消时只移交收尾资源。"""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._dispatched = threading.Event()
        self._resources: tuple[ThreadPoolExecutor, CandidateDirectoryLease] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="chapter-vision-deferred-cleanup",
            daemon=True,
        )

    def start(self, timeout_seconds: float) -> None:
        self._thread.start()
        if self._ready.wait(timeout=timeout_seconds):
            return
        self._dispatched.set()
        raise VideoDemoError(
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            "章节视觉收尾资源启动超时",
        )

    def defer(
        self,
        executor: ThreadPoolExecutor,
        lease: CandidateDirectoryLease,
    ) -> None:
        self._resources = (executor, lease)
        self._dispatched.set()

    def close(self) -> None:
        self._dispatched.set()

    def _run(self) -> None:
        self._ready.set()
        self._dispatched.wait()
        if self._resources is None:
            return
        executor, lease = self._resources
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            lease.close()


class ChapterVisionService:
    """把章节候选帧收敛为可追溯的视觉观察和已晋升关键帧。"""

    def __init__(
        self,
        vision_port: ChapterVisionPort,
        identity: ModelInvocationIdentity,
        *,
        runtime_root: Path,
        concurrency: int,
        max_image_bytes: int,
        max_request_image_bytes: int,
        max_encoded_request_bytes: int,
        max_published_keyframe_bytes: int,
        max_published_keyframe_files: int,
        invocation_wait_timeout_seconds: float,
        candidate_lock_timeout_seconds: float,
    ) -> None:
        positive_ints = (
            concurrency,
            max_image_bytes,
            max_request_image_bytes,
            max_encoded_request_bytes,
            max_published_keyframe_bytes,
            max_published_keyframe_files,
        )
        if any(type(value) is not int or value < 1 for value in positive_ints):
            raise ValueError("章节视觉并发和预算必须是正整数")
        if concurrency > 2:
            raise ValueError("章节视觉并发不得超过 2")
        if (
            not math.isfinite(invocation_wait_timeout_seconds)
            or invocation_wait_timeout_seconds <= 0
            or not math.isfinite(candidate_lock_timeout_seconds)
            or candidate_lock_timeout_seconds <= 0
        ):
            raise ValueError("章节视觉锁超时必须为有限正数")
        if identity.logical_operation != "chapter_vision":
            raise ValueError("章节视觉缓存身份与逻辑操作不一致")
        if identity.main_prompt_version != "chapter-vlm-v1":
            raise ValueError("章节视觉缓存身份与 Prompt 版本不一致")
        if identity.repair_prompt_version != "chapter-vlm-repair-v1":
            raise ValueError("章节视觉缓存身份与修复 Prompt 版本不一致")
        if identity.main_response_schema_name != "chapter_vlm_v2":
            raise ValueError("章节视觉缓存身份与主响应 Schema 不一致")
        if identity.repair_response_schema_name != "chapter_vlm_repair_v2":
            raise ValueError("章节视觉缓存身份与修复响应 Schema 不一致")
        self._vision_port = vision_port
        self._identity = identity
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._concurrency = concurrency
        self._max_image_bytes = max_image_bytes
        self._max_request_image_bytes = max_request_image_bytes
        self._max_encoded_request_bytes = max_encoded_request_bytes
        self._max_published_keyframe_bytes = max_published_keyframe_bytes
        self._max_published_keyframe_files = max_published_keyframe_files
        self._invocation_wait_timeout_seconds = invocation_wait_timeout_seconds
        self._candidate_lock_timeout_seconds = candidate_lock_timeout_seconds

    def analyze_all(
        self,
        chapters: tuple[ChapterPlan, ...],
        frame_batch: ChapterFrameSearchBatch,
        transcript_evidence: tuple[TranscriptEvidence, ...],
        document_config: DocumentGenerationConfig,
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterVisionBatch:
        frame_sets, frame_status = _validate_batch_alignment(chapters, frame_batch)
        if document_config.max_visuals_per_chapter == 0:
            return _disabled_batch(chapters)
        _validate_frame_batch_semantics(
            chapters,
            frame_sets,
            frame_status,
            asset_sha256=frame_batch.asset_sha256,
            allowed_run_root=frame_batch.allowed_run_root,
        )
        transcripts = _validate_transcripts(chapters, transcript_evidence)
        if not any(frame_set.candidates for frame_set in frame_sets.values()):
            return _empty_batch(chapters, frame_status, frame_batch.status)
        lease = CandidateDirectoryLease.from_allowed_run_root(
            runtime_root=self._runtime_root,
            allowed_run_root=frame_batch.allowed_run_root,
            mode="EXCLUSIVE",
            is_cancel_requested=is_cancel_requested,
            wait_timeout_seconds=self._candidate_lock_timeout_seconds,
        )
        lease.acquire()
        lease_transferred = False
        cleanup_handoff = _DeferredCleanupHandoff()
        try:
            _verify_candidate_batch_files(
                frame_sets,
                frame_batch.asset_sha256,
                frame_batch.allowed_run_root,
                max_bytes=self._max_image_bytes,
            )
            cleanup_handoff.start(self._invocation_wait_timeout_seconds)
            analyses = self._analyze_chapters(
                chapters,
                frame_sets,
                frame_status,
                transcripts,
                document_config,
                cache,
                frame_batch.allowed_run_root,
                is_cancel_requested,
            )
            return self._assemble_batch(
                analyses,
                frame_batch,
                frame_batch.allowed_run_root,
                frame_tolerance_ms=frame_batch.frame_tolerance_ms,
            )
        except _DeferredExecutorCleanup as deferred:
            cleanup_handoff.defer(deferred.executor, lease)
            lease_transferred = True
            raise deferred.error.with_traceback(deferred.error.__traceback__) from None
        finally:
            if not lease_transferred:
                cleanup_handoff.close()
                lease.close()

    def _analyze_chapters(
        self,
        chapters: tuple[ChapterPlan, ...],
        frame_sets: Mapping[str, ChapterFrameSet],
        frame_status: Mapping[str, ChapterFrameStatus],
        transcripts: Mapping[str, tuple[TranscriptEvidence, ...]],
        document_config: DocumentGenerationConfig,
        cache: DocumentModelCache,
        allowed_run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[_ChapterAnalysis, ...]:
        ready: list[tuple[ChapterPlan, ChapterFrameSet]] = []
        immediate: dict[str, _ChapterAnalysis] = {}
        for chapter in chapters:
            frame_set = frame_sets[chapter.chapter_id]
            assert isinstance(frame_set, ChapterFrameSet)
            status = frame_status[chapter.chapter_id]
            skipped = _skipped_analysis(chapter, frame_set.candidates, status, document_config)
            if skipped is not None:
                immediate[chapter.chapter_id] = skipped
            else:
                ready.append((chapter, frame_set))
        completed = self._bounded_parallel_analysis(
            ready,
            transcripts,
            document_config,
            cache,
            allowed_run_root,
            is_cancel_requested,
        )
        by_id = {analysis.chapter.chapter_id: analysis for analysis in completed}
        by_id.update(immediate)
        return tuple(
            _inherit_frame_status(
                by_id[chapter.chapter_id],
                frame_status[chapter.chapter_id],
            )
            for chapter in chapters
        )

    def _bounded_parallel_analysis(
        self,
        ready: list[tuple[ChapterPlan, ChapterFrameSet]],
        transcripts: Mapping[str, tuple[TranscriptEvidence, ...]],
        document_config: DocumentGenerationConfig,
        cache: DocumentModelCache,
        allowed_run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[_ChapterAnalysis, ...]:
        results: dict[str, _ChapterAnalysis] = {}
        iterator = iter(ready)
        executor = ThreadPoolExecutor(max_workers=self._concurrency)
        pending: dict[Future[_ChapterAnalysis], str] = {}
        try:
            for _ in range(min(self._concurrency, len(ready))):
                _raise_if_cancelled(is_cancel_requested)
                chapter, frame_set = next(iterator)
                assert isinstance(frame_set, ChapterFrameSet)
                future = executor.submit(
                    self._analyze_chapter,
                    chapter,
                    frame_set.candidates,
                    transcripts[chapter.chapter_id],
                    document_config,
                    cache,
                    allowed_run_root,
                    is_cancel_requested,
                )
                pending[future] = chapter.chapter_id
            while pending:
                _raise_if_cancelled(is_cancel_requested)
                done, _ = wait(
                    tuple(pending),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in done:
                    _raise_if_cancelled(is_cancel_requested)
                    chapter_id = pending.pop(future)
                    results[chapter_id] = future.result()
                    try:
                        chapter, frame_set = next(iterator)
                    except StopIteration:
                        continue
                    _raise_if_cancelled(is_cancel_requested)
                    assert isinstance(frame_set, ChapterFrameSet)
                    next_future = executor.submit(
                        self._analyze_chapter,
                        chapter,
                        frame_set.candidates,
                        transcripts[chapter.chapter_id],
                        document_config,
                        cache,
                        allowed_run_root,
                        is_cancel_requested,
                    )
                    pending[next_future] = chapter.chapter_id
        except BaseException as error:
            for future in pending:
                future.cancel()
            _done, unfinished = wait(
                tuple(pending),
                timeout=self._invocation_wait_timeout_seconds,
            )
            if unfinished:
                executor.shutdown(wait=False, cancel_futures=True)
                raise _DeferredExecutorCleanup(error, executor) from error
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return tuple(results[chapter.chapter_id] for chapter, _frame_set in ready)

    def _analyze_chapter(
        self,
        chapter: ChapterPlan,
        frames: tuple[FrameCandidateArtifact, ...],
        transcript_evidence: tuple[TranscriptEvidence, ...],
        document_config: DocumentGenerationConfig,
        cache: DocumentModelCache,
        allowed_run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> _ChapterAnalysis:
        counters = _ChapterCounters(logical_analyses=1)
        request_frames = self._admit_request_frames(
            chapter,
            frames,
            transcript_evidence,
            document_config,
        )
        if request_frames is None:
            counters.fallback = 1
            return _ChapterAnalysis(
                chapter,
                (),
                None,
                "DEGRADED",
                f"CHAPTER_VLM_INPUT_BUDGET_DEGRADED:{chapter.chapter_id}",
                counters,
            )
        request = _vision_request(chapter, request_frames, transcript_evidence, document_config)
        canonical_input = _cache_input(
            chapter,
            request_frames,
            transcript_evidence,
            document_config,
        )

        def validate(response: ChapterVisionResponse) -> None:
            _validate_response(response, request, chapter)

        cached = cache.get(self._identity, canonical_input, ChapterVisionResponse, validate)
        if cached is not None:
            counters.cache_hits = 1
            response = cached.response
        else:
            logical_response = self._logical_call(
                cache,
                canonical_input,
                request,
                validate,
                counters,
                allowed_run_root,
                is_cancel_requested,
            )
            if logical_response is None:
                counters.fallback = 1
                return _ChapterAnalysis(
                    chapter,
                    request_frames,
                    None,
                    "DEGRADED",
                    f"CHAPTER_VLM_DEGRADED:{chapter.chapter_id}",
                    counters,
                )
            response = logical_response
        if not response.observations:
            counters.no_value = 1
            return _ChapterAnalysis(chapter, request_frames, response, "NO_VALUE", None, counters)
        return _ChapterAnalysis(chapter, request_frames, response, "SUCCEEDED", None, counters)

    def _logical_call(
        self,
        cache: DocumentModelCache,
        canonical_input: ChapterVisionCacheInput,
        request: ChapterVisionRequest,
        validate: Callable[[ChapterVisionResponse], None],
        counters: _ChapterCounters,
        allowed_run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterVisionResponse | None:
        with cache.invocation_lock(
            self._identity,
            canonical_input,
            wait_timeout_seconds=self._invocation_wait_timeout_seconds,
            is_cancel_requested=is_cancel_requested,
        ):
            cached = cache.get(self._identity, canonical_input, ChapterVisionResponse, validate)
            if cached is not None:
                counters.cache_hits = 1
                return cached.response
            try:
                response = self._vision_port.analyze_chapter(
                    request,
                    allowed_run_root=allowed_run_root,
                    on_provider_attempt=counters.provider_attempt,
                )
            except ModelResponseValidationError as error:
                invalid = error.invalid_response
                _log_validation_failure(
                    chapter_id=request.chapter_id,
                    phase="main",
                    code=error.code,
                    invalid=invalid,
                    provider_attempts=counters.provider_attempts,
                )
            except VideoDemoError as error:
                if error.code in _FALLBACK_CODES:
                    return None
                raise
            else:
                try:
                    validate(response)
                except _VisionSemanticValidationError as error:
                    invalid = _invalid_semantic_response(response, error)
                    _log_validation_failure(
                        chapter_id=request.chapter_id,
                        phase="main",
                        code=ErrorCode.QWEN_RESPONSE_INVALID,
                        invalid=invalid,
                        provider_attempts=counters.provider_attempts,
                    )
                else:
                    successful_path: Literal["MAIN", "REPAIR"] = "MAIN"
                    return cache.put(
                        self._identity,
                        canonical_input,
                        response,
                        successful_path=successful_path,
                        validate=validate,
                    ).response

            if invalid is not None:
                counters.structure_repairs = 1
                try:
                    response = self._vision_port.repair_chapter(
                        _repair_request(request, invalid),
                        allowed_run_root=allowed_run_root,
                        on_provider_attempt=counters.provider_attempt,
                    )
                    validate(response)
                except ModelResponseValidationError as error:
                    _log_validation_failure(
                        chapter_id=request.chapter_id,
                        phase="repair",
                        code=error.code,
                        invalid=error.invalid_response,
                        provider_attempts=counters.provider_attempts,
                    )
                    return None
                except _VisionSemanticValidationError as error:
                    _log_validation_failure(
                        chapter_id=request.chapter_id,
                        phase="repair",
                        code=ErrorCode.QWEN_RESPONSE_INVALID,
                        invalid=_invalid_semantic_response(response, error),
                        provider_attempts=counters.provider_attempts,
                    )
                    return None
                except VideoDemoError as error:
                    if error.code in _FALLBACK_CODES:
                        return None
                    raise
                successful_path = "REPAIR"
            return cache.put(
                self._identity,
                canonical_input,
                response,
                successful_path=successful_path,
                validate=validate,
            ).response

    def _admit_request_frames(
        self,
        chapter: ChapterPlan,
        frames: tuple[FrameCandidateArtifact, ...],
        transcript_evidence: tuple[TranscriptEvidence, ...],
        document_config: DocumentGenerationConfig,
    ) -> tuple[FrameCandidateArtifact, ...] | None:
        admitted = tuple(frame for frame in frames if frame.size_bytes <= self._max_image_bytes)
        required_targets = {target.target_id for target in _targets(chapter)}
        max_selected_frames = min(
            document_config.max_visuals_per_chapter,
            3 if chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
        )
        if max_selected_frames == 0:
            return None
        while admitted:
            request = _vision_request(chapter, admitted, transcript_evidence, document_config)
            if (
                len(admitted) <= max_selected_frames
                and sum(frame.size_bytes for frame in admitted) <= self._max_request_image_bytes
                and self._request_pair_fits(request)
                and _covers_targets(admitted, required_targets)
            ):
                return admitted
            removable = _lowest_removable_frame(admitted, required_targets)
            if removable is None:
                return None
            admitted = tuple(frame for frame in admitted if frame.frame_id != removable.frame_id)
        return None

    def _request_pair_fits(self, request: ChapterVisionRequest) -> bool:
        frames = tuple(
            (frame.frame_id, frame.size_bytes)
            for frame in sorted(request.frames, key=lambda item: (item.timestamp_ms, item.frame_id))
        )
        main_size = vision_payload_size_upper_bound(
            prompt_for_vision(request),
            model_id=self._identity.model_id,
            schema_name=self._identity.main_response_schema_name,
            response_schema=ChapterVisionResponse.model_json_schema(),
            ordered_frames=frames,
        )
        repair = _repair_request(request, _worst_invalid_response())
        repair_size = vision_payload_size_upper_bound(
            prompt_for_vision_repair(repair),
            model_id=self._identity.model_id,
            schema_name=self._identity.repair_response_schema_name,
            response_schema=ChapterVisionResponse.model_json_schema(),
            ordered_frames=frames,
        )
        return max(main_size, repair_size) <= self._max_encoded_request_bytes

    def _assemble_batch(
        self,
        analyses: tuple[_ChapterAnalysis, ...],
        frame_batch: ChapterFrameSearchBatch,
        allowed_run_root: Path,
        *,
        frame_tolerance_ms: int,
    ) -> ChapterVisionBatch:
        observations = _collect_materialized_observations(analyses)
        artifacts = KeyframeArtifactSession(
            allowed_run_root,
            max_files=self._max_published_keyframe_files,
            max_bytes=self._max_published_keyframe_bytes,
        )
        with artifacts:
            existing = artifacts.snapshot()
            retained, budget_chapters = _apply_published_budget(
                observations,
                existing,
                max_files=self._max_published_keyframe_files,
                max_bytes=self._max_published_keyframe_bytes,
                new_artifact_multiplier=artifacts.new_artifact_budget_multiplier,
            )
            frames = _unique_frames(retained)
            batch = _materialize_batch(
                analyses,
                frame_batch,
                retained,
                frames,
                budget_chapters,
                frame_tolerance_ms=frame_tolerance_ms,
            )
            artifacts.publish(frames, existing)
            return batch


def _validate_batch_alignment(
    chapters: tuple[ChapterPlan, ...],
    frame_batch: ChapterFrameSearchBatch,
) -> tuple[dict[str, ChapterFrameSet], dict[str, ChapterFrameStatus]]:
    chapter_ids = tuple(chapter.chapter_id for chapter in chapters)
    frame_ids = tuple(frame_set.chapter_id for frame_set in frame_batch.frame_sets)
    status_ids = tuple(chapter_id for chapter_id, _status in frame_batch.chapter_status)
    if not chapters or chapter_ids != frame_ids or chapter_ids != status_ids:
        raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节与抽帧批次未有序对齐")
    return (
        {frame_set.chapter_id: frame_set for frame_set in frame_batch.frame_sets},
        dict(frame_batch.chapter_status),
    )


def _validate_frame_batch_semantics(
    chapters: tuple[ChapterPlan, ...],
    frame_sets: Mapping[str, ChapterFrameSet],
    frame_status: Mapping[str, ChapterFrameStatus],
    *,
    asset_sha256: str,
    allowed_run_root: Path,
) -> None:
    frame_metadata: dict[str, tuple[object, ...]] = {}
    for chapter in chapters:
        chapter_id = chapter.chapter_id
        frame_set = frame_sets[chapter_id]
        has_candidates = bool(frame_set.candidates)
        status = frame_status[chapter_id]
        if (
            (status == "SUCCEEDED" and not has_candidates)
            or (status in {"NO_CANDIDATE", "DISABLED"} and has_candidates)
        ):
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "抽帧章节状态与候选帧集合矛盾",
            )
        frame_ids = tuple(frame.frame_id for frame in frame_set.candidates)
        target_ids = {target.target_id for target in _targets(chapter)}
        if len(frame_ids) != len(set(frame_ids)):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "章节候选帧 ID 重复")
        for frame in frame_set.candidates:
            validate_candidate_descriptor(
                frame,
                asset_sha256,
                allowed_run_root=allowed_run_root,
            )
            if (
                not chapter.start_ms <= frame.timestamp_ms < chapter.end_ms
                or not set(frame.target_ids) <= target_ids
            ):
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "章节候选帧时间或目标绑定非法",
                )
            metadata = (
                frame.timestamp_ms,
                frame.sha256,
                frame.size_bytes,
                frame.relative_path,
                frame.mime_type,
                frame.perceptual_hash,
                frame.target_ids,
            )
            if frame.frame_id in frame_metadata and frame_metadata[frame.frame_id] != metadata:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "同 frame ID 候选帧元数据不一致",
                )
            frame_metadata[frame.frame_id] = metadata


def _verify_candidate_batch_files(
    frame_sets: Mapping[str, ChapterFrameSet],
    asset_sha256: str,
    allowed_run_root: Path,
    *,
    max_bytes: int,
) -> None:
    """在候选租约内按唯一 SHA 完成一次文件级完整性复验。"""

    seen: set[str] = set()
    for frame_set in frame_sets.values():
        for frame in frame_set.candidates:
            if frame.sha256 in seen:
                continue
            validate_candidate_descriptor(
                frame,
                asset_sha256,
                allowed_run_root=allowed_run_root,
            )
            read_verified_candidate_jpeg(
                allowed_run_root,
                frame,
                max_bytes=max_bytes,
            )
            seen.add(frame.sha256)


def _validate_transcripts(
    chapters: tuple[ChapterPlan, ...],
    transcript_evidence: tuple[TranscriptEvidence, ...],
) -> dict[str, tuple[TranscriptEvidence, ...]]:
    ordered = tuple(
        sorted(
            transcript_evidence,
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    ids = tuple(item.evidence_id for item in ordered)
    if transcript_evidence != ordered or len(ids) != len(set(ids)):
        raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "转写证据必须有序且唯一")
    return {
        chapter.chapter_id: tuple(
            item
            for item in ordered
            if chapter.start_ms <= item.start_ms and item.end_ms <= chapter.end_ms
        )
        for chapter in chapters
    }


def _skipped_analysis(
    chapter: ChapterPlan,
    frames: tuple[FrameCandidateArtifact, ...],
    frame_status: ChapterFrameStatus,
    config: DocumentGenerationConfig,
) -> _ChapterAnalysis | None:
    if config.max_visuals_per_chapter == 0 or frame_status == "DISABLED":
        return _ChapterAnalysis(
            chapter,
            (),
            None,
            "DISABLED",
            None,
            _ChapterCounters(logical_analyses=0),
        )
    if not frames:
        status: ChapterVisionStatus = "DEGRADED" if frame_status == "DEGRADED" else "NO_CANDIDATE"
        return _ChapterAnalysis(
            chapter,
            (),
            None,
            status,
            None,
            _ChapterCounters(logical_analyses=0),
        )
    return None


def _inherit_frame_status(
    analysis: _ChapterAnalysis,
    frame_status: ChapterFrameStatus,
) -> _ChapterAnalysis:
    if frame_status != "DEGRADED" or analysis.status == "DEGRADED":
        return analysis
    return _ChapterAnalysis(
        chapter=analysis.chapter,
        frames=analysis.frames,
        response=analysis.response,
        status="DEGRADED",
        warning=analysis.warning,
        counters=analysis.counters,
    )


def _empty_batch(
    chapters: tuple[ChapterPlan, ...],
    frame_status: Mapping[str, ChapterFrameStatus],
    batch_status: str,
) -> ChapterVisionBatch:
    statuses: list[tuple[str, ChapterVisionStatus]] = []
    for chapter in chapters:
        source = frame_status[chapter.chapter_id]
        mapped: ChapterVisionStatus = (
            "DEGRADED" if source == "DEGRADED" else source
        )
        statuses.append((chapter.chapter_id, mapped))
    return ChapterVisionBatch(
        observations=(),
        evidence=(),
        keyframe_evidence=(),
        chapter_status=tuple(statuses),
        status="PARTIAL_SUCCEEDED" if batch_status == "PARTIAL_SUCCEEDED" else "SUCCEEDED",
        metrics=_zero_metrics(),
    )


def _disabled_batch(chapters: tuple[ChapterPlan, ...]) -> ChapterVisionBatch:
    return ChapterVisionBatch(
        observations=(),
        evidence=(),
        keyframe_evidence=(),
        chapter_status=tuple((chapter.chapter_id, "DISABLED") for chapter in chapters),
        status="SUCCEEDED",
        metrics=_zero_metrics(),
    )


def _vision_request(
    chapter: ChapterPlan,
    frames: tuple[FrameCandidateArtifact, ...],
    transcript_evidence: tuple[TranscriptEvidence, ...],
    config: DocumentGenerationConfig,
) -> ChapterVisionRequest:
    return ChapterVisionRequest(
        chapter_id=chapter.chapter_id,
        targets=_targets(chapter),
        frames=tuple(sorted(frames, key=lambda item: (item.timestamp_ms, item.frame_id))),
        transcript_evidence=transcript_evidence,
        document_config=config,
        prompt_version="chapter-vlm-v1",
    )


def _cache_input(
    chapter: ChapterPlan,
    frames: tuple[FrameCandidateArtifact, ...],
    transcript_evidence: tuple[TranscriptEvidence, ...],
    config: DocumentGenerationConfig,
) -> ChapterVisionCacheInput:
    return ChapterVisionCacheInput(
        chapter=chapter,
        frames=tuple(
            ChapterVisionFrameCacheInput(
                frame_id=frame.frame_id,
                timestamp_ms=frame.timestamp_ms,
                sha256=frame.sha256,
                target_ids=frame.target_ids,
            )
            for frame in sorted(frames, key=lambda item: (item.timestamp_ms, item.frame_id))
        ),
        transcript_evidence=transcript_evidence,
        document_config=config,
    )


def _targets(chapter: ChapterPlan):  # type: ignore[no-untyped-def]
    return (*chapter.semantic_targets, *chapter.base_coverage_targets)


def _repair_request(
    request: ChapterVisionRequest,
    invalid: InvalidModelResponse,
) -> ChapterVisionRepairRequest:
    return ChapterVisionRepairRequest(
        request=request,
        invalid_response=invalid,
        allowed_frame_ids=tuple(frame.frame_id for frame in request.frames),
        allowed_target_ids=tuple(target.target_id for target in request.targets),
        allowed_transcript_evidence_ids=tuple(
            item.evidence_id for item in request.transcript_evidence
        ),
        prompt_version="chapter-vlm-repair-v1",
    )


def _worst_invalid_response() -> InvalidModelResponse:
    errors = tuple(f"{index:02d}:" + "." * 497 for index in range(32))
    return InvalidModelResponse(
        content_sha256="f" * 64,
        validation_errors=errors,
        safe_json_excerpt="." * 8_000,
    )


def _validate_response(
    response: ChapterVisionResponse,
    request: ChapterVisionRequest,
    chapter: ChapterPlan,
) -> None:
    cap = min(
        request.document_config.max_visuals_per_chapter,
        3 if chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2,
    )
    try:
        validate_chapter_vision_response(
            response,
            request,
            max_selected_frames=cap,
        )
    except ValueError as error:
        raise _VisionSemanticValidationError(str(error)) from None


class _VisionSemanticValidationError(ValueError):
    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def _log_validation_failure(
    *,
    chapter_id: StableId,
    phase: Literal["main", "repair"],
    code: ErrorCode,
    invalid: InvalidModelResponse,
    provider_attempts: int,
) -> None:
    _LOGGER.warning(
        "章节视觉响应校验失败 chapter_id=%s phase=%s code=%s "
        "provider_attempts=%d validation_errors=%s",
        chapter_id,
        phase,
        code,
        provider_attempts,
        ",".join(invalid.validation_errors[:8]),
    )


def _invalid_semantic_response(
    response: ChapterVisionResponse,
    error: _VisionSemanticValidationError,
) -> InvalidModelResponse:
    parsed = response.model_dump(mode="json")
    return invalid_model_response(
        response.model_dump_json().encode("utf-8"),
        (error.summary,),
        parsed_json=parsed,
    )


def _covers_targets(
    frames: tuple[FrameCandidateArtifact, ...],
    required_targets: set[str],
) -> bool:
    covered = {target_id for frame in frames for target_id in frame.target_ids}
    return required_targets <= covered


def _lowest_removable_frame(
    frames: tuple[FrameCandidateArtifact, ...],
    required_targets: set[str],
) -> FrameCandidateArtifact | None:
    for frame in reversed(frames):
        remaining = tuple(item for item in frames if item.frame_id != frame.frame_id)
        if _covers_targets(remaining, required_targets):
            return frame
    return None


def _collect_materialized_observations(
    analyses: tuple[_ChapterAnalysis, ...],
) -> tuple[_MaterializedObservation, ...]:
    collected: list[_MaterializedObservation] = []
    for analysis in analyses:
        if analysis.response is None:
            continue
        frame_by_id = {frame.frame_id: frame for frame in analysis.frames}
        for draft in analysis.response.observations:
            collected.append(
                _MaterializedObservation(
                    analysis.chapter,
                    draft,
                    tuple(frame_by_id[frame_id] for frame_id in draft.selected_frame_ids),
                ),
            )
    return tuple(collected)


def _apply_published_budget(
    observations: tuple[_MaterializedObservation, ...],
    existing: Mapping[str, int],
    *,
    max_files: int,
    max_bytes: int,
    new_artifact_multiplier: int,
) -> tuple[tuple[_MaterializedObservation, ...], tuple[str, ...]]:
    retained = list(observations)
    removed_chapters: list[str] = []
    while not _published_closure_fits(
        retained,
        existing,
        max_files=max_files,
        max_bytes=max_bytes,
        new_artifact_multiplier=new_artifact_multiplier,
    ):
        if not retained:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "既有关键帧超过运行时预算")
        removed = min(retained, key=_observation_value)
        retained.remove(removed)
        if removed.chapter.chapter_id not in removed_chapters:
            removed_chapters.append(removed.chapter.chapter_id)
    return tuple(retained), tuple(removed_chapters)


def _published_closure_fits(
    observations: Iterable[_MaterializedObservation],
    existing: Mapping[str, int],
    *,
    max_files: int,
    max_bytes: int,
    new_artifact_multiplier: int,
) -> bool:
    new = {
        frame.sha256: frame.size_bytes
        for observation in observations
        for frame in observation.frames
        if frame.sha256 not in existing
    }
    return (
        len(existing) + len(new) * new_artifact_multiplier <= max_files
        and sum(existing.values()) + sum(new.values()) * new_artifact_multiplier <= max_bytes
    )


def _observation_value(item: _MaterializedObservation) -> tuple[int, int, float, str]:
    complex_changes = sum(len(relation.changes) for relation in item.draft.frame_relations)
    stable = stable_identifier("visual_observation_draft", item.draft.model_dump(mode="json"))
    return (
        _RELATION_VALUE[item.draft.relation_to_transcript],
        complex_changes,
        item.draft.certainty,
        stable,
    )


def _unique_frames(
    observations: tuple[_MaterializedObservation, ...],
) -> tuple[FrameCandidateArtifact, ...]:
    by_id: dict[str, FrameCandidateArtifact] = {}
    for observation in observations:
        for frame in observation.frames:
            by_id.setdefault(frame.frame_id, frame)
    return tuple(sorted(by_id.values(), key=lambda item: (item.timestamp_ms, item.frame_id)))


def _materialize_batch(
    analyses: tuple[_ChapterAnalysis, ...],
    frame_batch: ChapterFrameSearchBatch,
    retained: tuple[_MaterializedObservation, ...],
    frames: tuple[FrameCandidateArtifact, ...],
    budget_chapters: tuple[str, ...],
    *,
    frame_tolerance_ms: int,
) -> ChapterVisionBatch:
    keyframes = tuple(_keyframe_evidence(frame) for frame in frames)
    keyframe_by_frame = {item.keyframe_id: item for item in keyframes}
    evidence_observations = tuple(
        _to_evidence(item, keyframe_by_frame, frame_tolerance_ms=frame_tolerance_ms)
        for item in retained
    )
    used_ids = {ref for item in evidence_observations for ref in item.keyframe_refs}
    used_keyframes = tuple(
        sorted(
            (item for item in keyframes if item.evidence_id in used_ids),
            key=lambda item: (item.timestamp_ms, item.evidence_id),
        ),
    )
    ordered_observations = tuple(
        sorted(
            evidence_observations,
            key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
        ),
    )
    status_by_id = {analysis.chapter.chapter_id: analysis.status for analysis in analyses}
    warnings = [analysis.warning for analysis in analyses if analysis.warning is not None]
    for chapter_id in budget_chapters:
        status_by_id[chapter_id] = "DEGRADED"
        warnings.append(f"VISUAL_PUBLISHED_BUDGET_DEGRADED:{chapter_id}")
    chapter_status = tuple(
        (analysis.chapter.chapter_id, status_by_id[analysis.chapter.chapter_id])
        for analysis in analyses
    )
    partial = frame_batch.status == "PARTIAL_SUCCEEDED" or any(
        status == "DEGRADED" for _chapter_id, status in chapter_status
    )
    return ChapterVisionBatch(
        observations=ordered_observations,
        keyframe_evidence=used_keyframes,
        evidence=(*used_keyframes, *ordered_observations),
        chapter_status=chapter_status,
        warnings=tuple(warnings),
        status="PARTIAL_SUCCEEDED" if partial else "SUCCEEDED",
        metrics=_merge_metrics(analyses, len(budget_chapters)),
    )


def _keyframe_evidence(frame: FrameCandidateArtifact) -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id=stable_identifier("keyframe_evidence", {"keyframe_id": frame.frame_id}),
        keyframe_id=frame.frame_id,
        start_ms=frame.timestamp_ms,
        end_ms=frame.timestamp_ms + 1,
        timestamp_ms=frame.timestamp_ms,
        relative_path=f"visual/keyframes/{frame.sha256}.jpg",
        mime_type="image/jpeg",
        sha256=frame.sha256,
        perceptual_hash=frame.perceptual_hash,
        size_bytes=frame.size_bytes,
    )


def _to_evidence(
    item: _MaterializedObservation,
    keyframe_by_frame: Mapping[str, KeyframeEvidence],
    *,
    frame_tolerance_ms: int,
) -> VisualObservationEvidence:
    keyframes = tuple(keyframe_by_frame[frame.frame_id] for frame in item.frames)
    draft = item.draft
    frame_to_evidence = {
        frame.keyframe_id: frame.evidence_id
        for frame in keyframes
    }
    start_ms = max(
        item.chapter.start_ms,
        min(frame.timestamp_ms for frame in keyframes) - frame_tolerance_ms,
    )
    end_ms = min(
        item.chapter.end_ms,
        max(frame.timestamp_ms for frame in keyframes) + frame_tolerance_ms + 1,
    )
    payload: dict[str, object] = {
        "chapter_id": item.chapter.chapter_id,
        "target_ids": draft.target_ids,
        "keyframe_refs": tuple(frame.evidence_id for frame in keyframes),
        "transcript_evidence_refs": draft.transcript_evidence_refs,
        "draft": draft.model_dump(mode="json"),
    }
    observation_id = stable_identifier("visual_observation", payload)
    return VisualObservationEvidence(
        evidence_id=observation_id,
        chapter_id=item.chapter.chapter_id,
        start_ms=start_ms,
        end_ms=end_ms,
        target_ids=draft.target_ids,
        keyframe_refs=tuple(frame.evidence_id for frame in keyframes),
        transcript_evidence_refs=draft.transcript_evidence_refs,
        visual_type=draft.visual_type,
        caption=draft.caption,
        content_blocks=tuple(
            _content_block(block, frame_to_evidence, observation_id, ordinal)
            for ordinal, block in enumerate(draft.content_blocks)
        ),
        visual_facts=tuple(
            _visual_fact(fact, frame_to_evidence, observation_id, ordinal)
            for ordinal, fact in enumerate(draft.visual_facts)
        ),
        frame_relations=tuple(
            _frame_relation(relation, frame_to_evidence) for relation in draft.frame_relations
        ),
        relation_to_transcript=draft.relation_to_transcript,
        certainty=draft.certainty,
    )


def _content_block(  # type: ignore[no-untyped-def]
    block: object,
    refs: Mapping[str, str],
    observation_id: str,
    ordinal: int,
):
    source = tuple(refs[frame_id] for frame_id in block.source_frame_ids)  # type: ignore[attr-defined]
    content_id = stable_identifier(
        "visual_content",
        {
            "visual_observation_id": observation_id,
            "content_type": block.content_type,  # type: ignore[attr-defined]
            "ordinal": ordinal,
        },
    )
    if isinstance(block, VisualTextContentDraft):
        return VisualTextContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            text=block.text,
        )
    if isinstance(block, VisualCodeContentDraft):
        return VisualCodeContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            language=block.language,
            code=block.code,
        )
    if isinstance(block, VisualTableContentDraft):
        return VisualTableContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            columns=block.columns,
            rows=block.rows,
        )
    if isinstance(block, VisualFormulaContentDraft):
        return VisualFormulaContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            latex=block.latex,
            explanation=block.explanation,
        )
    if isinstance(block, VisualDiagramContentDraft):
        return VisualDiagramContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            description=block.description,
            labels=block.labels,
            relations=block.relations,
        )
    if isinstance(block, VisualStateContentDraft):
        return VisualStateContent(
            visual_content_id=content_id,
            source_keyframe_refs=source,
            state_type=block.state_type,
            description=block.description,
            key_values=block.key_values,
        )
    raise TypeError("未知视觉内容草稿")


def _visual_fact(
    fact: VisualFactDraft,
    refs: Mapping[str, str],
    observation_id: str,
    ordinal: int,
) -> GroundedVisualFact:
    return GroundedVisualFact(
        visual_fact_id=stable_identifier(
            "visual_fact",
            {
                "visual_observation_id": observation_id,
                "ordinal": ordinal,
            },
        ),
        text=fact.text,
        source_keyframe_refs=tuple(refs[frame_id] for frame_id in fact.source_frame_ids),
    )


def _frame_relation(
    relation: VisualFrameRelationDraft,
    refs: Mapping[str, str],
) -> VisualFrameRelation:
    return VisualFrameRelation(
        relation_type=relation.relation_type,
        from_keyframe_ref=refs[relation.from_frame_id],
        to_keyframe_ref=refs[relation.to_frame_id],
        description=relation.description,
        changes=tuple(
            VisualFieldChange(field=item.field, before=item.before, after=item.after)
            for item in relation.changes
        ),
    )


def _merge_metrics(
    analyses: tuple[_ChapterAnalysis, ...],
    published_budget_chapters: int,
) -> dict[str, int]:
    return {
        "vlm_logical_analyses": sum(item.counters.logical_analyses for item in analyses),
        "vlm_provider_attempts": sum(item.counters.provider_attempts for item in analyses),
        "vlm_structure_repairs": sum(item.counters.structure_repairs for item in analyses),
        "vlm_cache_hits": sum(item.counters.cache_hits for item in analyses),
        "vlm_no_value_chapters": sum(item.counters.no_value for item in analyses),
        "vlm_fallback_chapters": sum(item.counters.fallback for item in analyses),
        "visual_published_budget_degraded_chapters": published_budget_chapters,
    }


def _zero_metrics() -> dict[str, int]:
    return {name: 0 for name in _METRIC_NAMES}


def _raise_if_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
    if is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")
