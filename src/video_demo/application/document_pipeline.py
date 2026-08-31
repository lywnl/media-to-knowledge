from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from video_demo.application.base_segments import (
    build_base_segments,
)
from video_demo.application.chapter_frames import ChapterFrameSearchBatch
from video_demo.application.chapter_planning import ChapterPlanningBatch
from video_demo.application.chapter_vision import ChapterVisionBatch
from video_demo.application.document_rendering import RenderedDocument, render_markdown
from video_demo.application.document_writing import WrittenDocument
from video_demo.application.pipeline_contracts import (
    DocumentWritingContext,
    EvidencePreparationLimits,
    PipelineContext,
    PipelineOutcome,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SpeechAnalysis,
    StageMetric,
    TranscriptionCheckpoint,
    merge_model_metrics,
    merge_run_statuses,
    require_result_evidence_budget,
    stable_merge_document_evidence,
    stable_merge_warnings,
)
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_artifact import (
    MAX_METRIC_VALUE,
    RESULT_STAGE_NAMES,
)
from video_demo.domain.document_plan import BaseSegment, ChapterPlan
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.document_cache import DocumentModelCache
from video_demo.storage.workspace import safe_runtime_path

StageResult = TypeVar("StageResult")
_LOGGER = logging.getLogger(__name__)


class AssetRegistrar(Protocol):
    def register(self, context: PipelineContext) -> RegisteredAsset: ...


class AssetProbe(Protocol):
    def probe(self, asset: RegisteredAsset) -> ProbedAsset: ...


class MediaTranscoder(Protocol):
    def transcode(
        self,
        probed: ProbedAsset,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> PreparedMedia: ...


class SpeechAnalyzer(Protocol):
    def analyze(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis: ...


class ChapterPlannerPort(Protocol):
    def plan(
        self,
        *,
        cache: DocumentModelCache,
        asset_sha256: str,
        title_hint: str,
        duration_ms: int,
        segments: tuple[BaseSegment, ...],
        transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
        document_config: DocumentGenerationConfig,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterPlanningBatch: ...


class ChapterFrameSearcherPort(Protocol):
    def search(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        document_config: DocumentGenerationConfig,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterFrameSearchBatch: ...


class ChapterVisionPort(Protocol):
    def analyze_all(
        self,
        chapters: tuple[ChapterPlan, ...],
        frame_batch: ChapterFrameSearchBatch,
        transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
        document_config: DocumentGenerationConfig,
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterVisionBatch: ...


class DocumentWriterPort(Protocol):
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
    ) -> WrittenDocument: ...


class _StageMetrics:
    def __init__(self) -> None:
        self._values = dict.fromkeys(RESULT_STAGE_NAMES, 0)

    def add(self, stage: str, duration_ms: int) -> None:
        if stage not in RESULT_STAGE_NAMES:
            raise ValueError("阶段指标包含未知白名单键")
        if type(duration_ms) is not int or duration_ms < 0:
            raise ValueError("阶段指标必须是非负严格整数")
        total = self._values[stage] + duration_ms
        if total > MAX_METRIC_VALUE:
            raise ValueError("阶段指标累计值溢出 2^63-1")
        self._values[stage] = total

    def add_many(self, metrics: tuple[StageMetric, ...]) -> None:
        for metric in metrics:
            self.add(metric.stage, metric.duration_ms)

    def complete(self) -> dict[str, int]:
        return dict(self._values)


class VideoUnderstandingPipeline:
    """从注册到确定性 Markdown 的唯一 4.1 生产编排。"""

    def __init__(
        self,
        registrar: AssetRegistrar,
        probe: AssetProbe,
        transcoder: MediaTranscoder,
        speech_analyzer: SpeechAnalyzer,
        chapter_planner: ChapterPlannerPort,
        frame_searcher: ChapterFrameSearcherPort,
        chapter_vision: ChapterVisionPort,
        document_writer: DocumentWriterPort,
        model_cache_factory: Callable[[Path], DocumentModelCache],
        *,
        runtime_root: Path,
        evidence_preparation_limits: EvidencePreparationLimits,
        max_result_evidence_items: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(max_result_evidence_items) is not int or max_result_evidence_items < 1:
            raise ValueError("max_result_evidence_items 必须是正整数")
        self._registrar = registrar
        self._probe = probe
        self._transcoder = transcoder
        self._speech_analyzer = speech_analyzer
        self._chapter_planner = chapter_planner
        self._frame_searcher = frame_searcher
        self._chapter_vision = chapter_vision
        self._document_writer = document_writer
        self._model_cache_factory = model_cache_factory
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._evidence_preparation_limits = evidence_preparation_limits
        self._max_result_evidence_items = max_result_evidence_items
        self._clock = clock

    @property
    def registrar(self) -> AssetRegistrar:
        return self._registrar

    @property
    def probe(self) -> AssetProbe:
        return self._probe

    @property
    def transcoder(self) -> MediaTranscoder:
        return self._transcoder

    @property
    def speech_analyzer(self) -> SpeechAnalyzer:
        return self._speech_analyzer

    @property
    def chapter_planner(self) -> ChapterPlannerPort:
        return self._chapter_planner

    @property
    def frame_searcher(self) -> ChapterFrameSearcherPort:
        return self._frame_searcher

    @property
    def chapter_vision(self) -> ChapterVisionPort:
        return self._chapter_vision

    @property
    def document_writer(self) -> DocumentWriterPort:
        return self._document_writer

    def run(self, context: PipelineContext) -> PipelineOutcome:
        checkpoint = self.run_transcription(context)
        return self.run_llm(context, checkpoint)

    def run_transcription(self, context: PipelineContext) -> TranscriptionCheckpoint:
        """执行媒体准备和转写；结果可独立持久化后交接给 LLM 阶段。"""

        stage_metrics = _StageMetrics()
        registered = self._run_stage(
            context,
            stage_metrics,
            "REGISTER",
            lambda: self._registrar.register(context),
        )
        model_cache = self._new_run_cache(registered)
        probed = self._run_stage(
            context,
            stage_metrics,
            "PROBE",
            lambda: self._probe.probe(registered),
        )
        prepared = self._run_stage(
            context,
            stage_metrics,
            "TRANSCODE",
            lambda: self._transcoder.transcode(
                probed,
                is_cancel_requested=context.is_cancel_requested,
            ),
        )
        speech, base_segments = self._run_evidence_preparation(
            context,
            registered,
            prepared,
            stage_metrics,
        )
        return TranscriptionCheckpoint(
            registered=registered,
            prepared=prepared,
            speech=speech,
            base_segments=base_segments,
            stage_metrics=stage_metrics.complete(),
            stage_cache_hits=speech.stage_cache_hits,
            model_cache=model_cache,
        )

    def run_llm(
        self,
        context: PipelineContext,
        checkpoint: TranscriptionCheckpoint,
    ) -> PipelineOutcome:
        """从转写快照继续执行章节规划、视觉补充和文档发布前编排。"""

        stage_metrics = _StageMetrics()
        for stage, duration_ms in checkpoint.stage_metrics.items():
            stage_metrics.add(stage, duration_ms)
        registered = checkpoint.registered
        prepared = checkpoint.prepared
        speech = checkpoint.speech
        base_segments = checkpoint.base_segments
        model_cache = checkpoint.model_cache or self._new_run_cache(registered)
        planning_batch = self._run_chapter_planning(
            context,
            stage_metrics,
            model_cache,
            registered,
            prepared,
            speech,
            base_segments,
        )
        chapter_plans = planning_batch.plans
        frame_batch = self._run_stage(
            context,
            stage_metrics,
            "FRAME_SEARCH",
            lambda: self._frame_searcher.search(
                prepared,
                chapter_plans,
                speech.transcript_by_id,
                context.document_config,
                is_cancel_requested=context.is_cancel_requested,
            ),
        )
        visual_batch = self._run_stage(
            context,
            stage_metrics,
            "VISUAL_EVIDENCE",
            lambda: self._chapter_vision.analyze_all(
                chapter_plans,
                frame_batch,
                speech.transcript_evidence,
                context.document_config,
                cache=model_cache,
                is_cancel_requested=context.is_cancel_requested,
            ),
        )
        evidence = stable_merge_document_evidence(
            speech.transcript_evidence,
            visual_batch.evidence,
        )
        require_result_evidence_budget(evidence, self._max_result_evidence_items)
        written = self._run_document_writing(
            context,
            stage_metrics,
            model_cache,
            registered,
            prepared,
            speech,
            chapter_plans,
            visual_batch,
        )
        document = self._run_stage(
            context,
            stage_metrics,
            "DOCUMENT_ASSEMBLY",
            lambda: render_markdown(written.result, evidence),
        )
        return self._run_result_stage(
            context,
            stage_metrics,
            speech,
            planning_batch,
            frame_batch,
            visual_batch,
            written,
            evidence,
            document,
        )

    def _run_chapter_planning(
        self,
        context: PipelineContext,
        stage_metrics: _StageMetrics,
        model_cache: DocumentModelCache,
        registered: RegisteredAsset,
        prepared: PreparedMedia,
        speech: SpeechAnalysis,
        base_segments: tuple[BaseSegment, ...],
    ) -> ChapterPlanningBatch:
        return self._run_stage(
            context,
            stage_metrics,
            "CHAPTER_PLAN",
            lambda: self._chapter_planner.plan(
                cache=model_cache,
                asset_sha256=registered.source_sha256,
                title_hint=context.title_hint,
                duration_ms=prepared.source.duration_ms,
                segments=base_segments,
                transcript_evidence=speech.transcript_evidence,
                document_config=context.document_config,
                is_cancel_requested=context.is_cancel_requested,
            ),
        )

    def _run_document_writing(
        self,
        context: PipelineContext,
        stage_metrics: _StageMetrics,
        model_cache: DocumentModelCache,
        registered: RegisteredAsset,
        prepared: PreparedMedia,
        speech: SpeechAnalysis,
        chapter_plans: tuple[ChapterPlan, ...],
        visual_batch: ChapterVisionBatch,
    ) -> WrittenDocument:
        writing_context = DocumentWritingContext(
            run_id=context.run_id,
            asset_sha256=registered.source_sha256,
            title_hint=context.title_hint,
            duration_ms=prepared.source.duration_ms,
            transcript_source=speech.transcript_source,
            document_config=context.document_config,
        )
        return self._run_stage(
            context,
            stage_metrics,
            "CHAPTER_WRITE",
            lambda: self._document_writer.write(
                writing_context,
                chapter_plans,
                speech.transcript_evidence,
                visual_batch.observations,
                visual_batch.keyframe_evidence,
                cache=model_cache,
                is_cancel_requested=context.is_cancel_requested,
            ),
        )

    def _run_evidence_preparation(
        self,
        context: PipelineContext,
        registered: RegisteredAsset,
        prepared: PreparedMedia,
        stage_metrics: _StageMetrics,
    ) -> tuple[SpeechAnalysis, tuple[BaseSegment, ...]]:
        self._start_stage(context, "EVIDENCE_PREP")
        started_at = self._clock()
        speech, speech_duration = self._timed_speech(prepared, context.is_cancel_requested)
        self._check_cancelled(context)
        base_segments = build_base_segments(
            registered.source_sha256,
            prepared.source.duration_ms,
            speech.transcript_evidence,
            speech.boundary_candidates,
            self._evidence_preparation_limits,
        )
        self._check_cancelled(context)
        stage_metrics.add("SPEECH", speech_duration)
        stage_metrics.add_many(speech.stage_metrics)
        stage_metrics.add("EVIDENCE_PREP", self._elapsed_ms(started_at))
        return speech, base_segments

    def _timed_speech(
        self,
        prepared: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[SpeechAnalysis, int]:
        started_at = self._clock()
        result = self._speech_analyzer.analyze(
            prepared,
            is_cancel_requested=is_cancel_requested,
        )
        return result, self._elapsed_ms(started_at)

    def _run_result_stage(
        self,
        context: PipelineContext,
        stage_metrics: _StageMetrics,
        speech: SpeechAnalysis,
        planning_batch: ChapterPlanningBatch,
        frame_batch: ChapterFrameSearchBatch,
        visual_batch: ChapterVisionBatch,
        written: WrittenDocument,
        evidence: tuple[DocumentEvidenceItem, ...],
        document: RenderedDocument,
    ) -> PipelineOutcome:
        self._start_stage(context, "RESULT")
        started_at = self._clock()
        status = merge_run_statuses(
            planning_batch.status,
            visual_batch.status,
            written.status,
        )
        warnings = stable_merge_warnings(
            speech.warnings,
            planning_batch.warnings,
            frame_batch.warnings,
            visual_batch.warnings,
            written.warnings,
        )
        model_metrics = merge_model_metrics(
            planning_batch.metrics,
            frame_batch.metrics,
            visual_batch.metrics,
            written.metrics,
        )
        stage_metrics.add("RESULT", self._elapsed_ms(started_at))
        complete_stage_metrics = stage_metrics.complete()
        stage_cache_hits = self._validated_cache_hits(
            speech.stage_cache_hits,
            complete_stage_metrics,
        )
        return PipelineOutcome(
            status=status,
            result=written.result,
            evidence=evidence,
            document=document,
            warnings=warnings,
            stage_metrics=complete_stage_metrics,
            model_metrics=model_metrics,
            stage_cache_hits=stage_cache_hits,
            transcript_source=speech.transcript_source,
            frame_batch=frame_batch,
            visual_batch=visual_batch,
        )

    def _run_stage(
        self,
        context: PipelineContext,
        stage_metrics: _StageMetrics,
        stage: str,
        operation: Callable[[], StageResult],
    ) -> StageResult:
        self._start_stage(context, stage)
        started_at = self._clock()
        result = operation()
        self._check_cancelled(context)
        stage_metrics.add(stage, self._elapsed_ms(started_at))
        return result

    def _new_run_cache(self, registered: RegisteredAsset) -> DocumentModelCache:
        run_root = safe_runtime_path(self._runtime_root, registered.run_relative_root)
        return self._model_cache_factory(run_root)

    def _elapsed_ms(self, started_at: float) -> int:
        return max(0, round((self._clock() - started_at) * 1_000))

    @staticmethod
    def _validated_cache_hits(
        cache_hits: tuple[str, ...],
        stage_metrics: Mapping[str, int],
    ) -> tuple[str, ...]:
        if len(cache_hits) != len(set(cache_hits)):
            raise ValueError("stage_cache_hits 不得重复")
        if any(
            name not in stage_metrics or stage_metrics[name] != 0
            for name in cache_hits
        ):
            raise ValueError("缓存命中阶段必须存在且耗时为 0")
        return cache_hits

    @staticmethod
    def _check_cancelled(context: PipelineContext) -> None:
        if context.is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")

    @classmethod
    def _start_stage(cls, context: PipelineContext, stage: str) -> None:
        cls._check_cancelled(context)
        context.on_stage_start(stage)
        cls._check_cancelled(context)
