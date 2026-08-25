from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Protocol, TypeVar

from video_demo.application.pipeline_contracts import PipelineRunConfig as PipelineRunConfig
from video_demo.application.pipeline_contracts import PreparedMedia as PreparedMedia
from video_demo.application.pipeline_contracts import ProbedAsset as ProbedAsset
from video_demo.application.pipeline_contracts import RegisteredAsset as RegisteredAsset
from video_demo.application.pipeline_contracts import SpeechAnalysis as SpeechAnalysis
from video_demo.application.pipeline_contracts import (
    SpeechBoundaryCandidate as SpeechBoundaryCandidate,
)
from video_demo.application.pipeline_contracts import StageMetric as StageMetric
from video_demo.application.pipeline_contracts import (
    pipeline_run_config_from_snapshot as pipeline_run_config_from_snapshot,
)
from video_demo.application.queries import ResultQueryService, ResultWriteFence
from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import (
    EvidenceItem,
    KeyframeEvidence,
    OcrEvidence,
    SceneBoundary,
    TimelineEvidence,
)
from video_demo.domain.result import (
    SummaryChapter,
    VideoUnderstandingResult,
    validate_evidence_references,
)
from video_demo.domain.result_artifact import TranscriptSource
from video_demo.domain.run import RunStatus, TimeRange
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.fusion.merge import (
    BoundaryPoint,
    WindowUnderstanding,
    merge_segment_understandings,
)
from video_demo.fusion.result_builder import build_video_summary
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.video_port import (
    VideoClipInput,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingPort,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowInput,
)
from video_demo.persistence.database import Database
from video_demo.persistence.models import RunStatusValue
from video_demo.persistence.repositories import ClaimedJob, JobRepository, Scope

StageInput = TypeVar("StageInput")
StageOutput = TypeVar("StageOutput")

_MAX_QWEN_FULL_VIDEO_BYTES = 128 * 1024 * 1024

@dataclass(frozen=True, slots=True)
class PipelineContext:
    run_id: str
    scope: Scope | None = None
    is_cancel_requested: Callable[[], bool] = lambda: False
    on_stage_start: Callable[[str], None] = lambda _stage: None


@dataclass(frozen=True, slots=True)
class VisualPreparation:
    proxy_sha256: str
    proxy_size_bytes: int
    run_relative_root: Path
    duration_ms: int
    frame_tolerance_ms: int
    scenes: tuple[SceneBoundary, ...]
    preparation_sha256: str
    observation_windows: tuple[TimeRange, ...] = ()
    keyframes: tuple[KeyframeEvidence, ...] = ()
    ocr: tuple[OcrEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    stage_metrics: tuple[StageMetric, ...] = ()


@dataclass(frozen=True, slots=True)
class VisualAnalysis:
    evidence: tuple[EvidenceItem, ...]
    boundaries: tuple[BoundaryPoint, ...]
    clips: tuple[VideoClipInput, ...] = ()
    windows: tuple[TimeRange, ...] = ()
    warnings: tuple[str, ...] = ()
    stage_metrics: tuple[StageMetric, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    status: RunStatus
    result: VideoUnderstandingResult
    evidence: tuple[EvidenceItem, ...]
    warnings: tuple[str, ...]
    stage_metrics: tuple[StageMetric, ...]
    transcript_source: TranscriptSource
    stage_cache_hits: tuple[str, ...] = ()


class AssetRegistrar(Protocol):
    def register(self, context: PipelineContext) -> RegisteredAsset: ...


class AssetProbe(Protocol):
    def probe(self, asset: RegisteredAsset) -> ProbedAsset: ...


class MediaTranscoder(Protocol):
    def transcode(
        self,
        probed: ProbedAsset,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> PreparedMedia: ...


class SpeechAnalyzer(Protocol):
    def analyze(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> SpeechAnalysis: ...


class VisualAnalyzer(Protocol):
    def prepare(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> VisualPreparation: ...

    def finalize(
        self,
        media: PreparedMedia,
        preparation: VisualPreparation,
        *,
        speech: SpeechAnalysis,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> VisualAnalysis: ...


class PipelineRunner(Protocol):
    def run(self, context: PipelineContext) -> PipelineOutcome: ...


class VideoUnderstandingPipeline:
    def __init__(
        self,
        registrar: AssetRegistrar,
        probe: AssetProbe,
        transcoder: MediaTranscoder,
        speech_analyzer: SpeechAnalyzer,
        visual_analyzer: VisualAnalyzer,
        understanding: WholeVideoUnderstandingPort,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registrar = registrar
        self._probe = probe
        self._transcoder = transcoder
        self._speech_analyzer = speech_analyzer
        self._visual_analyzer = visual_analyzer
        self._understanding = understanding
        self._clock = clock

    def run(self, context: PipelineContext) -> PipelineOutcome:
        metrics: list[StageMetric] = []
        registered = self._run_stage(
            context,
            metrics,
            "REGISTER",
            self._registrar.register,
            context,
        )
        probed = self._run_stage(context, metrics, "PROBE", self._probe.probe, registered)
        prepared = self._run_transcode(context, metrics, probed)
        speech, visual, branch_metrics = self._run_audio_visual(context, prepared)
        metrics.extend(branch_metrics)
        evidence = (*speech.evidence, *visual.evidence)
        timeline = self._run_stage(context, metrics, "FUSION", build_timeline, evidence)

        self._start_stage(context, "UNDERSTANDING")
        started_at = self._clock()
        whole_request = self._build_whole_video_request(
            prepared,
            visual,
            timeline,
            evidence,
        )
        self._check_cancelled(context)
        whole_understanding = self._understanding.understand_video(whole_request)
        window_results = self._map_whole_video_understanding(
            whole_request,
            whole_understanding,
        )
        metrics.append(self._metric("UNDERSTANDING", started_at))

        self._start_stage(context, "RESULT")
        started_at = self._clock()
        segments = merge_segment_understandings(
            window_results,
            boundaries=visual.boundaries,
            evidence=evidence,
            video_title=whole_understanding.summary.title,
        )
        chapters = tuple(
            SummaryChapter(
                title=segment.title,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                segment_ids=(segment.segment_id,),
            )
            for segment in segments
        )
        summary = build_video_summary(
            whole_understanding.summary,
            duration_ms=probed.duration_ms,
            segments=segments,
            chapters=chapters,
        )
        result = VideoUnderstandingResult(
            run_id=context.run_id,
            asset_sha256=registered.source_sha256,
            segments=segments,
            summary=summary,
        )
        validate_evidence_references(result, evidence)
        metrics.append(self._metric("RESULT", started_at))
        warnings = tuple(
            dict.fromkeys(
                (
                    *speech.warnings,
                    *visual.warnings,
                    *self._understanding_warnings(),
                ),
            ),
        )
        status = RunStatus.SUCCEEDED
        if any(warning.startswith("DEMO_DEGRADED_") for warning in warnings):
            status = RunStatus.PARTIAL_SUCCEEDED
        return PipelineOutcome(
            status=status,
            result=result,
            evidence=tuple(evidence),
            warnings=warnings,
            stage_metrics=tuple(metrics),
            stage_cache_hits=speech.stage_cache_hits,
            transcript_source=speech.transcript_source,
        )

    def _understanding_warnings(self) -> tuple[str, ...]:
        warnings = getattr(self._understanding, "degraded_warnings", ())
        if not isinstance(warnings, tuple) or any(not isinstance(item, str) for item in warnings):
            return ()
        return warnings

    def _run_audio_visual(
        self,
        context: PipelineContext,
        media: PreparedMedia,
    ) -> tuple[SpeechAnalysis, VisualAnalysis, tuple[StageMetric, ...]]:
        self._check_cancelled(context)
        self._start_stage(context, "SPEECH")
        self._start_stage(context, "VISUAL")
        self._check_cancelled(context)
        completion_times: dict[str, float] = {}
        completion_events = {"speech": Event(), "visual": Event()}

        def record_completion(name: str) -> None:
            completion_times[name] = self._clock()
            completion_events[name].set()

        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="video-understanding",
        ) as executor:
            self._check_cancelled(context)
            speech_future = executor.submit(
                self._timed_speech_analysis,
                media,
                context.is_cancel_requested,
            )
            self._check_cancelled(context)
            visual_started_at = self._clock()
            visual_future = executor.submit(
                self._visual_analyzer.prepare,
                media,
                is_cancel_requested=context.is_cancel_requested,
            )
            speech_future.add_done_callback(
                lambda _future: record_completion("speech"),
            )
            visual_future.add_done_callback(
                lambda _future: record_completion("visual"),
            )
            speech, speech_duration = speech_future.result()
            completion_events["speech"].wait()
            speech_completed_at = completion_times["speech"]
            preparation = visual_future.result()
            completion_events["visual"].wait()
            visual_completed_at = completion_times["visual"]
        self._check_cancelled(context)
        visual = self._visual_analyzer.finalize(
            media,
            preparation,
            speech=speech,
            is_cancel_requested=context.is_cancel_requested,
        )
        visual_duration = max(0, round((self._clock() - visual_started_at) * 1000))
        visual_wait_speech = max(0, round((speech_completed_at - visual_completed_at) * 1000))
        speech_metrics = tuple(
            metric for metric in speech.stage_metrics
            if metric.stage == "SPEECH_ASR"
        )
        visual_metrics = tuple((*preparation.stage_metrics, *visual.stage_metrics))
        visual_metrics = tuple(
            metric for metric in visual_metrics
            if metric.stage not in {"VISUAL", "VISUAL_WAIT_SPEECH", "VISUAL_FUSION"}
        )
        visual_fusion = next(
            (
                metric.duration_ms
                for metric in visual.stage_metrics
                if metric.stage == "VISUAL_FUSION"
            ),
            0,
        )
        return (
            speech,
            visual,
            (
                StageMetric(stage="SPEECH", duration_ms=speech_duration),
                *speech_metrics,
                *visual_metrics,
                StageMetric(stage="VISUAL_WAIT_SPEECH", duration_ms=visual_wait_speech),
                StageMetric(stage="VISUAL_FUSION", duration_ms=visual_fusion),
                StageMetric(stage="VISUAL", duration_ms=visual_duration),
            ),
        )

    def _run_transcode(
        self,
        context: PipelineContext,
        metrics: list[StageMetric],
        probed: ProbedAsset,
    ) -> PreparedMedia:
        self._check_cancelled(context)
        self._start_stage(context, "TRANSCODE")
        started_at = self._clock()
        prepared = self._transcoder.transcode(
            probed,
            is_cancel_requested=context.is_cancel_requested,
        )
        metrics.append(self._metric("TRANSCODE", started_at))
        return prepared

    def _build_whole_video_request(
        self,
        media: PreparedMedia,
        visual: VisualAnalysis,
        timeline: tuple[TimelineEvidence, ...],
        evidence: tuple[EvidenceItem, ...],
    ) -> WholeVideoUnderstandingRequest:
        if media.proxy_size_bytes > _MAX_QWEN_FULL_VIDEO_BYTES:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 完整视频理解输入不得超过 128 MiB",
            )
        ranges = self._understanding_ranges(visual)
        windows: list[WholeVideoWindowInput] = []
        for window in ranges:
            window_evidence = tuple(item for item in evidence if window.contains(item))
            window_timeline = tuple(item for item in timeline if window.contains(item))
            if not window_evidence or not window_timeline:
                raise VideoDemoError(
                    ErrorCode.QWEN_RESPONSE_INVALID,
                    "全片理解窗口缺少本地证据或时间轴",
                )
            windows.append(
                WholeVideoWindowInput(
                    window_id=stable_identifier(
                        "window",
                        {
                            "video_sha256": media.proxy_sha256,
                            "start_ms": window.start_ms,
                            "end_ms": window.end_ms,
                        },
                    ),
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    timeline=window_timeline,
                    evidence=window_evidence,
                )
            )
        video = VideoClipInput(
            clip_id=stable_identifier(
                "full_video",
                {
                    "sha256": media.proxy_sha256,
                    "duration_ms": media.source.duration_ms,
                },
            ),
            start_ms=0,
            end_ms=media.source.duration_ms,
            path=media.proxy_path,
            mime_type="video/mp4",
            sha256=media.proxy_sha256,
        )
        return WholeVideoUnderstandingRequest(video=video, windows=tuple(windows))

    @staticmethod
    def _understanding_ranges(visual: VisualAnalysis) -> tuple[TimeRange, ...]:
        if visual.windows:
            return visual.windows
        return tuple(
            TimeRange(start_ms=clip.start_ms, end_ms=clip.end_ms)
            for clip in sorted(
                visual.clips,
                key=lambda item: (item.start_ms, item.end_ms, item.clip_id),
            )
        )

    @staticmethod
    def _map_whole_video_understanding(
        request: WholeVideoUnderstandingRequest,
        result: WholeVideoUnderstanding,
    ) -> tuple[WindowUnderstanding, ...]:
        returned_by_id = {item.window_id: item.understanding for item in result.windows}
        expected_ids = {item.window_id for item in request.windows}
        if set(returned_by_id) != expected_ids:
            raise VideoDemoError(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 全片响应窗口集合与请求不一致",
            )
        mapped: list[WindowUnderstanding] = []
        for window in request.windows:
            understanding = returned_by_id[window.window_id]
            allowed_refs = {item.evidence_id for item in window.evidence}
            if not set(understanding.evidence_refs).issubset(allowed_refs):
                raise VideoDemoError(
                    ErrorCode.QWEN_RESPONSE_INVALID,
                    "Qwen 全片响应引用了窗口外证据",
                )
            mapped.append(
                WindowUnderstanding(
                    window_id=window.window_id,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    understanding=understanding,
                )
            )
        return tuple(mapped)

    def _run_stage(
        self,
        context: PipelineContext,
        metrics: list[StageMetric],
        name: str,
        function: Callable[[StageInput], StageOutput],
        argument: StageInput,
    ) -> StageOutput:
        self._check_cancelled(context)
        self._start_stage(context, name)
        started_at = self._clock()
        result = function(argument)
        metrics.append(self._metric(name, started_at))
        return result

    def _timed_speech_analysis(
        self,
        media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> tuple[SpeechAnalysis, int]:
        started_at = self._clock()
        result = self._speech_analyzer.analyze(
            media,
            is_cancel_requested=is_cancel_requested,
        )
        return result, max(0, round((self._clock() - started_at) * 1000))

    def _metric(self, stage: str, started_at: float) -> StageMetric:
        return StageMetric(
            stage=stage,
            duration_ms=max(0, round((self._clock() - started_at) * 1000)),
        )

    @staticmethod
    def _check_cancelled(context: PipelineContext) -> None:
        if context.is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")

    @staticmethod
    def _start_stage(context: PipelineContext, stage: str) -> None:
        context.on_stage_start(stage)


class PipelineJobHandler:
    """把可靠任务租约转换为流水线上下文，不承载媒体或模型实现。"""

    def __init__(
        self,
        database: Database,
        pipeline: PipelineRunner,
        result_queries: ResultQueryService,
    ) -> None:
        self._database = database
        self._pipeline = pipeline
        self._result_queries = result_queries

    def __call__(self, job: ClaimedJob) -> None:
        self._mark_running(job)
        context = PipelineContext(
            run_id=job.resource_id,
            scope=job.scope,
            is_cancel_requested=lambda: self._is_cancel_requested(job),
            on_stage_start=lambda stage: self._mark_stage(job, stage),
        )
        try:
            outcome = self._pipeline.run(context)
            self._result_queries.persist(
                job.scope,
                outcome.result,
                evidence=outcome.evidence,
                stage_metrics={
                    metric.stage: metric.duration_ms
                    for metric in outcome.stage_metrics
                },
                stage_cache_hits=outcome.stage_cache_hits,
                status=outcome.status,
                warnings=outcome.warnings,
                transcript_source=outcome.transcript_source,
                fence=ResultWriteFence(
                    job_pk=job.id,
                    worker_id=job.worker_id,
                    attempt_count=job.attempt_count,
                ),
            )
        except VideoDemoError as error:
            self._mark_unsuccessful(job, error)
            raise
        except MemoryError:
            memory_failure = VideoDemoError(ErrorCode.OUT_OF_MEMORY, "任务内存不足")
            self._mark_unsuccessful(job, memory_failure)
            raise memory_failure from None
        except Exception:
            self._mark_unsuccessful(
                job,
                VideoDemoError(ErrorCode.SYSTEM_FAILURE, "任务发生未分类系统错误"),
            )
            raise

    def _mark_running(self, job: ClaimedJob) -> None:
        with self._database.session() as session:
            JobRepository(session).update_owned_video_run(
                job,
                values={
                    "status": RunStatusValue.RUNNING,
                    "current_stage": "REGISTER",
                    "error_code": None,
                },
            )

    def _is_cancel_requested(self, job: ClaimedJob) -> bool:
        with self._database.session() as session:
            return JobRepository(session).is_cancel_requested(
                job.id,
                job.worker_id,
                attempt_count=job.attempt_count,
            )

    def _mark_stage(self, job: ClaimedJob, stage: str) -> None:
        with self._database.session() as session:
            JobRepository(session).update_owned_video_run(
                job,
                values={"current_stage": stage},
            )

    def _mark_unsuccessful(self, job: ClaimedJob, error: VideoDemoError) -> None:
        with self._database.session() as session:
            repository = JobRepository(session)
            if error.code == ErrorCode.JOB_CANCELLED:
                repository.mark_cancelled(
                    job.id,
                    job.worker_id,
                    attempt_count=job.attempt_count,
                )
                return
            if (
                is_retryable_error_code(error.code)
                and job.attempt_count < job.max_attempts
            ):
                status = RunStatusValue.PENDING
            else:
                status = RunStatusValue.FAILED
            repository.update_owned_video_run(
                job,
                values={"status": status, "error_code": error.code.value},
            )
            repository.mark_failed(
                job.id,
                job.worker_id,
                error_code=error.code,
                retryable=is_retryable_error_code(error.code),
                attempt_count=job.attempt_count,
            )
