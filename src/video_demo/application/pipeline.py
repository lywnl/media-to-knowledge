from __future__ import annotations

from typing import Protocol

from video_demo.application.document_pipeline import (
    VideoUnderstandingPipeline as VideoUnderstandingPipeline,
)
from video_demo.application.document_publication import ResultWriteFence
from video_demo.application.pipeline_contracts import PipelineContext as PipelineContext
from video_demo.application.pipeline_contracts import PipelineOutcome as PipelineOutcome
from video_demo.application.pipeline_contracts import PipelineRunConfig as PipelineRunConfig
from video_demo.application.pipeline_contracts import PreparedMedia as PreparedMedia
from video_demo.application.pipeline_contracts import ProbedAsset as ProbedAsset
from video_demo.application.pipeline_contracts import RegisteredAsset as RegisteredAsset
from video_demo.application.pipeline_contracts import SceneIndex as SceneIndex
from video_demo.application.pipeline_contracts import SpeechAnalysis as SpeechAnalysis
from video_demo.application.pipeline_contracts import (
    SpeechBoundaryCandidate as SpeechBoundaryCandidate,
)
from video_demo.application.pipeline_contracts import StageMetric as StageMetric
from video_demo.application.pipeline_contracts import (
    pipeline_run_config_from_snapshot as pipeline_run_config_from_snapshot,
)
from video_demo.application.queries import ResultQueryService
from video_demo.domain.title import sanitize_document_title
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.persistence.database import Database
from video_demo.persistence.models import RunStatusValue
from video_demo.persistence.repositories import (
    ClaimedJob,
    JobRepository,
    VideoObjectRepository,
    VideoRunRepository,
)


class PipelineRunner(Protocol):
    def run(self, context: PipelineContext) -> PipelineOutcome: ...


class PipelineJobHandler:
    """把可靠任务租约转换为 4.1 流水线上下文并原子发布唯一结果。"""

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
        try:
            config, original_filename = self._load_inputs(job)
            title_hint = sanitize_document_title(
                config.document_config.document_title,
                original_filename,
            ) or "视频知识文档"
            context = PipelineContext(
                run_id=job.resource_id,
                scope=job.scope,
                title_hint=title_hint,
                document_config=config.document_config,
                is_cancel_requested=lambda: self._is_cancel_requested(job),
                on_stage_start=lambda stage: self._mark_stage(job, stage),
            )
            outcome = self._pipeline.run(context)
            self._result_queries.persist(
                job.scope,
                outcome.result,
                evidence=outcome.evidence,
                document=outcome.document,
                stage_metrics=dict(outcome.stage_metrics),
                model_metrics=dict(outcome.model_metrics),
                stage_cache_hits=outcome.stage_cache_hits,
                status=outcome.status,
                warnings=outcome.warnings,
                transcript_source=outcome.transcript_source,
                published_keyframes=outcome.visual_batch.keyframe_evidence,
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
            failure = VideoDemoError(ErrorCode.OUT_OF_MEMORY, "任务内存不足")
            self._mark_unsuccessful(job, failure)
            raise failure from None
        except Exception:
            self._mark_unsuccessful(
                job,
                VideoDemoError(ErrorCode.SYSTEM_FAILURE, "任务发生未分类系统错误"),
            )
            raise

    def _load_inputs(self, job: ClaimedJob) -> tuple[PipelineRunConfig, str]:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(job.scope, job.resource_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            video = VideoObjectRepository(session).get(job.scope, run.object_ref)
            if video is None:
                raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")
            return pipeline_run_config_from_snapshot(run.config_snapshot), video.original_filename

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
            JobRepository(session).update_owned_video_run(job, values={"current_stage": stage})

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
            status = (
                RunStatusValue.PENDING
                if is_retryable_error_code(error.code) and job.attempt_count < job.max_attempts
                else RunStatusValue.FAILED
            )
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
