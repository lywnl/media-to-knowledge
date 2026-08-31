from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Protocol

from sqlalchemy import select

from video_demo.application.document_pipeline import (
    VideoUnderstandingPipeline as VideoUnderstandingPipeline,
)
from video_demo.application.document_publication import ResultWriteFence, scope_key
from video_demo.application.pipeline_contracts import (
    PipelineContext as PipelineContext,
)
from video_demo.application.pipeline_contracts import (
    PipelineOutcome as PipelineOutcome,
)
from video_demo.application.pipeline_contracts import (
    PipelineRunConfig as PipelineRunConfig,
)
from video_demo.application.pipeline_contracts import (
    PreparedMedia as PreparedMedia,
)
from video_demo.application.pipeline_contracts import (
    ProbedAsset as ProbedAsset,
)
from video_demo.application.pipeline_contracts import (
    RegisteredAsset as RegisteredAsset,
)
from video_demo.application.pipeline_contracts import (
    SceneIndex as SceneIndex,
)
from video_demo.application.pipeline_contracts import (
    SpeechAnalysis as SpeechAnalysis,
)
from video_demo.application.pipeline_contracts import (
    SpeechBoundaryCandidate as SpeechBoundaryCandidate,
)
from video_demo.application.pipeline_contracts import (
    StageMetric as StageMetric,
)
from video_demo.application.pipeline_contracts import (
    TranscriptionCheckpoint,
    transcription_checkpoint_from_payload,
    transcription_checkpoint_to_payload,
)
from video_demo.application.pipeline_contracts import (
    pipeline_run_config_from_snapshot as pipeline_run_config_from_snapshot,
)
from video_demo.application.queries import ResultQueryService
from video_demo.domain.title import sanitize_document_title
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.persistence.database import Database
from video_demo.persistence.models import (
    JobStatus,
    RunStatusValue,
    VideoAssetModel,
    VideoStageName,
)
from video_demo.persistence.repositories import (
    ClaimedJob,
    JobRepository,
    VideoObjectRepository,
    VideoRunRepository,
    VideoStageLease,
    VideoStageRepository,
)
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.workspace import verified_run_file, verified_visual_file


class PipelineRunner(Protocol):
    def run(self, context: PipelineContext) -> PipelineOutcome: ...


class VideoStagePipelineExecutor:
    """以阶段租约驱动生产 Pipeline，并保存可验证的转写 checkpoint。"""

    def __init__(
        self,
        database: Database,
        pipeline: VideoUnderstandingPipeline,
        result_queries: ResultQueryService,
        runtime_root: Path,
        *,
        lease_seconds: int = 120,
    ) -> None:
        self._database = database
        self._pipeline = pipeline
        self._result_queries = result_queries
        self._store = AtomicArtifactStore(runtime_root)
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._lease_seconds = lease_seconds
        self._leases: dict[tuple[str, str, str], VideoStageLease] = {}
        self._jobs: dict[tuple[str, str, str], ClaimedJob] = {}

    def run_transcription(self, scope: Scope, run_id: str) -> object:
        lease = self._claim(scope, run_id, VideoStageName.TRANSCRIPTION)
        job = self._claim_job(scope, run_id, lease.worker_id)
        self._jobs[self._lease_key(scope, run_id, VideoStageName.TRANSCRIPTION)] = job
        context = self._context(scope, run_id, lease, job)
        checkpoint = self._run_with_heartbeat(
            lease,
            job,
            lambda: self._pipeline.run_transcription(context),
        )
        assert isinstance(checkpoint, TranscriptionCheckpoint)
        payload = transcription_checkpoint_to_payload(checkpoint)
        relative = (
            Path("runs")
            / scope_key(scope)
            / run_id
            / "stages"
            / "transcription-checkpoint.json"
        )
        receipt = self._store.write_json(
            relative,
            payload,
            schema_version="1.0.0",
            upstream_sha256=checkpoint.registered.source_sha256,
            file_mode=0o600,
            exclusive=False,
            max_bytes=16 * 1024 * 1024,
        )
        self._complete_transcription(lease, job, receipt.relative_path, receipt.sha256)
        return checkpoint

    def run_llm(self, scope: Scope, run_id: str, checkpoint: object) -> None:
        if not isinstance(checkpoint, TranscriptionCheckpoint):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "转写阶段快照类型非法")
        lease = self._claim(scope, run_id, VideoStageName.LLM)
        try:
            job = self._claim_job(scope, run_id, lease.worker_id)
        except VideoDemoError as error:
            if error.code != ErrorCode.JOB_LEASE_LOST or not self._result_exists(scope, run_id):
                raise
            self._reconcile_published_stage(scope, run_id, lease)
            return
        self._jobs[self._lease_key(scope, run_id, VideoStageName.LLM)] = job
        outcome = self._run_with_heartbeat(
            lease,
            job,
            lambda: self._pipeline.run_llm(self._context(scope, run_id, lease, job), checkpoint),
        )
        assert isinstance(outcome, PipelineOutcome)
        self._result_queries.persist(
            scope,
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
        self._mark_stage_succeeded(lease, None, None)

    def load_checkpoint(self, scope: Scope, run_id: str) -> object | None:
        with self._database.session() as session:
            record = VideoStageRepository(session).get(scope, run_id, VideoStageName.TRANSCRIPTION)
        if record is None or not record.checkpoint_relative_path or not record.checkpoint_sha256:
            return None
        try:
            payload = self._store.read_verified_json_limited(
                ArtifactReceipt(
                    relative_path=record.checkpoint_relative_path,
                    schema_version="1.0.0",
                    sha256=record.checkpoint_sha256,
                    upstream_sha256=self._asset_source_sha256(scope, run_id),
                ),
                max_bytes=16 * 1024 * 1024,
            )
            if not isinstance(payload, dict):
                raise ValueError("转写快照 payload 非法")
            checkpoint = transcription_checkpoint_from_payload(payload)
            self._validate_checkpoint(scope, run_id, checkpoint)
            return checkpoint
        except (OSError, ValueError, TypeError, VideoDemoError) as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "转写快照无法恢复") from error

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> None:
        """关闭无法安全恢复的阶段，避免静默重跑已完成转写。"""

        stage_name = VideoStageName(stage)
        with self._database.session() as session:
            changed = VideoStageRepository(session).mark_recovery_failed(
                scope,
                run_id,
                stage_name,
                error_code=error.code,
            )
            if changed:
                JobRepository(session).fail_unclaimed_video_run(
                    scope,
                    run_id,
                    error_code=error.code,
                    current_stage=stage,
                )

    def stage_succeeded(self, scope: Scope, run_id: str, stage: str, result: object) -> None:
        del scope, run_id, stage, result

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        with self._database.session() as session:
            job = JobRepository(session).get_by_resource(scope, run_id)
            return job is not None and job.status == JobStatus.CANCELLED

    def mark_stage_started(self, scope: Scope, run_id: str, stage: str) -> None:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is not None:
                run.status = RunStatusValue.RUNNING
                run.current_stage = stage
                run.error_code = None

    def stage_failed(self, scope: Scope, run_id: str, stage: str, error: VideoDemoError) -> bool:
        key = self._lease_key(scope, run_id, stage)
        lease = self._leases.pop(key, None)
        job = self._jobs.pop(key, None)
        if lease is None:
            return False
        if (
            error.code == ErrorCode.JOB_LEASE_LOST
            and stage == VideoStageName.LLM.value
            and self._result_exists(scope, run_id)
        ):
            self._reconcile_published_stage(scope, run_id, lease)
            return False
        with self._database.session() as session:
            repository = VideoStageRepository(session)
            if error.code == ErrorCode.JOB_CANCELLED:
                repository.mark_cancelled(lease)
                if job is not None:
                    repository_job = JobRepository(session)
                    repository_job.mark_cancelled(
                        job.id,
                        job.worker_id,
                        attempt_count=job.attempt_count,
                    )
                return False
            retryable = is_retryable_error_code(error.code)
            repository.mark_failed(
                lease,
                error_code=error.code,
                retryable=retryable,
            )
            if job is not None:
                JobRepository(session).release_video_stage(
                    job,
                    status=(JobStatus.RETRY_WAIT if retryable else JobStatus.FAILED),
                    current_stage=stage,
                    error_code=error.code,
                )
            return retryable and lease.attempt_count < lease.max_attempts

    def _claim(self, scope: Scope, run_id: str, stage: VideoStageName) -> VideoStageLease:
        worker_id = f"api-{uuid.uuid4().hex}"
        with self._database.session() as session:
            lease = VideoStageRepository(session).claim(
                scope,
                run_id,
                stage,
                worker_id,
                lease_seconds=self._lease_seconds,
            )
        if lease is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频阶段无法领取")
        self._leases[self._lease_key(scope, run_id, stage)] = lease
        return lease

    def _mark_stage_succeeded(
        self,
        lease: VideoStageLease,
        path: str | None,
        digest: str | None,
    ) -> None:
        key = self._lease_key(lease.scope, lease.run_id, lease.stage_name)
        try:
            with self._database.session() as session:
                VideoStageRepository(session).mark_succeeded(
                    lease,
                    checkpoint_relative_path=path,
                    checkpoint_sha256=digest,
                )
        except VideoDemoError as error:
            if error.code != ErrorCode.JOB_LEASE_LOST or not self._result_exists(
                lease.scope,
                lease.run_id,
            ):
                raise
            self._reconcile_published_stage(lease.scope, lease.run_id, lease)
        finally:
            self._leases.pop(key, None)
            self._jobs.pop(key, None)

    def _reconcile_published_stage(
        self,
        scope: Scope,
        run_id: str,
        lease: VideoStageLease,
    ) -> None:
        """结果已发布但阶段租约写回失败时，只收敛状态，不重复调用模型。"""

        with self._database.session() as session:
            VideoStageRepository(session).reconcile_succeeded(
                scope,
                run_id,
                lease.stage_name,
            )

    def _complete_transcription(
        self,
        lease: VideoStageLease,
        job: ClaimedJob,
        path: str,
        digest: str,
    ) -> None:
        key = self._lease_key(lease.scope, lease.run_id, lease.stage_name)
        with self._database.session() as session:
            VideoStageRepository(session).mark_succeeded(
                lease,
                checkpoint_relative_path=path,
                checkpoint_sha256=digest,
            )
            JobRepository(session).release_video_stage(
                job,
                status=JobStatus.PENDING,
                current_stage="LLM",
            )
        self._leases.pop(key, None)
        self._jobs.pop(key, None)

    def _run_with_heartbeat(
        self,
        lease: VideoStageLease,
        job: ClaimedJob,
        operation: Callable[[], object],
    ) -> object:
        stop = Event()
        errors: list[VideoDemoError] = []

        def heartbeat() -> None:
            while not stop.wait(max(0.5, self._lease_seconds / 3)):
                try:
                    with self._database.session() as session:
                        VideoStageRepository(session).heartbeat(
                            lease,
                            lease_seconds=self._lease_seconds,
                        )
                        JobRepository(session).heartbeat(
                            job.id,
                            job.worker_id,
                            attempt_count=job.attempt_count,
                            lease_seconds=self._lease_seconds,
                        )
                except VideoDemoError as error:
                    errors.append(error)
                    return

        thread = Thread(
            target=heartbeat,
            name=f"video-stage-heartbeat-{lease.run_id}",
            daemon=True,
        )
        thread.start()
        try:
            result = operation()
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]
        return result

    def _context(
        self,
        scope: Scope,
        run_id: str,
        lease: VideoStageLease,
        job: ClaimedJob | None = None,
    ) -> PipelineContext:
        config, title = self._load_inputs(scope, run_id)
        return PipelineContext(
            run_id=run_id,
            scope=scope,
            title_hint=title,
            document_config=config.document_config,
            is_cancel_requested=(
                (lambda: self._is_job_cancelled(job))
                if job is not None
                else (lambda: self._is_cancelled(lease))
            ),
            on_stage_start=lambda stage: self._update_stage(scope, run_id, stage),
        )

    def _load_inputs(self, scope: Scope, run_id: str) -> tuple[PipelineRunConfig, str]:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            video = VideoObjectRepository(session).get(scope, run.object_ref)
            if video is None:
                raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")
            document_snapshot = run.config_snapshot.get("document_config")
            title_snapshot = (
                document_snapshot.get("document_title")
                if isinstance(document_snapshot, dict)
                else None
            )
            title = sanitize_document_title(title_snapshot, video.original_filename)
            return pipeline_run_config_from_snapshot(run.config_snapshot), title or "视频知识文档"

    def _asset_source_sha256(self, scope: Scope, run_id: str) -> str:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            asset = session.scalar(
                select(VideoAssetModel).where(
                    VideoAssetModel.tenant_id == scope.tenant_id,
                    VideoAssetModel.application_id == scope.application_id,
                    VideoAssetModel.knowledge_base_id == scope.knowledge_base_id,
                    VideoAssetModel.asset_id == run.asset_id,
                )
            )
            if asset is None:
                raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "视频资产不存在")
            return asset.source_sha256

    def _validate_checkpoint(
        self,
        scope: Scope,
        run_id: str,
        checkpoint: TranscriptionCheckpoint,
    ) -> None:
        expected_root = Path("runs") / scope_key(scope) / run_id
        if checkpoint.registered.run_relative_root != expected_root:
            raise ValueError("转写快照运行目录不匹配")
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            video = VideoObjectRepository(session).get(scope, run.object_ref)
            if (
                video is None
                or video.object_ref != checkpoint.registered.object_ref
                or video.sha256 != checkpoint.registered.source_sha256
            ):
                raise ValueError("转写快照原始视频事实不匹配")
        source_path = self._run_candidate_path(checkpoint.registered.source_path)
        proxy_path = self._run_candidate_path(checkpoint.prepared.proxy_path)
        verified_visual_file(
            self._runtime_root,
            expected_root,
            source_path,
            expected_sha256=checkpoint.registered.source_sha256,
            expected_size_bytes=checkpoint.registered.source_size_bytes,
            max_size_bytes=checkpoint.registered.source_size_bytes,
            message="转写快照原始视频路径非法",
        )
        verified_visual_file(
            self._runtime_root,
            expected_root,
            proxy_path,
            expected_sha256=checkpoint.prepared.proxy_sha256,
            expected_size_bytes=checkpoint.prepared.proxy_size_bytes,
            max_size_bytes=checkpoint.prepared.proxy_size_bytes,
            message="转写快照视觉输入路径非法",
        )
        if checkpoint.prepared.audio_path is not None:
            audio_sha256 = checkpoint.prepared.audio_sha256
            if audio_sha256 is None:
                raise ValueError("转写快照音频摘要缺失")
            verified_run_file(
                self._runtime_root,
                expected_root,
                self._run_candidate_path(checkpoint.prepared.audio_path),
                expected_sha256=audio_sha256,
                digest=_sha256_file,
                message="转写快照音频路径非法",
            )

    def _run_candidate_path(self, candidate: Path | None) -> Path:
        if candidate is None:
            raise ValueError("运行路径不能为空")
        if candidate.is_absolute():
            try:
                return candidate.resolve(strict=False).relative_to(self._runtime_root)
            except ValueError as error:
                raise ValueError("运行路径不在运行目录内") from error
        return candidate

    def _result_exists(self, scope: Scope, run_id: str) -> bool:
        try:
            self._result_queries.get_result(scope, run_id)
        except VideoDemoError:
            return False
        return True

    def _is_cancelled(self, lease: VideoStageLease) -> bool:
        with self._database.session() as session:
            job = JobRepository(session).get_by_resource(lease.scope, lease.run_id)
            return job is not None and job.status == JobStatus.CANCELLED

    def _is_job_cancelled(self, job: ClaimedJob) -> bool:
        with self._database.session() as session:
            return JobRepository(session).is_cancel_requested(
                job.id,
                job.worker_id,
                attempt_count=job.attempt_count,
            )

    def _claim_job(self, scope: Scope, run_id: str, worker_id: str) -> ClaimedJob:
        with self._database.session() as session:
            job = JobRepository(session).claim_video_run(scope, run_id, worker_id)
        if job is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频总任务无法领取")
        return job

    def _update_stage(self, scope: Scope, run_id: str, stage: str) -> None:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is not None:
                run.current_stage = stage

    @staticmethod
    def _lease_key(
        scope: Scope,
        run_id: str,
        stage: str | VideoStageName,
    ) -> tuple[str, str, str]:
        return (
            scope.tenant_id + "\x00" + scope.application_id,
            scope.knowledge_base_id + "\x00" + run_id,
            str(stage),
        )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
