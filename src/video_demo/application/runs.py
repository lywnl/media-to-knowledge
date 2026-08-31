from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import ValidationError

from video_demo.application.pipeline import (
    PipelineRunConfig,
    pipeline_run_config_from_snapshot,
)
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import (
    JobModel,
    JobStatus,
    RunStatusValue,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.repositories import (
    JobRepository,
    VideoObjectRepository,
    VideoRunRepository,
    VideoStageRepository,
)
from video_demo.persistence.scope import Scope


@dataclass(frozen=True, slots=True)
class RunView:
    run_id: str
    job_id: str
    status: RunStatusValue
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


@dataclass(frozen=True, slots=True)
class RunHistoryView:
    run_id: str
    object_ref: str
    original_filename: str
    detected_mime: str
    size_bytes: int
    status: RunStatusValue
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class JobView:
    job_id: str
    resource_id: str
    status: JobStatus
    attempt_count: int
    max_attempts: int
    error_code: str | None


class RunService:
    def __init__(self, database: Database, scheduler: object | None = None) -> None:
        self._database = database
        self._scheduler = scheduler

    def create(
        self,
        *,
        scope: Scope,
        object_ref: str,
        idempotency_key: str,
        language_hints: tuple[str, ...],
        hotwords: tuple[str, ...] = (),
        core_context: str | None = None,
        document_config: DocumentGenerationConfig | None = None,
        result_schema_version: Literal["4.1.0"] = "4.1.0",
    ) -> RunView:
        config = PipelineRunConfig(
            language_hints=language_hints,
            hotwords=hotwords,
            core_context=core_context,
            document_config=document_config or DocumentGenerationConfig(),
            result_schema_version=result_schema_version,
        )
        config_snapshot = config.model_dump(mode="json")
        with self._database.session() as session:
            object_model = VideoObjectRepository(session).get_ready(scope, object_ref)
            if object_model is None:
                raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")

            runs = VideoRunRepository(session)
            existing = runs.get_by_idempotency(scope, idempotency_key)
            if existing is not None:
                if existing.object_ref != object_ref or not self._same_config(
                    existing.config_snapshot,
                    config,
                ):
                    raise VideoDemoError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "幂等键已用于另一个视频对象",
                    )
                job = JobRepository(session).get_by_resource(scope, existing.run_id)
                assert job is not None
                return _run_view(existing, job.job_id)

            asset = runs.get_or_create_asset(
                scope=scope,
                asset_id=f"asset_{uuid.uuid4().hex}",
                object_ref=object_ref,
                source_sha256=object_model.sha256,
            )
            run_id = f"run_{uuid.uuid4().hex}"
            job_id = f"job_{uuid.uuid4().hex}"
            run = runs.add(
                scope=scope,
                run_id=run_id,
                asset_id=asset.asset_id,
                object_ref=object_ref,
                idempotency_key=idempotency_key,
                config_snapshot=config_snapshot,
            )
            JobRepository(session).enqueue_video_run(
                scope=scope,
                job_id=job_id,
                run_id=run_id,
            )
            VideoStageRepository(session).ensure(scope, run_id)
            view = _run_view(run, job_id)
        if self._scheduler is not None:
            result = self._scheduler.submit(scope, run_id)
            if result == "rejected":
                raise VideoDemoError(ErrorCode.JOB_NOT_RETRYABLE, "视频调度器暂不可用")
        return view

    @staticmethod
    def _same_config(snapshot: dict[str, object], expected: PipelineRunConfig) -> bool:
        try:
            existing = pipeline_run_config_from_snapshot(snapshot)
        except (ValidationError, VideoDemoError):
            return False
        return existing == expected

    def get(self, scope: Scope, run_id: str) -> RunView:
        with self._database.session() as session:
            run = VideoRunRepository(session).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            job = JobRepository(session).get_by_resource(scope, run_id)
            assert job is not None
            return _run_view(run, job.job_id)

    def list_history(self, scope: Scope) -> tuple[RunHistoryView, ...]:
        with self._database.session() as session:
            records = VideoRunRepository(session).list_with_objects(scope)
            return tuple(
                RunHistoryView(
                    run_id=run.run_id,
                    object_ref=run.object_ref,
                    original_filename=video.original_filename,
                    detected_mime=video.detected_mime,
                    size_bytes=video.size_bytes,
                    status=run.status,
                    current_stage=run.current_stage,
                    warning_codes=tuple(run.warning_codes),
                    error_code=run.error_code,
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                )
                for run, video in records
            )

    def require_result_ready(self, scope: Scope, run_id: str) -> RunView:
        view = self.get(scope, run_id)
        if view.status not in (RunStatusValue.SUCCEEDED, RunStatusValue.PARTIAL_SUCCEEDED):
            raise VideoDemoError(ErrorCode.VIDEO_RESULT_NOT_READY, "视频理解结果尚未就绪")
        return view

    def get_job(self, scope: Scope, job_id: str) -> JobView:
        with self._database.session() as session:
            job = JobRepository(session).get(scope, job_id)
            if job is None:
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "任务不存在")
            return _job_view(job)

    def cancel_job(self, scope: Scope, job_id: str) -> JobView:
        with self._database.session() as session:
            repository = JobRepository(session)
            if not repository.request_cancel(scope, job_id):
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "任务不存在")
            job = repository.get(scope, job_id)
            assert job is not None
            return _job_view(job)

    def retry_job(self, scope: Scope, job_id: str) -> JobView:
        stage_to_submit = None
        resource_id = None
        with self._database.session() as session:
            repository = JobRepository(session)
            job = repository.retry(scope, job_id)
            if job.resource_type == "VIDEO_UNDERSTANDING_RUN":
                resource_id = job.resource_id
                reset = VideoStageRepository(session).reset_for_retry(scope, resource_id)
                stage_to_submit = reset[0] if reset else None
            view = _job_view(job)
        if self._scheduler is not None and resource_id is not None and stage_to_submit is not None:
            self._scheduler.submit(scope, resource_id, stage_to_submit)
        return view


def _run_view(run: VideoUnderstandingRunModel, job_id: str) -> RunView:
    return RunView(
        run_id=run.run_id,
        job_id=job_id,
        status=run.status,
        current_stage=run.current_stage,
        warning_codes=tuple(run.warning_codes),
        error_code=run.error_code,
    )


def _job_view(job: JobModel) -> JobView:
    return JobView(
        job_id=job.job_id,
        resource_id=job.resource_id,
        status=job.status,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        error_code=job.error_code,
    )
