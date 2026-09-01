from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from video_demo.domain.document import DocumentGenerationConfig
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository, MediaRunRepository
from video_demo.persistence.models import (
    ImageObjectModel,
    ImageUnderstandingRunModel,
)
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope


@dataclass(frozen=True, slots=True)
class MediaRunView:
    run_id: str
    job_id: str
    status: str
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


class ImageSchedulerPort(Protocol):
    def submit(
        self,
        scope: Scope,
        run_id: str,
    ) -> Literal["accepted", "already_queued", "rejected"]: ...


class MediaRunService:
    """图片运行服务；音频运行由 AudioRunService 独立处理。"""

    def __init__(
        self,
        database: Database,
        scheduler: ImageSchedulerPort | None = None,
    ) -> None:
        self._database = database
        self._scheduler = scheduler

    def create(
        self,
        *,
        scope: Scope,
        object_ref: str,
        idempotency_key: str,
        language_hints: tuple[str, ...] = (),
        hotwords: tuple[str, ...] = (),
        core_context: str | None = None,
        document_config: DocumentGenerationConfig | None = None,
    ) -> MediaRunView:
        from video_demo.application.pipeline_contracts import PipelineRunConfig

        config = PipelineRunConfig(
            language_hints=language_hints,
            hotwords=hotwords,
            core_context=core_context,
            document_config=document_config or DocumentGenerationConfig(),
        )
        with self._database.session() as session:
            objects = MediaObjectRepository(session, ImageObjectModel)
            if objects.get_ready(scope, object_ref) is None:
                raise VideoDemoError(self._object_not_found_code(), "媒体对象不存在")
            runs = MediaRunRepository(session, ImageUnderstandingRunModel)
            existing = runs.get_by_idempotency(scope, idempotency_key)
            if existing is not None:
                config_snapshot = config.model_dump(mode="json")
                if existing.object_ref != object_ref or existing.config_snapshot != config_snapshot:
                    raise VideoDemoError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "幂等键已用于另一个媒体运行",
                    )
                job = JobRepository(session).get_by_resource_type(
                    scope,
                    existing.run_id,
                    "IMAGE_UNDERSTANDING_RUN",
                )
                if job is None:
                    raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "媒体运行任务不存在")
                return _view(existing, job.job_id)
            config_snapshot = config.model_dump(mode="json")
            run_id = f"run_{uuid.uuid4().hex}"
            job_id = f"job_{uuid.uuid4().hex}"
            run = runs.add(
                scope=scope,
                run_id=run_id,
                object_ref=object_ref,
                idempotency_key=idempotency_key,
                config_snapshot=config_snapshot,
            )
            JobRepository(session).enqueue_media_run(
                scope=scope,
                job_id=job_id,
                resource_id=run_id,
                job_type="IMAGE_UNDERSTANDING",
                resource_type="IMAGE_UNDERSTANDING_RUN",
            )
            view = _view(run, job_id)
        if self._scheduler is not None:
            result = self._scheduler.submit(scope, run_id)
            if result == "rejected":
                raise VideoDemoError(ErrorCode.JOB_NOT_RETRYABLE, "图片调度器暂不可用")
        return view

    def get(self, scope: Scope, run_id: str) -> MediaRunView:
        with self._database.session() as session:
            run = MediaRunRepository(session, ImageUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(self._run_not_found_code(), "媒体运行不存在")
            job = JobRepository(session).get_by_resource_type(
                scope,
                run_id,
                "IMAGE_UNDERSTANDING_RUN",
            )
            if job is None:
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "媒体运行任务不存在")
            return _view(run, job.job_id)

    def list_history(self, scope: Scope) -> tuple[dict[str, Any], ...]:
        with self._database.session() as session:
            objects = MediaObjectRepository(session, ImageObjectModel)
            result = []
            runs = MediaRunRepository(session, ImageUnderstandingRunModel)
            for run in runs.list_with_objects(scope):
                obj = objects.get(scope, run.object_ref)
                job = JobRepository(session).get_by_resource_type(
                    scope,
                    run.run_id,
                    "IMAGE_UNDERSTANDING_RUN",
                )
                if obj is None or job is None:
                    continue
                result.append(
                    {
                        "run_id": run.run_id,
                        "job_id": job.job_id,
                        "status": run.status.value,
                        "current_stage": run.current_stage,
                        "warning_codes": tuple(run.warning_codes),
                        "error_code": run.error_code,
                        "object_ref": run.object_ref,
                        "original_filename": obj.original_filename,
                        "detected_mime": obj.detected_mime,
                        "size_bytes": obj.size_bytes,
                        "created_at": run.created_at,
                        "updated_at": run.updated_at,
                    },
                )
            return tuple(result)

    def require_result_ready(self, scope: Scope, run_id: str) -> MediaRunView:
        view = self.get(scope, run_id)
        if view.status not in {"SUCCEEDED", "PARTIAL_SUCCEEDED"}:
            raise VideoDemoError(
                ErrorCode.VIDEO_RESULT_NOT_READY,
                "媒体理解结果尚未就绪",
            )
        return view

    def get_job(self, scope: Scope, job_id: str) -> Any:
        from video_demo.application.runs import _job_view

        with self._database.session() as session:
            job = JobRepository(session).get(scope, job_id)
            if job is None or job.resource_type != "IMAGE_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "图片任务不存在")
            return _job_view(job)

    def cancel_job(self, scope: Scope, job_id: str) -> Any:
        from video_demo.application.runs import _job_view

        with self._database.session() as session:
            repository = JobRepository(session)
            job = repository.get(scope, job_id)
            if job is None or job.resource_type != "IMAGE_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "图片任务不存在")
            repository.request_cancel(scope, job_id)
            updated = repository.get(scope, job_id)
            assert updated is not None
            return _job_view(updated)

    def retry_job(self, scope: Scope, job_id: str) -> Any:
        from video_demo.application.runs import _job_view

        with self._database.session() as session:
            repository = JobRepository(session)
            job = repository.get(scope, job_id)
            if job is None or job.resource_type != "IMAGE_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "图片任务不存在")
            retried = repository.retry(scope, job_id)
            view = _job_view(retried)
            run_id = retried.resource_id
        if self._scheduler is not None:
            result = self._scheduler.submit(scope, run_id)
            if result == "rejected":
                raise VideoDemoError(ErrorCode.JOB_NOT_RETRYABLE, "图片调度器暂不可用")
        return view

    def _object_not_found_code(self) -> ErrorCode:
        return ErrorCode.IMAGE_OBJECT_NOT_FOUND

    def _run_not_found_code(self) -> ErrorCode:
        return ErrorCode.IMAGE_RUN_NOT_FOUND


def _view(run: Any, job_id: str) -> MediaRunView:
    return MediaRunView(
        run_id=run.run_id,
        job_id=job_id,
        status=run.status.value,
        current_stage=run.current_stage,
        warning_codes=tuple(run.warning_codes),
        error_code=run.error_code,
    )
