from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from video_demo.application.document_publication import ResultWriteFence
from video_demo.application.image_pipeline import ImageAnalyzer, run_image_pipeline
from video_demo.application.media_publication import MediaPublicationService
from video_demo.application.pipeline_contracts import PipelineRunConfig
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import (
    MediaObjectRepository,
    MediaRunRepository,
)
from video_demo.persistence.models import (
    ImageObjectModel,
    ImageUnderstandingRunModel,
    RunStatusValue,
)
from video_demo.persistence.repositories import ClaimedJob, JobRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.image_object_store import ImageObjectStore
from video_demo.storage.media_object_store import MediaObjectRecord


class ImageAnalyzerFactory(Protocol):
    def __call__(self) -> ImageAnalyzer: ...


def _fence(job: ClaimedJob) -> ResultWriteFence:
    return ResultWriteFence(
        job_pk=job.id,
        worker_id=job.worker_id,
        attempt_count=job.attempt_count,
    )


def _scope_key(scope: Scope) -> str:
    value = "\x00".join(
        (scope.tenant_id, scope.application_id, scope.knowledge_base_id),
    ).encode()
    return hashlib.sha256(value).hexdigest()[:24]


class ImageJobHandler:
    """领取图片任务，执行单图 VLM 文档流水线并发布 Markdown。"""

    def __init__(
        self,
        database: Database,
        analyzer_factory: ImageAnalyzerFactory,
        publication: MediaPublicationService,
        object_store: ImageObjectStore,
        *,
        runtime_root: Path,
        max_image_bytes: int,
    ) -> None:
        self._database = database
        self._analyzer_factory = analyzer_factory
        self._publication = publication
        self._object_store = object_store
        self._runtime_root = runtime_root
        self._max_image_bytes = max_image_bytes

    def __call__(self, job: ClaimedJob) -> None:
        self._mark_running(job)
        try:
            source, config, filename, asset_sha, relative_path, mime_type = self._load_input(job)
            if source.stat().st_size > self._max_image_bytes:
                raise VideoDemoError(ErrorCode.IMAGE_FILE_TOO_LARGE, "图片文件超过大小限制")
            self._mark_stage(job, "VLM")
            outcome = run_image_pipeline(
                run_id=job.resource_id,
                asset_sha256=asset_sha,
                source=source,
                relative_path=relative_path,
                mime_type=mime_type,
                title_hint=config.document_config.document_title or filename,
                analyzer=self._analyzer_factory(),
                runtime_root=self._runtime_root,
                max_image_bytes=self._max_image_bytes,
            )
            self._publication.persist(
                job.scope,
                outcome.result,
                document=outcome.document,
                status="PARTIAL_SUCCEEDED" if outcome.warnings else "SUCCEEDED",
                warnings=outcome.warnings,
                fence=_fence(job),
            )
        except VideoDemoError as error:
            self._mark_unsuccessful(job, error)
            raise
        except Exception as system_error:
            failure = VideoDemoError(ErrorCode.SYSTEM_FAILURE, "图片任务发生未分类系统错误")
            self._mark_unsuccessful(job, failure)
            raise failure from system_error

    def _load_input(
        self,
        job: ClaimedJob,
    ) -> tuple[Path, PipelineRunConfig, str, str, str, str]:
        with self._database.session() as session:
            run = MediaRunRepository(session, ImageUnderstandingRunModel).get(
                job.scope,
                job.resource_id,
            )
            if run is None:
                raise VideoDemoError(ErrorCode.IMAGE_RUN_NOT_FOUND, "图片运行不存在")
            obj = MediaObjectRepository(session, ImageObjectModel).get_ready(
                job.scope,
                run.object_ref,
            )
            if obj is None:
                raise VideoDemoError(ErrorCode.IMAGE_OBJECT_NOT_FOUND, "图片对象不存在")
            record = MediaObjectRecord(
                object_ref=obj.object_ref,
                original_filename=obj.original_filename,
                declared_mime=obj.declared_mime,
                detected_mime=obj.detected_mime,
                size_bytes=obj.size_bytes,
                sha256=obj.sha256,
                relative_path=obj.relative_path,
                scope_key=self._object_store.scope_key(job.scope),
            )
            source = self._object_store.materialize(
                job.scope,
                record,
                job.resource_id,
                obj.sha256,
            )
            try:
                config = PipelineRunConfig.model_validate(run.config_snapshot)
            except ValueError as error:
                raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "图片运行配置非法") from error
            return (
                source,
                config,
                obj.original_filename,
                obj.sha256,
                str(Path("runs") / _scope_key(job.scope) / job.resource_id / "input" / source.name),
                obj.detected_mime,
            )

    def _mark_running(self, job: ClaimedJob) -> None:
        self._update(job, {"status": RunStatusValue.RUNNING, "current_stage": "PROBE"})

    def _mark_stage(self, job: ClaimedJob, stage: str) -> None:
        self._update(job, {"current_stage": stage})

    def _update(self, job: ClaimedJob, values: dict[str, object]) -> None:
        with self._database.session() as session:
            JobRepository(session).update_owned_media_run(
                job,
                values=values,
                run_model=ImageUnderstandingRunModel,
                resource_type="IMAGE_UNDERSTANDING_RUN",
            )

    def _mark_unsuccessful(self, job: ClaimedJob, error: VideoDemoError) -> None:
        with self._database.session() as session:
            repository = JobRepository(session)
            if error.code == ErrorCode.JOB_CANCELLED:
                repository.mark_cancelled(job.id, job.worker_id, attempt_count=job.attempt_count)
                return
            repository.update_owned_media_run(
                job,
                values={"status": RunStatusValue.FAILED, "error_code": error.code.value},
                run_model=ImageUnderstandingRunModel,
                resource_type="IMAGE_UNDERSTANDING_RUN",
            )
            repository.mark_failed(
                job.id,
                job.worker_id,
                error_code=error.code,
                retryable=is_retryable_error_code(error.code),
                attempt_count=job.attempt_count,
            )
