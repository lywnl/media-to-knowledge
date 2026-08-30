from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, cast

from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.application.pipeline_contracts import PipelineRunConfig
from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository, MediaRunRepository
from video_demo.persistence.models import (
    AudioObjectModel,
    AudioUnderstandingRunModel,
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


class MediaRunService:
    def __init__(
        self,
        database: Database,
        *,
        kind: Literal["AUDIO", "IMAGE"],
    ) -> None:
        self._database = database
        self._kind = kind
        if kind == "AUDIO":
            self._object_model = cast(type[Any], AudioObjectModel)
            self._run_model = cast(type[Any], AudioUnderstandingRunModel)
            self._job_type = "AUDIO_UNDERSTANDING"
            self._resource_type = "AUDIO_UNDERSTANDING_RUN"
        else:
            self._object_model = cast(type[Any], ImageObjectModel)
            self._run_model = cast(type[Any], ImageUnderstandingRunModel)
            self._job_type = "IMAGE_UNDERSTANDING"
            self._resource_type = "IMAGE_UNDERSTANDING_RUN"

    def create(
        self,
        *,
        scope: Scope,
        object_ref: str,
        idempotency_key: str,
        language_hints: tuple[str, ...] = (),
        hotwords: tuple[str, ...] = (),
        core_context: str | None = None,
        document_config: DocumentGenerationConfig | AudioDocumentConfig | None = None,
    ) -> MediaRunView:
        config: PipelineRunConfig | AudioRunConfig
        if self._kind == "AUDIO":
            config = AudioRunConfig(
                language_hints=language_hints,
                hotwords=hotwords,
                core_context=core_context,
                document_config=document_config or AudioDocumentConfig(),
            )
        else:
            config = PipelineRunConfig(
                language_hints=language_hints,
                hotwords=hotwords,
                core_context=core_context,
                document_config=document_config or DocumentGenerationConfig(),
            )
        with self._database.session() as session:
            objects = MediaObjectRepository(session, self._object_model)
            if objects.get_ready(scope, object_ref) is None:
                raise VideoDemoError(self._object_not_found_code(), "媒体对象不存在")
            runs = MediaRunRepository(session, self._run_model)
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
                    self._resource_type,
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
                job_type=self._job_type,
                resource_type=self._resource_type,
            )
            return _view(run, job_id)

    def get(self, scope: Scope, run_id: str) -> MediaRunView:
        with self._database.session() as session:
            run = MediaRunRepository(session, self._run_model).get(scope, run_id)
            if run is None:
                raise VideoDemoError(self._run_not_found_code(), "媒体运行不存在")
            job = JobRepository(session).get_by_resource_type(
                scope,
                run_id,
                self._resource_type,
            )
            if job is None:
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "媒体运行任务不存在")
            return _view(run, job.job_id)

    def list_history(self, scope: Scope) -> tuple[dict[str, Any], ...]:
        with self._database.session() as session:
            objects = MediaObjectRepository(session, self._object_model)
            result = []
            for run in MediaRunRepository(session, self._run_model).list_with_objects(scope):
                obj = objects.get(scope, run.object_ref)
                job = JobRepository(session).get_by_resource_type(
                    scope,
                    run.run_id,
                    self._resource_type,
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
            code = (
                ErrorCode.AUDIO_RESULT_NOT_READY
                if self._kind == "AUDIO"
                else ErrorCode.VIDEO_RESULT_NOT_READY
            )
            raise VideoDemoError(code, "媒体理解结果尚未就绪")
        return view

    def _object_not_found_code(self) -> ErrorCode:
        return (
            ErrorCode.AUDIO_OBJECT_NOT_FOUND
            if self._kind == "AUDIO"
            else ErrorCode.IMAGE_OBJECT_NOT_FOUND
        )

    def _run_not_found_code(self) -> ErrorCode:
        return (
            ErrorCode.AUDIO_RUN_NOT_FOUND
            if self._kind == "AUDIO"
            else ErrorCode.IMAGE_RUN_NOT_FOUND
        )


def _view(run: Any, job_id: str) -> MediaRunView:
    return MediaRunView(
        run_id=run.run_id,
        job_id=job_id,
        status=run.status.value,
        current_stage=run.current_stage,
        warning_codes=tuple(run.warning_codes),
        error_code=run.error_code,
    )
