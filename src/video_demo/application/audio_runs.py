"""音频运行创建、查询和历史服务。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_stage_repository import AudioStageRepository
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository, MediaRunRepository
from video_demo.persistence.models import (
    AudioObjectModel,
    AudioStageName,
    AudioUnderstandingRunModel,
    JobModel,
    JobStatus,
)
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope


@dataclass(frozen=True, slots=True)
class AudioRunView:
    run_id: str
    job_id: str
    status: str
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


@dataclass(frozen=True, slots=True)
class AudioJobView:
    job_id: str
    resource_id: str
    status: JobStatus
    attempt_count: int
    max_attempts: int
    error_code: str | None


class AudioSchedulerPort(Protocol):
    def submit(
        self,
        scope: Scope,
        run_id: str,
        stage: AudioStageName,
    ) -> Literal["accepted", "already_queued", "rejected"]: ...


class AudioRunService:
    """只操作 audio_object 与 audio_understanding_run。"""

    def __init__(
        self,
        database: Database,
        scheduler: AudioSchedulerPort | None = None,
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
        document_config: AudioDocumentConfig | None = None,
    ) -> AudioRunView:
        config = AudioRunConfig(
            language_hints=language_hints,
            hotwords=hotwords,
            core_context=core_context,
            document_config=document_config or AudioDocumentConfig(),
        )
        config_snapshot = config.model_dump(mode="json")
        with self._database.session() as session:
            objects = MediaObjectRepository(session, AudioObjectModel)
            if objects.get_ready(scope, object_ref) is None:
                raise VideoDemoError(ErrorCode.AUDIO_OBJECT_NOT_FOUND, "音频对象不存在")
            runs = MediaRunRepository(session, AudioUnderstandingRunModel)
            existing = runs.get_by_idempotency(scope, idempotency_key)
            if existing is not None:
                if (
                    existing.object_ref != object_ref
                    or existing.config_snapshot != config_snapshot
                ):
                    raise VideoDemoError(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        "幂等键已用于另一个音频运行",
                    )
                job = JobRepository(session).get_by_resource_type(
                    scope, existing.run_id, "AUDIO_UNDERSTANDING_RUN"
                )
                if job is None:
                    raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频运行任务不存在")
                return _view(existing, job.job_id)
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
                job_type="AUDIO_UNDERSTANDING",
                resource_type="AUDIO_UNDERSTANDING_RUN",
            )
            AudioStageRepository(session).ensure(scope, run_id)
            view = _view(run, job_id)
        if self._scheduler is not None:
            result = self._scheduler.submit(scope, run_id, AudioStageName.TRANSCRIPTION)
            if result == "rejected":
                raise VideoDemoError(ErrorCode.JOB_NOT_RETRYABLE, "音频调度器暂不可用")
        return view

    def get(self, scope: Scope, run_id: str) -> AudioRunView:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            job = JobRepository(session).get_by_resource_type(
                scope, run_id, "AUDIO_UNDERSTANDING_RUN"
            )
            if job is None:
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频运行任务不存在")
            return _view(run, job.job_id)

    def get_job(self, scope: Scope, job_id: str) -> AudioJobView:
        with self._database.session() as session:
            job = JobRepository(session).get(scope, job_id)
            if job is None or job.resource_type != "AUDIO_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频任务不存在")
            return _job_view(job)

    def cancel_job(self, scope: Scope, job_id: str) -> AudioJobView:
        with self._database.session() as session:
            repository = JobRepository(session)
            job = repository.get(scope, job_id)
            if job is None or job.resource_type != "AUDIO_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频任务不存在")
            if not repository.request_cancel(scope, job_id):
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频任务不存在")
            updated = repository.get(scope, job_id)
            assert updated is not None
            return _job_view(updated)

    def retry_job(self, scope: Scope, job_id: str) -> AudioJobView:
        stage_to_submit: AudioStageName | None = None
        run_id: str | None = None
        with self._database.session() as session:
            repository = JobRepository(session)
            job = repository.get(scope, job_id)
            if job is None or job.resource_type != "AUDIO_UNDERSTANDING_RUN":
                raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "音频任务不存在")
            retried = repository.retry(scope, job_id)
            run_id = retried.resource_id
            reset = AudioStageRepository(session).reset_for_retry(scope, run_id)
            stage_to_submit = reset[0] if reset else None
            view = _job_view(retried)
        if self._scheduler is not None and run_id is not None and stage_to_submit is not None:
            self._scheduler.submit(scope, run_id, stage_to_submit)
        return view

    def list_history(self, scope: Scope) -> tuple[dict[str, Any], ...]:
        with self._database.session() as session:
            objects = MediaObjectRepository(session, AudioObjectModel)
            history: list[dict[str, Any]] = []
            runs = MediaRunRepository(session, AudioUnderstandingRunModel)
            for run in runs.list_with_objects(scope):
                audio = objects.get(scope, run.object_ref)
                job = JobRepository(session).get_by_resource_type(
                    scope, run.run_id, "AUDIO_UNDERSTANDING_RUN"
                )
                if audio is None or job is None:
                    continue
                history.append(
                    {
                        "run_id": run.run_id,
                        "job_id": job.job_id,
                        "status": run.status.value,
                        "current_stage": run.current_stage,
                        "warning_codes": tuple(run.warning_codes),
                        "error_code": run.error_code,
                        "object_ref": run.object_ref,
                        "original_filename": audio.original_filename,
                        "detected_mime": audio.detected_mime,
                        "size_bytes": audio.size_bytes,
                        "created_at": run.created_at,
                        "updated_at": run.updated_at,
                    },
                )
            return tuple(history)


def _view(run: Any, job_id: str) -> AudioRunView:
    return AudioRunView(
        run_id=run.run_id,
        job_id=job_id,
        status=run.status.value,
        current_stage=run.current_stage,
        warning_codes=tuple(run.warning_codes),
        error_code=run.error_code,
    )


def _job_view(job: JobModel) -> AudioJobView:
    return AudioJobView(
        job_id=job.job_id,
        resource_id=job.resource_id,
        status=job.status,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        error_code=job.error_code,
    )
