from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from video_demo.application.audio_pipeline import AudioPipeline
from video_demo.application.document_publication import ResultWriteFence
from video_demo.application.image_pipeline import ImageAnalyzer, run_image_pipeline
from video_demo.application.media_publication import MediaPublicationService
from video_demo.application.pipeline_contracts import PipelineRunConfig
from video_demo.application.production_media import TranscodeClient
from video_demo.domain.document import sanitize_document_title
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.media.audio_probe import AudioProbeClient
from video_demo.media.transcode import NoAudioArtifact
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import (
    MediaObjectRepository,
    MediaRunRepository,
)
from video_demo.persistence.models import (
    AudioObjectModel,
    AudioUnderstandingRunModel,
    ImageObjectModel,
    ImageUnderstandingRunModel,
    RunStatusValue,
)
from video_demo.persistence.repositories import ClaimedJob, JobRepository, Scope
from video_demo.storage.audio_object_store import AudioObjectStore
from video_demo.storage.document_cache import DocumentModelCache
from video_demo.storage.image_object_store import ImageObjectStore
from video_demo.storage.media_object_store import MediaObjectRecord


class AudioPipelineFactory(Protocol):
    def __call__(self, duration_ms: int) -> AudioPipeline: ...


class AudioTranscoderFactory(Protocol):
    def __call__(self, is_cancel_requested: Callable[[], bool]) -> TranscodeClient: ...


def audio_run_scope_key(scope: Scope) -> str:
    return _scope_key(scope)


class AudioJobHandler:
    """领取音频任务，执行独立 ASR/章节流水线并发布 Markdown。"""

    def __init__(
        self,
        database: Database,
        probe: AudioProbeClient,
        pipeline_factory: AudioPipelineFactory,
        publication: MediaPublicationService,
        object_store: AudioObjectStore,
        transcoder_factory: AudioTranscoderFactory,
        *,
        runtime_root: Path,
        max_duration_ms: int,
        max_cache_entry_bytes: int,
        max_cache_run_bytes: int,
    ) -> None:
        self._database = database
        self._probe = probe
        self._pipeline_factory = pipeline_factory
        self._publication = publication
        self._object_store = object_store
        self._transcoder_factory = transcoder_factory
        self._runtime_root = runtime_root
        self._max_duration_ms = max_duration_ms
        self._max_cache_entry_bytes = max_cache_entry_bytes
        self._max_cache_run_bytes = max_cache_run_bytes

    def __call__(self, job: ClaimedJob) -> None:
        self._mark_running(job)
        try:
            source, config, filename, asset_sha = self._load_input(job)
            probe = self._probe.probe(source, max_duration_ms=self._max_duration_ms)
            self._mark_stage(job, "SPEECH")
            run_relative_root = Path("runs") / _scope_key(job.scope) / job.resource_id
            run_root = self._runtime_root / run_relative_root
            run_root.mkdir(parents=True, exist_ok=True)
            transcoder = self._transcoder_factory(lambda: self._is_cancel_requested(job))
            audio_artifact = transcoder.extract_audio(
                source,
                run_relative_root,
                has_audio=True,
                duration_ms=probe.duration_ms,
            )
            if isinstance(audio_artifact, NoAudioArtifact):
                raise VideoDemoError(ErrorCode.AUDIO_ASR_UNAVAILABLE, "音频没有可用音轨")
            source = self._runtime_root / audio_artifact.relative_path
            cache = DocumentModelCache(
                run_root,
                max_entry_bytes=self._max_cache_entry_bytes,
                max_run_bytes=self._max_cache_run_bytes,
            )
            title = sanitize_document_title(config.document_config.document_title, filename)
            outcome = self._pipeline_factory(probe.duration_ms).run(
                run_id=job.resource_id,
                asset_sha256=asset_sha,
                source=source,
                duration_ms=probe.duration_ms,
                title_hint=title or "音频知识文档",
                config=config,
                run_root=run_relative_root,
                cache=cache,
                is_cancel_requested=lambda: self._is_cancel_requested(job),
            )
            self._publication.persist(
                job.scope,
                outcome.result,
                document=outcome.document,
                status=outcome.status,
                warnings=outcome.warnings,
                fence=_fence(job),
            )
        except VideoDemoError as error:
            self._mark_unsuccessful(job, error)
            raise
        except MemoryError as memory_error:
            failure = VideoDemoError(ErrorCode.OUT_OF_MEMORY, "任务内存不足")
            self._mark_unsuccessful(job, failure)
            raise failure from memory_error
        except Exception as system_error:
            failure = VideoDemoError(ErrorCode.SYSTEM_FAILURE, "音频任务发生未分类系统错误")
            self._mark_unsuccessful(job, failure)
            raise failure from system_error

    def _load_input(self, job: ClaimedJob) -> tuple[Path, PipelineRunConfig, str, str]:
        with self._database.session() as session:
            run = MediaRunRepository(
                session,
                AudioUnderstandingRunModel,
            ).get(job.scope, job.resource_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            obj = MediaObjectRepository(session, AudioObjectModel).get_ready(
                job.scope,
                run.object_ref,
            )
            if obj is None:
                raise VideoDemoError(ErrorCode.AUDIO_OBJECT_NOT_FOUND, "音频对象不存在")
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
                raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "音频运行配置非法") from error
            return source, config, obj.original_filename, obj.sha256

    def _mark_running(self, job: ClaimedJob) -> None:
        with self._database.session() as session:
            JobRepository(session).update_owned_media_run(
                job,
                values={
                    "status": RunStatusValue.RUNNING,
                    "current_stage": "PROBE",
                    "error_code": None,
                },
                run_model=AudioUnderstandingRunModel,
                resource_type="AUDIO_UNDERSTANDING_RUN",
            )

    def _mark_stage(self, job: ClaimedJob, stage: str) -> None:
        with self._database.session() as session:
            JobRepository(session).update_owned_media_run(
                job,
                values={"current_stage": stage},
                run_model=AudioUnderstandingRunModel,
                resource_type="AUDIO_UNDERSTANDING_RUN",
            )

    def _is_cancel_requested(self, job: ClaimedJob) -> bool:
        with self._database.session() as session:
            return JobRepository(session).is_cancel_requested(
                job.id,
                job.worker_id,
                attempt_count=job.attempt_count,
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
                run_model=AudioUnderstandingRunModel,
                resource_type="AUDIO_UNDERSTANDING_RUN",
            )
            repository.mark_failed(
                job.id,
                job.worker_id,
                error_code=error.code,
                retryable=is_retryable_error_code(error.code),
                attempt_count=job.attempt_count,
            )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ImageAnalyzerFactory(Protocol):
    def __call__(self) -> ImageAnalyzer: ...


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
