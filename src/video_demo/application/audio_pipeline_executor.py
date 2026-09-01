"""音频阶段租约执行器。

该执行器只编排音频预检、转码、ASR、章节写作和音频结果发布；它不依赖
视频流水线或视觉阶段。阶段状态和总任务租约分别写入音频阶段表和音频任务。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from typing import Protocol, TypeAlias, cast

from video_demo.application.audio_contracts import AudioTranscriptionCheckpoint
from video_demo.application.audio_pipeline import AudioPipeline, AudioPipelineOutcome
from video_demo.application.audio_publication import AudioPublicationService
from video_demo.application.audio_run_config import AudioRunConfig
from video_demo.application.checkpoint_contracts import (
    CheckpointStaleError,
    cleanup_stale_checkpoint_artifacts,
)
from video_demo.application.publication_contracts import ResultWriteFence, scope_key
from video_demo.domain.title import sanitize_document_title
from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.media.audio_format import AUDIO_FORMAT_VERSION
from video_demo.persistence.audio_stage_repository import (
    AudioStageLease,
    AudioStageRepository,
)
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository, MediaRunRepository
from video_demo.persistence.models import (
    AudioObjectModel,
    AudioStageName,
    AudioUnderstandingRunModel,
    JobStatus,
    RunStatusValue,
)
from video_demo.persistence.repositories import ClaimedJob, JobRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.audio_object_store import AudioObjectStore
from video_demo.storage.document_cache import DocumentModelCache
from video_demo.storage.media_object_store import MediaObjectRecord

AudioPipelineFactory: TypeAlias = Callable[
    [int, Callable[[], bool]],
    AudioPipeline,
]
AudioTranscoderFactory: TypeAlias = Callable[[Callable[[], bool]], "AudioTranscoder"]


class AudioProbe(Protocol):
    def probe(self, source: Path, *, max_duration_ms: int) -> object: ...


class AudioTranscoder(Protocol):
    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool,
        duration_ms: int,
    ) -> object: ...


class AudioStagePipelineExecutor:
    """执行音频 TRANSCRIPTION/LLM 两阶段，并维护音频专用租约。"""

    def __init__(
        self,
        database: Database,
        pipeline_factory: AudioPipelineFactory,
        probe: AudioProbe,
        transcoder: AudioTranscoder | AudioTranscoderFactory,
        object_store: AudioObjectStore,
        runtime_root: Path,
        publication: AudioPublicationService | None = None,
        *,
        max_duration_ms: int = 7_200_000,
        max_cache_entry_bytes: int = 8 * 1024 * 1024,
        max_cache_run_bytes: int = 64 * 1024 * 1024,
        lease_seconds: int = 120,
        owned_resources: tuple[object, ...] = (),
    ) -> None:
        if max_duration_ms < 1 or lease_seconds < 1:
            raise ValueError("音频执行器时长和租约参数必须大于 0")
        self._database = database
        self._pipeline_factory = pipeline_factory
        self._probe = probe
        self._transcoder = transcoder
        self._object_store = object_store
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._publication = publication
        self._max_duration_ms = max_duration_ms
        self._max_cache_entry_bytes = max_cache_entry_bytes
        self._max_cache_run_bytes = max_cache_run_bytes
        self._lease_seconds = lease_seconds
        self._leases: dict[tuple[str, str, str], AudioStageLease] = {}
        self._jobs: dict[tuple[str, str, str], ClaimedJob] = {}
        self._owned_resources = owned_resources
        self._closed = False

    def run_transcription(self, scope: Scope, run_id: str) -> AudioTranscriptionCheckpoint:
        lease = self._claim(scope, run_id, AudioStageName.TRANSCRIPTION)
        job = self._claim_job(scope, run_id, lease.worker_id)
        key = self._lease_key(scope, run_id, AudioStageName.TRANSCRIPTION)
        self._jobs[key] = job
        checkpoint = cast(
            AudioTranscriptionCheckpoint,
            self._run_with_heartbeat(
                lease,
                job,
                lambda: self._run_transcription_body(scope, run_id, job),
            ),
        )
        self._validate_checkpoint(
            checkpoint,
            run_id,
            checkpoint.asset_sha256,
            checkpoint.duration_ms,
        )
        run_relative_root = Path("runs") / scope_key(scope) / run_id
        relative = run_relative_root / "stages" / "transcription-checkpoint.json"
        receipt = AtomicArtifactStore(self._runtime_root).write_json(
            relative,
            _checkpoint_payload(checkpoint),
            schema_version="1.0.0",
            upstream_sha256=checkpoint.asset_sha256,
            file_mode=0o600,
            exclusive=False,
            max_bytes=16 * 1024 * 1024,
        )
        self._complete_transcription(lease, job, receipt.relative_path, receipt.sha256)
        return checkpoint

    def _run_transcription_body(
        self,
        scope: Scope,
        run_id: str,
        job: ClaimedJob,
    ) -> AudioTranscriptionCheckpoint:
        """在同一阶段心跳内完成输入加载、预检、转码和音频转写。"""

        source, config, title, asset_sha256 = self._load_input(scope, run_id)
        probe = self._probe.probe(source, max_duration_ms=self._max_duration_ms)
        duration_ms = _duration_ms(probe)
        run_relative_root = Path("runs") / scope_key(scope) / run_id
        run_root = self._runtime_root / run_relative_root
        run_root.mkdir(parents=True, exist_ok=True)
        self._update_stage(scope, run_id, "PROBE")
        artifact = self._transcoder_for(job).extract_audio(
            source,
            run_relative_root,
            has_audio=True,
            duration_ms=duration_ms,
        )
        if _is_no_audio_artifact(artifact):
            raise VideoDemoError(ErrorCode.AUDIO_ASR_UNAVAILABLE, "音频没有可用音轨")
        audio_source = self._artifact_path(run_relative_root, artifact)
        self._update_stage(scope, run_id, "TRANSCRIPTION")
        result = self._pipeline_factory(
            duration_ms,
            lambda: self._is_job_cancelled(job),
        ).run_transcription(
            run_id=run_id,
            asset_sha256=asset_sha256,
            source=audio_source,
            duration_ms=duration_ms,
            title_hint=title,
            config=config,
            run_root=run_relative_root,
            is_cancel_requested=lambda: self._is_job_cancelled(job),
        )
        if not isinstance(result, AudioTranscriptionCheckpoint):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频转写阶段结果类型非法")
        return result

    def run_llm(self, scope: Scope, run_id: str, checkpoint: object) -> None:
        if not isinstance(checkpoint, AudioTranscriptionCheckpoint):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频转写快照类型非法")
        lease = self._claim(scope, run_id, AudioStageName.LLM)
        try:
            job = self._claim_job(scope, run_id, lease.worker_id)
        except VideoDemoError as error:
            if error.code != ErrorCode.JOB_LEASE_LOST or not self._result_exists(scope, run_id):
                raise
            self._reconcile_published_stage(scope, run_id, lease)
            return
        key = self._lease_key(scope, run_id, AudioStageName.LLM)
        self._jobs[key] = job
        _config, _title, _asset_sha256 = self._load_run_config(scope, run_id)
        run_root = self._runtime_root / (Path("runs") / scope_key(scope) / run_id)
        cache = DocumentModelCache(
            run_root,
            max_entry_bytes=self._max_cache_entry_bytes,
            max_run_bytes=self._max_cache_run_bytes,
        )
        outcome = self._run_with_heartbeat(
            lease,
            job,
            lambda: self._pipeline_factory(
                checkpoint.duration_ms,
                lambda: self._is_job_cancelled(job),
            ).run_llm(
                checkpoint,
                config=_config,
                cache=cache,
                is_cancel_requested=lambda: self._is_job_cancelled(job),
            ),
        )
        if not isinstance(outcome, AudioPipelineOutcome):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "音频 LLM 阶段结果类型非法")
        if self._publication is None:
            raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "音频结果发布器未配置")
        self._publication.persist(
            scope,
            outcome.result,
            document=outcome.document,
            status=outcome.status,
            warnings=outcome.warnings,
            fence=ResultWriteFence(job.id, job.worker_id, job.attempt_count),
        )
        self._mark_stage_succeeded(lease)

    def load_checkpoint(
        self,
        scope: Scope,
        run_id: str,
    ) -> AudioTranscriptionCheckpoint | None:
        with self._database.session() as session:
            record = AudioStageRepository(session).get(
                scope,
                run_id,
                AudioStageName.TRANSCRIPTION,
            )
        if record is None or not record.checkpoint_relative_path or not record.checkpoint_sha256:
            return None
        try:
            from video_demo.application.audio_contracts import (
                audio_transcription_checkpoint_from_payload,
            )

            store = AtomicArtifactStore(self._runtime_root)
            payload = store.read_verified_json_limited(
                ArtifactReceipt(
                    relative_path=record.checkpoint_relative_path,
                    schema_version="1.0.0",
                    sha256=record.checkpoint_sha256,
                    upstream_sha256=self._asset_sha256(scope, run_id),
                ),
                max_bytes=16 * 1024 * 1024,
            )
            if not isinstance(payload, dict):
                raise ValueError("音频转写快照 payload 非法")
            if _is_stale_audio_checkpoint(payload):
                raise CheckpointStaleError()
            checkpoint = audio_transcription_checkpoint_from_payload(payload)
            self._validate_checkpoint(
                checkpoint,
                run_id,
                self._asset_sha256(scope, run_id),
                checkpoint.duration_ms,
            )
            return checkpoint
        except CheckpointStaleError:
            raise
        except (OSError, ValueError, TypeError, VideoDemoError) as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "音频转写快照无法恢复",
            ) from error

    def reset_stale_checkpoint(self, scope: Scope, run_id: str) -> None:
        with self._database.session() as session:
            repository = AudioStageRepository(session)
            record = repository.get(scope, run_id, AudioStageName.TRANSCRIPTION)
            repository.reset_stale_checkpoint(scope, run_id)
        cleanup_stale_checkpoint_artifacts(
            self._runtime_root,
            Path("runs") / scope_key(scope) / run_id,
            record.checkpoint_relative_path if record is not None else None,
        )

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> None:
        stage_name = AudioStageName(stage)
        with self._database.session() as session:
            changed = AudioStageRepository(session).mark_recovery_failed(
                scope,
                run_id,
                stage_name,
                error_code=error.code,
            )
            if changed:
                JobRepository(session).fail_unclaimed_audio_run(
                    scope,
                    run_id,
                    error_code=error.code,
                    current_stage=stage,
                )

    def mark_stage_started(self, scope: Scope, run_id: str, stage: str) -> None:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is not None:
                run.status = RunStatusValue.RUNNING
                run.current_stage = stage
                run.error_code = None

    def stage_succeeded(self, scope: Scope, run_id: str, stage: str, result: object) -> None:
        del scope, run_id, stage, result

    def stage_failed(self, scope: Scope, run_id: str, stage: str, error: VideoDemoError) -> bool:
        stage_name = AudioStageName(stage)
        key = self._lease_key(scope, run_id, stage_name)
        lease = self._leases.pop(key, None)
        job = self._jobs.pop(key, None)
        if lease is None:
            return False
        if (
            error.code == ErrorCode.JOB_LEASE_LOST
            and stage_name == AudioStageName.LLM
            and self._result_exists(scope, run_id)
        ):
            self._reconcile_published_stage(scope, run_id, lease)
            return False
        with self._database.session() as session:
            repository = AudioStageRepository(session)
            if error.code == ErrorCode.JOB_CANCELLED:
                repository.mark_cancelled(lease)
                if job is not None:
                    JobRepository(session).mark_cancelled(
                        job.id,
                        job.worker_id,
                        attempt_count=job.attempt_count,
                    )
                return False
            retryable = is_retryable_error_code(error.code)
            repository.mark_failed(lease, error_code=error.code, retryable=retryable)
            if job is not None:
                JobRepository(session).release_audio_stage(
                    job,
                    status=JobStatus.RETRY_WAIT if retryable else JobStatus.FAILED,
                    current_stage=stage,
                    error_code=error.code,
                )
            return retryable and lease.attempt_count < lease.max_attempts

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        with self._database.session() as session:
            job = JobRepository(session).get_by_resource_type(
                scope,
                run_id,
                "AUDIO_UNDERSTANDING_RUN",
            )
            return job is not None and job.status == JobStatus.CANCELLED

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def _claim(self, scope: Scope, run_id: str, stage: AudioStageName) -> AudioStageLease:
        worker_id = f"audio-api-{uuid.uuid4().hex}"
        with self._database.session() as session:
            lease = AudioStageRepository(session).claim(
                scope,
                run_id,
                stage,
                worker_id,
                lease_seconds=self._lease_seconds,
            )
        if lease is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频阶段无法领取")
        self._leases[self._lease_key(scope, run_id, stage)] = lease
        return lease

    def _claim_job(self, scope: Scope, run_id: str, worker_id: str) -> ClaimedJob:
        with self._database.session() as session:
            job = JobRepository(session).claim_audio_run(
                scope,
                run_id,
                worker_id,
                lease_seconds=self._lease_seconds,
            )
        if job is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频总任务无法领取")
        return job

    def _load_input(self, scope: Scope, run_id: str) -> tuple[Path, AudioRunConfig, str, str]:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            obj = MediaObjectRepository(session, AudioObjectModel).get_ready(scope, run.object_ref)
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
                scope_key=self._object_store.scope_key(scope),
            )
            source = self._object_store.materialize(scope, record, run_id, obj.sha256)
            config = _parse_config(run.config_snapshot)
            title = sanitize_document_title(
                config.document_config.document_title,
                obj.original_filename,
            )
            return source, config, title or "音频知识文档", obj.sha256

    def _load_run_config(self, scope: Scope, run_id: str) -> tuple[AudioRunConfig, str, str]:
        _source, config, title, asset_sha256 = self._load_input(scope, run_id)
        return config, title, asset_sha256

    def _transcoder_for(self, job: ClaimedJob) -> AudioTranscoder:
        if callable(self._transcoder) and not hasattr(self._transcoder, "extract_audio"):
            return self._transcoder(lambda: self._is_job_cancelled(job))
        return cast(AudioTranscoder, self._transcoder)

    def _artifact_path(self, run_relative_root: Path, artifact: object) -> Path:
        relative = getattr(artifact, "relative_path", None)
        if not isinstance(relative, str) or not relative:
            raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_INVALID, "音频转码制品路径非法")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_INVALID, "音频转码制品路径非法")
        output = path if path.is_relative_to(run_relative_root) else run_relative_root / path
        resolved = (self._runtime_root / output).resolve(strict=False)
        run_root = (self._runtime_root / run_relative_root).resolve(strict=False)
        if not resolved.is_relative_to(run_root) or resolved.is_symlink() or not resolved.is_file():
            raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_INVALID, "音频转码制品不存在")
        return resolved

    def _complete_transcription(
        self,
        lease: AudioStageLease,
        job: ClaimedJob,
        path: str,
        digest: str,
    ) -> None:
        key = self._lease_key(lease.scope, lease.run_id, lease.stage_name)
        with self._database.session() as session:
            AudioStageRepository(session).mark_succeeded(
                lease,
                checkpoint_relative_path=path,
                checkpoint_sha256=digest,
            )
            JobRepository(session).release_audio_stage(
                job,
                status=JobStatus.PENDING,
                current_stage=AudioStageName.LLM.value,
            )
        self._leases.pop(key, None)
        self._jobs.pop(key, None)

    def _mark_stage_succeeded(self, lease: AudioStageLease) -> None:
        key = self._lease_key(lease.scope, lease.run_id, lease.stage_name)
        try:
            with self._database.session() as session:
                AudioStageRepository(session).mark_succeeded(lease)
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
        lease: AudioStageLease,
    ) -> None:
        with self._database.session() as session:
            AudioStageRepository(session).reconcile_succeeded(scope, run_id, lease.stage_name)

    def _run_with_heartbeat(
        self,
        lease: AudioStageLease,
        job: ClaimedJob,
        operation: Callable[[], object],
    ) -> object:
        stop = Event()
        errors: list[VideoDemoError] = []

        def heartbeat() -> None:
            while not stop.wait(max(0.5, self._lease_seconds / 3)):
                try:
                    with self._database.session() as session:
                        AudioStageRepository(session).heartbeat(
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

        thread = Thread(target=heartbeat, name=f"audio-stage-heartbeat-{lease.run_id}", daemon=True)
        thread.start()
        try:
            result = operation()
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]
        return result

    def _is_job_cancelled(self, job: ClaimedJob) -> bool:
        with self._database.session() as session:
            return JobRepository(session).is_cancel_requested(
                job.id,
                job.worker_id,
                attempt_count=job.attempt_count,
            )

    def _update_stage(self, scope: Scope, run_id: str, stage: str) -> None:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is not None:
                run.current_stage = stage

    def _asset_sha256(self, scope: Scope, run_id: str) -> str:
        with self._database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            audio = MediaObjectRepository(session, AudioObjectModel).get(scope, run.object_ref)
            if audio is None:
                raise VideoDemoError(ErrorCode.AUDIO_OBJECT_NOT_FOUND, "音频对象不存在")
            return str(audio.sha256)

    def _validate_checkpoint(
        self,
        checkpoint: AudioTranscriptionCheckpoint,
        run_id: str,
        asset_sha256: str,
        duration_ms: int,
    ) -> None:
        if checkpoint.run_id != run_id or checkpoint.asset_sha256 != asset_sha256:
            raise VideoDemoError(ErrorCode.AUDIO_DIGEST_MISMATCH, "音频转写结果资产不匹配")
        if checkpoint.duration_ms != duration_ms:
            raise VideoDemoError(ErrorCode.AUDIO_PROBE_INVALID, "音频转写结果时长不匹配")
        checkpoint.validate_consistency()

    def _result_exists(self, scope: Scope, run_id: str) -> bool:
        if self._publication is None:
            return False
        try:
            self._publication.get(scope, run_id)
        except VideoDemoError:
            return False
        return True

    @staticmethod
    def _lease_key(scope: Scope, run_id: str, stage: AudioStageName | str) -> tuple[str, str, str]:
        return (
            scope.tenant_id + "\x00" + scope.application_id,
            scope.knowledge_base_id + "\x00" + run_id,
            str(stage),
        )


def _duration_ms(probe: object) -> int:
    value = getattr(probe, "duration_ms", None)
    if not isinstance(value, int) or value < 1:
        raise VideoDemoError(ErrorCode.AUDIO_PROBE_INVALID, "音频预检时长非法")
    return value


def _is_stale_audio_checkpoint(payload: dict[str, object]) -> bool:
    if payload.get("schema_version") != "2.0.0":
        return True
    if payload.get("audio_format_version") != AUDIO_FORMAT_VERSION:
        return True
    audio_path = payload.get("audio_path")
    return isinstance(audio_path, str) and not audio_path.casefold().endswith(".mp3")


def _parse_config(snapshot: object) -> AudioRunConfig:
    try:
        return AudioRunConfig.model_validate(snapshot)
    except ValueError as error:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "音频运行配置非法") from error


def _checkpoint_payload(checkpoint: object) -> dict[str, object]:
    from video_demo.application.audio_contracts import audio_transcription_checkpoint_to_payload

    return audio_transcription_checkpoint_to_payload(checkpoint)  # type: ignore[arg-type]


def _is_no_audio_artifact(value: object) -> bool:
    return value.__class__.__name__ == "NoAudioArtifact"


__all__ = ["AudioStagePipelineExecutor"]
