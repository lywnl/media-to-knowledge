"""音频转写与 LLM 阶段的持久化租约和恢复查询。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import exists, or_, select, update
from sqlalchemy.orm import Session, aliased

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import (
    AudioPipelineStageModel,
    AudioStageName,
    JobStatus,
)
from video_demo.persistence.scope import Scope


@dataclass(frozen=True, slots=True)
class AudioStageRecord:
    scope: Scope
    run_id: str
    stage_name: AudioStageName
    status: JobStatus
    attempt_count: int
    max_attempts: int
    checkpoint_relative_path: str | None
    checkpoint_sha256: str | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class AudioStageLease:
    id: int
    scope: Scope
    run_id: str
    stage_name: AudioStageName
    worker_id: str
    attempt_count: int
    max_attempts: int


class AudioStageRepository:
    """只操作 ``audio_pipeline_stage`` 的阶段租约仓储。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure(self, scope: Scope, run_id: str, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("阶段最大尝试次数必须大于 0")
        now = datetime.now(UTC)
        for stage_name in (AudioStageName.TRANSCRIPTION, AudioStageName.LLM):
            existing = self._session.scalar(
                select(AudioPipelineStageModel).where(
                    AudioPipelineStageModel.tenant_id == scope.tenant_id,
                    AudioPipelineStageModel.application_id == scope.application_id,
                    AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                    AudioPipelineStageModel.run_id == run_id,
                    AudioPipelineStageModel.stage_name == stage_name,
                ),
            )
            if existing is None:
                self._session.add(
                    AudioPipelineStageModel(
                        tenant_id=scope.tenant_id,
                        application_id=scope.application_id,
                        knowledge_base_id=scope.knowledge_base_id,
                        run_id=run_id,
                        stage_name=stage_name,
                        status=JobStatus.PENDING,
                        attempt_count=0,
                        max_attempts=max_attempts,
                        next_attempt_at=now,
                    ),
                )
        self._session.flush()

    def get(
        self,
        scope: Scope,
        run_id: str,
        stage_name: AudioStageName,
    ) -> AudioStageRecord | None:
        model = self._session.scalar(
            select(AudioPipelineStageModel).where(
                AudioPipelineStageModel.tenant_id == scope.tenant_id,
                AudioPipelineStageModel.application_id == scope.application_id,
                AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                AudioPipelineStageModel.run_id == run_id,
                AudioPipelineStageModel.stage_name == stage_name,
            ),
        )
        return _record(model) if model is not None else None

    def list_recoverable(self, *, now: datetime | None = None) -> tuple[AudioStageRecord, ...]:
        current_time = now or datetime.now(UTC)
        transcription = aliased(AudioPipelineStageModel)
        transcription_ready = exists(
            select(transcription.id).where(
                transcription.tenant_id == AudioPipelineStageModel.tenant_id,
                transcription.application_id == AudioPipelineStageModel.application_id,
                transcription.knowledge_base_id == AudioPipelineStageModel.knowledge_base_id,
                transcription.run_id == AudioPipelineStageModel.run_id,
                transcription.stage_name == AudioStageName.TRANSCRIPTION,
                transcription.status == JobStatus.SUCCEEDED,
            ),
        )
        models = self._session.scalars(
            select(AudioPipelineStageModel).where(
                or_(
                    (
                        AudioPipelineStageModel.status.in_(
                            (JobStatus.PENDING, JobStatus.RETRY_WAIT),
                        )
                        & (AudioPipelineStageModel.next_attempt_at <= current_time)
                    ),
                    (
                        (AudioPipelineStageModel.status == JobStatus.RUNNING)
                        & (AudioPipelineStageModel.lease_expires_at <= current_time)
                    ),
                ),
                or_(
                    AudioPipelineStageModel.stage_name != AudioStageName.LLM,
                    transcription_ready,
                ),
            ).order_by(AudioPipelineStageModel.id),
        )
        return tuple(_record(model) for model in models)

    def claim(
        self,
        scope: Scope,
        run_id: str,
        stage_name: AudioStageName,
        worker_id: str,
        *,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> AudioStageLease | None:
        if lease_seconds < 1 or not worker_id.strip():
            raise ValueError("音频阶段租约参数非法")
        current_time = now or datetime.now(UTC)
        model = self._session.scalar(
            select(AudioPipelineStageModel).where(
                AudioPipelineStageModel.tenant_id == scope.tenant_id,
                AudioPipelineStageModel.application_id == scope.application_id,
                AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                AudioPipelineStageModel.run_id == run_id,
                AudioPipelineStageModel.stage_name == stage_name,
                AudioPipelineStageModel.attempt_count < AudioPipelineStageModel.max_attempts,
                or_(
                    (
                        AudioPipelineStageModel.status.in_(
                            (JobStatus.PENDING, JobStatus.RETRY_WAIT),
                        )
                        & (AudioPipelineStageModel.next_attempt_at <= current_time)
                    ),
                    (
                        (AudioPipelineStageModel.status == JobStatus.RUNNING)
                        & (AudioPipelineStageModel.lease_expires_at <= current_time)
                    ),
                ),
            ).limit(1),
        )
        if model is None:
            return None
        expected_status = model.status
        statement = update(AudioPipelineStageModel).where(
            AudioPipelineStageModel.id == model.id,
            AudioPipelineStageModel.status == expected_status,
            AudioPipelineStageModel.attempt_count == model.attempt_count,
        )
        if expected_status == JobStatus.RUNNING:
            statement = statement.where(AudioPipelineStageModel.lease_expires_at <= current_time)
        else:
            statement = statement.where(AudioPipelineStageModel.next_attempt_at <= current_time)
        result = self._session.execute(
            statement.values(
                status=JobStatus.RUNNING,
                worker_id=worker_id,
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                attempt_count=AudioPipelineStageModel.attempt_count + 1,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            return None
        self._session.expire_all()
        claimed = self._session.get(AudioPipelineStageModel, model.id)
        assert claimed is not None
        return AudioStageLease(
            id=claimed.id,
            scope=scope,
            run_id=run_id,
            stage_name=stage_name,
            worker_id=worker_id,
            attempt_count=claimed.attempt_count,
            max_attempts=claimed.max_attempts,
        )

    def heartbeat(
        self,
        lease: AudioStageLease,
        *,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("阶段租约时长必须大于 0")
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.id == lease.id,
                AudioPipelineStageModel.worker_id == lease.worker_id,
                AudioPipelineStageModel.attempt_count == lease.attempt_count,
                AudioPipelineStageModel.status == JobStatus.RUNNING,
                AudioPipelineStageModel.lease_expires_at > current_time,
            ).values(
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频阶段租约已丢失")

    def mark_succeeded(
        self,
        lease: AudioStageLease,
        *,
        checkpoint_relative_path: str | None = None,
        checkpoint_sha256: str | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.id == lease.id,
                AudioPipelineStageModel.worker_id == lease.worker_id,
                AudioPipelineStageModel.attempt_count == lease.attempt_count,
                AudioPipelineStageModel.status == JobStatus.RUNNING,
            ).values(
                status=JobStatus.SUCCEEDED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                checkpoint_relative_path=checkpoint_relative_path,
                checkpoint_sha256=checkpoint_sha256,
                error_code=None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频阶段租约已丢失")

    def mark_failed(
        self,
        lease: AudioStageLease,
        *,
        error_code: ErrorCode,
        retryable: bool,
        retry_delay_seconds: int = 5,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        status = (
            JobStatus.RETRY_WAIT
            if retryable and lease.attempt_count < lease.max_attempts
            else JobStatus.FAILED
        )
        result = self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.id == lease.id,
                AudioPipelineStageModel.worker_id == lease.worker_id,
                AudioPipelineStageModel.attempt_count == lease.attempt_count,
                AudioPipelineStageModel.status == JobStatus.RUNNING,
            ).values(
                status=status,
                next_attempt_at=current_time + timedelta(seconds=retry_delay_seconds),
                error_code=str(error_code),
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频阶段租约已丢失")

    def mark_cancelled(self, lease: AudioStageLease, *, now: datetime | None = None) -> None:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.id == lease.id,
                AudioPipelineStageModel.worker_id == lease.worker_id,
                AudioPipelineStageModel.attempt_count == lease.attempt_count,
                AudioPipelineStageModel.status == JobStatus.RUNNING,
            ).values(
                status=JobStatus.CANCELLED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=str(ErrorCode.JOB_CANCELLED),
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            return
        already_cancelled = self._session.scalar(
            select(AudioPipelineStageModel.id).where(
                AudioPipelineStageModel.id == lease.id,
                AudioPipelineStageModel.attempt_count == lease.attempt_count,
                AudioPipelineStageModel.status == JobStatus.CANCELLED,
            ),
        )
        if already_cancelled is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频阶段租约已丢失")

    def reconcile_succeeded(
        self,
        scope: Scope,
        run_id: str,
        stage_name: AudioStageName,
        *,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.tenant_id == scope.tenant_id,
                AudioPipelineStageModel.application_id == scope.application_id,
                AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                AudioPipelineStageModel.run_id == run_id,
                AudioPipelineStageModel.stage_name == stage_name,
                AudioPipelineStageModel.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
            ).values(
                status=JobStatus.SUCCEEDED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )

    def reset_for_retry(
        self,
        scope: Scope,
        run_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[AudioStageName, ...]:
        current_time = now or datetime.now(UTC)
        failed = tuple(
            self._session.scalars(
                select(AudioPipelineStageModel.stage_name).where(
                    AudioPipelineStageModel.tenant_id == scope.tenant_id,
                    AudioPipelineStageModel.application_id == scope.application_id,
                    AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                    AudioPipelineStageModel.run_id == run_id,
                    AudioPipelineStageModel.status == JobStatus.FAILED,
                ).order_by(AudioPipelineStageModel.id),
            ),
        )
        if not failed:
            return ()
        self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.tenant_id == scope.tenant_id,
                AudioPipelineStageModel.application_id == scope.application_id,
                AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                AudioPipelineStageModel.run_id == run_id,
                AudioPipelineStageModel.status == JobStatus.FAILED,
            ).values(
                status=JobStatus.PENDING,
                attempt_count=0,
                next_attempt_at=current_time,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        return failed

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage_name: AudioStageName,
        *,
        error_code: ErrorCode,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(AudioPipelineStageModel).where(
                AudioPipelineStageModel.tenant_id == scope.tenant_id,
                AudioPipelineStageModel.application_id == scope.application_id,
                AudioPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                AudioPipelineStageModel.run_id == run_id,
                AudioPipelineStageModel.stage_name == stage_name,
                AudioPipelineStageModel.status.in_(
                    (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.RUNNING),
                ),
            ).values(
                status=JobStatus.FAILED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=str(error_code),
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


def _record(model: AudioPipelineStageModel) -> AudioStageRecord:
    return AudioStageRecord(
        scope=Scope(model.tenant_id, model.application_id, model.knowledge_base_id),
        run_id=model.run_id,
        stage_name=model.stage_name,
        status=model.status,
        attempt_count=model.attempt_count,
        max_attempts=model.max_attempts,
        checkpoint_relative_path=model.checkpoint_relative_path,
        checkpoint_sha256=model.checkpoint_sha256,
        error_code=model.error_code,
    )


__all__ = ["AudioStageLease", "AudioStageRecord", "AudioStageRepository"]
