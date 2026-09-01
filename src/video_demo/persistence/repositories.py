from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import Any, cast

from sqlalchemy import Select, and_, exists, func, or_, select, update
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement

import video_demo.domain as _domain_package
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import (
    JobModel,
    JobStatus,
    RunStatusValue,
    VideoAssetModel,
    VideoObjectModel,
    VideoObjectStatus,
    VideoPipelineStageModel,
    VideoStageName,
    VideoSummaryModel,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.scope import Scope

_SENSITIVE_KEY_PARTS = ("secret", "token", "api_key", "apikey", "authorization", "password")

# 保持生产导入闭包对领域包的历史可见性；不触发任何视频契约加载。
_ = _domain_package


def __getattr__(name: str) -> object:
    if name == "ResultRepository":
        value = getattr(import_module("video_demo.persistence.document_repository"), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class PublishedRunCleanupRecord:
    run_pk: int
    scope: Scope
    run_id: str


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: int
    job_id: str
    resource_id: str
    worker_id: str
    attempt_count: int
    max_attempts: int
    scope: Scope


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    job_id: str
    object_ref: str
    status: RunStatusValue
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


@dataclass(frozen=True, slots=True)
class MediaObjectRecord:
    object_ref: str
    original_filename: str
    declared_mime: str
    detected_mime: str
    size_bytes: int
    sha256: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class MediaRunRecord:
    run_id: str
    object_ref: str
    status: RunStatusValue
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


@dataclass(frozen=True, slots=True)
class VideoStageRecord:
    scope: Scope
    run_id: str
    stage_name: VideoStageName
    status: JobStatus
    attempt_count: int
    max_attempts: int
    checkpoint_relative_path: str | None
    checkpoint_sha256: str | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class VideoStageLease:
    id: int
    scope: Scope
    run_id: str
    stage_name: VideoStageName
    worker_id: str
    attempt_count: int
    max_attempts: int


class VideoStageRepository:
    """视频转写与 LLM 阶段的持久化租约和恢复查询。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure(self, scope: Scope, run_id: str, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("阶段最大尝试次数必须大于 0")
        now = datetime.now(UTC)
        for stage_name in (VideoStageName.TRANSCRIPTION, VideoStageName.LLM):
            existing = self._session.scalar(
                select(VideoPipelineStageModel).where(
                    VideoPipelineStageModel.tenant_id == scope.tenant_id,
                    VideoPipelineStageModel.application_id == scope.application_id,
                    VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                    VideoPipelineStageModel.run_id == run_id,
                    VideoPipelineStageModel.stage_name == stage_name,
                ),
            )
            if existing is None:
                self._session.add(
                    VideoPipelineStageModel(
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

    def get(self, scope: Scope, run_id: str, stage_name: VideoStageName) -> VideoStageRecord | None:
        model = self._session.scalar(
            select(VideoPipelineStageModel).where(
                VideoPipelineStageModel.tenant_id == scope.tenant_id,
                VideoPipelineStageModel.application_id == scope.application_id,
                VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                VideoPipelineStageModel.run_id == run_id,
                VideoPipelineStageModel.stage_name == stage_name,
            ),
        )
        return _video_stage_record(model) if model is not None else None

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage_name: VideoStageName,
        *,
        error_code: ErrorCode,
        now: datetime | None = None,
    ) -> bool:
        """将无法安全恢复的阶段关闭，避免重启后静默重跑已完成阶段。"""

        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(VideoPipelineStageModel)
            .where(
                VideoPipelineStageModel.tenant_id == scope.tenant_id,
                VideoPipelineStageModel.application_id == scope.application_id,
                VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                VideoPipelineStageModel.run_id == run_id,
                VideoPipelineStageModel.stage_name == stage_name,
                VideoPipelineStageModel.status.in_(
                    (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.RUNNING)
                ),
            )
            .values(
                status=JobStatus.FAILED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=str(error_code),
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]

    def list_recoverable(self, *, now: datetime | None = None) -> tuple[VideoStageRecord, ...]:
        current_time = now or datetime.now(UTC)
        transcription = aliased(VideoPipelineStageModel)
        transcription_ready = exists(
            select(transcription.id).where(
                transcription.tenant_id == VideoPipelineStageModel.tenant_id,
                transcription.application_id == VideoPipelineStageModel.application_id,
                transcription.knowledge_base_id == VideoPipelineStageModel.knowledge_base_id,
                transcription.run_id == VideoPipelineStageModel.run_id,
                transcription.stage_name == VideoStageName.TRANSCRIPTION,
                transcription.status == JobStatus.SUCCEEDED,
            ),
        )
        models = self._session.scalars(
            select(VideoPipelineStageModel).where(
                or_(
                    (
                        VideoPipelineStageModel.status.in_(
                            (JobStatus.PENDING, JobStatus.RETRY_WAIT),
                        )
                        & (VideoPipelineStageModel.next_attempt_at <= current_time)
                    ),
                    (
                        (VideoPipelineStageModel.status == JobStatus.RUNNING)
                        & (VideoPipelineStageModel.lease_expires_at <= current_time)
                    ),
                ),
                or_(
                    VideoPipelineStageModel.stage_name != VideoStageName.LLM,
                    transcription_ready,
                ),
            ).order_by(VideoPipelineStageModel.id),
        )
        return tuple(_video_stage_record(model) for model in models)

    def heartbeat(
        self,
        lease: VideoStageLease,
        *,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("阶段租约时长必须大于 0")
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(VideoPipelineStageModel).where(
                VideoPipelineStageModel.id == lease.id,
                VideoPipelineStageModel.worker_id == lease.worker_id,
                VideoPipelineStageModel.attempt_count == lease.attempt_count,
                VideoPipelineStageModel.status == JobStatus.RUNNING,
                VideoPipelineStageModel.lease_expires_at > current_time,
            ).values(
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频阶段租约已丢失")

    def mark_cancelled(self, lease: VideoStageLease, *, now: datetime | None = None) -> None:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(VideoPipelineStageModel).where(
                VideoPipelineStageModel.id == lease.id,
                VideoPipelineStageModel.worker_id == lease.worker_id,
                VideoPipelineStageModel.attempt_count == lease.attempt_count,
                VideoPipelineStageModel.status == JobStatus.RUNNING,
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
            select(VideoPipelineStageModel.id).where(
                VideoPipelineStageModel.id == lease.id,
                VideoPipelineStageModel.attempt_count == lease.attempt_count,
                VideoPipelineStageModel.status == JobStatus.CANCELLED,
            )
        )
        if already_cancelled is not None:
            return
        raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频阶段租约已丢失")

    def claim(
        self,
        scope: Scope,
        run_id: str,
        stage_name: VideoStageName,
        worker_id: str,
        *,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> VideoStageLease | None:
        if lease_seconds < 1 or not worker_id.strip():
            raise ValueError("阶段租约参数非法")
        current_time = now or datetime.now(UTC)
        model = self._session.scalar(
            select(VideoPipelineStageModel).where(
                VideoPipelineStageModel.tenant_id == scope.tenant_id,
                VideoPipelineStageModel.application_id == scope.application_id,
                VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                VideoPipelineStageModel.run_id == run_id,
                VideoPipelineStageModel.stage_name == stage_name,
                VideoPipelineStageModel.attempt_count < VideoPipelineStageModel.max_attempts,
                or_(
                    (
                        VideoPipelineStageModel.status.in_(
                            (JobStatus.PENDING, JobStatus.RETRY_WAIT),
                        )
                        & (VideoPipelineStageModel.next_attempt_at <= current_time)
                    ),
                    (
                        (VideoPipelineStageModel.status == JobStatus.RUNNING)
                        & (VideoPipelineStageModel.lease_expires_at <= current_time)
                    ),
                ),
            ).limit(1),
        )
        if model is None:
            return None
        expected_status = model.status
        statement = update(VideoPipelineStageModel).where(
            VideoPipelineStageModel.id == model.id,
            VideoPipelineStageModel.status == expected_status,
            VideoPipelineStageModel.attempt_count == model.attempt_count,
        )
        if expected_status == JobStatus.RUNNING:
            statement = statement.where(VideoPipelineStageModel.lease_expires_at <= current_time)
        else:
            statement = statement.where(VideoPipelineStageModel.next_attempt_at <= current_time)
        result = self._session.execute(
            statement.values(
                status=JobStatus.RUNNING,
                worker_id=worker_id,
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                attempt_count=VideoPipelineStageModel.attempt_count + 1,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            return None
        self._session.expire_all()
        claimed = self._session.get(VideoPipelineStageModel, model.id)
        assert claimed is not None
        return VideoStageLease(
            id=claimed.id,
            scope=scope,
            run_id=run_id,
            stage_name=stage_name,
            worker_id=worker_id,
            attempt_count=claimed.attempt_count,
            max_attempts=claimed.max_attempts,
        )

    def mark_succeeded(
        self,
        lease: VideoStageLease,
        *,
        checkpoint_relative_path: str | None = None,
        checkpoint_sha256: str | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(VideoPipelineStageModel).where(
                VideoPipelineStageModel.id == lease.id,
                VideoPipelineStageModel.worker_id == lease.worker_id,
                VideoPipelineStageModel.attempt_count == lease.attempt_count,
                VideoPipelineStageModel.status == JobStatus.RUNNING,
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
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频阶段租约已丢失")

    def reconcile_succeeded(
        self,
        scope: Scope,
        run_id: str,
        stage_name: VideoStageName,
        *,
        now: datetime | None = None,
    ) -> None:
        """结果已经提交时，将同一阶段收敛为成功，保证发布幂等。"""

        current_time = now or datetime.now(UTC)
        self._session.execute(
            update(VideoPipelineStageModel)
            .where(
                VideoPipelineStageModel.tenant_id == scope.tenant_id,
                VideoPipelineStageModel.application_id == scope.application_id,
                VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                VideoPipelineStageModel.run_id == run_id,
                VideoPipelineStageModel.stage_name == stage_name,
                VideoPipelineStageModel.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
            )
            .values(
                status=JobStatus.SUCCEEDED,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=None,
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )

    def reset_for_retry(
        self,
        scope: Scope,
        run_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[VideoStageName, ...]:
        """人工重试时只重置失败阶段，已成功转写阶段保持不变。"""

        current_time = now or datetime.now(UTC)
        failed = tuple(
            self._session.scalars(
                select(VideoPipelineStageModel.stage_name)
                .where(
                    VideoPipelineStageModel.tenant_id == scope.tenant_id,
                    VideoPipelineStageModel.application_id == scope.application_id,
                    VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                    VideoPipelineStageModel.run_id == run_id,
                    VideoPipelineStageModel.status == JobStatus.FAILED,
                )
                .order_by(VideoPipelineStageModel.id)
            )
        )
        if not failed:
            return ()
        self._session.execute(
            update(VideoPipelineStageModel)
            .where(
                VideoPipelineStageModel.tenant_id == scope.tenant_id,
                VideoPipelineStageModel.application_id == scope.application_id,
                VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                VideoPipelineStageModel.run_id == run_id,
                VideoPipelineStageModel.status == JobStatus.FAILED,
            )
            .values(
                status=JobStatus.PENDING,
                attempt_count=0,
                next_attempt_at=current_time,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                error_code=None,
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        return failed

    def mark_failed(
        self,
        lease: VideoStageLease,
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
            update(VideoPipelineStageModel).where(
                VideoPipelineStageModel.id == lease.id,
                VideoPipelineStageModel.worker_id == lease.worker_id,
                VideoPipelineStageModel.attempt_count == lease.attempt_count,
                VideoPipelineStageModel.status == JobStatus.RUNNING,
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
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "视频阶段租约已丢失")


def _video_stage_record(model: VideoPipelineStageModel) -> VideoStageRecord:
    return VideoStageRecord(
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


def reject_sensitive_json(value: object, path: str = "$") -> None:
    """阻止 Secret 以 JSON 正文形式持久化。"""

    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(f"检测到敏感字段: {path}.{key}")
            reject_sensitive_json(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_sensitive_json(nested, f"{path}[{index}]")


def _scope_filter(
    statement: Select[tuple[VideoObjectModel]],
    scope: Scope,
) -> Select[tuple[VideoObjectModel]]:
    return statement.where(
        VideoObjectModel.tenant_id == scope.tenant_id,
        VideoObjectModel.application_id == scope.application_id,
        VideoObjectModel.knowledge_base_id == scope.knowledge_base_id,
    )


class VideoObjectRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_ready(
        self,
        *,
        scope: Scope,
        object_ref: str,
        original_filename: str,
        declared_mime: str,
        detected_mime: str,
        size_bytes: int,
        sha256: str,
        relative_path: str,
    ) -> VideoObjectModel:
        model = VideoObjectModel(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            object_ref=object_ref,
            original_filename=original_filename,
            declared_mime=declared_mime,
            detected_mime=detected_mime,
            size_bytes=size_bytes,
            sha256=sha256,
            relative_path=relative_path,
            status=VideoObjectStatus.READY,
            scan_details={},
        )
        self._session.add(model)
        self._session.flush()
        return model

    def get_ready(self, scope: Scope, object_ref: str) -> VideoObjectModel | None:
        statement = _scope_filter(select(VideoObjectModel), scope).where(
            VideoObjectModel.object_ref == object_ref,
            VideoObjectModel.status == VideoObjectStatus.READY,
        )
        return self._session.scalar(statement)

    def get(self, scope: Scope, object_ref: str) -> VideoObjectModel | None:
        statement = _scope_filter(select(VideoObjectModel), scope).where(
            VideoObjectModel.object_ref == object_ref,
        )
        return self._session.scalar(statement)

    def update_scan_details(
        self,
        *,
        scope: Scope,
        object_ref: str,
        details: dict[str, Any],
    ) -> bool:
        reject_sensitive_json(details)
        statement = _scope_filter(select(VideoObjectModel), scope).where(
            VideoObjectModel.object_ref == object_ref,
        )
        model = self._session.scalar(statement)
        if model is None:
            return False
        model.scan_details = details
        self._session.flush()
        return True


class VideoRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_or_create_asset(
        self,
        *,
        scope: Scope,
        asset_id: str,
        object_ref: str,
        source_sha256: str,
    ) -> VideoAssetModel:
        existing = self._session.scalar(
            select(VideoAssetModel).where(
                VideoAssetModel.tenant_id == scope.tenant_id,
                VideoAssetModel.application_id == scope.application_id,
                VideoAssetModel.knowledge_base_id == scope.knowledge_base_id,
                VideoAssetModel.object_ref == object_ref,
                VideoAssetModel.source_sha256 == source_sha256,
            ),
        )
        if existing is not None:
            return existing
        model = VideoAssetModel(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            asset_id=asset_id,
            object_ref=object_ref,
            source_sha256=source_sha256,
            manifest_relative_path=f"assets/{asset_id}/manifest.json",
            manifest_sha256="0" * 64,
            schema_version="1.0.0",
        )
        self._session.add(model)
        self._session.flush()
        return model

    def add(
        self,
        *,
        scope: Scope,
        run_id: str,
        asset_id: str,
        object_ref: str,
        idempotency_key: str,
        config_snapshot: dict[str, Any],
    ) -> VideoUnderstandingRunModel:
        reject_sensitive_json(config_snapshot)
        model = VideoUnderstandingRunModel(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            run_id=run_id,
            asset_id=asset_id,
            object_ref=object_ref,
            idempotency_key=idempotency_key,
            status=RunStatusValue.PENDING,
            current_stage="REGISTER",
            warning_codes=[],
            config_snapshot=config_snapshot,
        )
        self._session.add(model)
        self._session.flush()
        return model

    def get(self, scope: Scope, run_id: str) -> VideoUnderstandingRunModel | None:
        return self._session.scalar(
            select(VideoUnderstandingRunModel).where(
                VideoUnderstandingRunModel.tenant_id == scope.tenant_id,
                VideoUnderstandingRunModel.application_id == scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
                VideoUnderstandingRunModel.run_id == run_id,
            ),
        )

    def list_with_objects(
        self,
        scope: Scope,
    ) -> list[tuple[VideoUnderstandingRunModel, VideoObjectModel]]:
        """按创建时间倒序返回当前知识库的任务及其上传文件信息。"""
        statement = (
            select(VideoUnderstandingRunModel, VideoObjectModel)
            .join(
                VideoObjectModel,
                and_(
                    VideoObjectModel.tenant_id == VideoUnderstandingRunModel.tenant_id,
                    VideoObjectModel.application_id
                    == VideoUnderstandingRunModel.application_id,
                    VideoObjectModel.knowledge_base_id
                    == VideoUnderstandingRunModel.knowledge_base_id,
                    VideoObjectModel.object_ref == VideoUnderstandingRunModel.object_ref,
                ),
            )
            .where(
                VideoUnderstandingRunModel.tenant_id == scope.tenant_id,
                VideoUnderstandingRunModel.application_id == scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
            )
            .order_by(
                VideoUnderstandingRunModel.created_at.desc(),
                VideoUnderstandingRunModel.id.desc(),
            )
        )
        return [
            (run, video)
            for run, video in self._session.execute(statement).all()
        ]

    def get_by_idempotency(
        self,
        scope: Scope,
        idempotency_key: str,
    ) -> VideoUnderstandingRunModel | None:
        return self._session.scalar(
            select(VideoUnderstandingRunModel).where(
                VideoUnderstandingRunModel.tenant_id == scope.tenant_id,
                VideoUnderstandingRunModel.application_id == scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
                VideoUnderstandingRunModel.idempotency_key == idempotency_key,
            ),
        )

    def list_published_4_for_cleanup(
        self,
        *,
        after_id: int,
        limit: int = 100,
    ) -> tuple[PublishedRunCleanupRecord, ...]:
        """固定 keyset 扫描已发布 4.1 Run；返回轻量恢复记录。"""

        if after_id < 0 or limit != 100:
            raise ValueError("视觉恢复只允许固定 100 条 keyset 扫描")
        statement = (
            select(VideoUnderstandingRunModel)
            .join(
                VideoSummaryModel,
                and_(
                    VideoSummaryModel.tenant_id == VideoUnderstandingRunModel.tenant_id,
                    VideoSummaryModel.application_id == VideoUnderstandingRunModel.application_id,
                    VideoSummaryModel.knowledge_base_id
                    == VideoUnderstandingRunModel.knowledge_base_id,
                    VideoSummaryModel.run_id == VideoUnderstandingRunModel.run_id,
                    VideoSummaryModel.schema_version == "4.2.0",
                ),
            )
            .where(
                VideoUnderstandingRunModel.id > after_id,
                VideoUnderstandingRunModel.status.in_(
                    (RunStatusValue.SUCCEEDED, RunStatusValue.PARTIAL_SUCCEEDED),
                ),
                VideoUnderstandingRunModel.artifact_manifest_relative_path.is_not(None),
                VideoUnderstandingRunModel.artifact_manifest_sha256.is_not(None),
                VideoUnderstandingRunModel.document_relative_path.is_not(None),
                VideoUnderstandingRunModel.document_sha256.is_not(None),
                VideoUnderstandingRunModel.document_size_bytes.is_not(None),
            )
            .order_by(VideoUnderstandingRunModel.id)
            .limit(limit)
        )
        return tuple(
            PublishedRunCleanupRecord(
                run_pk=run.id,
                scope=Scope(run.tenant_id, run.application_id, run.knowledge_base_id),
                run_id=run.run_id,
            )
            for run in self._session.scalars(statement)
        )


class JobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def enqueue_video_run(
        self,
        *,
        scope: Scope,
        job_id: str,
        run_id: str,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobModel:
        current_time = now or datetime.now(UTC)
        model = JobModel(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            job_id=job_id,
            job_type="VIDEO_UNDERSTANDING",
            resource_type="VIDEO_UNDERSTANDING_RUN",
            resource_id=run_id,
            status=JobStatus.PENDING,
            attempt_count=0,
            max_attempts=max_attempts,
            next_attempt_at=current_time,
            cancel_requested=False,
        )
        self._session.add(model)
        self._session.flush()
        return model

    def enqueue_media_run(
        self,
        *,
        scope: Scope,
        job_id: str,
        resource_id: str,
        job_type: str,
        resource_type: str,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobModel:
        """为音频或图片运行创建独立任务；视频继续使用专用方法。"""

        if job_type not in {"AUDIO_UNDERSTANDING", "IMAGE_UNDERSTANDING"}:
            raise ValueError("媒体任务类型非法")
        if resource_type not in {"AUDIO_UNDERSTANDING_RUN", "IMAGE_UNDERSTANDING_RUN"}:
            raise ValueError("媒体资源类型非法")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("任务最大尝试次数必须大于 0")
        current_time = now or datetime.now(UTC)
        model = JobModel(
            tenant_id=scope.tenant_id,
            application_id=scope.application_id,
            knowledge_base_id=scope.knowledge_base_id,
            job_id=job_id,
            job_type=job_type,
            resource_type=resource_type,
            resource_id=resource_id,
            status=JobStatus.PENDING,
            attempt_count=0,
            max_attempts=max_attempts,
            next_attempt_at=current_time,
            cancel_requested=False,
        )
        self._session.add(model)
        self._session.flush()
        return model

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        job_type: str = "VIDEO_UNDERSTANDING",
        now: datetime | None = None,
    ) -> ClaimedJob | None:
        if job_type not in {
            "VIDEO_UNDERSTANDING",
            "AUDIO_UNDERSTANDING",
            "IMAGE_UNDERSTANDING",
        }:
            raise ValueError("任务类型非法")
        current_time = now or datetime.now(UTC)
        self._cancel_pending_jobs()
        candidate = self._session.scalar(
            select(JobModel)
            .where(
                JobModel.job_type == job_type,
                JobModel.cancel_requested.is_(False),
                JobModel.attempt_count < JobModel.max_attempts,
                or_(
                    (
                        JobModel.status.in_((JobStatus.PENDING, JobStatus.RETRY_WAIT))
                        & (JobModel.next_attempt_at <= current_time)
                    ),
                    (
                        (JobModel.status == JobStatus.RUNNING)
                        & (JobModel.lease_expires_at <= current_time)
                    ),
                ),
            )
            .order_by(JobModel.next_attempt_at, JobModel.id)
            .limit(1),
        )
        if candidate is None:
            return None

        eligible_status = candidate.status
        claim_statement = update(JobModel).where(
            JobModel.id == candidate.id,
            JobModel.status == eligible_status,
            JobModel.cancel_requested.is_(False),
            JobModel.attempt_count == candidate.attempt_count,
        )
        if eligible_status == JobStatus.RUNNING:
            claim_statement = claim_statement.where(JobModel.lease_expires_at <= current_time)
        else:
            claim_statement = claim_statement.where(JobModel.next_attempt_at <= current_time)
        result = self._session.execute(
            claim_statement
            .values(
                status=JobStatus.RUNNING,
                worker_id=worker_id,
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                attempt_count=JobModel.attempt_count + 1,
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            self._session.expire_all()
            return None
        self._session.expire_all()
        claimed = self._session.get(JobModel, candidate.id)
        assert claimed is not None
        assert claimed.worker_id is not None
        return ClaimedJob(
            id=claimed.id,
            job_id=claimed.job_id,
            resource_id=claimed.resource_id,
            worker_id=claimed.worker_id,
            attempt_count=claimed.attempt_count,
            max_attempts=claimed.max_attempts,
            scope=Scope(
                claimed.tenant_id,
                claimed.application_id,
                claimed.knowledge_base_id,
            ),
        )

    def claim_video_run(
        self,
        scope: Scope,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> ClaimedJob | None:
        """为 LLM 发布阶段领取指定视频总任务，避免重新执行转写。"""

        if not worker_id.strip() or lease_seconds < 1:
            raise ValueError("视频任务租约参数非法")
        current_time = now or datetime.now(UTC)
        candidate = self._session.scalar(
            select(JobModel).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
                JobModel.resource_id == run_id,
                JobModel.status.in_((JobStatus.PENDING, JobStatus.RETRY_WAIT)),
                JobModel.cancel_requested.is_(False),
                JobModel.next_attempt_at <= current_time,
            ),
        )
        if candidate is None:
            return None
        result = self._session.execute(
            update(JobModel).where(
                JobModel.id == candidate.id,
                JobModel.status == candidate.status,
                JobModel.attempt_count == candidate.attempt_count,
                JobModel.cancel_requested.is_(False),
            ).values(
                status=JobStatus.RUNNING,
                worker_id=worker_id,
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                attempt_count=JobModel.attempt_count + 1,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            return None
        self._session.expire_all()
        claimed = self._session.get(JobModel, candidate.id)
        assert claimed is not None and claimed.worker_id is not None
        return ClaimedJob(
            id=claimed.id,
            job_id=claimed.job_id,
            resource_id=claimed.resource_id,
            worker_id=claimed.worker_id,
            attempt_count=claimed.attempt_count,
            max_attempts=claimed.max_attempts,
            scope=scope,
        )

    def heartbeat(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> JobModel:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(JobModel)
            .where(
                JobModel.id == job_pk,
                JobModel.worker_id == worker_id,
                JobModel.attempt_count == attempt_count,
                JobModel.status == JobStatus.RUNNING,
                JobModel.cancel_requested.is_(False),
                JobModel.lease_expires_at > current_time,
            )
            .values(
                heartbeat_at=current_time,
                lease_expires_at=current_time + timedelta(seconds=lease_seconds),
                updated_at=current_time,
            ),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")
        self._session.expire_all()
        model = self._session.get(JobModel, job_pk)
        assert model is not None
        return model

    def request_cancel(self, scope: Scope, job_id: str) -> bool:
        result = self._session.execute(
            update(JobModel)
            .where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.job_id == job_id,
                JobModel.status.in_(
                    (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.RUNNING),
                ),
            )
            .values(
                cancel_requested=True,
                status=JobStatus.CANCELLED,
                error_code=str(ErrorCode.JOB_CANCELLED),
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
            )
            .execution_options(synchronize_session=False),
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            exists_in_scope = self._session.scalar(
                select(JobModel.id).where(
                    JobModel.tenant_id == scope.tenant_id,
                    JobModel.application_id == scope.application_id,
                    JobModel.knowledge_base_id == scope.knowledge_base_id,
                    JobModel.job_id == job_id,
                ),
            )
            return exists_in_scope is not None
        resource = self._session.execute(
            select(JobModel.resource_type, JobModel.resource_id).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.job_id == job_id,
            ),
        ).one()
        self._mark_run_cancelled(
            scope=scope,
            resource_type=resource.resource_type,
            resource_id=resource.resource_id,
        )
        return True

    def is_cancel_requested(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        model = self._session.scalar(
            select(JobModel).where(
                JobModel.id == job_pk,
                JobModel.attempt_count == attempt_count,
            ),
        )
        if model is not None and model.status == JobStatus.CANCELLED and model.cancel_requested:
            return True
        if (
            model is None
            or model.worker_id != worker_id
            or model.status != JobStatus.RUNNING
            or model.lease_expires_at is None
            or model.lease_expires_at <= current_time
        ):
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")
        return model.cancel_requested

    def update_owned_video_run(
        self,
        job: ClaimedJob,
        *,
        values: dict[str, object],
        allow_cancel_requested: bool = False,
        now: datetime | None = None,
    ) -> None:
        """仅允许仍持有有效租约的 Worker 修改对应运行。"""

        current_time = now or datetime.now(UTC)
        owned_job = select(JobModel.id).where(
            JobModel.id == job.id,
            JobModel.worker_id == job.worker_id,
            JobModel.attempt_count == job.attempt_count,
            JobModel.status == JobStatus.RUNNING,
            JobModel.lease_expires_at > current_time,
            JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
            JobModel.resource_id == job.resource_id,
            JobModel.tenant_id == job.scope.tenant_id,
            JobModel.application_id == job.scope.application_id,
            JobModel.knowledge_base_id == job.scope.knowledge_base_id,
        )
        if not allow_cancel_requested:
            owned_job = owned_job.where(JobModel.cancel_requested.is_(False))
        result = self._session.execute(
            update(VideoUnderstandingRunModel)
            .where(
                VideoUnderstandingRunModel.tenant_id == job.scope.tenant_id,
                VideoUnderstandingRunModel.application_id == job.scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id
                == job.scope.knowledge_base_id,
                VideoUnderstandingRunModel.run_id == job.resource_id,
                exists(owned_job),
            )
            .values(**values)
            .execution_options(synchronize_session=False),
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            return
        run_exists = self._session.scalar(
            select(VideoUnderstandingRunModel.id).where(
                VideoUnderstandingRunModel.tenant_id == job.scope.tenant_id,
                VideoUnderstandingRunModel.application_id == job.scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id
                == job.scope.knowledge_base_id,
                VideoUnderstandingRunModel.run_id == job.resource_id,
            ),
        )
        if run_exists is None:
            raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
        raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")

    def update_owned_media_run(
        self,
        job: ClaimedJob,
        *,
        values: dict[str, object],
        run_model: type[object],
        resource_type: str,
    ) -> None:
        """按 Worker 租约更新音频或图片运行，且拒绝跨资源类型写入。"""

        current_time = datetime.now(UTC)
        owned_job = select(JobModel.id).where(
            JobModel.id == job.id,
            JobModel.worker_id == job.worker_id,
            JobModel.attempt_count == job.attempt_count,
            JobModel.status == JobStatus.RUNNING,
            JobModel.lease_expires_at > current_time,
            JobModel.resource_type == resource_type,
            JobModel.resource_id == job.resource_id,
            JobModel.tenant_id == job.scope.tenant_id,
            JobModel.application_id == job.scope.application_id,
            JobModel.knowledge_base_id == job.scope.knowledge_base_id,
            JobModel.cancel_requested.is_(False),
        )
        result = self._session.execute(
            update(run_model)
            .where(
                run_model.tenant_id == job.scope.tenant_id,  # type: ignore[attr-defined]
                run_model.application_id == job.scope.application_id,  # type: ignore[attr-defined]
                run_model.knowledge_base_id == job.scope.knowledge_base_id,  # type: ignore[attr-defined]
                run_model.run_id == job.resource_id,  # type: ignore[attr-defined]
                exists(owned_job),
            )
            .values(**values)
            .execution_options(synchronize_session=False),
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            return
        raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")

    def mark_succeeded(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        now: datetime | None = None,
    ) -> bool:
        return self._mark_succeeded(
            job_pk,
            worker_id,
            attempt_count=attempt_count,
            target=None,
            now=now,
        )

    def mark_result_published(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        scope: Scope,
        run_id: str,
        resource_type: str = "VIDEO_UNDERSTANDING_RUN",
        now: datetime | None = None,
    ) -> bool:
        """仅允许目标运行对应的租约赢得结果发布。"""

        return self._mark_succeeded(
            job_pk,
            worker_id,
            attempt_count=attempt_count,
            target=(scope, run_id),
            resource_type=resource_type,
            now=now,
        )

    def _mark_succeeded(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        target: tuple[Scope, str] | None,
        resource_type: str = "VIDEO_UNDERSTANDING_RUN",
        now: datetime | None,
    ) -> bool:
        try:
            return self._finish_owned_job(
                job_pk,
                worker_id,
                attempt_count=attempt_count,
                status=JobStatus.SUCCEEDED,
                error_code=None,
                allow_cancel_requested=False,
                target=target,
                resource_type=resource_type,
                now=now,
            )
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_LEASE_LOST and self._is_owned_cancellation(
                job_pk,
                worker_id,
                attempt_count=attempt_count,
                now=now,
            ):
                raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消") from error
            raise

    def mark_failed(
        self,
        job_pk: int,
        worker_id: str,
        *,
        error_code: ErrorCode,
        retryable: bool,
        attempt_count: int,
        retry_delay_seconds: int = 5,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        owned = self._owned_running_filter(
            job_pk=job_pk,
            worker_id=worker_id,
            attempt_count=attempt_count,
            current_time=current_time,
            allow_cancel_requested=False,
        )
        should_retry = retryable and attempt_count < self._max_attempts(job_pk)
        status = JobStatus.RETRY_WAIT if should_retry else JobStatus.FAILED
        result = self._session.execute(
            update(JobModel)
            .where(and_(*owned))
            .values(
                status=status,
                next_attempt_at=current_time + timedelta(seconds=retry_delay_seconds),
                error_code=str(error_code),
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            return
        if self._attempt_is_closed(job_pk, attempt_count=attempt_count):
            return
        raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")

    def release_video_stage(
        self,
        job: ClaimedJob,
        *,
        status: JobStatus = JobStatus.PENDING,
        current_stage: str | None = None,
        error_code: ErrorCode | None = None,
        retry_delay_seconds: int = 5,
        now: datetime | None = None,
    ) -> None:
        """释放视频阶段占用的总任务租约，并原子更新运行阶段。"""

        if status not in (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.FAILED):
            raise ValueError("视频阶段只能释放为可重试或失败状态")
        if retry_delay_seconds < 0:
            raise ValueError("任务重试延迟不能为负数")
        current_time = now or datetime.now(UTC)
        if current_stage is not None:
            run = self._session.scalar(
                select(VideoUnderstandingRunModel).where(
                    VideoUnderstandingRunModel.tenant_id == job.scope.tenant_id,
                    VideoUnderstandingRunModel.application_id == job.scope.application_id,
                    VideoUnderstandingRunModel.knowledge_base_id == job.scope.knowledge_base_id,
                    VideoUnderstandingRunModel.run_id == job.resource_id,
                ),
            )
            if run is None:
                raise VideoDemoError(ErrorCode.VIDEO_RUN_NOT_FOUND, "视频理解运行不存在")
            run.current_stage = current_stage
            if status == JobStatus.FAILED:
                run.status = RunStatusValue.FAILED
                run.error_code = str(error_code) if error_code else None
            else:
                run.status = RunStatusValue.PENDING
                run.error_code = str(error_code) if error_code else None
        result = self._session.execute(
            update(JobModel).where(
                JobModel.id == job.id,
                JobModel.worker_id == job.worker_id,
                JobModel.attempt_count == job.attempt_count,
                JobModel.status == JobStatus.RUNNING,
                JobModel.lease_expires_at > current_time,
            ).values(
                status=status,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                next_attempt_at=(
                    current_time + timedelta(seconds=retry_delay_seconds)
                    if status == JobStatus.RETRY_WAIT
                    else current_time
                ),
                error_code=str(error_code) if error_code else None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")

    def release_audio_stage(
        self,
        job: ClaimedJob,
        *,
        status: JobStatus = JobStatus.PENDING,
        current_stage: str | None = None,
        error_code: ErrorCode | None = None,
        retry_delay_seconds: int = 5,
        now: datetime | None = None,
    ) -> None:
        """释放音频阶段租约，并原子更新音频运行阶段。"""

        if status not in (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.FAILED):
            raise ValueError("音频阶段只能释放为可重试或失败状态")
        if retry_delay_seconds < 0:
            raise ValueError("任务重试延迟不能为负数")
        current_time = now or datetime.now(UTC)
        if current_stage is not None:
            from video_demo.persistence.models import AudioUnderstandingRunModel

            run = self._session.scalar(
                select(AudioUnderstandingRunModel).where(
                    AudioUnderstandingRunModel.tenant_id == job.scope.tenant_id,
                    AudioUnderstandingRunModel.application_id == job.scope.application_id,
                    AudioUnderstandingRunModel.knowledge_base_id == job.scope.knowledge_base_id,
                    AudioUnderstandingRunModel.run_id == job.resource_id,
                ),
            )
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            run.current_stage = current_stage
            run.status = (
                RunStatusValue.FAILED if status == JobStatus.FAILED else RunStatusValue.PENDING
            )
            run.error_code = str(error_code) if error_code else None
        result = self._session.execute(
            update(JobModel).where(
                JobModel.id == job.id,
                JobModel.worker_id == job.worker_id,
                JobModel.attempt_count == job.attempt_count,
                JobModel.status == JobStatus.RUNNING,
                JobModel.resource_type == "AUDIO_UNDERSTANDING_RUN",
                JobModel.resource_id == job.resource_id,
                JobModel.lease_expires_at > current_time,
            ).values(
                status=status,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                next_attempt_at=(
                    current_time + timedelta(seconds=retry_delay_seconds)
                    if status == JobStatus.RETRY_WAIT
                    else current_time
                ),
                error_code=str(error_code) if error_code else None,
                updated_at=current_time,
            ).execution_options(synchronize_session=False),
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "音频任务租约已丢失")

    def fail_unclaimed_video_run(
        self,
        scope: Scope,
        run_id: str,
        *,
        error_code: ErrorCode,
        current_stage: str,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(JobModel)
            .where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
                JobModel.resource_id == run_id,
                JobModel.status.in_((JobStatus.PENDING, JobStatus.RETRY_WAIT)),
            )
            .values(
                status=JobStatus.FAILED,
                error_code=str(error_code),
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        run = self._session.scalar(
            select(VideoUnderstandingRunModel).where(
                VideoUnderstandingRunModel.tenant_id == scope.tenant_id,
                VideoUnderstandingRunModel.application_id == scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
                VideoUnderstandingRunModel.run_id == run_id,
            )
        )
        if run is not None:
            run.status = RunStatusValue.FAILED
            run.current_stage = current_stage
            run.error_code = str(error_code)
        return bool(result.rowcount)  # type: ignore[attr-defined]

    def fail_unclaimed_audio_run(
        self,
        scope: Scope,
        run_id: str,
        *,
        error_code: ErrorCode,
        current_stage: str,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        result = self._session.execute(
            update(JobModel)
            .where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == "AUDIO_UNDERSTANDING_RUN",
                JobModel.resource_id == run_id,
                JobModel.status.in_((JobStatus.PENDING, JobStatus.RETRY_WAIT)),
            )
            .values(
                status=JobStatus.FAILED,
                error_code=str(error_code),
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        from video_demo.persistence.models import AudioUnderstandingRunModel

        run = self._session.scalar(
            select(AudioUnderstandingRunModel).where(
                AudioUnderstandingRunModel.tenant_id == scope.tenant_id,
                AudioUnderstandingRunModel.application_id == scope.application_id,
                AudioUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
                AudioUnderstandingRunModel.run_id == run_id,
            ),
        )
        if run is not None:
            run.status = RunStatusValue.FAILED
            run.current_stage = current_stage
            run.error_code = str(error_code)
        return bool(result.rowcount)  # type: ignore[attr-defined]

    def mark_cancelled(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        now: datetime | None = None,
    ) -> None:
        self._finish_owned_job(
            job_pk,
            worker_id,
            attempt_count=attempt_count,
            status=JobStatus.CANCELLED,
            error_code=str(ErrorCode.JOB_CANCELLED),
            allow_cancel_requested=True,
            target=None,
            now=now,
        )
        model = self._session.get(JobModel, job_pk)
        if model is not None:
            self._mark_run_cancelled(
                scope=Scope(
                    model.tenant_id,
                    model.application_id,
                    model.knowledge_base_id,
                ),
                resource_type=model.resource_type,
                resource_id=model.resource_id,
            )

    def get(self, scope: Scope, job_id: str) -> JobModel | None:
        return self._session.scalar(
            select(JobModel).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.job_id == job_id,
            ),
        )

    def get_by_resource(self, scope: Scope, resource_id: str) -> JobModel | None:
        return self.get_by_resource_type(scope, resource_id, "VIDEO_UNDERSTANDING_RUN")

    def get_by_resource_type(
        self,
        scope: Scope,
        resource_id: str,
        resource_type: str,
    ) -> JobModel | None:
        return self._session.scalar(
            select(JobModel).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == resource_type,
                JobModel.resource_id == resource_id,
            ),
        )

    def has_active_owner(
        self,
        scope: Scope,
        run_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        return self._session.scalar(
            select(JobModel.id).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
                JobModel.resource_id == run_id,
                JobModel.status == JobStatus.RUNNING,
                JobModel.worker_id.is_not(None),
                func.trim(JobModel.worker_id, " \t\r\n\v\f") != "",
                JobModel.lease_expires_at > current_time,
            )
        ) is not None

    def retry(self, scope: Scope, job_id: str, *, now: datetime | None = None) -> JobModel:
        model = self.get(scope, job_id)
        if model is None:
            raise VideoDemoError(ErrorCode.JOB_NOT_FOUND, "任务不存在")
        if model.status != JobStatus.FAILED:
            raise VideoDemoError(ErrorCode.JOB_NOT_RETRYABLE, "只有失败任务可以重试")
        model.status = JobStatus.PENDING
        model.attempt_count = 0
        model.next_attempt_at = now or datetime.now(UTC)
        model.error_code = None
        model.cancel_requested = False
        model.worker_id = None
        model.lease_expires_at = None
        model.heartbeat_at = None
        self._session.flush()
        return model

    def _cancel_pending_jobs(self) -> None:
        pending = self._session.scalars(
            select(JobModel).where(
                JobModel.cancel_requested.is_(True),
                JobModel.status.in_((JobStatus.PENDING, JobStatus.RETRY_WAIT)),
            ),
        )
        for model in pending:
            model.status = JobStatus.CANCELLED
            model.error_code = str(ErrorCode.JOB_CANCELLED)
            model.worker_id = None
            model.lease_expires_at = None
            model.heartbeat_at = None
            self._mark_run_cancelled(
                scope=Scope(
                    model.tenant_id,
                    model.application_id,
                    model.knowledge_base_id,
                ),
                resource_type=model.resource_type,
                resource_id=model.resource_id,
            )

    def _mark_run_cancelled(
        self,
        *,
        scope: Scope,
        resource_type: str,
        resource_id: str,
    ) -> None:
        from video_demo.persistence.models import (
            AudioUnderstandingRunModel,
            ImageUnderstandingRunModel,
        )

        run_model = cast(
            type[Any] | None,
            {
                "VIDEO_UNDERSTANDING_RUN": VideoUnderstandingRunModel,
                "AUDIO_UNDERSTANDING_RUN": AudioUnderstandingRunModel,
                "IMAGE_UNDERSTANDING_RUN": ImageUnderstandingRunModel,
            }.get(resource_type),
        )
        if run_model is None:
            return
        result = self._session.execute(
            update(run_model)
            .where(
                run_model.tenant_id == scope.tenant_id,
                run_model.application_id == scope.application_id,
                run_model.knowledge_base_id == scope.knowledge_base_id,
                run_model.run_id == resource_id,
            )
            .values(
                status=RunStatusValue.CANCELLED,
                error_code=str(ErrorCode.JOB_CANCELLED),
            )
            .execution_options(synchronize_session=False),
        )
        # 通用 Job 单测和未来非视频资源可以没有对应的视频运行。
        if result.rowcount not in (0, 1):  # type: ignore[attr-defined]
            raise RuntimeError("视频运行取消更新影响了意外数量的记录")
        if resource_type == "VIDEO_UNDERSTANDING_RUN":
            self._session.execute(
                update(VideoPipelineStageModel)
                .where(
                    VideoPipelineStageModel.tenant_id == scope.tenant_id,
                    VideoPipelineStageModel.application_id == scope.application_id,
                    VideoPipelineStageModel.knowledge_base_id == scope.knowledge_base_id,
                    VideoPipelineStageModel.run_id == resource_id,
                    VideoPipelineStageModel.status.in_(
                        (JobStatus.PENDING, JobStatus.RETRY_WAIT, JobStatus.RUNNING)
                    ),
                )
                .values(
                    status=JobStatus.CANCELLED,
                    error_code=str(ErrorCode.JOB_CANCELLED),
                    worker_id=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    updated_at=datetime.now(UTC),
                )
                .execution_options(synchronize_session=False),
            )

    def _is_owned_cancellation(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        now: datetime | None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        model = self._session.get(JobModel, job_pk)
        if (
            model is None
            or model.attempt_count != attempt_count
            or not model.cancel_requested
        ):
            return False
        if model.status == JobStatus.CANCELLED:
            return True
        return (
            model.worker_id == worker_id
            and model.status == JobStatus.RUNNING
            and model.lease_expires_at is not None
            and model.lease_expires_at > current_time
        )

    def _finish_owned_job(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        status: JobStatus,
        error_code: str | None,
        allow_cancel_requested: bool,
        target: tuple[Scope, str] | None,
        resource_type: str = "VIDEO_UNDERSTANDING_RUN",
        now: datetime | None,
    ) -> bool:
        current_time = now or datetime.now(UTC)
        owned = self._owned_running_filter(
            job_pk=job_pk,
            worker_id=worker_id,
            attempt_count=attempt_count,
            current_time=current_time,
            allow_cancel_requested=allow_cancel_requested,
        )
        if target is not None:
            scope, run_id = target
            owned = (
                *owned,
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == resource_type,
                JobModel.resource_id == run_id,
            )
        result = self._session.execute(
            update(JobModel)
            .where(and_(*owned))
            .values(
                status=status,
                error_code=error_code,
                worker_id=None,
                lease_expires_at=None,
                heartbeat_at=None,
                updated_at=current_time,
            )
            .execution_options(synchronize_session=False),
        )
        if result.rowcount == 1:  # type: ignore[attr-defined]
            return True
        completed: tuple[ColumnElement[bool], ...] = (
            JobModel.id == job_pk,
            JobModel.attempt_count == attempt_count,
            JobModel.status == status,
            JobModel.error_code == error_code,
        )
        if target is not None:
            scope, run_id = target
            completed = (
                *completed,
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == resource_type,
                JobModel.resource_id == run_id,
            )
        if self._session.scalar(select(JobModel.id).where(and_(*completed))) is not None:
            return False
        raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")

    def _attempt_is_closed(self, job_pk: int, *, attempt_count: int) -> bool:
        completed = self._session.execute(
            select(JobModel.attempt_count, JobModel.status).where(JobModel.id == job_pk),
        ).one_or_none()
        return completed is not None and completed[0] == attempt_count and completed[1] in {
            JobStatus.RETRY_WAIT,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    @staticmethod
    def _owned_running_filter(
        *,
        job_pk: int,
        worker_id: str,
        attempt_count: int,
        current_time: datetime,
        allow_cancel_requested: bool,
    ) -> tuple[ColumnElement[bool], ...]:
        conditions: tuple[ColumnElement[bool], ...] = (
            JobModel.id == job_pk,
            JobModel.worker_id == worker_id,
            JobModel.attempt_count == attempt_count,
            JobModel.status == JobStatus.RUNNING,
            JobModel.lease_expires_at > current_time,
        )
        if not allow_cancel_requested:
            conditions = (*conditions, JobModel.cancel_requested.is_(False))
        return conditions

    def _max_attempts(self, job_pk: int) -> int:
        value = self._session.scalar(
            select(JobModel.max_attempts).where(JobModel.id == job_pk),
        )
        if value is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "任务租约已丢失")
        return value


__all__ = [
    "ClaimedJob",
    "JobRepository",
    "PublishedRunCleanupRecord",
    "RunRecord",
    "Scope",
    "VideoObjectRepository",
    "VideoRunRepository",
    "reject_sensitive_json",
]
