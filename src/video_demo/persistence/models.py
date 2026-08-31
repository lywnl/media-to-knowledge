from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator[datetime]):
    """在 SQLite 中存 naive UTC，读回时恢复 UTC 时区。"""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("持久化时间必须包含时区")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class VideoObjectStatus(StrEnum):
    QUARANTINED = "QUARANTINED"
    SCANNING = "SCANNING"
    READY = "READY"
    REJECTED = "REJECTED"


class RunStatusValue(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCEEDED = "PARTIAL_SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class VideoStageName(StrEnum):
    TRANSCRIPTION = "TRANSCRIPTION"
    LLM = "LLM"


class ScopeColumns:
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    application_id: Mapped[str] = mapped_column(String(128), nullable=False)
    knowledge_base_id: Mapped[str] = mapped_column(String(128), nullable=False)


class TimestampColumns:
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class VideoObjectModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "video_object"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
            name="uq_video_object_scope_ref",
        ),
        Index(
            "ix_video_object_scope_sha",
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "sha256",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[VideoObjectStatus] = mapped_column(
        Enum(VideoObjectStatus, native_enum=False, length=32),
        nullable=False,
    )
    scan_details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class VideoAssetModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "video_asset"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "object_ref",
            "source_sha256",
            name="uq_video_asset_scope_object_sha",
        ),
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "asset_id",
            name="uq_video_asset_scope_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)


class VideoUnderstandingRunModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "video_understanding_run"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            name="uq_video_run_scope_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "idempotency_key",
            name="uq_video_run_scope_idempotency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[RunStatusValue] = mapped_column(
        Enum(RunStatusValue, native_enum=False, length=32),
        nullable=False,
        default=RunStatusValue.PENDING,
    )
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="REGISTER")
    warning_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128))
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_manifest_relative_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    document_relative_path: Mapped[str | None] = mapped_column(String(1024))
    document_sha256: Mapped[str | None] = mapped_column(String(64))
    document_size_bytes: Mapped[int | None] = mapped_column(BigInteger)


class AudioObjectModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "audio_object"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "object_ref",
            name="uq_audio_object_scope_ref",
        ),
        Index(
            "ix_audio_object_scope_sha",
            "tenant_id", "application_id", "knowledge_base_id", "sha256",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[VideoObjectStatus] = mapped_column(
        Enum(VideoObjectStatus, native_enum=False, length=32), nullable=False,
    )
    scan_details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class ImageObjectModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "image_object"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "object_ref",
            name="uq_image_object_scope_ref",
        ),
        Index(
            "ix_image_object_scope_sha",
            "tenant_id", "application_id", "knowledge_base_id", "sha256",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    declared_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_mime: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[VideoObjectStatus] = mapped_column(
        Enum(VideoObjectStatus, native_enum=False, length=32), nullable=False,
    )
    scan_details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class AudioUnderstandingRunModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "audio_understanding_run"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_audio_run_scope_id",
        ),
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "idempotency_key",
            name="uq_audio_run_scope_idempotency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[RunStatusValue] = mapped_column(
        Enum(RunStatusValue, native_enum=False, length=32), nullable=False,
        default=RunStatusValue.PENDING,
    )
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="REGISTER")
    warning_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128))
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_relative_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    document_relative_path: Mapped[str | None] = mapped_column(String(1024))
    document_sha256: Mapped[str | None] = mapped_column(String(64))
    document_size_bytes: Mapped[int | None] = mapped_column(BigInteger)


class AudioAssetModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "audio_asset"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_audio_asset_scope_run",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    asset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)


class AudioSegmentModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "audio_segment"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id", "segment_id",
            name="uq_audio_segment_scope_run_id",
        ),
        Index(
            "ix_audio_segment_scope_run_time",
            "tenant_id", "application_id", "knowledge_base_id", "run_id", "start_ms",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    segment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class AudioSummaryModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "audio_summary"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_audio_summary_scope_run",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ImageUnderstandingRunModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "image_understanding_run"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "run_id",
            name="uq_image_run_scope_id",
        ),
        UniqueConstraint(
            "tenant_id", "application_id", "knowledge_base_id", "idempotency_key",
            name="uq_image_run_scope_idempotency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[RunStatusValue] = mapped_column(
        Enum(RunStatusValue, native_enum=False, length=32), nullable=False,
        default=RunStatusValue.PENDING,
    )
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="REGISTER")
    warning_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error_code: Mapped[str | None] = mapped_column(String(128))
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    artifact_relative_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    document_relative_path: Mapped[str | None] = mapped_column(String(1024))
    document_sha256: Mapped[str | None] = mapped_column(String(64))
    document_size_bytes: Mapped[int | None] = mapped_column(BigInteger)


class JobModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "job"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "job_id",
            name="uq_job_scope_id",
        ),
        Index("ix_job_claim", "status", "next_attempt_at", "lease_expires_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(128), nullable=False)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=32),
        nullable=False,
        default=JobStatus.PENDING,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_attempt_at: Mapped[datetime] = mapped_column(
        UtcDateTime(),
        nullable=False,
        default=utc_now,
    )
    worker_id: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    heartbeat_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(128))


class VideoPipelineStageModel(ScopeColumns, TimestampColumns, Base):
    """视频跨进程可恢复的阶段状态；内存队列只负责调度，不承载事实。"""

    __tablename__ = "video_pipeline_stage"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "stage_name",
            name="uq_video_stage_scope_run_name",
        ),
        Index(
            "ix_video_stage_recovery",
            "stage_name",
            "status",
            "next_attempt_at",
            "lease_expires_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    stage_name: Mapped[VideoStageName] = mapped_column(
        Enum(VideoStageName, native_enum=False, length=32),
        nullable=False,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=32),
        nullable=False,
        default=JobStatus.PENDING,
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_attempt_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), nullable=False, default=utc_now,
    )
    worker_id: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    heartbeat_at: Mapped[datetime | None] = mapped_column(UtcDateTime())
    checkpoint_relative_path: Mapped[str | None] = mapped_column(String(1024))
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(128))


class VideoSegmentModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "video_segment"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "segment_id",
            name="uq_video_segment_scope_run_id",
        ),
        Index(
            "ix_video_segment_scope_run_time",
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            "start_ms",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    segment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class VideoSummaryModel(ScopeColumns, TimestampColumns, Base):
    __tablename__ = "video_summary"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "application_id",
            "knowledge_base_id",
            "run_id",
            name="uq_video_summary_scope_run",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
