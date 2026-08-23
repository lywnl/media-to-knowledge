from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, and_, delete, exists, or_, select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from video_demo.domain.result import (
    SUPPORTED_RESULT_SCHEMA_VERSIONS,
    VideoSegment,
    VideoSummary,
    VideoUnderstandingResult,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import (
    JobModel,
    JobStatus,
    RunStatusValue,
    VideoAssetModel,
    VideoObjectModel,
    VideoObjectStatus,
    VideoSegmentModel,
    VideoSummaryModel,
    VideoUnderstandingRunModel,
)

_SENSITIVE_KEY_PARTS = ("secret", "token", "api_key", "apikey", "authorization", "password")


@dataclass(frozen=True, slots=True)
class Scope:
    tenant_id: str
    application_id: str
    knowledge_base_id: str


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

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> ClaimedJob | None:
        current_time = now or datetime.now(UTC)
        self._cancel_pending_jobs()
        candidate = self._session.scalar(
            select(JobModel)
            .where(
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
        now: datetime | None = None,
    ) -> bool:
        """仅允许目标运行对应的租约赢得结果发布。"""

        return self._mark_succeeded(
            job_pk,
            worker_id,
            attempt_count=attempt_count,
            target=(scope, run_id),
            now=now,
        )

    def _mark_succeeded(
        self,
        job_pk: int,
        worker_id: str,
        *,
        attempt_count: int,
        target: tuple[Scope, str] | None,
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
        return self._session.scalar(
            select(JobModel).where(
                JobModel.tenant_id == scope.tenant_id,
                JobModel.application_id == scope.application_id,
                JobModel.knowledge_base_id == scope.knowledge_base_id,
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
                JobModel.resource_id == resource_id,
            ),
        )

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
        if resource_type != "VIDEO_UNDERSTANDING_RUN":
            return
        result = self._session.execute(
            update(VideoUnderstandingRunModel)
            .where(
                VideoUnderstandingRunModel.tenant_id == scope.tenant_id,
                VideoUnderstandingRunModel.application_id == scope.application_id,
                VideoUnderstandingRunModel.knowledge_base_id == scope.knowledge_base_id,
                VideoUnderstandingRunModel.run_id == resource_id,
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
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
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
                JobModel.resource_type == "VIDEO_UNDERSTANDING_RUN",
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


class ResultRepository:
    """在既有结果表中保存严格领域模型，不持久化证据正文。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace(
        self,
        scope: Scope,
        result: VideoUnderstandingResult,
    ) -> None:
        for model in (VideoSegmentModel, VideoSummaryModel):
            self._session.execute(
                delete(model).where(
                    model.tenant_id == scope.tenant_id,
                    model.application_id == scope.application_id,
                    model.knowledge_base_id == scope.knowledge_base_id,
                    model.run_id == result.run_id,
                ),
            )
        for segment in result.segments:
            payload = segment.model_dump(mode="json", exclude_computed_fields=True)
            reject_sensitive_json(payload)
            self._session.add(
                VideoSegmentModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=result.run_id,
                    segment_id=segment.segment_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    schema_version=result.schema_version,
                    payload_json=payload,
                    retrieval_text=segment.retrieval_text,
                    retrieval_hash=segment.retrieval_hash,
                ),
            )
        summary_payload = result.summary.model_dump(mode="json", exclude_computed_fields=True)
        reject_sensitive_json(summary_payload)
        self._session.add(
            VideoSummaryModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                run_id=result.run_id,
                schema_version=result.schema_version,
                payload_json=summary_payload,
                retrieval_text=result.summary.retrieval_text,
                retrieval_hash=result.summary.retrieval_hash,
            ),
        )
        self._session.flush()

    def get(self, scope: Scope, run_id: str, asset_sha256: str) -> VideoUnderstandingResult | None:
        segments = self._session.scalars(
            select(VideoSegmentModel)
            .where(
                VideoSegmentModel.tenant_id == scope.tenant_id,
                VideoSegmentModel.application_id == scope.application_id,
                VideoSegmentModel.knowledge_base_id == scope.knowledge_base_id,
                VideoSegmentModel.run_id == run_id,
            )
            .order_by(VideoSegmentModel.start_ms, VideoSegmentModel.end_ms, VideoSegmentModel.id),
        ).all()
        summary = self._session.scalar(
            select(VideoSummaryModel).where(
                VideoSummaryModel.tenant_id == scope.tenant_id,
                VideoSummaryModel.application_id == scope.application_id,
                VideoSummaryModel.knowledge_base_id == scope.knowledge_base_id,
                VideoSummaryModel.run_id == run_id,
            ),
        )
        if not segments or summary is None:
            return None
        versions = {str(item.schema_version) for item in segments} | {str(summary.schema_version)}
        if not versions.issubset(SUPPORTED_RESULT_SCHEMA_VERSIONS) or len(versions) != 1:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "结果行 Schema 版本非法或不一致",
            )
        return VideoUnderstandingResult(
            # 旧结果行仍可按 1.0.0 双读，新写入结果由领域模型默认升级到 2.0.0。
            schema_version=summary.schema_version,
            run_id=run_id,
            asset_sha256=asset_sha256,
            segments=tuple(VideoSegment.model_validate(item.payload_json) for item in segments),
            summary=VideoSummary.model_validate(summary.payload_json),
        )
