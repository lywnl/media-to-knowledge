from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import update

from video_demo.application.image_pipeline_executor import ImageStagePipelineExecutor
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import ImageUnderstandingRunModel, JobModel, JobStatus
from video_demo.persistence.repositories import ClaimedJob, JobRepository
from video_demo.persistence.scope import Scope


class _Handler:
    def process(self, job, *, is_cancel_requested):
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已取消")
        return object()


class _Database:
    @contextmanager
    def session(self):
        yield object()


class _Repository:
    def __init__(self, session) -> None:
        del session

    def claim_image_run(self, scope, run_id, worker_id, *, lease_seconds):
        return ClaimedJob(1, "job-1", run_id, worker_id, 1, 3, scope)

    def is_cancel_requested(self, *args, **kwargs):
        return True


def test_executor_respects_cancellation_before_handler(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_demo.application.image_pipeline_executor.JobRepository",
        _Repository,
    )
    executor = ImageStagePipelineExecutor(
        _Database(),
        _Handler(),
        runtime_root=Path("/tmp"),
    )

    with pytest.raises(VideoDemoError) as error:
        executor.run(Scope("t", "a", "k"), "run-1")

    assert error.value.code == ErrorCode.JOB_CANCELLED


class _ReclaimingHandler:
    def __init__(self, database: Database) -> None:
        self._database = database

    def process(self, job, *, is_cancel_requested):
        del is_cancel_requested
        with self._database.session() as session:
            session.execute(
                update(JobModel)
                .where(JobModel.id == job.id)
                .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
            reclaimed = JobRepository(session).claim_image_run(
                job.scope,
                job.resource_id,
                "new-image-owner",
                lease_seconds=60,
            )
            assert reclaimed is not None
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "旧租约失败")


def test_expired_executor_cannot_fail_new_image_lease(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'image-lease-fence.db'}")
    database.create_schema()
    scope = Scope("tenant", "application", "kb")
    run_id = "run_image_lease_fence"
    now = datetime.now(UTC)
    with database.session() as session:
        MediaRunRepository(session, ImageUnderstandingRunModel).add(
            scope=scope,
            run_id=run_id,
            object_ref="obj_image",
            idempotency_key="image-lease-fence",
            config_snapshot={},
        )
        JobRepository(session).enqueue_media_run(
            scope=scope,
            job_id="job_image_lease_fence",
            resource_id=run_id,
            job_type="IMAGE_UNDERSTANDING",
            resource_type="IMAGE_UNDERSTANDING_RUN",
            now=now,
        )

    executor = ImageStagePipelineExecutor(
        database,
        _ReclaimingHandler(database),
        runtime_root=tmp_path,
        lease_seconds=1,
    )
    failure = VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "旧租约失败")

    with pytest.raises(VideoDemoError) as run_error:
        executor.run(scope, run_id)
    assert run_error.value.code == failure.code

    with pytest.raises(VideoDemoError) as stage_error:
        executor.stage_failed(scope, run_id, failure)
    assert stage_error.value.code == ErrorCode.JOB_LEASE_LOST

    with database.session() as session:
        job = JobRepository(session).get_by_resource_type(
            scope,
            run_id,
            "IMAGE_UNDERSTANDING_RUN",
        )
        assert job is not None
        assert job.status == JobStatus.RUNNING
        assert job.worker_id == "new-image-owner"


def test_retry_exhaustion_marks_image_run_failed(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'image-retry-exhausted.db'}")
    database.create_schema()
    scope = Scope("tenant", "application", "kb")
    run_id = "run_image_retry_exhausted"
    with database.session() as session:
        MediaRunRepository(session, ImageUnderstandingRunModel).add(
            scope=scope,
            run_id=run_id,
            object_ref="obj_image",
            idempotency_key="image-retry-exhausted",
            config_snapshot={},
        )
        JobRepository(session).enqueue_media_run(
            scope=scope,
            job_id="job_image_retry_exhausted",
            resource_id=run_id,
            job_type="IMAGE_UNDERSTANDING",
            resource_type="IMAGE_UNDERSTANDING_RUN",
            max_attempts=1,
        )

    executor = ImageStagePipelineExecutor(
        database,
        object(),  # stage_failed 不需要调用图片处理器
        runtime_root=tmp_path,
    )
    error = VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "重试次数耗尽")
    with database.session() as session:
        job = JobRepository(session).claim_image_run(
            scope,
            run_id,
            "image-worker",
            lease_seconds=60,
        )
        assert job is not None
    assert executor.stage_failed(scope, run_id, error, job=job) is False

    with database.session() as session:
        current_job = JobRepository(session).get_by_resource_type(
            scope,
            run_id,
            "IMAGE_UNDERSTANDING_RUN",
        )
        current_run = MediaRunRepository(session, ImageUnderstandingRunModel).get(
            scope,
            run_id,
        )
        assert current_job is not None
        assert current_job.status == JobStatus.FAILED
        assert current_run is not None
        assert current_run.status.value == "FAILED"
