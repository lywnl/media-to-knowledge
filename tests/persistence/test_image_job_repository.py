from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import update

from video_demo.persistence.database import Database
from video_demo.persistence.models import JobModel, JobStatus
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope


def _database(tmp_path: Path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'image-jobs.db'}")
    database.create_schema()
    return database


def _scope() -> Scope:
    return Scope("tenant", "application", "kb")


def test_claim_image_run_reclaims_expired_running_lease(tmp_path: Path) -> None:
    database = _database(tmp_path)
    scope = _scope()
    first_now = datetime(2026, 9, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=3)
    with database.session() as session:
        repository = JobRepository(session)
        repository.enqueue_media_run(
            scope=scope,
            job_id="job-image",
            resource_id="run-image",
            job_type="IMAGE_UNDERSTANDING",
            resource_type="IMAGE_UNDERSTANDING_RUN",
            now=first_now,
        )
        first = repository.claim_image_run(
            scope,
            "run-image",
            "image-a",
            lease_seconds=1,
            now=first_now,
        )
        assert first is not None

    with database.session() as session:
        reclaimed = JobRepository(session).claim_image_run(
            scope,
            "run-image",
            "image-b",
            lease_seconds=60,
            now=second_now,
        )
        assert reclaimed is not None
        assert reclaimed.worker_id == "image-b"
        assert reclaimed.attempt_count == 2


def test_claim_image_run_does_not_consume_audio_job(tmp_path: Path) -> None:
    database = _database(tmp_path)
    scope = _scope()
    now = datetime(2026, 9, 1, tzinfo=UTC)
    with database.session() as session:
        repository = JobRepository(session)
        repository.enqueue_media_run(
            scope=scope,
            job_id="job-audio",
            resource_id="run-audio",
            job_type="AUDIO_UNDERSTANDING",
            resource_type="AUDIO_UNDERSTANDING_RUN",
            now=now,
        )
        assert repository.claim_image_run(
            scope,
            "run-audio",
            "image-worker",
            now=now,
        ) is None
        audio = repository.get(scope, "job-audio")
        assert audio is not None
    assert audio.status == JobStatus.PENDING


def test_claim_image_run_does_not_reclaim_after_attempt_limit(tmp_path: Path) -> None:
    database = _database(tmp_path)
    scope = _scope()
    first_now = datetime(2026, 9, 1, tzinfo=UTC)
    expired_now = first_now + timedelta(seconds=3)
    with database.session() as session:
        repository = JobRepository(session)
        repository.enqueue_media_run(
            scope=scope,
            job_id="job-image-limit",
            resource_id="run-image-limit",
            job_type="IMAGE_UNDERSTANDING",
            resource_type="IMAGE_UNDERSTANDING_RUN",
            max_attempts=1,
            now=first_now,
        )
        claimed = repository.claim_image_run(
            scope,
            "run-image-limit",
            "image-first",
            lease_seconds=1,
            now=first_now,
        )
        assert claimed is not None

    with database.session() as session:
        reclaimed = JobRepository(session).claim_image_run(
            scope,
            "run-image-limit",
            "image-second",
            lease_seconds=60,
            now=expired_now,
        )
        assert reclaimed is None


def test_list_recoverable_image_runs_filters_by_state_and_attempt_limit(tmp_path: Path) -> None:
    database = _database(tmp_path)
    scope = _scope()
    now = datetime(2026, 9, 1, tzinfo=UTC)
    with database.session() as session:
        repository = JobRepository(session)
        for run_id in (
            "run-pending",
            "run-retry-due",
            "run-retry-future",
            "run-running-expired",
            "run-running-active",
            "run-cancelled",
            "run-succeeded",
            "run-attempt-limit",
        ):
            repository.enqueue_media_run(
                scope=scope,
                job_id=f"job-{run_id}",
                resource_id=run_id,
                job_type="IMAGE_UNDERSTANDING",
                resource_type="IMAGE_UNDERSTANDING_RUN",
                max_attempts=1 if run_id == "run-attempt-limit" else 3,
                now=now,
            )
        repository.enqueue_media_run(
            scope=scope,
            job_id="job-audio-not-image",
            resource_id="run-audio-not-image",
            job_type="AUDIO_UNDERSTANDING",
            resource_type="AUDIO_UNDERSTANDING_RUN",
            now=now,
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-retry-due")
            .values(
                status=JobStatus.RETRY_WAIT,
                next_attempt_at=now - timedelta(seconds=1),
            )
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-retry-future")
            .values(
                status=JobStatus.RETRY_WAIT,
                next_attempt_at=now + timedelta(seconds=10),
            )
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-cancelled")
            .values(status=JobStatus.CANCELLED, cancel_requested=True)
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-succeeded")
            .values(status=JobStatus.SUCCEEDED)
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-attempt-limit")
            .values(attempt_count=1)
        )
        session.execute(
            update(JobModel)
            .where(JobModel.resource_id == "run-running-expired")
            .values(next_attempt_at=now - timedelta(seconds=5))
        )
        expired = repository.claim_image_run(
            scope,
            "run-running-expired",
            "expired-owner",
            lease_seconds=1,
            now=now,
        )
        assert expired is not None
        active = repository.claim_image_run(
            scope,
            "run-running-active",
            "active-owner",
            lease_seconds=60,
            now=now,
        )
        assert active is not None

        recoverable = repository.list_recoverable_image_runs(
            now=now + timedelta(seconds=2),
        )

    assert recoverable == (
        (scope, "run-pending"),
        (scope, "run-retry-due"),
        (scope, "run-running-expired"),
    )
