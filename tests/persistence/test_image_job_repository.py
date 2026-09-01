from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus
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
