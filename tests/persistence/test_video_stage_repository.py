from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from video_demo.errors import ErrorCode
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus, VideoPipelineStageModel, VideoStageName
from video_demo.persistence.repositories import (
    JobRepository,
    Scope,
    VideoStageRepository,
)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'stages.db'}")
    database.create_schema()
    return database


@pytest.fixture
def scope() -> Scope:
    return Scope("tenant-a", "app-a", "kb-a")


def test_stage_repository_creates_claims_and_recovers_only_due_stages(
    database: Database,
    scope: Scope,
) -> None:
    now = datetime.now(UTC) + timedelta(seconds=1)
    with database.session() as session:
        repository = VideoStageRepository(session)
        repository.ensure(scope, "run-001")
        lease = repository.claim(
            scope,
            "run-001",
            VideoStageName.TRANSCRIPTION,
            "worker-a",
            now=now,
        )
        assert lease is not None
        repository.mark_failed(
            lease,
            error_code=ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            retryable=True,
            retry_delay_seconds=30,
            now=now,
        )

    with database.session() as session:
        repository = VideoStageRepository(session)
        assert repository.list_recoverable(now=now) == ()
        due = repository.list_recoverable(now=now + timedelta(seconds=30))
        assert len(due) == 1
        assert due[0].status == JobStatus.RETRY_WAIT
        reclaimed = repository.claim(
            scope,
            "run-001",
            VideoStageName.TRANSCRIPTION,
            "worker-b",
            now=now + timedelta(seconds=30),
        )
        assert reclaimed is not None
        assert reclaimed.attempt_count == 2


def test_release_video_stage_preserves_total_job_attempt_count(
    database: Database,
    scope: Scope,
) -> None:
    with database.session() as session:
        job = JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job-001",
            run_id="run-001",
        )
        claimed = JobRepository(session).claim_video_run(scope, "run-001", "worker-a")
        assert claimed is not None
        JobRepository(session).release_video_stage(
            claimed,
            status=JobStatus.PENDING,
        )
        assert job.attempt_count == 1

    with database.session() as session:
        job = JobRepository(session).get(scope, "job-001")
        assert job is not None
        assert job.status == JobStatus.PENDING
        assert job.attempt_count == 1


def test_stage_scope_and_run_name_are_unique(database: Database, scope: Scope) -> None:
    with database.session() as session:
        repository = VideoStageRepository(session)
        repository.ensure(scope, "run-001")
        repository.ensure(scope, "run-001")
        assert len(session.scalars(select(VideoPipelineStageModel)).all()) == 2


def test_stage_cancel_is_idempotent_after_job_cancel(database: Database, scope: Scope) -> None:
    with database.session() as session:
        repository = VideoStageRepository(session)
        repository.ensure(scope, "run-001")
        lease = repository.claim(
            scope,
            "run-001",
            VideoStageName.TRANSCRIPTION,
            "worker-a",
        )
        assert lease is not None
        repository.mark_cancelled(lease)
        repository.mark_cancelled(lease)


def test_retry_resets_only_failed_video_stage(database: Database, scope: Scope) -> None:
    with database.session() as session:
        repository = VideoStageRepository(session)
        repository.ensure(scope, "run-001")
        transcription = repository.claim(
            scope,
            "run-001",
            VideoStageName.TRANSCRIPTION,
            "worker-a",
        )
        assert transcription is not None
        repository.mark_succeeded(transcription)
        llm = repository.claim(scope, "run-001", VideoStageName.LLM, "worker-a")
        assert llm is not None
        repository.mark_failed(
            llm,
            error_code=ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            retryable=False,
        )
        assert repository.reset_for_retry(scope, "run-001") == (VideoStageName.LLM,)
        transcription_record = repository.get(scope, "run-001", VideoStageName.TRANSCRIPTION)
        llm_record = repository.get(scope, "run-001", VideoStageName.LLM)
        assert transcription_record is not None
        assert llm_record is not None
        assert transcription_record.status == JobStatus.SUCCEEDED
        assert llm_record.status == JobStatus.PENDING
