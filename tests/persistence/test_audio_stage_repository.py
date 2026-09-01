from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_stage_repository import AudioStageRepository
from video_demo.persistence.database import Database
from video_demo.persistence.models import AudioStageName, JobStatus
from video_demo.persistence.repositories import Scope


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'audio-stage.db'}")
    database.create_schema()
    return database


def _scope() -> Scope:
    return Scope("tenant-a", "app-a", "kb-a")


def test_audio_stage_repository_ensures_and_claims_two_audio_stages(database: Database) -> None:
    scope = _scope()
    with database.session() as session:
        repository = AudioStageRepository(session)
        repository.ensure(scope, "run-001")
        assert tuple(item.stage_name for item in repository.list_recoverable()) == (
            AudioStageName.TRANSCRIPTION,
        )
        lease = repository.claim(scope, "run-001", AudioStageName.TRANSCRIPTION, "audio-worker")
        assert lease is not None
        assert lease.stage_name == AudioStageName.TRANSCRIPTION
        assert lease.attempt_count == 1
        record = repository.get(scope, "run-001", AudioStageName.TRANSCRIPTION)
        assert record is not None
        assert record.status == JobStatus.RUNNING


def test_audio_stage_repository_marks_failure_retry_and_recovers(database: Database) -> None:
    scope = _scope()
    now = datetime.now(UTC) + timedelta(seconds=1)
    with database.session() as session:
        repository = AudioStageRepository(session)
        repository.ensure(scope, "run-002")
        lease = repository.claim(
            scope,
            "run-002",
            AudioStageName.TRANSCRIPTION,
            "audio-worker",
            now=now,
        )
        assert lease is not None
        repository.mark_failed(
            lease,
            error_code=ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            retryable=True,
            retry_delay_seconds=5,
            now=now,
        )
        record = repository.get(scope, "run-002", AudioStageName.TRANSCRIPTION)
        assert record is not None
        assert record.status == JobStatus.RETRY_WAIT
        assert repository.list_recoverable(now=now + timedelta(seconds=6))


def test_audio_stage_repository_rejects_lost_lease(database: Database) -> None:
    scope = _scope()
    with database.session() as session:
        repository = AudioStageRepository(session)
        repository.ensure(scope, "run-003")
        lease = repository.claim(scope, "run-003", AudioStageName.LLM, "audio-worker")
        assert lease is not None
        with pytest.raises(VideoDemoError) as raised:
            repository.heartbeat(
                lease.__class__(
                    id=lease.id,
                    scope=lease.scope,
                    run_id=lease.run_id,
                    stage_name=lease.stage_name,
                    worker_id="other-worker",
                    attempt_count=lease.attempt_count,
                    max_attempts=lease.max_attempts,
                ),
            )
        assert raised.value.code == ErrorCode.JOB_LEASE_LOST
