from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus
from video_demo.persistence.repositories import JobRepository, Scope
from video_demo.worker.runtime import ReliableWorker


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'worker.db'}")
    database.create_schema()
    return database


@pytest.fixture
def scope() -> Scope:
    return Scope("tenant-a", "app-a", "kb-a")


TEST_NOW = datetime(2026, 8, 17, tzinfo=UTC)


def _enqueue(
    database: Database,
    scope: Scope,
    *,
    max_attempts: int = 3,
    now: datetime = TEST_NOW,
) -> str:
    with database.session() as session:
        return JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
            max_attempts=max_attempts,
            now=now,
        ).job_id


def test_claim_returns_a_job_only_once_before_lease_expires(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)
    now = TEST_NOW

    with database.session() as session:
        first = JobRepository(session).claim("worker-a", lease_seconds=30, now=now)
    with database.session() as session:
        second = JobRepository(session).claim("worker-b", lease_seconds=30, now=now)

    assert first is not None
    assert first.worker_id == "worker-a"
    assert first.attempt_count == 1
    assert second is None


def test_expired_lease_can_be_reclaimed_and_old_worker_loses_ownership(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)
    now = TEST_NOW
    with database.session() as session:
        original = JobRepository(session).claim("worker-a", lease_seconds=10, now=now)
    assert original is not None
    with database.session() as session:
        reclaimed = JobRepository(session).claim(
            "worker-b",
            lease_seconds=10,
            now=now + timedelta(seconds=11),
        )
    assert reclaimed is not None
    assert reclaimed.worker_id == "worker-b"
    assert reclaimed.attempt_count == 2

    with database.session() as session, pytest.raises(VideoDemoError) as raised:
        JobRepository(session).heartbeat(
            original.id,
            "worker-a",
            attempt_count=original.attempt_count,
            lease_seconds=10,
            now=now + timedelta(seconds=12),
        )
    assert raised.value.code == ErrorCode.JOB_LEASE_LOST

    with database.session() as session, pytest.raises(VideoDemoError) as completion_error:
        JobRepository(session).mark_succeeded(
            original.id,
            "worker-a",
            attempt_count=original.attempt_count,
            now=now + timedelta(seconds=12),
        )
    assert completion_error.value.code == ErrorCode.JOB_LEASE_LOST


def test_heartbeat_extends_the_owned_lease(database: Database, scope: Scope) -> None:
    _enqueue(database, scope)
    now = TEST_NOW
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=10, now=now)
    assert claimed is not None
    with database.session() as session:
        heartbeat = JobRepository(session).heartbeat(
            claimed.id,
            "worker-a",
            attempt_count=claimed.attempt_count,
            lease_seconds=30,
            now=now + timedelta(seconds=5),
        )

    assert heartbeat.lease_expires_at == now + timedelta(seconds=35)


def test_cancelled_job_is_not_executed(database: Database, scope: Scope) -> None:
    _enqueue(database, scope)
    with database.session() as session:
        JobRepository(session).request_cancel(scope, "job_001")

    executed: list[str] = []
    worker = ReliableWorker(database, "worker-a", lambda job: executed.append(job.job_id))

    assert worker.run_once() is False
    assert executed == []
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.CANCELLED


def test_retryable_failure_stops_after_max_attempts(database: Database, scope: Scope) -> None:
    _enqueue(database, scope, max_attempts=2)

    def fail(_job: object) -> None:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "依赖暂时不可用")

    now = TEST_NOW
    worker = ReliableWorker(database, "worker-a", fail, clock=lambda: now)
    assert worker.run_once() is True
    now += timedelta(seconds=10)
    assert worker.run_once() is True

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.attempt_count == 2
        assert job.error_code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE


def test_successful_worker_marks_job_succeeded(database: Database, scope: Scope) -> None:
    _enqueue(database, scope)
    executed: list[str] = []
    worker = ReliableWorker(database, "worker-a", lambda job: executed.append(job.resource_id))

    assert worker.run_once() is True

    assert executed == ["run_001"]
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED


def test_worker_completion_does_not_touch_same_job_id_in_another_scope(
    database: Database,
    scope: Scope,
) -> None:
    other_scope = Scope("tenant-b", "app-a", "kb-a")
    _enqueue(database, scope)
    with database.session() as session:
        JobRepository(session).enqueue_video_run(
            scope=other_scope,
            job_id="job_001",
            run_id="run_other",
            now=TEST_NOW,
        )

    with database.session() as session:
        repository = JobRepository(session)
        first_claim = repository.claim("worker-shared", lease_seconds=60, now=TEST_NOW)
        second_claim = repository.claim("worker-shared", lease_seconds=60, now=TEST_NOW)
    assert first_claim is not None
    assert second_claim is not None
    assert first_claim.id != second_claim.id

    with database.session() as session:
        JobRepository(session).mark_succeeded(
            first_claim.id,
            "worker-shared",
            attempt_count=first_claim.attempt_count,
            now=TEST_NOW,
        )

    with database.session() as session:
        first = JobRepository(session).get(scope, "job_001")
        other = JobRepository(session).get(other_scope, "job_001")
        assert first is not None
        assert other is not None
        assert first.status == JobStatus.SUCCEEDED
        assert other.status == JobStatus.RUNNING


def test_unexpected_handler_error_is_recorded_as_non_retryable_failure(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)

    def fail(_job: object) -> None:
        raise RuntimeError("unexpected content that must not become a persisted error")

    worker = ReliableWorker(database, "worker-a", fail, clock=lambda: TEST_NOW)

    assert worker.run_once() is True
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == ErrorCode.SYSTEM_FAILURE


def test_worker_renews_lease_while_long_running_handler_is_active(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)
    heartbeat_observed = Event()

    def handler(job: object) -> None:
        assert heartbeat_observed.wait(timeout=2)

    worker = ReliableWorker(
        database,
        "worker-a",
        handler,
        lease_seconds=1,
        heartbeat_interval_seconds=0.05,
        on_heartbeat=lambda: heartbeat_observed.set(),
    )

    assert worker.run_once() is True
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.SUCCEEDED


def test_cancellation_during_handler_marks_job_cancelled(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)

    def handler(_job: object) -> None:
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")

    worker = ReliableWorker(database, "worker-a", handler)

    assert worker.run_once() is True
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.JOB_CANCELLED


def test_cancellation_after_handler_prevents_successful_completion(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)

    def handler(_job: object) -> None:
        with database.session() as session:
            assert JobRepository(session).request_cancel(scope, "job_001") is True

    worker = ReliableWorker(database, "worker-a", handler)

    assert worker.run_once() is True
    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.JOB_CANCELLED


def test_running_cancellation_is_terminal_without_worker_acknowledgement(
    database: Database,
    scope: Scope,
) -> None:
    _enqueue(database, scope)
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=10, now=TEST_NOW)
    assert claimed is not None

    with database.session() as session:
        assert JobRepository(session).request_cancel(scope, "job_001") is True

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.JOB_CANCELLED
        assert job.worker_id is None
        assert job.lease_expires_at is None
        assert JobRepository(session).claim(
            "worker-b",
            lease_seconds=10,
            now=TEST_NOW + timedelta(seconds=11),
        ) is None
