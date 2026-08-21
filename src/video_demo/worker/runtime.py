from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event, Thread

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import ClaimedJob, JobRepository

JobHandler = Callable[[ClaimedJob], None]
Clock = Callable[[], datetime]


class ReliableWorker:
    """每次只领取一个带租约任务，适合 M1 默认单并发执行。"""

    def __init__(
        self,
        database: Database,
        worker_id: str,
        handler: JobHandler,
        *,
        lease_seconds: int = 60,
        heartbeat_interval_seconds: float | None = None,
        on_heartbeat: Callable[[], None] = lambda: None,
        clock: Clock | None = None,
        owned_resources: tuple[object, ...] = (),
    ) -> None:
        actual_interval = heartbeat_interval_seconds or max(0.1, lease_seconds / 3)
        if actual_interval <= 0 or actual_interval >= lease_seconds:
            raise ValueError("心跳间隔必须大于 0 且小于租约时长")
        self._database = database
        self._worker_id = worker_id
        self._handler = handler
        self._lease_seconds = lease_seconds
        self._heartbeat_interval_seconds = actual_interval
        self._on_heartbeat = on_heartbeat
        self._clock = clock or (lambda: datetime.now(UTC))
        self._owned_resources = owned_resources
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def run_once(self) -> bool:
        now = self._clock()
        with self._database.session() as session:
            job = JobRepository(session).claim(
                self._worker_id,
                lease_seconds=self._lease_seconds,
                now=now,
            )
        if job is None:
            return False

        try:
            with self._database.session() as session:
                if JobRepository(session).is_cancel_requested(
                    job.id,
                    self._worker_id,
                    attempt_count=job.attempt_count,
                    now=self._clock(),
                ):
                    JobRepository(session).mark_cancelled(
                        job.id,
                        self._worker_id,
                        attempt_count=job.attempt_count,
                        now=self._clock(),
                    )
                    return True
            heartbeat_error = self._run_with_heartbeat(job, lambda: self._handler(job))
            if heartbeat_error is not None:
                raise heartbeat_error
            with self._database.session() as session:
                repository = JobRepository(session)
                try:
                    repository.mark_succeeded(
                        job.id,
                        self._worker_id,
                        attempt_count=job.attempt_count,
                        now=self._clock(),
                    )
                except VideoDemoError as error:
                    if (
                        error.code == ErrorCode.JOB_LEASE_LOST
                        and repository.is_cancel_requested(
                            job.id,
                            self._worker_id,
                            attempt_count=job.attempt_count,
                            now=self._clock(),
                        )
                    ):
                        repository.mark_cancelled(
                            job.id,
                            self._worker_id,
                            attempt_count=job.attempt_count,
                            now=self._clock(),
                        )
                    else:
                        raise
        except VideoDemoError as error:
            if error.code == ErrorCode.JOB_CANCELLED:
                self._record_cancellation(job)
            elif error.code == ErrorCode.JOB_LEASE_LOST:
                pass
            else:
                self._record_failure(job, error)
        except Exception:
            self._record_failure(
                job,
                VideoDemoError(ErrorCode.SYSTEM_FAILURE, "任务发生未分类系统错误"),
            )
        return True

    def _run_with_heartbeat(
        self,
        job: ClaimedJob,
        handler: Callable[[], None],
    ) -> VideoDemoError | None:
        stop = Event()
        errors: list[VideoDemoError] = []

        def heartbeat_loop() -> None:
            while not stop.wait(self._heartbeat_interval_seconds):
                try:
                    with self._database.session() as session:
                        JobRepository(session).heartbeat(
                            job.id,
                            self._worker_id,
                            attempt_count=job.attempt_count,
                            lease_seconds=self._lease_seconds,
                            now=self._clock(),
                        )
                    self._on_heartbeat()
                except VideoDemoError as error:
                    errors.append(error)
                    return

        heartbeat = Thread(
            target=heartbeat_loop,
            name=f"job-heartbeat-{self._worker_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            handler()
        finally:
            stop.set()
            heartbeat.join()
        return errors[0] if errors else None

    def _record_failure(self, job: ClaimedJob, error: VideoDemoError) -> None:
        retryable = error.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
        with self._database.session() as session:
            JobRepository(session).mark_failed(
                job.id,
                self._worker_id,
                error_code=error.code,
                retryable=retryable,
                attempt_count=job.attempt_count,
                now=self._clock(),
            )

    def _record_cancellation(self, job: ClaimedJob) -> None:
        with self._database.session() as session:
            JobRepository(session).mark_cancelled(
                job.id,
                self._worker_id,
                attempt_count=job.attempt_count,
                now=self._clock(),
            )
