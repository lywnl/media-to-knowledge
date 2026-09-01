"""图片阶段执行器：负责数据库租约、心跳和图片业务调用。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread, local
from typing import Protocol

from video_demo.errors import ErrorCode, VideoDemoError, is_retryable_error_code
from video_demo.persistence.database import Database
from video_demo.persistence.models import ImageUnderstandingRunModel, JobStatus, RunStatusValue
from video_demo.persistence.repositories import ClaimedJob, JobRepository
from video_demo.persistence.scope import Scope


class ImageHandler(Protocol):
    def process(self, job: ClaimedJob, *, is_cancel_requested: Callable[[], bool]) -> None: ...


class ImageStagePipelineExecutor:
    """每个图片 Run 独立领取租约，失去租约后禁止继续写入结果。"""

    def __init__(
        self,
        database: Database,
        handler: ImageHandler,
        *,
        runtime_root: Path,
        lease_seconds: int = 120,
        owned_resources: tuple[object, ...] = (),
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("图片任务租约时长必须大于 0")
        self._database = database
        self._handler = handler
        self._runtime_root = runtime_root
        self._lease_seconds = lease_seconds
        self._owned_resources = owned_resources
        self._closed = False
        self._active_job = local()

    def run(self, scope: Scope, run_id: str) -> None:
        worker_id = f"image-api-{uuid.uuid4().hex}"
        with self._database.session() as session:
            job = JobRepository(session).claim_image_run(
                scope,
                run_id,
                worker_id,
                lease_seconds=self._lease_seconds,
            )
        if job is None:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "图片任务无法领取")
        self._active_job.claim = job
        try:
            self._run_with_heartbeat(job, lambda: self._handler.process(
                job,
                is_cancel_requested=lambda: self._is_cancelled(job),
            ))
        except BaseException:
            # 调度器会在同一执行线程中调用 stage_failed()，因此保留本次
            # 领取的租约，禁止失败处理重新按 run_id 查找其他执行者的租约。
            raise
        else:
            self._clear_active_job(job)

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        with self._database.session() as session:
            job = JobRepository(session).get_by_resource_type(
                scope,
                run_id,
                "IMAGE_UNDERSTANDING_RUN",
            )
            return job is not None and job.status == JobStatus.CANCELLED

    def stage_failed(
        self,
        scope: Scope,
        run_id: str,
        error: VideoDemoError,
        *,
        job: ClaimedJob | None = None,
    ) -> bool:
        if job is None:
            job = self._current_active_job(scope, run_id)
        try:
            with self._database.session() as session:
                repository = JobRepository(session)
                if error.code == ErrorCode.JOB_CANCELLED:
                    repository.mark_cancelled(
                        job.id,
                        job.worker_id,
                        attempt_count=job.attempt_count,
                    )
                    return False
                retryable = is_retryable_error_code(error.code)
                should_retry = retryable and job.attempt_count < job.max_attempts
                repository.update_owned_media_run(
                    job,
                    values={
                        "status": (
                            RunStatusValue.PENDING
                            if should_retry
                            else RunStatusValue.FAILED
                        ),
                        "current_stage": "VLM",
                        "error_code": error.code.value,
                    },
                    run_model=ImageUnderstandingRunModel,
                    resource_type="IMAGE_UNDERSTANDING_RUN",
                )
                repository.mark_failed(
                    job.id,
                    job.worker_id,
                    error_code=error.code,
                    retryable=retryable,
                    attempt_count=job.attempt_count,
                )
                return should_retry
        finally:
            self._clear_active_job(job)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def _run_with_heartbeat(self, job: ClaimedJob, operation: Callable[[], None]) -> None:
        stop = Event()
        errors: list[VideoDemoError] = []

        def heartbeat() -> None:
            while not stop.wait(max(0.5, self._lease_seconds / 3)):
                try:
                    with self._database.session() as session:
                        JobRepository(session).heartbeat(
                            job.id,
                            job.worker_id,
                            attempt_count=job.attempt_count,
                            lease_seconds=self._lease_seconds,
                        )
                except VideoDemoError as error:
                    errors.append(error)
                    return

        thread = Thread(target=heartbeat, name=f"image-heartbeat-{job.resource_id}", daemon=True)
        thread.start()
        try:
            operation()
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]

    def _is_cancelled(self, job: ClaimedJob) -> bool:
        with self._database.session() as session:
            return JobRepository(session).is_cancel_requested(
                job.id,
                job.worker_id,
                attempt_count=job.attempt_count,
            )

    def _current_active_job(self, scope: Scope, run_id: str) -> ClaimedJob:
        job = getattr(self._active_job, "claim", None)
        if (
            not isinstance(job, ClaimedJob)
            or job.scope != scope
            or job.resource_id != run_id
        ):
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "图片任务租约已丢失")
        return job

    def _clear_active_job(self, job: ClaimedJob) -> None:
        if getattr(self._active_job, "claim", None) == job:
            del self._active_job.claim


__all__ = ["ImageStagePipelineExecutor"]
