from __future__ import annotations

import threading
import time

from video_demo.application.image_scheduler import ImageTaskScheduler
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.scope import Scope


def _scope() -> Scope:
    return Scope("tenant", "application", "kb")


class _Executor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.running = 0
        self.max_running = 0
        self.lock = threading.Lock()
        self.calls: list[str] = []
        self.fail_once: set[str] = set()

    def run(self, scope: Scope, run_id: str) -> None:
        del scope
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.calls.append(run_id)
        self.started.set()
        self.release.wait(2)
        with self.lock:
            self.running -= 1
        if run_id in self.fail_once:
            self.fail_once.remove(run_id)
            raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "可重试失败")

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        del scope, run_id
        return False

    def stage_failed(
        self,
        scope: Scope,
        run_id: str,
        error: VideoDemoError,
    ) -> bool:
        del scope, run_id
        return error.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE

    def close(self) -> None:
        return None


class _SequenceExecutor:
    def __init__(self, failures: dict[str, tuple[ErrorCode, int]] | None = None) -> None:
        self.failures = failures or {}
        self.calls: list[str] = []
        self.failure_records: list[tuple[str, ErrorCode]] = []
        self.cancelled: set[str] = set()
        self.lock = threading.Lock()

    def run(self, scope: Scope, run_id: str) -> None:
        del scope
        with self.lock:
            self.calls.append(run_id)
            failure = self.failures.get(run_id)
            if failure is not None and failure[1] > 0:
                self.failures[run_id] = (failure[0], failure[1] - 1)
                raise VideoDemoError(failure[0], "测试失败")

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        del scope
        return run_id in self.cancelled

    def stage_failed(self, scope: Scope, run_id: str, error: VideoDemoError) -> bool:
        del scope
        self.failure_records.append((run_id, error.code))
        return error.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE

    def close(self) -> None:
        return None


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_image_scheduler_runs_two_images_and_queues_the_third() -> None:
    executor = _Executor()
    scheduler = ImageTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-1") == "accepted"
        assert scheduler.submit(_scope(), "run-2") == "accepted"
        assert scheduler.submit(_scope(), "run-3") == "accepted"
        assert scheduler.submit(_scope(), "run-1") == "already_queued"
        assert executor.started.wait(1)
        deadline = time.monotonic() + 1
        snapshot = scheduler.snapshot()
        while snapshot["running"] < 2 and time.monotonic() < deadline:  # type: ignore[operator]
            time.sleep(0.01)
            snapshot = scheduler.snapshot()
        assert snapshot["running"] == 2
        assert snapshot["pending"] == 1
        deadline = time.monotonic() + 1
        while executor.max_running < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor.max_running == 2
        executor.release.set()
    finally:
        scheduler.shutdown(wait=True, timeout=2)


def test_image_scheduler_rejects_after_shutdown() -> None:
    scheduler = ImageTaskScheduler(_Executor())
    scheduler.start()
    scheduler.shutdown(wait=True, timeout=1)

    assert scheduler.submit(_scope(), "run-closed") == "rejected"


def test_image_failure_does_not_block_another_run() -> None:
    executor = _SequenceExecutor(
        {"run-failed": (ErrorCode.IMAGE_VLM_UNAVAILABLE, 1)},
    )
    scheduler = ImageTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-failed") == "accepted"
        assert scheduler.submit(_scope(), "run-success") == "accepted"
        _wait_until(lambda: len(executor.calls) >= 2)
        _wait_until(lambda: scheduler.snapshot()["running"] == 0)
        assert executor.failure_records == [("run-failed", ErrorCode.IMAGE_VLM_UNAVAILABLE)]
        assert scheduler.snapshot()["completed"]
    finally:
        scheduler.shutdown(wait=True, timeout=2)


def test_retryable_image_failure_is_reenqueued() -> None:
    executor = _SequenceExecutor(
        {"run-retry": (ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 1)},
    )
    scheduler = ImageTaskScheduler(executor)
    scheduler._retry_delay_seconds = 0  # type: ignore[attr-defined]
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-retry") == "accepted"
        _wait_until(lambda: executor.calls.count("run-retry") == 2)
        _wait_until(lambda: scheduler.snapshot()["running"] == 0)
        assert executor.failure_records == [
            ("run-retry", ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        ]
    finally:
        scheduler.shutdown(wait=True, timeout=2)


def test_non_retryable_image_failure_is_not_reenqueued() -> None:
    executor = _SequenceExecutor(
        {"run-failed": (ErrorCode.IMAGE_VLM_UNAVAILABLE, 1)},
    )
    scheduler = ImageTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-failed") == "accepted"
        _wait_until(lambda: executor.calls == ["run-failed"])
        _wait_until(lambda: scheduler.snapshot()["running"] == 0)
        assert executor.calls == ["run-failed"]
    finally:
        scheduler.shutdown(wait=True, timeout=2)


def test_cancelled_pending_image_is_never_executed() -> None:
    executor = _SequenceExecutor()
    executor.cancelled.add("run-cancelled")
    scheduler = ImageTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-cancelled") == "accepted"
        _wait_until(lambda: scheduler.snapshot()["running"] == 0)
        assert executor.calls == []
    finally:
        scheduler.shutdown(wait=True, timeout=2)


def test_scheduler_survives_failure_recording_exception() -> None:
    class _BrokenFailureExecutor(_SequenceExecutor):
        def stage_failed(self, scope: Scope, run_id: str, error: VideoDemoError) -> bool:
            del scope, run_id, error
            raise RuntimeError("记录失败")

    executor = _BrokenFailureExecutor(
        {"run-broken": (ErrorCode.IMAGE_VLM_UNAVAILABLE, 1)},
    )
    scheduler = ImageTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-broken") == "accepted"
        _wait_until(lambda: scheduler.snapshot()["running"] == 0)
        assert scheduler.submit(_scope(), "run-after-broken") == "accepted"
        _wait_until(lambda: scheduler.snapshot()["completed"])
        assert executor.calls[-1] == "run-after-broken"
    finally:
        scheduler.shutdown(wait=True, timeout=2)
