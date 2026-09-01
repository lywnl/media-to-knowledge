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
