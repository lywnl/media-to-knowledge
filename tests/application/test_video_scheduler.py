from __future__ import annotations

import threading
import time

from video_demo.application.video_scheduler import VideoTaskScheduler
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import VideoStageName
from video_demo.persistence.scope import Scope


class _Executor:
    def __init__(self) -> None:
        self.transcription_started = threading.Event()
        self.release_transcription = threading.Event()
        self.llm_started = threading.Event()
        self.calls: list[tuple[str, str]] = []
        self.recovery_failures: list[tuple[str, str, str]] = []
        self.cancelled: set[str] = set()

    def run_transcription(self, scope: Scope, run_id: str) -> object:
        self.calls.append(("TRANSCRIPTION", run_id))
        self.transcription_started.set()
        self.release_transcription.wait(2)
        return {"run_id": run_id}

    def run_llm(self, scope: Scope, run_id: str, checkpoint: object) -> None:
        self.calls.append(("LLM", run_id))
        self.llm_started.set()

    def load_checkpoint(self, scope: Scope, run_id: str) -> object | None:
        return None

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        del scope
        return run_id in self.cancelled

    def stage_succeeded(self, scope: Scope, run_id: str, stage: str, result: object) -> None:
        del scope, run_id, stage, result

    def stage_failed(self, scope: Scope, run_id: str, stage: str, error: object) -> bool:
        del scope, run_id, stage, error
        return False

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> None:
        self.recovery_failures.append((run_id, stage, error.code.value))


def _scope() -> Scope:
    return Scope("tenant", "application", "kb")


def test_scheduler_limits_slots_and_handoffs_transcription_to_llm() -> None:
    executor = _Executor()
    scheduler = VideoTaskScheduler(executor, transcription_concurrency=2, llm_concurrency=2)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-1") == "accepted"
        assert scheduler.submit(_scope(), "run-2") == "accepted"
        assert scheduler.submit(_scope(), "run-1") == "already_queued"
        assert executor.transcription_started.wait(1)
        deadline = time.monotonic() + 1
        snapshot = scheduler.snapshot()
        while snapshot["queues"]["TRANSCRIPTION"]["running"] < 2:  # type: ignore[index]
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
            snapshot = scheduler.snapshot()
        assert snapshot["queues"]["TRANSCRIPTION"]["running"] == 2  # type: ignore[index]
        executor.release_transcription.set()
        assert executor.llm_started.wait(1)
    finally:
        scheduler.shutdown(wait=True, timeout=2)

    assert {stage for stage, _ in executor.calls} == {"TRANSCRIPTION", "LLM"}


def test_scheduler_rejects_submission_after_shutdown() -> None:
    scheduler = VideoTaskScheduler(_Executor())
    scheduler.start()
    scheduler.shutdown(wait=True, timeout=1)

    assert scheduler.submit(_scope(), "run-closed") == "rejected"


def test_scheduler_recovers_checkpointed_run_into_llm_queue() -> None:
    executor = _Executor()
    executor.load_checkpoint = lambda _scope, _run_id: {"checkpoint": True}  # type: ignore[method-assign]
    scheduler = VideoTaskScheduler(executor)
    assert scheduler.recover(((_scope(), "run-recover"),)) == 1
    snapshot = scheduler.snapshot()

    assert snapshot["queues"]["LLM"]["pending"] == 1  # type: ignore[index]


def test_scheduler_does_not_fallback_to_transcription_when_llm_checkpoint_is_missing() -> None:
    executor = _Executor()
    scheduler = VideoTaskScheduler(executor)

    assert scheduler.recover(((_scope(), "run-missing", VideoStageName.LLM),)) == 0
    assert executor.recovery_failures == [
        ("run-missing", "LLM", ErrorCode.VIDEO_RESULT_NOT_READY.value),
    ]
    snapshot = scheduler.snapshot()
    assert snapshot["queues"]["TRANSCRIPTION"]["pending"] == 0  # type: ignore[index]


def test_scheduler_does_not_leave_running_threads_after_shutdown() -> None:
    executor = _Executor()
    scheduler = VideoTaskScheduler(executor)
    scheduler.start()
    assert scheduler.submit(_scope(), "run-1") == "accepted"
    assert executor.transcription_started.wait(1)
    scheduler.shutdown(wait=False)
    executor.release_transcription.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and scheduler.snapshot()["job_threads"]:
        time.sleep(0.01)
    scheduler.shutdown(wait=True, timeout=1)
    assert scheduler.snapshot()["job_threads"] == 0


def test_scheduler_skips_cancelled_pending_run() -> None:
    executor = _Executor()
    executor.cancelled.add("run-cancelled")
    scheduler = VideoTaskScheduler(executor)
    scheduler.start()
    try:
        assert scheduler.submit(_scope(), "run-cancelled") == "accepted"
        time.sleep(0.05)
        assert executor.calls == []
        assert scheduler.snapshot()["queues"]["TRANSCRIPTION"]["running"] == 0  # type: ignore[index]
    finally:
        scheduler.shutdown(wait=True, timeout=1)
