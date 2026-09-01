from __future__ import annotations

import threading
import time

from video_demo.application.audio_scheduler import AudioTaskScheduler
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import AudioStageName
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
        del scope
        self.calls.append(("TRANSCRIPTION", run_id))
        self.transcription_started.set()
        self.release_transcription.wait(2)
        return {"run_id": run_id}

    def run_llm(self, scope: Scope, run_id: str, checkpoint: object) -> None:
        del scope, checkpoint
        self.calls.append(("LLM", run_id))
        self.llm_started.set()

    def load_checkpoint(self, scope: Scope, run_id: str) -> object | None:
        del scope, run_id
        return None

    def is_cancelled(self, scope: Scope, run_id: str) -> bool:
        del scope
        return run_id in self.cancelled

    def stage_succeeded(self, scope: Scope, run_id: str, stage: str, result: object) -> None:
        del scope, run_id, stage, result

    def stage_failed(self, scope: Scope, run_id: str, stage: str, error: VideoDemoError) -> bool:
        del scope, run_id, stage, error
        return False

    def mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> None:
        del scope
        self.recovery_failures.append((run_id, stage, error.code.value))


def _scope() -> Scope:
    return Scope("tenant", "application", "kb")


def test_audio_scheduler_limits_slots_and_handoffs_transcription_to_llm() -> None:
    executor = _Executor()
    scheduler = AudioTaskScheduler(executor, transcription_concurrency=2, llm_concurrency=2)
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


def test_audio_scheduler_rejects_submission_after_shutdown() -> None:
    scheduler = AudioTaskScheduler(_Executor())
    scheduler.start()
    scheduler.shutdown(wait=True, timeout=1)

    assert scheduler.submit(_scope(), "run-closed") == "rejected"


def test_audio_scheduler_recovers_checkpointed_run_into_llm_queue() -> None:
    executor = _Executor()
    executor.load_checkpoint = lambda _scope, _run_id: {"checkpoint": True}  # type: ignore[method-assign]
    scheduler = AudioTaskScheduler(executor)

    assert scheduler.recover(((_scope(), "run-recover"),)) == 1
    assert scheduler.snapshot()["queues"]["LLM"]["pending"] == 1  # type: ignore[index]


def test_audio_scheduler_does_not_fallback_to_transcription_without_checkpoint() -> None:
    executor = _Executor()
    scheduler = AudioTaskScheduler(executor)

    assert scheduler.recover(((_scope(), "run-missing", AudioStageName.LLM),)) == 0
    assert executor.recovery_failures == [
        ("run-missing", "LLM", ErrorCode.AUDIO_RESULT_NOT_READY.value),
    ]
    assert scheduler.snapshot()["queues"]["TRANSCRIPTION"]["pending"] == 0  # type: ignore[index]
