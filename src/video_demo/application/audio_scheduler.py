"""音频转写和 LLM 阶段的进程内可恢复调度器。"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Thread, current_thread
from typing import Literal, Protocol

from video_demo.application.checkpoint_contracts import CheckpointStaleError
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.models import AudioStageName
from video_demo.persistence.scope import Scope

_LOGGER = logging.getLogger(__name__)


class AudioStageExecutor(Protocol):
    """调度器只依赖音频阶段执行器，不感知流水线内部细节。"""

    def run_transcription(self, scope: Scope, run_id: str) -> object: ...

    def run_llm(self, scope: Scope, run_id: str, checkpoint: object) -> None: ...

    def load_checkpoint(self, scope: Scope, run_id: str) -> object | None: ...

    def reset_stale_checkpoint(self, scope: Scope, run_id: str) -> None: ...

    def is_cancelled(self, scope: Scope, run_id: str) -> bool: ...

    def stage_succeeded(self, scope: Scope, run_id: str, stage: str, result: object) -> None: ...

    def stage_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> bool: ...


@dataclass
class _StageQueue:
    name: str
    concurrency: int

    def __post_init__(self) -> None:
        self.concurrency = max(1, int(self.concurrency))
        self.pending: deque[tuple[Scope, str]] = deque()
        self.pending_ids: set[str] = set()
        self.running_ids: set[str] = set()

    def key(self, scope: Scope, run_id: str) -> str:
        return f"{scope.tenant_id}/{scope.application_id}/{scope.knowledge_base_id}/{run_id}"

    def clear(self) -> int:
        count = len(self.pending)
        self.pending.clear()
        self.pending_ids.clear()
        return count


class AudioTaskScheduler:
    """音频双阶段调度器，独立维护 TRANSCRIPTION 和 LLM 两个队列。"""

    def __init__(
        self,
        executor: AudioStageExecutor,
        *,
        transcription_concurrency: int = 2,
        llm_concurrency: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        if transcription_concurrency < 1 or llm_concurrency < 1:
            raise ValueError("音频阶段并发数必须大于 0")
        self._executor = executor
        self._transcription = _StageQueue("TRANSCRIPTION", transcription_concurrency)
        self._llm = _StageQueue("LLM", llm_concurrency)
        self._condition = Condition()
        self._logger = logger or _LOGGER
        self._accept_new_work = True
        self._shutdown_requested = False
        self._started = False
        self._dispatch_threads: list[Thread] = []
        self._job_threads: set[Thread] = set()
        self._completed: dict[str, str] = {}
        self._checkpoints: dict[str, object] = {}
        self._retry_delay_seconds = 5.0

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._accept_new_work = True
            self._shutdown_requested = False
            self._dispatch_threads = [
                Thread(
                    target=self._dispatch_loop,
                    args=(self._transcription, self._run_transcription),
                    daemon=True,
                    name="audio-transcription-dispatcher",
                ),
                Thread(
                    target=self._dispatch_loop,
                    args=(self._llm, self._run_llm),
                    daemon=True,
                    name="audio-llm-dispatcher",
                ),
            ]
            for thread in self._dispatch_threads:
                thread.start()
        self._logger.info(
            "audio scheduler started transcription_concurrency=%d llm_concurrency=%d",
            self._transcription.concurrency,
            self._llm.concurrency,
        )

    def submit(
        self,
        scope: Scope,
        run_id: str,
        stage: AudioStageName = AudioStageName.TRANSCRIPTION,
    ) -> Literal["accepted", "already_queued", "rejected"]:
        queue = self._queue_for_stage(stage)
        key = queue.key(scope, run_id)
        with self._condition:
            if not self._accept_new_work or self._shutdown_requested:
                return "rejected"
            if self._contains_locked(key):
                return "already_queued"
            queue.pending.append((scope, run_id))
            queue.pending_ids.add(key)
            self._condition.notify_all()
        self._logger.info("audio scheduler enqueue run_id=%s stage=%s", run_id, queue.name)
        return "accepted"

    def _queue_for_stage(self, stage: AudioStageName) -> _StageQueue:
        if stage == AudioStageName.TRANSCRIPTION:
            return self._transcription
        if stage == AudioStageName.LLM:
            return self._llm
        raise ValueError(f"音频阶段不受调度器支持：{stage}")

    def recover(
        self,
        items: tuple[
            tuple[Scope, str] | tuple[Scope, str, AudioStageName],
            ...,
        ] = (),
    ) -> int:
        recovered = 0
        for item in items:
            scope, run_id = item[:2]
            key = self._transcription.key(scope, run_id)
            requested_stage = item[2] if len(item) == 3 else None
            checkpoint = None
            if requested_stage != AudioStageName.TRANSCRIPTION:
                try:
                    checkpoint = self._executor.load_checkpoint(scope, run_id)
                    if requested_stage == AudioStageName.LLM and checkpoint is None:
                        error = VideoDemoError(
                            ErrorCode.AUDIO_RESULT_NOT_READY,
                            "转写阶段快照不存在",
                        )
                        self._mark_recovery_failed(scope, run_id, requested_stage, error)
                        continue
                except CheckpointStaleError:
                    self._executor.reset_stale_checkpoint(scope, run_id)
                    with self._condition:
                        if not self._contains_locked(key):
                            self._transcription.pending.append((scope, run_id))
                            self._transcription.pending_ids.add(key)
                            recovered += 1
                            self._condition.notify_all()
                    self._logger.info(
                        "audio scheduler stale checkpoint reset run_id=%s stage=%s",
                        run_id,
                        requested_stage or "UNKNOWN",
                    )
                    continue
                except VideoDemoError as error:
                    self._mark_recovery_failed(scope, run_id, requested_stage, error)
                    self._logger.error(
                        "audio scheduler recovery failed run_id=%s stage=%s error_code=%s",
                        run_id,
                        requested_stage or "UNKNOWN",
                        error.code,
                    )
                    continue
            with self._condition:
                if self._contains_locked(key):
                    continue
                target = (
                    self._transcription
                    if requested_stage == AudioStageName.TRANSCRIPTION
                    else self._llm if checkpoint is not None else self._transcription
                )
                target.pending.append((scope, run_id))
                target.pending_ids.add(key)
                if checkpoint is not None:
                    self._checkpoints[key] = checkpoint
                recovered += 1
                self._condition.notify_all()
            self._logger.info(
                "audio scheduler recovered run_id=%s stage=%s",
                run_id,
                target.name,
            )
        self._logger.info("audio scheduler recovered count=%d", recovered)
        return recovered

    def _mark_recovery_failed(
        self,
        scope: Scope,
        run_id: str,
        stage: AudioStageName | None,
        error: VideoDemoError,
    ) -> None:
        if stage is None:
            return
        handler = getattr(self._executor, "mark_recovery_failed", None)
        if callable(handler):
            try:
                handler(scope, run_id, stage.value, error)
            except Exception:
                self._logger.exception(
                    "audio scheduler recovery failure recording crashed run_id=%s stage=%s",
                    run_id,
                    stage.value,
                )

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "started": self._started,
                "accept_new_work": self._accept_new_work,
                "shutdown_requested": self._shutdown_requested,
                "dispatch_threads": sum(thread.is_alive() for thread in self._dispatch_threads),
                "job_threads": sum(thread.is_alive() for thread in self._job_threads),
                "completed": dict(self._completed),
                "queues": {
                    self._transcription.name: self._queue_snapshot(self._transcription),
                    self._llm.name: self._queue_snapshot(self._llm),
                },
            }

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
        with self._condition:
            self._accept_new_work = False
            self._shutdown_requested = True
            self._transcription.clear()
            self._llm.clear()
            self._condition.notify_all()
        if wait:
            deadline = None if timeout is None else time.monotonic() + timeout
            for thread in self._dispatch_threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                thread.join(remaining)
            while True:
                with self._condition:
                    running = tuple(thread for thread in self._job_threads if thread.is_alive())
                if not running:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                running[0].join(None if deadline is None else max(0.0, deadline - time.monotonic()))
            with self._condition:
                self._started = False
        close = getattr(self._executor, "close", None)
        if callable(close):
            close()
        self._logger.info("audio scheduler stopped")

    def _contains_locked(self, key: str) -> bool:
        return any(
            key in queue.pending_ids or key in queue.running_ids
            for queue in (self._transcription, self._llm)
        )

    @staticmethod
    def _queue_snapshot(queue: _StageQueue) -> dict[str, int]:
        return {
            "concurrency": queue.concurrency,
            "pending": len(queue.pending),
            "running": len(queue.running_ids),
        }

    def _dispatch_loop(
        self,
        queue: _StageQueue,
        runner: Callable[[Scope, str], object],
    ) -> None:
        while True:
            item = self._take(queue)
            if item is None:
                return
            scope, run_id = item
            if self._is_cancelled(scope, run_id):
                with self._condition:
                    queue.running_ids.discard(queue.key(scope, run_id))
                    self._condition.notify_all()
                self._logger.info(
                    "audio scheduler skip cancelled run_id=%s stage=%s",
                    run_id,
                    queue.name,
                )
                continue
            if hasattr(self._executor, "mark_stage_started"):
                self._executor.mark_stage_started(scope, run_id, queue.name)
            thread = Thread(
                target=self._execute,
                args=(queue, scope, run_id, runner),
                daemon=True,
                name=f"audio-{queue.name.lower()}-{run_id}",
            )
            with self._condition:
                self._job_threads.add(thread)
                slot = len(queue.running_ids)
            self._logger.info(
                "audio scheduler start run_id=%s stage=%s slot=%d/%d",
                run_id,
                queue.name,
                slot,
                queue.concurrency,
            )
            thread.start()

    def _take(self, queue: _StageQueue) -> tuple[Scope, str] | None:
        with self._condition:
            while True:
                if self._shutdown_requested and not queue.pending and not queue.running_ids:
                    return None
                if not self._accept_new_work and not queue.pending and not queue.running_ids:
                    return None
                if queue.pending and len(queue.running_ids) < queue.concurrency:
                    scope, run_id = queue.pending.popleft()
                    key = queue.key(scope, run_id)
                    queue.pending_ids.discard(key)
                    queue.running_ids.add(key)
                    self._logger.info(
                        "audio scheduler dequeue run_id=%s stage=%s pending=%d running=%d/%d",
                        run_id,
                        queue.name,
                        len(queue.pending),
                        len(queue.running_ids),
                        queue.concurrency,
                    )
                    return scope, run_id
                self._condition.wait()

    def _execute(
        self,
        queue: _StageQueue,
        scope: Scope,
        run_id: str,
        runner: Callable[[Scope, str], object],
    ) -> None:
        key = queue.key(scope, run_id)
        retry = False
        started_at = time.monotonic()
        try:
            result = runner(scope, run_id)
            if queue is self._transcription and result is not None:
                with self._condition:
                    if not self._shutdown_requested:
                        self._llm.pending.append((scope, run_id))
                        self._llm.pending_ids.add(key)
                        self._checkpoints[key] = result
                        self._condition.notify_all()
                self._logger.info("audio scheduler handoff run_id=%s stage=LLM", run_id)
            else:
                with self._condition:
                    self._completed[key] = queue.name
            if hasattr(self._executor, "stage_succeeded"):
                self._executor.stage_succeeded(scope, run_id, queue.name, result)
        except CheckpointStaleError:
            self._executor.reset_stale_checkpoint(scope, run_id)
            with self._condition:
                self._enqueue_locked(self._transcription, scope, run_id)
                self._condition.notify_all()
            self._logger.info("audio scheduler stale checkpoint requeued run_id=%s", run_id)
        except VideoDemoError as error:
            retry = self._record_failure(scope, run_id, queue.name, error)
            self._logger.warning(
                "audio scheduler stage failed run_id=%s stage=%s error_code=%s retry=%s",
                run_id,
                queue.name,
                error.code,
                retry,
            )
        except Exception:
            system_error = VideoDemoError(
                ErrorCode.SYSTEM_FAILURE,
                "音频阶段发生未分类系统错误",
            )
            retry = self._record_failure(scope, run_id, queue.name, system_error)
            self._logger.exception(
                "audio scheduler stage crashed run_id=%s stage=%s retry=%s",
                run_id,
                queue.name,
                retry,
            )
        finally:
            with self._condition:
                queue.running_ids.discard(key)
                self._job_threads.discard(current_thread())
                running = len(queue.running_ids)
                if retry and not self._shutdown_requested:
                    Thread(
                        target=self._delayed_enqueue,
                        args=(queue, scope, run_id),
                        daemon=True,
                        name=f"audio-{queue.name.lower()}-retry-{run_id}",
                    ).start()
                self._condition.notify_all()
            self._logger.info(
                "audio scheduler finish run_id=%s stage=%s duration_ms=%d retry=%s running=%d/%d",
                run_id,
                queue.name,
                round((time.monotonic() - started_at) * 1_000),
                retry,
                running,
                queue.concurrency,
            )

    def _delayed_enqueue(self, queue: _StageQueue, scope: Scope, run_id: str) -> None:
        deadline = time.monotonic() + self._retry_delay_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))
            with self._condition:
                if self._shutdown_requested:
                    return
        with self._condition:
            if self._shutdown_requested:
                return
            self._enqueue_locked(queue, scope, run_id)
            self._condition.notify_all()

    def _enqueue_locked(self, queue: _StageQueue, scope: Scope, run_id: str) -> None:
        key = queue.key(scope, run_id)
        if key in queue.pending_ids or key in queue.running_ids:
            return
        queue.pending.append((scope, run_id))
        queue.pending_ids.add(key)
        self._logger.info("audio scheduler retry enqueue run_id=%s stage=%s", run_id, queue.name)

    def _run_transcription(self, scope: Scope, run_id: str) -> object:
        return self._executor.run_transcription(scope, run_id)

    def _run_llm(self, scope: Scope, run_id: str) -> object:
        key = self._llm.key(scope, run_id)
        with self._condition:
            checkpoint = self._checkpoints.pop(key, None)
        checkpoint = checkpoint or self._executor.load_checkpoint(scope, run_id)
        if checkpoint is None:
            raise VideoDemoError(ErrorCode.AUDIO_RESULT_NOT_READY, "转写阶段快照不存在")
        self._executor.run_llm(scope, run_id, checkpoint)
        return None

    def _record_failure(
        self,
        scope: Scope,
        run_id: str,
        stage: str,
        error: VideoDemoError,
    ) -> bool:
        try:
            return self._executor.stage_failed(scope, run_id, stage, error)
        except Exception:
            self._logger.exception(
                "audio scheduler failure recording crashed run_id=%s stage=%s",
                run_id,
                stage,
            )
            return False

    def _is_cancelled(self, scope: Scope, run_id: str) -> bool:
        checker = getattr(self._executor, "is_cancelled", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(scope, run_id))
        except VideoDemoError:
            return False


__all__ = ["AudioStageExecutor", "AudioTaskScheduler"]
