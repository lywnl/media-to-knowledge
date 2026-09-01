"""图片理解任务的进程内双并发调度器。"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread, current_thread
from typing import Literal, Protocol

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.scope import Scope

IMAGE_CONCURRENCY = 2
_LOGGER = logging.getLogger(__name__)


class ImageStageExecutor(Protocol):
    def run(self, scope: Scope, run_id: str) -> None: ...

    def is_cancelled(self, scope: Scope, run_id: str) -> bool: ...

    def stage_failed(self, scope: Scope, run_id: str, error: VideoDemoError) -> bool: ...

    def close(self) -> None: ...


@dataclass
class _ImageQueue:
    concurrency: int = IMAGE_CONCURRENCY

    def __post_init__(self) -> None:
        self.pending: deque[tuple[Scope, str]] = deque()
        self.pending_ids: set[str] = set()
        self.running_ids: set[str] = set()

    @staticmethod
    def key(scope: Scope, run_id: str) -> str:
        return f"{scope.tenant_id}/{scope.application_id}/{scope.knowledge_base_id}/{run_id}"


class ImageTaskScheduler:
    """在 FastAPI 进程内最多同时执行两个图片 Run。"""

    def __init__(
        self,
        executor: ImageStageExecutor,
        *,
        concurrency: int = IMAGE_CONCURRENCY,
        logger: logging.Logger | None = None,
    ) -> None:
        if concurrency != IMAGE_CONCURRENCY:
            raise ValueError("图片并发数必须固定为 2")
        self._executor = executor
        self._queue = _ImageQueue(int(concurrency))
        self._condition = Condition()
        self._logger = logger or _LOGGER
        self._accept_new_work = True
        self._shutdown_requested = False
        self._started = False
        self._dispatcher: Thread | None = None
        self._job_threads: set[Thread] = set()
        self._completed: dict[str, str] = {}
        self._retry_delay_seconds = 5.0

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._accept_new_work = True
            self._shutdown_requested = False
            self._dispatcher = Thread(
                target=self._dispatch_loop,
                daemon=True,
                name="image-dispatcher",
            )
            self._dispatcher.start()
        self._logger.info("image scheduler started concurrency=%d", self._queue.concurrency)

    def submit(
        self,
        scope: Scope,
        run_id: str,
    ) -> Literal["accepted", "already_queued", "rejected"]:
        key = self._queue.key(scope, run_id)
        with self._condition:
            if not self._accept_new_work or self._shutdown_requested:
                return "rejected"
            if key in self._queue.pending_ids or key in self._queue.running_ids:
                return "already_queued"
            self._queue.pending.append((scope, run_id))
            self._queue.pending_ids.add(key)
            self._condition.notify_all()
            pending = len(self._queue.pending)
        self._logger.info("image scheduler enqueue run_id=%s pending=%d", run_id, pending)
        return "accepted"

    def recover(self, items: tuple[tuple[Scope, str], ...] = ()) -> int:
        recovered = 0
        for scope, run_id in items:
            if self.submit(scope, run_id) == "accepted":
                recovered += 1
        self._logger.info("image scheduler recovered count=%d", recovered)
        return recovered

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "started": self._started,
                "accept_new_work": self._accept_new_work,
                "shutdown_requested": self._shutdown_requested,
                "pending": len(self._queue.pending),
                "running": len(self._queue.running_ids),
                "concurrency": self._queue.concurrency,
                "dispatch_threads": int(
                    self._dispatcher is not None and self._dispatcher.is_alive()
                ),
                "job_threads": sum(thread.is_alive() for thread in self._job_threads),
                "completed": dict(self._completed),
            }

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
        with self._condition:
            self._accept_new_work = False
            self._shutdown_requested = True
            self._queue.pending.clear()
            self._queue.pending_ids.clear()
            self._condition.notify_all()
            dispatcher = self._dispatcher
        if wait and dispatcher is not None:
            deadline = None if timeout is None else time.monotonic() + timeout
            dispatcher.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
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
        self._logger.info("image scheduler stopped")

    def _dispatch_loop(self) -> None:
        while True:
            item = self._take()
            if item is None:
                return
            scope, run_id = item
            if self._is_cancelled(scope, run_id):
                self._finish_queue_item(scope, run_id)
                self._logger.info("image scheduler skip cancelled run_id=%s", run_id)
                continue
            with self._condition:
                slot = len(self._queue.running_ids)
            thread = Thread(
                target=self._execute,
                args=(scope, run_id),
                daemon=True,
                name=f"image-run-{run_id}",
            )
            with self._condition:
                self._job_threads.add(thread)
            self._logger.info(
                "image scheduler start run_id=%s slot=%d/%d",
                run_id,
                slot,
                self._queue.concurrency,
            )
            thread.start()

    def _take(self) -> tuple[Scope, str] | None:
        with self._condition:
            while True:
                if (
                    self._shutdown_requested
                    and not self._queue.pending
                    and not self._queue.running_ids
                ):
                    return None
                if self._queue.pending and len(self._queue.running_ids) < self._queue.concurrency:
                    scope, run_id = self._queue.pending.popleft()
                    self._queue.pending_ids.discard(self._queue.key(scope, run_id))
                    self._queue.running_ids.add(self._queue.key(scope, run_id))
                    return scope, run_id
                self._condition.wait()

    def _execute(self, scope: Scope, run_id: str) -> None:
        key = self._queue.key(scope, run_id)
        started_at = time.monotonic()
        retry = False
        try:
            self._executor.run(scope, run_id)
            with self._condition:
                self._completed[key] = "SUCCEEDED"
        except VideoDemoError as error:
            retry = self._record_failure(scope, run_id, error)
            self._logger.warning(
                "image scheduler failed run_id=%s error_code=%s retry=%s",
                run_id,
                error.code,
                retry,
            )
        except Exception as system_error:
            failure = VideoDemoError(ErrorCode.SYSTEM_FAILURE, "图片阶段发生未分类系统错误")
            retry = self._record_failure(scope, run_id, failure)
            self._logger.exception(
                "image scheduler crashed run_id=%s",
                run_id,
                exc_info=system_error,
            )
        finally:
            with self._condition:
                self._queue.running_ids.discard(key)
                self._job_threads.discard(current_thread())
                self._condition.notify_all()
            if retry and not self._shutdown_requested:
                Thread(
                    target=self._delayed_enqueue,
                    args=(scope, run_id),
                    daemon=True,
                    name=f"image-retry-{run_id}",
                ).start()
            self._logger.info(
                "image scheduler finish run_id=%s duration_ms=%d retry=%s",
                run_id,
                round((time.monotonic() - started_at) * 1_000),
                retry,
            )

    def _record_failure(self, scope: Scope, run_id: str, error: VideoDemoError) -> bool:
        try:
            return self._executor.stage_failed(scope, run_id, error)
        except VideoDemoError:
            self._logger.exception("image scheduler failure recording failed run_id=%s", run_id)
            return False
        except Exception:
            # 失败写回属于状态收口；即使数据库或其他基础设施异常，也不能
            # 让单个图片任务线程把整个调度器拖死。
            self._logger.exception("image scheduler failure recording crashed run_id=%s", run_id)
            return False

    def _delayed_enqueue(self, scope: Scope, run_id: str) -> None:
        deadline = time.monotonic() + self._retry_delay_seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            with self._condition:
                if self._shutdown_requested:
                    return
        self.submit(scope, run_id)

    def _finish_queue_item(self, scope: Scope, run_id: str) -> None:
        with self._condition:
            self._queue.running_ids.discard(self._queue.key(scope, run_id))
            self._condition.notify_all()

    def _is_cancelled(self, scope: Scope, run_id: str) -> bool:
        try:
            return self._executor.is_cancelled(scope, run_id)
        except VideoDemoError:
            return False
        except Exception:
            # 取消预检是调度边界，数据库瞬时故障不能让 dispatcher 线程退出；
            # 执行器后续仍会在领取租约和业务处理边界再次校验状态。
            self._logger.exception(
                "image scheduler cancel check failed run_id=%s",
                run_id,
            )
            return False


__all__ = ["IMAGE_CONCURRENCY", "ImageStageExecutor", "ImageTaskScheduler"]
