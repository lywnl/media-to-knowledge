from __future__ import annotations

import argparse
import math
import socket
import time
import uuid
from typing import Protocol

from video_demo.application.composition import build_worker
from video_demo.config import Settings


class WorkerLoop(Protocol):
    def run_once(self) -> bool: ...

    def close(self) -> None: ...


def run_worker(
    worker: WorkerLoop,
    *,
    once: bool,
    poll_interval_seconds: float,
) -> None:
    try:
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds 必须大于 0")
        while True:
            claimed = worker.run_once()
            if once:
                return
            if not claimed:
                time.sleep(poll_interval_seconds)
    finally:
        worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="视频理解可靠任务 Worker")
    parser.add_argument("--once", action="store_true", help="只尝试领取一次任务")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="空闲轮询间隔秒数")
    parser.add_argument("--worker-id", default=None, help="稳定 Worker 标识")
    arguments = parser.parse_args()
    worker_id = arguments.worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"
    run_worker(
        build_worker(Settings(), worker_id=worker_id),
        once=arguments.once,
        poll_interval_seconds=arguments.poll_interval,
    )


if __name__ == "__main__":
    main()
