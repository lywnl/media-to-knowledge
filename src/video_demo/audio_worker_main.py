from __future__ import annotations

import argparse
import socket
import time
import uuid

from video_demo.application.composition import build_audio_worker
from video_demo.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="音频理解可靠任务 Worker")
    parser.add_argument("--once", action="store_true", help="只尝试领取一次任务")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args()
    worker = build_audio_worker(
        Settings(),
        worker_id=args.worker_id or f"audio-{socket.gethostname()}-{uuid.uuid4().hex[:12]}",
    )
    try:
        while True:
            claimed = worker.run_once()
            if args.once:
                return
            if not claimed:
                time.sleep(args.poll_interval)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
