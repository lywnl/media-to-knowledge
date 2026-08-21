from __future__ import annotations

import pytest

from video_demo.worker_main import run_worker


def test_worker_once_exits_after_one_claim_attempt() -> None:
    calls: list[str] = []

    class Worker:
        def run_once(self) -> bool:
            calls.append("run_once")
            return False

        def close(self) -> None:
            calls.append("close")

    run_worker(Worker(), once=True, poll_interval_seconds=1.0)  # type: ignore[arg-type]

    assert calls == ["run_once", "close"]


@pytest.mark.parametrize("raises", [False, True])
def test_worker_loop_closes_lifecycle_resources_exactly_once(raises: bool) -> None:
    calls: list[str] = []

    class Worker:
        def run_once(self) -> bool:
            calls.append("run_once")
            if raises:
                raise RuntimeError("worker failed")
            return False

        def close(self) -> None:
            calls.append("close")

    worker = Worker()
    if raises:
        with pytest.raises(RuntimeError, match="worker failed"):
            run_worker(worker, once=True, poll_interval_seconds=1.0)
    else:
        run_worker(worker, once=True, poll_interval_seconds=1.0)

    assert calls == ["run_once", "close"]


@pytest.mark.parametrize(
    "poll_interval_seconds",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0],
)
def test_worker_closes_when_poll_interval_is_invalid(
    poll_interval_seconds: float,
) -> None:
    calls: list[str] = []

    class Worker:
        def run_once(self) -> bool:
            calls.append("run_once")
            return False

        def close(self) -> None:
            calls.append("close")

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        run_worker(  # type: ignore[arg-type]
            Worker(),
            once=True,
            poll_interval_seconds=poll_interval_seconds,
        )

    assert calls == ["close"]


def test_worker_loop_closes_when_interrupted() -> None:
    calls: list[str] = []

    class Worker:
        def run_once(self) -> bool:
            calls.append("run_once")
            if calls.count("run_once") == 2:
                raise KeyboardInterrupt
            return True

        def close(self) -> None:
            calls.append("close")

    with pytest.raises(KeyboardInterrupt):
        run_worker(Worker(), once=False, poll_interval_seconds=1.0)  # type: ignore[arg-type]

    assert calls == ["run_once", "run_once", "close"]
