from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from video_demo.application.image_pipeline_executor import ImageStagePipelineExecutor
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.repositories import ClaimedJob
from video_demo.persistence.scope import Scope


class _Handler:
    def process(self, job, *, is_cancel_requested):
        if is_cancel_requested():
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已取消")
        return object()


class _Database:
    @contextmanager
    def session(self):
        yield object()


class _Repository:
    def __init__(self, session) -> None:
        del session

    def claim_image_run(self, scope, run_id, worker_id, *, lease_seconds):
        return ClaimedJob(1, "job-1", run_id, worker_id, 1, 3, scope)

    def is_cancel_requested(self, *args, **kwargs):
        return True


def test_executor_respects_cancellation_before_handler(monkeypatch) -> None:
    monkeypatch.setattr(
        "video_demo.application.image_pipeline_executor.JobRepository",
        _Repository,
    )
    executor = ImageStagePipelineExecutor(
        _Database(),
        _Handler(),
        runtime_root=Path("/tmp"),
    )

    with pytest.raises(VideoDemoError) as error:
        executor.run(Scope("t", "a", "k"), "run-1")

    assert error.value.code == ErrorCode.JOB_CANCELLED
