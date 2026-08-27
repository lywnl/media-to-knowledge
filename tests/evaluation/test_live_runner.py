from __future__ import annotations

from pathlib import Path

import pytest

from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import EvidenceStore
from video_demo.evaluation.live_runner import LiveValidationRunner


def test_live_validation_runner_exposes_active_entries() -> None:
    assert hasattr(LiveValidationRunner, "run_chapter_vlm")
    assert hasattr(LiveValidationRunner, "run_local_model_stack")
    assert not hasattr(LiveValidationRunner, "run_baidu")
    assert not hasattr(LiveValidationRunner, "run_qwen")


def test_invalid_run_id_is_rejected_before_live_work(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )
    with pytest.raises(VideoDemoError) as raised:
        runner.run_workspace_chapter_vlm("../escape")
    assert raised.value.code == ErrorCode.INVALID_PATH_COMPONENT


def test_workspace_chapter_vlm_without_dataset_is_not_run(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    runner = LiveValidationRunner(
        Settings(workspace_root=tmp_path),
        EvidenceStore(tmp_path, runtime_root),
    )
    issues = runner._preflight_chapter_vlm("run-001", None)
    assert ErrorCode.INVALID_CONFIGURATION in issues
    assert ErrorCode.LIVE_AUTHORIZED_CHAPTER_FRAMES_UNAVAILABLE in issues
