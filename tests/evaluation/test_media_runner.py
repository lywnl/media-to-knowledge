from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import (
    EvidenceStore,
    RealMediaRawReport,
    load_machine_evidence,
)
from video_demo.evaluation.report import GateStatus


def _runner(tmp_path: Path):
    from video_demo.evaluation.media_runner import RealMediaRunner

    runtime = tmp_path / ".codex" / "video-rag-demo"
    runtime.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    return RealMediaRunner(settings, EvidenceStore(tmp_path, runtime)), runtime


@pytest.fixture(autouse=True)
def _stable_implementation_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    import video_demo.evaluation.gate as gate_module
    import video_demo.evaluation.media_runner as runner_module

    monkeypatch.setattr(
        gate_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )
    monkeypatch.setattr(
        runner_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )


def _complete_dependencies(runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_resolve_binary",
        lambda name: Path("/opt/homebrew/bin") / name,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())


def _successful_versions(runner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_run_version",
        lambda name, _path: (0, f"{name} version test\n".encode(), b""),
    )


def test_preflight_only_requires_ffmpeg_and_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    import video_demo.evaluation.gate as gate_module
    import video_demo.evaluation.media_runner as runner_module

    monkeypatch.setattr(
        gate_module,
        "_current_real_media_implementation_sha256",
        lambda _root: "a" * 64,
    )
    monkeypatch.setattr(
        runner_module,
        "_current_real_media_implementation_sha256",
        lambda _root: "a" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_binary",
        lambda name: None if name == "ffmpeg" else Path("/opt/homebrew/bin") / name,
    )

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.NOT_RUN
    report = load_machine_evidence(
        runtime / "eval/reports/run-1/real-media.json", workspace_root=tmp_path
    )
    assert report.status == GateStatus.NOT_RUN
    raw = json.loads((runtime / "eval/reports/run-1/preflight.json").read_text(encoding="utf-8"))
    assert ErrorCode.VIDEO_FFMPEG_UNAVAILABLE.value in json.dumps(raw)
    assert "opencv" not in json.dumps(raw).lower()
    assert "scenedetect" not in json.dumps(raw).lower()


def test_visual_dependency_modules_are_not_loaded() -> None:
    __import__("video_demo.evaluation.real_media_execution")

    assert "cv2" not in sys.modules
    assert "scenedetect" not in sys.modules


def test_real_media_implementation_uses_four_current_phases() -> None:
    from video_demo.evaluation.evidence import _REAL_MEDIA_PHASES

    assert _REAL_MEDIA_PHASES == (
        "generate",
        "probe",
        "audio",
        "ffmpeg_frame_extract",
    )


def test_media_runner_exposes_current_ffmpeg_frame_contract() -> None:
    from video_demo.evaluation.media_runner import _MEDIA_PHASE_EXECUTABLES

    assert _MEDIA_PHASE_EXECUTABLES["ffmpeg_frame_extract"] == "FFmpegFrameExtractor"
    assert "opencv_decode" not in _MEDIA_PHASE_EXECUTABLES
    assert "scene_detect" not in _MEDIA_PHASE_EXECUTABLES


def test_real_media_execution_failure_is_reported_without_partial_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, runtime = _runner(tmp_path)
    _complete_dependencies(runner, tmp_path, monkeypatch)
    _successful_versions(runner, monkeypatch)

    def fail_execution(*_args: object, **_kwargs: object) -> object:
        raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "controlled failure")

    monkeypatch.setattr(runner, "_execute_media_port", fail_execution)

    check = runner.run(evaluation_run_id="run-1")

    assert check.status == GateStatus.FAIL
    raw = RealMediaRawReport.model_validate_json(
        (runtime / "eval/reports/run-1/raw.json").read_bytes()
    )
    assert raw.samples[0].execution_status == "FAILED"
    assert all(sample.execution_status == "NOT_STARTED" for sample in raw.samples[1:])
    assert raw.failure_code == ErrorCode.VIDEO_PROCESS_FAILED
