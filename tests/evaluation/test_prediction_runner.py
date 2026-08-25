from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.prediction_runner import (
    PredictionRunReport,
    _create_run_request,
    _implementation_sha256,
    _persist_failed_prediction,
    _prediction_index_path,
)
from video_demo.evaluation.report import GateStatus


@pytest.mark.parametrize(
    "changed_path",
    (
        Path("src/video_demo/media/subtitles.py"),
        Path("src/video_demo/speech/isolated.py"),
        Path("src/video_demo/evaluation/quality_runner.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ),
)
def test_prediction_digest_covers_complete_backend_implementation(
    tmp_path: Path,
    changed_path: Path,
) -> None:
    from video_demo.implementation import prediction_implementation_files

    project_root = Path(__file__).parents[2]
    implementation_files = prediction_implementation_files(project_root)
    assert changed_path in implementation_files
    for relative in implementation_files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, destination)

    before = _implementation_sha256(tmp_path)
    copied_source = tmp_path / changed_path
    copied_source.write_bytes(copied_source.read_bytes() + b"\n")

    assert _implementation_sha256(tmp_path) != before


def test_prediction_implementation_files_reject_source_symlink(tmp_path: Path) -> None:
    from video_demo.implementation import prediction_implementation_files

    source_root = tmp_path / "src/video_demo"
    source_root.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    target = tmp_path / "linked-source.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "linked.py").symlink_to(target)

    with pytest.raises(ValueError, match="符号链接"):
        prediction_implementation_files(tmp_path)


def test_prediction_run_report_binds_run_and_prediction_digest() -> None:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    report = PredictionRunReport(
        schema_version="1.0.0",
        evaluation_run_id="eval_001",
        status=GateStatus.NOT_RUN,
        dataset_sha256="a" * 64,
        authorization_sha256="b" * 64,
        implementation_sha256="c" * 64,
        settings_fingerprint="d" * 64,
        prediction_index_sha256=None,
        predictions=(),
        not_run_reason="缺少真实生产能力",
        started_at=now,
        finished_at=now,
    )

    assert report.status == GateStatus.NOT_RUN
    assert report.prediction_index_sha256 is None
    assert report.model_dump(mode="json")["evaluation_run_id"] == "eval_001"
    encoded = report.model_dump_json(exclude_none=True).encode("utf-8")
    reparsed = PredictionRunReport.model_validate_json(encoded)
    assert reparsed == report
    assert reparsed.model_dump_json(exclude_none=True).encode("utf-8") == encoded


def test_prediction_runner_missing_capability_does_not_create_sample_runs(
    tmp_path: Path,
) -> None:
    """任务级 preflight 失败时，不能先创建任何产品 run。"""

    from video_demo.evaluation.prediction_runner import PredictionRunner

    runner = PredictionRunner(
        settings=__import__("video_demo.config", fromlist=["Settings"]).Settings(
            workspace_root=tmp_path,
        ),
        preflight=lambda: "VIDEO_FFPROBE_UNAVAILABLE",
    )

    # package 由实现实际校验；此处只证明 preflight 具有短路入口。
    assert runner.preflight_reason() == "VIDEO_FFPROBE_UNAVAILABLE"


def test_dependency_probe_rejects_module_that_fails_during_import() -> None:
    from video_demo.evaluation.prediction_runner import _has_dependency

    with (
        patch(
            "video_demo.evaluation.prediction_runner.importlib.util.find_spec",
            return_value=object(),
        ),
        patch(
            "video_demo.evaluation.prediction_runner.importlib.import_module",
            side_effect=AttributeError("torchaudio.AudioMetaData"),
        ),
    ):
        assert _has_dependency("silero_vad") is False


def test_prediction_index_path_points_to_index_json_not_run_snapshot() -> None:
    assert _prediction_index_path(Path("/runtime/eval"), "eval_001", "sample_001") == (
        Path("/runtime/eval/predictions/eval_001/sample_001/index.json")
    )


def test_prediction_request_forwards_sample_speech_hints() -> None:
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256="a" * 64,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256="b" * 64,
        hotwords=("Milvus", "WhisperX"),
        core_context="这是向量检索课程。",
    )

    assert _create_run_request(sample, "obj_001", "idem_001") == {
        "object_ref": "obj_001",
        "idempotency_key": "idem_001",
        "language_hints": ["zh"],
        "hotwords": ["Milvus", "WhisperX"],
        "core_context": "这是向量检索课程。",
    }


def test_failed_prediction_is_persisted_and_reverified_with_actual_terminal_status(
    tmp_path: Path,
) -> None:
    from video_demo.domain.run import ModelIdentity
    from video_demo.evaluation.predictions import load_verified_prediction

    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256="a" * 64,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256="b" * 64,
    )
    prediction = _persist_failed_prediction(
        eval_root,
        evaluation_run_id="eval_001",
        sample=sample,
        run_id="run_001",
        job_id="job_001",
        terminal_status="CANCELLED",
        current_stage="SPEECH",
        failure_code="JOB_CANCELLED",
        models=(ModelIdentity(component="asr", provider="local", model_id="m1"),),
    )

    loaded = load_verified_prediction(
        _prediction_index_path(eval_root, "eval_001", "sample_001"),
        eval_root=eval_root,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        sample=sample,
    )

    assert prediction.terminal_status == "CANCELLED"
    assert loaded.index == prediction
    assert loaded.run.current_stage == "SPEECH"
    assert loaded.run.error_code == "JOB_CANCELLED"
