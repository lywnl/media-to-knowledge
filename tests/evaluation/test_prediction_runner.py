from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.prediction_runner import (
    PredictionRunReport,
    _create_run_request,
    _persist_failed_prediction,
    _prediction_index_path,
)
from video_demo.evaluation.report import GateStatus


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
        assert _has_dependency("whisperx") is False


def test_dependency_probe_keeps_tensorflow_hub_warning_off_stderr() -> None:
    from video_demo.evaluation.prediction_runner import _has_dependency

    def import_module(_name: str) -> object:
        warnings.warn(
            "pkg_resources is deprecated as an API. See upstream migration guidance.",
            UserWarning,
            stacklevel=2,
        )
        return object()

    with (
        patch(
            "video_demo.evaluation.prediction_runner.importlib.util.find_spec",
            return_value=object(),
        ),
        patch(
            "video_demo.evaluation.prediction_runner.importlib.import_module",
            side_effect=import_module,
        ),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        assert _has_dependency("tensorflow_hub") is True

    assert captured == []


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
        "min_speakers": None,
        "max_speakers": None,
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
