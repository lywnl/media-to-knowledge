from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from video_demo.config import Settings
from video_demo.evaluation import cli
from video_demo.evaluation.evidence import (
    EvidenceKind,
    EvidenceLevel,
    EvidenceReference,
)
from video_demo.evaluation.final_runner import StageExecutionResult
from video_demo.evaluation.gate import GateCheck
from video_demo.evaluation.prediction_runner import PredictionRunReport
from video_demo.evaluation.report import GateStatus


@pytest.mark.parametrize(
    ("statuses", "expected"),
    (
        ((GateStatus.PASS,), 0),
        ((GateStatus.PASS, GateStatus.FAIL, GateStatus.NOT_RUN), 1),
        ((GateStatus.PASS, GateStatus.NOT_RUN), 2),
    ),
)
def test_exit_code_uses_fail_not_run_pass_priority(
    statuses: tuple[GateStatus, ...],
    expected: int,
) -> None:
    assert cli.exit_code_for(statuses) == expected


@pytest.mark.parametrize(
    "argv",
    (
        ["media", "--evaluation-run-id", "../escape"],
        ["final", "--evaluation-run-id", "eval_001", "/tmp/foreign"],
        ["live", "--evaluation-run-id", "eval_001", "--token", "visible"],
        ["quality", "predict", "--evaluation-run-id", "eval_001", "--api-key", "x"],
    ),
)
def test_parser_rejects_paths_and_secret_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


def test_parser_accepts_cleanup_as_the_only_destructive_subcommand() -> None:
    args = cli.parse_args(["cleanup", "--evaluation-run-id", "eval_cleanup"])

    assert args.command == "cleanup"
    assert args.evaluation_run_id == "eval_cleanup"


@pytest.mark.parametrize("quality_command", ["visual", "visual-resolution"])
def test_parser_accepts_visual_quality_commands(quality_command: str) -> None:
    args = cli.parse_args(
        ["quality", quality_command, "--evaluation-run-id", "eval_visual"]
    )

    assert args.command == "quality"
    assert args.quality_command == quality_command


def _gate_check(check_id: str, status: GateStatus) -> GateCheck:
    evidence = EvidenceReference(
        kind=EvidenceKind.LIVE_SERVICE_REPORT,
        level=EvidenceLevel.REAL_SERVICE,
        relative_path=(
            f".codex/video-rag-demo/eval/reports/{check_id}/{check_id}.json"
        ),
        sha256="a" * 64,
        covered_items=(check_id,),
        summary=f"{check_id} 测试报告",
    )
    return GateCheck(check_id=check_id, status=status, evidence=(evidence,))


def test_live_dispatch_writes_outer_summary_instead_of_returning_chapter_vlm_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = {
        "chapter_vlm": _gate_check("chapter_vlm_live", GateStatus.PASS),
        "models": _gate_check("five_language_models", GateStatus.PASS),
    }

    class FakeLiveRunner:
        def __init__(self, *_args: object) -> None:
            pass

        def run_workspace_chapter_vlm(self, _run_id: str) -> GateCheck:
            return checks["chapter_vlm"]

        def run_workspace_local_model_stack(self, _run_id: str) -> GateCheck:
            return checks["models"]

    monkeypatch.setattr(cli, "LiveValidationRunner", FakeLiveRunner)
    settings = Settings(workspace_root=tmp_path)
    result = cli.dispatch(
        cli.parse_args(["live", "--evaluation-run-id", "eval_live"]),
        settings,
    )

    assert result.status == GateStatus.PASS
    assert result.reason is None
    assert result.report_path.endswith("eval/reports/eval_live/live-summary.json")
    summary = __import__("json").loads(
        (tmp_path / result.report_path).read_text(encoding="utf-8")
    )
    assert [item["check_id"] for item in summary["checks"]] == [
        "chapter_vlm_live",
        "five_language_models",
    ]


def test_non_pass_stage_result_requires_stable_reason() -> None:
    with pytest.raises(ValidationError, match="原因"):
        StageExecutionResult(
            status=GateStatus.FAIL,
            report_path=".codex/video-rag-demo/eval/reports/run/report.json",
        )


def test_quality_predict_fail_has_stable_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedPredictionRunner:
        def __init__(self, _settings: Settings) -> None:
            pass

        def predict(self, _package: object, *, evaluation_run_id: str) -> object:
            return SimpleNamespace(
                status=GateStatus.FAIL,
                not_run_reason=None,
                evaluation_run_id=evaluation_run_id,
            )

    monkeypatch.setattr(cli, "PredictionRunner", FailedPredictionRunner)
    monkeypatch.setattr(cli, "load_evaluation_package", lambda *_args, **_kwargs: object())
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    eval_root = settings.runtime_root / "eval"
    eval_root.mkdir(parents=True)
    (eval_root / "dataset.jsonl").write_text("fixture", encoding="utf-8")
    (eval_root / "authorization.json").write_text("fixture", encoding="utf-8")
    result = cli.dispatch(
        cli.parse_args(
            ["quality", "predict", "--evaluation-run-id", "eval_predict_fail"]
        ),
        settings,
    )

    assert result.status == GateStatus.FAIL
    assert result.reason == "预测阶段存在失败样本"


@pytest.mark.parametrize("present_input", (None, "dataset.jsonl", "authorization.json"))
def test_quality_predict_missing_package_returns_structured_not_run(
    tmp_path: Path,
    present_input: str | None,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    eval_root = settings.runtime_root / "eval"
    eval_root.mkdir(parents=True)
    if present_input is not None:
        (eval_root / present_input).write_text("incomplete", encoding="utf-8")

    result = cli.dispatch(
        cli.parse_args(
            ["quality", "predict", "--evaluation-run-id", "eval_predict_missing"]
        ),
        settings,
    )

    assert result.status == GateStatus.NOT_RUN
    assert result.reason == "缺少授权五语评测集或授权记录"
    payload = json.loads((tmp_path / result.report_path).read_text(encoding="utf-8"))
    assert payload == {
        "created_at": payload["created_at"],
        "evaluation_run_id": "eval_predict_missing",
        "reason": "缺少授权五语评测集或授权记录",
        "schema_version": "1.0.0",
        "stage": "quality_predict",
        "status": "NOT_RUN",
    }


def test_quality_score_missing_prediction_returns_structured_not_run(
    tmp_path: Path,
) -> None:
    settings = Settings(workspace_root=tmp_path)

    result = cli.dispatch(
        cli.parse_args(
            ["quality", "score", "--evaluation-run-id", "eval_score_missing"]
        ),
        settings,
    )

    assert result.status == GateStatus.NOT_RUN
    assert result.reason == "缺少预测报告，请先执行 quality predict"
    payload = json.loads((tmp_path / result.report_path).read_text(encoding="utf-8"))
    assert payload["stage"] == "quality_score"
    assert payload["status"] == "NOT_RUN"


def test_quality_score_propagates_verified_prediction_not_run(
    tmp_path: Path,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    report_path = (
        settings.runtime_root / "eval/reports/eval_score_preflight/prediction.json"
    )
    report_path.parent.mkdir(parents=True)
    report = PredictionRunReport(
        schema_version="1.0.0",
        evaluation_run_id="eval_score_preflight",
        status=GateStatus.NOT_RUN,
        dataset_sha256="a" * 64,
        authorization_sha256="b" * 64,
        implementation_sha256="c" * 64,
        settings_fingerprint="d" * 64,
        prediction_index_sha256=None,
        predictions=(),
        not_run_reason="VIDEO_FFMPEG_UNAVAILABLE",
        started_at=datetime(2026, 8, 20, tzinfo=UTC),
        finished_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    report_path.write_bytes(report.model_dump_json(exclude_none=True).encode("utf-8"))

    result = cli.dispatch(
        cli.parse_args(
            ["quality", "score", "--evaluation-run-id", "eval_score_preflight"]
        ),
        settings,
    )

    assert result.status == GateStatus.NOT_RUN
    assert result.reason == "预测阶段未运行"


def test_main_prints_only_stable_status_relative_report_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "dispatch",
        lambda _args, _settings: StageExecutionResult(
            status=GateStatus.NOT_RUN,
            report_path=".codex/video-rag-demo/eval/reports/eval_001/preflight.json",
            reason="缺少工作区工具",
        ),
    )

    code = cli.main(["preflight", "--evaluation-run-id", "eval_001"])
    output = capsys.readouterr()

    assert code == 2
    assert output.err == ""
    assert output.out.splitlines() == [
        "状态: NOT_RUN",
        "报告: .codex/video-rag-demo/eval/reports/eval_001/preflight.json",
        "原因: 缺少工作区工具",
    ]
    assert str(tmp_path) not in output.out


def test_main_returns_three_and_hides_internal_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail(*_args: object) -> StageExecutionResult:
        raise ValueError(f"visible-secret {tmp_path}")

    monkeypatch.setattr(cli, "dispatch", fail)

    code = cli.main(["final", "--evaluation-run-id", "eval_001"])
    output = capsys.readouterr()

    assert code == 3
    assert output.err == ""
    assert output.out.splitlines() == [
        "状态: ERROR",
        "报告: -",
        "原因: 验收器配置或证据损坏",
    ]
    assert "visible-secret" not in output.out
    assert str(tmp_path) not in output.out
