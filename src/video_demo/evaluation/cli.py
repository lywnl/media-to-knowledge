from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from video_demo.config import Settings
from video_demo.evaluation.annotations import load_evaluation_package
from video_demo.evaluation.durability import DurabilityRunner
from video_demo.evaluation.evidence import EvidenceStore
from video_demo.evaluation.final_runner import (
    FinalValidationRunner,
    StageExecutionResult,
    cleanup_evaluation_run,
    stage_evaluation_run_id,
    write_live_validation_summary,
    write_stage_not_run_summary,
)
from video_demo.evaluation.gate import GateCheck
from video_demo.evaluation.live_runner import LiveValidationRunner
from video_demo.evaluation.media_runner import RealMediaRunner
from video_demo.evaluation.prediction_runner import (
    PredictionRunner,
    PredictionRunReport,
    score_prediction_run,
)
from video_demo.evaluation.report import GateStatus
from video_demo.storage.workspace import validate_path_component


def exit_code_for(statuses: Sequence[GateStatus]) -> int:
    if GateStatus.FAIL in statuses:
        return 1
    if GateStatus.NOT_RUN in statuses:
        return 2
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="video-demo-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "media", "live", "durability", "final", "cleanup"):
        command = commands.add_parser(name)
        _add_run_id(command)
    quality = commands.add_parser("quality")
    quality_commands = quality.add_subparsers(dest="quality_command", required=True)
    for name in ("predict", "score", "visual", "visual-resolution"):
        command = quality_commands.add_parser(name)
        _add_run_id(command)
    return parser.parse_args(argv)


def dispatch(args: argparse.Namespace, settings: Settings) -> StageExecutionResult:
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(settings.workspace_root, settings.runtime_root)
    run_id = str(args.evaluation_run_id)
    if args.command == "preflight":
        return FinalValidationRunner(settings, store).preflight(run_id)
    if args.command == "media":
        check = RealMediaRunner(settings, store).run(
            evaluation_run_id=stage_evaluation_run_id(run_id, "media")
        )
        return _check_result(check)
    if args.command == "live":
        runner = LiveValidationRunner(settings, store)
        checks = (
            runner.run_workspace_chapter_vlm(stage_evaluation_run_id(run_id, "chapter-vlm")),
            runner.run_workspace_local_model_stack(stage_evaluation_run_id(run_id, "models")),
        )
        return write_live_validation_summary(
            evaluation_run_id=run_id,
            checks=checks,
            workspace_root=settings.workspace_root,
        )
    if args.command == "durability":
        manifest = settings.runtime_root / "eval/durability/dataset.jsonl"
        check = DurabilityRunner(settings, store).run(
            manifest,
            evaluation_run_id=stage_evaluation_run_id(run_id, "durability"),
        )
        return _check_result(check)
    if args.command == "final":
        return FinalValidationRunner(settings, store).final(run_id)
    if args.command == "cleanup":
        cleanup = cleanup_evaluation_run(
            settings.workspace_root,
            run_id,
            settings=settings,
        )
        return StageExecutionResult(
            status=GateStatus.PASS,
            report_path=cleanup.manifest_path,
        )
    if args.command == "quality" and args.quality_command == "predict":
        dataset_path = settings.runtime_root / "eval/dataset.jsonl"
        authorization_path = settings.runtime_root / "eval/authorization.json"
        if not dataset_path.is_file() or not authorization_path.is_file():
            return write_stage_not_run_summary(
                evaluation_run_id=run_id,
                stage="quality_predict",
                reason="缺少授权五语评测集或授权记录",
                workspace_root=settings.workspace_root,
            )
        package = load_evaluation_package(
            dataset_path,
            authorization_path,
            workspace_root=settings.workspace_root,
            runtime_root=settings.runtime_root,
            max_video_bytes=settings.max_video_bytes,
        )
        prediction_report = PredictionRunner(settings).predict(
            package,
            evaluation_run_id=run_id,
        )
        path = settings.runtime_root / "eval/reports" / run_id / "prediction.json"
        return StageExecutionResult(
            status=prediction_report.status,
            report_path=path.relative_to(settings.workspace_root).as_posix(),
            reason=(
                prediction_report.not_run_reason
                if prediction_report.status == GateStatus.NOT_RUN
                else (
                    "预测阶段存在失败样本" if prediction_report.status == GateStatus.FAIL else None
                )
            ),
        )
    if args.command == "quality" and args.quality_command == "score":
        prediction_path = settings.runtime_root / "eval/reports" / run_id / "prediction.json"
        if not prediction_path.is_file():
            return write_stage_not_run_summary(
                evaluation_run_id=run_id,
                stage="quality_score",
                reason="缺少预测报告，请先执行 quality predict",
                workspace_root=settings.workspace_root,
            )
        prediction = _load_canonical_prediction_report(prediction_path, run_id)
        if prediction.status == GateStatus.NOT_RUN:
            return write_stage_not_run_summary(
                evaluation_run_id=run_id,
                stage="quality_score",
                reason="预测阶段未运行",
                workspace_root=settings.workspace_root,
            )
        visual_quality = None
        visual_path = settings.runtime_root / "eval/reports" / run_id / "visual-quality.json"
        dataset_path = settings.runtime_root / "eval/dataset.jsonl"
        authorization_path = settings.runtime_root / "eval/authorization.json"
        if visual_path.is_file() and dataset_path.is_file() and authorization_path.is_file():
            from video_demo.evaluation.visual_quality import verify_visual_quality_report

            package = load_evaluation_package(
                dataset_path,
                authorization_path,
                workspace_root=settings.workspace_root,
                runtime_root=settings.runtime_root,
                max_video_bytes=settings.max_video_bytes,
            )
            visual_report = __import__(
                "video_demo.evaluation.visual_quality",
                fromlist=["VisualQualityReport"],
            ).VisualQualityReport.model_validate_json(visual_path.read_bytes())
            quality_set = __import__(
                "video_demo.evaluation.visual_quality",
                fromlist=["build_visual_quality_set"],
            ).build_visual_quality_set(
                package,
                parent_evaluation_run_id=run_id,
                proxy_max_edge=1_920,
                jpeg_quality=settings.keyframe_jpeg_quality,
            )
            visual_quality = verify_visual_quality_report(visual_report, quality_set, package)
        quality_report = score_prediction_run(
            run_id,
            eval_root=settings.runtime_root / "eval",
            visual_quality_report=visual_quality,
        )
        path = settings.runtime_root / "eval/reports" / run_id / "quality.json"
        return StageExecutionResult(
            status=quality_report.status,
            report_path=path.relative_to(settings.workspace_root).as_posix(),
            reason=(
                "质量指标存在未运行项"
                if quality_report.status == GateStatus.NOT_RUN
                else ("质量指标存在失败项" if quality_report.status == GateStatus.FAIL else None)
            ),
        )
    if args.command == "quality" and args.quality_command in {
        "visual",
        "visual-resolution",
    }:
        dataset_path = settings.runtime_root / "eval/dataset.jsonl"
        authorization_path = settings.runtime_root / "eval/authorization.json"
        if not dataset_path.is_file() or not authorization_path.is_file():
            return StageExecutionResult(
                status=GateStatus.NOT_RUN,
                report_path=(
                    settings.runtime_root
                    / "eval"
                    / "reports"
                    / run_id
                    / f"{args.quality_command}.json"
                )
                .relative_to(settings.workspace_root)
                .as_posix(),
                reason="缺少授权视觉评测集或授权记录",
            )
        from video_demo.evaluation.visual_quality_runner import VisualQualityRunner

        package = load_evaluation_package(
            dataset_path,
            authorization_path,
            workspace_root=settings.workspace_root,
            runtime_root=settings.runtime_root,
            max_video_bytes=settings.max_video_bytes,
        )
        visual_runner = VisualQualityRunner(settings)
        if args.quality_command == "visual":
            report = visual_runner.run(package, evaluation_run_id=run_id)
            path = settings.runtime_root / "eval" / "reports" / run_id / "visual-quality.json"
            return StageExecutionResult(
                status=_visual_gate_status(report.status),
                report_path=path.relative_to(settings.workspace_root).as_posix(),
                reason=report.not_run_reason
                or ("视觉质量报告存在失败 case" if report.status == "FAIL" else None),
            )
        from video_demo.evaluation.visual_quality import (
            build_visual_resolution_pair,
            build_visual_resolution_report,
        )

        quality_set_1920 = visual_runner.build_quality_set(
            package, evaluation_run_id=run_id, proxy_max_edge=1_920
        )
        report_1280 = visual_runner.run(package, evaluation_run_id=run_id, proxy_max_edge=1_280)
        report_1920 = visual_runner.run(package, evaluation_run_id=run_id, proxy_max_edge=1_920)
        pairs = tuple(
            build_visual_resolution_pair(
                next(case for case in report_1280.cases if case.sample_id == sample.sample_id),
                next(case for case in report_1920.cases if case.sample_id == sample.sample_id),
                expected_parent_evaluation_run_id=run_id,
                expected_sample_id=sample.sample_id,
                expected_requested_reference_frame_ids=sample.requested_reference_frame_ids,
                expected_jpeg_quality=quality_set_1920.jpeg_quality,
                quality_report_1280=report_1280,
                quality_report_1920=report_1920,
            )
            for sample in quality_set_1920.samples
        )
        resolution = build_visual_resolution_report(
            quality_set_1920,
            package,
            pairs,
            report_1280,
            report_1920,
        )
        path = visual_runner.write_report(
            resolution, evaluation_run_id=run_id, filename="visual-resolution.json"
        )
        return StageExecutionResult(
            status=_visual_gate_status(resolution.status),
            report_path=path.relative_to(settings.workspace_root).as_posix(),
            reason=resolution.not_run_reason
            or ("分辨率对照存在失败 case" if resolution.status == "FAIL" else None),
        )
    raise ValueError("未知验收子命令")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings(workspace_root=Path.cwd())
        result = dispatch(args, settings)
    except Exception:
        _print_result("ERROR", "-", "验收器配置或证据损坏")
        return 3
    _print_result(result.status.value, result.report_path, result.reason)
    return exit_code_for((result.status,))


def _add_run_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evaluation-run-id",
        required=True,
        type=_evaluation_run_id,
    )


def _evaluation_run_id(value: str) -> str:
    try:
        return validate_path_component(value, "evaluation_run_id")
    except Exception:
        raise argparse.ArgumentTypeError("evaluation_run_id 非法") from None


def _print_result(status: str, report_path: str, reason: str | None) -> None:
    print(f"状态: {status}")
    print(f"报告: {report_path}")
    if reason is not None:
        print(f"原因: {reason}")


def _visual_gate_status(status: str) -> GateStatus:
    if status == "SUCCEEDED":
        return GateStatus.PASS
    return GateStatus(status)


def _check_result(check: GateCheck) -> StageExecutionResult:
    status = check.status
    evidence = check.evidence
    if not evidence:
        raise ValueError("阶段检查缺少权威报告")
    return StageExecutionResult(
        status=status,
        report_path=evidence[0].relative_path,
        reason=(
            check.not_run_reason
            if status == GateStatus.NOT_RUN
            else (f"失败门禁: {check.check_id}" if status == GateStatus.FAIL else None)
        ),
    )


def _load_canonical_prediction_report(
    path: Path,
    evaluation_run_id: str,
) -> PredictionRunReport:
    try:
        if path.is_symlink():
            raise ValueError("预测报告不得是符号链接")
        encoded = path.read_bytes()
        report = PredictionRunReport.model_validate_json(encoded)
        canonical = report.model_dump_json(exclude_none=True).encode("utf-8")
        if encoded != canonical or report.evaluation_run_id != evaluation_run_id:
            raise ValueError("预测报告不是当前运行的规范序列化")
        return report
    except (OSError, ValueError, ValidationError):
        raise ValueError("预测报告非法或损坏") from None


if __name__ == "__main__":
    raise SystemExit(main())
