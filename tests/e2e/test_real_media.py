from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from video_demo.capabilities import resolve_workspace_binary
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.evidence import (
    EvidenceStore,
    PreflightRawReport,
    RealMediaRawReport,
)
from video_demo.evaluation.gate import (
    FINAL_GATE_CHECKS,
    build_final_gate_report,
)
from video_demo.evaluation.media_runner import RealMediaRunner
from video_demo.evaluation.report import GateStatus, build_quality_report
from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS


def test_absent_real_media_models_credentials_and_durability_stay_not_run(
    tmp_path: Path,
) -> None:
    report = build_final_gate_report(
        quality=build_quality_report({}, QUALITY_THRESHOLDS),
        checks=(),
        workspace_root=tmp_path,
    )
    by_id = {check.check_id: check for check in report.checks}

    for check_id in (
        "authorized_dataset",
        "real_media_chain",
        "baidu_ocr_live",
        "qwen_live",
        "pyannote_live",
        "five_language_models",
        "m1_durability",
    ):
        assert check_id in FINAL_GATE_CHECKS
        assert by_id[check_id].status == GateStatus.NOT_RUN
        assert by_id[check_id].not_run_reason


@pytest.mark.integration
def test_workspace_real_media_chain() -> None:
    """真实工作区入口只能产生可信 PASS 或精确的整体前置条件 NOT_RUN。"""

    settings = Settings(workspace_root=Path.cwd())
    assert settings.runtime_root is not None
    generated_root = settings.runtime_root / "eval" / "generated"
    before_generated = _generated_media_paths(generated_root)
    evaluation_run_id = _workspace_real_media_run_id()
    runner = RealMediaRunner(
        settings,
        EvidenceStore(settings.workspace_root, settings.runtime_root),
    )
    expected_issues = _independent_preflight_issues(settings)

    check = runner.run(evaluation_run_id=evaluation_run_id)

    assert check.status in {GateStatus.PASS, GateStatus.NOT_RUN}
    assert check.status != GateStatus.FAIL
    report_root = settings.runtime_root / "eval" / "reports" / evaluation_run_id
    if check.status == GateStatus.NOT_RUN:
        preflight = PreflightRawReport.model_validate_json(
            (report_root / "preflight.json").read_bytes()
        )
        assert preflight.execution_started is False
        assert expected_issues
        assert tuple(issue.code for issue in preflight.issues) == expected_issues
        assert _generated_media_paths(generated_root) == before_generated
        return

    assert not expected_issues
    raw = RealMediaRawReport.model_validate_json((report_root / "raw.json").read_bytes())
    assert raw.status == GateStatus.PASS
    assert tuple(sample.case_id for sample in raw.samples) == (
        "normal_audio",
        "no_audio",
        "rotation",
        "vfr",
    )


def _generated_media_paths(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _workspace_real_media_run_id() -> str:
    """每次集成入口均创建新 run，避免复用不同依赖状态的旧 authority。"""

    return f"workspace-real-media-{uuid4().hex}"


def test_workspace_real_media_run_ids_force_current_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 NOT_RUN 不能阻止下一次工作区入口重新探测当前依赖。"""

    import video_demo.evaluation.gate as gate_module
    import video_demo.evaluation.media_runner as runner_module

    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    settings = Settings(workspace_root=tmp_path)
    runner = RealMediaRunner(settings, EvidenceStore(tmp_path, runtime_root))
    preflight_results = iter(
        (
            (
                {},
                (
                    ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
                    ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
                    ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE,
                ),
            ),
            ({"ffmpeg": tmp_path / "ffmpeg"}, (ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,)),
        )
    )
    observed: list[None] = []

    def current_preflight() -> tuple[dict[str, Path], tuple[ErrorCode, ...]]:
        observed.append(None)
        return next(preflight_results)

    monkeypatch.setattr(runner, "_preflight", current_preflight)
    monkeypatch.setattr(
        gate_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )
    monkeypatch.setattr(
        runner_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )

    first_id = _workspace_real_media_run_id()
    second_id = _workspace_real_media_run_id()
    assert first_id != second_id
    assert runner.run(evaluation_run_id=first_id).status == GateStatus.NOT_RUN
    assert runner.run(evaluation_run_id=second_id).status == GateStatus.NOT_RUN
    assert observed == [None, None]

    second = PreflightRawReport.model_validate_json(
        (runtime_root / "eval" / "reports" / second_id / "preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in second.issues) == (
        ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
    )


def test_workspace_real_media_chain_accepts_current_partial_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实入口必须精确接受本次仅缺 ffprobe 的整体前置条件。"""

    import video_demo.evaluation.gate as gate_module
    import video_demo.evaluation.media_runner as runner_module

    real_media_test_module = sys.modules[__name__]
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    ffmpeg = runtime_root / "tools" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"fixture executable")
    ffmpeg.chmod(0o700)
    settings = Settings(workspace_root=tmp_path)
    store = EvidenceStore(tmp_path, runtime_root)
    runner = RealMediaRunner(settings, store)
    evaluation_run_id = "workspace-real-media-partial-preflight"

    monkeypatch.setattr(runner_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        gate_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )
    monkeypatch.setattr(
        runner_module, "_current_real_media_implementation_sha256", lambda _root: "a" * 64
    )
    expected_issues = _independent_preflight_issues(settings)
    assert expected_issues == (ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,)
    monkeypatch.setattr(real_media_test_module, "Settings", lambda **_kwargs: settings)
    monkeypatch.setattr(real_media_test_module, "EvidenceStore", lambda *_args: store)
    monkeypatch.setattr(real_media_test_module, "RealMediaRunner", lambda *_args: runner)
    monkeypatch.setattr(
        real_media_test_module,
        "_workspace_real_media_run_id",
        lambda: evaluation_run_id,
    )

    real_media_test_module.test_workspace_real_media_chain()

    raw = PreflightRawReport.model_validate_json(
        (runtime_root / "eval" / "reports" / evaluation_run_id / "preflight.json").read_bytes()
    )
    assert tuple(issue.code for issue in raw.issues) == expected_issues


def _independent_preflight_issues(settings: Settings) -> tuple[ErrorCode, ...]:
    """不调用 runner 私有逻辑，按当前工作区能力形成入口预期。"""

    assert settings.runtime_root is not None
    issues: list[ErrorCode] = []
    for name, configured, code in (
        ("ffmpeg", settings.ffmpeg_path, ErrorCode.VIDEO_FFMPEG_UNAVAILABLE),
        ("ffprobe", settings.ffprobe_path, ErrorCode.VIDEO_FFPROBE_UNAVAILABLE),
    ):
        try:
            resolve_workspace_binary(
                configured or settings.runtime_root / "tools" / name,
                workspace_root=settings.workspace_root,
                unavailable_code=code,
            )
        except VideoDemoError:
            issues.append(code)
    if (
        importlib.util.find_spec("cv2") is None
        or importlib.util.find_spec("scenedetect") is None
    ):
        issues.append(ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE)
    return tuple(issues)
