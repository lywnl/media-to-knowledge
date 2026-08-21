from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from video_demo.config import Settings
from video_demo.evaluation.evidence import EvidenceStore, PreflightRawReport
from video_demo.evaluation.live_runner import LiveValidationRunner
from video_demo.evaluation.report import GateStatus


def test_missing_workspace_package_writes_not_run_instead_of_skipping(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.gate import _LIVE_IMPLEMENTATION_FILES

    project_root = Path(__file__).resolve().parents[2]
    for relative_path in _LIVE_IMPLEMENTATION_FILES:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative_path, target)
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    runner = LiveValidationRunner(
        settings,
        EvidenceStore(tmp_path, settings.runtime_root),
        components_factory=lambda _settings: pytest.fail("缺授权输入不得构造组件"),
    )

    check = runner.run_workspace_baidu("workspace-baidu-missing")

    assert check.status == GateStatus.NOT_RUN
    raw = PreflightRawReport.model_validate_json(
        (
            settings.runtime_root
            / "eval/reports/workspace-baidu-missing/preflight.json"
        ).read_bytes()
    )
    assert raw.execution_started is False
    assert "LIVE_AUTHORIZED_KEYFRAME_UNAVAILABLE" in {
        issue.code.value for issue in raw.issues
    }


@pytest.mark.integration
def test_workspace_live_services_emit_authoritative_results() -> None:
    """真实入口不 skip；按当前工作区事实形成 PASS、FAIL 或具体 NOT_RUN。"""

    settings = Settings(workspace_root=Path.cwd())
    assert settings.runtime_root is not None
    runner = LiveValidationRunner(
        settings,
        EvidenceStore(settings.workspace_root, settings.runtime_root),
    )
    suffix = uuid4().hex
    checks = {
        "baidu_ocr_live": runner.run_workspace_baidu(f"workspace-baidu-{suffix}"),
        "qwen_live": runner.run_workspace_qwen(f"workspace-qwen-{suffix}"),
        "pyannote_live": runner.run_workspace_pyannote(f"workspace-pyannote-{suffix}"),
        "five_language_models": runner.run_workspace_local_model_stack(
            f"workspace-local-{suffix}"
        ),
    }

    assert set(checks) == {
        "baidu_ocr_live",
        "qwen_live",
        "pyannote_live",
        "five_language_models",
    }
    for check_id, check in checks.items():
        assert check.check_id == check_id
        assert check.status in {GateStatus.PASS, GateStatus.FAIL, GateStatus.NOT_RUN}
        if check.status == GateStatus.NOT_RUN:
            assert check.not_run_reason
