from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from video_demo.config import Settings
from video_demo.evaluation.durability import DurabilityRunner
from video_demo.evaluation.evidence import EvidenceStore, PreflightRawReport
from video_demo.evaluation.report import GateStatus


@pytest.mark.performance
def test_workspace_m1_durability_emits_structured_not_run_without_real_inputs() -> None:
    """真实入口不 skip；当前缺正式样本/依赖时发布结构化 NOT_RUN。"""

    settings = Settings(workspace_root=Path.cwd())
    assert settings.runtime_root is not None
    evaluation_run_id = f"workspace-m1-durability-{uuid4().hex}"
    report_root = settings.runtime_root / "eval/reports" / evaluation_run_id
    manifest = settings.runtime_root / "eval/durability/dataset.jsonl"

    check = DurabilityRunner(
        settings,
        EvidenceStore(settings.workspace_root, settings.runtime_root),
    ).run(manifest, evaluation_run_id=evaluation_run_id)

    assert check.status == GateStatus.NOT_RUN
    assert check.not_run_reason
    preflight = PreflightRawReport.model_validate_json(
        (report_root / "preflight.json").read_bytes()
    )
    assert preflight.execution_started is False
    assert preflight.evaluation_run_id == evaluation_run_id
    assert preflight.issues
    assert not (report_root / ".real-media.commit.json").exists()
