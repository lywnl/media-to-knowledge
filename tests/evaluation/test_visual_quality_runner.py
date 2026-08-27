from __future__ import annotations

import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_demo.config import Settings
from video_demo.evaluation.visual_quality import (
    VisualQualityReport,
    VisualQualitySample,
    VisualQualitySet,
)
from video_demo.evaluation.visual_quality_runner import VisualQualityRunner


def test_report_not_run_preserves_every_planned_case(tmp_path: Path) -> None:
    runner = VisualQualityRunner(Settings(workspace_root=tmp_path))
    quality_set = VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id="eval_visual",
        dataset_sha256="a" * 64,
        authorization_sha256="b" * 64,
        status="NOT_RUN",
        not_run_reason="代表性质量集不足",
        samples=(
            VisualQualitySample(
                sample_id="sample_001",
                requested_reference_frame_ids=("frame_001", "frame_002"),
            ),
        ),
    )

    report = runner.report_not_run(quality_set)

    assert report.status == "NOT_RUN"
    assert len(report.cases) == len(report.planned_case_ids) == 1
    assert report.cases[0].case_status == "NOT_RUN"


def test_report_digest_tampering_is_rejected(tmp_path: Path) -> None:
    runner = VisualQualityRunner(Settings(workspace_root=tmp_path))
    quality_set = VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id="eval_visual",
        dataset_sha256="a" * 64,
        authorization_sha256="b" * 64,
        status="NOT_RUN",
        not_run_reason="代表性质量集不足",
    )
    report = runner.report_not_run(quality_set)
    payload = report.model_dump(mode="python")
    payload["not_run_reason"] = "被篡改"

    with pytest.raises(ValidationError, match="摘要"):
        VisualQualityReport.model_validate(payload)


def test_write_report_uses_private_atomic_artifact_permissions(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path)
    runner = VisualQualityRunner(settings)
    quality_set = VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id="eval_visual",
        dataset_sha256="a" * 64,
        authorization_sha256="b" * 64,
        status="NOT_RUN",
        not_run_reason="代表性质量集不足",
    )

    report = runner.report_not_run(quality_set)
    path = runner.write_report(report, evaluation_run_id="eval_visual")

    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
