from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_demo.errors import ErrorCode
from video_demo.evaluation.evidence import (
    ChapterVlmLiveRawReport,
    EvidenceStore,
    LiveInputArtifact,
    MachineEvidenceReport,
    PreflightIssue,
    PreflightRawReport,
)
from video_demo.evaluation.report import GateStatus


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_input() -> dict[str, object]:
    digest = "a" * 64
    return {
        "kind": "SOURCE_MEDIA",
        "sample_id": "sample-001",
        "relative_path": ".codex/video-rag-demo/eval/media/sample.mp4",
        "sha256": digest,
        "source_media_sha256": digest,
        "size_bytes": 1,
    }


def _chapter_failure_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "check_id": "chapter_vlm_live",
        "status": "FAIL",
        "execution_started": True,
        "parent_evaluation_run_id": "parent-run",
        "evaluation_run_id": "run-001",
        "sample_id": "sample-001",
        "annotation_sha256": "b" * 64,
        "source_media_input": _source_input(),
        "model": {
            "component": "chapter_vlm",
            "provider": "qwen",
            "model_id": "qwen3-vl-flash",
        },
        "operation": "analyze_chapter",
        "settings_fingerprint": "c" * 64,
        "implementation_sha256": "d" * 64,
        "failure_code": ErrorCode.ARTIFACT_SCHEMA_INVALID,
        "failure_component": "chapter_vlm",
    }


def test_live_input_artifact_rejects_workspace_escape() -> None:
    with pytest.raises(ValidationError):
        LiveInputArtifact.model_validate(
            {**_source_input(), "relative_path": ".codex/video-rag-demo/../escape.mp4"}
        )


def test_chapter_vlm_early_failure_may_omit_manifest_but_not_response() -> None:
    report = ChapterVlmLiveRawReport.model_validate(_chapter_failure_payload())
    assert report.frame_manifest_input is None
    with pytest.raises(ValidationError):
        ChapterVlmLiveRawReport.model_validate(
            {**_chapter_failure_payload(), "response_sha256": "e" * 64}
        )


def test_preflight_issues_are_nonempty_and_ordered() -> None:
    issues = (
        PreflightIssue(code=ErrorCode.INVALID_CONFIGURATION),
        PreflightIssue(code=ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE),
    )
    report = PreflightRawReport(
        schema_version="1.0.0",
        check_id="chapter_vlm_live",
        reason_code="CHAPTER_VLM_INPUT_UNAVAILABLE",
        execution_started=False,
        issues=issues,
        implementation_sha256="a" * 64,
        evaluation_run_id="run-001",
    )
    assert tuple(issue.code for issue in report.issues or ()) == tuple(
        issue.code for issue in issues
    )
    with pytest.raises(ValidationError):
        PreflightRawReport.model_validate(
            {**report.model_dump(mode="python"), "issues": []}
        )


def test_evidence_store_writes_only_under_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    store = EvidenceStore(tmp_path, runtime_root)
    artifact = store.write_artifact(
        Path("eval/reports/run-001/stdout.txt"), "COMMAND_STDOUT", b"ok\n"
    )
    assert artifact.sha256 == _sha(b"ok\n")
    assert (runtime_root / "eval/reports/run-001/stdout.txt").read_bytes() == b"ok\n"
    with pytest.raises(ValueError):
        store.write_artifact(Path("../escape.txt"), "COMMAND_STDOUT", b"no")


def test_machine_report_rejects_retired_details_discriminator() -> None:
    with pytest.raises(ValidationError):
        MachineEvidenceReport.model_validate(
            {
                "schema_version": "1.0.0",
                "check_id": "chapter_vlm_live",
                "status": GateStatus.PASS,
                "kind": "LIVE_SERVICE_REPORT",
                "level": "REAL_SERVICE",
                "covered_items": ["chapter_vlm_live"],
                "summary": "x",
                "producer": "test",
                "started_at": "2026-08-18T01:00:00Z",
                "finished_at": "2026-08-18T01:00:01Z",
                "details": {"type": "RETIRED"},
            }
        )
