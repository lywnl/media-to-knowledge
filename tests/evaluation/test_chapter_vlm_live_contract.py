from __future__ import annotations

import hashlib
from typing import get_args

import pytest

from video_demo.errors import ErrorCode
from video_demo.evaluation.evidence import (
    _LIVE_EXECUTED_CHECKS,
    ChapterVlmLiveDetails,
    ChapterVlmLiveRawReport,
    LiveCheckId,
    LiveInputKind,
)
from video_demo.evaluation.final_runner import LiveValidationSummary
from video_demo.evaluation.gate import FINAL_GATE_CHECKS


def test_active_live_contract_uses_chapter_vlm_and_speech_only() -> None:
    assert {
        "chapter_vlm_live",
        "five_language_models",
    } == _LIVE_EXECUTED_CHECKS
    assert "baidu_ocr_live" not in FINAL_GATE_CHECKS
    assert "qwen_live" not in FINAL_GATE_CHECKS
    assert "chapter_vlm_live" in FINAL_GATE_CHECKS
    assert set(get_args(LiveInputKind)) >= {
        "SOURCE_MEDIA",
        "AUDIO",
        "FRAME_MANIFEST",
        "CHAPTER_FRAME",
    }


def test_chapter_vlm_live_uses_authority_reverification() -> None:
    from video_demo.evaluation import evidence as evidence_module

    assert getattr(evidence_module, "_LIVE_AUTHORITY_CHECKS", None) == _LIVE_EXECUTED_CHECKS


def test_durability_active_preflight_excludes_retired_qwen_configuration() -> None:
    from video_demo.evaluation.durability import _PREFLIGHT_ORDER
    from video_demo.evaluation.evidence import _DURABILITY_PREFLIGHT_CODES

    retired = set()
    assert not retired.intersection(_PREFLIGHT_ORDER)
    assert not retired.intersection(_DURABILITY_PREFLIGHT_CODES)


def test_durability_disk_preflight_is_supported_by_report_reason_mapping() -> None:
    from video_demo.errors import ErrorCode
    from video_demo.evaluation.durability import _PREFLIGHT_ORDER
    from video_demo.evaluation.evidence import _DURABILITY_PREFLIGHT_CODES
    from video_demo.evaluation.gate import build_durability_not_run_reason

    code = ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT
    assert code in _PREFLIGHT_ORDER
    assert code in _DURABILITY_PREFLIGHT_CODES
    assert "磁盘" in build_durability_not_run_reason((code,))


def test_package_media_path_is_relative_to_dataset_eval_root(tmp_path: object) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from video_demo.evaluation import live_runner as live_runner_module

    root = Path(tmp_path) / "eval"
    media = root / "media" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    package = SimpleNamespace(
        dataset=SimpleNamespace(
            eval_root=root,
            samples=(SimpleNamespace(sample_id="sample", media_relative_path="media/sample.mp4"),),
        )
    )
    helper = getattr(live_runner_module, "_package_media_path", None)
    assert callable(helper)
    assert helper(package, "sample") == media


def test_live_check_id_exposes_new_active_check() -> None:
    assert "chapter_vlm_live" in get_args(LiveCheckId)


def test_chapter_vlm_raw_report_rejects_not_run_and_incomplete_pass() -> None:
    with pytest.raises(ValueError):
        ChapterVlmLiveRawReport.model_validate(
            {
                "schema_version": "1.0.0",
                "check_id": "chapter_vlm_live",
                "status": "NOT_RUN",
                "execution_started": False,
                "parent_evaluation_run_id": "parent",
                "evaluation_run_id": "run",
                "sample_id": "sample",
                "annotation_sha256": "a" * 64,
                "source_media_input": {
                    "kind": "SOURCE_MEDIA",
                    "sample_id": "sample",
                    "relative_path": ".codex/video-rag-demo/eval/media.mp4",
                    "sha256": "b" * 64,
                    "source_media_sha256": "b" * 64,
                    "size_bytes": 1,
                },
                "model": {
                    "component": "chapter_vlm",
                    "provider": "qwen",
                    "model_id": "qwen3-vl-flash",
                },
                "operation": "analyze_chapter",
            }
        )


def test_chapter_vlm_details_has_stable_discriminator() -> None:
    assert ChapterVlmLiveDetails.model_fields["type"].default == "CHAPTER_VLM"


def test_gate_rejects_tampered_chapter_manifest_bytes(tmp_path: object) -> None:
    """门禁不能只相信 raw 中的 Manifest SHA，必须重算正文规范摘要。"""

    from pathlib import Path

    import video_demo.evaluation.gate as gate_module
    from video_demo.evaluation.evidence import (
        FileIdentity,
        FileSnapshot,
        TraceArtifact,
        VerifiedArtifact,
    )
    canonical = b'{"schema_version":"1.0.0"}'
    path = Path(tmp_path) / "manifest.json"
    path.write_bytes(canonical + b" ")
    snapshot = FileSnapshot(
        path=path,
        identity=FileIdentity(1, 1, path.stat().st_size, 1, 1),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        content=path.read_bytes(),
    )
    artifact = VerifiedArtifact(
        reference=TraceArtifact(
            role="INPUT_MEDIA",
            relative_path=".codex/video-rag-demo/eval/reports/run/visual/chapter-vlm-input.json",
            sha256=hashlib.sha256(canonical).hexdigest(),
            max_bytes=2 * 1024 * 1024,
        ),
        snapshot=snapshot,
    )

    with pytest.raises(ValueError, match="Manifest"):
        gate_module._load_chapter_vlm_manifest(artifact)


def test_summary_requires_chapter_vlm_then_speech() -> None:
    fields = LiveValidationSummary.model_fields["checks"]
    assert fields is not None


def _early_fail_raw(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "check_id": "chapter_vlm_live",
        "status": "FAIL",
        "execution_started": True,
        "parent_evaluation_run_id": "parent_run",
        "evaluation_run_id": "stage_run",
        "sample_id": "sample_001",
        "annotation_sha256": "a" * 64,
        "source_media_input": {
            "kind": "SOURCE_MEDIA",
            "sample_id": "sample_001",
            "relative_path": ".codex/video-rag-demo/eval/media/sample.mp4",
            "sha256": "b" * 64,
            "source_media_sha256": "b" * 64,
            "size_bytes": 1,
        },
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
    payload.update(overrides)
    return payload


def test_early_failure_may_omit_manifest_but_cannot_forge_response_or_score() -> None:
    raw = ChapterVlmLiveRawReport.model_validate(_early_fail_raw())
    assert raw.frame_manifest_input is None
    assert raw.response_sha256 is None
    assert raw.visual_text_score_fact is None

    with pytest.raises(ValueError):
        ChapterVlmLiveRawReport.model_validate(
            _early_fail_raw(
                failure_code=ErrorCode.QWEN_RESPONSE_INVALID,
                response_sha256="e" * 64,
            )
        )

    with pytest.raises(ValueError):
        ChapterVlmLiveRawReport.model_validate(
            _early_fail_raw(
                visual_text_score_fact={
                    "schema_version": "1.0.0",
                    "parent_evaluation_run_id": "parent_run",
                    "evaluation_run_id": "stage_run",
                    "sample_id": "sample_001",
                    "manifest_sha256": "f" * 64,
                    "response_sha256": "e" * 64,
                    "reference_sha256": "1" * 64,
                    "hypothesis_sha256": "2" * 64,
                    "errors": 0,
                    "reference_units": 0,
                    "key_field_matches": 0,
                    "key_field_reference_units": 0,
                    "quality_categories": ["GENERAL_TEXT"],
                    "selected_reference_frame_count": 1,
                }
            )
        )
