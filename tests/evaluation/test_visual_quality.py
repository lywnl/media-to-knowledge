from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from video_demo.evaluation.visual_quality import (
    VisualQualityCase,
    VisualQualityReport,
    VisualQualitySample,
    VisualQualitySet,
    build_visual_quality_report,
    build_visual_resolution_pair,
    build_visual_resolution_report,
    visual_quality_case_id,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _not_run_case(edge: int = 1920) -> VisualQualityCase:
    parent = "eval_parent"
    sample = "sample_001"
    refs = ("frame_001", "frame_002")
    return VisualQualityCase(
        case_id=visual_quality_case_id(parent, sample, refs, edge, 90),
        parent_evaluation_run_id=parent,
        sample_id=sample,
        requested_reference_frame_ids=refs,
        proxy_max_edge=edge,
        jpeg_quality=90,
        case_status="NOT_RUN",
        implementation_sha256="a" * 64,
        settings_fingerprint="b" * 64,
    )


def test_case_id_is_stable_and_report_preserves_planned_not_run_cases() -> None:
    case = _not_run_case()
    quality_set = VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id="eval_parent",
        dataset_sha256="c" * 64,
        authorization_sha256="d" * 64,
        status="NOT_RUN",
        not_run_reason="代表性质量集不足",
        samples=(
            VisualQualitySample(
                sample_id="sample_001",
                requested_reference_frame_ids=("frame_001", "frame_002"),
            ),
        ),
    )
    report = build_visual_quality_report(
        quality_set,
        object(),
        (case,),
    )

    assert visual_quality_case_id(
        "eval_parent", "sample_001", ("frame_001", "frame_002"), 1920, 90
    ) == case.case_id
    assert report.status == "NOT_RUN"
    assert report.planned_case_ids == (case.case_id,)
    assert report.cases == (case,)
    assert report.visual_text_accuracy is None
    assert report.visual_text_accuracy_not_run_reason


def test_not_run_case_cannot_claim_a_created_run() -> None:
    with pytest.raises(ValidationError):
        payload = _not_run_case().model_dump(mode="python")
        payload["evaluation_run_id"] = "run_001"
        VisualQualityCase(
            **payload,
        )


def test_report_rejects_forged_digest_and_fail_report_does_not_aggregate() -> None:
    case = _not_run_case()
    with pytest.raises(ValidationError):
        VisualQualityReport(
            schema_version="1.0.0",
            parent_evaluation_run_id="eval_parent",
            dataset_sha256="c" * 64,
            authorization_sha256="d" * 64,
            status="NOT_RUN",
            not_run_reason="缺少授权样本",
            planned_case_ids=(case.case_id,),
            cases=(case,),
            visual_text_accuracy=None,
            visual_text_accuracy_not_run_reason="缺少授权样本",
            visual_key_field_recall=None,
            visual_key_field_recall_not_run_reason="缺少授权样本",
            report_sha256="e" * 64,
        )


def test_resolution_pair_requires_two_resolution_cases() -> None:
    left = _not_run_case(1280)
    right = _not_run_case(1920)
    pair = build_visual_resolution_pair(
        left,
        right,
        expected_parent_evaluation_run_id="eval_parent",
        expected_sample_id="sample_001",
        expected_requested_reference_frame_ids=("frame_001", "frame_002"),
        expected_jpeg_quality=90,
        quality_report_1280=None,
        quality_report_1920=None,
    )
    assert pair.status == "NOT_RUN"


def test_resolution_report_keeps_not_run_pair_and_inconclusive_decision() -> None:
    left = _not_run_case(1280)
    right = _not_run_case(1920)
    pair = build_visual_resolution_pair(
        left,
        right,
        expected_parent_evaluation_run_id="eval_parent",
        expected_sample_id="sample_001",
        expected_requested_reference_frame_ids=("frame_001", "frame_002"),
        expected_jpeg_quality=90,
        quality_report_1280=None,
        quality_report_1920=None,
    )
    quality_set = VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id="eval_parent",
        dataset_sha256="c" * 64,
        authorization_sha256="d" * 64,
        status="NOT_RUN",
        not_run_reason="代表性质量集不足",
        samples=(
            VisualQualitySample(
                sample_id="sample_001",
                requested_reference_frame_ids=("frame_001", "frame_002"),
            ),
        ),
    )
    report = build_visual_resolution_report(
        quality_set,
        object(),
        (pair,),
        None,
        None,
    )

    assert report.status == "NOT_RUN"
    assert report.resolution_decision == "INCONCLUSIVE"
    assert report.pairs == (pair,)
