from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_demo.evaluation.annotations import (
    AuthorizationFile,
    AuthorizationRecord,
    EvaluationAnnotation,
    ReferenceVisualFrame,
    ValidatedEvaluationPackage,
    VerifiedAnnotation,
)
from video_demo.evaluation.chapter_vlm_live import ChapterVlmCallReceipt, VisualTextScoreFact
from video_demo.evaluation.dataset import EvaluationDataset, EvaluationSample
from video_demo.evaluation.visual_quality import (
    VisualQualityCase,
    VisualQualityReport,
    VisualQualitySample,
    VisualQualitySet,
    build_visual_quality_report,
    build_visual_quality_set,
    build_visual_resolution_pair,
    build_visual_resolution_report,
    visual_quality_case_id,
)
from video_demo.integrations.qwen_vl import QwenVisionProviderReceipt


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


def _package_with_frames(
    frames: tuple[ReferenceVisualFrame, ...],
    *,
    duration_ms: int = 10_000,
) -> ValidatedEvaluationPackage:
    media_sha = "a" * 64
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=media_sha,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256="b" * 64,
    )
    annotation = EvaluationAnnotation(
        schema_version="2.0.0",
        sample_id=sample.sample_id,
        media_sha256=media_sha,
        duration_ms=duration_ms,
        language="zh",
        reference_text="参考文本",
        visual_frames=frames,
        scene_boundaries_ms=(5_000,),
        semantic_boundaries_ms=(5_000,),
        supported_facts=(
            {"fact_id": "fact_001", "canonical_text": "事实"},
        ),
        key_fact_ids=("fact_001",),
    )
    dataset = EvaluationDataset(
        samples=(sample,),
        eval_root=Path("/tmp/eval"),
        runtime_root=Path("/tmp/runtime"),
        workspace_root=Path("/tmp/workspace"),
    )
    authorization = AuthorizationFile(
        schema_version="1.0.0",
        records=(
            AuthorizationRecord(
                schema_version="1.0.0",
                authorization_id="auth_001",
                source_category="OWNED",
                allowed_purposes=("VIDEO_QUALITY_EVALUATION",),
                confirmed_at="2026-01-01T00:00:00Z",
                media_sha256=(media_sha,),
            ),
        ),
    )
    return ValidatedEvaluationPackage(
        dataset=dataset,
        authorization=authorization,
        annotations=(VerifiedAnnotation(annotation=annotation, sha256="c" * 64),),
        dataset_sha256="d" * 64,
        authorization_sha256="e" * 64,
    )


def test_quality_set_selects_first_last_and_evenly_spaced_frames() -> None:
    frames = tuple(
        ReferenceVisualFrame(
            frame_id=f"frame_{index:03d}",
            timestamp_ms=index * 1_000,
            text_lines=(f"文本 {index}",),
            quality_categories=(
                "CODE" if index == 5 else "GENERAL_TEXT",
            ),
        )
        for index in range(6)
    )
    quality_set = build_visual_quality_set(
        _package_with_frames(frames),
        parent_evaluation_run_id="eval_parent",
    )

    assert quality_set.samples[0].requested_reference_frame_ids == (
        "frame_000",
        "frame_002",
        "frame_003",
        "frame_005",
    )


def test_quality_set_selects_the_earliest_largest_five_minute_frame_cluster() -> None:
    timestamps = (0, 100_000, 200_000, 300_000, 700_000, 800_000, 900_000)
    frames = tuple(
        ReferenceVisualFrame(
            frame_id=f"frame_{index:03d}",
            timestamp_ms=timestamp_ms,
            text_lines=(f"文本 {index}",),
        )
        for index, timestamp_ms in enumerate(timestamps)
    )
    quality_set = build_visual_quality_set(
        _package_with_frames(frames, duration_ms=1_000_000),
        parent_evaluation_run_id="eval_parent",
    )

    assert quality_set.samples[0].requested_reference_frame_ids == (
        "frame_000",
        "frame_001",
        "frame_002",
        "frame_003",
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


def test_failed_case_cannot_retain_a_successful_score_fact() -> None:
    parent = "eval_parent"
    sample = "sample_001"
    refs = ("frame_001", "frame_002")
    manifest_sha = "c" * 64
    response_sha = "d" * 64
    receipt = ChapterVlmCallReceipt(
        logical_analysis_count=1,
        parent_evaluation_run_id=parent,
        evaluation_run_id="run_001",
        sample_id=sample,
        manifest_sha256=manifest_sha,
        provider=QwenVisionProviderReceipt(
            provider_attempt_count=1,
            final_http_status=200,
            provider_response_sha256="e" * 64,
            request_json_bytes=1,
            encoded_request_bytes=1,
            elapsed_ms=1,
        ),
        ordered_input_frame_ids=refs,
        request_json_bytes=1,
        encoded_request_bytes=1,
        vlm_elapsed_ms=1,
        response_sha256=response_sha,
    )
    score = VisualTextScoreFact(
        schema_version="1.0.0",
        parent_evaluation_run_id=parent,
        evaluation_run_id="run_001",
        sample_id=sample,
        manifest_sha256=manifest_sha,
        response_sha256=response_sha,
        reference_sha256="f" * 64,
        hypothesis_sha256="a" * 64,
        errors=0,
        reference_units=1,
        key_field_matches=0,
        key_field_reference_units=0,
        quality_categories=("GENERAL_TEXT",),
        selected_reference_frame_count=1,
    )
    with pytest.raises(ValidationError, match="评分事实"):
        VisualQualityCase(
            case_id=visual_quality_case_id(parent, sample, refs, 1920, 90),
            parent_evaluation_run_id=parent,
            evaluation_run_id="run_001",
            sample_id=sample,
            requested_reference_frame_ids=refs,
            proxy_max_edge=1920,
            jpeg_quality=90,
            case_status="FAIL",
            error_code="VISUAL_RESULT_INVALID",
            manifest_sha256=manifest_sha,
            call_receipt=receipt,
            response_sha256=response_sha,
            score_fact=score,
            implementation_sha256="1" * 64,
            settings_fingerprint="2" * 64,
            request_json_bytes=1,
            encoded_request_bytes=1,
            vlm_elapsed_ms=1,
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
