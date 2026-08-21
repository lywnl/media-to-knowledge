from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_demo.evaluation.metrics import runtime_resource_metrics
from video_demo.evaluation.report import (
    GateStatus,
    MetricObservation,
    MetricResult,
    QualityReport,
    build_quality_report,
)
from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS


def test_missing_real_measurement_is_not_run_not_zero_or_pass() -> None:
    report = build_quality_report(
        {"zh_cer": MetricObservation(value=None, not_run_reason="缺少授权中文标注")},
        QUALITY_THRESHOLDS,
    )

    assert report.status == GateStatus.NOT_RUN
    by_name = {metric.name: metric for metric in report.metrics}
    assert by_name["zh_cer"].status == GateStatus.NOT_RUN
    assert by_name["zh_cer"].value is None
    assert set(by_name) == set(QUALITY_THRESHOLDS)
    assert by_name["en_wer"].status == GateStatus.NOT_RUN
    assert by_name["en_wer"].not_run_reason == "未提供真实测量"


def test_threshold_report_distinguishes_pass_and_fail() -> None:
    report = build_quality_report(
        {
            "zh_cer": MetricObservation(value=0.14),
            "en_wer": MetricObservation(value=0.19),
        },
        QUALITY_THRESHOLDS,
    )

    assert report.status == GateStatus.FAIL
    by_name = {metric.name: metric.status for metric in report.metrics}
    assert by_name["en_wer"] == GateStatus.FAIL
    assert by_name["zh_cer"] == GateStatus.PASS


def test_quality_report_is_strict_machine_readable_json() -> None:
    report = build_quality_report({}, QUALITY_THRESHOLDS)

    restored = type(report).model_validate_json(report.model_dump_json())
    schema = type(report).model_json_schema()

    assert restored == report
    assert schema["additionalProperties"] is False


def test_quality_report_records_peak_resources() -> None:
    resources = runtime_resource_metrics(
        video_duration_ms=60_000,
        elapsed_seconds=120.0,
        peak_rss_bytes=2_000,
        peak_disk_bytes=3_000,
    )

    report = build_quality_report({}, QUALITY_THRESHOLDS, resources=resources)

    assert report.resources == resources
    assert report.resources_not_run_reason is None


def test_quality_report_rejects_inconsistent_direct_deserialization() -> None:
    with pytest.raises(ValidationError):
        QualityReport.model_validate(
            {
                "status": "PASS",
                "metrics": [],
                "resources": None,
                "resources_not_run_reason": None,
            },
        )


def test_missing_resource_measurement_prevents_overall_pass() -> None:
    thresholds = {"quality": QUALITY_THRESHOLDS["zh_cer"]}
    report = build_quality_report(
        {"quality": MetricObservation(value=0.10)},
        thresholds,
    )

    assert report.metrics[0].status == GateStatus.PASS
    assert report.status == GateStatus.NOT_RUN


def test_metric_result_rejects_status_that_disagrees_with_value() -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            name="zh_cer",
            value=0.20,
            threshold=0.15,
            direction="max",
            status=GateStatus.PASS,
        )


def test_resource_rtf_is_the_single_source_for_threshold_evaluation() -> None:
    resources = runtime_resource_metrics(
        video_duration_ms=60_000,
        elapsed_seconds=600.0,
        peak_rss_bytes=2_000,
        peak_disk_bytes=3_000,
    )

    report = build_quality_report({}, {"rtf": QUALITY_THRESHOLDS["rtf"]}, resources=resources)

    assert report.metrics[0].value == 10.0
    assert report.metrics[0].status == GateStatus.FAIL
    assert report.status == GateStatus.FAIL


def test_conflicting_rtf_observation_is_rejected() -> None:
    resources = runtime_resource_metrics(
        video_duration_ms=60_000,
        elapsed_seconds=600.0,
        peak_rss_bytes=2_000,
        peak_disk_bytes=3_000,
    )

    with pytest.raises(ValueError, match="RTF"):
        build_quality_report(
            {"rtf": MetricObservation(value=1.0)},
            {"rtf": QUALITY_THRESHOLDS["rtf"]},
            resources=resources,
        )


def test_machine_report_rejects_rtf_that_conflicts_with_resources() -> None:
    with pytest.raises(ValidationError, match="RTF"):
        QualityReport.model_validate(
            {
                "status": "PASS",
                "metrics": [
                    {
                        "name": "rtf",
                        "value": 1.0,
                        "threshold": 3.0,
                        "direction": "max",
                        "status": "PASS",
                    },
                ],
                "resources": {
                    "rtf": 10.0,
                    "peak_rss_bytes": 2_000,
                    "peak_disk_bytes": 3_000,
                },
                "resources_not_run_reason": None,
            },
        )


def test_report_level_failure_is_explicit_and_backward_compatible() -> None:
    report = build_quality_report(
        {"zh_cer": MetricObservation(value=0.1)},
        {"zh_cer": QUALITY_THRESHOLDS["zh_cer"]},
        failure_code="SEMANTIC_REVIEW_INCOMPLETE",
    )

    assert report.status == GateStatus.FAIL
    assert report.failure_code == "SEMANTIC_REVIEW_INCOMPLETE"
    assert QualityReport.model_validate_json(report.model_dump_json()) == report
