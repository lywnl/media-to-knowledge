from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.evaluation.metrics import RuntimeResourceMetrics
from video_demo.evaluation.thresholds import MetricThreshold


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class MetricObservation(FrozenModel):
    value: float | None
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_observation(self) -> MetricObservation:
        if self.value is None and self.not_run_reason is None:
            raise ValueError("缺失指标必须提供 NOT_RUN 原因")
        if self.value is not None and (
            not math.isfinite(self.value) or self.not_run_reason is not None
        ):
            raise ValueError("已运行指标必须是有限数值且不得包含 NOT_RUN 原因")
        return self


class MetricResult(FrozenModel):
    name: str
    value: float | None
    threshold: float
    direction: Literal["max", "min"]
    status: GateStatus
    not_run_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> MetricResult:
        if self.value is None:
            expected_status = GateStatus.NOT_RUN
            if self.not_run_reason is None:
                raise ValueError("NOT_RUN 指标必须提供原因")
        else:
            passes = (
                self.value <= self.threshold
                if self.direction == "max"
                else self.value >= self.threshold
            )
            expected_status = GateStatus.PASS if passes else GateStatus.FAIL
            if self.not_run_reason is not None:
                raise ValueError("已运行指标不得包含 NOT_RUN 原因")
        if self.status != expected_status:
            raise ValueError("指标状态必须与数值和阈值一致")
        return self


class QualityReport(FrozenModel):
    status: GateStatus
    metrics: tuple[MetricResult, ...] = Field(min_length=1)
    resources: RuntimeResourceMetrics | None = None
    resources_not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)

    @model_validator(mode="after")
    def validate_report(self) -> QualityReport:
        expected_status = _aggregate_status(self.metrics)
        if self.failure_code is not None:
            expected_status = GateStatus.FAIL
        if self.resources is None and expected_status == GateStatus.PASS:
            expected_status = GateStatus.NOT_RUN
        if self.status != expected_status:
            raise ValueError("报告总状态必须与指标状态一致")
        if (self.resources is None) == (self.resources_not_run_reason is None):
            raise ValueError("资源统计与 NOT_RUN 原因必须且只能提供一个")
        rtf_metrics = tuple(metric for metric in self.metrics if metric.name == "rtf")
        if len(rtf_metrics) > 1:
            raise ValueError("报告只能包含一个 RTF 指标")
        if (
            self.resources is not None
            and rtf_metrics
            and rtf_metrics[0].value != self.resources.rtf
        ):
            raise ValueError("RTF 指标必须与资源统计一致")
        return self


class BoundQualityReport(QualityReport):
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    prediction_index_sha256: Sha256
    judgment_index_sha256: Sha256
    sample_details_sha256: Sha256
    durability_report_sha256: Sha256 | None


def build_quality_report(
    observations: dict[str, MetricObservation],
    thresholds: dict[str, MetricThreshold],
    *,
    resources: RuntimeResourceMetrics | None = None,
    resources_not_run_reason: str | None = None,
    failure_code: str | None = None,
) -> QualityReport:
    unknown = sorted(set(observations) - set(thresholds))
    if unknown:
        raise ValueError(f"未知质量指标: {', '.join(unknown)}")
    effective_observations = dict(observations)
    if resources is not None and "rtf" in thresholds:
        supplied_rtf = observations.get("rtf")
        if supplied_rtf is not None and supplied_rtf.value != resources.rtf:
            raise ValueError("RTF 指标必须来自同一份资源统计")
        effective_observations["rtf"] = MetricObservation(value=resources.rtf)
    results = tuple(
        _evaluate(
            name,
            effective_observations.get(
                name,
                MetricObservation(value=None, not_run_reason="未提供真实测量"),
            ),
            thresholds[name],
        )
        for name in sorted(thresholds)
    )
    if resources is None and resources_not_run_reason is None:
        resources_not_run_reason = "未提供真实资源测量"
    status = _aggregate_status(results)
    if failure_code is not None:
        status = GateStatus.FAIL
    if resources is None and status == GateStatus.PASS:
        status = GateStatus.NOT_RUN
    return QualityReport(
        status=status,
        metrics=results,
        resources=resources,
        resources_not_run_reason=resources_not_run_reason,
        failure_code=failure_code,
    )


def _evaluate(
    name: str,
    observation: MetricObservation,
    threshold: MetricThreshold,
) -> MetricResult:
    if observation.value is None:
        status = GateStatus.NOT_RUN
    elif threshold.direction == "max":
        status = GateStatus.PASS if observation.value <= threshold.limit else GateStatus.FAIL
    else:
        status = GateStatus.PASS if observation.value >= threshold.limit else GateStatus.FAIL
    return MetricResult(
        name=name,
        value=observation.value,
        threshold=threshold.limit,
        direction=threshold.direction,
        status=status,
        not_run_reason=observation.not_run_reason,
    )


def _aggregate_status(metrics: tuple[MetricResult, ...]) -> GateStatus:
    if any(result.status == GateStatus.FAIL for result in metrics):
        return GateStatus.FAIL
    if any(result.status == GateStatus.NOT_RUN for result in metrics):
        return GateStatus.NOT_RUN
    return GateStatus.PASS
