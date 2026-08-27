"""代表性视觉文字质量事实与分辨率对照契约。

本模块只保存脱敏计数和可重验的闭包摘要，不保存请求正文、图片或模型原文。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, StrictInt, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.manifest import Rational
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode
from video_demo.evaluation.annotations import ValidatedEvaluationPackage
from video_demo.evaluation.chapter_vlm_live import (
    ChapterVlmCallReceipt,
    VisualTextScoreFact,
)
from video_demo.integrations.qwen_vl import QwenVisionProviderFailureReceipt

QualitySetStatus = Literal["NOT_RUN", "READY"]
VisualQualityStatus = Literal["NOT_RUN", "SUCCEEDED", "FAIL"]
CaseStatus = Literal["READY", "NOT_RUN", "FAIL"]


class VisualQualitySample(FrozenModel):
    sample_id: StableId
    requested_reference_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def reject_duplicate_frames(self) -> VisualQualitySample:
        if len(self.requested_reference_frame_ids) != len(set(self.requested_reference_frame_ids)):
            raise ValueError("质量样本参考帧不得重复")
        return self


class VisualQualitySet(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["1.0.0"]
    parent_evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    status: QualitySetStatus
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    samples: tuple[VisualQualitySample, ...] = Field(default=(), max_length=240)
    proxy_max_edge: StrictInt = Field(default=1_920, ge=1_280, le=2_560)
    jpeg_quality: StrictInt = Field(default=90, ge=1, le=100)

    @model_validator(mode="after")
    def validate_quality_set(self) -> VisualQualitySet:
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("质量集样本不得重复")
        if self.status == "NOT_RUN" and not self.not_run_reason:
            raise ValueError("未运行质量集必须提供原因")
        if self.status == "READY" and (self.not_run_reason or not self.samples):
            raise ValueError("可运行质量集必须有样本且不得有未运行原因")
        return self


def build_visual_quality_set(
    package: ValidatedEvaluationPackage,
    *,
    parent_evaluation_run_id: StableId,
    proxy_max_edge: int = 1_920,
    jpeg_quality: int = 90,
) -> VisualQualitySet:
    """从已重验标注确定性生成代表性质量集身份闭包。"""

    frames_by_sample = {
        verified.annotation.sample_id: verified.annotation.visual_frames
        for verified in package.annotations
    }
    samples: list[VisualQualitySample] = []
    media_ids: set[str] = set()
    category_counts: dict[str, int] = {"CODE": 0, "TABLE": 0, "UI_SMALL_TEXT": 0}
    frame_count = 0
    for sample in sorted(package.dataset.samples, key=lambda item: item.sample_id):
        frames = sorted(
            frames_by_sample.get(sample.sample_id, ()),
            key=lambda item: (item.timestamp_ms, item.frame_id),
        )
        if len(frames) < 2:
            continue
        selected = tuple(frame.frame_id for frame in frames[:4])
        samples.append(
            VisualQualitySample(
                sample_id=sample.sample_id,
                requested_reference_frame_ids=selected,
            )
        )
        media_ids.add(sample.media_sha256)
        frame_count += len(selected)
        for frame in frames[:4]:
            for category in frame.quality_categories:
                if category in category_counts:
                    category_counts[category] += 1
    insufficient = (
        len(media_ids) < 3
        or frame_count < 12
        or any(count < 3 for count in category_counts.values())
    )
    if insufficient:
        return VisualQualitySet(
            schema_version="1.0.0",
            parent_evaluation_run_id=parent_evaluation_run_id,
            dataset_sha256=package.dataset_sha256,
            authorization_sha256=package.authorization_sha256,
            status="NOT_RUN",
            not_run_reason=(
                "代表性质量集不足：至少需要 3 个媒体、12 个参考帧，"
                "且 CODE/TABLE/UI_SMALL_TEXT 各 3 帧"
            ),
            samples=tuple(samples),
            proxy_max_edge=proxy_max_edge,
            jpeg_quality=jpeg_quality,
        )
    return VisualQualitySet(
        schema_version="1.0.0",
        parent_evaluation_run_id=parent_evaluation_run_id,
        dataset_sha256=package.dataset_sha256,
        authorization_sha256=package.authorization_sha256,
        status="READY",
        samples=tuple(samples),
        proxy_max_edge=proxy_max_edge,
        jpeg_quality=jpeg_quality,
    )


def visual_quality_case_id(
    parent_evaluation_run_id: StableId,
    sample_id: StableId,
    requested_reference_frame_ids: tuple[StableId, ...],
    proxy_max_edge: int,
    jpeg_quality: int,
) -> StableId:
    return stable_identifier(
        "visual_quality_case",
        {
            "parent_evaluation_run_id": parent_evaluation_run_id,
            "sample_id": sample_id,
            "requested_reference_frame_ids": requested_reference_frame_ids,
            "proxy_max_edge": proxy_max_edge,
            "jpeg_quality": jpeg_quality,
        },
    )


class VisualQualityCase(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    case_id: StableId
    parent_evaluation_run_id: StableId
    evaluation_run_id: StableId | None = None
    sample_id: StableId
    requested_reference_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)
    proxy_max_edge: StrictInt = Field(ge=1_280, le=2_560)
    jpeg_quality: StrictInt = Field(ge=1, le=100)
    case_status: CaseStatus
    error_code: ErrorCode | None = None
    manifest_sha256: Sha256 | None = None
    call_receipt: ChapterVlmCallReceipt | None = None
    failure_receipt: QwenVisionProviderFailureReceipt | None = None
    response_sha256: Sha256 | None = None
    score_fact: VisualTextScoreFact | None = None
    model: ModelIdentity | None = None
    implementation_sha256: Sha256
    settings_fingerprint: Sha256
    proxy_width: StrictInt | None = Field(default=None, gt=0)
    proxy_height: StrictInt | None = Field(default=None, gt=0)
    proxy_frame_rate: Rational | None = None
    proxy_size_bytes: StrictInt | None = Field(default=None, gt=0)
    proxy_elapsed_ms: StrictInt | None = Field(default=None, ge=0)
    resolution_settings_fingerprint: Sha256 | None = None
    request_json_bytes: StrictInt | None = Field(default=None, ge=0)
    encoded_request_bytes: StrictInt | None = Field(default=None, ge=0)
    vlm_elapsed_ms: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_case_state(self) -> VisualQualityCase:
        if self.case_id != visual_quality_case_id(
            self.parent_evaluation_run_id,
            self.sample_id,
            self.requested_reference_frame_ids,
            self.proxy_max_edge,
            self.jpeg_quality,
        ):
            raise ValueError("质量 case ID 与身份闭包不一致")
        if self.case_status == "NOT_RUN":
            if self.evaluation_run_id is not None or any(
                value is not None
                for value in (
                    self.error_code,
                    self.manifest_sha256,
                    self.call_receipt,
                    self.failure_receipt,
                    self.response_sha256,
                    self.score_fact,
                    self.model,
                    self.proxy_width,
                    self.proxy_height,
                    self.proxy_frame_rate,
                    self.proxy_size_bytes,
                    self.proxy_elapsed_ms,
                    self.resolution_settings_fingerprint,
                    self.request_json_bytes,
                    self.encoded_request_bytes,
                    self.vlm_elapsed_ms,
                )
            ):
                raise ValueError("NOT_RUN case 不得携带已开始运行事实")
        elif self.evaluation_run_id is None:
            raise ValueError("已开始的质量 case 必须绑定产品 Run")
        elif self.evaluation_run_id == self.parent_evaluation_run_id:
            raise ValueError("质量 case 产品 Run 必须区别于父评测 Run")
        if self.case_status == "READY":
            if (
                any(
                    value is None
                    for value in (
                        self.manifest_sha256,
                        self.call_receipt,
                        self.response_sha256,
                        self.score_fact,
                        self.model,
                        self.proxy_width,
                        self.proxy_height,
                        self.proxy_frame_rate,
                        self.proxy_size_bytes,
                        self.proxy_elapsed_ms,
                        self.request_json_bytes,
                        self.encoded_request_bytes,
                        self.vlm_elapsed_ms,
                    )
                )
                or self.error_code is not None
                or self.failure_receipt is not None
            ):
                raise ValueError("READY case 必须包含完整成功事实")
            if self.call_receipt is not None and (
                self.call_receipt.parent_evaluation_run_id != self.parent_evaluation_run_id
                or self.call_receipt.evaluation_run_id != self.evaluation_run_id
                or self.call_receipt.sample_id != self.sample_id
                or self.call_receipt.manifest_sha256 != self.manifest_sha256
                or self.call_receipt.response_sha256 != self.response_sha256
            ):
                raise ValueError("READY case 调用回执与身份闭包不一致")
            if self.score_fact is not None and (
                self.score_fact.parent_evaluation_run_id != self.parent_evaluation_run_id
                or self.score_fact.evaluation_run_id != self.evaluation_run_id
                or self.score_fact.sample_id != self.sample_id
                or self.score_fact.manifest_sha256 != self.manifest_sha256
                or self.score_fact.response_sha256 != self.response_sha256
            ):
                raise ValueError("READY case 评分事实与身份闭包不一致")
        elif self.case_status == "FAIL":
            if self.error_code is None:
                raise ValueError("FAIL case 必须包含稳定错误码")
            if self.score_fact is not None:
                if self.call_receipt is None:
                    raise ValueError("失败 case 的评分事实必须绑定正式调用回执")
                if self.response_sha256 is None:
                    raise ValueError("失败 case 的评分事实必须绑定响应摘要")
                if self.error_code != ErrorCode.VISUAL_RESULT_INVALID:
                    raise ValueError(
                        "失败 case 的评分事实只允许用于视觉结果非法分支"
                    )
                if any(
                    value != 0
                    for value in (
                        self.score_fact.key_field_matches,
                    )
                ):
                    raise ValueError("视觉结果非法分支的评分事实必须是零命中")
                if (
                    self.score_fact.reference_units > 0
                    and self.score_fact.errors != self.score_fact.reference_units
                ):
                    raise ValueError("视觉结果非法分支的评分事实必须是零命中")
            if self.failure_receipt is not None and self.call_receipt is not None:
                raise ValueError("同一失败 case 不得同时保存成功调用回执和失败回执")
            for fact in (self.call_receipt, self.score_fact):
                if fact is not None and (
                    fact.parent_evaluation_run_id != self.parent_evaluation_run_id
                    or fact.evaluation_run_id != self.evaluation_run_id
                    or fact.sample_id != self.sample_id
                ):
                    raise ValueError("失败 case 事实与身份闭包不一致")
        if self.response_sha256 is not None and self.call_receipt is None:
            raise ValueError("响应摘要必须绑定正式调用回执")
        if self.call_receipt is not None and (
            self.call_receipt.parent_evaluation_run_id != self.parent_evaluation_run_id
            or self.call_receipt.evaluation_run_id != self.evaluation_run_id
            or self.call_receipt.sample_id != self.sample_id
            or self.call_receipt.manifest_sha256 != self.manifest_sha256
            or self.call_receipt.response_sha256 != self.response_sha256
        ):
            raise ValueError("调用回执与质量 case 闭包不一致")
        if self.score_fact is not None and (
            self.score_fact.parent_evaluation_run_id != self.parent_evaluation_run_id
            or self.score_fact.evaluation_run_id != self.evaluation_run_id
            or self.score_fact.sample_id != self.sample_id
            or self.score_fact.manifest_sha256 != self.manifest_sha256
            or self.score_fact.response_sha256 != self.response_sha256
        ):
            raise ValueError("评分事实与质量 case 闭包不一致")
        if self.call_receipt is not None and (
            (self.request_json_bytes, self.encoded_request_bytes, self.vlm_elapsed_ms)
            != (
                self.call_receipt.request_json_bytes,
                self.call_receipt.encoded_request_bytes,
                self.call_receipt.vlm_elapsed_ms,
            )
        ):
            raise ValueError("调用字节数和耗时必须来自正式调用回执")
        return self


class VisualQualityReport(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["1.0.0"]
    parent_evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    status: VisualQualityStatus
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    failure_code: ErrorCode | None = None
    planned_case_ids: tuple[StableId, ...] = Field(max_length=240)
    cases: tuple[VisualQualityCase, ...] = Field(max_length=240)
    visual_text_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_text_accuracy_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    visual_key_field_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_key_field_recall_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_report_contract(self) -> VisualQualityReport:
        case_ids = tuple(case.case_id for case in self.cases)
        if case_ids != self.planned_case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("质量报告 cases 必须按 planned_case_ids 同序完整覆盖")
        if any(
            case.parent_evaluation_run_id != self.parent_evaluation_run_id for case in self.cases
        ):
            raise ValueError("质量报告 case 父 Run 不一致")
        for value, reason, name in (
            (
                self.visual_text_accuracy,
                self.visual_text_accuracy_not_run_reason,
                "visual_text_accuracy",
            ),
            (
                self.visual_key_field_recall,
                self.visual_key_field_recall_not_run_reason,
                "visual_key_field_recall",
            ),
        ):
            if (value is None) == (reason is None):
                raise ValueError(f"{name} 必须恰好包含数值或未运行原因")
        if self.status == "NOT_RUN" and (not self.not_run_reason or self.failure_code is not None):
            raise ValueError("NOT_RUN 报告必须有原因且不得有失败码")
        if self.status == "NOT_RUN" and any(case.case_status != "NOT_RUN" for case in self.cases):
            raise ValueError("NOT_RUN 报告只能包含未启动 case")
        if self.status == "FAIL" and (self.failure_code is None or self.not_run_reason is not None):
            raise ValueError("FAIL 报告必须有失败码且不得有未运行原因")
        if self.status == "FAIL" and (
            self.visual_text_accuracy is not None or self.visual_key_field_recall is not None
        ):
            raise ValueError("FAIL 报告不得聚合成功子集")
        if self.status == "SUCCEEDED" and (
            self.not_run_reason is not None
            or self.failure_code is not None
            or any(case.case_status != "READY" for case in self.cases)
        ):
            raise ValueError("SUCCEEDED 报告必须完整覆盖 READY case")
        if self.report_sha256 != _report_sha(self):
            raise ValueError("质量报告摘要不匹配")
        return self


class VisualResolutionPair(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    parent_evaluation_run_id: StableId
    pair_id: StableId
    sample_id: StableId
    requested_reference_frame_ids: tuple[StableId, ...] = Field(min_length=2, max_length=4)
    jpeg_quality: StrictInt = Field(ge=1, le=100)
    case_1280_id: StableId
    case_1920_id: StableId
    actual_1280_width: StrictInt | None = Field(default=None, gt=0)
    actual_1280_height: StrictInt | None = Field(default=None, gt=0)
    actual_1920_width: StrictInt | None = Field(default=None, gt=0)
    actual_1920_height: StrictInt | None = Field(default=None, gt=0)
    status: Literal["COMPARABLE", "INCONCLUSIVE_SOURCE_RESOLUTION", "NOT_RUN", "FAIL"]
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    failure_code: ErrorCode | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> VisualResolutionPair:
        expected = stable_identifier(
            "visual_resolution_pair",
            {
                "parent_evaluation_run_id": self.parent_evaluation_run_id,
                "sample_id": self.sample_id,
                "requested_reference_frame_ids": self.requested_reference_frame_ids,
                "jpeg_quality": self.jpeg_quality,
            },
        )
        if self.pair_id != expected:
            raise ValueError("分辨率 Pair ID 不匹配")
        pixels = (
            self.actual_1280_width,
            self.actual_1280_height,
            self.actual_1920_width,
            self.actual_1920_height,
        )
        if self.status in {"COMPARABLE", "INCONCLUSIVE_SOURCE_RESOLUTION"} and any(
            value is None for value in pixels
        ):
            raise ValueError("已完成分辨率 Pair 必须有实际像素")
        if self.status == "NOT_RUN" and (
            any(value is not None for value in pixels) or not self.not_run_reason
        ):
            raise ValueError("NOT_RUN Pair 不得有像素且必须说明原因")
        if self.status == "FAIL" and self.failure_code is None:
            raise ValueError("FAIL Pair 必须包含失败码")
        return self


class VisualResolutionReport(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["1.0.0"]
    parent_evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    status: VisualQualityStatus
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    failure_code: ErrorCode | None = None
    quality_report_1280_sha256: Sha256 | None = None
    quality_report_1920_sha256: Sha256 | None = None
    visual_text_accuracy_1280: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_text_accuracy_1280_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    visual_text_accuracy_1920: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_text_accuracy_1920_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    visual_key_field_recall_1280: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_key_field_recall_1280_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    visual_key_field_recall_1920: float | None = Field(default=None, ge=0.0, le=1.0)
    visual_key_field_recall_1920_not_run_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    planned_pair_ids: tuple[StableId, ...] = Field(max_length=240)
    pairs: tuple[VisualResolutionPair, ...] = Field(max_length=240)
    resolution_decision: Literal[
        "INCONCLUSIVE", "KEEP_1920", "PREFER_1280", "TRADEOFF_REVIEW_REQUIRED"
    ]
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_resolution_report(self) -> VisualResolutionReport:
        pair_ids = tuple(pair.pair_id for pair in self.pairs)
        if pair_ids != self.planned_pair_ids or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("分辨率报告 pairs 必须按 planned_pair_ids 同序完整覆盖")
        if any(
            pair.parent_evaluation_run_id != self.parent_evaluation_run_id for pair in self.pairs
        ):
            raise ValueError("分辨率 Pair 父 Run 不一致")
        if self.status == "NOT_RUN":
            if not self.not_run_reason or self.failure_code is not None:
                raise ValueError("NOT_RUN 分辨率报告必须有原因且不得有失败码")
            if any(
                value is not None
                for value in (
                    self.quality_report_1280_sha256,
                    self.quality_report_1920_sha256,
                    self.visual_text_accuracy_1280,
                    self.visual_text_accuracy_1920,
                    self.visual_key_field_recall_1280,
                    self.visual_key_field_recall_1920,
                )
            ) or any(pair.status != "NOT_RUN" for pair in self.pairs):
                raise ValueError("NOT_RUN 分辨率报告不得携带执行事实")
            if self.resolution_decision != "INCONCLUSIVE":
                raise ValueError("NOT_RUN 分辨率结论必须是 INCONCLUSIVE")
        elif self.status == "FAIL":
            if self.failure_code is None or self.not_run_reason is not None:
                raise ValueError("FAIL 分辨率报告必须有失败码且不得有未运行原因")
            if (
                any(
                    value is not None
                    for value in (
                        self.visual_text_accuracy_1280,
                        self.visual_text_accuracy_1920,
                        self.visual_key_field_recall_1280,
                        self.visual_key_field_recall_1920,
                    )
                )
                or self.resolution_decision != "INCONCLUSIVE"
            ):
                raise ValueError("FAIL 分辨率报告不得聚合分数")
        else:
            if self.not_run_reason is not None or self.failure_code is not None:
                raise ValueError("SUCCEEDED 分辨率报告不得包含失败或未运行原因")
            if any(
                pair.status not in {"COMPARABLE", "INCONCLUSIVE_SOURCE_RESOLUTION"}
                for pair in self.pairs
            ):
                raise ValueError("SUCCEEDED 分辨率报告必须完整覆盖可比较 Pair")
            if self.quality_report_1280_sha256 is None or self.quality_report_1920_sha256 is None:
                raise ValueError("SUCCEEDED 分辨率报告必须绑定两侧质量报告")
        metric_values = (
            (
                self.visual_text_accuracy_1280,
                self.visual_text_accuracy_1280_not_run_reason,
                "1280 视觉文字准确率",
            ),
            (
                self.visual_text_accuracy_1920,
                self.visual_text_accuracy_1920_not_run_reason,
                "1920 视觉文字准确率",
            ),
            (
                self.visual_key_field_recall_1280,
                self.visual_key_field_recall_1280_not_run_reason,
                "1280 关键字段召回率",
            ),
            (
                self.visual_key_field_recall_1920,
                self.visual_key_field_recall_1920_not_run_reason,
                "1920 关键字段召回率",
            ),
        )
        if self.status == "SUCCEEDED":
            for value, reason, name in metric_values:
                if (value is None) == (reason is None):
                    raise ValueError(f"{name} 必须恰好包含数值或未运行原因")
        elif any(value is not None or reason is not None for value, reason, _ in metric_values):
            raise ValueError("未成功分辨率报告不得携带聚合指标或指标原因")
        if self.report_sha256 != _resolution_report_sha(self):
            raise ValueError("分辨率报告摘要不匹配")
        return self


class VerifiedVisualQualityReport(FrozenModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    report: VisualQualityReport
    verified_case_ids: tuple[StableId, ...] = ()
    source_manifest_sha256s: tuple[Sha256, ...] = ()

    @model_validator(mode="after")
    def validate_verified_closure(self) -> VerifiedVisualQualityReport:
        if len(self.verified_case_ids) != len(self.source_manifest_sha256s):
            raise ValueError("验证 case 与 Manifest 摘要数量不一致")
        if len(set(self.verified_case_ids)) != len(self.verified_case_ids):
            raise ValueError("验证 case 不得重复")
        expected_ids = tuple(
            case.case_id for case in self.report.cases if case.case_status != "NOT_RUN"
        )
        if self.verified_case_ids != expected_ids:
            raise ValueError("验证 case 必须按报告计划顺序排列")
        report_cases = {case.case_id: case for case in self.report.cases}
        if any(case_id not in report_cases for case_id in self.verified_case_ids):
            raise ValueError("验证 case 必须来自报告")
        for case_id, manifest_sha in zip(
            self.verified_case_ids, self.source_manifest_sha256s, strict=True
        ):
            case = report_cases[case_id]
            if case.manifest_sha256 != manifest_sha or case.manifest_sha256 is None:
                raise ValueError("验证 case 与 Manifest 摘要未绑定同一事实")
        if self.report.status == "NOT_RUN" and (
            self.verified_case_ids or self.source_manifest_sha256s
        ):
            raise ValueError("NOT_RUN 报告不得列出已验证 case")
        return self


def build_visual_quality_report(
    quality_set: VisualQualitySet,
    verified_parent_package: object,
    case_results: Iterable[VisualQualityCase],
) -> VisualQualityReport:
    _validate_quality_set_parent(quality_set, verified_parent_package)
    cases = tuple(case_results)
    planned = tuple(
        visual_quality_case_id(
            quality_set.parent_evaluation_run_id,
            sample.sample_id,
            sample.requested_reference_frame_ids,
            quality_set.proxy_max_edge,
            quality_set.jpeg_quality,
        )
        for sample in quality_set.samples
    )
    by_id = {case.case_id: case for case in cases}
    if len(cases) != len(planned) or len(by_id) != len(cases) or set(by_id) != set(planned):
        raise ValueError("质量报告必须回填全部计划 case")
    ordered = tuple(by_id[item] for item in planned)
    if quality_set.status == "NOT_RUN":
        status: VisualQualityStatus = "NOT_RUN"
        failure_code = None
        reason = quality_set.not_run_reason
    elif ordered and all(case.case_status == "NOT_RUN" for case in ordered):
        status = "NOT_RUN"
        failure_code = None
        reason = "所有视觉质量 case 尚未启动"
    elif any(case.case_status == "FAIL" for case in ordered):
        status = "FAIL"
        failure_code = next(case.error_code for case in ordered if case.case_status == "FAIL")
        reason = None
    elif any(case.case_status == "NOT_RUN" for case in ordered):
        status = "FAIL"
        failure_code = ErrorCode.ARTIFACT_SCHEMA_INVALID
        reason = None
    else:
        status = "SUCCEEDED"
        failure_code = None
        reason = None
    facts = tuple(case.score_fact for case in ordered if case.score_fact is not None)
    text_units = sum(fact.reference_units for fact in facts)
    text_errors = sum(fact.errors for fact in facts)
    key_units = sum(fact.key_field_reference_units for fact in facts)
    key_matches = sum(fact.key_field_matches for fact in facts)
    text_value = (
        None if status != "SUCCEEDED" or text_units == 0 else max(0.0, 1 - text_errors / text_units)
    )
    key_value = None if status != "SUCCEEDED" or key_units == 0 else key_matches / key_units
    payload = {
        "schema_version": "1.0.0",
        "parent_evaluation_run_id": quality_set.parent_evaluation_run_id,
        "dataset_sha256": quality_set.dataset_sha256,
        "authorization_sha256": quality_set.authorization_sha256,
        "status": status,
        "not_run_reason": reason,
        "failure_code": failure_code,
        "planned_case_ids": planned,
        "cases": ordered,
        "visual_text_accuracy": text_value,
        "visual_text_accuracy_not_run_reason": (
            "总体报告失败，未聚合" if status == "FAIL" else "代表性质量集没有参考字符"
        )
        if text_value is None
        else None,
        "visual_key_field_recall": key_value,
        "visual_key_field_recall_not_run_reason": (
            "总体报告失败，未聚合" if status == "FAIL" else "代表性质量集没有关键字段"
        )
        if key_value is None
        else None,
        "report_sha256": "0" * 64,
    }
    provisional = VisualQualityReport.model_construct(**cast(Any, payload))
    payload["report_sha256"] = _report_sha(provisional)
    return VisualQualityReport.model_validate(payload)


def verify_visual_quality_report(
    report: VisualQualityReport,
    quality_set: VisualQualitySet,
    verified_parent_package: object,
) -> VerifiedVisualQualityReport:
    _validate_quality_set_parent(quality_set, verified_parent_package)
    expected = build_visual_quality_report(quality_set, verified_parent_package, report.cases)
    if expected != report:
        raise ValueError("质量报告与显式质量集重验结果不一致")
    ready = tuple(case for case in report.cases if case.case_status in {"READY", "FAIL"})
    if report.status == "SUCCEEDED" and len(ready) != len(report.cases):
        raise ValueError("SUCCEEDED 报告必须验证全部 case")
    if report.status == "FAIL" and any(
        case.case_status in {"READY", "FAIL"} and case.manifest_sha256 is None
        for case in report.cases
    ):
        raise ValueError("已开始 case 缺少 Manifest 摘要，不能通过重验")
    return VerifiedVisualQualityReport(
        report=report,
        verified_case_ids=tuple(case.case_id for case in ready),
        source_manifest_sha256s=tuple(
            case.manifest_sha256 for case in ready if case.manifest_sha256 is not None
        ),
    )


def build_visual_resolution_pair(
    case_1280: VisualQualityCase,
    case_1920: VisualQualityCase,
    *,
    expected_parent_evaluation_run_id: StableId,
    expected_sample_id: StableId,
    expected_requested_reference_frame_ids: tuple[StableId, ...],
    expected_jpeg_quality: int,
    quality_report_1280: VisualQualityReport | None,
    quality_report_1920: VisualQualityReport | None,
) -> VisualResolutionPair:
    if (
        case_1280.parent_evaluation_run_id,
        case_1920.parent_evaluation_run_id,
    ) != (expected_parent_evaluation_run_id,) * 2:
        raise ValueError("分辨率 Pair 父 Run 不一致")
    if any(
        (case.sample_id, case.requested_reference_frame_ids, case.jpeg_quality)
        != (expected_sample_id, expected_requested_reference_frame_ids, expected_jpeg_quality)
        for case in (case_1280, case_1920)
    ):
        raise ValueError("分辨率 Pair 输入闭包不一致")
    if {case_1280.proxy_max_edge, case_1920.proxy_max_edge} != {1280, 1920}:
        raise ValueError("分辨率 Pair 必须是 1280/1920")
    if (case_1280.proxy_max_edge, case_1920.proxy_max_edge) != (1280, 1920):
        raise ValueError("分辨率 Pair 必须按 1280、1920 顺序传入")
    if case_1280.case_status == "READY" and (
        quality_report_1280 is None or quality_report_1280.status != "SUCCEEDED"
    ):
        raise ValueError("READY 的 1280 case 必须绑定成功质量报告")
    if case_1920.case_status == "READY" and (
        quality_report_1920 is None or quality_report_1920.status != "SUCCEEDED"
    ):
        raise ValueError("READY 的 1920 case 必须绑定成功质量报告")
    if quality_report_1280 is not None and not _report_contains_case(
        quality_report_1280, case_1280.case_id
    ):
        raise ValueError("1280 质量报告未覆盖 Pair case")
    if quality_report_1920 is not None and not _report_contains_case(
        quality_report_1920, case_1920.case_id
    ):
        raise ValueError("1920 质量报告未覆盖 Pair case")
    if case_1280.case_status == "READY" and case_1920.case_status == "READY":
        shared = (
            case_1280.sample_id,
            case_1280.requested_reference_frame_ids,
            case_1280.jpeg_quality,
            case_1280.model,
            case_1280.implementation_sha256,
            case_1280.resolution_settings_fingerprint or case_1280.settings_fingerprint,
        )
        other = (
            case_1920.sample_id,
            case_1920.requested_reference_frame_ids,
            case_1920.jpeg_quality,
            case_1920.model,
            case_1920.implementation_sha256,
            case_1920.resolution_settings_fingerprint or case_1920.settings_fingerprint,
        )
        if shared != other:
            raise ValueError("分辨率 Pair 除代理分辨率外的实现闭包必须一致")
    pair_id = stable_identifier(
        "visual_resolution_pair",
        {
            "parent_evaluation_run_id": expected_parent_evaluation_run_id,
            "sample_id": expected_sample_id,
            "requested_reference_frame_ids": expected_requested_reference_frame_ids,
            "jpeg_quality": expected_jpeg_quality,
        },
    )
    if case_1280.case_status == "NOT_RUN" and case_1920.case_status == "NOT_RUN":
        return VisualResolutionPair(
            parent_evaluation_run_id=expected_parent_evaluation_run_id,
            pair_id=pair_id,
            sample_id=expected_sample_id,
            requested_reference_frame_ids=expected_requested_reference_frame_ids,
            jpeg_quality=expected_jpeg_quality,
            case_1280_id=case_1280.case_id,
            case_1920_id=case_1920.case_id,
            status="NOT_RUN",
            not_run_reason="两侧分辨率实验尚未开始",
        )
    if case_1280.case_status == "FAIL" or case_1920.case_status == "FAIL":
        failed = case_1280 if case_1280.case_status == "FAIL" else case_1920
        return VisualResolutionPair(
            parent_evaluation_run_id=expected_parent_evaluation_run_id,
            pair_id=pair_id,
            sample_id=expected_sample_id,
            requested_reference_frame_ids=expected_requested_reference_frame_ids,
            jpeg_quality=expected_jpeg_quality,
            case_1280_id=case_1280.case_id,
            case_1920_id=case_1920.case_id,
            status="FAIL",
            failure_code=failed.error_code,
        )
    if case_1280.case_status != "READY" or case_1920.case_status != "READY":
        return VisualResolutionPair(
            parent_evaluation_run_id=expected_parent_evaluation_run_id,
            pair_id=pair_id,
            sample_id=expected_sample_id,
            requested_reference_frame_ids=expected_requested_reference_frame_ids,
            jpeg_quality=expected_jpeg_quality,
            case_1280_id=case_1280.case_id,
            case_1920_id=case_1920.case_id,
            status="FAIL",
            failure_code=ErrorCode.ARTIFACT_SCHEMA_INVALID,
        )
    pixels = (
        case_1280.proxy_width,
        case_1280.proxy_height,
        case_1920.proxy_width,
        case_1920.proxy_height,
    )
    comparable = (case_1280.proxy_width, case_1280.proxy_height) != (
        case_1920.proxy_width,
        case_1920.proxy_height,
    )
    return VisualResolutionPair(
        parent_evaluation_run_id=expected_parent_evaluation_run_id,
        pair_id=pair_id,
        sample_id=expected_sample_id,
        requested_reference_frame_ids=expected_requested_reference_frame_ids,
        jpeg_quality=expected_jpeg_quality,
        case_1280_id=case_1280.case_id,
        case_1920_id=case_1920.case_id,
        actual_1280_width=pixels[0],
        actual_1280_height=pixels[1],
        actual_1920_width=pixels[2],
        actual_1920_height=pixels[3],
        status="COMPARABLE" if comparable else "INCONCLUSIVE_SOURCE_RESOLUTION",
    )


def build_visual_resolution_report(
    quality_set: VisualQualitySet,
    verified_parent_package: object,
    pairs: Iterable[VisualResolutionPair],
    quality_report_1280: VisualQualityReport | None,
    quality_report_1920: VisualQualityReport | None,
) -> VisualResolutionReport:
    """按质量集完整身份闭包构造分辨率对照报告。"""

    _validate_quality_set_parent(quality_set, verified_parent_package)
    ordered_pairs = tuple(pairs)
    planned = tuple(
        stable_identifier(
            "visual_resolution_pair",
            {
                "parent_evaluation_run_id": quality_set.parent_evaluation_run_id,
                "sample_id": sample.sample_id,
                "requested_reference_frame_ids": sample.requested_reference_frame_ids,
                "jpeg_quality": quality_set.jpeg_quality,
            },
        )
        for sample in quality_set.samples
    )
    if tuple(pair.pair_id for pair in ordered_pairs) != planned:
        raise ValueError("分辨率报告必须按质量集顺序完整覆盖 Pair")
    if quality_set.status == "NOT_RUN" or (
        quality_report_1280 is None and quality_report_1920 is None
    ):
        status: VisualQualityStatus = "NOT_RUN"
        failure_code = None
        not_run_reason = quality_set.not_run_reason or "两侧分辨率质量报告尚未生成"
    elif (
        any(pair.status == "FAIL" for pair in ordered_pairs)
        or quality_report_1280 is None
        or quality_report_1920 is None
        or quality_report_1280.status != "SUCCEEDED"
        or quality_report_1920.status != "SUCCEEDED"
    ):
        status = "FAIL"
        failure_code = next(
            (pair.failure_code for pair in ordered_pairs if pair.status == "FAIL"),
            ErrorCode.ARTIFACT_SCHEMA_INVALID,
        )
        not_run_reason = None
    else:
        status = "SUCCEEDED"
        failure_code = None
        not_run_reason = None
    if status == "SUCCEEDED":
        assert quality_report_1280 is not None and quality_report_1920 is not None
        if (
            quality_report_1280.status != "SUCCEEDED"
            or quality_report_1920.status != "SUCCEEDED"
            or quality_report_1280.dataset_sha256 != quality_set.dataset_sha256
            or quality_report_1920.dataset_sha256 != quality_set.dataset_sha256
            or quality_report_1280.authorization_sha256 != quality_set.authorization_sha256
            or quality_report_1920.authorization_sha256 != quality_set.authorization_sha256
        ):
            raise ValueError("分辨率报告必须绑定同一已重验质量集")
    text_1280 = None if quality_report_1280 is None else quality_report_1280.visual_text_accuracy
    text_1920 = None if quality_report_1920 is None else quality_report_1920.visual_text_accuracy
    key_1280 = None if quality_report_1280 is None else quality_report_1280.visual_key_field_recall
    key_1920 = None if quality_report_1920 is None else quality_report_1920.visual_key_field_recall
    text_1280_reason = None if text_1280 is not None else "1280 质量报告没有参考字符分母"
    text_1920_reason = None if text_1920 is not None else "1920 质量报告没有参考字符分母"
    key_1280_reason = None if key_1280 is not None else "1280 质量报告没有关键字段分母"
    key_1920_reason = None if key_1920 is not None else "1920 质量报告没有关键字段分母"
    if status == "NOT_RUN":
        text_1280 = text_1920 = key_1280 = key_1920 = None
        text_1280_reason = text_1920_reason = key_1280_reason = key_1920_reason = None
    elif status != "SUCCEEDED":
        reason = "总体报告失败，未聚合"
        text_1280 = text_1920 = key_1280 = key_1920 = None
        text_1280_reason = text_1920_reason = key_1280_reason = key_1920_reason = reason
    decision = _resolution_decision(text_1280, text_1920, key_1280, key_1920)
    if status == "SUCCEEDED" and not any(pair.status == "COMPARABLE" for pair in ordered_pairs):
        decision = "INCONCLUSIVE"
    payload = {
        "schema_version": "1.0.0",
        "parent_evaluation_run_id": quality_set.parent_evaluation_run_id,
        "dataset_sha256": quality_set.dataset_sha256,
        "authorization_sha256": quality_set.authorization_sha256,
        "status": status,
        "not_run_reason": not_run_reason,
        "failure_code": failure_code,
        "quality_report_1280_sha256": (
            None
            if status != "SUCCEEDED" or quality_report_1280 is None
            else quality_report_1280.report_sha256
        ),
        "quality_report_1920_sha256": (
            None
            if status != "SUCCEEDED" or quality_report_1920 is None
            else quality_report_1920.report_sha256
        ),
        "visual_text_accuracy_1280": text_1280,
        "visual_text_accuracy_1280_not_run_reason": text_1280_reason,
        "visual_text_accuracy_1920": text_1920,
        "visual_text_accuracy_1920_not_run_reason": text_1920_reason,
        "visual_key_field_recall_1280": key_1280,
        "visual_key_field_recall_1280_not_run_reason": key_1280_reason,
        "visual_key_field_recall_1920": key_1920,
        "visual_key_field_recall_1920_not_run_reason": key_1920_reason,
        "planned_pair_ids": planned,
        "pairs": ordered_pairs,
        "resolution_decision": decision,
        "report_sha256": "0" * 64,
    }
    provisional = VisualResolutionReport.model_construct(**cast(Any, payload))
    payload["report_sha256"] = _resolution_report_sha(provisional)
    return VisualResolutionReport.model_validate(payload)


def _resolution_decision(
    text_1280: float | None,
    text_1920: float | None,
    key_1280: float | None,
    key_1920: float | None,
) -> Literal["INCONCLUSIVE", "KEEP_1920", "PREFER_1280", "TRADEOFF_REVIEW_REQUIRED"]:
    if text_1280 is None or text_1920 is None or key_1280 is None or key_1920 is None:
        return "INCONCLUSIVE"
    text_up = text_1920 > text_1280
    key_up = key_1920 > key_1280
    text_down = text_1920 < text_1280
    key_down = key_1920 < key_1280
    if (text_up or key_up) and not (text_down or key_down):
        return "KEEP_1920"
    if not text_up and not key_up:
        return "PREFER_1280"
    return "TRADEOFF_REVIEW_REQUIRED"


def _report_sha(report: VisualQualityReport) -> Sha256:
    payload = report.model_dump(mode="json", exclude={"report_sha256"})
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolution_report_sha(report: VisualResolutionReport) -> Sha256:
    payload = report.model_dump(mode="json", exclude={"report_sha256"})
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_quality_set_parent(quality_set: VisualQualitySet, package: object) -> None:
    """校验质量集确实来自显式重验评测包，而不是调用方自报摘要。"""

    if not isinstance(package, ValidatedEvaluationPackage):
        # 质量集只在前置条件不足时允许暂存计划身份；可执行集合必须绑定正式包。
        if quality_set.status != "NOT_RUN":
            raise ValueError("质量集必须绑定已重验评测包")
        return
    if (
        package.dataset_sha256 != quality_set.dataset_sha256
        or package.authorization_sha256 != quality_set.authorization_sha256
    ):
        raise ValueError("质量集与评测包摘要不一致")
    annotations = {item.annotation.sample_id: item.annotation for item in package.annotations}
    samples = {item.sample_id: item for item in package.dataset.samples}
    for sample in quality_set.samples:
        annotation = annotations.get(sample.sample_id)
        dataset_sample = samples.get(sample.sample_id)
        if annotation is None or dataset_sample is None:
            raise ValueError("质量集样本不在已重验评测包中")
        if annotation.media_sha256 != dataset_sample.media_sha256:
            raise ValueError("质量集样本媒体摘要不一致")
        frame_ids = {frame.frame_id for frame in annotation.visual_frames}
        if not set(sample.requested_reference_frame_ids).issubset(frame_ids):
            raise ValueError("质量集请求帧不在已重验标注中")


def _report_contains_case(report: VisualQualityReport, case_id: StableId) -> bool:
    return case_id in report.planned_case_ids and any(
        case.case_id == case_id for case in report.cases
    )


__all__ = [
    "QualitySetStatus",
    "VerifiedVisualQualityReport",
    "VisualQualityCase",
    "VisualQualityReport",
    "VisualQualitySample",
    "VisualQualitySet",
    "VisualResolutionPair",
    "VisualResolutionReport",
    "build_visual_quality_report",
    "build_visual_quality_set",
    "build_visual_resolution_pair",
    "build_visual_resolution_report",
    "verify_visual_quality_report",
    "visual_quality_case_id",
]
