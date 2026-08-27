"""知识文档质量判断与确定性质量指标。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import AwareDatetime, ConfigDict, Field, ValidationError, model_validator

from video_demo.application.document_rendering import render_markdown
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.document import validate_evidence_references
from video_demo.domain.evidence import VisualObservationEvidence
from video_demo.evaluation.annotations import VerifiedAnnotation
from video_demo.evaluation.visual_quality import VerifiedVisualQualityReport

if TYPE_CHECKING:
    from video_demo.evaluation.predictions import VerifiedPrediction

RelevanceVerdict = Literal["RELEVANT", "PARTIAL", "IRRELEVANT"]
VisualSupplementVerdict = Literal["SUPPLEMENTS", "DUPLICATES", "CONFLICTS", "UNCLEAR"]


class ChapterDocumentJudgment(FrozenModel):
    """一个章节的标题、摘要和全部视觉观察判断。"""

    chapter_id: StableId
    title_relevance: RelevanceVerdict
    summary_relevance: RelevanceVerdict
    visual_observation_refs: tuple[StableId, ...] = ()
    visual_supplement_judgments: tuple[
        tuple[StableId, VisualSupplementVerdict], ...
    ] = ()

    @model_validator(mode="after")
    def validate_visual_coverage(self) -> Self:
        refs = self.visual_observation_refs
        judged_refs = tuple(item[0] for item in self.visual_supplement_judgments)
        if len(refs) != len(set(refs)):
            raise ValueError("视觉观察引用不得重复")
        if len(judged_refs) != len(set(judged_refs)) or judged_refs != refs:
            raise ValueError("视觉观察判断必须按引用顺序完整覆盖")
        return self


class DocumentQualityJudgment(FrozenModel):
    """绑定一份生产文档的完整人工审阅。"""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    sample_id: StableId
    annotation_sha256: Sha256
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    prediction_index_sha256: Sha256
    result_sha256: Sha256
    evidence_sha256: Sha256
    document_sha256: Sha256
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: AwareDatetime
    rubric_version: Literal["document-quality-v1"]
    chapters: tuple[ChapterDocumentJudgment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_chapters(self) -> Self:
        chapter_ids = tuple(item.chapter_id for item in self.chapters)
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("章节判断不得重复")
        return self


class DocumentMetricObservation(FrozenModel):
    """一个质量指标的数值，或明确说明为什么本次没有运行。"""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    value: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_value_or_reason(self) -> Self:
        if (self.value is None) == (self.not_run_reason is None):
            raise ValueError("指标必须恰好包含 value 或 not_run_reason")
        return self


class DocumentQualityReport(FrozenModel):
    """独立于通用语义质量报告的文档质量事实。"""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")
    schema_version: Literal["1.0.0"]
    evaluation_run_id: StableId
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    prediction_index_sha256: Sha256
    status: Literal["NOT_RUN", "SUCCEEDED", "FAIL"]
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    failure_code: str | None = Field(default=None, min_length=3, max_length=128)
    automatic_metrics: Mapping[str, float | None] | None = None
    visual_quality_status: Literal["NOT_RUN", "SUCCEEDED", "FAIL"]
    visual_quality_metrics: Mapping[str, float | None]
    visual_quality_not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)
    visual_quality_failure_code: str | None = Field(default=None, min_length=3, max_length=128)
    human_metrics: Mapping[str, DocumentMetricObservation] | None = None
    judgment_sha256: Sha256 | None = None
    report_sha256: Sha256

    @property
    def metrics(self) -> tuple[object, ...]:
        """兼容旧质量查看器；正式 JSON 契约使用独立的自动/人工映射。"""

        class MetricView:
            def __init__(self, name: str, value: float | None, reason: str | None = None) -> None:
                self.name = name
                self.value = value
                self.not_run_reason = reason

        if self.automatic_metrics is None:
            return ()
        values: list[object] = [
            MetricView(name, value) for name, value in self.automatic_metrics.items()
        ]
        if self.human_metrics is not None:
            values.extend(
                MetricView(name, metric.value, metric.not_run_reason)
                for name, metric in self.human_metrics.items()
            )
        return tuple(values)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if set(self.visual_quality_metrics) != {
            "visual_text_accuracy",
            "visual_key_field_recall",
        }:
            raise ValueError("视觉质量指标必须恰好包含两个固定键")
        self._validate_visual_quality_state()
        if self.status == "NOT_RUN":
            if (
                self.automatic_metrics is not None
                or self.human_metrics is not None
                or self.judgment_sha256 is not None
                or self.not_run_reason is None
                or self.failure_code is not None
            ):
                raise ValueError("NOT_RUN 文档报告不得携带质量分数或失败码")
        elif self.status == "FAIL":
            if (
                self.automatic_metrics is not None
                or self.human_metrics is not None
                or self.judgment_sha256 is not None
                or self.failure_code is None
                or self.not_run_reason is not None
            ):
                raise ValueError("FAIL 文档报告不得携带部分质量分数")
        else:
            if (
                self.automatic_metrics is None
                or set(self.automatic_metrics) != {
                    "chapter_time_coverage",
                    "claim_evidence_rate",
                    "markdown_json_consistency",
                }
                or self.human_metrics is None
                or set(self.human_metrics) != {
                    "title_relevance",
                    "summary_relevance",
                    "visual_supplement_rate",
                    "visual_duplicate_rate",
                }
                or self.judgment_sha256 is None
                or self.not_run_reason is not None
                or self.failure_code is not None
            ):
                raise ValueError("SUCCEEDED 文档报告必须完整包含自动和人工指标")
            if any(
                value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.automatic_metrics.values()
            ):
                raise ValueError("SUCCEEDED 自动指标必须是有限数值")
        if self.report_sha256 != document_quality_report_sha256(self):
            raise ValueError("文档质量报告摘要不匹配")
        return self

    def _validate_visual_quality_state(self) -> None:
        values = tuple(self.visual_quality_metrics.values())
        if any(
            value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
            for value in values
        ):
            raise ValueError("视觉质量指标必须是 0 到 1 的有限数值")
        if self.visual_quality_status == "NOT_RUN":
            if (
                any(value is not None for value in values)
                or self.visual_quality_not_run_reason is None
                or self.visual_quality_failure_code is not None
            ):
                raise ValueError("NOT_RUN 视觉质量必须只有原因")
        elif self.visual_quality_status == "FAIL":
            if (
                any(value is not None for value in values)
                or self.visual_quality_not_run_reason is not None
                or self.visual_quality_failure_code is None
            ):
                raise ValueError("FAIL 视觉质量必须只有失败码")
        else:
            if self.visual_quality_failure_code is not None:
                raise ValueError("SUCCEEDED 视觉质量不得包含失败码")
            if (
                any(value is None for value in values)
                and self.visual_quality_not_run_reason is None
            ):
                raise ValueError("SUCCEEDED 视觉质量缺失指标必须说明未运行原因")
            if (
                all(value is not None for value in values)
                and self.visual_quality_not_run_reason is not None
            ):
                raise ValueError("SUCCEEDED 视觉质量完整指标不得携带未运行原因")

def verify_document_quality_judgment(
    judgment: DocumentQualityJudgment,
    annotation: VerifiedAnnotation,
    prediction: VerifiedPrediction,
    *,
    dataset_sha256: Sha256 | None = None,
    authorization_sha256: Sha256 | None = None,
) -> DocumentQualityJudgment:
    """验证人工判断只引用当前样本的章节和视觉观察，并完整覆盖它们。"""

    if prediction.result is None:
        raise ValueError("失败预测没有可审阅文档")
    index = prediction.index
    expected: dict[str, object] = {
        "evaluation_run_id": index.evaluation_run_id,
        "sample_id": annotation.annotation.sample_id,
        "annotation_sha256": annotation.sha256,
        "prediction_index_sha256": prediction.index_sha256,
        "result_sha256": index.result_sha256,
        "evidence_sha256": index.evidence_sha256,
        "document_sha256": index.document_sha256,
    }
    if dataset_sha256 is not None:
        expected["dataset_sha256"] = dataset_sha256
    if authorization_sha256 is not None:
        expected["authorization_sha256"] = authorization_sha256
    actual = judgment.model_dump(mode="python", include=set(expected))
    if any(actual[key] != value for key, value in expected.items()):
        raise ValueError("文档判断与样本或生产制品摘要不匹配")
    validate_evidence_references(prediction.result, prediction.evidence)
    chapter_ids = tuple(chapter.chapter_id for chapter in prediction.result.chapters)
    if tuple(item.chapter_id for item in judgment.chapters) != chapter_ids:
        raise ValueError("文档判断必须按结果顺序完整覆盖所有章节")
    expected_visuals: dict[str, tuple[str, ...]] = {}
    for item in prediction.evidence:
        if isinstance(item, VisualObservationEvidence):
            expected_visuals[item.chapter_id] = (
                *expected_visuals.get(item.chapter_id, ()),
                item.evidence_id,
            )
    for chapter_judgment in judgment.chapters:
        if chapter_judgment.visual_observation_refs != expected_visuals.get(
            chapter_judgment.chapter_id, ()
        ):
            raise ValueError("文档判断必须按结果顺序完整覆盖所有视觉观察")
    return judgment


def load_document_quality_judgment(
    path: Path,
    *,
    annotation: VerifiedAnnotation,
    prediction: VerifiedPrediction,
    dataset_sha256: Sha256 | None = None,
    authorization_sha256: Sha256 | None = None,
) -> DocumentQualityJudgment:
    """读取并立即绑定一份人工判断，底层路径错误不向调用方泄露。"""

    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("判断文件不存在")
        judgment = DocumentQualityJudgment.model_validate_json(path.read_bytes())
        return verify_document_quality_judgment(
            judgment,
            annotation,
            prediction,
            dataset_sha256=dataset_sha256,
            authorization_sha256=authorization_sha256,
        )
    except (OSError, ValidationError, ValueError):
        raise ValueError("文档质量判断非法") from None


def build_document_quality_report(
    *,
    evaluation_run_id: StableId,
    dataset_sha256: Sha256,
    authorization_sha256: Sha256,
    predictions: Sequence[VerifiedPrediction],
    annotations: Sequence[VerifiedAnnotation],
    judgments: Sequence[DocumentQualityJudgment] = (),
    visual_quality_report: VerifiedVisualQualityReport | None = None,
) -> DocumentQualityReport:
    """从重验预测重算自动指标，并在人工完整时聚合主观指标。"""

    if len(predictions) != len(annotations):
        raise ValueError("文档质量预测和标注数量不一致")
    for annotation, prediction in zip(annotations, predictions, strict=True):
        if prediction.index.sample_id != annotation.annotation.sample_id:
            raise ValueError("文档质量预测和标注顺序不一致")
    ordered_judgments = tuple(judgments)
    by_id = {item.sample_id: item for item in ordered_judgments}
    expected_ids = {item.annotation.sample_id for item in annotations}
    if len(by_id) != len(ordered_judgments) or not set(by_id).issubset(expected_ids):
        raise ValueError("文档人工判断只能引用当前数据集且不得重复")
    for annotation, prediction in zip(annotations, predictions, strict=True):
        supplied = by_id.get(annotation.annotation.sample_id)
        if supplied is not None:
            verify_document_quality_judgment(
                supplied,
                annotation,
                prediction,
                dataset_sha256=dataset_sha256,
                authorization_sha256=authorization_sha256,
            )
    visual_state = _visual_quality_state(visual_quality_report)
    complete_manual = bool(annotations) and len(ordered_judgments) == len(annotations)
    automatic = _automatic_metrics(predictions)
    if not complete_manual:
        payload = _base_payload(
            evaluation_run_id,
            dataset_sha256,
            authorization_sha256,
            predictions,
            status="NOT_RUN",
            not_run_reason="尚未提供完整文档人工审阅",
            failure_code=None,
            automatic_metrics=None,
            human_metrics=None,
            judgment_sha256=None,
            visual_state=visual_state,
        )
    elif min(automatic.values()) < 1.0:
        payload = _base_payload(
            evaluation_run_id,
            dataset_sha256,
            authorization_sha256,
            predictions,
            status="FAIL",
            not_run_reason=None,
            failure_code="DOCUMENT_AUTOMATIC_CONTRACT_FAILED",
            automatic_metrics=None,
            human_metrics=None,
            judgment_sha256=None,
            visual_state=visual_state,
        )
    else:
        payload = _base_payload(
            evaluation_run_id,
            dataset_sha256,
            authorization_sha256,
            predictions,
            status="SUCCEEDED",
            not_run_reason=None,
            failure_code=None,
            automatic_metrics=automatic,
            human_metrics=_manual_metrics(ordered_judgments),
            judgment_sha256=_judgment_digest(ordered_judgments),
            visual_state=visual_state,
        )
    provisional = DocumentQualityReport.model_construct(**cast(Any, payload))
    payload["report_sha256"] = document_quality_report_sha256(provisional)
    return DocumentQualityReport.model_validate(payload)


def _base_payload(
    evaluation_run_id: StableId,
    dataset_sha256: Sha256,
    authorization_sha256: Sha256,
    predictions: Sequence[VerifiedPrediction],
    *,
    status: Literal["NOT_RUN", "SUCCEEDED", "FAIL"],
    not_run_reason: str | None,
    failure_code: str | None,
    automatic_metrics: Mapping[str, float] | None,
    human_metrics: Mapping[str, DocumentMetricObservation] | None,
    judgment_sha256: Sha256 | None,
    visual_state: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "evaluation_run_id": evaluation_run_id,
        "dataset_sha256": dataset_sha256,
        "authorization_sha256": authorization_sha256,
        "prediction_index_sha256": _prediction_digest(predictions),
        "status": status,
        "not_run_reason": not_run_reason,
        "failure_code": failure_code,
        "automatic_metrics": automatic_metrics,
        "visual_quality_status": visual_state["status"],
        "visual_quality_metrics": visual_state["metrics"],
        "visual_quality_not_run_reason": visual_state["not_run_reason"],
        "visual_quality_failure_code": visual_state["failure_code"],
        "human_metrics": human_metrics,
        "judgment_sha256": judgment_sha256,
        "report_sha256": "0" * 64,
    }


def _visual_quality_state(report: VerifiedVisualQualityReport | None) -> dict[str, object]:
    if report is None:
        return {
            "status": "NOT_RUN",
            "metrics": {"visual_text_accuracy": None, "visual_key_field_recall": None},
            "not_run_reason": "尚未提供代表性视觉质量报告",
            "failure_code": None,
        }
    source = report.report
    return {
        "status": source.status,
        "metrics": {
            "visual_text_accuracy": source.visual_text_accuracy,
            "visual_key_field_recall": source.visual_key_field_recall,
        },
        "not_run_reason": source.not_run_reason if source.status == "NOT_RUN" else None,
        "failure_code": (
            source.failure_code.value
            if source.status == "FAIL" and source.failure_code is not None
            else None
        ),
    }


def _automatic_metrics(predictions: Sequence[VerifiedPrediction]) -> dict[str, float]:
    return {
        "chapter_time_coverage": _chapter_coverage(predictions),
        "claim_evidence_rate": _claim_evidence_rate(predictions),
        "markdown_json_consistency": _markdown_consistency(predictions),
    }


def _chapter_coverage(predictions: Sequence[VerifiedPrediction]) -> float:
    if not predictions or any(prediction.result is None for prediction in predictions):
        return 0.0
    total = sum(
        prediction.result.summary.duration_ms
        for prediction in predictions
        if prediction.result
    )
    covered = sum(
        prediction.result.summary.duration_ms
        for prediction in predictions
        if prediction.result is not None and _has_contiguous_chapters(prediction)
    )
    return 1.0 if total == 0 else covered / total


def _claim_evidence_rate(predictions: Sequence[VerifiedPrediction]) -> float:
    if not predictions or any(prediction.result is None for prediction in predictions):
        return 0.0
    total = 0
    closed = 0
    for prediction in predictions:
        if prediction.result is None:
            continue
        evidence_ids = {item.evidence_id for item in prediction.evidence}
        for chapter in prediction.result.chapters:
            for claim in chapter.claims:
                total += 1
                if claim.evidence_refs and set(claim.evidence_refs).issubset(evidence_ids):
                    closed += 1
    return 1.0 if total == 0 else closed / total


def _markdown_consistency(predictions: Sequence[VerifiedPrediction]) -> float:
    return (
        1.0
        if predictions and all(_markdown_matches(prediction) for prediction in predictions)
        else 0.0
    )


def _has_contiguous_chapters(prediction: VerifiedPrediction) -> bool:
    result = prediction.result
    if result is None or not result.chapters:
        return False
    return (
        result.chapters[0].start_ms == 0
        and result.chapters[-1].end_ms == result.summary.duration_ms
        and all(
            left.end_ms == right.start_ms
            for left, right in zip(result.chapters, result.chapters[1:], strict=False)
        )
    )


def _markdown_matches(prediction: VerifiedPrediction) -> bool:
    if prediction.result is None or prediction.index.document_relative_path is None:
        return False
    expected = render_markdown(prediction.result, prediction.evidence)
    path = prediction.eval_root / prediction.index.document_relative_path
    try:
        content = path.read_bytes()
    except OSError:
        return False
    return content == expected.content and (
        hashlib.sha256(content).hexdigest() == prediction.index.document_sha256
    )


_RELEVANCE_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}


def _manual_metrics(
    judgments: Sequence[DocumentQualityJudgment],
) -> dict[str, DocumentMetricObservation]:
    chapters = [chapter for judgment in judgments for chapter in judgment.chapters]
    verdicts = [
        verdict
        for chapter in chapters
        for _, verdict in chapter.visual_supplement_judgments
    ]
    return {
        "title_relevance": DocumentMetricObservation(
            value=sum(_RELEVANCE_SCORE[item.title_relevance] for item in chapters) / len(chapters)
        ),
        "summary_relevance": DocumentMetricObservation(
            value=sum(_RELEVANCE_SCORE[item.summary_relevance] for item in chapters) / len(chapters)
        ),
        "visual_supplement_rate": _verdict_rate(verdicts, "SUPPLEMENTS"),
        "visual_duplicate_rate": _verdict_rate(verdicts, "DUPLICATES"),
    }


def _verdict_rate(
    verdicts: Sequence[VisualSupplementVerdict],
    expected: VisualSupplementVerdict,
) -> DocumentMetricObservation:
    if not verdicts:
        return DocumentMetricObservation(not_run_reason="当前文档没有可审阅视觉观察")
    return DocumentMetricObservation(
        value=sum(item == expected for item in verdicts) / len(verdicts)
    )


def _judgment_digest(judgments: Sequence[DocumentQualityJudgment]) -> Sha256:
    encoded = json.dumps(
        [judgment.model_dump(mode="json") for judgment in judgments],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prediction_digest(predictions: Sequence[VerifiedPrediction]) -> Sha256:
    encoded = json.dumps(
        [prediction.index_sha256 for prediction in predictions],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def document_quality_report_sha256(report: DocumentQualityReport) -> Sha256:
    payload = report.model_dump(mode="json", exclude={"report_sha256"})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ChapterDocumentJudgment",
    "DocumentMetricObservation",
    "DocumentQualityJudgment",
    "DocumentQualityReport",
    "build_document_quality_report",
    "document_quality_report_sha256",
    "load_document_quality_judgment",
    "verify_document_quality_judgment",
]
