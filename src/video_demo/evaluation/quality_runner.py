from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.evidence import (
    EvidenceItem,
    OcrEvidence,
    SceneBoundary,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import VideoUnderstandingResult, validate_evidence_references
from video_demo.domain.result_artifact import TranscriptSource
from video_demo.errors import VideoDemoError
from video_demo.evaluation.annotations import (
    EvaluationAnnotation,
    SemanticJudgment,
    ValidatedEvaluationPackage,
    VerifiedAnnotation,
    reverify_evaluation_package,
)
from video_demo.evaluation.dataset import EvaluationSample, ValidationLanguage
from video_demo.evaluation.metrics import (
    EditCounts,
    MatchCounts,
    boundary_match_counts,
    character_edit_counts,
    match_counts_f1,
    nfkc_character_edit_counts,
    word_edit_counts,
)
from video_demo.evaluation.predictions import VerifiedPrediction, reverify_verified_prediction
from video_demo.evaluation.report import (
    BoundQualityReport,
    MetricObservation,
    build_quality_report,
)
from video_demo.evaluation.thresholds import (
    QUALITY_THRESHOLDS,
    SCENE_BOUNDARY_TOLERANCE_MS,
    SEMANTIC_BOUNDARY_TOLERANCE_MS,
)

_CER_LANGUAGES = frozenset({"zh", "ja", "ko"})
_LANGUAGE_METRICS: Mapping[ValidationLanguage, str] = {
    "zh": "zh_cer",
    "ja": "ja_cer",
    "ko": "ko_cer",
    "en": "en_wer",
    "es": "es_wer",
}
_EVIDENCE_ADAPTER = TypeAdapter(tuple[EvidenceItem, ...])
class SampleQualityDetail(FrozenModel):
    sample_id: StableId
    language: ValidationLanguage
    prediction_status: str
    transcript_source: TranscriptSource | None
    metric_inputs: dict[str, int | float]
    failure_code: str | None


class HintPairEffect(FrozenModel):
    pair_id: StableId
    language: ValidationLanguage
    none_sample_id: StableId
    correct_sample_id: StableId
    metric_name: Literal["CER", "WER"]
    term_count: int = Field(gt=0)
    none_term_recall: float = Field(ge=0, le=1)
    correct_term_recall: float = Field(ge=0, le=1)
    term_recall_delta: float = Field(ge=-1, le=1)
    none_text_error_rate: float = Field(ge=0)
    correct_text_error_rate: float = Field(ge=0)
    text_error_rate_delta: float

    @model_validator(mode="after")
    def validate_deltas(self) -> HintPairEffect:
        if abs(
            self.term_recall_delta
            - (self.correct_term_recall - self.none_term_recall)
        ) > 1e-12:
            raise ValueError("术语召回差值与两端结果不一致")
        if abs(
            self.text_error_rate_delta
            - (self.correct_text_error_rate - self.none_text_error_rate)
        ) > 1e-12:
            raise ValueError("文本错误率差值与两端结果不一致")
        return self


HintPairExclusion = Literal[
    "PREDICTION_NOT_SUCCESSFUL",
    "TERMS_EMPTY",
    "TRANSCRIPT_SOURCE_NOT_ASR",
]


class HintEffectReport(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation_run_id: StableId
    status: Literal["RUN", "NOT_RUN"]
    dataset_sha256: Sha256
    authorization_sha256: Sha256
    prediction_index_sha256: Sha256
    candidate_pair_count: int = Field(ge=0)
    eligible_pair_count: int = Field(ge=0)
    excluded_pair_counts: dict[HintPairExclusion, int]
    pairs: tuple[HintPairEffect, ...]
    not_run_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("excluded_pair_counts")
    @classmethod
    def reject_empty_or_nonpositive_exclusion_counts(
        cls,
        value: dict[HintPairExclusion, int],
    ) -> dict[HintPairExclusion, int]:
        if any(count <= 0 for count in value.values()):
            raise ValueError("提示配对排除计数必须为正整数")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_report_contract(self) -> HintEffectReport:
        if self.eligible_pair_count != len(self.pairs):
            raise ValueError("合格提示配对计数与明细不一致")
        if self.candidate_pair_count != self.eligible_pair_count + sum(
            self.excluded_pair_counts.values()
        ):
            raise ValueError("候选提示配对计数未闭合")
        if tuple(pair.pair_id for pair in self.pairs) != tuple(
            sorted(pair.pair_id for pair in self.pairs)
        ):
            raise ValueError("提示配对明细必须按配对 ID 排序")
        if self.status == "RUN":
            if not self.pairs or self.not_run_reason is not None:
                raise ValueError("已运行提示报告必须包含配对且不得有未运行原因")
        elif self.pairs or self.not_run_reason is None:
            raise ValueError("未运行提示报告不得包含配对且必须有原因")
        return self


class QualityScoreArtifacts(FrozenModel):
    report: BoundQualityReport
    sample_details: tuple[SampleQualityDetail, ...]
    hint_effect_report: HintEffectReport


def score_quality(
    package: ValidatedEvaluationPackage,
    predictions: tuple[VerifiedPrediction, ...],
    judgments: tuple[SemanticJudgment, ...],
    *,
    evaluation_run_id: str,
) -> QualityScoreArtifacts:
    """从受验签来源确定性重建非运行时质量指标及逐样本明细。"""

    verified_package = reverify_evaluation_package(package)
    ordered_predictions = _ordered_predictions(
        verified_package, predictions, evaluation_run_id
    )
    ordered_predictions = tuple(
        reverify_verified_prediction(prediction, sample=sample)
        for sample, prediction in zip(
            verified_package.dataset.samples, ordered_predictions, strict=True
        )
    )
    ordered_annotations = _ordered_annotations(verified_package)
    ordered_judgments = _ordered_judgments(
        ordered_annotations, ordered_predictions, judgments
    )
    accumulators = _Accumulators()
    details = tuple(
        _score_sample(annotation, prediction, accumulators)
        for annotation, prediction in zip(
            ordered_annotations, ordered_predictions, strict=True
        )
    )
    observations = accumulators.observations()
    prediction_index_sha256 = _canonical_digest(
        [prediction.index_sha256 for prediction in ordered_predictions]
    )
    hint_effect_report = _build_hint_effect_report(
        samples=verified_package.dataset.samples,
        annotations=ordered_annotations,
        predictions=ordered_predictions,
        evaluation_run_id=evaluation_run_id,
        dataset_sha256=verified_package.dataset_sha256,
        authorization_sha256=verified_package.authorization_sha256,
        prediction_index_sha256=prediction_index_sha256,
    )
    failure_code: str | None = None
    if not judgments:
        reason = "尚未提供人工语义审阅"
        for name in ("fact_support_rate", "key_fact_recall", "fabricated_name_count"):
            observations[name] = MetricObservation(value=None, not_run_reason=reason)
    elif len(judgments) != len(ordered_predictions):
        reason = "人工语义审阅未完整覆盖数据集"
        for name in ("fact_support_rate", "key_fact_recall", "fabricated_name_count"):
            observations[name] = MetricObservation(value=None, not_run_reason=reason)
        failure_code = "SEMANTIC_REVIEW_INCOMPLETE"
    else:
        assert ordered_judgments is not None
        _add_semantic_observations(
            observations, ordered_annotations, ordered_predictions, ordered_judgments
        )

    base_report = build_quality_report(
        observations,
        QUALITY_THRESHOLDS,
        resources_not_run_reason="本切片不执行运行时资源测量",
        failure_code=failure_code,
    )
    report = BoundQualityReport(
        **base_report.model_dump(mode="python"),
        schema_version="1.0.0",
        evaluation_run_id=evaluation_run_id,
        dataset_sha256=verified_package.dataset_sha256,
        authorization_sha256=verified_package.authorization_sha256,
        prediction_index_sha256=prediction_index_sha256,
        judgment_index_sha256=_canonical_digest(
            []
            if ordered_judgments is None
            else [judgment.model_dump(mode="json") for judgment in ordered_judgments]
        ),
        sample_details_sha256=_canonical_digest(
            [detail.model_dump(mode="json") for detail in details]
        ),
        durability_report_sha256=None,
    )
    return QualityScoreArtifacts(
        report=report,
        sample_details=details,
        hint_effect_report=hint_effect_report,
    )


class _Accumulators:
    def __init__(self) -> None:
        self.text_counts: dict[str, EditCounts] = {
            name: EditCounts(0, 0) for name in _LANGUAGE_METRICS.values()
        }
        self.ocr_errors = 0
        self.ocr_units = 0
        self.scene_counts = MatchCounts(0, 0, 0)
        self.semantic_boundary_counts = MatchCounts(0, 0, 0)
        self.unknown_evidence_count = 0
        self.valid_results = 0
        self.attempted_results = 0

    def observations(self) -> dict[str, MetricObservation]:
        observations: dict[str, MetricObservation] = {}
        for name, counts in self.text_counts.items():
            observations[name] = _count_observation(counts)
        observations["ocr_accuracy"] = _accuracy_observation(
            self.ocr_errors, self.ocr_units, "没有 OCR 参考字符"
        )
        observations["scene_f1"] = MetricObservation(
            value=match_counts_f1(self.scene_counts)
        )
        observations["semantic_boundary_f1"] = MetricObservation(
            value=match_counts_f1(self.semantic_boundary_counts)
        )
        observations["unknown_evidence_count"] = MetricObservation(
            value=float(self.unknown_evidence_count)
        )
        observations["schema_time_valid_rate"] = MetricObservation(
            value=self.valid_results / self.attempted_results
        )
        observations["rtf"] = MetricObservation(
            value=None, not_run_reason="本切片不执行运行时测量"
        )
        return observations


def _score_sample(
    verified_annotation: VerifiedAnnotation,
    prediction: VerifiedPrediction,
    accumulators: _Accumulators,
) -> SampleQualityDetail:
    annotation = verified_annotation.annotation
    accumulators.attempted_results += 1
    transcript_source = prediction.index.transcript_source
    hypothesis_text = _transcript_text(prediction)
    counts = (
        character_edit_counts(hypothesis_text, annotation.reference_text)
        if annotation.language in _CER_LANGUAGES
        else word_edit_counts(hypothesis_text, annotation.reference_text)
    )
    metric_name = _LANGUAGE_METRICS[annotation.language]
    previous = accumulators.text_counts[metric_name]
    accumulators.text_counts[metric_name] = EditCounts(
        previous.errors + counts.errors,
        previous.reference_units + counts.reference_units,
    )
    reference_ocr = "".join(
        line
        for frame in sorted(annotation.ocr_frames, key=lambda item: item.timestamp_ms)
        for line in frame.text_lines
    )
    predicted_ocr = "".join(
        line.text
        for item in sorted(
            (item for item in prediction.evidence if isinstance(item, OcrEvidence)),
            key=lambda item: item.timestamp_ms,
        )
        for line in item.lines
    )
    ocr_counts = nfkc_character_edit_counts(predicted_ocr, reference_ocr)
    accumulators.ocr_errors += ocr_counts.errors
    accumulators.ocr_units += ocr_counts.reference_units

    scene_counts = boundary_match_counts(
        reference_ms=annotation.scene_boundaries_ms,
        hypothesis_ms=tuple(
            item.start_ms
            for item in prediction.evidence
            if isinstance(item, SceneBoundary) and item.start_ms > 0
        ),
        tolerance_ms=SCENE_BOUNDARY_TOLERANCE_MS,
    )
    semantic_counts = boundary_match_counts(
        reference_ms=annotation.semantic_boundaries_ms,
        hypothesis_ms=tuple(
            segment.start_ms
            for segment in (() if prediction.result is None else prediction.result.segments)
            if segment.start_ms > 0
        ),
        tolerance_ms=SEMANTIC_BOUNDARY_TOLERANCE_MS,
    )
    accumulators.scene_counts = _sum_match_counts(
        accumulators.scene_counts, scene_counts
    )
    accumulators.semantic_boundary_counts = _sum_match_counts(
        accumulators.semantic_boundary_counts, semantic_counts
    )
    scene_score = match_counts_f1(scene_counts)
    semantic_score = match_counts_f1(semantic_counts)

    unknown_count, is_valid = _schema_time_check(annotation, prediction)
    accumulators.unknown_evidence_count += unknown_count
    accumulators.valid_results += int(is_valid)
    diagnostics: dict[str, int | float] = {
        "text_errors": counts.errors,
        "text_reference_units": counts.reference_units,
        "unknown_evidence_count": unknown_count,
        "schema_time_valid": float(is_valid),
        "ocr_errors": ocr_counts.errors,
        "ocr_reference_units": ocr_counts.reference_units,
        "scene_f1": scene_score,
        "semantic_boundary_f1": semantic_score,
    }
    return SampleQualityDetail(
        sample_id=annotation.sample_id,
        language=annotation.language,
        prediction_status=prediction.index.terminal_status,
        transcript_source=transcript_source,
        metric_inputs=diagnostics,
        failure_code=prediction.index.failure_code,
    )


def _transcript_text(
    prediction: VerifiedPrediction,
) -> str:
    transcript_type = (
        SubtitleCue
        if prediction.index.transcript_source == "SUBTITLE"
        else SpeechSegment
    )
    items = sorted(
        (item for item in prediction.evidence if isinstance(item, transcript_type)),
        key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
    )
    return " ".join(item.text for item in items)


def _build_hint_effect_report(
    *,
    samples: tuple[EvaluationSample, ...],
    annotations: tuple[VerifiedAnnotation, ...],
    predictions: tuple[VerifiedPrediction, ...],
    evaluation_run_id: str,
    dataset_sha256: str,
    authorization_sha256: str,
    prediction_index_sha256: str,
) -> HintEffectReport:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    annotation_by_id = {
        item.annotation.sample_id: item.annotation for item in annotations
    }
    prediction_by_id = {
        prediction.index.sample_id: prediction for prediction in predictions
    }
    pair_ids = sorted(
        {sample.pair_id for sample in samples if sample.pair_id is not None}
    )
    pairs: list[HintPairEffect] = []
    exclusions: dict[HintPairExclusion, int] = defaultdict(int)
    for pair_id in pair_ids:
        pair_samples = tuple(
            sample for sample in samples if sample.pair_id == pair_id
        )
        by_variant = {sample.hint_variant: sample for sample in pair_samples}
        none_sample = by_variant["NONE"]
        correct_sample = by_variant["CORRECT"]
        none_prediction = prediction_by_id[none_sample.sample_id]
        correct_prediction = prediction_by_id[correct_sample.sample_id]
        if any(
            prediction.index.terminal_status
            not in {"SUCCEEDED", "PARTIAL_SUCCEEDED"}
            for prediction in (none_prediction, correct_prediction)
        ):
            exclusions["PREDICTION_NOT_SUCCESSFUL"] += 1
            continue
        if any(
            prediction.index.transcript_source != "ASR"
            for prediction in (none_prediction, correct_prediction)
        ):
            exclusions["TRANSCRIPT_SOURCE_NOT_ASR"] += 1
            continue
        annotation = annotation_by_id[none_sample.sample_id]
        if not annotation.terms:
            exclusions["TERMS_EMPTY"] += 1
            continue
        correct_annotation = annotation_by_id[correct_sample.sample_id]
        if correct_annotation.terms != annotation.terms:
            raise ValueError("提示配对的明确术语不一致")
        pairs.append(
            _score_hint_pair(
                pair_id=pair_id,
                none_sample=sample_by_id[none_sample.sample_id],
                correct_sample=sample_by_id[correct_sample.sample_id],
                annotation=annotation,
                none_prediction=none_prediction,
                correct_prediction=correct_prediction,
            )
        )
    not_run_reason: str | None = None
    status: Literal["RUN", "NOT_RUN"] = "RUN"
    if not pairs:
        status = "NOT_RUN"
        not_run_reason = (
            "数据集没有 NONE/CORRECT 提示效果配对"
            if not pair_ids
            else "没有同时满足成功、ASR 来源和明确术语的提示效果配对"
        )
    return HintEffectReport(
        evaluation_run_id=evaluation_run_id,
        status=status,
        dataset_sha256=dataset_sha256,
        authorization_sha256=authorization_sha256,
        prediction_index_sha256=prediction_index_sha256,
        candidate_pair_count=len(pair_ids),
        eligible_pair_count=len(pairs),
        excluded_pair_counts=dict(exclusions),
        pairs=tuple(pairs),
        not_run_reason=not_run_reason,
    )


def _score_hint_pair(
    *,
    pair_id: str,
    none_sample: EvaluationSample,
    correct_sample: EvaluationSample,
    annotation: EvaluationAnnotation,
    none_prediction: VerifiedPrediction,
    correct_prediction: VerifiedPrediction,
) -> HintPairEffect:
    none_text = _prediction_transcript_text(none_prediction)
    correct_text = _prediction_transcript_text(correct_prediction)
    none_counts = _text_edit_counts(
        none_text,
        annotation.reference_text,
        annotation.language,
    )
    correct_counts = _text_edit_counts(
        correct_text,
        annotation.reference_text,
        annotation.language,
    )
    none_recall = _exact_term_recall(none_text, annotation.terms)
    correct_recall = _exact_term_recall(correct_text, annotation.terms)
    none_error_rate = none_counts.errors / none_counts.reference_units
    correct_error_rate = correct_counts.errors / correct_counts.reference_units
    return HintPairEffect(
        pair_id=pair_id,
        language=annotation.language,
        none_sample_id=none_sample.sample_id,
        correct_sample_id=correct_sample.sample_id,
        metric_name="CER" if annotation.language in _CER_LANGUAGES else "WER",
        term_count=len(annotation.terms),
        none_term_recall=none_recall,
        correct_term_recall=correct_recall,
        term_recall_delta=correct_recall - none_recall,
        none_text_error_rate=none_error_rate,
        correct_text_error_rate=correct_error_rate,
        text_error_rate_delta=correct_error_rate - none_error_rate,
    )


def _prediction_transcript_text(prediction: VerifiedPrediction) -> str:
    return _transcript_text(prediction)


def _text_edit_counts(
    hypothesis: str,
    reference: str,
    language: ValidationLanguage,
) -> EditCounts:
    return (
        character_edit_counts(hypothesis, reference)
        if language in _CER_LANGUAGES
        else word_edit_counts(hypothesis, reference)
    )


def _exact_term_recall(hypothesis: str, terms: tuple[str, ...]) -> float:
    normalized_hypothesis = _normalize_hint_text(hypothesis)
    matches = sum(
        _normalize_hint_text(term) in normalized_hypothesis for term in terms
    )
    return matches / len(terms)


def _normalize_hint_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _ordered_predictions(
    package: ValidatedEvaluationPackage,
    predictions: tuple[VerifiedPrediction, ...],
    evaluation_run_id: str,
) -> tuple[VerifiedPrediction, ...]:
    expected_ids = tuple(sample.sample_id for sample in package.dataset.samples)
    supplied_ids = tuple(prediction.index.sample_id for prediction in predictions)
    if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != set(expected_ids):
        raise ValueError("预测必须恰好覆盖数据集且不得重复或包含外来样本")
    if any(
        prediction.index.evaluation_run_id != evaluation_run_id
        for prediction in predictions
    ):
        raise ValueError("预测必须全部属于指定评测运行")
    by_id = {prediction.index.sample_id: prediction for prediction in predictions}
    if any(
        by_id[sample.sample_id].index.media_sha256 != sample.media_sha256
        for sample in package.dataset.samples
    ):
        raise ValueError("预测媒体摘要与数据集样本不匹配")
    return tuple(by_id[sample_id] for sample_id in expected_ids)


def _ordered_annotations(
    package: ValidatedEvaluationPackage,
) -> tuple[VerifiedAnnotation, ...]:
    expected_ids = tuple(sample.sample_id for sample in package.dataset.samples)
    by_id = {item.annotation.sample_id: item for item in package.annotations}
    if len(by_id) != len(package.annotations) or set(by_id) != set(expected_ids):
        raise ValueError("受验签标注必须恰好覆盖数据集")
    return tuple(by_id[sample_id] for sample_id in expected_ids)


def _ordered_judgments(
    annotations: tuple[VerifiedAnnotation, ...],
    predictions: tuple[VerifiedPrediction, ...],
    judgments: tuple[SemanticJudgment, ...],
) -> tuple[SemanticJudgment, ...] | None:
    if not judgments:
        return None
    expected_ids = tuple(item.annotation.sample_id for item in annotations)
    supplied_ids = tuple(judgment.sample_id for judgment in judgments)
    if len(supplied_ids) != len(set(supplied_ids)) or not set(supplied_ids).issubset(
        expected_ids
    ):
        raise ValueError("审阅不得重复或包含外来样本")
    annotations_by_id = {item.annotation.sample_id: item for item in annotations}
    predictions_by_id = {item.index.sample_id: item for item in predictions}
    by_id = {judgment.sample_id: judgment for judgment in judgments}
    for sample_id, judgment in by_id.items():
        annotation = annotations_by_id[sample_id]
        prediction = predictions_by_id[sample_id]
        if (
            judgment.annotation_sha256 != annotation.sha256
            or judgment.prediction_sha256 != prediction.index_sha256
        ):
            raise ValueError("审阅与当前标注或预测摘要不匹配")
        if {item.claim_id for item in judgment.claim_judgments} != {
            claim.claim_id for claim in prediction.claims
        }:
            raise ValueError("审阅必须恰好覆盖当前预测 claims")
        if not set(judgment.matched_key_fact_ids).issubset(
            annotation.annotation.key_fact_ids
        ):
            raise ValueError("审阅引用了未知关键事实")
    return tuple(by_id[sample_id] for sample_id in expected_ids if sample_id in by_id)


def _add_semantic_observations(
    observations: dict[str, MetricObservation],
    annotations: tuple[VerifiedAnnotation, ...],
    predictions: tuple[VerifiedPrediction, ...],
    judgments: tuple[SemanticJudgment, ...],
) -> None:
    total_claims = sum(len(prediction.claims) for prediction in predictions)
    supported_claims = sum(
        item.verdict == "SUPPORTED"
        for judgment in judgments
        for item in judgment.claim_judgments
    )
    total_key_facts = sum(len(item.annotation.key_fact_ids) for item in annotations)
    matched_key_facts = sum(len(judgment.matched_key_fact_ids) for judgment in judgments)
    observations["fact_support_rate"] = (
        MetricObservation(value=supported_claims / total_claims)
        if total_claims
        else MetricObservation(value=None, not_run_reason="完整审阅中没有预测 claim")
    )
    observations["key_fact_recall"] = MetricObservation(
        value=matched_key_facts / total_key_facts
    )
    observations["fabricated_name_count"] = MetricObservation(
        value=float(sum(len(judgment.fabricated_names) for judgment in judgments))
    )


def _schema_time_check(
    annotation: EvaluationAnnotation, prediction: VerifiedPrediction
) -> tuple[int, bool]:
    if prediction.result is None:
        return 0, False
    evidence_ids = [item.evidence_id for item in prediction.evidence]
    evidence_id_set = set(evidence_ids)
    references = [
        reference
        for segment in prediction.result.segments
        for reference in segment.evidence_refs
    ]
    unknown_count = sum(reference not in evidence_id_set for reference in references)
    try:
        if prediction.index.terminal_status not in ("SUCCEEDED", "PARTIAL_SUCCEEDED"):
            raise ValueError("失败预测不得携带成功结果")
        checked_result = VideoUnderstandingResult.model_validate(
            prediction.result.model_dump(mode="python", exclude_computed_fields=True)
        )
        checked_evidence = _EVIDENCE_ADAPTER.validate_python(
            [
                item.model_dump(mode="python", exclude_computed_fields=True)
                for item in prediction.evidence
            ]
        )
        if len(evidence_ids) != len(evidence_id_set):
            raise ValueError("证据 ID 重复")
        if (
            checked_result.asset_sha256 != annotation.media_sha256
            or checked_result.run_id != prediction.index.run_id
            or checked_result.summary.duration_ms != annotation.duration_ms
            or any(item.end_ms > annotation.duration_ms for item in checked_evidence)
        ):
            raise ValueError("结果绑定或时间越界")
        validate_evidence_references(checked_result, checked_evidence)
    except (ValueError, RuntimeError, VideoDemoError):
        return unknown_count, False
    return unknown_count, True


def _count_observation(counts: EditCounts) -> MetricObservation:
    return _ratio_observation(
        float(counts.errors), counts.reference_units, "该语言没有参考文本单元"
    )


def _accuracy_observation(
    errors: int, units: int, not_run_reason: str
) -> MetricObservation:
    if not units:
        return MetricObservation(value=None, not_run_reason=not_run_reason)
    return MetricObservation(value=1 - errors / units)


def _ratio_observation(
    numerator: float, denominator: int, not_run_reason: str
) -> MetricObservation:
    if not denominator:
        return MetricObservation(value=None, not_run_reason=not_run_reason)
    return MetricObservation(value=numerator / denominator)


def _canonical_digest(value: Sequence[object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sum_match_counts(left: MatchCounts, right: MatchCounts) -> MatchCounts:
    return MatchCounts(
        left.matches + right.matches,
        left.predicted_units + right.predicted_units,
        left.reference_units + right.reference_units,
    )
