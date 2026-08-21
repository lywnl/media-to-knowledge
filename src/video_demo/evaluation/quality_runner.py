from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise

from pydantic import TypeAdapter

from video_demo.domain.base import FrozenModel, StableId
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    EvidenceItem,
    OcrEvidence,
    SceneBoundary,
    SpeakerTurn,
)
from video_demo.domain.result import VideoUnderstandingResult, validate_evidence_references
from video_demo.errors import VideoDemoError
from video_demo.evaluation.annotations import (
    EvaluationAnnotation,
    SemanticJudgment,
    ValidatedEvaluationPackage,
    VerifiedAnnotation,
    reverify_evaluation_package,
)
from video_demo.evaluation.dataset import ValidationLanguage
from video_demo.evaluation.metrics import (
    EditCounts,
    MatchCounts,
    aligned_word_time_errors_ms,
    boundary_match_counts,
    character_edit_counts,
    diarization_counts_by_overlap,
    event_match_counts,
    match_counts_f1,
    nfkc_character_edit_counts,
    percentile_90,
    word_edit_counts,
)
from video_demo.evaluation.predictions import VerifiedPrediction, reverify_verified_prediction
from video_demo.evaluation.report import (
    BoundQualityReport,
    MetricObservation,
    build_quality_report,
)
from video_demo.evaluation.thresholds import (
    AUDIO_EVENT_TOLERANCE_MS,
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
    metric_inputs: dict[str, int | float]
    failure_code: str | None


class QualityScoreArtifacts(FrozenModel):
    report: BoundQualityReport
    sample_details: tuple[SampleQualityDetail, ...]


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
        prediction_index_sha256=_canonical_digest(
            [prediction.index_sha256 for prediction in ordered_predictions]
        ),
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
    return QualityScoreArtifacts(report=report, sample_details=details)


class _Accumulators:
    def __init__(self) -> None:
        self.text_counts: dict[str, EditCounts] = {
            name: EditCounts(0, 0) for name in _LANGUAGE_METRICS.values()
        }
        self.word_times: list[int] = []
        self.der_errors = [0, 0]
        self.der_units = [0, 0]
        self.ocr_errors = 0
        self.ocr_units = 0
        self.audio_counts: dict[str, MatchCounts] = {}
        self.scene_counts = MatchCounts(0, 0, 0)
        self.semantic_boundary_counts = MatchCounts(0, 0, 0)
        self.unknown_evidence_count = 0
        self.valid_results = 0
        self.attempted_results = 0

    def observations(self) -> dict[str, MetricObservation]:
        observations: dict[str, MetricObservation] = {}
        for name, counts in self.text_counts.items():
            observations[name] = _count_observation(counts)
        observations["word_time_p90_ms"] = (
            MetricObservation(value=percentile_90(self.word_times))
            if self.word_times
            else MetricObservation(value=None, not_run_reason="没有文字匹配的词时间单元")
        )
        observations["der_non_overlap"] = _ratio_observation(
            self.der_errors[0], self.der_units[0], "没有非重叠参考说话人时长"
        )
        observations["der_overlap"] = _ratio_observation(
            self.der_errors[1], self.der_units[1], "没有重叠参考说话人时长"
        )
        observations["ocr_accuracy"] = _accuracy_observation(
            self.ocr_errors, self.ocr_units, "没有 OCR 参考字符"
        )
        observations["audio_event_macro_f1"] = MetricObservation(
            value=sum(match_counts_f1(counts) for counts in self.audio_counts.values())
            / len(self.audio_counts)
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
    aligned_words = tuple(
        item for item in prediction.evidence if isinstance(item, AlignedWord)
    )
    hypothesis_text = " ".join(item.text for item in aligned_words)
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
    word_time_errors = aligned_word_time_errors_ms(
        reference=tuple(
            (word.text, word.start_ms, word.end_ms) for word in annotation.words
        ),
        hypothesis=tuple(
            (word.text, word.start_ms, word.end_ms) for word in aligned_words
        ),
    )
    accumulators.word_times.extend(word_time_errors)

    speaker_turns = tuple(
        item for item in prediction.evidence if isinstance(item, SpeakerTurn)
    )
    reference_turns = tuple(
        (turn.start_ms, turn.end_ms, turn.speaker_id) for turn in annotation.speaker_turns
    )
    hypothesis_turns = tuple(
        (turn.start_ms, turn.end_ms, speaker)
        for turn in speaker_turns
        for speaker in {turn.speaker, *turn.overlap_speakers}
    )
    der_counts = diarization_counts_by_overlap(
        reference=reference_turns, hypothesis=hypothesis_turns
    )
    for index, counts_for_partition in enumerate(der_counts):
        accumulators.der_errors[index] += counts_for_partition.error_speaker_ms
        accumulators.der_units[index] += counts_for_partition.reference_speaker_ms

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

    audio_reference = _event_intervals(annotation)
    audio_hypothesis = _predicted_event_intervals(prediction)
    audio_counts = event_match_counts(
        reference=audio_reference,
        hypothesis=audio_hypothesis,
        tolerance_ms=AUDIO_EVENT_TOLERANCE_MS,
    )
    for label, counts_for_label in audio_counts.items():
        accumulators.audio_counts[label] = _sum_match_counts(
            accumulators.audio_counts.get(label, MatchCounts(0, 0, 0)),
            counts_for_label,
        )
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
    audio_score = (
        sum(match_counts_f1(counts_for_label) for counts_for_label in audio_counts.values())
        / len(audio_counts)
    )
    scene_score = match_counts_f1(scene_counts)
    semantic_score = match_counts_f1(semantic_counts)

    unknown_count, is_valid = _schema_time_check(annotation, prediction)
    accumulators.unknown_evidence_count += unknown_count
    accumulators.valid_results += int(is_valid)
    reference_speaker_count = len({turn.speaker_id for turn in annotation.speaker_turns})
    predicted_speaker_count = len(
        {
            speaker
            for turn in speaker_turns
            for speaker in (turn.speaker, *turn.overlap_speakers)
            if speaker != "SPEAKER_UNKNOWN"
        }
    )
    diagnostics = {
        "text_errors": counts.errors,
        "text_reference_units": counts.reference_units,
        "word_time_match_count": len(word_time_errors),
        "speaker_count_accuracy": float(reference_speaker_count == predicted_speaker_count),
        "unknown_evidence_count": unknown_count,
        "schema_time_valid": float(is_valid),
        "ocr_errors": ocr_counts.errors,
        "ocr_reference_units": ocr_counts.reference_units,
        "audio_event_macro_f1": audio_score,
        "scene_f1": scene_score,
        "semantic_boundary_f1": semantic_score,
    }
    return SampleQualityDetail(
        sample_id=annotation.sample_id,
        language=annotation.language,
        prediction_status=prediction.index.terminal_status,
        metric_inputs=diagnostics,
        failure_code=prediction.index.failure_code,
    )


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


def _event_intervals(
    annotation: EvaluationAnnotation,
) -> dict[str, tuple[tuple[int, int], ...]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in annotation.audio_events:
        grouped[item.normalized_event].append((item.start_ms, item.end_ms))
    return {label: tuple(intervals) for label, intervals in grouped.items()}


def _predicted_event_intervals(
    prediction: VerifiedPrediction,
) -> dict[str, tuple[tuple[int, int], ...]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in prediction.evidence:
        if isinstance(item, AudioEvent):
            grouped[item.normalized_event].append((item.start_ms, item.end_ms))
    return {label: tuple(intervals) for label, intervals in grouped.items()}


def _der_reference_units(annotation: EvaluationAnnotation) -> tuple[int, int]:
    boundaries = sorted(
        {point for turn in annotation.speaker_turns for point in (turn.start_ms, turn.end_ms)}
    )
    units = [0, 0]
    for start, end in pairwise(boundaries):
        active = sum(
            turn.start_ms <= start < turn.end_ms for turn in annotation.speaker_turns
        )
        if active:
            units[int(active >= 2)] += (end - start) * active
    return units[0], units[1]


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
