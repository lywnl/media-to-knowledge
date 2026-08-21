from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    BoundingBox,
    OcrEvidence,
    OcrLine,
    SceneBoundary,
    SpeakerTurn,
    SubtitleCue,
)
from video_demo.domain.result import VideoSegment, VideoSummary, VideoUnderstandingResult
from video_demo.domain.run import ModelIdentity
from video_demo.errors import VideoDemoError
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    AuthorizationRecord,
    ClaimJudgment,
    EvaluationAnnotation,
    ReferenceAudioEvent,
    ReferenceOcrFrame,
    ReferenceSpeakerTurn,
    ReferenceWord,
    SemanticJudgment,
    SupportedFact,
    ValidatedEvaluationPackage,
    VerifiedAnnotation,
    load_evaluation_package,
    pair_reference_sha256,
)
from video_demo.evaluation.dataset import EvaluationDataset, EvaluationSample
from video_demo.evaluation.predictions import (
    EvaluationPrediction,
    PredictionClaim,
    PredictionRunSnapshot,
    VerifiedPrediction,
    load_verified_prediction,
)
from video_demo.evaluation.report import GateStatus

_NOW = datetime(2026, 8, 18, tzinfo=UTC)
_MEDIA_BYTES = b"fixture-media"
_SHA = hashlib.sha256(_MEDIA_BYTES).hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _segment(sample_id: str, index: int, start: int, end: int, evidence_ref: str) -> VideoSegment:
    text = f"segment {sample_id} {index}"
    return VideoSegment(
        segment_id=f"seg_{sample_id}_{index}",
        start_ms=start,
        end_ms=end,
        title="片段",
        summary_zh="片段事实",
        speakers=("SPEAKER_01",),
        languages=("zh",),
        evidence_refs=(evidence_ref,),
        retrieval_text=text,
        retrieval_hash=_hash(text),
    )


def _successful_prediction(
    sample_id: str,
    language: str,
    predicted_words: tuple[str, ...],
    *,
    run_id: str = "eval_001",
) -> VerifiedPrediction:
    words = tuple(
        AlignedWord(
            evidence_id=f"word_{sample_id}_{index}",
            start_ms=index * 500 + 100,
            end_ms=index * 500 + 450,
            text=text,
            language=language,
            probability=0.9,
            speaker="SPEAKER_01",
        )
        for index, text in enumerate(predicted_words)
    )
    first_ref = words[0].evidence_id
    ocr = OcrEvidence(
        evidence_id=f"ocr_{sample_id}",
        start_ms=7_000,
        end_ms=9_000,
        keyframe_id=f"key_{sample_id}",
        timestamp_ms=8_000,
        language=language,
        lines=(
            OcrLine(
                text="A",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.9,
            ),
        ),
        provider_request_id=f"request-{sample_id}",
    )
    segments = (
        _segment(sample_id, 1, 0, 7_000, first_ref),
        _segment(sample_id, 2, 7_000, 10_000, ocr.evidence_id),
    )
    summary_text = f"summary {sample_id}"
    result = VideoUnderstandingResult(
        run_id=f"run_{sample_id}",
        asset_sha256=_SHA,
        segments=segments,
        summary=VideoSummary(
            title="摘要",
            summary_zh="视频事实",
            speakers=("SPEAKER_01",),
            languages=(language,),
            duration_ms=10_000,
            chapters=(),
            retrieval_text=summary_text,
            retrieval_hash=_hash(summary_text),
        ),
    )
    evidence = (
        *words,
        SpeakerTurn(
            evidence_id=f"turn_{sample_id}_1",
            start_ms=0,
            end_ms=4_000,
            speaker="SPEAKER_01",
        ),
        SpeakerTurn(
            evidence_id=f"turn_{sample_id}_2",
            start_ms=2_000,
            end_ms=4_000,
            speaker="SPEAKER_02",
        ),
        ocr,
        AudioEvent(
            evidence_id=f"audio_{sample_id}",
            start_ms=2_000,
            end_ms=3_000,
            audioset_class="Music",
            normalized_event="music",
            confidence=0.9,
            threshold_version="v1",
        ),
        SceneBoundary(
            evidence_id=f"scene_{sample_id}_1",
            start_ms=0,
            end_ms=4_000,
            transition="candidate",
            score=0.9,
        ),
        SceneBoundary(
            evidence_id=f"scene_{sample_id}_2",
            start_ms=4_000,
            end_ms=10_000,
            transition="hard_cut",
            score=0.9,
        ),
    )
    index_sha = _hash(f"index-{sample_id}-{run_id}")
    claims = tuple(
        PredictionClaim(
            claim_id=f"claim_{sample_id}_{position}",
            source_kind="SEGMENT_SUMMARY" if position < 2 else "VIDEO_SUMMARY",
            source_id=(segments[position].segment_id if position < 2 else result.run_id),
            text="事实",
        )
        for position in range(3)
    )
    return VerifiedPrediction(
        index=EvaluationPrediction(
            schema_version="1.1.0",
            evaluation_run_id=run_id,
            sample_id=sample_id,
            media_sha256=_SHA,
            run_id=result.run_id,
            job_id=f"job_{sample_id}",
            terminal_status="SUCCEEDED",
            run_relative_path=f"predictions/{sample_id}/run.json",
            run_sha256=_hash(f"run-{sample_id}"),
            result_relative_path=f"predictions/{sample_id}/result.json",
            result_sha256=_hash(f"result-{sample_id}"),
            evidence_relative_path=f"predictions/{sample_id}/evidence.jsonl",
            evidence_sha256=_hash(f"evidence-{sample_id}"),
            artifact_manifest_relative_path=f"predictions/{sample_id}/manifest.json",
            artifact_manifest_sha256=_hash(f"manifest-{sample_id}"),
            transcript_source="ASR",
            started_at=_NOW,
            finished_at=_NOW,
        ),
        index_sha256=index_sha,
        run=PredictionRunSnapshot(
            schema_version="1.0.0",
            run_id=result.run_id,
            job_id=f"job_{sample_id}",
            terminal_status="SUCCEEDED",
            current_stage="RESULT",
            models=(ModelIdentity(component="asr", provider="local", model_id="m1"),),
        ),
        result=result,
        evidence=evidence,
        claims=claims,
        artifact_manifest_sha256=_hash(f"manifest-{sample_id}"),
        eval_root=Path("."),
    )


def _failed_prediction(sample_id: str, *, run_id: str = "eval_001") -> VerifiedPrediction:
    return VerifiedPrediction(
        index=EvaluationPrediction(
            schema_version="1.1.0",
            evaluation_run_id=run_id,
            sample_id=sample_id,
            media_sha256=_SHA,
            run_id=f"run_{sample_id}",
            job_id=f"job_{sample_id}",
            terminal_status="FAILED",
            run_relative_path=f"predictions/{sample_id}/run.json",
            run_sha256=_hash(f"run-{sample_id}"),
            failure_code="MODEL_FAILED",
            started_at=_NOW,
            finished_at=_NOW,
        ),
        index_sha256=_hash(f"index-{sample_id}-{run_id}"),
        run=PredictionRunSnapshot(
            schema_version="1.0.0",
            run_id=f"run_{sample_id}",
            job_id=f"job_{sample_id}",
            terminal_status="FAILED",
            current_stage="SPEECH",
            error_code="MODEL_FAILED",
            models=(ModelIdentity(component="asr", provider="local", model_id="m1"),),
        ),
        result=None,
        evidence=(),
        claims=(),
        artifact_manifest_sha256=None,
        eval_root=Path("."),
    )


def _subtitle_prediction(
    sample_id: str,
    language: str,
    cues: tuple[tuple[int, int, str, str], ...],
) -> VerifiedPrediction:
    base = _successful_prediction(sample_id, language, ("占位",))
    subtitle_cues = tuple(
        SubtitleCue(
            evidence_id=evidence_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            language=language,
            stream_index=2,
        )
        for start_ms, end_ms, evidence_id, text in cues
    )
    visual_evidence = tuple(
        item
        for item in base.evidence
        if isinstance(item, (OcrEvidence, SceneBoundary))
    )
    segments = (
        base.result.segments[0].model_copy(
            update={"evidence_refs": (subtitle_cues[0].evidence_id,)}
        ),
        base.result.segments[1],
    )
    result = base.result.model_copy(update={"segments": segments})
    return base.model_copy(
        update={
            "index": base.index.model_copy(update={"transcript_source": "SUBTITLE"}),
            "result": result,
            "evidence": (*subtitle_cues, *visual_evidence),
        }
    )


def _none_prediction(sample_id: str, language: str) -> VerifiedPrediction:
    base = _successful_prediction(sample_id, language, ("占位",))
    visual_evidence = tuple(
        item
        for item in base.evidence
        if isinstance(item, (OcrEvidence, SceneBoundary))
    )
    first_visual_id = visual_evidence[0].evidence_id
    result = base.result.model_copy(
        update={
            "segments": tuple(
                segment.model_copy(update={"evidence_refs": (first_visual_id,)})
                for segment in base.result.segments
            )
        }
    )
    return base.model_copy(
        update={
            "index": base.index.model_copy(update={"transcript_source": "NONE"}),
            "result": result,
            "evidence": visual_evidence,
        }
    )


def _annotation(sample_id: str, language: str, text: str) -> VerifiedAnnotation:
    tokens = tuple(text.split()) if language in ("en", "es") else tuple(text.replace(" ", ""))
    annotation = EvaluationAnnotation(
        schema_version="1.0.0",
        sample_id=sample_id,
        media_sha256=_SHA,
        duration_ms=10_000,
        language=language,
        reference_text=text,
        words=tuple(
            ReferenceWord(
                word_id=f"refword_{sample_id}_{index}",
                text=token,
                start_ms=index * 500,
                end_ms=index * 500 + 400,
            )
            for index, token in enumerate(tokens)
        ),
        speaker_turns=(
            ReferenceSpeakerTurn(
                turn_id=f"refturn_{sample_id}_1",
                speaker_id=f"speaker_{sample_id}_1",
                start_ms=0,
                end_ms=4_000,
            ),
            ReferenceSpeakerTurn(
                turn_id=f"refturn_{sample_id}_2",
                speaker_id=f"speaker_{sample_id}_2",
                start_ms=2_000,
                end_ms=4_000,
            ),
        ),
        ocr_frames=(
            ReferenceOcrFrame(
                frame_id=f"frame_{sample_id}",
                timestamp_ms=8_000,
                text_lines=("\uff21",),
            ),
        ),
        audio_events=(
            ReferenceAudioEvent(
                event_id=f"refaudio_{sample_id}",
                normalized_event="music",
                start_ms=1_000,
                end_ms=2_000,
            ),
        ),
        scene_boundaries_ms=(3_000,),
        semantic_boundaries_ms=(5_000,),
        supported_facts=(SupportedFact(fact_id=f"fact_{sample_id}", canonical_text="事实"),),
        key_fact_ids=(f"fact_{sample_id}",),
    )
    return VerifiedAnnotation(annotation=annotation, sha256=_hash(f"annotation-{sample_id}"))


def _fixture(tmp_path: Path) -> tuple[ValidatedEvaluationPackage, tuple[VerifiedPrediction, ...]]:
    specs = (
        ("sample_zh_1", "zh", "甲乙", ("甲", "丙")),
        ("sample_zh_2", "zh", "甲乙丙丁戊己庚辛", tuple("甲乙丙丁戊己庚辛")),
        ("sample_ja", "ja", "かな", tuple("かな")),
        ("sample_ko", "ko", "가나", tuple("가나")),
        ("sample_en", "en", "one two three", ("one", "three")),
        ("sample_es", "es", "uno dos", ("uno", "dos")),
        ("sample_failed", "en", "lost", ()),
    )
    annotations = tuple(
        _annotation(sample_id, language, text)
        for sample_id, language, text, _ in specs
    )
    samples = tuple(
        EvaluationSample(
            sample_id=sample_id,
            language=language,
            authorization_id="auth_001",
            media_relative_path=f"media/{sample_id}.mp4",
            media_sha256=_SHA,
            annotations_relative_path=f"annotations/{sample_id}.json",
            annotations_sha256=annotation.sha256,
        )
        for (sample_id, language, _text, _predicted), annotation in zip(
            specs, annotations, strict=True
        )
    )
    package = ValidatedEvaluationPackage(
        dataset=EvaluationDataset(
            samples=samples,
            eval_root=Path("."),
            runtime_root=Path("."),
            workspace_root=Path("."),
        ),
        authorization=AuthorizationFile(
            schema_version="1.0.0",
            records=(
                AuthorizationRecord(
                    schema_version="1.0.0",
                    authorization_id="auth_001",
                    source_category="OWNED",
                    allowed_purposes=("VIDEO_QUALITY_EVALUATION",),
                    confirmed_at=_NOW,
                    media_sha256=(_SHA,),
                ),
            ),
        ),
        annotations=annotations,
        dataset_sha256=_hash("dataset"),
        authorization_sha256=_hash("authorization"),
    )
    predictions = tuple(
        _failed_prediction(sample_id)
        if sample_id == "sample_failed"
        else _successful_prediction(sample_id, language, predicted)
        for sample_id, language, _text, predicted in specs
    )
    return _materialize_fixture(tmp_path, package, predictions)


def _materialize_fixture(
    tmp_path: Path,
    package: ValidatedEvaluationPackage,
    predictions: tuple[VerifiedPrediction, ...],
) -> tuple[ValidatedEvaluationPackage, tuple[VerifiedPrediction, ...]]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    dataset_path = eval_root / "dataset.jsonl"
    authorization_path = eval_root / "authorization.json"
    annotation_by_id = {item.annotation.sample_id: item.annotation for item in package.annotations}
    dataset_lines = []
    for sample in package.dataset.samples:
        media_path = eval_root / sample.media_relative_path
        media_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(_MEDIA_BYTES)
        annotation_path = eval_root / sample.annotations_relative_path
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            annotation_by_id[sample.sample_id].model_dump_json(exclude_computed_fields=True),
            encoding="utf-8",
        )
        dataset_lines.append(
            sample.model_copy(
                update={
                    "annotations_sha256": hashlib.sha256(
                        annotation_path.read_bytes()
                    ).hexdigest()
                }
            ).model_dump_json()
        )
    dataset_path.write_text("\n".join(dataset_lines), encoding="utf-8")
    authorization_path.write_text(package.authorization.model_dump_json(), encoding="utf-8")
    loaded_package = load_evaluation_package(
        dataset_path,
        authorization_path,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
    )
    samples = {sample.sample_id: sample for sample in loaded_package.dataset.samples}
    loaded_predictions = []
    for prediction in predictions:
        sample_id = prediction.index.sample_id
        prediction_dir = eval_root / "predictions" / "eval_001" / sample_id
        prediction_dir.mkdir(parents=True, exist_ok=True)
        run_path = prediction_dir / "run.json"
        run_path.write_text(
            prediction.run.model_dump_json(exclude_computed_fields=True), encoding="utf-8"
        )
        updates: dict[str, object] = {
            "run_relative_path": run_path.relative_to(eval_root).as_posix(),
            "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        }
        if prediction.result is not None:
            result_path = prediction_dir / "result.json"
            result_path.write_text(
                prediction.result.model_dump_json(exclude_computed_fields=True),
                encoding="utf-8",
            )
            evidence_path = prediction_dir / "evidence.jsonl"
            evidence_path.write_text(
                "\n".join(
                    item.model_dump_json(exclude_computed_fields=True)
                    for item in prediction.evidence
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = prediction_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "upstream_sha256": _SHA,
                        "payload": {
                            "result": prediction.result.model_dump(
                                mode="json", exclude_computed_fields=True
                            ),
                            "evidence": [
                                item.model_dump(mode="json", exclude_computed_fields=True)
                                for item in prediction.evidence
                            ],
                            "stage_metrics": {"RESULT": 1},
                            "status": prediction.index.terminal_status,
                            "warnings": list(prediction.run.warning_codes),
                            "transcript_source": prediction.index.transcript_source,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            updates.update(
                result_relative_path=result_path.relative_to(eval_root).as_posix(),
                result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                evidence_relative_path=evidence_path.relative_to(eval_root).as_posix(),
                evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                artifact_manifest_relative_path=manifest_path.relative_to(eval_root).as_posix(),
                artifact_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
        index_path = prediction_dir / "index.json"
        index_path.write_text(
            prediction.index.model_copy(update=updates).model_dump_json(), encoding="utf-8"
        )
        loaded_predictions.append(
            load_verified_prediction(
                index_path,
                eval_root=eval_root,
                workspace_root=tmp_path,
                runtime_root=runtime_root,
                sample=samples[sample_id],
            )
        )
    return loaded_package, tuple(loaded_predictions)


def _judgments(
    package: ValidatedEvaluationPackage,
    predictions: tuple[VerifiedPrediction, ...],
) -> tuple[SemanticJudgment, ...]:
    annotations = {item.annotation.sample_id: item for item in package.annotations}
    result = []
    for prediction in predictions:
        sample_id = prediction.index.sample_id
        result.append(
            SemanticJudgment(
                schema_version="1.0.0",
                sample_id=sample_id,
                annotation_sha256=annotations[sample_id].sha256,
                prediction_sha256=prediction.index_sha256,
                claim_judgments=tuple(
                    ClaimJudgment(
                        claim_id=claim.claim_id,
                        verdict="UNSUPPORTED" if index == 2 else "SUPPORTED",
                    )
                    for index, claim in enumerate(prediction.claims)
                ),
                matched_key_fact_ids=(
                    () if sample_id == "sample_failed" else (f"fact_{sample_id}",)
                ),
                fabricated_names=(("虚构姓名",) if sample_id == "sample_es" else ()),
                reviewer_id="reviewer_001",
                reviewed_at=_NOW,
                rubric_version="v1",
            )
        )
    return tuple(result)


def _score(
    package: ValidatedEvaluationPackage,
    predictions: tuple[VerifiedPrediction, ...],
    judgments: tuple[SemanticJudgment, ...] = (),
):
    from video_demo.evaluation.quality_runner import score_quality

    return score_quality(package, predictions, judgments, evaluation_run_id="eval_001")


def _hint_effect_fixture(
    tmp_path: Path,
    *,
    exclusion: str | None = None,
) -> tuple[ValidatedEvaluationPackage, tuple[VerifiedPrediction, ...]]:
    pair_specs = (
        (
            "pair_b_zh",
            "zh",
            "向量 数据库",
            ("向量 数据库",),
            ("向量", "数据湖"),
            ("向量", "数据库"),
        ),
        (
            "pair_a_en",
            "en",
            "Milvus DB course",
            ("Milvus   DB",),
            ("milvus", "da", "course"),
            ("MILVUS", "db", "course"),
        ),
    )
    samples: list[EvaluationSample] = []
    annotations: list[VerifiedAnnotation] = []
    predictions: list[VerifiedPrediction] = []
    for pair_index, (
        pair_id,
        language,
        reference_text,
        original_terms,
        none_words,
        correct_words,
    ) in enumerate(pair_specs):
        terms = () if exclusion == "TERMS_EMPTY" and pair_index == 0 else original_terms
        source_annotation = _annotation(
            f"{pair_id}_none", language, reference_text
        ).annotation
        base = source_annotation.model_copy(update={"terms": terms})
        # model_copy 不执行字段校验，因此显式从 JSON 重新解析，冻结术语规范化契约。
        base = EvaluationAnnotation.model_validate_json(
            base.model_dump_json(exclude_computed_fields=True)
        )
        reference_sha = pair_reference_sha256(base)
        for variant, predicted_words in (("NONE", none_words), ("CORRECT", correct_words)):
            sample_id = f"{pair_id}_{variant.lower()}"
            annotation = base.model_copy(update={"sample_id": sample_id})
            verified = VerifiedAnnotation(
                annotation=annotation,
                sha256=_hash(f"annotation-{sample_id}"),
            )
            annotations.append(verified)
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    language=language,
                    authorization_id="auth_001",
                    media_relative_path=f"media/{sample_id}.mp4",
                    media_sha256=_SHA,
                    annotations_relative_path=f"annotations/{sample_id}.json",
                    annotations_sha256=verified.sha256,
                    hotwords=(
                        original_terms
                        if variant == "CORRECT" and original_terms
                        else ()
                    ),
                    pair_id=pair_id,
                    hint_variant=variant,
                    pair_reference_sha256=reference_sha,
                )
            )
            prediction = _successful_prediction(sample_id, language, predicted_words)
            if pair_index == 0 and variant == "CORRECT":
                if exclusion == "PREDICTION_NOT_SUCCESSFUL":
                    prediction = _failed_prediction(sample_id)
                elif exclusion == "TRANSCRIPT_SOURCE_NOT_ASR":
                    prediction = _subtitle_prediction(
                        sample_id,
                        language,
                        ((0, 900, f"subtitle_{sample_id}", " ".join(predicted_words)),),
                    )
            predictions.append(prediction)
    package = ValidatedEvaluationPackage(
        dataset=EvaluationDataset(
            samples=tuple(samples),
            eval_root=Path("."),
            runtime_root=Path("."),
            workspace_root=Path("."),
        ),
        authorization=AuthorizationFile(
            schema_version="1.0.0",
            records=(
                AuthorizationRecord(
                    schema_version="1.0.0",
                    authorization_id="auth_001",
                    source_category="OWNED",
                    allowed_purposes=("VIDEO_QUALITY_EVALUATION",),
                    confirmed_at=_NOW,
                    media_sha256=(_SHA,),
                ),
            ),
        ),
        annotations=tuple(annotations),
        dataset_sha256=_hash("hint-dataset"),
        authorization_sha256=_hash("hint-authorization"),
    )
    return _materialize_fixture(tmp_path, package, tuple(predictions))


def test_score_quality_micro_averages_languages_and_counts_failed_schema_denominator(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)

    artifacts = _score(package, predictions)
    metrics = {metric.name: metric for metric in artifacts.report.metrics}

    assert metrics["zh_cer"].value == 0.1
    assert metrics["ja_cer"].value == 0.0
    assert metrics["ko_cer"].value == 0.0
    assert metrics["en_wer"].value == 0.5  # (1 删除 + 失败样本 1 删除) / 4 词
    assert metrics["es_wer"].value == 0.0
    assert metrics["schema_time_valid_rate"].value == 6 / 7
    assert [detail.sample_id for detail in artifacts.sample_details] == [
        sample.sample_id for sample in package.dataset.samples
    ]


def test_score_quality_uses_alignment_der_diagnostics_nfkc_and_fixed_tolerances(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)

    artifacts = _score(package, predictions)
    metrics = {metric.name: metric for metric in artifacts.report.metrics}
    signature = inspect.signature(__import__(
        "video_demo.evaluation.quality_runner", fromlist=["score_quality"]
    ).score_quality)

    assert metrics["word_time_p90_ms"].value == 100.0
    assert metrics["der_non_overlap"].value == pytest.approx(1 / 7)
    assert metrics["der_overlap"].value == pytest.approx(1 / 7)
    assert metrics["ocr_accuracy"].value == pytest.approx(6 / 7)
    assert metrics["audio_event_macro_f1"].value == pytest.approx(12 / 13)
    assert metrics["scene_f1"].value == pytest.approx(12 / 13)
    assert metrics["semantic_boundary_f1"].value == pytest.approx(12 / 13)
    assert "speaker_count_accuracy" in artifacts.sample_details[0].metric_inputs
    assert "speaker_count_accuracy" not in metrics
    assert tuple(signature.parameters) == (
        "package",
        "predictions",
        "judgments",
        "evaluation_run_id",
    )


def test_subtitle_sample_uses_sorted_cues_and_marks_unexecuted_metrics_not_run() -> None:
    from video_demo.evaluation import quality_runner

    annotation = _annotation("sample_subtitle", "en", "vector database")
    prediction = _subtitle_prediction(
        "sample_subtitle",
        "en",
        (
            (500, 900, "subtitle_002", "database"),
            (0, 400, "subtitle_001", "vector"),
        ),
    )
    accumulators = quality_runner._Accumulators()

    detail = quality_runner._score_sample(annotation, prediction, accumulators)
    observations = accumulators.observations()

    assert observations["en_wer"].value == 0
    assert detail.transcript_source == "SUBTITLE"
    assert detail.not_run_metrics == (
        "audio_event_macro_f1",
        "der_non_overlap",
        "der_overlap",
        "speaker_count_accuracy",
        "word_time_p90_ms",
    )
    assert {
        "audio_event_macro_f1",
        "speaker_count_accuracy",
        "word_time_match_count",
    }.isdisjoint(detail.metric_inputs)
    for name in (
        "audio_event_macro_f1",
        "der_non_overlap",
        "der_overlap",
        "word_time_p90_ms",
    ):
        assert observations[name].value is None


def test_hint_effect_report_uses_bound_pairs_and_normalized_exact_terms(
    tmp_path: Path,
) -> None:
    from video_demo.evaluation.thresholds import QUALITY_THRESHOLDS

    package, predictions = _hint_effect_fixture(tmp_path)

    artifacts = _score(package, predictions)
    report = artifacts.hint_effect_report

    assert report.status == "RUN"
    assert report.candidate_pair_count == 2
    assert report.eligible_pair_count == 2
    assert report.excluded_pair_counts == {}
    assert [pair.pair_id for pair in report.pairs] == ["pair_a_en", "pair_b_zh"]
    english, chinese = report.pairs
    assert english.metric_name == "WER"
    assert english.none_term_recall == 0
    assert english.correct_term_recall == 1
    assert english.term_recall_delta == 1
    assert english.none_text_error_rate == pytest.approx(1 / 3)
    assert english.correct_text_error_rate == 0
    assert english.text_error_rate_delta == pytest.approx(-1 / 3)
    assert chinese.metric_name == "CER"
    assert chinese.none_term_recall == 0
    assert chinese.correct_term_recall == 1
    assert chinese.none_text_error_rate == pytest.approx(1 / 5)
    assert chinese.correct_text_error_rate == 0
    assert not any(name.startswith("hint_") for name in QUALITY_THRESHOLDS)
    assert not any(
        metric.name.startswith("hint_") for metric in artifacts.report.metrics
    )


def test_hint_effect_report_not_run_without_candidate_pairs(tmp_path: Path) -> None:
    package, predictions = _fixture(tmp_path)

    report = _score(package, predictions).hint_effect_report

    assert report.status == "NOT_RUN"
    assert report.candidate_pair_count == 0
    assert report.eligible_pair_count == 0
    assert report.excluded_pair_counts == {}
    assert report.pairs == ()
    assert report.not_run_reason == "数据集没有 NONE/CORRECT 提示效果配对"


@pytest.mark.parametrize(
    "exclusion",
    [
        "PREDICTION_NOT_SUCCESSFUL",
        "TERMS_EMPTY",
        "TRANSCRIPT_SOURCE_NOT_ASR",
    ],
)
def test_hint_effect_report_counts_stable_pair_exclusions(
    exclusion: str,
    tmp_path: Path,
) -> None:
    package, predictions = _hint_effect_fixture(tmp_path, exclusion=exclusion)

    report = _score(package, predictions).hint_effect_report

    assert report.status == "RUN"
    assert report.candidate_pair_count == 2
    assert report.eligible_pair_count == 1
    assert report.excluded_pair_counts == {exclusion: 1}
    assert len(report.pairs) == 1


def test_mixed_quality_metrics_only_exclude_subtitle_samples() -> None:
    from video_demo.evaluation import quality_runner

    asr_annotation = _annotation("sample_asr", "en", "one two")
    asr_prediction = _successful_prediction("sample_asr", "en", ("one", "two"))
    subtitle_annotation = _annotation("sample_subtitle", "en", "vector database")
    subtitle_prediction = _subtitle_prediction(
        "sample_subtitle",
        "en",
        (
            (0, 400, "subtitle_001", "vector"),
            (500, 900, "subtitle_002", "database"),
        ),
    )
    asr_only = quality_runner._Accumulators()
    mixed = quality_runner._Accumulators()

    quality_runner._score_sample(asr_annotation, asr_prediction, asr_only)
    quality_runner._score_sample(subtitle_annotation, subtitle_prediction, mixed)
    quality_runner._score_sample(asr_annotation, asr_prediction, mixed)

    asr_observations = asr_only.observations()
    mixed_observations = mixed.observations()
    for name in (
        "audio_event_macro_f1",
        "der_non_overlap",
        "der_overlap",
        "word_time_p90_ms",
    ):
        assert mixed_observations[name] == asr_observations[name]


@pytest.mark.parametrize(
    ("prediction", "expected_source"),
    [
        (_failed_prediction("sample_failed_scope"), None),
        (_none_prediction("sample_none_scope", "en"), "NONE"),
    ],
)
def test_failed_and_none_samples_keep_existing_metric_denominators(
    prediction: VerifiedPrediction,
    expected_source: str | None,
) -> None:
    from video_demo.evaluation import quality_runner

    annotation = _annotation(prediction.index.sample_id, "en", "lost term")
    accumulators = quality_runner._Accumulators()

    detail = quality_runner._score_sample(annotation, prediction, accumulators)

    assert detail.transcript_source == expected_source
    assert detail.not_run_metrics == ()
    assert accumulators.text_counts["en_wer"].reference_units == 2
    assert sum(accumulators.der_units) > 0
    assert accumulators.audio_counts


def test_semantic_metrics_only_come_from_bound_complete_reviews(tmp_path: Path) -> None:
    package, predictions = _fixture(tmp_path)
    judgments = _judgments(package, predictions)

    no_review = _score(package, predictions)
    partial = _score(package, predictions, judgments[:-1])
    complete = _score(package, predictions, judgments)
    no_review_metrics = {metric.name: metric for metric in no_review.report.metrics}
    complete_metrics = {metric.name: metric for metric in complete.report.metrics}

    assert no_review_metrics["fact_support_rate"].status == GateStatus.NOT_RUN
    assert no_review.report.status != GateStatus.PASS
    assert partial.report.status == GateStatus.FAIL
    assert partial.report.failure_code == "SEMANTIC_REVIEW_INCOMPLETE"
    assert complete_metrics["fact_support_rate"].value == 12 / 18
    assert complete_metrics["key_fact_recall"].value == 6 / 7
    assert complete_metrics["fabricated_name_count"].value == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "foreign", "cross_run"])
def test_predictions_must_exactly_cover_dataset_and_run(
    mutation: str, tmp_path: Path
) -> None:
    package, predictions = _fixture(tmp_path)
    if mutation == "duplicate":
        invalid = (*predictions[:-1], predictions[0])
    elif mutation == "missing":
        invalid = predictions[:-1]
    elif mutation == "foreign":
        invalid = (*predictions[:-1], _failed_prediction("sample_foreign"))
    else:
        invalid = (*predictions[:-1], _failed_prediction("sample_failed", run_id="eval_other"))

    with pytest.raises(ValueError):
        _score(package, tuple(invalid))


@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "annotation", "prediction"])
def test_present_judgments_must_be_unique_and_bound(
    mutation: str, tmp_path: Path
) -> None:
    package, predictions = _fixture(tmp_path)
    judgments = list(_judgments(package, predictions))
    if mutation == "duplicate":
        judgments[-1] = judgments[0]
    elif mutation == "foreign":
        judgments[-1] = judgments[-1].model_copy(update={"sample_id": "sample_foreign"})
    elif mutation == "annotation":
        judgments[-1] = judgments[-1].model_copy(update={"annotation_sha256": "f" * 64})
    else:
        judgments[-1] = judgments[-1].model_copy(update={"prediction_sha256": "f" * 64})

    with pytest.raises(ValueError):
        _score(package, predictions, tuple(judgments))


def test_unknown_evidence_and_schema_validity_are_recomputed_from_verified_prediction(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)
    first = predictions[0]
    assert first.result is not None
    bad_segment = first.result.segments[0].model_copy(
        update={"evidence_refs": ("missing_001",)}
    )
    bad_result = first.result.model_copy(
        update={"segments": (bad_segment, *first.result.segments[1:])}
    )
    bad_prediction = first.model_copy(update={"result": bad_result})

    with pytest.raises(Exception, match="来源"):
        _score(package, (bad_prediction, *predictions[1:]))


def test_speaker_turn_overlap_speakers_participate_in_der_and_count_diagnostic(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)
    first = predictions[0]
    turns = tuple(item for item in first.evidence if isinstance(item, SpeakerTurn))
    evidence = (
        *(item for item in first.evidence if not isinstance(item, SpeakerTurn)),
        turns[0].model_copy(update={"overlap_speakers": ("SPEAKER_02",)}),
    )
    prediction = first.model_copy(update={"evidence": evidence})

    from video_demo.evaluation import quality_runner

    detail = quality_runner._score_sample(
        package.annotations[0], prediction, quality_runner._Accumulators()
    )

    assert detail.metric_inputs["speaker_count_accuracy"] == 1.0


def test_bound_report_digests_use_one_canonical_json_rule_and_bind_details(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)
    judgments = _judgments(package, predictions)

    artifacts = _score(package, predictions, judgments)

    def digest(value: object) -> str:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    assert artifacts.report.prediction_index_sha256 == digest(
        [prediction.index_sha256 for prediction in predictions]
    )
    assert artifacts.report.judgment_index_sha256 == digest(
        [judgment.model_dump(mode="json") for judgment in judgments]
    )
    assert artifacts.report.sample_details_sha256 == digest(
        [detail.model_dump(mode="json") for detail in artifacts.sample_details]
    )
    tampered = list(detail.model_dump(mode="json") for detail in artifacts.sample_details)
    tampered[0]["metric_inputs"]["speaker_count_accuracy"] = 0
    assert digest(tampered) != artifacts.report.sample_details_sha256
    assert digest(["f" * 64, *(item.index_sha256 for item in predictions[1:])]) != (
        artifacts.report.prediction_index_sha256
    )
    tampered_judgments = [judgment.model_dump(mode="json") for judgment in judgments]
    tampered_judgments[0]["rubric_version"] = "tampered"
    assert digest(tampered_judgments) != artifacts.report.judgment_index_sha256


def test_score_quality_rejects_predictions_and_package_without_loader_sources(
    tmp_path: Path,
) -> None:
    package, predictions = _fixture(tmp_path)
    package = package.model_copy(
        update={"dataset_path": None, "authorization_path": None}
    )

    with pytest.raises(Exception, match="来源"):
        _score(package, predictions)


@pytest.mark.parametrize("field", ["index_path", "workspace_root", "runtime_root"])
def test_score_quality_rejects_prediction_missing_real_source_without_path_leak(
    field: str, tmp_path: Path
) -> None:
    package, predictions = _fixture(tmp_path)
    altered = predictions[0].model_copy(update={field: None})

    with pytest.raises(VideoDemoError) as raised:
        _score(package, (altered, *predictions[1:]))

    assert raised.value.__cause__ is None
    assert "来源" in raised.value.message
    assert str(tmp_path) not in raised.value.message


def test_loader_source_paths_are_excluded_from_all_serialization(tmp_path: Path) -> None:
    package, predictions = _fixture(tmp_path)
    source_fields = {
        "source_path",
        "dataset_path",
        "authorization_path",
        "workspace_root",
        "runtime_root",
        "max_video_bytes",
        "index_path",
        "eval_root",
    }
    models = (package.dataset, package, predictions[0])

    for model in models:
        dumped = model.model_dump(mode="json")
        encoded = model.model_dump_json()
        assert source_fields.isdisjoint(dumped)
        assert all(f'"{field}"' not in encoded for field in source_fields)


def test_score_sample_aligns_word_timing_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from video_demo.evaluation import quality_runner

    package, predictions = _fixture(tmp_path)
    original = quality_runner.aligned_word_time_errors_ms
    call_count = 0

    def counted_alignment(*args: object, **kwargs: object) -> tuple[int, ...]:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(quality_runner, "aligned_word_time_errors_ms", counted_alignment)
    quality_runner._score_sample(
        package.annotations[0], predictions[0], quality_runner._Accumulators()
    )

    assert call_count == 1
