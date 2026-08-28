from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from video_demo.application.document_rendering import render_markdown
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import (
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
    DocumentArtifactPayload,
)
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import ModelIdentity
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    AuthorizationRecord,
    EvaluationAnnotation,
    ReferenceVisualFrame,
    SupportedFact,
    ValidatedEvaluationPackage,
    load_evaluation_package,
)
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.predictions import (
    EvaluationPrediction,
    PredictionRunSnapshot,
    VerifiedPrediction,
    load_verified_prediction,
)
from video_demo.evaluation.quality_runner import score_quality
from video_demo.storage.artifacts import (
    RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
    canonical_artifact_envelope_bytes,
)

_NOW = datetime(2026, 8, 18, tzinfo=UTC)
_MEDIA_BYTES = b"fixture-media"


def _hash(content: bytes | str) -> str:
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(encoded).hexdigest()


def _write_prediction(
    eval_root: Path,
    sample: EvaluationSample,
) -> Path:
    directory = eval_root / "predictions" / "eval_001" / sample.sample_id
    directory.mkdir(parents=True)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=250,
        text="你好",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    first_chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=250,
        title="问候",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="讲者问好。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(ParagraphBlock(text="讲者问好。", evidence_refs=(speech.evidence_id,)),),
        claims=(
            GroundedClaim(
                text="讲者进行了问候。",
                evidence_refs=(speech.evidence_id,),
                certainty=0.9,
            ),
        ),
        evidence_refs=(speech.evidence_id,),
        transcript_source="ASR",
        retrieval_text="你好",
        retrieval_hash=_hash("你好"),
    )
    second_chapter = SemanticChapter(
        chapter_id="chapter_002",
        start_ms=250,
        end_ms=500,
        title="本时段未提取到可验证语义内容",
        title_evidence_refs=(),
        summary_zh="本时段未提取到可验证语义内容",
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash=_hash(""),
    )
    result = VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256=sample.media_sha256,
        summary=VideoDocumentSummary(
            title="视频",
            duration_ms=500,
            overview_zh="视频包含问候。",
            key_points=(),
            retrieval_text="视频问候",
            retrieval_hash=_hash("视频问候"),
        ),
        chapters=(first_chapter, second_chapter),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text-model",
            vlm_model_id="qwen3-vl-flash",
            prompt_versions=PromptVersions(
                chapter_planner="chapter-planner-v1",
                chapter_planner_repair="chapter-planner-repair-v1",
                chapter_vlm="chapter-vlm-v1",
                chapter_vlm_repair="chapter-vlm-repair-v1",
                chapter_writer="chapter-writer-v1",
                chapter_writer_repair="chapter-writer-repair-v1",
                global_editor="global-editor-v1",
                global_editor_repair="global-editor-repair-v1",
            ),
        ),
    )
    evidence = (speech,)
    document = render_markdown(result, evidence)
    artifact = DocumentArtifactPayload(
        result=result,
        evidence=evidence,
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        status="SUCCEEDED",
        warnings=(),
        transcript_source="ASR",
        document_sha256=document.sha256,
        document_size_bytes=document.size_bytes,
    )
    snapshot = PredictionRunSnapshot(
        schema_version="1.0.0",
        run_id=result.run_id,
        job_id="job_001",
        terminal_status="SUCCEEDED",
        current_stage="RESULT",
        models=(ModelIdentity(component="asr", provider="test", model_id="model"),),
    )
    result_bytes = result.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    evidence_bytes = speech.model_dump_json(exclude_computed_fields=True).encode("utf-8") + b"\n"
    run_bytes = snapshot.model_dump_json(exclude_none=True).encode("utf-8")
    manifest_bytes = canonical_artifact_envelope_bytes(
        artifact.model_dump(mode="json", exclude_computed_fields=True),
        RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
        sample.media_sha256,
    )
    files = {
        "run.json": run_bytes,
        "result.json": result_bytes,
        "evidence.jsonl": evidence_bytes,
        "document.md": document.content,
        "artifact-manifest.json": manifest_bytes,
    }
    for name, content in files.items():
        (directory / name).write_bytes(content)
    prefix = f"predictions/eval_001/{sample.sample_id}"
    index = EvaluationPrediction(
        schema_version="1.2.0",
        evaluation_run_id="eval_001",
        sample_id=sample.sample_id,
        media_sha256=sample.media_sha256,
        run_id=result.run_id,
        job_id="job_001",
        terminal_status="SUCCEEDED",
        run_relative_path=f"{prefix}/run.json",
        run_sha256=_hash(run_bytes),
        result_relative_path=f"{prefix}/result.json",
        result_sha256=_hash(result_bytes),
        evidence_relative_path=f"{prefix}/evidence.jsonl",
        evidence_sha256=_hash(evidence_bytes),
        document_relative_path=f"{prefix}/document.md",
        document_sha256=document.sha256,
        document_size_bytes=document.size_bytes,
        artifact_manifest_relative_path=f"{prefix}/artifact-manifest.json",
        artifact_manifest_sha256=_hash(manifest_bytes),
        transcript_source="ASR",
        started_at=_NOW,
        finished_at=_NOW,
    )
    index_path = directory / "index.json"
    index_path.write_bytes(index.model_dump_json(exclude_none=True).encode("utf-8"))
    return index_path


def _materialized_fixture(
    tmp_path: Path,
) -> tuple[ValidatedEvaluationPackage, tuple[VerifiedPrediction, ...]]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    media_path = eval_root / "media/sample.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(_MEDIA_BYTES)
    annotation = EvaluationAnnotation(
        schema_version="2.0.0",
        sample_id="sample_001",
        media_sha256=_hash(_MEDIA_BYTES),
        duration_ms=500,
        language="zh",
        reference_text="你好",
        visual_frames=(
            ReferenceVisualFrame(
                frame_id="frame_001",
                timestamp_ms=100,
                text_lines=("你好",),
            ),
        ),
        scene_boundaries_ms=(250,),
        semantic_boundaries_ms=(250,),
        supported_facts=(SupportedFact(fact_id="fact_001", canonical_text="你好"),),
        key_fact_ids=("fact_001",),
    )
    annotation_path = eval_root / "annotations/sample.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_bytes = annotation.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    annotation_path.write_bytes(annotation_bytes)
    sample = EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=_hash(_MEDIA_BYTES),
        annotations_relative_path="annotations/sample.json",
        annotations_sha256=_hash(annotation_bytes),
    )
    dataset_path = eval_root / "dataset.jsonl"
    dataset_path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    authorization = AuthorizationFile(
        schema_version="1.0.0",
        records=(
            AuthorizationRecord(
                schema_version="1.0.0",
                authorization_id="auth_001",
                source_category="OWNED",
                allowed_purposes=("VIDEO_QUALITY_EVALUATION",),
                confirmed_at=_NOW,
                media_sha256=(sample.media_sha256,),
            ),
        ),
    )
    authorization_path = eval_root / "authorization.json"
    authorization_path.write_text(authorization.model_dump_json(), encoding="utf-8")
    index = _write_prediction(eval_root, sample)
    package = load_evaluation_package(
        dataset_path,
        authorization_path,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
    )
    prediction = load_verified_prediction(
        index,
        eval_root=eval_root,
        workspace_root=tmp_path,
        runtime_root=runtime_root,
        sample=sample,
    )
    return package, (prediction,)


def test_quality_uses_chapter_boundaries_and_marks_retired_visual_metrics_not_run(
    tmp_path: Path,
) -> None:
    package, predictions = _materialized_fixture(tmp_path)

    artifacts = score_quality(package, predictions, (), evaluation_run_id="eval_001")

    metrics = {item.name: item for item in artifacts.report.metrics}
    assert metrics["visual_text_accuracy"].value is None
    assert (
        metrics["visual_text_accuracy"].not_run_reason
        == "代表性视觉质量事实尚未接入"
    )
    assert metrics["visual_key_field_recall"].value is None
    assert (
        metrics["visual_key_field_recall"].not_run_reason
        == "代表性视觉质量事实尚未接入"
    )
    assert metrics["scene_f1"].value is None
    assert metrics["scene_f1"].not_run_reason == "3.0 生产结果不再公开场景边界证据"
    assert metrics["semantic_boundary_f1"].value == 1.0
    assert metrics["schema_time_valid_rate"].value == 1.0
    assert artifacts.sample_details[0].metric_inputs["semantic_boundary_f1"] == 1.0
    assert artifacts.document_quality_report.automatic_metrics is None
    assert artifacts.document_quality_report.human_metrics is None
    assert artifacts.document_quality_report.status == "NOT_RUN"


def test_quality_report_is_bound_to_prediction_index_and_has_no_runtime_metrics(
    tmp_path: Path,
) -> None:
    package, predictions = _materialized_fixture(tmp_path)

    artifacts = score_quality(package, predictions, (), evaluation_run_id="eval_001")

    assert artifacts.report.evaluation_run_id == "eval_001"
    assert artifacts.report.prediction_index_sha256 == _hash(
        json.dumps(
            [predictions[0].index_sha256],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    metrics = {item.name: item for item in artifacts.report.metrics}
    assert metrics["rtf"].value is None
    assert artifacts.hint_effect_report.status == "NOT_RUN"
