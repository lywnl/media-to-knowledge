from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    VisualBlock,
)
from video_demo.domain.document_artifact import (
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
    DocumentArtifactPayload,
)
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.predictions import (
    EvaluationPrediction,
    PredictionRunSnapshot,
    load_verified_prediction,
    reverify_verified_prediction,
)
from video_demo.storage.artifacts import (
    RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION,
    canonical_artifact_envelope_bytes,
)

_MEDIA_BYTES = b"fixture-media"


def _digest(content: bytes | str) -> str:
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(encoded).hexdigest()


def _sample() -> EvaluationSample:
    return EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=_digest(_MEDIA_BYTES),
        annotations_relative_path="annotations/sample.json",
        annotations_sha256="b" * 64,
    )


def _result() -> tuple[VideoUnderstandingResult, tuple[DocumentEvidenceItem, ...], bytes]:
    sample = _sample()
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=500,
        text="你好",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    image = b"\xff\xd8\xffprediction-keyframe\xff\xd9"
    image_sha = _digest(image)
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_001",
        start_ms=100,
        end_ms=101,
        keyframe_id="frame_001",
        timestamp_ms=100,
        relative_path=f"visual/keyframes/{image_sha}.jpg",
        mime_type="image/jpeg",
        sha256=image_sha,
        perceptual_hash="0123456789abcdef",
        size_bytes=len(image),
    )
    observation = VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="chapter_001",
        start_ms=90,
        end_ms=110,
        target_ids=("target_001",),
        keyframe_refs=(keyframe.evidence_id,),
        transcript_evidence_refs=(speech.evidence_id,),
        visual_type="TEXT",
        caption="画面显示你好。",
        content_blocks=(
            VisualTextContent(
                visual_content_id="visual_content_001",
                source_keyframe_refs=(keyframe.evidence_id,),
                text="你好",
            ),
        ),
        relation_to_transcript="SUPPORTING",
        certainty=0.9,
    )
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=500,
        title="问候",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="讲者问好。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(
            ParagraphBlock(text="讲者问好。", evidence_refs=(speech.evidence_id,)),
            VisualBlock(
                visual_observation_ref=observation.evidence_id,
                visual_content_refs=("visual_content_001",),
                caption=observation.caption,
                evidence_refs=(observation.evidence_id,),
            ),
        ),
        claims=(
            GroundedClaim(
                text="讲者进行了问候。",
                evidence_refs=(speech.evidence_id,),
                certainty=0.9,
            ),
        ),
        evidence_refs=(speech.evidence_id, keyframe.evidence_id, observation.evidence_id),
        selected_keyframe_refs=(keyframe.evidence_id,),
        transcript_source="ASR",
    )
    result = VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256=sample.media_sha256,
        summary=VideoDocumentSummary(
            title="视频",
            duration_ms=500,
            overview_zh="视频包含问候。",
        ),
        chapters=(chapter,),
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
    return result, (speech, keyframe, observation), image


def _write_prediction(tmp_path: Path) -> tuple[Path, EvaluationSample]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    eval_root = runtime_root / "eval"
    directory = eval_root / "predictions" / "eval_001" / "sample_001"
    directory.mkdir(parents=True)
    sample = _sample()
    result, evidence, image = _result()
    document = render_markdown(result, evidence)
    payload = DocumentArtifactPayload(
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
    evidence_bytes = b"".join(
        item.model_dump_json(exclude_computed_fields=True).encode("utf-8") + b"\n"
        for item in evidence
    )
    run_bytes = snapshot.model_dump_json(exclude_none=True).encode("utf-8")
    manifest_bytes = canonical_artifact_envelope_bytes(
        payload.model_dump(mode="json", exclude_computed_fields=True),
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
    keyframe = evidence[1]
    assert isinstance(keyframe, KeyframeEvidence)
    keyframe_path = directory / keyframe.relative_path
    keyframe_path.parent.mkdir(parents=True)
    keyframe_path.write_bytes(image)
    keyframe_path.chmod(0o600)
    prefix = "predictions/eval_001/sample_001"
    index = EvaluationPrediction(
        schema_version="1.2.0",
        evaluation_run_id="eval_001",
        sample_id=sample.sample_id,
        media_sha256=sample.media_sha256,
        run_id=result.run_id,
        job_id="job_001",
        terminal_status="SUCCEEDED",
        run_relative_path=f"{prefix}/run.json",
        run_sha256=_digest(run_bytes),
        result_relative_path=f"{prefix}/result.json",
        result_sha256=_digest(result_bytes),
        evidence_relative_path=f"{prefix}/evidence.jsonl",
        evidence_sha256=_digest(evidence_bytes),
        document_relative_path=f"{prefix}/document.md",
        document_sha256=document.sha256,
        document_size_bytes=document.size_bytes,
        artifact_manifest_relative_path=f"{prefix}/artifact-manifest.json",
        artifact_manifest_sha256=_digest(manifest_bytes),
        transcript_source="ASR",
        started_at="2026-08-18T00:00:00Z",
        finished_at="2026-08-18T00:00:01Z",
    )
    index_path = directory / "index.json"
    index_path.write_bytes(index.model_dump_json(exclude_none=True).encode("utf-8"))
    return index_path, sample


def _kwargs(tmp_path: Path) -> dict[str, Path]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    return {
        "eval_root": runtime_root / "eval",
        "workspace_root": tmp_path,
        "runtime_root": runtime_root,
    }


def _refresh_index_digest(index_path: Path, field: str, artifact: Path) -> None:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload[field] = _digest(artifact.read_bytes())
    if field == "document_sha256":
        payload["document_size_bytes"] = artifact.stat().st_size
    index_path.write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_round_trip_uses_three_layer_versions_and_chapter_claims(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)

    prediction = load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)

    assert prediction.index.schema_version == "1.2.0"
    assert prediction.result is not None
    assert prediction.result.schema_version == "4.1.0"
    assert prediction.claims[0].source_kind == "CHAPTER_CLAIM"
    assert prediction.claims[0].source_id == "chapter_001"
    assert prediction.claims[0].evidence_refs == ("asr_001",)
    assert prediction.claims[0].certainty == 0.9
    assert reverify_verified_prediction(prediction, sample=sample) == prediction


@pytest.mark.parametrize("schema", ["1.0.0", "1.1.0", "3.0.0"])
def test_prediction_index_rejects_non_1_2_schema(tmp_path: Path, schema: str) -> None:
    index, sample = _write_prediction(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["schema_version"] = schema
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_failed_prediction_requires_all_success_artifacts_to_be_empty() -> None:
    base = {
        "schema_version": "1.2.0",
        "evaluation_run_id": "eval_001",
        "sample_id": "sample_001",
        "media_sha256": "a" * 64,
        "run_id": "run_001",
        "job_id": "job_001",
        "terminal_status": "FAILED",
        "run_relative_path": "predictions/eval_001/sample_001/run.json",
        "run_sha256": "b" * 64,
        "failure_code": "PIPELINE_FAILED",
        "started_at": "2026-08-18T00:00:00Z",
        "finished_at": "2026-08-18T00:00:01Z",
    }
    EvaluationPrediction(**base)

    with pytest.raises(ValidationError):
        EvaluationPrediction(
            **base,
            document_relative_path="predictions/eval_001/sample_001/document.md",
            document_sha256="c" * 64,
            document_size_bytes=10,
        )


@pytest.mark.parametrize("layer", ["envelope", "payload", "result"])
def test_prediction_rejects_each_tampered_bundle_version(
    tmp_path: Path,
    layer: str,
) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    envelope = json.loads(manifest.read_text(encoding="utf-8"))
    if layer == "envelope":
        envelope["schema_version"] = "3.0.0"
    elif layer == "payload":
        envelope["payload"]["artifact_schema_version"] = "2.0.0"
    else:
        envelope["payload"]["result"]["schema_version"] = "2.0.0"
    manifest.write_text(json.dumps(envelope), encoding="utf-8")
    _refresh_index_digest(index, "artifact_manifest_sha256", manifest)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("mutation", ["content", "index_digest", "size"])
def test_prediction_rejects_tampered_markdown(tmp_path: Path, mutation: str) -> None:
    index, sample = _write_prediction(tmp_path)
    document = index.parent / "document.md"
    if mutation == "content":
        document.write_text("# 伪造文档\n", encoding="utf-8")
        _refresh_index_digest(index, "document_sha256", document)
    else:
        payload = json.loads(index.read_text(encoding="utf-8"))
        if mutation == "index_digest":
            payload["document_sha256"] = "f" * 64
        else:
            payload["document_size_bytes"] += 1
        index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("mutation", ["missing", "extra", "png", "replace", "hardlink"])
def test_prediction_keyframe_copy_is_an_exact_jpeg_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    index, sample = _write_prediction(tmp_path)
    keyframe_root = index.parent / "visual" / "keyframes"
    path = next(keyframe_root.iterdir())
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (keyframe_root / f"{'f' * 64}.jpg").write_bytes(b"\xff\xd8\xffextra\xff\xd9")
    elif mutation == "png":
        path.rename(path.with_suffix(".png"))
    elif mutation == "replace":
        path.write_bytes(b"\xff\xd8\xffreplaced\xff\xd9")
    else:
        os.link(path, keyframe_root / "second-link.jpg")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_reverification_rejects_in_memory_claim_replacement(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    prediction = load_verified_prediction(index, **_kwargs(tmp_path), sample=sample)
    altered = prediction.model_copy(update={"claims": ()})

    with pytest.raises(VideoDemoError) as raised:
        reverify_verified_prediction(altered, sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
