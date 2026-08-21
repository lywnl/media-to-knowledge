from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from video_demo.domain.evidence import EvidenceItem
from video_demo.domain.result import VideoUnderstandingResult
from video_demo.domain.result_artifact import ResultArtifactPayload
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import VerifiedAnnotation, load_semantic_judgment
from video_demo.evaluation.dataset import EvaluationSample
from video_demo.evaluation.predictions import EvaluationPrediction, load_verified_prediction
from video_demo.storage.artifacts import AtomicArtifactStore

_EVIDENCE_ADAPTER = TypeAdapter(tuple[EvidenceItem, ...])


def _prediction_kwargs(tmp_path: Path) -> dict[str, Path]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    return {
        "eval_root": runtime_root / "eval",
        "workspace_root": tmp_path,
        "runtime_root": runtime_root,
    }


def _prediction_root_kwargs(tmp_path: Path) -> dict[str, Path]:
    values = _prediction_kwargs(tmp_path)
    del values["eval_root"]
    return values


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(media_sha256: str) -> dict[str, object]:
    segment_text = "标题：问候"
    summary_text = "标题：视频"
    return {
        "schema_version": "1.0.0",
        "run_id": "run_001",
        "asset_sha256": media_sha256,
        "segments": [
            {
                "result_type": "VIDEO_SEGMENT",
                "segment_id": "seg_001",
                "start_ms": 0,
                "end_ms": 500,
                "title": "问候",
                "summary_zh": "讲者问好",
                "speakers": ["SPEAKER_01"],
                "languages": ["zh"],
                "topics": [],
                "entities": [],
                "actions": [],
                "keywords": [],
                "original_keywords": [],
                "evidence_refs": ["ev_001"],
                "retrieval_text": segment_text,
                "retrieval_hash": hashlib.sha256(segment_text.encode()).hexdigest(),
            }
        ],
        "summary": {
            "result_type": "VIDEO_SUMMARY",
            "title": "视频",
            "duration_ms": 500,
            "summary_zh": "视频问好",
            "chapters": [
                {"title": "问候", "start_ms": 0, "end_ms": 500, "segment_ids": ["seg_001"]}
            ],
            "speakers": ["SPEAKER_01"],
            "languages": ["zh"],
            "topics": [],
            "entities": [],
            "actions": [],
            "keywords": [],
            "original_keywords": [],
            "retrieval_text": summary_text,
            "retrieval_hash": hashlib.sha256(summary_text.encode()).hexdigest(),
        },
    }


def _evidence() -> dict[str, object]:
    return {
        "evidence_type": "ASR_SEGMENT",
        "evidence_id": "ev_001",
        "start_ms": 0,
        "end_ms": 500,
        "text": "你好",
        "language": "zh",
        "confidence": 0.9,
        "is_fully_evaluated_language": True,
    }


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": "run_001",
        "job_id": "job_001",
        "terminal_status": "SUCCEEDED",
        "current_stage": "RESULT",
        "warning_codes": [],
        "error_code": None,
        "models": [{"component": "asr", "provider": "local", "model_id": "model_001"}],
    }


def _write_prediction(tmp_path: Path) -> tuple[Path, EvaluationSample]:
    root = tmp_path / ".codex" / "video-rag-demo" / "eval"
    prediction_dir = root / "predictions" / "eval_001" / "sample_001"
    prediction_dir.mkdir(parents=True)
    media_sha256 = "a" * 64
    result = prediction_dir / "result.json"
    result.write_text(json.dumps(_result(media_sha256)), encoding="utf-8")
    evidence = prediction_dir / "evidence.jsonl"
    evidence.write_text(json.dumps(_evidence()) + "\n", encoding="utf-8")
    run = prediction_dir / "run.json"
    run.write_text(json.dumps(_snapshot()), encoding="utf-8")
    manifest = prediction_dir / "artifact-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "upstream_sha256": media_sha256,
                "payload": {
                    "result": _result(media_sha256),
                    "evidence": [_evidence()],
                    "stage_metrics": {"RESULT": 1},
                    "status": "SUCCEEDED",
                    "warnings": [],
                    "transcript_source": "ASR",
                },
            }
        ),
        encoding="utf-8",
    )
    index = prediction_dir / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "evaluation_run_id": "eval_001",
                "sample_id": "sample_001",
                "media_sha256": media_sha256,
                "run_id": "run_001",
                "job_id": "job_001",
                "terminal_status": "SUCCEEDED",
                "run_relative_path": "predictions/eval_001/sample_001/run.json",
                "run_sha256": _digest(run),
                "result_relative_path": "predictions/eval_001/sample_001/result.json",
                "result_sha256": _digest(result),
                "evidence_relative_path": "predictions/eval_001/sample_001/evidence.jsonl",
                "evidence_sha256": _digest(evidence),
                "artifact_manifest_relative_path": (
                    "predictions/eval_001/sample_001/artifact-manifest.json"
                ),
                "artifact_manifest_sha256": _digest(manifest),
                "failure_code": None,
                "transcript_source": "ASR",
                "started_at": "2026-08-18T00:00:00Z",
                "finished_at": "2026-08-18T00:00:01Z",
            }
        ),
        encoding="utf-8",
    )
    return index, EvaluationSample(
        sample_id="sample_001",
        language="zh",
        authorization_id="auth_001",
        media_relative_path="media/sample.mp4",
        media_sha256=media_sha256,
        annotations_relative_path="annotations/sample.json",
        annotations_sha256="b" * 64,
    )


def _replace_with_atomic_production_manifest(index: Path) -> None:
    root = index.parents[3]
    relative = index.parent.relative_to(root) / "artifact-manifest.json"
    media_sha256 = "a" * 64
    result = VideoUnderstandingResult.model_validate(_result(media_sha256))
    evidence = _EVIDENCE_ADAPTER.validate_python([_evidence()])
    payload = ResultArtifactPayload(
        result=result,
        evidence=evidence,
        stage_metrics={"RESULT": 1},
        status="SUCCEEDED",
        warnings=(),
        transcript_source="ASR",
    )
    receipt = AtomicArtifactStore(root).write_json(
        relative,
        payload.model_dump(mode="json", exclude_computed_fields=True),
        schema_version="1.0.0",
        upstream_sha256=media_sha256,
    )
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    index_payload["artifact_manifest_sha256"] = receipt.sha256
    index.write_text(json.dumps(index_payload), encoding="utf-8")


def _update_manifest_digest(index: Path) -> None:
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["artifact_manifest_sha256"] = _digest(index.parent / "artifact-manifest.json")
    index.write_text(json.dumps(payload), encoding="utf-8")


def test_verified_prediction_reparses_production_artifacts_and_derives_stable_claims(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)

    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert [claim.source_kind for claim in prediction.claims] == [
        "SEGMENT_SUMMARY",
        "VIDEO_SUMMARY",
    ]
    assert prediction.result is not None


@pytest.mark.parametrize("mutation", ["legacy_schema", "missing_source"])
def test_prediction_index_rejects_legacy_schema_or_missing_transcript_source(
    mutation: str,
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    if mutation == "legacy_schema":
        payload["schema_version"] = "1.0.0"
    else:
        payload.pop("transcript_source")
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None


def test_failed_prediction_contract_rejects_transcript_source() -> None:
    with pytest.raises(ValidationError):
        EvaluationPrediction(
            schema_version="1.1.0",
            evaluation_run_id="eval_001",
            sample_id="sample_001",
            media_sha256="a" * 64,
            run_id="run_001",
            job_id="job_001",
            terminal_status="FAILED",
            run_relative_path="predictions/eval_001/sample_001/run.json",
            run_sha256="b" * 64,
            failure_code="PIPELINE_FAILED",
            transcript_source="ASR",
            started_at="2026-08-18T00:00:00Z",
            finished_at="2026-08-18T00:00:01Z",
        )


def test_verified_prediction_rejects_index_and_manifest_transcript_source_mismatch(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["transcript_source"] = "SUBTITLE"
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("mutation", ["word", "ocr", "event", "summary", "claims", "run"])
def test_reverify_prediction_rejects_schema_valid_in_memory_replacement(
    mutation: str,
    tmp_path: Path,
) -> None:
    from video_demo.domain.evidence import (
        AlignedWord,
        AudioEvent,
        BoundingBox,
        OcrEvidence,
        OcrLine,
    )
    from video_demo.evaluation.predictions import PredictionClaim, reverify_verified_prediction

    index, sample = _write_prediction(tmp_path)
    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)
    assert prediction.result is not None
    if mutation == "word":
        altered = prediction.model_copy(
            update={
                "evidence": (
                    AlignedWord(
                        evidence_id="word_altered",
                        start_ms=0,
                        end_ms=100,
                        text="合法替换",
                        language="zh",
                        probability=0.9,
                    ),
                )
            }
        )
    elif mutation == "ocr":
        altered = prediction.model_copy(
            update={
                "evidence": (
                    OcrEvidence(
                        evidence_id="ocr_altered",
                        start_ms=0,
                        end_ms=100,
                        keyframe_id="key_001",
                        timestamp_ms=50,
                        language="zh",
                        lines=(
                            OcrLine(
                                text="合法替换",
                                bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
                                confidence=0.9,
                            ),
                        ),
                        provider_request_id="request-001",
                    ),
                )
            }
        )
    elif mutation == "event":
        altered = prediction.model_copy(
            update={
                "evidence": (
                    AudioEvent(
                        evidence_id="event_altered",
                        start_ms=0,
                        end_ms=100,
                        audioset_class="Music",
                        normalized_event="music",
                        confidence=0.9,
                        threshold_version="v1",
                    ),
                )
            }
        )
    elif mutation == "summary":
        summary = prediction.result.summary.model_copy(update={"summary_zh": "合法替换"})
        altered = prediction.model_copy(
            update={"result": prediction.result.model_copy(update={"summary": summary})}
        )
    elif mutation == "claims":
        altered = prediction.model_copy(
            update={
                "claims": (
                    PredictionClaim(
                        claim_id="claim_altered",
                        source_kind="VIDEO_SUMMARY",
                        source_id=prediction.result.run_id,
                        text="合法替换",
                    ),
                )
            }
        )
    else:
        altered = prediction.model_copy(
            update={"run": prediction.run.model_copy(update={"current_stage": "ALTERED"})}
        )

    with pytest.raises(VideoDemoError) as raised:
        reverify_verified_prediction(altered, sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None
    assert str(tmp_path) not in raised.value.message


def test_reverify_prediction_accepts_unmodified_loader_output(tmp_path: Path) -> None:
    from video_demo.evaluation.predictions import reverify_verified_prediction

    index, sample = _write_prediction(tmp_path)
    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert reverify_verified_prediction(prediction, sample=sample) == prediction


def test_verified_prediction_rejects_external_plain_eval_root(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    external = tmp_path / "external-eval"
    external.mkdir()

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(
            index,
            eval_root=external,
            workspace_root=tmp_path,
            runtime_root=tmp_path / ".codex" / "video-rag-demo",
            sample=sample,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None
    assert str(external) not in raised.value.message


def test_verified_prediction_accepts_complete_atomic_artifact_store_envelope(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    _replace_with_atomic_production_manifest(index)

    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert prediction.result is not None


def test_verified_prediction_rejects_tampered_manifest_binding_and_failed_result(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["payload"]["status"] = "FAILED"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    index_payload["artifact_manifest_sha256"] = _digest(manifest)
    index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_manifest_missing_production_fields(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["payload"]["stage_metrics"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    index_payload["artifact_manifest_sha256"] = _digest(manifest)
    index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize(
    "field",
    ["result", "evidence", "stage_metrics", "status", "warnings", "transcript_source"],
)
def test_verified_prediction_rejects_each_missing_manifest_payload_field(
    field: str,
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["payload"][field]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _update_manifest_digest(index)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None


def test_verified_prediction_rejects_extra_manifest_payload_field(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["payload"]["forged"] = "value"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _update_manifest_digest(index)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_manifest_warning_mismatch(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["payload"]["warnings"] = ["WINDOW_FAILED"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    index_payload["artifact_manifest_sha256"] = _digest(manifest)
    index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_manifest_bool_stage_metric(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    manifest = index.parent / "artifact-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["payload"]["stage_metrics"] = {"RESULT": True}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    index_payload = json.loads(index.read_text(encoding="utf-8"))
    index_payload["artifact_manifest_sha256"] = _digest(manifest)
    index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("content", [b"\xff", b""])
def test_verified_prediction_rejects_non_utf8_or_empty_index(
    content: bytes, tmp_path: Path
) -> None:
    index, sample = _write_prediction(tmp_path)
    index.write_bytes(content)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None


def test_verified_prediction_rejects_oversized_index(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    with index.open("r+b") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize(
    "mutation", ["result_sha", "result_media", "result_run", "run_job", "run_status"]
)
def test_verified_prediction_rejects_artifact_identity_mismatch(
    mutation: str, tmp_path: Path
) -> None:
    index, sample = _write_prediction(tmp_path)
    if mutation.startswith("result"):
        result = index.parent / "result.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        if mutation == "result_sha":
            payload["summary"]["summary_zh"] = "篡改"
        elif mutation == "result_media":
            payload["asset_sha256"] = "f" * 64
        else:
            payload["run_id"] = "run_999"
        result.write_text(json.dumps(payload), encoding="utf-8")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_payload["result_sha256"] = _digest(result)
        index.write_text(json.dumps(index_payload), encoding="utf-8")
    else:
        run = index.parent / "run.json"
        payload = json.loads(run.read_text(encoding="utf-8"))
        payload["job_id" if mutation == "run_job" else "terminal_status"] = (
            "job_999" if mutation == "run_job" else "PARTIAL_SUCCEEDED"
        )
        run.write_text(json.dumps(payload), encoding="utf-8")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_payload["run_sha256"] = _digest(run)
        index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("mutation", ["retrieval_hash", "unknown_evidence", "duplicate_evidence"])
def test_verified_prediction_rejects_result_or_evidence_integrity(
    mutation: str, tmp_path: Path
) -> None:
    index, sample = _write_prediction(tmp_path)
    if mutation == "retrieval_hash":
        result = index.parent / "result.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["segments"][0]["retrieval_hash"] = "f" * 64
        result.write_text(json.dumps(payload), encoding="utf-8")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_payload["result_sha256"] = _digest(result)
        index.write_text(json.dumps(index_payload), encoding="utf-8")
    elif mutation == "unknown_evidence":
        result = index.parent / "result.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["segments"][0]["evidence_refs"] = ["ev_unknown"]
        result.write_text(json.dumps(payload), encoding="utf-8")
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_payload["result_sha256"] = _digest(result)
        index.write_text(json.dumps(index_payload), encoding="utf-8")
    else:
        evidence = index.parent / "evidence.jsonl"
        evidence.write_text(
            json.dumps(_evidence()) + "\n" + json.dumps(_evidence()) + "\n",
            encoding="utf-8",
        )
        index_payload = json.loads(index.read_text(encoding="utf-8"))
        index_payload["evidence_sha256"] = _digest(evidence)
        index.write_text(json.dumps(index_payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_symlink_eval_root_before_resolution(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    outside = tmp_path / "outside"
    (tmp_path / ".codex" / "video-rag-demo" / "eval").rename(outside)
    linked_root = tmp_path / ".codex" / "video-rag-demo" / "eval"
    linked_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(
            linked_root / index.relative_to(tmp_path / ".codex" / "video-rag-demo" / "eval"),
            eval_root=linked_root,
            workspace_root=tmp_path,
            runtime_root=tmp_path / ".codex" / "video-rag-demo",
            sample=sample,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_symlink_root_with_real_target_index_path(
    tmp_path: Path,
) -> None:
    index, sample = _write_prediction(tmp_path)
    real_root = tmp_path / ".codex" / "video-rag-demo" / "eval"
    linked_root = tmp_path / "linked-eval"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(
            index, eval_root=linked_root, **_prediction_root_kwargs(tmp_path), sample=sample
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_symlink_index(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    link = index.with_name("linked-index.json")
    link.symlink_to(index)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(link, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_verified_prediction_rejects_index_directory(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index.parent, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("relative_path", ["/tmp/result.json", "../result.json"])
def test_prediction_contract_rejects_unsafe_result_path(relative_path: str, tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["result_relative_path"] = relative_path
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_prediction_contract_rejects_failed_terminal_with_success_artifact(
    tmp_path: Path,
) -> None:
    index, _ = _write_prediction(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["terminal_status"] = "FAILED"
    payload["failure_code"] = "PIPELINE_FAILED"
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_verified_prediction(
            index,
            eval_root=tmp_path / ".codex" / "video-rag-demo" / "eval",
            workspace_root=tmp_path,
            runtime_root=tmp_path / ".codex" / "video-rag-demo",
            sample=EvaluationSample(
                sample_id="sample_001",
                language="zh",
                authorization_id="auth_001",
                media_relative_path="media/sample.mp4",
                media_sha256="a" * 64,
                annotations_relative_path="annotations/sample.json",
                annotations_sha256="b" * 64,
            ),
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def _verified_annotation_for_prediction_test() -> VerifiedAnnotation:
    return VerifiedAnnotation.model_validate(
        {
            "annotation": {
                "schema_version": "1.0.0",
                "sample_id": "sample_001",
                "media_sha256": "a" * 64,
                "duration_ms": 500,
                "language": "zh",
                "reference_text": "你好",
                "words": [{"word_id": "word_001", "text": "你", "start_ms": 0, "end_ms": 100}],
                "speaker_turns": [
                    {
                        "turn_id": "turn_001",
                        "speaker_id": "speaker_001",
                        "start_ms": 0,
                        "end_ms": 100,
                    }
                ],
                "ocr_frames": [
                    {"frame_id": "frame_001", "timestamp_ms": 10, "text_lines": ["你好"]}
                ],
                "audio_events": [
                    {
                        "event_id": "event_001",
                        "normalized_event": "speech",
                        "start_ms": 0,
                        "end_ms": 100,
                    }
                ],
                "scene_boundaries_ms": [100],
                "semantic_boundaries_ms": [100],
                "supported_facts": [{"fact_id": "fact_001", "canonical_text": "问好"}],
                "key_fact_ids": ["fact_001"],
                "known_people": [],
            },
            "sha256": "b" * 64,
        }
    )


def _write_valid_prediction_judgment(
    path: Path, prediction_sha256: str, claim_ids: list[str]
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "sample_id": "sample_001",
                "annotation_sha256": "b" * 64,
                "prediction_sha256": prediction_sha256,
                "claim_judgments": [
                    {"claim_id": claim_id, "verdict": "SUPPORTED"} for claim_id in claim_ids
                ],
                "matched_key_fact_ids": ["fact_001"],
                "fabricated_names": [],
                "reviewer_id": "reviewer_001",
                "reviewed_at": "2026-08-18T00:02:00Z",
                "rubric_version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )


def test_semantic_judgment_rejects_stale_prediction_sha(tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)
    judgment = tmp_path / ".codex" / "video-rag-demo" / "eval" / "judgment.json"
    _write_valid_prediction_judgment(
        judgment, "f" * 64, [claim.claim_id for claim in prediction.claims]
    )

    with pytest.raises(VideoDemoError) as raised:
        load_semantic_judgment(
            judgment,
            workspace_root=tmp_path,
            runtime_root=tmp_path / ".codex" / "video-rag-demo",
            annotation=_verified_annotation_for_prediction_test(),
            prediction=prediction,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_semantic_judgment_rejects_non_exact_claim_coverage(change: str, tmp_path: Path) -> None:
    index, sample = _write_prediction(tmp_path)
    prediction = load_verified_prediction(index, **_prediction_kwargs(tmp_path), sample=sample)
    judgment = tmp_path / ".codex" / "video-rag-demo" / "eval" / "judgment.json"
    claim_ids = [claim.claim_id for claim in prediction.claims]
    if change == "missing":
        claim_ids.pop()
    elif change == "extra":
        claim_ids.append("claim_extra")
    else:
        claim_ids.append(claim_ids[0])
    _write_valid_prediction_judgment(judgment, prediction.index_sha256, claim_ids)

    with pytest.raises(VideoDemoError) as raised:
        load_semantic_judgment(
            judgment,
            workspace_root=tmp_path,
            runtime_root=tmp_path / ".codex" / "video-rag-demo",
            annotation=_verified_annotation_for_prediction_test(),
            prediction=prediction,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
