from __future__ import annotations

# ruff: noqa: RUF001
import hashlib
import json
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import (
    AuthorizationFile,
    EvaluationAnnotation,
    VerifiedAnnotation,
    load_evaluation_package,
    load_semantic_judgment,
)
from video_demo.evaluation.predictions import VerifiedPrediction


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / ".codex" / "video-rag-demo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _loader_kwargs(tmp_path: Path) -> dict[str, Path]:
    return {"workspace_root": tmp_path, "runtime_root": _runtime_root(tmp_path)}


def _annotation(media_sha256: str, *, duration_ms: int = 1_000) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sample_id": "sample_001",
        "media_sha256": media_sha256,
        "duration_ms": duration_ms,
        "language": "zh",
        "reference_text": "你好",
        "words": [{"word_id": "word_001", "text": "你", "start_ms": 0, "end_ms": 500}],
        "speaker_turns": [
            {"turn_id": "turn_001", "speaker_id": "speaker_001", "start_ms": 0, "end_ms": 500}
        ],
        "ocr_frames": [{"frame_id": "frame_001", "timestamp_ms": 100, "text_lines": ["你好"]}],
        "audio_events": [
            {
                "event_id": "event_001",
                "normalized_event": "speech",
                "start_ms": 0,
                "end_ms": 500,
            }
        ],
        "scene_boundaries_ms": [500],
        "semantic_boundaries_ms": [500],
        "supported_facts": [{"fact_id": "fact_001", "canonical_text": "有人问好"}],
        "key_fact_ids": ["fact_001"],
        "known_people": [{"person_id": "person_001", "allowed_names": ["张三"]}],
    }


def _write_package(tmp_path: Path) -> tuple[Path, Path, str]:
    eval_root = _runtime_root(tmp_path) / "eval"
    media = eval_root / "media" / "sample.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"real-media")
    media_sha256 = _sha256(media.read_bytes())
    annotation = eval_root / "annotations" / "sample.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(_annotation(media_sha256), ensure_ascii=False), encoding="utf-8"
    )
    dataset = eval_root / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "language": "zh",
                "authorization_id": "auth_001",
                "media_relative_path": "media/sample.mp4",
                "media_sha256": media_sha256,
                "annotations_relative_path": "annotations/sample.json",
                "annotations_sha256": _sha256(annotation.read_bytes()),
            },
        ),
        encoding="utf-8",
    )
    authorization = eval_root / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "records": [
                    {
                        "schema_version": "1.0.0",
                        "authorization_id": "auth_001",
                        "source_category": "OWNED",
                        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
                        "confirmed_at": "2026-08-18T00:00:00Z",
                        "media_sha256": [media_sha256],
                    }
                ],
            },
        ),
        encoding="utf-8",
    )
    return dataset, authorization, media_sha256


def test_annotation_contract_rejects_time_overrun() -> None:
    payload = _annotation("a" * 64)
    payload["words"][0]["end_ms"] = 1_001

    with pytest.raises(ValueError, match="标注时间"):
        EvaluationAnnotation.model_validate(payload)


def test_annotation_contract_rejects_duplicate_id() -> None:
    payload = _annotation("a" * 64)
    payload["speaker_turns"] = [
        {"turn_id": "turn_001", "speaker_id": "speaker_001", "start_ms": 0, "end_ms": 500},
        {"turn_id": "turn_001", "speaker_id": "speaker_002", "start_ms": 500, "end_ms": 800},
    ]

    with pytest.raises(ValueError, match="稳定 ID"):
        EvaluationAnnotation.model_validate(payload)


def test_annotation_contract_rejects_unknown_key_fact() -> None:
    payload = _annotation("a" * 64)
    payload["key_fact_ids"] = ["fact_unknown"]

    with pytest.raises(ValueError, match="关键事实"):
        EvaluationAnnotation.model_validate(payload)


def _authorization_record() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "authorization_id": "auth_001",
        "source_category": "OWNED",
        "allowed_purposes": ["VIDEO_QUALITY_EVALUATION"],
        "confirmed_at": "2026-08-18T00:00:00Z",
        "media_sha256": ["a" * 64],
    }


def test_authorization_contract_rejects_extra_field() -> None:
    record = _authorization_record()
    record["secret"] = "must-not-appear"

    with pytest.raises(ValueError):
        AuthorizationFile.model_validate(
            {
                "schema_version": "1.0.0",
                "records": [record],
            },
        )


def test_authorization_contract_rejects_empty_evaluation_purpose() -> None:
    record = _authorization_record()
    record["allowed_purposes"] = []

    with pytest.raises(ValueError, match="allowed_purposes"):
        AuthorizationFile.model_validate({"schema_version": "1.0.0", "records": [record]})


def test_package_loader_rejects_uncovered_media_and_does_not_leak_absolute_path(
    tmp_path: Path,
) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    payload["records"][0]["media_sha256"] = ["f" * 64]
    authorization.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert str(tmp_path) not in raised.value.message


def test_package_loader_rejects_external_plain_dataset_before_reading(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    dataset = external / "dataset.jsonl"
    authorization = external / "authorization.json"
    dataset.write_bytes(b"\xff")
    authorization.write_text("{}", encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None
    assert str(external) not in raised.value.message


def test_package_loader_rejects_symlink_in_any_path_component(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    media_root = _runtime_root(tmp_path) / "eval" / "media"
    media_root.rename(outside / "media")
    media_root.symlink_to(outside / "media", target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_package_loader_rejects_authorization_parent_symlink(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    authorization.rename(outside / "authorization.json")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(
            dataset, linked_parent / "authorization.json", **_loader_kwargs(tmp_path)
        )

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_package_loader_rejects_empty_media(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    (_runtime_root(tmp_path) / "eval" / "media" / "sample.mp4").write_bytes(b"")

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


@pytest.mark.parametrize("binding", ["annotation_sha", "media_sha", "language"])
def test_package_loader_rejects_annotation_binding_mismatch(binding: str, tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    annotation_path = _runtime_root(tmp_path) / "eval" / "annotations" / "sample.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if binding == "annotation_sha":
        payload["reference_text"] = "被篡改"
    elif binding == "media_sha":
        payload["media_sha256"] = "f" * 64
    else:
        payload["language"] = "en"
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None
    assert str(tmp_path) not in raised.value.message


def test_package_loader_rejects_media_sha_tampering(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    (_runtime_root(tmp_path) / "eval" / "media" / "sample.mp4").write_bytes(b"tampered-media")

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_reverify_package_rejects_in_memory_annotation_replacement(tmp_path: Path) -> None:
    from video_demo.evaluation.annotations import reverify_evaluation_package

    dataset, authorization, _ = _write_package(tmp_path)
    package = load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))
    annotation = package.annotations[0]
    altered_annotation = annotation.model_copy(
        update={
            "annotation": annotation.annotation.model_copy(
                update={"reference_text": "合法但未绑定的替换"}
            )
        }
    )
    altered = package.model_copy(update={"annotations": (altered_annotation,)})

    with pytest.raises(VideoDemoError) as raised:
        reverify_evaluation_package(altered)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None
    assert str(tmp_path) not in raised.value.message


def test_reverify_package_accepts_unmodified_loader_output(tmp_path: Path) -> None:
    from video_demo.evaluation.annotations import reverify_evaluation_package

    dataset, authorization, _ = _write_package(tmp_path)
    package = load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert reverify_evaluation_package(package) == package


@pytest.mark.parametrize("content", [b"\xff", b""])
def test_package_loader_rejects_non_utf8_or_empty_authorization(
    content: bytes, tmp_path: Path
) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    authorization.write_bytes(content)

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_package_loader_rejects_oversized_authorization(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    with authorization.open("r+b") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(dataset, authorization, **_loader_kwargs(tmp_path))

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def _annotation_for_judgment() -> VerifiedAnnotation:
    payload = _annotation("a" * 64, duration_ms=1_000)
    payload["supported_facts"] = [
        {"fact_id": "fact_key", "canonical_text": "关键事实"},
        {"fact_id": "fact_nonkey", "canonical_text": "非关键事实"},
    ]
    payload["key_fact_ids"] = ["fact_key"]
    payload["known_people"] = [
        {"person_id": "person_001", "allowed_names": ["Ａlice  Zhang", "张三"]}
    ]
    return VerifiedAnnotation(
        annotation=EvaluationAnnotation.model_validate(payload),
        sha256="b" * 64,
    )


def _prediction_for_judgment(eval_root: Path) -> VerifiedPrediction:
    from video_demo.evaluation.predictions import (
        EvaluationPrediction,
        PredictionClaim,
        PredictionRunSnapshot,
    )

    return VerifiedPrediction(
        index=EvaluationPrediction(
            schema_version="1.0.0",
            evaluation_run_id="eval_001",
            sample_id="sample_001",
            media_sha256="a" * 64,
            run_id="run_001",
            job_id="job_001",
            terminal_status="FAILED",
            run_relative_path="predictions/eval_001/sample_001/run.json",
            run_sha256="c" * 64,
            failure_code="PIPELINE_FAILED",
            started_at="2026-08-18T00:00:00Z",
            finished_at="2026-08-18T00:00:01Z",
        ),
        index_sha256="d" * 64,
        run=PredictionRunSnapshot(
            schema_version="1.0.0",
            run_id="run_001",
            job_id="job_001",
            terminal_status="FAILED",
            current_stage="RESULT",
            error_code="PIPELINE_FAILED",
            models=({"component": "asr", "provider": "local", "model_id": "model_001"},),
        ),
        result=None,
        evidence=(),
        claims=(
            PredictionClaim(
                claim_id="claim_001",
                source_kind="VIDEO_SUMMARY",
                source_id="run_001",
                text="摘要",
            ),
        ),
        artifact_manifest_sha256=None,
        eval_root=eval_root,
    )


def _write_valid_judgment(path: Path, prediction_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "sample_id": "sample_001",
                "annotation_sha256": "b" * 64,
                "prediction_sha256": prediction_sha256,
                "claim_judgments": [{"claim_id": "claim_001", "verdict": "SUPPORTED"}],
                "matched_key_fact_ids": ["fact_key"],
                "fabricated_names": [],
                "reviewer_id": "reviewer_001",
                "reviewed_at": "2026-08-18T00:02:00Z",
                "rubric_version": "1.0.0",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_judgment_rejects_supported_but_non_key_fact(tmp_path: Path) -> None:
    prediction = _prediction_for_judgment(_runtime_root(tmp_path) / "eval")
    judgment = _runtime_root(tmp_path) / "eval" / "judgment.json"
    _write_valid_judgment(judgment, prediction.index_sha256)
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    payload["matched_key_fact_ids"] = ["fact_nonkey"]
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_semantic_judgment(
            judgment,
            **_loader_kwargs(tmp_path),
            annotation=_annotation_for_judgment(),
            prediction=prediction,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_judgment_rejects_normalized_known_person_alias(tmp_path: Path) -> None:
    prediction = _prediction_for_judgment(_runtime_root(tmp_path) / "eval")
    judgment = _runtime_root(tmp_path) / "eval" / "judgment.json"
    _write_valid_judgment(judgment, prediction.index_sha256)
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    payload["fabricated_names"] = ["alice   zhang"]
    judgment.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        load_semantic_judgment(
            judgment,
            **_loader_kwargs(tmp_path),
            annotation=_annotation_for_judgment(),
            prediction=prediction,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID


def test_judgment_rejects_normalized_duplicate_fabricated_names() -> None:
    with pytest.raises(ValueError, match="虚构姓名"):
        from video_demo.evaluation.annotations import SemanticJudgment

        SemanticJudgment.model_validate(
            {
                "schema_version": "1.0.0",
                "sample_id": "sample_001",
                "annotation_sha256": "a" * 64,
                "prediction_sha256": "b" * 64,
                "claim_judgments": [],
                "matched_key_fact_ids": [],
                "fabricated_names": ["Ａlice Zhang", "alice   zhang"],
                "reviewer_id": "reviewer_001",
                "reviewed_at": "2026-08-18T00:00:00Z",
                "rubric_version": "1.0.0",
            }
        )


def test_judgment_rejects_blank_fabricated_name() -> None:
    from video_demo.evaluation.annotations import SemanticJudgment

    payload = {
        "schema_version": "1.0.0",
        "sample_id": "sample_001",
        "annotation_sha256": "a" * 64,
        "prediction_sha256": "b" * 64,
        "claim_judgments": [],
        "matched_key_fact_ids": [],
        "fabricated_names": ["   "],
        "reviewer_id": "reviewer_001",
        "reviewed_at": "2026-08-18T00:00:00Z",
        "rubric_version": "1.0.0",
    }

    with pytest.raises(ValueError, match="虚构姓名"):
        SemanticJudgment.model_validate(payload)


def test_package_loader_rejects_video_over_configured_limit(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)

    with pytest.raises(VideoDemoError) as raised:
        load_evaluation_package(
            dataset, authorization, **_loader_kwargs(tmp_path), max_video_bytes=1
        )

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_package_loader_rejects_non_positive_or_boolean_video_limit(tmp_path: Path) -> None:
    dataset, authorization, _ = _write_package(tmp_path)
    for invalid_limit in (0, True):
        with pytest.raises(VideoDemoError) as raised:
            load_evaluation_package(
                dataset, authorization, **_loader_kwargs(tmp_path), max_video_bytes=invalid_limit
            )

        assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_judgment_allows_truly_unknown_fabricated_name(tmp_path: Path) -> None:
    prediction = _prediction_for_judgment(_runtime_root(tmp_path) / "eval")
    judgment = _runtime_root(tmp_path) / "eval" / "judgment.json"
    _write_valid_judgment(judgment, prediction.index_sha256)
    payload = json.loads(judgment.read_text(encoding="utf-8"))
    payload["fabricated_names"] = ["未知人物"]
    judgment.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    actual = load_semantic_judgment(
        judgment,
        **_loader_kwargs(tmp_path),
        annotation=_annotation_for_judgment(),
        prediction=prediction,
    )

    assert actual.fabricated_names == ("未知人物",)


def test_judgment_rejects_handmade_prediction_with_external_eval_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    prediction = _prediction_for_judgment(external)
    judgment = external / "judgment.json"
    _write_valid_judgment(judgment, prediction.index_sha256)

    with pytest.raises(VideoDemoError) as raised:
        load_semantic_judgment(
            judgment,
            **_loader_kwargs(tmp_path),
            annotation=_annotation_for_judgment(),
            prediction=prediction,
        )

    assert raised.value.code == ErrorCode.EVALUATION_ARTIFACT_INVALID
    assert raised.value.__cause__ is None
    assert str(external) not in raised.value.message
