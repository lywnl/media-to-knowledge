from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.dataset import EvaluationDataset


def _runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / ".codex" / "video-rag-demo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _eval_root(tmp_path: Path) -> Path:
    root = _runtime_root(tmp_path) / "eval"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_dataset(manifest: Path, tmp_path: Path) -> EvaluationDataset:
    return EvaluationDataset.load(
        manifest,
        workspace_root=tmp_path,
        runtime_root=_runtime_root(tmp_path),
    )


def test_dataset_rejects_external_plain_directory_before_reading(tmp_path: Path) -> None:
    runtime_root = _runtime_root(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    manifest = external / "dataset.jsonl"
    manifest.write_bytes(b"\xff")

    with pytest.raises(VideoDemoError) as raised:
        EvaluationDataset.load(manifest, workspace_root=tmp_path, runtime_root=runtime_root)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None
    assert str(external) not in raised.value.message


def test_dataset_rejects_noncanonical_runtime_under_workspace(tmp_path: Path) -> None:
    alternate_runtime = tmp_path / "other-runtime"
    alternate_runtime.mkdir()
    manifest = alternate_runtime / "dataset.jsonl"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        EvaluationDataset.load(
            manifest,
            workspace_root=tmp_path,
            runtime_root=alternate_runtime,
        )

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_final_gate_rejects_media_over_injected_limit(tmp_path: Path) -> None:
    runtime_root = _runtime_root(tmp_path)
    eval_root = runtime_root / "eval"
    manifest = eval_root / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)
    oversized = eval_root / "media" / "zh_0.mp4"
    oversized.write_bytes(b"12")
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    lines[0]["media_sha256"] = hashlib.sha256(b"12").hexdigest()
    manifest.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    dataset = EvaluationDataset.load(manifest, workspace_root=tmp_path, runtime_root=runtime_root)
    with pytest.raises(VideoDemoError) as raised:
        dataset.validate_final_gate(max_video_bytes=1)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


@pytest.mark.parametrize("max_video_bytes", [0, True])
def test_final_gate_rejects_invalid_media_limit(max_video_bytes: int, tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)

    with pytest.raises(ValueError, match="媒体大小上限必须是正整数"):
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=max_video_bytes)


def _write_dataset(path: Path, languages: tuple[str, ...], count_per_language: int) -> None:
    lines = []
    for language in languages:
        for index in range(count_per_language):
            media_relative_path = f"media/{language}_{index}.mp4"
            annotations_relative_path = f"annotations/{language}_{index}.json"
            media = path.parent / media_relative_path
            annotations = path.parent / annotations_relative_path
            media.parent.mkdir(parents=True, exist_ok=True)
            annotations.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(f"media-{language}-{index}".encode())
            annotations.write_text(f'{{"language":"{language}"}}', encoding="utf-8")
            lines.append(
                json.dumps(
                    {
                        "sample_id": f"{language}_{index}",
                        "language": language,
                        "authorization_id": "auth_001",
                        "media_relative_path": media_relative_path,
                        "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                        "annotations_relative_path": annotations_relative_path,
                        "annotations_sha256": hashlib.sha256(
                            annotations.read_bytes(),
                        ).hexdigest(),
                    },
                    ensure_ascii=False,
                ),
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_final_dataset_requires_30_samples_and_six_per_validation_language(
    tmp_path: Path,
) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)

    dataset = _load_dataset(manifest, tmp_path)

    assert len(dataset.samples) == 30
    dataset.validate_final_gate(max_video_bytes=1024)


def test_final_dataset_rejects_insufficient_language_coverage(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 5)

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=1024)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_dataset_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "language": "zh",
                "authorization_id": "auth_001",
                "media_relative_path": "../secret.mp4",
                "media_sha256": "a" * 64,
                "annotations_relative_path": "annotations/sample.json",
                "annotations_sha256": "b" * 64,
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_dataset_rejects_missing_authorization_id(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "language": "zh",
                "media_relative_path": "media/sample.mp4",
                "media_sha256": "a" * 64,
                "annotations_relative_path": "annotations/sample.json",
                "annotations_sha256": "b" * 64,
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


@pytest.mark.parametrize("content", [b"\xff", b""])
def test_dataset_rejects_non_utf8_or_empty_manifest(content: bytes, tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    manifest.write_bytes(content)

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_dataset_rejects_oversized_manifest(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    with manifest.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_final_dataset_rejects_missing_or_changed_artifacts(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)
    (manifest.parent / "media/zh_0.mp4").write_bytes(b"tampered")

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=1024)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def test_final_dataset_rejects_oversized_annotation_file(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)
    annotation = manifest.parent / "annotations/zh_0.json"
    with annotation.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    lines[0]["annotations_sha256"] = _sha256_file(annotation)
    manifest.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=1024)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_final_dataset_rejects_non_utf8_annotation_file(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)
    annotation = manifest.parent / "annotations/zh_0.json"
    annotation.write_bytes(b"\xff")
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    lines[0]["annotations_sha256"] = hashlib.sha256(b"\xff").hexdigest()
    manifest.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=1024)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID
    assert raised.value.__cause__ is None


def test_final_dataset_rejects_resolved_symlink_escape(tmp_path: Path) -> None:
    manifest = _eval_root(tmp_path) / "dataset.jsonl"
    _write_dataset(manifest, ("zh", "en", "ja", "ko", "es"), 6)
    escaped = tmp_path / "escaped-video.mp4"
    escaped.write_bytes(b"outside")
    media = manifest.parent / "media/zh_0.mp4"
    media.unlink()
    media.symlink_to(escaped)

    with pytest.raises(VideoDemoError) as raised:
        _load_dataset(manifest, tmp_path).validate_final_gate(max_video_bytes=1024)

    assert raised.value.code == ErrorCode.EVALUATION_DATASET_INVALID


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
