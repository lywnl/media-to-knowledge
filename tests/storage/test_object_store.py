from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.repositories import Scope
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.object_store import LocalVideoObjectStore, VideoObjectRecord

VIDEO_SAMPLES = {
    "lesson.mp4": ("video/mp4", b"\x00\x00\x00\x18ftypisom" + b"m" * 128),
    "lesson.mov": ("video/quicktime", b"\x00\x00\x00\x18ftypqt  " + b"q" * 128),
    "lesson.mkv": ("video/x-matroska", b"\x1aE\xdf\xa3matroska" + b"m" * 128),
    "lesson.webm": ("video/webm", b"\x1aE\xdf\xa3webm" + b"w" * 128),
}


@pytest.fixture
def scope() -> Scope:
    return Scope("tenant-a", "app-a", "kb-a")


@pytest.fixture
def store(tmp_path: Path) -> LocalVideoObjectStore:
    return LocalVideoObjectStore(tmp_path / ".codex" / "video-rag-demo", max_video_bytes=1024)


@pytest.mark.parametrize(("filename", "sample"), VIDEO_SAMPLES.items())
def test_ingest_accepts_supported_container_and_calculates_sha256(
    store: LocalVideoObjectStore,
    scope: Scope,
    filename: str,
    sample: tuple[str, bytes],
) -> None:
    declared_mime, content = sample

    record = store.ingest(io.BytesIO(content), filename, declared_mime, scope)

    assert record.detected_mime == declared_mime
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.size_bytes == len(content)
    assert not Path(record.relative_path).is_absolute()
    assert store.resolve_record_path(record).read_bytes() == content


def test_ingest_rejects_extension_and_declared_mime_mismatch(
    store: LocalVideoObjectStore,
    scope: Scope,
) -> None:
    _, content = VIDEO_SAMPLES["lesson.webm"]

    with pytest.raises(VideoDemoError) as raised:
        store.ingest(io.BytesIO(content), "lesson.mp4", "video/mp4", scope)

    assert raised.value.code == ErrorCode.VIDEO_MIME_MISMATCH


@pytest.mark.parametrize("filename", ["../lesson.mp4", "/tmp/lesson.mp4", "folder/lesson.mp4"])
def test_ingest_rejects_client_filename_paths(
    store: LocalVideoObjectStore,
    scope: Scope,
    filename: str,
) -> None:
    mime, content = VIDEO_SAMPLES["lesson.mp4"]

    with pytest.raises(VideoDemoError) as raised:
        store.ingest(io.BytesIO(content), filename, mime, scope)

    assert raised.value.code == ErrorCode.INVALID_VIDEO_FILENAME


def test_ingest_rejects_file_over_size_limit(
    tmp_path: Path,
    scope: Scope,
) -> None:
    store = LocalVideoObjectStore(tmp_path / "runtime", max_video_bytes=16)
    mime, content = VIDEO_SAMPLES["lesson.mp4"]

    with pytest.raises(VideoDemoError) as raised:
        store.ingest(io.BytesIO(content), "lesson.mp4", mime, scope)

    assert raised.value.code == ErrorCode.VIDEO_FILE_TOO_LARGE


def test_materialize_rejects_wrong_scope_and_digest(
    store: LocalVideoObjectStore,
    scope: Scope,
) -> None:
    mime, content = VIDEO_SAMPLES["lesson.mp4"]
    record = store.ingest(io.BytesIO(content), "lesson.mp4", mime, scope)

    with pytest.raises(VideoDemoError) as wrong_scope:
        store.materialize(Scope("tenant-b", "app-a", "kb-a"), record, "run_001", record.sha256)
    assert wrong_scope.value.code == ErrorCode.VIDEO_OBJECT_NOT_FOUND

    with pytest.raises(VideoDemoError) as wrong_digest:
        store.materialize(scope, record, "run_001", "f" * 64)
    assert wrong_digest.value.code == ErrorCode.VIDEO_DIGEST_MISMATCH


def test_materialize_isolates_same_run_id_by_scope(
    store: LocalVideoObjectStore,
    scope: Scope,
) -> None:
    other_scope = Scope("tenant-b", "app-a", "kb-a")
    mime, first_content = VIDEO_SAMPLES["lesson.mp4"]
    _, second_content = VIDEO_SAMPLES["lesson.mov"]
    first = store.ingest(io.BytesIO(first_content), "lesson.mp4", mime, scope)
    second = store.ingest(
        io.BytesIO(second_content),
        "lesson.mov",
        "video/quicktime",
        other_scope,
    )

    first_path = store.materialize(scope, first, "run_shared", first.sha256)
    second_path = store.materialize(other_scope, second, "run_shared", second.sha256)

    assert first_path == (
        store.runtime_root / "runs" / store.scope_key(scope) / "run_shared/input/source.mp4"
    )
    assert second_path == (
        store.runtime_root
        / "runs"
        / store.scope_key(other_scope)
        / "run_shared/input/source.mov"
    )
    assert first_path.read_bytes() == first_content
    assert second_path.read_bytes() == second_content


def test_materialize_rejects_symlink_source_escape(
    store: LocalVideoObjectStore,
    scope: Scope,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(VIDEO_SAMPLES["lesson.mp4"][1])
    record = VideoObjectRecord(
        object_ref="obj_symlink",
        original_filename="lesson.mp4",
        declared_mime="video/mp4",
        detected_mime="video/mp4",
        size_bytes=outside.stat().st_size,
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        relative_path="objects/escape/source.mp4",
        scope_key=store.scope_key(scope),
    )
    source = store.runtime_root / record.relative_path
    source.parent.mkdir(parents=True)
    source.symlink_to(outside)

    with pytest.raises(VideoDemoError) as raised:
        store.materialize(scope, record, "run_001", record.sha256)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


def test_atomic_artifact_detects_content_corruption(tmp_path: Path) -> None:
    artifact_store = AtomicArtifactStore(tmp_path)
    receipt = artifact_store.write_json(
        Path("runs/run_001/probe/ffprobe.json"),
        {"format": "mp4"},
        schema_version="1.0.0",
        upstream_sha256="a" * 64,
    )
    payload = artifact_store.read_verified_json(receipt)
    assert payload == {"format": "mp4"}

    artifact_path = tmp_path / receipt.relative_path
    artifact_path.write_text(json.dumps({"format": "webm"}), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        artifact_store.read_verified_json(receipt)

    assert raised.value.code == ErrorCode.ARTIFACT_DIGEST_MISMATCH
