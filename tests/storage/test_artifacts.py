from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

import video_demo.storage.artifacts as artifacts_module
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.artifacts import ArtifactBytesReceipt, AtomicArtifactStore


def test_bytes_artifact_round_trip_receipt_permissions_and_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    store = AtomicArtifactStore(runtime_root)
    content = "确定性文档\n".encode()
    fsynced_modes: list[int] = []
    original_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(artifacts_module.os, "fsync", tracked_fsync)

    receipt = store.write_bytes(
        Path("runs/run_001/documents/result.md"),
        content,
        max_bytes=1_024,
    )
    artifact = runtime_root / receipt.relative_path

    assert receipt == ArtifactBytesReceipt(
        relative_path="runs/run_001/documents/result.md",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    assert store.read_verified_bytes(receipt, max_bytes=1_024) == content
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert stat.S_IMODE(artifact.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.parent.parent.stat().st_mode) == 0o700
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_open_runtime_directory_allows_entry_changes_after_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    original_open = os.open
    changed = False

    def change_directory_after_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal changed
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == runtime_root and not changed:
            (runtime_root / "concurrent-directory").mkdir()
            changed = True
        return descriptor

    monkeypatch.setattr(artifacts_module.os, "open", change_directory_after_open)

    receipt = AtomicArtifactStore(runtime_root).write_bytes(
        Path("runs/run_001/result.md"),
        b"document",
        max_bytes=100,
    )

    assert changed is True
    assert (runtime_root / receipt.relative_path).read_bytes() == b"document"


@pytest.mark.parametrize("max_bytes", (0, -1))
def test_bytes_artifact_requires_positive_limits(tmp_path: Path, max_bytes: int) -> None:
    store = AtomicArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="大于 0"):
        store.write_bytes(Path("result.md"), b"x", max_bytes=max_bytes)
    with pytest.raises(ValueError, match="大于 0"):
        store.read_verified_bytes(
            ArtifactBytesReceipt(
                relative_path="result.md",
                sha256=hashlib.sha256(b"x").hexdigest(),
                size_bytes=1,
            ),
            max_bytes=max_bytes,
        )


@pytest.mark.parametrize("relative_path", ("../escape.md", "/absolute.md", ".", "bad\x00.md"))
def test_bytes_receipt_requires_safe_relative_path(relative_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactBytesReceipt(
            relative_path=relative_path,
            sha256=hashlib.sha256(b"x").hexdigest(),
            size_bytes=1,
        )


def test_bytes_artifact_rejects_empty_or_oversized_content(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)

    with pytest.raises(VideoDemoError) as empty:
        store.write_bytes(Path("empty.md"), b"", max_bytes=10)
    assert empty.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID

    with pytest.raises(VideoDemoError) as oversized:
        store.write_bytes(Path("large.md"), b"12345", max_bytes=4)
    assert oversized.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert not (tmp_path / "large.md").exists()


@pytest.mark.parametrize("relative_path", (Path("../escape.md"), Path("nested/../../escape.md")))
def test_bytes_artifact_rejects_parent_traversal(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"

    with pytest.raises(VideoDemoError) as raised:
        AtomicArtifactStore(runtime_root).write_bytes(relative_path, b"private", max_bytes=100)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert not (tmp_path / "escape.md").exists()


def test_bytes_artifact_rejects_absolute_and_symlink_parent_paths(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime_root.mkdir()
    outside.mkdir()
    (runtime_root / "linked").symlink_to(outside, target_is_directory=True)
    absolute_target = outside / "absolute.md"

    store = AtomicArtifactStore(runtime_root)
    with pytest.raises(VideoDemoError) as absolute:
        store.write_bytes(absolute_target, b"private", max_bytes=100)
    assert absolute.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE

    with pytest.raises(VideoDemoError) as symlink:
        store.write_bytes(Path("linked/escape.md"), b"private", max_bytes=100)
    assert symlink.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert not absolute_target.exists()
    assert not (outside / "escape.md").exists()


def test_bytes_artifact_exclusive_publish_preserves_existing_content(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)
    path = Path("result.md")
    original = store.write_bytes(path, b"original", max_bytes=100)

    with pytest.raises(FileExistsError):
        store.write_bytes(path, b"replacement", max_bytes=100, exclusive=True)

    assert store.read_verified_bytes(original, max_bytes=100) == b"original"


def test_bytes_artifact_nonexclusive_publish_atomically_replaces_content(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)
    path = Path("result.md")
    store.write_bytes(path, b"original", max_bytes=100)

    replacement = store.write_bytes(path, b"replacement", max_bytes=100)

    assert store.read_verified_bytes(replacement, max_bytes=100) == b"replacement"


def test_bytes_artifact_detects_digest_and_declared_size_tampering(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)
    receipt = store.write_bytes(Path("result.md"), b"original", max_bytes=100)
    artifact = tmp_path / receipt.relative_path
    artifact.write_bytes(b"tampered")

    with pytest.raises(VideoDemoError) as digest:
        store.read_verified_bytes(receipt, max_bytes=100)
    assert digest.value.code == ErrorCode.ARTIFACT_DIGEST_MISMATCH

    valid = store.write_bytes(Path("result.md"), b"original", max_bytes=100)
    forged = valid.model_copy(update={"size_bytes": valid.size_bytes + 1})
    with pytest.raises(VideoDemoError) as size:
        store.read_verified_bytes(forged, max_bytes=100)
    assert size.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_bytes_artifact_rejects_symlink_leaf_without_reading_target(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside")
    runtime_root.mkdir()
    (runtime_root / "result.md").symlink_to(outside)
    receipt = ArtifactBytesReceipt(
        relative_path="result.md",
        sha256=hashlib.sha256(b"outside").hexdigest(),
        size_bytes=len(b"outside"),
    )

    with pytest.raises(VideoDemoError) as raised:
        AtomicArtifactStore(runtime_root).read_verified_bytes(receipt, max_bytes=100)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert outside.read_bytes() == b"outside"


def test_bytes_artifact_rejects_symlink_leaf_without_overwriting_target(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside")
    runtime_root.mkdir()
    (runtime_root / "result.md").symlink_to(outside)

    with pytest.raises(VideoDemoError) as raised:
        AtomicArtifactStore(runtime_root).write_bytes(
            Path("result.md"),
            b"replacement",
            max_bytes=100,
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert outside.read_bytes() == b"outside"


def test_bytes_artifact_reads_at_most_limit_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AtomicArtifactStore(tmp_path)
    receipt = store.write_bytes(Path("result.md"), b"12345", max_bytes=10)
    read_sizes: list[int] = []
    original_read = os.read

    def tracked_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(artifacts_module.os, "read", tracked_read)

    with pytest.raises(VideoDemoError) as raised:
        store.read_verified_bytes(receipt, max_bytes=4)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert read_sizes[0] == 5
    assert sum(read_sizes) <= 5


def test_bytes_artifact_rejects_file_replaced_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AtomicArtifactStore(tmp_path)
    receipt = store.write_bytes(Path("result.md"), b"original", max_bytes=100)
    artifact = tmp_path / receipt.relative_path
    replacement = tmp_path / "replacement.md"
    replacement.write_bytes(b"replaced")
    original_open = os.open
    replaced = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        is_target_read = (
            Path(path) == Path(artifact.name)
            and flags & os.O_RDONLY == os.O_RDONLY
            and not replaced
        )
        if is_target_read:
            replacement.replace(artifact)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts_module.os, "open", racing_open)

    with pytest.raises(VideoDemoError) as raised:
        store.read_verified_bytes(receipt, max_bytes=100)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_limited_json_rejects_symlink_leaf_and_limit_plus_one(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "runtime")
    receipt = store.write_json(
        Path("runs/run_001/result.json"),
        {"value": "正文"},
        schema_version="3.0.0",
        upstream_sha256="a" * 64,
        file_mode=0o600,
    )
    with pytest.raises(VideoDemoError) as oversized:
        store.read_verified_json_limited(receipt, max_bytes=1)
    assert oversized.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID

    artifact = store.runtime_root / receipt.relative_path
    outside = tmp_path / "outside.json"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)
    with pytest.raises(VideoDemoError) as symlink:
        store.read_verified_json_limited(receipt, max_bytes=1024)
    assert symlink.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_json_write_rejects_non_positive_limit(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="写入上限"):
        store.write_json(
            Path("result.json"),
            {},
            schema_version="3.0.0",
            upstream_sha256="a" * 64,
            max_bytes=0,
        )


def test_limited_json_rejects_file_replaced_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AtomicArtifactStore(tmp_path / "runtime")
    receipt = store.write_json(
        Path("result.json"),
        {"value": "original"},
        schema_version="3.0.0",
        upstream_sha256="a" * 64,
        file_mode=0o600,
    )
    artifact = store.runtime_root / receipt.relative_path
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"replacement")
    original_open = os.open
    replaced = False

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if Path(path) == Path(artifact.name) and not replaced:
            replacement.replace(artifact)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifacts_module.os, "open", racing_open)

    with pytest.raises(VideoDemoError) as raised:
        store.read_verified_json_limited(receipt, max_bytes=1024)
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_discard_bytes_deletes_only_unchanged_regular_artifact(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "runtime")
    receipt = store.write_bytes(Path("result.md"), b"document", max_bytes=100)
    artifact = store.runtime_root / receipt.relative_path

    assert store.discard_bytes(receipt.model_copy(update={"sha256": "b" * 64})) is False
    assert artifact.exists()
    assert store.discard_bytes(receipt.model_copy(update={"size_bytes": 1})) is False
    assert artifact.exists()
    assert store.discard_bytes(receipt) is True
    assert not artifact.exists()


def test_discard_bytes_rejects_symlink_without_deleting_target(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "runtime")
    receipt = store.write_bytes(Path("result.md"), b"document", max_bytes=100)
    artifact = store.runtime_root / receipt.relative_path
    outside = tmp_path / "outside.md"
    artifact.replace(outside)
    artifact.symlink_to(outside)

    assert store.discard_bytes(receipt) is False
    assert outside.read_bytes() == b"document"


def test_discard_artifact_rejects_replaced_or_symlink_json(tmp_path: Path) -> None:
    store = AtomicArtifactStore(tmp_path / "runtime")
    receipt = store.write_json(
        Path("result.json"),
        {"value": "original"},
        schema_version="3.0.0",
        upstream_sha256="a" * 64,
        file_mode=0o600,
    )
    artifact = store.runtime_root / receipt.relative_path
    artifact.write_bytes(b"replacement")
    assert store.discard_artifact(receipt, max_bytes=1024) is False
    assert artifact.read_bytes() == b"replacement"

    outside = tmp_path / "outside.json"
    artifact.replace(outside)
    artifact.symlink_to(outside)
    assert store.discard_artifact(receipt, max_bytes=1024) is False
    assert outside.read_bytes() == b"replacement"
