from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.errors import VideoDemoError
from video_demo.visual.keyframe_artifacts import KeyframeArtifactSession


def _candidate(run_root: Path, payload: bytes) -> FrameCandidateArtifact:
    digest = hashlib.sha256(payload).hexdigest()
    path = run_root / "visual/candidates" / f"{digest}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    (run_root / "visual").chmod(0o700)
    path.parent.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    return FrameCandidateArtifact(
        frame_id="frame_001",
        timestamp_ms=1_500,
        sha256=digest,
        size_bytes=len(payload),
        relative_path=f"visual/candidates/{digest}.jpg",
        mime_type="image/jpeg",
        perceptual_hash="0123456789abcdef",
        target_ids=("target_001",),
    )


def test_publish_never_follows_replaced_keyframe_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffstable-parent\xff\xd9")
    keyframe_root = run_root / "visual/keyframes"
    keyframe_root.mkdir(mode=0o700)
    moved_root = run_root / "visual/keyframes-before-swap"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = keyframe_artifacts.os.open
    swapped = False

    def replace_parent_before_leaf_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and dir_fd is not None and os.fspath(path) == f"{frame.sha256}.jpg":
            swapped = True
            keyframe_root.rename(moved_root)
            keyframe_root.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(keyframe_artifacts.os, "open", replace_parent_before_leaf_open)

    with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
        existing = session.snapshot()
        with pytest.raises(VideoDemoError):
            session.publish((frame,), existing)

    assert swapped
    assert tuple(outside.iterdir()) == ()
    assert tuple(moved_root.iterdir()) == ()


def test_publish_rejects_replaced_visual_parent_even_when_symlink_resolves_to_same_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffstable-visual-parent\xff\xd9")
    visual_root = run_root / "visual"
    keyframe_root = visual_root / "keyframes"
    keyframe_root.mkdir(mode=0o700)
    moved_visual = run_root / "visual-before-swap"
    real_open = keyframe_artifacts.os.open
    swapped = False

    def replace_visual_before_leaf_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and dir_fd is not None and os.fspath(path) == f"{frame.sha256}.jpg":
            swapped = True
            visual_root.rename(moved_visual)
            visual_root.symlink_to(moved_visual, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(keyframe_artifacts.os, "open", replace_visual_before_leaf_open)

    with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
        existing = session.snapshot()
        with pytest.raises(VideoDemoError):
            session.publish((frame,), existing)

    assert swapped
    assert tuple((moved_visual / "keyframes").iterdir()) == ()


@pytest.mark.parametrize("failed_operation", ("fchmod", "first_fstat"))
def test_exclusive_create_failure_removes_owned_name_without_initial_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_operation: str,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffcreate-failure\xff\xd9")
    real_open = keyframe_artifacts.os.open
    real_fchmod = keyframe_artifacts.os.fchmod
    real_fstat = keyframe_artifacts.os.fstat
    created_descriptor = -1

    def remember_exclusive_leaf(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if flags & os.O_EXCL and os.fspath(path) == f"{frame.sha256}.jpg":
            created_descriptor = descriptor
        return descriptor

    def fail_fchmod(descriptor: int, mode: int) -> None:
        if failed_operation == "fchmod" and descriptor == created_descriptor:
            raise OSError("模拟创建后 fchmod 失败")
        real_fchmod(descriptor, mode)

    def fail_first_fstat(descriptor: int) -> os.stat_result:
        if failed_operation == "first_fstat" and descriptor == created_descriptor:
            raise OSError("模拟创建后首次 fstat 失败")
        return real_fstat(descriptor)

    monkeypatch.setattr(keyframe_artifacts.os, "open", remember_exclusive_leaf)
    monkeypatch.setattr(keyframe_artifacts.os, "fchmod", fail_fchmod)
    monkeypatch.setattr(keyframe_artifacts.os, "fstat", fail_first_fstat)

    with (
        KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session,
        pytest.raises(VideoDemoError),
    ):
        session.publish((frame,), session.snapshot())

    keyframe_root = run_root / "visual/keyframes"
    assert keyframe_root.is_dir()
    assert tuple(keyframe_root.iterdir()) == ()
