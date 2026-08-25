from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import platform
from pathlib import Path

import pytest

from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.errors import ErrorCode, VideoDemoError
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
    real_publish = keyframe_artifacts._publish_verified_fd
    swapped = False

    def replace_parent_before_publish(
        source_descriptor: int,
        target_directory: int,
        target_name: str,
    ) -> None:
        nonlocal swapped
        swapped = True
        keyframe_root.rename(moved_root)
        keyframe_root.symlink_to(outside, target_is_directory=True)
        real_publish(source_descriptor, target_directory, target_name)

    monkeypatch.setattr(
        keyframe_artifacts,
        "_publish_verified_fd",
        replace_parent_before_publish,
    )

    with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
        existing = session.snapshot()
        with pytest.raises(VideoDemoError):
            session.publish((frame,), existing)

    assert swapped
    assert tuple(outside.iterdir()) == ()
    published = tuple(moved_root.iterdir())
    assert len(published) == 1
    assert published[0].name == f"{frame.sha256}.jpg"
    assert published[0].read_bytes() == b"\xff\xd8\xffstable-parent\xff\xd9"


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
    real_publish = keyframe_artifacts._publish_verified_fd
    swapped = False

    def replace_visual_before_publish(
        source_descriptor: int,
        target_directory: int,
        target_name: str,
    ) -> None:
        nonlocal swapped
        swapped = True
        visual_root.rename(moved_visual)
        visual_root.symlink_to(moved_visual, target_is_directory=True)
        real_publish(source_descriptor, target_directory, target_name)

    monkeypatch.setattr(
        keyframe_artifacts,
        "_publish_verified_fd",
        replace_visual_before_publish,
    )

    with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
        existing = session.snapshot()
        with pytest.raises(VideoDemoError):
            session.publish((frame,), existing)

    assert swapped
    published = tuple((moved_visual / "keyframes").iterdir())
    assert len(published) == 1
    assert published[0].name == f"{frame.sha256}.jpg"
    assert published[0].read_bytes() == b"\xff\xd8\xffstable-visual-parent\xff\xd9"


@pytest.mark.parametrize("failed_operation", ("fchmod", "first_fstat"))
def test_exclusive_create_failure_keeps_owned_name_instead_of_racy_unlink(
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
    failed_fstat = False
    unlink_calls: list[str] = []

    def remember_exclusive_leaf(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        temporary_flag = getattr(os, "O_TMPFILE", 0)
        if (
            flags & os.O_EXCL and os.fspath(path).endswith(".pending")
        ) or (temporary_flag and flags & temporary_flag == temporary_flag):
            created_descriptor = descriptor
        return descriptor

    def fail_fchmod(descriptor: int, mode: int) -> None:
        if failed_operation == "fchmod" and descriptor == created_descriptor:
            raise OSError("模拟创建后 fchmod 失败")
        real_fchmod(descriptor, mode)

    def fail_first_fstat(descriptor: int) -> os.stat_result:
        nonlocal failed_fstat
        if (
            failed_operation == "first_fstat"
            and descriptor == created_descriptor
            and not failed_fstat
        ):
            failed_fstat = True
            raise OSError("模拟创建后首次 fstat 失败")
        return real_fstat(descriptor)

    def record_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        del dir_fd
        value = os.fspath(path)
        unlink_calls.append(value.decode() if isinstance(value, bytes) else value)

    monkeypatch.setattr(keyframe_artifacts.os, "open", remember_exclusive_leaf)
    monkeypatch.setattr(keyframe_artifacts.os, "fchmod", fail_fchmod)
    monkeypatch.setattr(keyframe_artifacts.os, "fstat", fail_first_fstat)
    monkeypatch.setattr(keyframe_artifacts.os, "unlink", record_unlink)

    with (
        KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session,
        pytest.raises(VideoDemoError),
    ):
        session.publish((frame,), session.snapshot())

    keyframe_root = run_root / "visual/keyframes"
    assert keyframe_root.is_dir()
    staging = tuple((run_root / "visual/.keyframe-staging").iterdir())
    assert unlink_calls == []
    assert tuple(keyframe_root.iterdir()) == ()
    assert len(staging) == (1 if platform.system() == "Darwin" else 0)
    if staging:
        assert staging[0].stat().st_size == 0


def test_exclusive_create_failure_keeps_name_when_fd_identity_cannot_be_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffunknown-identity\xff\xd9")
    real_open = keyframe_artifacts.os.open
    real_fstat = keyframe_artifacts.os.fstat
    created_descriptor = -1
    failed_fstats = 0

    def remember_exclusive_leaf(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        temporary_flag = getattr(os, "O_TMPFILE", 0)
        if (
            flags & os.O_EXCL and os.fspath(path).endswith(".pending")
        ) or (temporary_flag and flags & temporary_flag == temporary_flag):
            created_descriptor = descriptor
        return descriptor

    def fail_created_fstat(descriptor: int) -> os.stat_result:
        nonlocal failed_fstats
        if descriptor == created_descriptor:
            failed_fstats += 1
            raise OSError("模拟无法证明新建文件身份")
        return real_fstat(descriptor)

    monkeypatch.setattr(keyframe_artifacts.os, "open", remember_exclusive_leaf)
    monkeypatch.setattr(keyframe_artifacts.os, "fstat", fail_created_fstat)

    with (
        KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session,
        pytest.raises(VideoDemoError),
    ):
        session.publish((frame,), session.snapshot())

    staging = tuple((run_root / "visual/.keyframe-staging").iterdir())
    assert failed_fstats == 1
    assert tuple((run_root / "visual/keyframes").iterdir()) == ()
    assert len(staging) == (1 if platform.system() == "Darwin" else 0)
    if staging:
        assert staging[0].stat().st_size == 0


def test_atomic_publish_never_overwrites_existing_formal_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    _candidate(run_root, b"\xff\xd8\xffreplaced-owned-name\xff\xd9")
    new_payload = b"\xff\xd8\xffnew-content\xff\xd9"
    replacement_payload = b"existing-formal-name"
    digest = hashlib.sha256(new_payload).hexdigest()
    leaf = run_root / "visual/keyframes" / f"{digest}.jpg"
    leaf.parent.mkdir(mode=0o700)
    leaf.write_bytes(replacement_payload)
    leaf.chmod(0o600)

    with (
        KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session,
        pytest.raises(VideoDemoError),
    ):
        session._open_keyframe_directory(create=True)
        session._open_staging_directory(create=True)
        session._write_private_jpeg(leaf.name, new_payload)

    assert leaf.read_bytes() == replacement_payload


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS 使用具名 staging 源")
def test_publish_uses_verified_fd_when_staging_name_is_replaced_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    payload = b"\xff\xd8\xffverified-source-fd\xff\xd9"
    replacement_payload = b"unverified-staging-replacement"
    frame = _candidate(run_root, payload)
    real_publish = keyframe_artifacts._publish_verified_fd
    replaced = False

    def replace_staging_name_then_publish(
        source_descriptor: int,
        target_directory: int,
        target_name: str,
    ) -> None:
        nonlocal replaced
        staging = run_root / "visual/.keyframe-staging"
        source = next(staging.iterdir())
        source.unlink()
        source.write_bytes(replacement_payload)
        source.chmod(0o600)
        replaced = True
        real_publish(source_descriptor, target_directory, target_name)

    monkeypatch.setattr(
        keyframe_artifacts,
        "_publish_verified_fd",
        replace_staging_name_then_publish,
    )

    with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
        session.publish((frame,), session.snapshot())

    formal = run_root / "visual/keyframes" / f"{frame.sha256}.jpg"
    assert replaced
    assert formal.read_bytes() == payload


def test_atomic_publish_fails_closed_when_noreplace_syscall_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffunsupported-publish\xff\xd9")

    def unsupported(*_arguments: object) -> None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持原子排他发布")

    monkeypatch.setattr(keyframe_artifacts, "_publish_verified_fd", unsupported)

    with (
        KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session,
        pytest.raises(VideoDemoError) as raised,
    ):
        session.publish((frame,), session.snapshot())

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert tuple((run_root / "visual/keyframes").iterdir()) == ()
    staging = tuple((run_root / "visual/.keyframe-staging").iterdir())
    assert len(staging) == (1 if platform.system() == "Darwin" else 0)
    if staging:
        assert staging[0].read_bytes() == b"\xff\xd8\xffunsupported-publish\xff\xd9"


def test_linux_publish_source_uses_linkable_anonymous_inode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    calls: list[tuple[object, int, int, int | None]] = []

    def record_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        calls.append((path, flags, mode, dir_fd))
        return 42

    monkeypatch.setattr(keyframe_artifacts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(keyframe_artifacts.os, "O_TMPFILE", 0o20_000_000, raising=False)
    monkeypatch.setattr(keyframe_artifacts.os, "open", record_open)

    descriptor = keyframe_artifacts._open_publish_source(7, "ignored.pending")

    assert descriptor == 42
    assert calls == [(".", os.O_RDWR | 0o20_000_000, 0o600, 7)]
    assert calls[0][1] & os.O_EXCL == 0


def test_linux_publish_falls_back_to_proc_fd_when_empty_path_returns_enoent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    calls: list[tuple[int, bytes, int, bytes, int]] = []

    class FakeLinkat:
        argtypes: object | None = None
        restype: object | None = None

        def __call__(
            self,
            source_directory: int,
            source_name: bytes,
            target_directory: int,
            target_name: bytes,
            flags: int,
        ) -> int:
            assert ctypes.get_errno() == 0
            calls.append(
                (
                    source_directory,
                    source_name,
                    target_directory,
                    target_name,
                    flags,
                )
            )
            if len(calls) == 1:
                ctypes.set_errno(errno.ENOENT)
                return -1
            return 0

    class FakeLibc:
        def __init__(self) -> None:
            self.linkat = FakeLinkat()

    monkeypatch.setattr(keyframe_artifacts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(keyframe_artifacts.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())

    keyframe_artifacts._publish_verified_fd(17, 23, "evidence.jpg")

    assert calls == [
        (17, b"", 23, b"evidence.jpg", 0x00001000),
        (-100, b"/proc/self/fd/17", 23, b"evidence.jpg", 0x00000400),
    ]


def test_staging_residue_counts_toward_combined_runtime_budget(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    _candidate(run_root, b"\xff\xd8\xffstaging-budget\xff\xd9")
    staging = run_root / "visual/.keyframe-staging"
    staging.mkdir(mode=0o700)
    pending = staging / f"{'a' * 32}.pending"
    pending.write_bytes(b"residue")
    pending.chmod(0o600)

    with KeyframeArtifactSession(run_root, max_files=1, max_bytes=7) as session:
        existing = session.snapshot()

    assert len(existing) == 1
    assert sum(existing.values()) == 7


def test_publish_forces_private_directory_permissions_under_restrictive_umask(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs/scope_001/run_001"
    frame = _candidate(run_root, b"\xff\xd8\xffprivate-directories\xff\xd9")
    previous_umask = os.umask(0o777)
    try:
        with KeyframeArtifactSession(run_root, max_files=10, max_bytes=10_000) as session:
            session.publish((frame,), session.snapshot())
    finally:
        os.umask(previous_umask)

    keyframes = run_root / "visual/keyframes"
    staging = run_root / "visual/.keyframe-staging"
    assert keyframes.stat().st_mode & 0o777 == 0o700
    assert staging.stat().st_mode & 0o777 == 0o700
