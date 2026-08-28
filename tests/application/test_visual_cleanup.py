from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from video_demo.application.visual_cleanup import (
    PublishedVisualCleaner,
)
from video_demo.application.visual_cleanup_recovery import PublishedVisualCleanupRecovery
from video_demo.domain.document_artifact import DocumentArtifactPayload
from video_demo.domain.evidence import KeyframeEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus, RunStatusValue, VideoSummaryModel
from video_demo.persistence.repositories import (
    JobRepository,
    PublishedRunCleanupRecord,
    Scope,
    VideoRunRepository,
)
from video_demo.visual.candidate_artifacts import CandidateDirectoryLease


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _private_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _jpeg(label: bytes) -> bytes:
    return b"\xff\xd8\xff" + label + b"\xff\xd9"


def _keyframe(payload: bytes, *, index: int = 1) -> KeyframeEvidence:
    digest = hashlib.sha256(payload).hexdigest()
    return KeyframeEvidence(
        evidence_id=f"keyframe_evidence_{index:03d}",
        start_ms=index,
        end_ms=index + 1,
        keyframe_id=f"keyframe_{index:03d}",
        timestamp_ms=index,
        relative_path=f"visual/keyframes/{digest}.jpg",
        mime_type="image/jpeg",
        sha256=digest,
        perceptual_hash=f"{index:016x}",
        size_bytes=len(payload),
    )


def _cleaner(
    runtime_root: Path,
    *,
    max_candidate_files: int = 10,
    max_candidate_bytes: int = 4_096,
    max_published_keyframe_files: int = 10,
    max_published_keyframe_bytes: int = 4_096,
    max_keyframe_bytes: int = 1_024,
) -> PublishedVisualCleaner:
    return PublishedVisualCleaner(
        runtime_root,
        max_candidate_files=max_candidate_files,
        max_candidate_bytes=max_candidate_bytes,
        max_published_keyframe_files=max_published_keyframe_files,
        max_published_keyframe_bytes=max_published_keyframe_bytes,
        max_keyframe_bytes=max_keyframe_bytes,
    )


def test_cleanup_stops_after_scanned_candidate_inode_is_replaced_and_keeps_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    staging_root = visual_root / ".keyframe-staging"
    candidate_root = visual_root / "candidates"
    keyframe_root = visual_root / "keyframes"
    for directory in (staging_root, candidate_root, keyframe_root):
        _private_directory(directory)
    staging = staging_root / f"{'a' * 32}.pending"
    _private_file(staging, b"staging")
    # 让被替换条目按名称排序在前，证明竞态后不会继续删后续条目。
    victim_payload = _jpeg(b"survivor")
    victim = candidate_root / f"{hashlib.sha256(victim_payload).hexdigest()}.jpg"
    survivor_payload = _jpeg(b"victim")
    survivor = candidate_root / f"{hashlib.sha256(survivor_payload).hexdigest()}.jpg"
    _private_file(victim, victim_payload)
    _private_file(survivor, survivor_payload)
    original_unlink = os.unlink
    raced = False

    def replace_after_scan(path: str | bytes | Path, *args: object, **kwargs: object) -> None:
        nonlocal raced
        if not raced and os.fspath(path).endswith(".pending"):
            raced = True
            original_unlink(victim)
            _private_file(victim, victim_payload)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", replace_after_scan)
    cleaner = _cleaner(runtime_root)

    with pytest.raises(VideoDemoError) as raised:
        cleaner.cleanup(run_relative_root, ())

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert raced is True
    assert victim.is_file()
    assert survivor.is_file()
    assert (visual_root / "candidate-cleanup.pending").is_file()


def test_cleanup_rescans_formal_directory_and_rejects_late_keyframe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    candidate_root = visual_root / "candidates"
    keyframe_root = visual_root / "keyframes"
    for directory in (candidate_root, keyframe_root):
        _private_directory(directory)
    candidate_payload = _jpeg(b"candidate")
    candidate = candidate_root / f"{hashlib.sha256(candidate_payload).hexdigest()}.jpg"
    _private_file(candidate, candidate_payload)
    late_payload = _jpeg(b"late-formal")
    late_keyframe = keyframe_root / f"{hashlib.sha256(late_payload).hexdigest()}.jpg"
    original_unlink = os.unlink
    injected = False

    def inject_after_candidate_delete(
        path: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        original_unlink(path, *args, **kwargs)
        if not injected and os.fspath(path) == candidate.name:
            injected = True
            _private_file(late_keyframe, late_payload)

    monkeypatch.setattr(os, "unlink", inject_after_candidate_delete)

    with pytest.raises(VideoDemoError) as raised:
        _cleaner(runtime_root).cleanup(run_relative_root, ())

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert injected is True
    assert late_keyframe.is_file()
    assert (visual_root / "candidate-cleanup.pending").is_file()


def test_cleanup_rejects_candidate_total_bytes_over_budget(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    _private_directory(candidate_root)
    for label in (b"candidate-a", b"candidate-b"):
        payload = _jpeg(label)
        _private_file(candidate_root / f"{hashlib.sha256(payload).hexdigest()}.jpg", payload)

    with pytest.raises(VideoDemoError) as raised:
        _cleaner(runtime_root, max_candidate_bytes=20).cleanup(run_relative_root, ())

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_cleanup_accepts_unselected_candidate_larger_than_single_keyframe_limit(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    _private_directory(candidate_root)
    payload = _jpeg(b"x" * 2_048)
    candidate = candidate_root / f"{hashlib.sha256(payload).hexdigest()}.jpg"
    _private_file(candidate, payload)

    assert (
        _cleaner(
            runtime_root,
            max_candidate_bytes=4_096,
            max_keyframe_bytes=1_024,
        ).cleanup(run_relative_root, ())
        is True
    )
    assert not candidate.exists()


def test_cleanup_rejects_broken_candidate_directory_symlink_and_marks_pending(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    _private_directory(visual_root)
    (visual_root / "candidates").symlink_to(visual_root / "missing-candidates")

    with pytest.raises(VideoDemoError) as raised:
        _cleaner(runtime_root).cleanup(run_relative_root, ())

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert (visual_root / "candidate-cleanup.pending").is_file()


def test_cleanup_rejects_combined_formal_and_staging_file_count_over_budget(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    keyframe_root = visual_root / "keyframes"
    staging_root = visual_root / ".keyframe-staging"
    for directory in (keyframe_root, staging_root):
        _private_directory(directory)
    payload = _jpeg(b"retained")
    keyframe = _keyframe(payload)
    _private_file(keyframe_root / f"{keyframe.sha256}.jpg", payload)
    _private_file(staging_root / f"{'a' * 32}.pending", b"staging")

    with pytest.raises(VideoDemoError) as raised:
        _cleaner(runtime_root, max_published_keyframe_files=1).cleanup(
            run_relative_root,
            (keyframe,),
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_cleanup_rejects_combined_formal_and_staging_bytes_over_budget(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    keyframe_root = visual_root / "keyframes"
    staging_root = visual_root / ".keyframe-staging"
    for directory in (keyframe_root, staging_root):
        _private_directory(directory)
    payload = _jpeg(b"retained")
    keyframe = _keyframe(payload)
    _private_file(keyframe_root / f"{keyframe.sha256}.jpg", payload)
    _private_file(staging_root / f"{'a' * 32}.pending", b"staging")

    with pytest.raises(VideoDemoError) as raised:
        _cleaner(
            runtime_root,
            max_published_keyframe_bytes=len(payload) + len(b"staging") - 1,
        ).cleanup(run_relative_root, (keyframe,))

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_cleanup_does_not_read_formal_jpeg_after_staging_exhausts_combined_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    keyframe_root = visual_root / "keyframes"
    staging_root = visual_root / ".keyframe-staging"
    for directory in (keyframe_root, staging_root):
        _private_directory(directory)
    staging_payload = b"staging"
    _private_file(staging_root / f"{'a' * 32}.pending", staging_payload)
    keyframe_payload = _jpeg(b"retained")
    keyframe = _keyframe(keyframe_payload)
    _private_file(keyframe_root / f"{keyframe.sha256}.jpg", keyframe_payload)
    cleaner = _cleaner(
        runtime_root,
        max_published_keyframe_bytes=len(keyframe_payload),
    )
    read_digests: list[str] = []
    original_read = cleaner._read_verified_jpeg_at

    def track_read(
        descriptor: int,
        snapshot: object,
        digest: str,
        size_bytes: int,
        *,
        maximum_bytes: int,
    ) -> None:
        read_digests.append(digest)
        original_read(
            descriptor,
            snapshot,  # type: ignore[arg-type]
            digest,
            size_bytes,
            maximum_bytes=maximum_bytes,
        )

    monkeypatch.setattr(cleaner, "_read_verified_jpeg_at", track_read)
    monkeypatch.setattr(cleaner, "_record_cleanup_failure", lambda *_args: None)

    with pytest.raises(VideoDemoError) as raised:
        cleaner.cleanup(run_relative_root, (keyframe,))

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert read_digests == []


def test_cleanup_does_not_read_formal_jpeg_after_staging_exhausts_file_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    keyframe_root = visual_root / "keyframes"
    staging_root = visual_root / ".keyframe-staging"
    for directory in (keyframe_root, staging_root):
        _private_directory(directory)
    _private_file(staging_root / f"{'a' * 32}.pending", b"staging")
    keyframe_payload = _jpeg(b"retained")
    keyframe = _keyframe(keyframe_payload)
    _private_file(keyframe_root / f"{keyframe.sha256}.jpg", keyframe_payload)
    cleaner = _cleaner(runtime_root, max_published_keyframe_files=1)
    read_digests: list[str] = []
    original_read = cleaner._read_verified_jpeg_at

    def track_read(
        descriptor: int,
        snapshot: object,
        digest: str,
        size_bytes: int,
        *,
        maximum_bytes: int,
    ) -> None:
        read_digests.append(digest)
        original_read(
            descriptor,
            snapshot,  # type: ignore[arg-type]
            digest,
            size_bytes,
            maximum_bytes=maximum_bytes,
        )

    monkeypatch.setattr(cleaner, "_read_verified_jpeg_at", track_read)
    monkeypatch.setattr(cleaner, "_record_cleanup_failure", lambda *_args: None)

    with pytest.raises(VideoDemoError) as raised:
        cleaner.cleanup(run_relative_root, (keyframe,))

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert read_digests == []


def test_cleanup_accepts_staging_exactly_filling_combined_budget_with_empty_formal(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    keyframe_root = visual_root / "keyframes"
    staging_root = visual_root / ".keyframe-staging"
    for directory in (keyframe_root, staging_root):
        _private_directory(directory)
    staging_payload = b"staging"
    _private_file(staging_root / f"{'a' * 32}.pending", staging_payload)

    assert (
        _cleaner(
            runtime_root,
            max_published_keyframe_files=1,
            max_published_keyframe_bytes=len(staging_payload),
        ).cleanup(run_relative_root, ())
        is True
    )
    assert tuple(staging_root.iterdir()) == ()
    assert tuple(keyframe_root.iterdir()) == ()


def test_cleanup_failure_logs_bounded_residual_usage_without_path_or_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    _private_directory(candidate_root)
    payload = _jpeg(b"candidate")
    digest = hashlib.sha256(payload).hexdigest()
    _private_file(candidate_root / f"{digest}.jpg", payload)
    cleaner = _cleaner(runtime_root)

    def fail_delete(*_args: object, **_kwargs: object) -> None:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "原始清理错误")

    monkeypatch.setattr(cleaner, "_delete_entries", fail_delete)
    with (
        caplog.at_level(logging.WARNING, logger="video_demo.application.visual_cleanup"),
        pytest.raises(VideoDemoError, match="原始清理错误"),
    ):
        cleaner.cleanup(run_relative_root, ())

    message = caplog.messages[-1]
    assert "剩余文件数=1" in message
    assert f"剩余字节={len(payload)}" in message
    assert str(runtime_root) not in message
    assert digest not in message


def test_cleanup_stats_failure_does_not_mask_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    _private_directory(candidate_root)
    payload = _jpeg(b"candidate")
    _private_file(candidate_root / f"{hashlib.sha256(payload).hexdigest()}.jpg", payload)
    cleaner = _cleaner(runtime_root)

    def fail_delete(*_args: object, **_kwargs: object) -> None:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "原始清理错误")

    def fail_stats(*_args: object, **_kwargs: object) -> tuple[int, int]:
        raise RuntimeError("统计错误")

    monkeypatch.setattr(cleaner, "_delete_entries", fail_delete)
    monkeypatch.setattr(cleaner, "_residual_usage", fail_stats)

    with (
        caplog.at_level(logging.WARNING, logger="video_demo.application.visual_cleanup"),
        pytest.raises(VideoDemoError, match="原始清理错误"),
    ):
        cleaner.cleanup(run_relative_root, ())

    message = caplog.messages[-1]
    assert "run_id=run_001" in message
    assert "剩余统计不可用" in message
    assert "统计错误" not in message
    assert str(runtime_root) not in message


def test_remove_pending_rejects_identity_replacement_before_dir_fd_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    cleaner = _cleaner(runtime_root)
    cleaner.mark_pending(run_relative_root)
    pending = runtime_root / run_relative_root / "visual/candidate-cleanup.pending"
    replacement = b'{"replacement":true}'
    original_stat = os.stat
    pending_stats = 0

    def replace_between_checks(
        path: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        nonlocal pending_stats
        if os.fspath(path) == pending.name and kwargs.get("dir_fd") is not None:
            pending_stats += 1
            if pending_stats == 2:
                original_unlink = os.unlink
                original_unlink(pending)
                _private_file(pending, replacement)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", replace_between_checks)

    with pytest.raises(VideoDemoError) as raised:
        cleaner._remove_pending(run_relative_root)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert pending.read_bytes() == replacement


def test_remove_pending_fsyncs_visual_directory_after_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    cleaner = _cleaner(runtime_root)
    cleaner.mark_pending(run_relative_root)
    fsynced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsynced.append(descriptor))

    cleaner._remove_pending(run_relative_root)

    assert len(fsynced) == 1


def test_open_directory_closes_descriptor_when_binding_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    visual_root = runtime_root / "runs/scope/run_001/visual"
    _private_directory(visual_root)
    cleaner = _cleaner(runtime_root)
    opened: list[int] = []
    original_open = os.open

    def capture_open(path: str | bytes | Path, *args: object, **kwargs: object) -> int:
        descriptor = original_open(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(visual_root):
            opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(
        cleaner,
        "_require_bound_directory",
        lambda _snapshot: cleaner._invalid("模拟目录绑定复验失败"),
    )

    with pytest.raises(VideoDemoError, match="模拟目录绑定复验失败"):
        cleaner._open_directory(visual_root)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_cleanup_rejects_pending_reappearance_after_unlink_and_refreshes_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    cleaner = _cleaner(runtime_root)
    cleaner.mark_pending(run_relative_root)
    pending = runtime_root / run_relative_root / "visual/candidate-cleanup.pending"
    replacement = b'{"replacement":true}'
    original_unlink = os.unlink
    replaced = False

    def replace_after_unlink(
        path: str | bytes | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        original_unlink(path, *args, **kwargs)
        if not replaced and os.fspath(path) == pending.name:
            replaced = True
            _private_file(pending, replacement)

    monkeypatch.setattr(os, "unlink", replace_after_unlink)

    with pytest.raises(VideoDemoError) as raised:
        cleaner.cleanup(run_relative_root, ())

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert replaced is True
    assert b"cleanup-required" in pending.read_bytes()


def test_cleanup_closes_earlier_directory_descriptors_when_later_scan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    staging_root = visual_root / ".keyframe-staging"
    keyframe_root = visual_root / "keyframes"
    candidate_root = visual_root / "candidates"
    for directory in (staging_root, keyframe_root, candidate_root):
        _private_directory(directory)
    _private_file(staging_root / f"{'a' * 32}.pending", b"staging")
    _private_file(candidate_root / "unknown.part", b"unknown")
    cleaner = _cleaner(runtime_root)
    descriptors: list[int] = []
    original_scan = cleaner._scan_directory

    def capture_scan(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        snapshot = original_scan(*args, **kwargs)  # type: ignore[arg-type]
        if snapshot is not None:
            descriptors.append(snapshot.descriptor)
        return snapshot

    monkeypatch.setattr(cleaner, "_scan_directory", capture_scan)
    monkeypatch.setattr(cleaner, "_record_cleanup_failure", lambda *_args: None)

    with pytest.raises(VideoDemoError):
        cleaner.cleanup(run_relative_root, ())

    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_startup_recovery_uses_fixed_keyset_batches_and_skips_active_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = Scope("tenant", "app", "kb")
    records = tuple(
        PublishedRunCleanupRecord(run_pk=index, scope=scope, run_id=f"run_{index:03d}")
        for index in range(1, 102)
    )
    keyset_calls: list[tuple[int, int]] = []

    def list_batch(
        _repository: VideoRunRepository,
        *,
        after_id: int,
        limit: int,
    ) -> tuple[PublishedRunCleanupRecord, ...]:
        keyset_calls.append((after_id, limit))
        return tuple(item for item in records if item.run_pk > after_id)[:limit]

    monkeypatch.setattr(VideoRunRepository, "list_published_4_for_cleanup", list_batch)
    monkeypatch.setattr(
        JobRepository,
        "has_active_owner",
        lambda _repository, candidate_scope, run_id, now=None: (
            candidate_scope == scope and run_id == "run_050"
        ),
    )

    class Database:
        @contextmanager
        def session(self) -> Iterator[object]:
            yield object()

    class Reader:
        def get_artifact(
            self,
            _scope: Scope,
            _run_id: str,
        ) -> tuple[DocumentArtifactPayload, bytes]:
            return DocumentArtifactPayload.model_construct(evidence=()), b"document"

    class Cleaner:
        def __init__(self) -> None:
            self.cleaned: list[Path] = []

        def has_residuals(self, _run_relative_root: Path) -> bool:
            return True

        def cleanup(
            self,
            run_relative_root: Path,
            _keyframes: tuple[KeyframeEvidence, ...],
        ) -> bool:
            self.cleaned.append(run_relative_root)
            return True

        def mark_pending(self, _run_relative_root: Path) -> None:
            raise AssertionError("正常恢复不应刷新 pending")

    cleaner = Cleaner()
    recovered = PublishedVisualCleanupRecovery(
        Database(),  # type: ignore[arg-type]
        Reader(),  # type: ignore[arg-type]
        cleaner,  # type: ignore[arg-type]
        clock=lambda: datetime.now(UTC) + timedelta(seconds=1),
    ).recover()

    assert keyset_calls == [(0, 100), (100, 100)]
    assert recovered == 100
    assert len(cleaner.cleaned) == 100


def test_startup_recovery_isolates_residual_probe_and_pending_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    scope = Scope("tenant", "app", "kb")
    records = (
        PublishedRunCleanupRecord(run_pk=1, scope=scope, run_id="run_001"),
        PublishedRunCleanupRecord(run_pk=2, scope=scope, run_id="run_002"),
    )
    monkeypatch.setattr(
        VideoRunRepository,
        "list_published_4_for_cleanup",
        lambda *_args, **_kwargs: records,
    )
    monkeypatch.setattr(JobRepository, "has_active_owner", lambda *_args, **_kwargs: False)

    class Database:
        @contextmanager
        def session(self) -> Iterator[object]:
            yield object()

    class Reader:
        def get_artifact(
            self,
            _scope: Scope,
            _run_id: str,
        ) -> tuple[DocumentArtifactPayload, bytes]:
            return DocumentArtifactPayload.model_construct(evidence=()), b"document"

    class Cleaner:
        def __init__(self) -> None:
            self.cleaned: list[Path] = []

        def has_residuals(self, run_relative_root: Path) -> bool:
            if run_relative_root.name == "run_001":
                raise OSError("残留目录损坏")
            return True

        def cleanup(
            self,
            run_relative_root: Path,
            _keyframes: tuple[KeyframeEvidence, ...],
        ) -> bool:
            self.cleaned.append(run_relative_root)
            return True

        def mark_pending(self, _run_relative_root: Path) -> None:
            raise OSError("标记目录损坏")

    cleaner = Cleaner()
    with caplog.at_level(
        logging.WARNING,
        logger="video_demo.application.visual_cleanup_recovery",
    ):
        recovered = PublishedVisualCleanupRecovery(
            Database(),  # type: ignore[arg-type]
            Reader(),  # type: ignore[arg-type]
            cleaner,  # type: ignore[arg-type]
        ).recover()

    assert recovered == 1
    assert [path.name for path in cleaner.cleaned] == ["run_002"]
    assert any(
        "run_id=run_001" in message
        and "阶段=残留探测" in message
        and "异常类型=OSError" in message
        for message in caplog.messages
    )
    assert any("阶段=pending 刷新" in message for message in caplog.messages)
    assert all("残留目录损坏" not in message for message in caplog.messages)
    assert all("标记目录损坏" not in message for message in caplog.messages)


def test_cleanup_without_residual_directories_does_not_create_them(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    run_root = runtime_root / run_relative_root
    _private_directory(run_root)
    cleaner = _cleaner(runtime_root)

    assert cleaner.cleanup(run_relative_root, ()) is True
    assert not (run_root / "visual/candidates").exists()
    assert not (run_root / "visual/.keyframe-staging").exists()


def test_cleanup_without_residuals_still_requires_exclusive_candidate_lease(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    run_root = runtime_root / run_relative_root
    _private_directory(run_root)
    held = CandidateDirectoryLease(
        runtime_root=runtime_root,
        run_relative_root=run_relative_root,
        mode="EXCLUSIVE",
        wait_timeout_seconds=0.0,
        create_candidate_directory=False,
    )
    held.acquire()
    try:
        assert _cleaner(runtime_root).cleanup(run_relative_root, ()) is False
    finally:
        held.close()

    assert (run_root / "visual/candidate-cleanup.pending").is_file()
    assert not (run_root / "visual/candidates").exists()


def test_cleanup_repository_uses_real_sqlite_keyset_and_active_lease(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'cleanup.db'}")
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    now = datetime.now(UTC)
    with database.session() as session:
        runs = VideoRunRepository(session)
        for index in range(1, 103):
            asset = runs.get_or_create_asset(
                scope=scope,
                asset_id=f"asset_{index:03d}",
                object_ref=f"object_{index:03d}",
                source_sha256=f"{index:064x}",
            )
            run = runs.add(
                scope=scope,
                run_id=f"run_{index:03d}",
                asset_id=asset.asset_id,
                object_ref=asset.object_ref,
                idempotency_key=f"key_{index:03d}",
                config_snapshot={},
            )
            run.status = RunStatusValue.SUCCEEDED
            run.artifact_manifest_relative_path = f"result/{index}.json"
            run.artifact_manifest_sha256 = "a" * 64
            run.document_relative_path = f"result/{index}.md"
            run.document_sha256 = "b" * 64
            run.document_size_bytes = 1
            session.add(
                VideoSummaryModel(
                    tenant_id=scope.tenant_id,
                    application_id=scope.application_id,
                    knowledge_base_id=scope.knowledge_base_id,
                    run_id=run.run_id,
                    schema_version="2.0.0" if index == 102 else "4.0.0",
                    payload_json={},
                    retrieval_text="",
                    retrieval_hash="c" * 64,
                )
            )
        active = JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_active",
            run_id="run_050",
            now=now,
        )
        active.status = JobStatus.RUNNING
        active.worker_id = "worker"
        active.lease_expires_at = now + timedelta(minutes=1)
        empty_owner = JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_empty_owner",
            run_id="run_051",
            now=now,
        )
        empty_owner.status = JobStatus.RUNNING
        empty_owner.worker_id = ""
        empty_owner.lease_expires_at = now + timedelta(minutes=1)
        whitespace_owner = JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_whitespace_owner",
            run_id="run_052",
            now=now,
        )
        whitespace_owner.status = JobStatus.RUNNING
        whitespace_owner.worker_id = "  \t  "
        whitespace_owner.lease_expires_at = now + timedelta(minutes=1)

    with database.session() as session:
        repository = VideoRunRepository(session)
        first = repository.list_published_4_for_cleanup(after_id=0, limit=100)
        second = repository.list_published_4_for_cleanup(
            after_id=first[-1].run_pk,
            limit=100,
        )
        jobs = JobRepository(session)
        assert len(first) == 100
        assert [item.run_id for item in second] == ["run_101"]
        assert jobs.has_active_owner(scope, "run_050", now=now) is True
        assert jobs.has_active_owner(scope, "run_051", now=now) is False
        assert jobs.has_active_owner(scope, "run_052", now=now) is False
        assert jobs.has_active_owner(
            scope,
            "run_050",
            now=now + timedelta(minutes=2),
        ) is False


def test_cleanup_record_is_owned_by_persistence_layer() -> None:
    assert PublishedRunCleanupRecord.__module__ == "video_demo.persistence.repositories"
