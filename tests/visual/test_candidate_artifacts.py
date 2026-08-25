from __future__ import annotations

import hashlib
import os
import stat
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Event, Thread

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual import candidate_artifacts
from video_demo.visual.candidate_artifacts import (
    CandidateArtifactSession,
    CandidateDirectoryLease,
)


def _jpeg(label: str) -> bytes:
    return b"\xff\xd8\xff" + label.encode("utf-8") + b"\xff\xd9"


def _write_existing_candidate(candidate_root: Path, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    destination = candidate_root / f"{digest}.jpg"
    destination.write_bytes(payload)
    destination.chmod(0o600)
    return destination


def _session(
    runtime_root: Path,
    *,
    max_unique_bytes: int = 1024,
    max_files: int = 20_000,
) -> CandidateArtifactSession:
    return CandidateArtifactSession(
        runtime_root=runtime_root,
        max_unique_bytes=max_unique_bytes,
        max_files=max_files,
        max_file_bytes=1024,
    )


def _acquire_lease_in_spawned_process(
    runtime_root: str,
    run_relative_root: str,
    connection: Connection,
) -> None:
    try:
        with CandidateDirectoryLease(
            runtime_root=Path(runtime_root),
            run_relative_root=Path(run_relative_root),
            mode="EXCLUSIVE",
            wait_timeout_seconds=0.2,
        ):
            connection.send("ACQUIRED")
    except VideoDemoError as error:
        connection.send(error.code.value)
    finally:
        connection.close()


def test_session_locks_current_run_visual_root_and_snapshots_existing_files(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    candidate_root.mkdir(parents=True, mode=0o700)
    jpeg = _jpeg("existing-budget")
    existing = _write_existing_candidate(candidate_root, jpeg)
    session = _session(runtime_root, max_unique_bytes=len(jpeg))

    session.prepare_run(run_relative_root)

    lock_file = candidate_root.parent / ".candidates.lock"
    assert lock_file.is_file()
    assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600
    assert session.unique_bytes == len(jpeg)
    assert session.reserve("b" * 64, 1) == "REJECTED"
    assert list(candidate_root.iterdir()) == [existing]
    session.close()


def test_session_rejects_budget_reservation_before_run_lease(tmp_path: Path) -> None:
    session = _session(tmp_path / "runtime")

    with pytest.raises(VideoDemoError) as raised:
        session.reserve("b" * 64, 1)

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    session.close()


def test_session_rejects_unknown_candidate_directory_artifact(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    candidate_root.mkdir(parents=True, mode=0o700)
    (candidate_root / "unknown.part").write_bytes(b"partial")
    session = _session(runtime_root)

    with pytest.raises(VideoDemoError) as raised:
        session.prepare_run(run_relative_root)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    session.close()


def test_session_enforces_configured_candidate_file_count(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    candidate_root.mkdir(parents=True, mode=0o700)
    _write_existing_candidate(candidate_root, _jpeg("first"))
    _write_existing_candidate(candidate_root, _jpeg("second"))
    session = _session(runtime_root, max_files=1)

    with pytest.raises(VideoDemoError) as raised:
        session.prepare_run(run_relative_root)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    session.close()


def test_second_session_honors_cancellation_while_waiting_for_same_run(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    first = _session(runtime_root)
    first.prepare_run(run_relative_root)
    second = CandidateArtifactSession(
        runtime_root=runtime_root,
        max_unique_bytes=1024,
        max_files=20_000,
        max_file_bytes=1024,
        is_cancel_requested=lambda: True,
        lock_timeout_seconds=1.0,
    )

    with pytest.raises(VideoDemoError) as raised:
        second.prepare_run(run_relative_root)

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    first.close()
    second.close()


def test_same_run_lease_is_serialized_and_released_for_next_caller(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    first = CandidateDirectoryLease(
        runtime_root=runtime_root,
        run_relative_root=run_relative_root,
        mode="SHARED",
    ).acquire()
    entered = Event()
    finished = Event()

    def acquire_second() -> None:
        with CandidateDirectoryLease(
            runtime_root=runtime_root,
            run_relative_root=run_relative_root,
            mode="SHARED",
            wait_timeout_seconds=1.0,
        ):
            entered.set()
        finished.set()

    thread = Thread(target=acquire_second)
    thread.start()
    assert not entered.wait(timeout=0.1)

    first.close()

    assert entered.wait(timeout=1.0)
    assert finished.wait(timeout=1.0)
    thread.join(timeout=1.0)


def test_different_run_leases_do_not_share_process_lock(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    first = CandidateDirectoryLease(
        runtime_root=runtime_root,
        run_relative_root=Path("runs/scope/run_001"),
        mode="EXCLUSIVE",
    ).acquire()
    second = CandidateDirectoryLease(
        runtime_root=runtime_root,
        run_relative_root=Path("runs/scope/run_002"),
        mode="EXCLUSIVE",
        wait_timeout_seconds=0.1,
    )

    second.acquire()

    assert second.is_acquired
    second.close()
    first.close()


def test_exclusive_lease_blocks_another_process_until_released(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    first = CandidateDirectoryLease(
        runtime_root=runtime_root,
        run_relative_root=run_relative_root,
        mode="EXCLUSIVE",
    ).acquire()
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_acquire_lease_in_spawned_process,
        args=(str(runtime_root), str(run_relative_root), child_connection),
    )
    process.start()
    child_connection.close()

    assert parent_connection.recv() == ErrorCode.VISUAL_RESULT_INVALID.value
    process.join(timeout=3.0)
    assert process.exitcode == 0
    parent_connection.close()
    first.close()


def test_zero_timeout_lease_attempts_once_and_succeeds_when_run_is_free(tmp_path: Path) -> None:
    lease = CandidateDirectoryLease(
        runtime_root=tmp_path / "runtime",
        run_relative_root=Path("runs/scope/run_001"),
        mode="EXCLUSIVE",
        wait_timeout_seconds=0.0,
    )

    lease.acquire()

    assert lease.is_acquired
    lease.close()


def test_session_rejects_symlink_run_lock_file(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    visual_root = runtime_root / run_relative_root / "visual"
    visual_root.mkdir(parents=True, mode=0o700)
    foreign = runtime_root / "foreign.lock"
    foreign.write_bytes(b"foreign")
    (visual_root / ".candidates.lock").symlink_to(foreign)
    session = _session(runtime_root)

    with pytest.raises(VideoDemoError) as raised:
        session.prepare_run(run_relative_root)

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert foreign.read_bytes() == b"foreign"
    session.close()


def test_session_reuses_existing_digest_without_temporary_file(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    candidate_root.mkdir(parents=True, mode=0o700)
    jpeg = _jpeg("existing")
    existing = _write_existing_candidate(candidate_root, jpeg)
    digest = hashlib.sha256(jpeg).hexdigest()
    session = _session(runtime_root)
    session.prepare_run(run_relative_root)

    publication = session.publish_jpeg(
        run_relative_root / "visual/candidates" / f"{digest}.jpg",
        jpeg,
        digest,
    )

    assert publication.status == "PUBLISHED"
    assert publication.created_by_call is False
    assert list(candidate_root.iterdir()) == [existing]
    session.close()


def test_session_fails_closed_when_unseen_digest_appears_after_snapshot(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    jpeg = _jpeg("racing-writer")
    digest = hashlib.sha256(jpeg).hexdigest()
    session = _session(runtime_root)
    session.prepare_run(run_relative_root)
    destination = runtime_root / run_relative_root / "visual/candidates" / f"{digest}.jpg"
    destination.write_bytes(jpeg)
    destination.chmod(0o600)

    with pytest.raises(VideoDemoError) as raised:
        session.publish_jpeg(
            run_relative_root / "visual/candidates" / f"{digest}.jpg",
            jpeg,
            digest,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert destination.read_bytes() == jpeg
    session.close()


def test_failed_publication_removes_only_newly_linked_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    jpeg = _jpeg("fsync-failure")
    digest = hashlib.sha256(jpeg).hexdigest()
    session = _session(runtime_root)
    session.prepare_run(run_relative_root)
    real_fsync_directory = candidate_artifacts._fsync_directory
    calls = 0

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(candidate_artifacts, "_fsync_directory", fail_once)

    with pytest.raises(VideoDemoError) as raised:
        session.publish_jpeg(
            run_relative_root / "visual/candidates" / f"{digest}.jpg",
            jpeg,
            digest,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    destination = runtime_root / run_relative_root / "visual/candidates" / f"{digest}.jpg"
    assert not destination.exists()
    assert session.owned_artifacts == ()
    session.close()


def test_failed_identity_read_after_link_does_not_leave_unowned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    jpeg = _jpeg("identity-read-failure")
    digest = hashlib.sha256(jpeg).hexdigest()
    destination = runtime_root / run_relative_root / "visual/candidates" / f"{digest}.jpg"
    session = _session(runtime_root)
    session.prepare_run(run_relative_root)
    real_lstat = candidate_artifacts.os.lstat
    real_link = candidate_artifacts.os.link
    linked = False
    failed = False

    def track_link(
        source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal linked
        real_link(source, target, follow_symlinks=follow_symlinks)
        linked = True

    def fail_first_destination_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        nonlocal failed
        if linked and Path(path) == destination and not failed:
            failed = True
            raise OSError("simulated identity read failure")
        return real_lstat(path)

    monkeypatch.setattr(candidate_artifacts.os, "link", track_link)
    monkeypatch.setattr(candidate_artifacts.os, "lstat", fail_first_destination_lstat)

    with pytest.raises(VideoDemoError) as raised:
        session.publish_jpeg(
            run_relative_root / "visual/candidates" / f"{digest}.jpg",
            jpeg,
            digest,
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert not destination.exists()
    assert session.owned_artifacts == ()
    session.close()


def test_session_rejects_budget_before_creating_candidate_file(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    jpeg = _jpeg("over-budget")
    digest = hashlib.sha256(jpeg).hexdigest()
    session = CandidateArtifactSession(
        runtime_root=runtime_root,
        max_unique_bytes=len(jpeg) - 1,
        max_files=20_000,
        max_file_bytes=1024,
    )
    session.prepare_run(run_relative_root)

    publication = session.publish_jpeg(
        run_relative_root / "visual/candidates" / f"{digest}.jpg",
        jpeg,
        digest,
    )

    assert publication.status == "BUDGET_REJECTED"
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    assert not tuple(candidate_root.iterdir())
    session.close()


def test_session_rejects_existing_candidate_with_extra_hard_link(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    candidate_root = runtime_root / run_relative_root / "visual/candidates"
    candidate_root.mkdir(parents=True, mode=0o700)
    existing = _write_existing_candidate(candidate_root, _jpeg("linked"))
    os.link(existing, runtime_root / "foreign-copy.jpg")
    session = _session(runtime_root)

    with pytest.raises(VideoDemoError) as raised:
        session.prepare_run(run_relative_root)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    session.close()


def test_cleanup_refuses_to_remove_replaced_owned_inode(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    run_relative_root = Path("runs/scope/run_001")
    relative_path = run_relative_root / "visual/candidates"
    jpeg = _jpeg("owned")
    digest = hashlib.sha256(jpeg).hexdigest()
    session = _session(runtime_root)
    session.prepare_run(run_relative_root)
    publication = session.publish_jpeg(relative_path / f"{digest}.jpg", jpeg, digest)
    assert publication.created_by_call is True
    candidate = runtime_root / relative_path / f"{digest}.jpg"

    replacement = candidate.with_name("replacement.jpg")
    replacement.write_bytes(jpeg)
    replacement.chmod(0o600)
    os.replace(replacement, candidate)

    with pytest.raises(VideoDemoError) as raised:
        session.cleanup_unretained(frozenset())

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert candidate.exists()
    session.close()
