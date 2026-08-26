from __future__ import annotations

import fcntl
import hashlib
import math
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import (
    reject_symlink_components,
    safe_runtime_path,
    validate_path_component,
)

CandidateReservation = Literal["NEW", "REUSED", "REJECTED"]
CandidatePublicationStatus = Literal["PUBLISHED", "BUDGET_REJECTED"]
CandidateLeaseMode = Literal["SHARED", "EXCLUSIVE"]

_LOCK_POLL_SECONDS = 0.05
_PROCESS_LOCK_GUARD = threading.Lock()


@dataclass(slots=True)
class _ProcessLockEntry:
    lock: threading.Lock
    users: int = 0


_PROCESS_LOCKS: dict[Path, _ProcessLockEntry] = {}


@dataclass(frozen=True, slots=True)
class OwnedCandidateArtifact:
    sha256: str
    path: Path
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class CandidatePublication:
    status: CandidatePublicationStatus
    created_by_call: bool = False


class CandidateDirectoryLease:
    """串行化同进程同 Run，并用 flock 协调跨进程候选目录访问。"""

    def __init__(
        self,
        *,
        runtime_root: Path,
        run_relative_root: Path,
        mode: CandidateLeaseMode,
        is_cancel_requested: Callable[[], bool] | None = None,
        wait_timeout_seconds: float = 300.0,
        create_candidate_directory: bool = True,
    ) -> None:
        if mode not in {"SHARED", "EXCLUSIVE"}:
            raise ValueError("候选帧目录租约模式非法")
        if not math.isfinite(wait_timeout_seconds) or wait_timeout_seconds < 0:
            raise ValueError("候选帧目录租约超时必须为有限非负数")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._mode = mode
        self._is_cancel_requested = is_cancel_requested or (lambda: False)
        self._wait_timeout_seconds = wait_timeout_seconds
        self._create_candidate_directory = create_candidate_directory
        self._visual_root = safe_runtime_path(
            self._runtime_root,
            run_relative_root / "visual",
        )
        self._candidate_root = safe_runtime_path(
            self._runtime_root,
            run_relative_root / "visual" / "candidates",
        )
        self._process_lock_entry: _ProcessLockEntry | None = None
        self._process_lock_acquired = False
        self._lock_descriptor = -1

    @classmethod
    def from_allowed_run_root(
        cls,
        *,
        runtime_root: Path,
        allowed_run_root: Path,
        mode: CandidateLeaseMode,
        is_cancel_requested: Callable[[], bool] | None = None,
        wait_timeout_seconds: float = 300.0,
    ) -> Self:
        run_relative_root = _validated_run_relative_root(runtime_root, allowed_run_root)
        return cls(
            runtime_root=runtime_root,
            run_relative_root=run_relative_root,
            mode=mode,
            is_cancel_requested=is_cancel_requested,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    @property
    def visual_root(self) -> Path:
        return self._visual_root

    @property
    def candidate_root(self) -> Path:
        return self._candidate_root

    @property
    def is_acquired(self) -> bool:
        return self._lock_descriptor >= 0

    def acquire(self) -> Self:
        if self.is_acquired:
            return self
        deadline = time.monotonic() + self._wait_timeout_seconds
        try:
            self._acquire_process_lock(deadline)
            _ensure_private_directory(self._visual_root, self._runtime_root)
            self._lock_descriptor = _open_lock_file(
                self._visual_root / ".candidates.lock",
            )
            _acquire_file_lock(
                self._lock_descriptor,
                mode=self._mode,
                deadline=deadline,
                is_cancel_requested=self._is_cancel_requested,
            )
            if self._create_candidate_directory:
                _ensure_private_directory(self._candidate_root, self._runtime_root)
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._lock_descriptor >= 0:
            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_descriptor)
                self._lock_descriptor = -1
        self._release_process_lock()

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def _acquire_process_lock(self, deadline: float) -> None:
        with _PROCESS_LOCK_GUARD:
            entry = _PROCESS_LOCKS.setdefault(
                self._visual_root,
                _ProcessLockEntry(lock=threading.Lock()),
            )
            entry.users += 1
        self._process_lock_entry = entry
        _raise_if_cancelled(self._is_cancel_requested)
        if entry.lock.acquire(blocking=False):
            self._process_lock_acquired = True
            return
        while True:
            _raise_if_cancelled(self._is_cancel_requested)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._release_process_lock(acquired=False)
                raise VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "候选帧 Run 锁等待超时",
                )
            if entry.lock.acquire(timeout=min(_LOCK_POLL_SECONDS, remaining)):
                self._process_lock_acquired = True
                return

    def _release_process_lock(self, *, acquired: bool = True) -> None:
        entry = self._process_lock_entry
        if entry is None:
            return
        if acquired and self._process_lock_acquired:
            entry.lock.release()
            self._process_lock_acquired = False
        with _PROCESS_LOCK_GUARD:
            entry.users -= 1
            if entry.users == 0 and _PROCESS_LOCKS.get(self._visual_root) is entry:
                _PROCESS_LOCKS.pop(self._visual_root, None)
        self._process_lock_entry = None


class CandidateArtifactSession:
    """在一个 Run 锁内管理候选 JPEG 的预算、发布和调用级所有权。"""

    def __init__(
        self,
        *,
        runtime_root: Path,
        max_unique_bytes: int,
        max_files: int,
        max_file_bytes: int,
        is_cancel_requested: Callable[[], bool] | None = None,
        lock_timeout_seconds: float = 300.0,
    ) -> None:
        if max_unique_bytes < 1 or max_files < 1 or max_file_bytes < 1:
            raise ValueError("候选帧会话预算必须大于 0")
        if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
            raise ValueError("候选帧会话锁超时必须为有限正数")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._max_unique_bytes = max_unique_bytes
        self._max_files = max_files
        self._max_file_bytes = max_file_bytes
        self._is_cancel_requested = is_cancel_requested or (lambda: False)
        self._lock_timeout_seconds = lock_timeout_seconds
        self._sizes: dict[str, int] = {}
        self._unique_bytes = 0
        self._owned: dict[str, OwnedCandidateArtifact] = {}
        self._candidate_root: Path | None = None
        self._lease: CandidateDirectoryLease | None = None

    @property
    def unique_bytes(self) -> int:
        return self._unique_bytes

    @property
    def owned_artifacts(self) -> tuple[OwnedCandidateArtifact, ...]:
        return tuple(self._owned.values())

    def prepare_run(self, run_relative_root: Path) -> None:
        candidate_root = safe_runtime_path(
            self._runtime_root,
            run_relative_root / "visual" / "candidates",
        )
        if self._lease is not None:
            if candidate_root != self._candidate_root:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧会话跨越多个 Run")
            return
        lease = CandidateDirectoryLease(
            runtime_root=self._runtime_root,
            run_relative_root=run_relative_root,
            mode="EXCLUSIVE",
            is_cancel_requested=self._is_cancel_requested,
            wait_timeout_seconds=self._lock_timeout_seconds,
        )
        try:
            lease.acquire()
            self._lease = lease
            self._candidate_root = candidate_root
            self._snapshot_existing_candidates()
        except Exception:
            lease.close()
            self._lease = None
            self._candidate_root = None
            raise

    def reserve(self, sha256: str, size_bytes: int) -> CandidateReservation:
        if self._lease is None or not self._lease.is_acquired:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧预算必须在 Run 锁内登记")
        if not _is_sha256(sha256) or size_bytes <= 0:
            raise ValueError("候选帧预算登记参数非法")
        if size_bytes > self._max_file_bytes:
            return "REJECTED"
        existing_size = self._sizes.get(sha256)
        if existing_size is not None:
            if existing_size != size_bytes:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "同 SHA 候选帧大小不一致")
            return "REUSED"
        if len(self._sizes) >= self._max_files:
            return "REJECTED"
        if self.unique_bytes + size_bytes > self._max_unique_bytes:
            return "REJECTED"
        self._sizes[sha256] = size_bytes
        self._unique_bytes += size_bytes
        return "NEW"

    def rollback(self, sha256: str, reservation: CandidateReservation) -> None:
        if reservation != "NEW":
            return
        size_bytes = self._sizes.pop(sha256, None)
        if size_bytes is not None:
            self._unique_bytes -= size_bytes

    def publish_jpeg(
        self,
        relative_path: Path,
        payload: bytes,
        sha256: str,
    ) -> CandidatePublication:
        destination = self._validate_publication(relative_path, payload, sha256)
        reservation = self.reserve(sha256, len(payload))
        if reservation == "REJECTED":
            return CandidatePublication(status="BUDGET_REJECTED")
        if reservation == "REUSED":
            _verify_existing_jpeg(
                destination,
                sha256,
                len(payload),
                self._max_file_bytes,
            )
            return CandidatePublication(status="PUBLISHED")
        try:
            status = _publish_new_jpeg(
                destination,
                payload,
                sha256,
                self._max_file_bytes,
            )
            self._record_created(sha256, destination, status)
            return CandidatePublication(
                status="PUBLISHED",
                created_by_call=True,
            )
        except Exception:
            self.rollback(sha256, reservation)
            raise

    def cleanup_unretained(self, retained_sha256: frozenset[str]) -> None:
        for digest, artifact in tuple(self._owned.items()):
            if digest in retained_sha256:
                continue
            _unlink_owned_artifact(artifact)
            self._owned.pop(digest, None)
            size_bytes = self._sizes.pop(digest, None)
            if size_bytes is not None:
                self._unique_bytes -= size_bytes

    def close(self) -> None:
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        self._candidate_root = None

    def _snapshot_existing_candidates(self) -> None:
        assert self._candidate_root is not None
        try:
            with os.scandir(self._candidate_root) as entries:
                for count, entry in enumerate(entries, start=1):
                    if count > self._max_files:
                        raise VideoDemoError(
                            ErrorCode.INPUT_BUDGET_EXCEEDED,
                            "候选帧目录条目超过配置上限",
                        )
                    self._snapshot_entry(Path(entry.path))
        except VideoDemoError:
            raise
        except OSError:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "候选帧目录快照读取失败",
            ) from None

    def _snapshot_entry(self, path: Path) -> None:
        if path.suffix != ".jpg" or not _is_sha256(path.stem):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "候选帧目录包含未知制品")
        digest = path.stem
        descriptor = -1
        try:
            before_open = os.lstat(path)
            descriptor = os.open(path, os.O_RDONLY | _no_follow_flag())
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_nlink != 1
                or not 0 < before.st_size <= self._max_file_bytes
                or _file_identity(before_open) != _file_identity(before)
            ):
                raise OSError
            payload = _read_bounded(descriptor, self._max_file_bytes)
            after = os.fstat(descriptor)
            after_path = os.lstat(path)
            if (
                _file_identity(before) != _file_identity(after)
                or _file_identity(after) != _file_identity(after_path)
                or not _is_jpeg(payload)
                or hashlib.sha256(payload).hexdigest() != digest
            ):
                raise OSError
            if self._unique_bytes + before.st_size > self._max_unique_bytes:
                raise VideoDemoError(
                    ErrorCode.INPUT_BUDGET_EXCEEDED,
                    "既有候选帧超过总字节预算",
                )
            self._sizes[digest] = before.st_size
            self._unique_bytes += before.st_size
        except VideoDemoError:
            raise
        except OSError:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "既有候选帧目录快照校验失败",
            ) from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_publication(
        self,
        relative_path: Path,
        payload: bytes,
        sha256: str,
    ) -> Path:
        if self._candidate_root is None:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧会话尚未准备 Run")
        if (
            not _is_sha256(sha256)
            or not _is_jpeg(payload)
            or hashlib.sha256(payload).hexdigest() != sha256
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧 JPEG 产物非法")
        destination = safe_runtime_path(self._runtime_root, relative_path)
        reject_symlink_components(
            self._runtime_root,
            destination,
            message="候选帧输出路径不能包含符号链接",
        )
        if (
            destination.parent != self._candidate_root
            or destination.name != f"{sha256}.jpg"
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧内容地址路径非法")
        return destination

    def _record_created(self, sha256: str, path: Path, status: os.stat_result) -> None:
        expected_size = self._sizes.get(sha256)
        if (
            expected_size is None
            or not stat.S_ISREG(status.st_mode)
            or status.st_size != expected_size
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧所有权登记失败")
        self._owned[sha256] = OwnedCandidateArtifact(
            sha256=sha256,
            path=path,
            size_bytes=status.st_size,
            device=status.st_dev,
            inode=status.st_ino,
        )


def read_verified_candidate_jpeg(
    allowed_run_root: Path,
    frame: FrameCandidateArtifact,
    *,
    max_bytes: int,
) -> bytes:
    """受限读取当前 Run 下与内容地址一致的私有候选 JPEG。"""

    if max_bytes < 1:
        raise ValueError("候选帧读取上限必须大于 0")
    run_root = allowed_run_root.expanduser()
    if not run_root.is_absolute() or not run_root.is_dir():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法")
    run_root = run_root.resolve(strict=True)
    expected_relative = Path("visual/candidates") / f"{frame.sha256}.jpg"
    if Path(frame.relative_path) != expected_relative or frame.mime_type != "image/jpeg":
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "候选帧内容地址路径非法")
    candidate_root = reject_symlink_components(
        run_root,
        run_root / "visual/candidates",
        message="候选帧必须位于当前 Run 且不能包含符号链接",
    )
    return _read_verified_candidate_at(
        run_root,
        candidate_root,
        f"{frame.sha256}.jpg",
        expected_digest=frame.sha256,
        expected_size=frame.size_bytes,
        max_bytes=max_bytes,
    )


def _read_verified_candidate_at(
    run_root: Path,
    candidate_root: Path,
    name: str,
    *,
    expected_digest: str,
    expected_size: int,
    max_bytes: int,
) -> bytes:
    directory_descriptor = -1
    file_descriptor = -1
    try:
        before_directory = os.lstat(candidate_root)
        directory_descriptor = os.open(
            candidate_root,
            os.O_RDONLY | _directory_flags() | _no_follow_flag(),
        )
        opened_directory = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or stat.S_IMODE(opened_directory.st_mode) != 0o700
            or _directory_identity(before_directory)
            != _directory_identity(opened_directory)
        ):
            raise OSError
        before_file = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        file_descriptor = os.open(
            name,
            os.O_RDONLY | _no_follow_flag(),
            dir_fd=directory_descriptor,
        )
        opened_file = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened_file.st_mode)
            or stat.S_IMODE(opened_file.st_mode) != 0o600
            or opened_file.st_nlink != 1
            or opened_file.st_size != expected_size
            or _file_identity(before_file) != _file_identity(opened_file)
        ):
            raise OSError
        payload = _read_bounded(file_descriptor, max_bytes)
        after_file = os.fstat(file_descriptor)
        current_file = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        current_directory = os.fstat(directory_descriptor)
        reject_symlink_components(
            run_root,
            candidate_root,
            message="候选帧必须位于当前 Run 且不能包含符号链接",
        )
        after_directory = os.lstat(candidate_root)
        if (
            _file_identity(opened_file) != _file_identity(after_file)
            or _file_identity(after_file) != _file_identity(current_file)
            or _directory_identity(opened_directory)
            != _directory_identity(current_directory)
            or _directory_identity(current_directory)
            != _directory_identity(after_directory)
            or not _is_jpeg(payload)
            or hashlib.sha256(payload).hexdigest() != expected_digest
        ):
            raise OSError
        return payload
    except (OSError, VideoDemoError):
        raise VideoDemoError(
            ErrorCode.ARTIFACT_SCHEMA_INVALID,
            "候选帧内容完整性校验失败",
        ) from None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _open_lock_file(path: Path) -> int:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | _no_follow_flag(), 0o600)
        os.fchmod(descriptor, 0o600)
        status = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_nlink != 1
            or _file_identity(status) != _file_identity(current)
        ):
            raise OSError
        return descriptor
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧 Run 锁文件非法") from None


def _acquire_file_lock(
    descriptor: int,
    *,
    mode: CandidateLeaseMode,
    deadline: float,
    is_cancel_requested: Callable[[], bool],
) -> None:
    operation = fcntl.LOCK_SH if mode == "SHARED" else fcntl.LOCK_EX
    _raise_if_cancelled(is_cancel_requested)
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    except OSError:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧 Run 加锁失败") from None
    while True:
        _raise_if_cancelled(is_cancel_requested)
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "候选帧 Run 锁等待超时",
                ) from None
            time.sleep(_LOCK_POLL_SECONDS)
        except OSError:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧 Run 加锁失败") from None


def _publish_new_jpeg(
    destination: Path,
    payload: bytes,
    digest: str,
    max_file_bytes: int,
) -> os.stat_result:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    destination_identity: tuple[int, int, int] | None = None
    publication_complete = False
    try:
        descriptor = os.open(temporary, _exclusive_write_flags(), 0o600)
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != len(payload):
            raise OSError
        temporary_identity = (status.st_dev, status.st_ino)
        os.close(descriptor)
        descriptor = -1
        current = os.lstat(temporary)
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or temporary_identity != (current.st_dev, current.st_ino)
        ):
            raise OSError
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            raise VideoDemoError(
                ErrorCode.VISUAL_RESULT_INVALID,
                "候选帧目录在快照后出现未知同名制品",
            ) from None
        destination_identity = (temporary_identity[0], temporary_identity[1], len(payload))
        published = os.lstat(destination)
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino, published.st_size)
            != destination_identity
        ):
            raise OSError
        temporary.unlink()
        _fsync_directory(destination.parent)
        final_status = os.lstat(destination)
        if (
            not stat.S_ISREG(final_status.st_mode)
            or (final_status.st_dev, final_status.st_ino, final_status.st_size)
            != destination_identity
        ):
            raise OSError
        publication_complete = True
        return final_status
    except VideoDemoError:
        raise
    except OSError:
        raise VideoDemoError(
            ErrorCode.VISUAL_RESULT_INVALID,
            "候选帧内容寻址写入失败",
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if destination_identity is not None and not publication_complete:
            _unlink_matching_identity(destination, destination_identity)


def _unlink_matching_identity(path: Path, identity: tuple[int, int, int]) -> None:
    try:
        current = os.lstat(path)
        if (
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino, current.st_size) == identity
        ):
            path.unlink()
            _fsync_directory(path.parent)
    except FileNotFoundError:
        return
    except OSError:
        raise VideoDemoError(
            ErrorCode.VISUAL_RESULT_INVALID,
            "失败的候选帧发布无法安全回滚",
        ) from None


def _ensure_private_directory(path: Path, runtime_root: Path) -> None:
    reject_symlink_components(
        runtime_root,
        path,
        message="候选帧目录不能包含符号链接",
    )
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current == runtime_root:
            break
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        descriptor = os.open(directory, os.O_RDONLY | _directory_flags())
        try:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    reject_symlink_components(
        runtime_root,
        path,
        message="候选帧目录不能包含符号链接",
    )
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | _directory_flags())
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode):
            raise OSError
        os.fchmod(descriptor, 0o700)
    except OSError:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧私有目录非法") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_existing_jpeg(
    path: Path,
    expected_digest: str,
    expected_size: int,
    max_bytes: int,
) -> None:
    _read_verified_jpeg(
        path,
        expected_digest=expected_digest,
        expected_size=expected_size,
        max_bytes=max_bytes,
        error_code=ErrorCode.VISUAL_RESULT_INVALID,
        message="既有候选帧与内容地址不一致",
    )


def _read_verified_jpeg(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int,
    max_bytes: int,
    error_code: ErrorCode,
    message: str,
) -> bytes:
    descriptor = -1
    try:
        before_open = os.lstat(path)
        descriptor = os.open(path, os.O_RDONLY | _no_follow_flag())
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size != expected_size
            or _file_identity(before_open) != _file_identity(before)
        ):
            raise OSError
        payload = _read_bounded(descriptor, max_bytes)
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(after_path)
            or not _is_jpeg(payload)
            or hashlib.sha256(payload).hexdigest() != expected_digest
        ):
            raise OSError
        return payload
    except OSError:
        raise VideoDemoError(error_code, message) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unlink_owned_artifact(artifact: OwnedCandidateArtifact) -> None:
    descriptor = -1
    try:
        descriptor = os.open(artifact.path, os.O_RDONLY | _no_follow_flag())
        before = os.fstat(descriptor)
        identity = (artifact.device, artifact.inode, artifact.size_bytes)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size) != identity
        ):
            raise OSError
        payload = _read_bounded(descriptor, artifact.size_bytes)
        after = os.fstat(descriptor)
        current = os.lstat(artifact.path)
        if (
            (after.st_dev, after.st_ino, after.st_size) != identity
            or (current.st_dev, current.st_ino, current.st_size) != identity
            or not stat.S_ISREG(current.st_mode)
            or not _is_jpeg(payload)
            or hashlib.sha256(payload).hexdigest() != artifact.sha256
        ):
            raise OSError
        artifact.path.unlink()
        _fsync_directory(artifact.path.parent)
    except FileNotFoundError:
        return
    except OSError:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧所有权清理失败") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _raise_if_cancelled(is_cancel_requested: Callable[[], bool]) -> None:
    if is_cancel_requested():
        raise VideoDemoError(ErrorCode.JOB_CANCELLED, "任务已请求取消")


def _exclusive_write_flags() -> int:
    return int(os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag())


def _no_follow_flag() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全候选帧访问")
    return int(no_follow)


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全候选帧目录")
    return int(no_follow | directory)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError
        view = view[written:]


def _read_bounded(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total)):
        total += len(chunk)
        if total > max_bytes:
            raise OSError
        chunks.append(chunk)
    return b"".join(chunks)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | _directory_flags())
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_jpeg(payload: bytes) -> bool:
    return payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")


def _validated_run_relative_root(runtime_root: Path, allowed_run_root: Path) -> Path:
    runtime = runtime_root.expanduser().resolve(strict=False)
    lexical = allowed_run_root.expanduser()
    if not lexical.is_absolute():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根必须是绝对路径")
    run_root = reject_symlink_components(
        runtime,
        lexical,
        message="当前 Run 根必须位于运行目录且不能包含符号链接",
    )
    try:
        relative = run_root.relative_to(runtime)
    except ValueError:
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法") from None
    if len(relative.parts) != 3 or relative.parts[0] != "runs" or not run_root.is_dir():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法")
    validate_path_component(relative.parts[1], "scope_key")
    validate_path_component(relative.parts[2], "run_id")
    return relative
