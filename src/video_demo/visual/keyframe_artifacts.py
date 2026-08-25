from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components
from video_demo.visual.candidate_artifacts import read_verified_candidate_jpeg

_READ_CHUNK_BYTES = 1024 * 1024


class KeyframeArtifactSession:
    """用稳定目录描述符快照并发布当前 Run 的关键帧。"""

    def __init__(self, run_root: Path, *, max_files: int, max_bytes: int) -> None:
        if max_files < 1 or max_bytes < 1:
            raise ValueError("关键帧目录预算必须为正整数")
        lexical_root = run_root.expanduser()
        if not lexical_root.is_absolute() or not lexical_root.is_dir():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法")
        self._run_root = lexical_root.resolve(strict=True)
        self._visual_root = self._run_root / "visual"
        self._keyframe_path = self._visual_root / "keyframes"
        self._max_files = max_files
        self._max_bytes = max_bytes
        self._visual_descriptor = -1
        self._visual_identity: tuple[int, int] | None = None
        self._keyframe_descriptor = -1
        self._keyframe_identity: tuple[int, int] | None = None

    def __enter__(self) -> Self:
        self._open_visual_directory()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def close(self) -> None:
        if self._keyframe_descriptor >= 0:
            os.close(self._keyframe_descriptor)
            self._keyframe_descriptor = -1
            self._keyframe_identity = None
        if self._visual_descriptor >= 0:
            os.close(self._visual_descriptor)
            self._visual_descriptor = -1
            self._visual_identity = None

    def snapshot(self) -> dict[str, int]:
        if not self._open_keyframe_directory(create=False):
            return {}
        self._require_bound_keyframe_directory("关键帧目录在快照前发生变化")
        existing: dict[str, int] = {}
        total_bytes = 0
        try:
            with os.scandir(self._keyframe_descriptor) as entries:
                for count, entry in enumerate(entries, start=1):
                    if count > self._max_files:
                        raise VideoDemoError(
                            ErrorCode.INPUT_BUDGET_EXCEEDED,
                            "关键帧目录文件数超限",
                        )
                    name = entry.name
                    if not _is_keyframe_name(name):
                        raise OSError
                    payload = self._read_private_jpeg(name, name[:-4], self._max_bytes)
                    existing[name[:-4]] = len(payload)
                    total_bytes += len(payload)
                    if total_bytes > self._max_bytes:
                        raise VideoDemoError(
                            ErrorCode.INPUT_BUDGET_EXCEEDED,
                            "既有关键帧超过字节预算",
                        )
        except VideoDemoError:
            raise
        except OSError:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "关键帧目录快照非法",
            ) from None
        self._require_bound_keyframe_directory("关键帧目录在快照后发生变化")
        return existing

    def publish(
        self,
        frames: tuple[FrameCandidateArtifact, ...],
        existing: Mapping[str, int],
    ) -> None:
        if not frames:
            return
        self._open_keyframe_directory(create=True)
        self._require_bound_keyframe_directory("关键帧目录在发布前发生变化")
        created: list[tuple[str, tuple[int, int, int]]] = []
        published_sha = set(existing)
        try:
            for frame in frames:
                name = f"{frame.sha256}.jpg"
                if frame.sha256 in existing:
                    if existing[frame.sha256] != frame.size_bytes:
                        raise VideoDemoError(
                            ErrorCode.ARTIFACT_SCHEMA_INVALID,
                            "既有关键帧大小与候选元数据不一致",
                        )
                    self._read_private_jpeg(name, frame.sha256, frame.size_bytes)
                    continue
                if frame.sha256 in published_sha:
                    self._read_private_jpeg(name, frame.sha256, frame.size_bytes)
                    continue
                payload = read_verified_candidate_jpeg(
                    self._run_root,
                    frame,
                    max_bytes=frame.size_bytes,
                )
                identity = self._write_private_jpeg(name, payload)
                created.append((name, identity))
                published_sha.add(frame.sha256)
            self._require_bound_keyframe_directory("关键帧目录在发布后发生变化")
        except BaseException:
            self._rollback_created(created)
            raise

    def _open_visual_directory(self) -> None:
        if self._visual_descriptor >= 0:
            return
        reject_symlink_components(
            self._run_root,
            self._visual_root,
            message="视觉制品目录不能包含符号链接",
        )
        descriptor = -1
        try:
            before = os.lstat(self._visual_root)
            descriptor = os.open(
                self._visual_root,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _no_follow(),
            )
            opened = os.fstat(descriptor)
            current = os.lstat(self._visual_root)
            if (
                not _is_private_directory(opened)
                or _directory_identity(before) != _directory_identity(opened)
                or _directory_identity(opened) != _directory_identity(current)
            ):
                raise OSError
            self._visual_descriptor = descriptor
            self._visual_identity = _directory_identity(opened)
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "视觉制品目录非法") from None

    def _open_keyframe_directory(self, *, create: bool) -> bool:
        if self._keyframe_descriptor >= 0:
            self._require_bound_keyframe_directory("关键帧目录发生变化")
            return True
        assert self._visual_descriptor >= 0
        try:
            before = os.stat("keyframes", dir_fd=self._visual_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if not create:
                return False
            try:
                os.mkdir("keyframes", 0o700, dir_fd=self._visual_descriptor)
                os.fsync(self._visual_descriptor)
                before = os.stat(
                    "keyframes",
                    dir_fd=self._visual_descriptor,
                    follow_symlinks=False,
                )
            except (FileExistsError, OSError):
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "关键帧目录非法",
                ) from None
        except OSError:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "关键帧目录非法") from None
        descriptor = -1
        try:
            descriptor = os.open(
                "keyframes",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _no_follow(),
                dir_fd=self._visual_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                not _is_private_directory(opened)
                or _directory_identity(before) != _directory_identity(opened)
            ):
                raise OSError
            self._keyframe_descriptor = descriptor
            self._keyframe_identity = _directory_identity(opened)
            self._require_bound_keyframe_directory("关键帧目录在打开时发生变化")
            return True
        except (OSError, VideoDemoError):
            if descriptor >= 0:
                os.close(descriptor)
            self._keyframe_descriptor = -1
            self._keyframe_identity = None
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "关键帧目录非法") from None

    def _require_bound_keyframe_directory(self, message: str) -> None:
        assert self._keyframe_descriptor >= 0
        assert self._keyframe_identity is not None
        try:
            self._require_bound_visual_directory()
            opened = os.fstat(self._keyframe_descriptor)
            through_parent = os.stat(
                "keyframes",
                dir_fd=self._visual_descriptor,
                follow_symlinks=False,
            )
            through_path = os.lstat(self._keyframe_path)
            expected = self._keyframe_identity
            if (
                not _is_private_directory(opened)
                or _directory_identity(opened) != expected
                or _directory_identity(through_parent) != expected
                or _directory_identity(through_path) != expected
            ):
                raise OSError
        except OSError:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message) from None

    def _require_bound_visual_directory(self) -> None:
        assert self._visual_descriptor >= 0
        assert self._visual_identity is not None
        opened = os.fstat(self._visual_descriptor)
        through_path = os.lstat(self._visual_root)
        if (
            not _is_private_directory(opened)
            or _directory_identity(opened) != self._visual_identity
            or _directory_identity(through_path) != self._visual_identity
        ):
            raise OSError

    def _read_private_jpeg(self, name: str, digest: str, max_bytes: int) -> bytes:
        descriptor = -1
        try:
            self._require_bound_keyframe_directory("关键帧目录在读取前发生变化")
            before = os.stat(name, dir_fd=self._keyframe_descriptor, follow_symlinks=False)
            descriptor = os.open(
                name,
                os.O_RDONLY | _no_follow(),
                dir_fd=self._keyframe_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                not _is_private_regular_file(opened)
                or _file_identity(before) != _file_identity(opened)
            ):
                raise OSError
            payload = _read_bounded(descriptor, max_bytes)
            after = os.fstat(descriptor)
            current = os.stat(name, dir_fd=self._keyframe_descriptor, follow_symlinks=False)
            if (
                _file_identity(opened) != _file_identity(after)
                or _file_identity(after) != _file_identity(current)
                or not _is_jpeg(payload)
                or hashlib.sha256(payload).hexdigest() != digest
            ):
                raise OSError
            self._require_bound_keyframe_directory("关键帧目录在读取后发生变化")
            return payload
        except VideoDemoError:
            raise
        except OSError:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "关键帧文件非法") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _write_private_jpeg(self, name: str, payload: bytes) -> tuple[int, int, int]:
        assert self._keyframe_descriptor >= 0
        self._require_bound_keyframe_directory("关键帧目录在创建前发生变化")
        descriptor = -1
        identity: tuple[int, int, int] | None = None
        created_by_call = False
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow(),
                0o600,
                dir_fd=self._keyframe_descriptor,
            )
            created_by_call = True
            os.fchmod(descriptor, 0o600)
            created = os.fstat(descriptor)
            if not _is_private_regular_file(created):
                raise OSError
            identity = (created.st_dev, created.st_ino, created.st_size)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            status = os.fstat(descriptor)
            identity = (status.st_dev, status.st_ino, status.st_size)
            if not _is_private_regular_file(status) or status.st_size != len(payload):
                raise OSError
            os.close(descriptor)
            descriptor = -1
            current = os.stat(name, dir_fd=self._keyframe_descriptor, follow_symlinks=False)
            if _owned_identity(current) != identity or not _is_private_regular_file(current):
                raise OSError
            self._require_bound_keyframe_directory("关键帧目录在创建后发生变化")
            os.fsync(self._keyframe_descriptor)
            verified = self._read_private_jpeg(
                name,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
            if verified != payload:
                raise OSError
            return identity
        except FileExistsError:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "关键帧发布发生同名竞争",
            ) from None
        except VideoDemoError:
            if created_by_call:
                self._unlink_owned_name(name, None)
            raise
        except OSError:
            if created_by_call:
                self._unlink_owned_name(name, None)
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "关键帧安全复制失败") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _unlink_owned_name(
        self,
        name: str,
        identity: tuple[int, int, int] | None,
    ) -> None:
        assert self._keyframe_descriptor >= 0
        try:
            current = os.stat(name, dir_fd=self._keyframe_descriptor, follow_symlinks=False)
            if identity is not None and (
                not _is_private_regular_file(current) or _owned_identity(current) != identity
            ):
                return
            os.unlink(name, dir_fd=self._keyframe_descriptor)
            os.fsync(self._keyframe_descriptor)
        except FileNotFoundError:
            return
        except OSError:
            raise VideoDemoError(
                ErrorCode.VISUAL_RESULT_INVALID,
                "失败的关键帧发布无法安全回滚",
            ) from None

    def _rollback_created(self, created: list[tuple[str, tuple[int, int, int]]]) -> None:
        rollback_error: VideoDemoError | None = None
        for name, identity in reversed(created):
            try:
                self._unlink_owned_name(name, identity)
            except VideoDemoError as error:
                rollback_error = error
        if rollback_error is not None:
            raise rollback_error


def _read_bounded(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1 - total)):
        total += len(chunk)
        if total > max_bytes:
            raise OSError
        chunks.append(chunk)
    return b"".join(chunks)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError
        view = view[written:]


def _is_private_directory(value: os.stat_result) -> bool:
    return stat.S_ISDIR(value.st_mode) and stat.S_IMODE(value.st_mode) == 0o700


def _is_private_regular_file(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_nlink == 1
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_nlink


def _owned_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_size


def _is_keyframe_name(value: str) -> bool:
    return len(value) == 68 and value.endswith(".jpg") and _is_sha256(value[:-4])


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_jpeg(payload: bytes) -> bool:
    return (
        len(payload) >= 4
        and payload.startswith(b"\xff\xd8\xff")
        and payload.endswith(b"\xff\xd9")
    )


def _no_follow() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    if flag == 0:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持 O_NOFOLLOW")
    return flag
