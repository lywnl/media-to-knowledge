from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from video_demo.domain.evidence import KeyframeEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.workspace import safe_runtime_path
from video_demo.visual.candidate_artifacts import CandidateDirectoryLease

_PENDING_RELATIVE = Path("visual/candidate-cleanup.pending")
_MAX_SCAN_ENTRIES = 20_000
_READ_CHUNK_BYTES = 1024 * 1024
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    name: str
    identity: tuple[int, int, int, int, int]

    @property
    def size_bytes(self) -> int:
        return self.identity[2]


@dataclass(slots=True)
class _DirectorySnapshot:
    path: Path
    descriptor: int
    identity: tuple[int, int]
    entries: tuple[_FileSnapshot, ...]

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class PublishedVisualCleaner:
    """只按已提交 3.0 bundle 授权清理当前 Run 的临时视觉制品。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        max_candidate_files: int,
        max_candidate_bytes: int,
        max_published_keyframe_files: int,
        max_published_keyframe_bytes: int,
        max_keyframe_bytes: int,
    ) -> None:
        if min(
            max_candidate_files,
            max_candidate_bytes,
            max_published_keyframe_files,
            max_published_keyframe_bytes,
            max_keyframe_bytes,
        ) < 1:
            raise ValueError("视觉清理预算必须大于 0")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._store = AtomicArtifactStore(self._runtime_root)
        self._max_candidate_files = max_candidate_files
        self._max_candidate_bytes = max_candidate_bytes
        self._max_keyframe_files = max_published_keyframe_files
        self._max_keyframe_bytes = max_published_keyframe_bytes
        self._max_single_keyframe_bytes = max_keyframe_bytes

    def cleanup(
        self,
        run_relative_root: Path,
        keyframes: tuple[KeyframeEvidence, ...],
    ) -> bool:
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        if not run_root.is_dir() or run_root.is_symlink():
            self._invalid("已发布 Run 根非法")
        lease = CandidateDirectoryLease(
            runtime_root=self._runtime_root,
            run_relative_root=run_relative_root,
            mode="EXCLUSIVE",
            wait_timeout_seconds=0.0,
            create_candidate_directory=False,
        )
        try:
            lease.acquire()
        except VideoDemoError as error:
            self._record_cleanup_failure(run_relative_root, run_root)
            if error.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE:
                return False
            raise
        except Exception:
            self._record_cleanup_failure(run_relative_root, run_root)
            raise
        try:
            return self._cleanup_under_lease(run_relative_root, run_root, keyframes)
        except Exception:
            self._record_cleanup_failure(run_relative_root, run_root)
            raise
        finally:
            lease.close()

    def mark_pending(self, run_relative_root: Path) -> None:
        """原子创建或刷新恢复标记；已存在标记不会导致排他写失败。"""

        payload = json.dumps(
            {"schema_version": "1.0.0", "reason": "cleanup-required"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._store.write_bytes(
            run_relative_root / _PENDING_RELATIVE,
            payload,
            max_bytes=1024,
            file_mode=0o600,
        )

    def has_residuals(self, run_relative_root: Path) -> bool:
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        pending = run_root / _PENDING_RELATIVE
        if pending.exists() or pending.is_symlink():
            return True
        for relative in ("visual/candidates", "visual/.keyframe-staging"):
            directory = run_root / relative
            if directory.is_symlink():
                return True
            if not directory.exists():
                continue
            if not directory.is_dir():
                return True
            with os.scandir(directory) as entries:
                if next(entries, None) is not None:
                    return True
        return False

    def _cleanup_under_lease(
        self,
        run_relative_root: Path,
        run_root: Path,
        keyframes: tuple[KeyframeEvidence, ...],
    ) -> bool:
        snapshots: list[_DirectorySnapshot] = []
        try:
            formal, staging = self._scan_published_directories(run_root, snapshots)
            candidates = self._scan_directory(
                run_root / "visual/candidates",
                kind="candidate",
                maximum=self._max_candidate_files,
                maximum_bytes=self._max_candidate_bytes,
            )
            if candidates is not None:
                snapshots.append(candidates)
            self._require_formal_staging_budget(formal, staging)
            self._verify_keyframe_closure(formal, keyframes, exact=False)
            retained = {item.sha256 for item in keyframes}
            self._delete_entries(staging, staging.entries if staging else ())
            formal_orphans = tuple(
                item
                for item in (formal.entries if formal else ())
                if item.name[:-4] not in retained
            )
            self._delete_entries(formal, formal_orphans)
            self._delete_entries(candidates, candidates.entries if candidates else ())
        finally:
            self._close_snapshots(snapshots)

        final_snapshots: list[_DirectorySnapshot] = []
        try:
            final_formal, final_staging = self._scan_published_directories(
                run_root,
                final_snapshots,
            )
            final_candidates = self._scan_directory(
                run_root / "visual/candidates",
                kind="candidate",
                maximum=self._max_candidate_files,
                maximum_bytes=self._max_candidate_bytes,
            )
            if final_candidates is not None:
                final_snapshots.append(final_candidates)
            self._require_formal_staging_budget(final_formal, final_staging)
            self._verify_keyframe_closure(final_formal, keyframes, exact=True)
            self._require_empty(final_staging)
            self._require_empty(final_candidates)
        finally:
            self._close_snapshots(final_snapshots)
        self._remove_pending(run_relative_root)
        return True

    def _verify_keyframe_closure(
        self,
        formal: _DirectorySnapshot | None,
        keyframes: tuple[KeyframeEvidence, ...],
        *,
        exact: bool,
    ) -> None:
        if len(keyframes) > self._max_keyframe_files:
            self._invalid("正式关键帧超过文件数上限")
        entries = {item.name: item for item in formal.entries} if formal is not None else {}
        seen: set[str] = set()
        total = 0
        for item in keyframes:
            expected_name = f"{item.sha256}.jpg"
            if (
                item.mime_type != "image/jpeg"
                or Path(item.relative_path) != Path("visual/keyframes") / expected_name
                or item.sha256 in seen
                or not 0 < item.size_bytes <= self._max_single_keyframe_bytes
                or expected_name not in entries
                or formal is None
            ):
                self._invalid("正式关键帧闭包非法")
            seen.add(item.sha256)
            total += item.size_bytes
            if total > self._max_keyframe_bytes:
                self._invalid("正式关键帧超过字节上限")
            assert formal is not None
            self._require_bound_directory(formal)
            self._read_verified_jpeg_at(
                formal.descriptor,
                entries[expected_name],
                item.sha256,
                item.size_bytes,
                maximum_bytes=self._max_single_keyframe_bytes,
            )
        if exact and set(entries) != {f"{digest}.jpg" for digest in seen}:
            self._invalid("正式关键帧闭包与已发布证据不一致")

    def _scan_directory(
        self,
        directory: Path,
        *,
        kind: str,
        maximum: int,
        maximum_bytes: int,
    ) -> _DirectorySnapshot | None:
        try:
            before = os.lstat(directory)
        except FileNotFoundError:
            return None
        descriptor = -1
        try:
            descriptor = os.open(directory, _directory_flags())
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o700
                or _directory_identity(before) != _directory_identity(opened)
            ):
                raise OSError
            snapshot = _DirectorySnapshot(
                path=directory,
                descriptor=descriptor,
                identity=_directory_identity(opened),
                entries=(),
            )
            with os.scandir(descriptor) as scanned:
                names: list[str] = []
                for entry in scanned:
                    if len(names) >= min(maximum, _MAX_SCAN_ENTRIES):
                        self._invalid("视觉目录扫描超限")
                    names.append(entry.name)
            names.sort()
            entries: list[_FileSnapshot] = []
            total_bytes = 0
            for name in names:
                self._require_name(name, kind)
                status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                file_snapshot = _FileSnapshot(name=name, identity=_file_identity(status))
                self._require_private_file(status)
                total_bytes += status.st_size
                if total_bytes > maximum_bytes:
                    self._invalid("视觉目录字节扫描超限")
                if kind in {"keyframe", "candidate"}:
                    self._read_verified_jpeg_at(
                        descriptor,
                        file_snapshot,
                        name[:-4],
                        status.st_size,
                        maximum_bytes=(
                            self._max_single_keyframe_bytes
                            if kind == "keyframe"
                            else self._max_candidate_bytes
                        ),
                    )
                entries.append(file_snapshot)
            snapshot.entries = tuple(entries)
            self._require_bound_directory(snapshot)
            descriptor = -1
            return snapshot
        except OSError:
            self._invalid("视觉制品目录非法")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _delete_entries(
        self,
        directory: _DirectorySnapshot | None,
        entries: tuple[_FileSnapshot, ...],
    ) -> None:
        if directory is None:
            return
        for item in entries:
            self._require_bound_directory(directory)
            try:
                current = os.stat(
                    item.name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                self._invalid("视觉条目在删除前发生变化")
            if _file_identity(current) != item.identity:
                self._invalid("视觉条目在删除前发生变化")
            os.unlink(item.name, dir_fd=directory.descriptor)
            os.fsync(directory.descriptor)
            self._require_bound_directory(directory)

    def _require_empty(self, directory: _DirectorySnapshot | None) -> None:
        if directory is None:
            return
        self._require_bound_directory(directory)
        with os.scandir(directory.descriptor) as entries:
            if next(entries, None) is not None:
                self._invalid("视觉临时目录清理不完整")

    def _read_verified_jpeg_at(
        self,
        directory_descriptor: int,
        snapshot: _FileSnapshot,
        digest: str,
        size_bytes: int,
        *,
        maximum_bytes: int,
    ) -> None:
        descriptor = -1
        try:
            before = os.stat(snapshot.name, dir_fd=directory_descriptor, follow_symlinks=False)
            if _file_identity(before) != snapshot.identity:
                raise OSError
            descriptor = os.open(
                snapshot.name,
                os.O_RDONLY | _no_follow(),
                dir_fd=directory_descriptor,
            )
            opened = os.fstat(descriptor)
            self._require_private_file(opened)
            if (
                _file_identity(opened) != snapshot.identity
                or opened.st_size != size_bytes
                or not 0 < size_bytes <= maximum_bytes
            ):
                raise OSError
            digest_builder = hashlib.sha256()
            prefix = b""
            suffix = b""
            total = 0
            while total <= maximum_bytes:
                chunk = os.read(
                    descriptor,
                    min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total),
                )
                if not chunk:
                    break
                digest_builder.update(chunk)
                prefix = (prefix + chunk)[:3]
                suffix = (suffix + chunk)[-2:]
                total += len(chunk)
            after = os.fstat(descriptor)
            current = os.stat(snapshot.name, dir_fd=directory_descriptor, follow_symlinks=False)
            if (
                total != size_bytes
                or _file_identity(after) != snapshot.identity
                or _file_identity(current) != snapshot.identity
                or prefix != b"\xff\xd8\xff"
                or suffix != b"\xff\xd9"
                or digest_builder.hexdigest() != digest
            ):
                raise OSError
        except OSError:
            self._invalid("视觉 JPEG 完整性非法")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _require_name(name: str, kind: str) -> None:
        if kind in {"keyframe", "candidate"}:
            valid = len(name) == 68 and name.endswith(".jpg") and _is_sha256(name[:-4])
        else:
            token = name.removesuffix(".pending")
            valid = (
                name.endswith(".pending")
                and len(token) == 32
                and all(character in "0123456789abcdef" for character in token)
            )
        if not valid:
            PublishedVisualCleaner._invalid("视觉目录包含未知条目")

    @staticmethod
    def _require_private_file(status: os.stat_result) -> None:
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_nlink != 1
        ):
            PublishedVisualCleaner._invalid("视觉条目类型非法")

    @staticmethod
    def _require_bound_directory(directory: _DirectorySnapshot) -> None:
        try:
            opened = os.fstat(directory.descriptor)
            through_path = os.lstat(directory.path)
        except OSError:
            PublishedVisualCleaner._invalid("视觉制品目录在清理期间发生变化")
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or _directory_identity(opened) != directory.identity
            or _directory_identity(through_path) != directory.identity
        ):
            PublishedVisualCleaner._invalid("视觉制品目录在清理期间发生变化")

    def _remove_pending(self, run_relative_root: Path) -> None:
        visual_path = safe_runtime_path(self._runtime_root, run_relative_root / "visual")
        directory = self._open_directory(visual_path)
        if directory is None:
            return
        file_descriptor = -1
        try:
            name = _PENDING_RELATIVE.name
            try:
                before = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return
            self._require_private_file(before)
            snapshot = _FileSnapshot(name=name, identity=_file_identity(before))
            file_descriptor = os.open(
                name,
                os.O_RDONLY | _no_follow(),
                dir_fd=directory.descriptor,
            )
            opened = os.fstat(file_descriptor)
            self._require_private_file(opened)
            current = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
            if (
                _file_identity(opened) != snapshot.identity
                or _file_identity(current) != snapshot.identity
            ):
                self._invalid("视觉清理标记在删除前发生变化")
            os.unlink(name, dir_fd=directory.descriptor)
            os.fsync(directory.descriptor)
            self._require_bound_directory(directory)
            try:
                os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                self._invalid("视觉清理标记在删除后重新出现")
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            directory.close()

    def _open_directory(self, directory: Path) -> _DirectorySnapshot | None:
        try:
            before = os.lstat(directory)
        except FileNotFoundError:
            return None
        descriptor = -1
        try:
            descriptor = os.open(directory, _directory_flags())
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o700
                or _directory_identity(before) != _directory_identity(opened)
            ):
                raise OSError
            snapshot = _DirectorySnapshot(
                path=directory,
                descriptor=descriptor,
                identity=_directory_identity(opened),
                entries=(),
            )
            self._require_bound_directory(snapshot)
            descriptor = -1
            return snapshot
        except OSError:
            self._invalid("视觉制品目录非法")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _require_formal_staging_budget(
        self,
        formal: _DirectorySnapshot | None,
        staging: _DirectorySnapshot | None,
    ) -> None:
        entries = (
            *(() if formal is None else formal.entries),
            *(() if staging is None else staging.entries),
        )
        if (
            len(entries) > self._max_keyframe_files
            or sum(item.size_bytes for item in entries) > self._max_keyframe_bytes
        ):
            self._invalid("正式关键帧与 staging 组合预算超限")

    @staticmethod
    def _snapshot_bytes(snapshot: _DirectorySnapshot | None) -> int:
        return 0 if snapshot is None else sum(item.size_bytes for item in snapshot.entries)

    def _scan_published_directories(
        self,
        run_root: Path,
        snapshots: list[_DirectorySnapshot],
    ) -> tuple[_DirectorySnapshot | None, _DirectorySnapshot | None]:
        staging = self._scan_directory(
            run_root / "visual/.keyframe-staging",
            kind="staging",
            maximum=self._max_keyframe_files,
            maximum_bytes=self._max_keyframe_bytes,
        )
        if staging is not None:
            snapshots.append(staging)
        used_files = 0 if staging is None else len(staging.entries)
        used_bytes = self._snapshot_bytes(staging)
        remaining_files = self._max_keyframe_files - used_files
        remaining_bytes = self._max_keyframe_bytes - used_bytes
        if remaining_files < 0 or remaining_bytes < 0:
            self._invalid("正式关键帧与 staging 组合预算超限")
        formal = self._scan_directory(
            run_root / "visual/keyframes",
            kind="keyframe",
            maximum=remaining_files,
            maximum_bytes=remaining_bytes,
        )
        if formal is not None:
            snapshots.append(formal)
        return formal, staging

    def _record_cleanup_failure(self, run_relative_root: Path, run_root: Path) -> None:
        with suppress(Exception):
            self.mark_pending(run_relative_root)
        try:
            files, bytes_used = self._residual_usage(run_root)
        except Exception:
            _LOGGER.warning(
                "视觉清理失败，run_id=%s，剩余统计不可用",
                run_relative_root.name,
            )
            return
        _LOGGER.warning(
            "视觉清理失败，run_id=%s，剩余文件数=%d，剩余字节=%d",
            run_relative_root.name,
            files,
            bytes_used,
        )

    def _residual_usage(self, run_root: Path) -> tuple[int, int]:
        snapshots: list[_DirectorySnapshot] = []
        try:
            candidates = self._scan_directory(
                run_root / "visual/candidates",
                kind="candidate",
                maximum=self._max_candidate_files,
                maximum_bytes=self._max_candidate_bytes,
            )
            if candidates is not None:
                snapshots.append(candidates)
            formal, staging = self._scan_published_directories(run_root, snapshots)
            self._require_formal_staging_budget(formal, staging)
            entries = tuple(item for snapshot in snapshots for item in snapshot.entries)
            return len(entries), sum(item.size_bytes for item in entries)
        finally:
            self._close_snapshots(snapshots)

    @staticmethod
    def _close_snapshots(snapshots: list[_DirectorySnapshot]) -> None:
        for snapshot in reversed(snapshots):
            snapshot.close()

    @staticmethod
    def _invalid(message: str) -> NoReturn:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, message)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or not isinstance(directory, int):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全视觉目录")
    return os.O_RDONLY | no_follow | directory


def _no_follow() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全视觉文件")
    return no_follow


def _directory_identity(status: os.stat_result) -> tuple[int, int]:
    return (status.st_dev, status.st_ino)


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        stat.S_IMODE(status.st_mode),
        status.st_nlink,
    )
