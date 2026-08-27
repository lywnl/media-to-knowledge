from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import (
    atomic_replace,
    reject_symlink_components,
    safe_runtime_path,
)

RESULT_BUNDLE_ENVELOPE_SCHEMA_VERSION = "1.0.0"


class ArtifactReceipt(FrozenModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    schema_version: str = Field(min_length=1, max_length=32)
    sha256: Sha256
    upstream_sha256: Sha256


class ArtifactBytesReceipt(FrozenModel):
    """可用于校验任意非空字节制品的最小回执。"""

    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    size_bytes: int = Field(gt=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        try:
            return _validated_bytes_relative_path(Path(value)).as_posix()
        except VideoDemoError:
            raise ValueError("字节制品回执路径必须是安全相对路径") from None


def canonical_artifact_envelope_bytes(
    payload: dict[str, Any] | list[Any],
    schema_version: str,
    upstream_sha256: str,
) -> bytes:
    """按阶段产物唯一规范编码 envelope。"""

    return json.dumps(
        {
            "schema_version": schema_version,
            "upstream_sha256": upstream_sha256,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class AtomicArtifactStore:
    """将阶段 JSON 以规范 UTF-8 和原子替换方式写入运行目录。"""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root.expanduser().resolve(strict=False)

    def write_json(
        self,
        relative_path: Path,
        payload: dict[str, Any] | list[Any],
        *,
        schema_version: str,
        upstream_sha256: str,
        file_mode: int | None = None,
        exclusive: bool = False,
        max_bytes: int | None = None,
    ) -> ArtifactReceipt:
        if file_mode is not None and not 0 <= file_mode <= 0o777:
            raise ValueError("文件权限模式非法")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("阶段产物写入上限必须大于 0")
        destination = safe_runtime_path(self.runtime_root, relative_path)
        if file_mode is not None:
            destination = reject_symlink_components(
                self.runtime_root,
                destination,
                message="私有产物路径不能包含符号链接",
            )
        encoded = canonical_artifact_envelope_bytes(
            payload,
            schema_version,
            upstream_sha256,
        )
        if max_bytes is not None and len(encoded) > max_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "阶段产物超过写入上限")
        digest = hashlib.sha256(encoded).hexdigest()
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            creation_mode = file_mode if file_mode is not None else 0o666
            descriptor = os.open(temporary, flags, creation_mode)
            try:
                if file_mode is not None:
                    os.fchmod(descriptor, file_mode)
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if exclusive:
                os.link(temporary, destination, follow_symlinks=False)
                temporary.unlink()
            else:
                atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return ArtifactReceipt(
            relative_path=relative_path.as_posix(),
            schema_version=schema_version,
            sha256=digest,
            upstream_sha256=upstream_sha256,
        )

    def write_bytes(
        self,
        relative_path: Path,
        content: bytes,
        *,
        max_bytes: int,
        file_mode: int = 0o600,
        exclusive: bool = False,
    ) -> ArtifactBytesReceipt:
        """以私有权限原子发布非空字节，并返回内容摘要回执。"""

        _require_positive_byte_limit(max_bytes)
        if not 0 <= file_mode <= 0o777:
            raise ValueError("文件权限模式非法")
        if not content:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品不能为空")
        if len(content) > max_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "字节制品超过写入上限")

        validated_path = _validated_bytes_relative_path(relative_path)
        parent_descriptor = _open_private_directory_tree(
            self.runtime_root,
            validated_path.parent,
            create=True,
        )
        temporary_name = f".{validated_path.name}.{uuid.uuid4().hex}.part"
        descriptor: int | None = None
        try:
            _reject_symlink_leaf(parent_descriptor, validated_path.name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _require_no_follow()
            descriptor = os.open(temporary_name, flags, file_mode, dir_fd=parent_descriptor)
            os.fchmod(descriptor, file_mode)
            with os.fdopen(descriptor, "wb", closefd=False) as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.close(descriptor)
            descriptor = None

            if exclusive:
                os.link(
                    temporary_name,
                    validated_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            else:
                os.replace(
                    temporary_name,
                    validated_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            os.fsync(parent_descriptor)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)

        return ArtifactBytesReceipt(
            relative_path=validated_path.as_posix(),
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )

    def read_verified_bytes(
        self,
        receipt: ArtifactBytesReceipt,
        *,
        max_bytes: int,
    ) -> bytes:
        """有界读取普通文件，并校验路径、身份、大小和内容摘要。"""

        _require_positive_byte_limit(max_bytes)
        validated_path = _validated_bytes_relative_path(Path(receipt.relative_path))
        parent_descriptor = _open_private_directory_tree(
            self.runtime_root,
            validated_path.parent,
            create=False,
        )
        try:
            before = os.lstat(
                validated_path.name,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            os.close(parent_descriptor)
            raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "字节制品不存在") from None
        except OSError as error:
            os.close(parent_descriptor)
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品文件非法") from error
        if stat.S_ISLNK(before.st_mode):
            os.close(parent_descriptor)
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "字节制品路径包含符号链接")
        if not stat.S_ISREG(before.st_mode):
            os.close(parent_descriptor)
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品文件非法")

        descriptor: int | None = None
        try:
            descriptor = os.open(
                validated_path.name,
                os.O_RDONLY | _require_no_follow(),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            _require_same_bytes_file(before, opened)
            chunks: list[bytes] = []
            total = 0
            while True:
                remaining = max_bytes + 1 - total
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise VideoDemoError(
                        ErrorCode.ARTIFACT_SCHEMA_INVALID,
                        "字节制品超过读取上限",
                    )
            after = os.fstat(descriptor)
            _require_same_bytes_file(opened, after)
            try:
                current = os.lstat(
                    validated_path.name,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise VideoDemoError(
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    "字节制品读取期间发生变化",
                ) from error
            _require_same_bytes_file(after, current)
        except FileNotFoundError:
            raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "字节制品不存在") from None
        except OSError as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品文件非法") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)

        encoded = b"".join(chunks)
        if len(encoded) != receipt.size_bytes:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品大小与回执不一致")
        if hashlib.sha256(encoded).hexdigest() != receipt.sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "字节制品摘要不匹配")
        return encoded

    def read_verified_json(self, receipt: ArtifactReceipt) -> dict[str, Any] | list[Any]:
        return self.read_verified_json_limited(receipt)

    def list_regular_artifacts(
        self,
        relative_directory: Path,
        *,
        max_entries: int,
    ) -> tuple[str, ...]:
        """有界列出安全目录中的普通文件；遇未知类型失败关闭。"""

        if max_entries <= 0:
            raise ValueError("制品目录条目上限必须大于 0")
        validated = _validated_bytes_relative_path(relative_directory)
        directory = _open_private_directory_tree(self.runtime_root, validated, create=False)
        names: list[str] = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if len(names) >= max_entries:
                        raise VideoDemoError(
                            ErrorCode.ARTIFACT_SCHEMA_INVALID,
                            "制品目录条目超过恢复上限",
                        )
                    if not entry.is_file(follow_symlinks=False):
                        raise VideoDemoError(
                            ErrorCode.ARTIFACT_SCHEMA_INVALID,
                            "制品目录包含非普通文件",
                        )
                    names.append(entry.name)
        except OSError as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "制品目录无法安全枚举",
            ) from error
        finally:
            os.close(directory)
        return tuple(sorted(names))

    def inspect_artifact_bytes(
        self,
        relative_path: Path,
        *,
        max_bytes: int,
    ) -> tuple[bytes, ArtifactBytesReceipt]:
        """有界读取当前普通文件，并返回供二次身份/摘要复验的回执。"""

        _require_positive_byte_limit(max_bytes)
        encoded = self._read_limited_artifact_bytes(relative_path.as_posix(), max_bytes)
        return encoded, ArtifactBytesReceipt(
            relative_path=relative_path.as_posix(),
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )

    def discard_bytes(self, receipt: ArtifactBytesReceipt) -> bool:
        """仅在普通文件身份和摘要均未变化时按目录描述符删除。"""

        validated = _validated_bytes_relative_path(Path(receipt.relative_path))
        try:
            parent = _open_private_directory_tree(
                self.runtime_root,
                validated.parent,
                create=False,
            )
        except VideoDemoError:
            return False
        descriptor: int | None = None
        try:
            before = os.lstat(validated.name, dir_fd=parent)
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
                return False
            descriptor = os.open(
                validated.name,
                os.O_RDONLY | _require_no_follow(),
                dir_fd=parent,
            )
            opened = os.fstat(descriptor)
            _require_same_bytes_file(before, opened)
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                if size > receipt.size_bytes:
                    return False
            current = os.lstat(validated.name, dir_fd=parent)
            _require_same_bytes_file(opened, current)
            if size != receipt.size_bytes or digest.hexdigest() != receipt.sha256:
                return False
            os.unlink(validated.name, dir_fd=parent)
            os.fsync(parent)
            return True
        except (OSError, VideoDemoError):
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def discard_artifact(self, receipt: ArtifactReceipt, *, max_bytes: int) -> bool:
        """以 JSON 制品的已知摘要和有界大小复用安全字节删除。"""

        _require_positive_byte_limit(max_bytes)
        try:
            encoded = self._read_limited_artifact_bytes(receipt.relative_path, max_bytes)
        except VideoDemoError:
            return False
        bytes_receipt = ArtifactBytesReceipt(
            relative_path=receipt.relative_path,
            sha256=receipt.sha256,
            size_bytes=len(encoded),
        )
        return self.discard_bytes(bytes_receipt)

    def read_verified_json_limited(
        self,
        receipt: ArtifactReceipt,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        if max_bytes is not None and max_bytes < 1:
            raise ValueError("阶段产物读取上限必须大于 0")
        if max_bytes is not None:
            encoded = self._read_limited_artifact_bytes(receipt.relative_path, max_bytes)
        else:
            artifact = safe_runtime_path(self.runtime_root, Path(receipt.relative_path))
            if artifact.is_symlink() or not artifact.is_file():
                raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "阶段产物不存在")
            encoded = artifact.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != receipt.sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_DIGEST_MISMATCH, "阶段产物摘要不匹配")
        try:
            envelope: object = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "阶段产物不是合法 JSON",
            ) from error
        if not isinstance(envelope, dict):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物外层必须是对象")
        if envelope.get("schema_version") != receipt.schema_version:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物 Schema 版本不匹配")
        if envelope.get("upstream_sha256") != receipt.upstream_sha256:
            raise VideoDemoError(ErrorCode.ARTIFACT_UPSTREAM_MISMATCH, "阶段产物上游摘要不匹配")
        payload = envelope.get("payload")
        if not isinstance(payload, (dict, list)):
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物 payload 类型非法")
        return payload

    def _read_limited_artifact_bytes(self, relative_path: str, max_bytes: int) -> bytes:
        validated = _validated_bytes_relative_path(Path(relative_path))
        parent = _open_private_directory_tree(self.runtime_root, validated.parent, create=False)
        descriptor: int | None = None
        try:
            before = os.lstat(validated.name, dir_fd=parent)
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物文件非法")
            descriptor = os.open(validated.name, os.O_RDONLY | _require_no_follow(), dir_fd=parent)
            opened = os.fstat(descriptor)
            _require_same_bytes_file(before, opened)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物超过读取上限")
            current = os.lstat(validated.name, dir_fd=parent)
            _require_same_bytes_file(opened, current)
            return b"".join(chunks)
        except FileNotFoundError:
            raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "阶段产物不存在") from None
        except OSError as error:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "阶段产物文件非法") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def discard(self, receipt: ArtifactReceipt) -> bool:
        """仅删除摘要仍与本次写入一致的未发布产物。"""

        artifact = reject_symlink_components(
            self.runtime_root,
            self.runtime_root / receipt.relative_path,
            message="待删除产物路径不能包含符号链接",
        )
        if artifact.is_symlink() or not artifact.is_file():
            return False
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt.sha256:
            return False
        artifact.unlink()
        return True


def _require_positive_byte_limit(max_bytes: int) -> None:
    if max_bytes <= 0:
        raise ValueError("字节制品大小上限必须大于 0")


def _require_no_follow() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全制品读写")
    return no_follow


def _validated_bytes_relative_path(relative_path: Path) -> Path:
    if (
        "\x00" in str(relative_path)
        or relative_path.is_absolute()
        or relative_path == Path()
        or ".." in relative_path.parts
    ):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "字节制品路径非法")
    return relative_path


def _open_private_directory_tree(
    runtime_root: Path,
    relative_directory: Path,
    *,
    create: bool,
) -> int:
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = _open_verified_directory(runtime_root)
    next_descriptor: int | None = None
    try:
        os.fchmod(descriptor, 0o700)
        for component in relative_directory.parts:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
            next_descriptor = os.open(
                component,
                os.O_RDONLY | _require_directory_flag() | _require_no_follow(),
                dir_fd=descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise OSError
            os.fchmod(next_descriptor, 0o700)
            os.close(descriptor)
            descriptor = next_descriptor
            next_descriptor = None
        return descriptor
    except FileNotFoundError:
        if next_descriptor is not None:
            os.close(next_descriptor)
        os.close(descriptor)
        raise VideoDemoError(ErrorCode.ARTIFACT_NOT_FOUND, "字节制品目录不存在") from None
    except VideoDemoError:
        if next_descriptor is not None:
            os.close(next_descriptor)
        os.close(descriptor)
        raise
    except OSError as error:
        if next_descriptor is not None:
            os.close(next_descriptor)
        os.close(descriptor)
        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "字节制品路径不能包含符号链接",
        ) from error


def _open_verified_directory(directory: Path) -> int:
    descriptor: int | None = None
    try:
        before = os.lstat(directory)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise OSError
        descriptor = os.open(
            directory,
            os.O_RDONLY | _require_directory_flag() | _require_no_follow(),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise OSError
        return descriptor
    except VideoDemoError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品目录非法") from error


def _require_directory_flag() -> int:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if not isinstance(directory_flag, int):
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全制品目录")
    return directory_flag


def _reject_symlink_leaf(parent_descriptor: int, name: str) -> None:
    try:
        current = os.lstat(name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return
    except OSError as error:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品目标非法") from error
    if stat.S_ISLNK(current.st_mode):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "字节制品路径包含符号链接")


def _require_same_bytes_file(before: os.stat_result, after: os.stat_result) -> None:
    if _bytes_file_identity(before) != _bytes_file_identity(after):
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "字节制品读取期间发生变化")


def _bytes_file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
