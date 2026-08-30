from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.scope import Scope
from video_demo.storage.workspace import atomic_replace, safe_runtime_path, validate_path_component

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MediaObjectRecord:
    object_ref: str
    original_filename: str
    declared_mime: str
    detected_mime: str
    size_bytes: int
    sha256: str
    relative_path: str
    scope_key: str


class BinaryMediaObjectStore:
    """为非视频媒体提供安全的内容寻址文件写入和运行目录物化。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        max_bytes: int,
        media_kind: str,
        allowed_mimes: Iterable[str],
        extension_to_mime: dict[str, str],
        detector: Callable[[bytes], str | None],
        error_prefix: str,
    ) -> None:
        self.runtime_root = runtime_root.expanduser().resolve(strict=False)
        self.max_bytes = max_bytes
        self.media_kind = media_kind
        self._allowed_mimes = frozenset(allowed_mimes)
        self._extension_to_mime = extension_to_mime
        self._detector = detector
        self._error_prefix = error_prefix
        if max_bytes < 1 or not self._allowed_mimes or not extension_to_mime:
            raise ValueError("媒体存储配置非法")

    @staticmethod
    def scope_key(scope: Scope) -> str:
        value = "\x00".join(
            (scope.tenant_id, scope.application_id, scope.knowledge_base_id),
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:24]

    def ingest(
        self,
        stream: BinaryIO,
        filename: str,
        declared_mime: str,
        scope: Scope,
    ) -> MediaObjectRecord:
        extension = self._validate_filename_and_mime(filename, declared_mime)
        object_ref = f"obj_{uuid.uuid4().hex}"
        scope_key = self.scope_key(scope)
        relative_path = (
            Path(self.media_kind.lower()) / scope_key / object_ref / f"source{extension}"
        )
        destination = safe_runtime_path(self.runtime_root, relative_path)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            size_bytes, sha256, header = self._write_quarantined(stream, temporary)
            detected_mime = self._detector(header)
            if detected_mime != declared_mime or detected_mime not in self._allowed_mimes:
                raise VideoDemoError(
                    self._error_code("MIME_MISMATCH"),
                    "文件声明 MIME 与内容签名不一致",
                )
            atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return MediaObjectRecord(
            object_ref=object_ref,
            original_filename=filename,
            declared_mime=declared_mime,
            detected_mime=detected_mime,
            size_bytes=size_bytes,
            sha256=sha256,
            relative_path=relative_path.as_posix(),
            scope_key=scope_key,
        )

    def materialize(
        self,
        scope: Scope,
        record: MediaObjectRecord,
        run_id: str,
        expected_sha256: str,
    ) -> Path:
        validate_path_component(run_id, "run_id")
        if record.scope_key != self.scope_key(scope):
            raise VideoDemoError(self._error_code("OBJECT_NOT_FOUND"), "媒体对象不存在")
        source = safe_runtime_path(self.runtime_root, Path(record.relative_path))
        if source.is_symlink() or not source.is_file():
            raise VideoDemoError(self._error_code("OBJECT_NOT_FOUND"), "媒体对象不存在")
        if source.stat().st_size != record.size_bytes:
            raise VideoDemoError(self._error_code("DIGEST_MISMATCH"), "媒体对象大小校验失败")
        if _sha256_file(source) != expected_sha256 or expected_sha256 != record.sha256:
            raise VideoDemoError(self._error_code("DIGEST_MISMATCH"), "媒体对象摘要校验失败")
        destination = safe_runtime_path(
            self.runtime_root,
            Path("runs")
            / record.scope_key
            / run_id
            / "input"
            / f"source{Path(record.relative_path).suffix}",
        )
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open("rb") as source_stream, temporary.open("xb") as output:
                shutil.copyfileobj(source_stream, output, length=_CHUNK_SIZE)
                output.flush()
                os.fsync(output.fileno())
            if _sha256_file(temporary) != expected_sha256:
                raise VideoDemoError(self._error_code("DIGEST_MISMATCH"), "运行输入摘要校验失败")
            atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def _validate_filename_and_mime(self, filename: str, declared_mime: str) -> str:
        candidate = Path(filename)
        if (
            not filename
            or candidate.name != filename
            or candidate.is_absolute()
            or "\x00" in filename
        ):
            raise VideoDemoError(self._error_code("INPUT_INVALID"), "媒体文件名不能包含路径")
        extension = candidate.suffix.lower()
        expected_mime = self._extension_to_mime.get(extension)
        if expected_mime is None:
            raise VideoDemoError(self._error_code("FORMAT_UNSUPPORTED"), "不支持该媒体格式")
        if declared_mime != expected_mime or declared_mime not in self._allowed_mimes:
            raise VideoDemoError(self._error_code("MIME_MISMATCH"), "声明 MIME 与扩展名不一致")
        return extension

    def _write_quarantined(self, stream: BinaryIO, temporary: Path) -> tuple[int, str, bytes]:
        digest = hashlib.sha256()
        size_bytes = 0
        header = bytearray()
        with temporary.open("xb") as output:
            while chunk := stream.read(_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > self.max_bytes:
                    raise VideoDemoError(self._error_code("FILE_TOO_LARGE"), "媒体文件超过大小限制")
                if len(header) < 4096:
                    header.extend(chunk[: 4096 - len(header)])
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size_bytes == 0:
            raise VideoDemoError(self._error_code("FILE_EMPTY"), "媒体文件不能为空")
        return size_bytes, digest.hexdigest(), bytes(header)

    def _error_code(self, suffix: str) -> ErrorCode:
        return ErrorCode[f"{self._error_prefix}_{suffix}"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
