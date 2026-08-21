from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from video_demo.config import resolve_workspace_path
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.repositories import Scope
from video_demo.storage.workspace import atomic_replace, safe_runtime_path, validate_path_component

_MIME_BY_EXTENSION = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class VideoObjectRecord:
    object_ref: str
    original_filename: str
    declared_mime: str
    detected_mime: str
    size_bytes: int
    sha256: str
    relative_path: str
    scope_key: str


def detect_video_mime(header: bytes) -> str | None:
    """基于容器签名进行第一层探测; 后续仍必须经过 ffprobe 解码门禁。"""

    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12]
        return "video/quicktime" if brand.startswith(b"qt") else "video/mp4"
    if header.startswith(b"\x1aE\xdf\xa3"):
        lowered = header.lower()
        return "video/webm" if b"webm" in lowered else "video/x-matroska"
    return None


class LocalVideoObjectStore:
    """只在工作区运行目录中持久化视频对象。"""

    def __init__(
        self,
        runtime_root: Path,
        *,
        max_video_bytes: int,
        mime_detector: Callable[[bytes], str | None] = detect_video_mime,
    ) -> None:
        self.runtime_root = runtime_root.expanduser().resolve(strict=False)
        self.max_video_bytes = max_video_bytes
        self._mime_detector = mime_detector

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
    ) -> VideoObjectRecord:
        extension = self._validate_filename_and_mime(filename, declared_mime)
        object_ref = f"obj_{uuid.uuid4().hex}"
        scope_key = self.scope_key(scope)
        relative_path = Path("objects") / scope_key / object_ref / f"source{extension}"
        destination = safe_runtime_path(self.runtime_root, relative_path)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            size_bytes, sha256, header = self._write_quarantined(stream, temporary)
            detected_mime = self._mime_detector(header)
            expected_mime = _MIME_BY_EXTENSION[extension]
            if detected_mime != declared_mime or detected_mime != expected_mime:
                raise VideoDemoError(
                    ErrorCode.VIDEO_MIME_MISMATCH,
                    "文件扩展名、声明 MIME 与容器签名不一致",
                    {"declared_mime": declared_mime, "detected_mime": detected_mime},
                )
            atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        return VideoObjectRecord(
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
        record: VideoObjectRecord,
        run_id: str,
        expected_sha256: str,
    ) -> Path:
        validate_path_component(run_id, "run_id")
        if record.scope_key != self.scope_key(scope):
            raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")

        source = safe_runtime_path(self.runtime_root, Path(record.relative_path))
        if source.is_symlink() or not source.is_file():
            if source.is_symlink():
                raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "视频对象符号链接越界")
            raise VideoDemoError(ErrorCode.VIDEO_OBJECT_NOT_FOUND, "视频对象不存在")
        source = resolve_workspace_path(self.runtime_root, source)
        actual_sha256 = _sha256_file(source)
        if actual_sha256 != expected_sha256 or actual_sha256 != record.sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "视频对象摘要校验失败")

        extension = Path(record.relative_path).suffix
        destination = safe_runtime_path(
            self.runtime_root,
            Path("runs") / record.scope_key / run_id / "input" / f"source{extension}",
        )
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open("rb") as source_stream, temporary.open("xb") as output:
                shutil.copyfileobj(source_stream, output, length=_CHUNK_SIZE)
                output.flush()
                os.fsync(output.fileno())
            if _sha256_file(temporary) != expected_sha256:
                raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "运行输入摘要校验失败")
            atomic_replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def resolve_record_path(self, record: VideoObjectRecord) -> Path:
        return safe_runtime_path(self.runtime_root, Path(record.relative_path))

    def _validate_filename_and_mime(self, filename: str, declared_mime: str) -> str:
        candidate = Path(filename)
        has_path = candidate.name != filename or candidate.is_absolute()
        if not filename or has_path or "\x00" in filename:
            raise VideoDemoError(ErrorCode.INVALID_VIDEO_FILENAME, "视频文件名不能包含路径")
        extension = candidate.suffix.lower()
        expected_mime = _MIME_BY_EXTENSION.get(extension)
        if expected_mime is None:
            raise VideoDemoError(ErrorCode.VIDEO_FORMAT_UNSUPPORTED, "不支持该视频格式")
        if declared_mime != expected_mime:
            raise VideoDemoError(ErrorCode.VIDEO_MIME_MISMATCH, "声明 MIME 与扩展名不一致")
        return extension

    def _write_quarantined(self, stream: BinaryIO, temporary: Path) -> tuple[int, str, bytes]:
        digest = hashlib.sha256()
        size_bytes = 0
        header = bytearray()
        with temporary.open("xb") as output:
            while chunk := stream.read(_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > self.max_video_bytes:
                    raise VideoDemoError(ErrorCode.VIDEO_FILE_TOO_LARGE, "视频文件超过大小限制")
                if len(header) < 4096:
                    header.extend(chunk[: 4096 - len(header)])
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size_bytes == 0:
            raise VideoDemoError(ErrorCode.VIDEO_FILE_EMPTY, "视频文件不能为空")
        return size_bytes, digest.hexdigest(), bytes(header)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
