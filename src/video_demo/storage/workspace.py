from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path

from video_demo.config import resolve_workspace_path
from video_demo.errors import ErrorCode, VideoDemoError

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{3,128}$")


def validate_path_component(value: str, field_name: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise VideoDemoError(
            ErrorCode.INVALID_PATH_COMPONENT,
            f"{field_name} 不是合法路径标识",
        )
    return value


def safe_runtime_path(runtime_root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "运行路径不能是绝对路径")
    return resolve_workspace_path(runtime_root, relative_path)


def atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def reject_symlink_components(root: Path, candidate: Path, *, message: str) -> Path:
    """按词法路径逐组件拒绝符号链接，并确认解析后仍在根内。"""

    resolved_root = root.expanduser().resolve(strict=False)
    lexical = candidate.expanduser()
    if not lexical.is_absolute():
        lexical = resolved_root / lexical
    try:
        relative = lexical.relative_to(resolved_root)
    except ValueError:
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message) from None
    current = resolved_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message)
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message)
    return resolved


def verified_run_file(
    runtime_root: Path,
    run_relative_root: Path,
    candidate: Path,
    *,
    expected_sha256: str | None = None,
    digest: Callable[[Path], str] | None = None,
    message: str = "产物必须位于当前运行目录内",
) -> Path:
    """验证普通文件的当前 run 归属、无符号链接和可选摘要。"""

    run_root = safe_runtime_path(runtime_root, run_relative_root)
    path = reject_symlink_components(runtime_root, candidate, message=message)
    if not path.is_relative_to(run_root) or not path.is_file():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message)
    if expected_sha256 is not None:
        if digest is None:
            raise ValueError("提供 expected_sha256 时必须提供 digest")
        if digest(path) != expected_sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "媒体产物摘要校验失败")
    return path


def verified_mp4_file(
    runtime_root: Path,
    allowed_relative_root: Path,
    candidate: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    max_size_bytes: int,
    message: str,
) -> Path:
    """流式验证当前运行目录内、受限且具有 ISO BMFF ftyp 的 MP4。"""

    if expected_size_bytes < 1:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "MP4 声明大小必须大于 0")
    if max_size_bytes < 1 or expected_size_bytes > max_size_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "MP4 大小超过限制")
    path = verified_run_file(
        runtime_root,
        allowed_relative_root,
        candidate,
        message=message,
    )
    digest = hashlib.sha256()
    actual_size = 0
    header = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            actual_size += len(chunk)
            if actual_size > max_size_bytes:
                raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "MP4 大小超过限制")
            digest.update(chunk)
            if len(header) < 32:
                header += chunk[: 32 - len(header)]
    if actual_size != expected_size_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "MP4 实际大小与声明不一致")
    if digest.hexdigest() != expected_sha256:
        raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "MP4 摘要校验失败")
    if not _has_iso_bmff_ftyp(header, actual_size):
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "MP4 缺少合法 ISO BMFF ftyp")
    return path


def _has_iso_bmff_ftyp(header: bytes, actual_size: int) -> bool:
    if len(header) < 8 or header[4:8] != b"ftyp":
        return False
    size32 = int.from_bytes(header[:4], byteorder="big")
    if size32 == 0:
        return False
    if size32 == 1:
        if len(header) < 24:
            return False
        large_size = int.from_bytes(header[8:16], byteorder="big")
        return (
            large_size >= 24
            and (large_size - 24) % 4 == 0
            and large_size <= actual_size
        )
    return size32 >= 16 and (size32 - 16) % 4 == 0 and size32 <= actual_size
