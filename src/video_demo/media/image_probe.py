from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components


@dataclass(frozen=True, slots=True)
class ImageProbeResult:
    width: int
    height: int
    mime_type: str


def probe_image(source: Path, *, runtime_root: Path, max_bytes: int) -> ImageProbeResult:
    path = reject_symlink_components(runtime_root, source, message="图片输入路径非法")
    if not path.is_file():
        raise VideoDemoError(ErrorCode.IMAGE_OBJECT_NOT_FOUND, "图片对象不存在")
    if path.stat().st_size < 1 or path.stat().st_size > max_bytes:
        raise VideoDemoError(ErrorCode.IMAGE_FILE_TOO_LARGE, "图片文件超过大小限制")
    data = path.read_bytes()
    result = _parse_image_header(data)
    if result is None:
        raise VideoDemoError(ErrorCode.IMAGE_INPUT_INVALID, "图片格式或尺寸无法识别")
    return result


def _parse_image_header(data: bytes) -> ImageProbeResult | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return _valid_dimensions(width, height, "image/png")
    if data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8X":
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return _valid_dimensions(width, height, "image/webp")
        if data[12:16] == b"VP8 " and len(data) >= 30:
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return _valid_dimensions(width, height, "image/webp")
    if data.startswith(b"\xff\xd8\xff"):
        dimensions = _jpeg_dimensions(data)
        return _valid_dimensions(*dimensions, "image/jpeg") if dimensions else None
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        length = int.from_bytes(data[index:index + 2], "big")
        if length < 2 or index + length > len(data):
            return None
        sof_markers = (
            set(range(0xC0, 0xC4))
            | set(range(0xC5, 0xC8))
            | set(range(0xC9, 0xCC))
            | set(range(0xCD, 0xD0))
        )
        if marker in sof_markers:
            if length < 7:
                return None
            return (
                int.from_bytes(data[index + 5:index + 7], "big"),
                int.from_bytes(data[index + 3:index + 5], "big"),
            )
        index += length
    return None


def _valid_dimensions(width: int, height: int, mime_type: str) -> ImageProbeResult | None:
    if not 1 <= width <= 20_000 or not 1 <= height <= 20_000:
        return None
    return ImageProbeResult(width=width, height=height, mime_type=mime_type)
