from __future__ import annotations

from pathlib import Path

from video_demo.storage.media_object_store import BinaryMediaObjectStore

_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def detect_image_mime(header: bytes) -> str | None:
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


class ImageObjectStore(BinaryMediaObjectStore):
    def __init__(self, runtime_root: Path, *, max_bytes: int) -> None:
        super().__init__(
            runtime_root,
            max_bytes=max_bytes,
            media_kind="IMAGE_OBJECT",
            allowed_mimes=_IMAGE_EXTENSIONS.values(),
            extension_to_mime=_IMAGE_EXTENSIONS,
            detector=detect_image_mime,
            error_prefix="IMAGE",
        )
