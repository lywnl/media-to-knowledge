from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import Field, StrictInt

from video_demo.domain.base import FrozenModel, LanguageCode, Sha256, stable_identifier
from video_demo.domain.evidence import OcrEvidence, OcrLine
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components


class OcrProviderResponse(FrozenModel):
    request_id: str
    http_status: StrictInt = Field(default=200, ge=200, le=299)
    lines: tuple[OcrLine, ...]


class OcrClient(Protocol):
    def recognize(self, image: bytes, language: str) -> OcrProviderResponse: ...


class KeyframeForOcr(FrozenModel):
    keyframe_id: str
    source_sha256: Sha256
    start_ms: int
    end_ms: int
    timestamp_ms: int
    path: Path
    language: LanguageCode


class OcrProcessor:
    def __init__(
        self,
        client: OcrClient,
        *,
        allowed_root: Path,
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes 必须大于等于 1")
        self._client = client
        self._allowed_root = allowed_root.resolve(strict=True)
        self._max_image_bytes = max_image_bytes

    def process(self, keyframes: Sequence[KeyframeForOcr]) -> tuple[OcrEvidence, ...]:
        evidence: list[OcrEvidence] = []
        for keyframe in keyframes:
            image = self._read_image(keyframe)
            response = self._client.recognize(
                image,
                keyframe.language,
            )
            evidence.append(
                OcrEvidence(
                    evidence_id=stable_identifier(
                        "ocr",
                        {
                            "keyframe_id": keyframe.keyframe_id,
                            "source_sha256": keyframe.source_sha256,
                            "timestamp_ms": keyframe.timestamp_ms,
                        },
                    ),
                    start_ms=keyframe.start_ms,
                    end_ms=keyframe.end_ms,
                    keyframe_id=keyframe.keyframe_id,
                    timestamp_ms=keyframe.timestamp_ms,
                    language=keyframe.language,
                    lines=response.lines,
                    provider_request_id=response.request_id,
                ),
            )
        return tuple(evidence)

    def _read_image(self, keyframe: KeyframeForOcr) -> bytes:
        path = reject_symlink_components(
            self._allowed_root,
            keyframe.path,
            message="OCR 图片必须位于当前运行目录内",
        )
        if not path.is_file():
            raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "OCR 图片不是普通文件")
        size = path.stat().st_size
        if size < 1:
            raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "OCR 图片不能为空")
        if size > self._max_image_bytes:
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "OCR 图片超过大小上限")
        image = path.read_bytes()
        if hashlib.sha256(image).hexdigest() != keyframe.source_sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "OCR 图片摘要不匹配")
        if not is_supported_ocr_image(image):
            raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "OCR 图片格式非法")
        return image


def is_supported_ocr_image(image: bytes) -> bool:
    """判断 OCR 输入是否具有受支持的 JPEG/PNG 文件签名。"""

    jpeg = image.startswith(b"\xff\xd8\xff") and image.endswith(b"\xff\xd9")
    png = image.startswith(b"\x89PNG\r\n\x1a\n")
    return jpeg or png
