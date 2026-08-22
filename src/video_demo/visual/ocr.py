from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    provider_attempt_count: StrictInt = Field(default=1, ge=1)


class OcrDeadlineExceeded(RuntimeError):
    """OCR 成本截止时间已到；仅用于内部正常降级，不代表供应商故障。"""

    def __init__(self, message: str, *, provider_attempt_count: int = 0) -> None:
        super().__init__(message)
        self.provider_attempt_count = provider_attempt_count


class OcrClient(Protocol):
    def recognize(
        self,
        image: bytes,
        language: str,
        *,
        deadline: float | None = None,
    ) -> OcrProviderResponse: ...


class KeyframeForOcr(FrozenModel):
    keyframe_id: str
    source_sha256: Sha256
    start_ms: int
    end_ms: int
    timestamp_ms: int
    path: Path
    language: LanguageCode


@dataclass(frozen=True, slots=True)
class OcrProcessResult:
    evidence: tuple[OcrEvidence, ...]
    provider_attempt_count: int
    image_sizes: tuple[tuple[int | None, int | None], ...]


class OcrProcessor:
    def __init__(
        self,
        client: OcrClient,
        *,
        allowed_root: Path,
        max_image_bytes: int = 20 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes 必须大于等于 1")
        self._client = client
        self._allowed_root = allowed_root.resolve(strict=True)
        self._max_image_bytes = max_image_bytes
        self._clock = clock

    def process(self, keyframes: Sequence[KeyframeForOcr]) -> tuple[OcrEvidence, ...]:
        return self.process_with_diagnostics(keyframes).evidence

    def process_with_diagnostics(
        self,
        keyframes: Sequence[KeyframeForOcr],
        *,
        deadline: float | None = None,
    ) -> OcrProcessResult:
        evidence: list[OcrEvidence] = []
        image_sizes: list[tuple[int | None, int | None]] = []
        provider_attempt_count = 0
        for keyframe in keyframes:
            self._check_deadline(deadline)
            image = self._read_image(keyframe)
            image_sizes.append(_image_size(image))
            self._check_deadline(deadline)
            response = (
                self._client.recognize(image, keyframe.language)
                if deadline is None
                else self._client.recognize(
                    image,
                    keyframe.language,
                    deadline=deadline,
                )
            )
            if deadline is not None and self._clock() >= deadline:
                raise OcrDeadlineExceeded(
                    "OCR 全局截止时间已到",
                    provider_attempt_count=response.provider_attempt_count,
                )
            provider_attempt_count += response.provider_attempt_count
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
        return OcrProcessResult(
            evidence=tuple(evidence),
            provider_attempt_count=provider_attempt_count,
            image_sizes=tuple(image_sizes),
        )

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and self._clock() >= deadline:
            raise OcrDeadlineExceeded("OCR 全局截止时间已到")

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


def _image_size(image: bytes) -> tuple[int | None, int | None]:
    if image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) >= 24:
        width = int.from_bytes(image[16:20], "big")
        height = int.from_bytes(image[20:24], "big")
        if width > 0 and height > 0:
            return width, height
    if image.startswith(b"\xff\xd8\xff"):
        return _jpeg_size(image)
    return None, None


def _jpeg_size(image: bytes) -> tuple[int | None, int | None]:
    offset = 2
    while offset + 9 <= len(image):
        if image[offset] != 0xFF:
            offset += 1
            continue
        marker = image[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(image):
            break
        segment_length = int.from_bytes(image[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(image):
            break
        if marker in {0xC0, 0xC1, 0xC2} and segment_length >= 7:
            height = int.from_bytes(image[offset + 3 : offset + 5], "big")
            width = int.from_bytes(image[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
        offset += segment_length
    return None, None
