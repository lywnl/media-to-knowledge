from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.evidence import BoundingBox, OcrLine
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual.ocr import KeyframeForOcr, OcrProcessor, OcrProviderResponse


def test_ocr_processor_preserves_frame_time_bbox_confidence_and_request_id(
    tmp_path: Path,
) -> None:
    calls: list[tuple[bytes, str]] = []

    class Client:
        def recognize(self, image: bytes, language: str) -> OcrProviderResponse:
            calls.append((image, language))
            return OcrProviderResponse(
                request_id="request-001",
                lines=(
                    OcrLine(
                        text="课程标题",
                        bounding_box=BoundingBox(x=1, y=2, width=100, height=20),
                        confidence=0.95,
                    ),
                ),
            )

    image = tmp_path / "keyframe.jpg"
    image.write_bytes(b"\xff\xd8\xffjpeg-data\xff\xd9")
    result = OcrProcessor(Client(), allowed_root=tmp_path).process(
        (
            KeyframeForOcr(
                keyframe_id="keyframe_001",
                source_sha256=hashlib.sha256(b"\xff\xd8\xffjpeg-data\xff\xd9").hexdigest(),
                start_ms=10_000,
                end_ms=20_000,
                timestamp_ms=12_000,
                path=image,
                language="zh",
            ),
        ),
    )

    assert calls == [(b"\xff\xd8\xffjpeg-data\xff\xd9", "zh")]
    assert result[0].timestamp_ms == 12_000
    assert result[0].provider_request_id == "request-001"
    assert result[0].lines[0].bounding_box.width == 100
    assert result[0].lines[0].confidence == 0.95


def test_ocr_evidence_id_is_stable_across_provider_request_ids(tmp_path: Path) -> None:
    image = tmp_path / "keyframe.jpg"
    image.write_bytes(b"\xff\xd8\xffjpeg-data\xff\xd9")
    request_ids = iter(("request-first", "request-second"))

    class Client:
        def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
            return OcrProviderResponse(request_id=next(request_ids), lines=())

    keyframe = KeyframeForOcr(
        keyframe_id="keyframe_001",
        source_sha256=hashlib.sha256(b"\xff\xd8\xffjpeg-data\xff\xd9").hexdigest(),
        start_ms=10_000,
        end_ms=20_000,
        timestamp_ms=12_000,
        path=image,
        language="zh",
    )
    processor = OcrProcessor(Client(), allowed_root=tmp_path)

    first = processor.process((keyframe,))[0]
    second = processor.process((keyframe,))[0]

    assert first.evidence_id == second.evidence_id
    assert first.provider_request_id != second.provider_request_id


def test_ocr_processor_rejects_file_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"sensitive")

    class Client:
        def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
            raise AssertionError("越界文件不得外发")

    with pytest.raises(VideoDemoError) as raised:
        OcrProcessor(Client(), allowed_root=allowed_root).process(
            (
                KeyframeForOcr(
                    keyframe_id="keyframe_001",
                    source_sha256=hashlib.sha256(b"sensitive").hexdigest(),
                    start_ms=0,
                    end_ms=1_000,
                    timestamp_ms=500,
                    path=outside,
                    language="zh",
                ),
            ),
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


@pytest.mark.parametrize("mutation", ["symlink", "digest", "invalid_magic", "too_large"])
def test_ocr_processor_validates_image_before_external_call(
    tmp_path: Path,
    mutation: str,
) -> None:
    allowed = tmp_path / "run"
    allowed.mkdir()
    target = allowed / "frame.jpg"
    payload = b"\xff\xd8\xffimage\xff\xd9"
    target.write_bytes(payload)
    path = target
    digest = hashlib.sha256(payload).hexdigest()
    max_bytes = 1024
    if mutation == "symlink":
        path = allowed / "link.jpg"
        path.symlink_to(target)
    elif mutation == "digest":
        digest = "0" * 64
    elif mutation == "invalid_magic":
        target.write_bytes(b"not-an-image")
        digest = hashlib.sha256(b"not-an-image").hexdigest()
    else:
        max_bytes = 1

    class Client:
        def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
            raise AssertionError("非法图片不得外发")

    processor = OcrProcessor(Client(), allowed_root=allowed, max_image_bytes=max_bytes)
    with pytest.raises(VideoDemoError):
        processor.process(
            (
                KeyframeForOcr(
                    keyframe_id="keyframe_001",
                    source_sha256=digest,
                    start_ms=0,
                    end_ms=1_000,
                    timestamp_ms=500,
                    path=path,
                    language="zh",
                ),
            )
        )
