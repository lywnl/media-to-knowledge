from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.evidence import BoundingBox, OcrLine
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual.ocr import (
    KeyframeForOcr,
    OcrDeadlineExceeded,
    OcrProcessor,
    OcrProviderResponse,
)


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


def test_ocr_processor_checks_deadline_before_reading_or_sending_image(tmp_path: Path) -> None:
    image = tmp_path / "keyframe.jpg"
    image.write_bytes(b"\xff\xd8\xffjpeg-data\xff\xd9")

    class Client:
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            raise AssertionError(f"截止后不得外发，deadline={deadline}")

    processor = OcrProcessor(Client(), allowed_root=tmp_path, clock=lambda: 10.0)

    with pytest.raises(OcrDeadlineExceeded):
        processor.process_with_diagnostics(
            (
                KeyframeForOcr(
                    keyframe_id="keyframe_001",
                    source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    start_ms=0,
                    end_ms=1_000,
                    timestamp_ms=500,
                    path=image,
                    language="zh",
                ),
            ),
            deadline=10.0,
        )


def test_ocr_processor_reports_provider_attempts_without_changing_evidence(tmp_path: Path) -> None:
    image = tmp_path / "keyframe.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (640).to_bytes(4, "big")
        + (480).to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00",
    )

    class Client:
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            assert deadline == 50.0
            return OcrProviderResponse(
                request_id="request-001",
                lines=(),
                provider_attempt_count=3,
            )

    result = OcrProcessor(
        Client(),
        allowed_root=tmp_path,
        clock=lambda: 1.0,
    ).process_with_diagnostics(
        (
            KeyframeForOcr(
                keyframe_id="keyframe_001",
                source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                start_ms=0,
                end_ms=1_000,
                timestamp_ms=500,
                path=image,
                language="zh",
            ),
        ),
        deadline=50.0,
    )

    assert len(result.evidence) == 1
    assert result.provider_attempt_count == 3
    assert result.image_sizes == ((640, 480),)


def test_ocr_processor_reads_jpeg_dimensions_for_subtitle_scoring(tmp_path: Path) -> None:
    image = tmp_path / "keyframe.jpg"
    image.write_bytes(
        b"\xff\xd8\xff\xc0\x00\x11\x08"
        + (720).to_bytes(2, "big")
        + (1_280).to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9",
    )

    class Client:
        def recognize(self, _image: bytes, _language: str) -> OcrProviderResponse:
            return OcrProviderResponse(request_id="request-001", lines=())

    result = OcrProcessor(Client(), allowed_root=tmp_path).process_with_diagnostics(
        (
            KeyframeForOcr(
                keyframe_id="keyframe_001",
                source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                start_ms=0,
                end_ms=1_000,
                timestamp_ms=500,
                path=image,
                language="zh",
            ),
        ),
    )

    assert result.image_sizes == ((1_280, 720),)


def test_ocr_processor_discards_response_returned_after_deadline(tmp_path: Path) -> None:
    now = [1.0]
    image = tmp_path / "keyframe.jpg"
    image.write_bytes(b"\xff\xd8\xffjpeg-data\xff\xd9")

    class Client:
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            now[0] = 2.0
            return OcrProviderResponse(
                request_id="late-response",
                lines=(),
                provider_attempt_count=2,
            )

    processor = OcrProcessor(Client(), allowed_root=tmp_path, clock=lambda: now[0])

    with pytest.raises(OcrDeadlineExceeded) as raised:
        processor.process_with_diagnostics(
            (
                KeyframeForOcr(
                    keyframe_id="keyframe_001",
                    source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    start_ms=0,
                    end_ms=1_000,
                    timestamp_ms=500,
                    path=image,
                    language="zh",
                ),
            ),
            deadline=2.0,
        )

    assert raised.value.provider_attempt_count == 2


def test_ocr_processor_does_not_send_image_when_read_crosses_deadline(tmp_path: Path) -> None:
    clock_values = iter((1.0, 2.0))
    image = tmp_path / "keyframe.jpg"
    image.write_bytes(b"\xff\xd8\xffjpeg-data\xff\xd9")

    class Client:
        def recognize(
            self,
            _image: bytes,
            _language: str,
            *,
            deadline: float | None = None,
        ) -> OcrProviderResponse:
            raise AssertionError(f"读图越过截止时间后不得外发，deadline={deadline}")

    processor = OcrProcessor(
        Client(),
        allowed_root=tmp_path,
        clock=lambda: next(clock_values, 2.0),
    )

    with pytest.raises(OcrDeadlineExceeded):
        processor.process_with_diagnostics(
            (
                KeyframeForOcr(
                    keyframe_id="keyframe_001",
                    source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    start_ms=0,
                    end_ms=1_000,
                    timestamp_ms=500,
                    path=image,
                    language="zh",
                ),
            ),
            deadline=2.0,
        )
