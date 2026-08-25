from __future__ import annotations

import hashlib
import stat
import traceback
from pathlib import Path
from typing import Literal

import pytest

from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.keyframes import (
    FrameCandidate,
    FrameSample,
    KeyframeSelector,
    OpenCvFrameExtractor,
)


def _frame(
    timestamp_ms: int,
    *,
    sharpness: float = 100.0,
    black_ratio: float = 0.0,
    perceptual_hash: str | None = None,
) -> FrameCandidate:
    return FrameCandidate(
        timestamp_ms=timestamp_ms,
        sharpness=sharpness,
        black_ratio=black_ratio,
        perceptual_hash=perceptual_hash or f"{timestamp_ms:016x}",
        relative_path=Path(f"visual/candidates/{timestamp_ms}.jpg"),
    )


def _artifact_session(runtime_root: Path) -> CandidateArtifactSession:
    return CandidateArtifactSession(
        runtime_root=runtime_root,
        max_unique_bytes=1024 * 1024,
        max_files=100,
        max_file_bytes=1024 * 1024,
    )


def test_short_window_selects_one_best_non_black_frame() -> None:
    selection = KeyframeSelector().select(
        TimeRange(start_ms=0, end_ms=8_000),
        (
            _frame(1_000, sharpness=50),
            _frame(4_000, sharpness=500, black_ratio=0.99),
            _frame(6_000, sharpness=120),
        ),
    )

    assert [frame.timestamp_ms for frame in selection.frames] == [6_000]


def test_all_black_candidates_produce_no_keyframe() -> None:
    selection = KeyframeSelector().select(
        TimeRange(start_ms=0, end_ms=8_000),
        (
            _frame(1_000, black_ratio=0.99),
            _frame(4_000, black_ratio=1.0),
        ),
    )

    assert selection.frames == ()


def test_long_window_selects_three_frames_with_temporal_coverage() -> None:
    selection = KeyframeSelector().select(
        TimeRange(start_ms=0, end_ms=30_000),
        (
            _frame(1_000, sharpness=100),
            _frame(9_000, sharpness=200),
            _frame(11_000, sharpness=100),
            _frame(19_000, sharpness=200),
            _frame(21_000, sharpness=100),
            _frame(29_000, sharpness=200),
        ),
    )

    assert [frame.timestamp_ms for frame in selection.frames] == [9_000, 19_000, 29_000]


def test_perceptual_duplicates_are_removed_even_across_buckets() -> None:
    selection = KeyframeSelector(max_hash_distance_for_duplicate=2).select(
        TimeRange(start_ms=0, end_ms=30_000),
        (
            _frame(5_000, sharpness=100, perceptual_hash="0000000000000000"),
            _frame(15_000, sharpness=200, perceptual_hash="0000000000000001"),
            _frame(25_000, sharpness=100, perceptual_hash="ffffffffffffffff"),
        ),
    )

    assert [frame.timestamp_ms for frame in selection.frames] == [15_000, 25_000]


def test_selection_rejects_candidates_outside_window() -> None:
    with pytest.raises(ValueError, match="候选帧超出窗口"):
        KeyframeSelector().select(
            TimeRange(start_ms=10_000, end_ms=20_000),
            (_frame(9_999),),
        )


def test_opencv_extractor_reuses_one_capture_seeks_and_atomically_encodes_jpeg(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    class Array:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.25

        def var(self) -> float:
            return 123.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.values)

    class Encoded:
        def tobytes(self) -> bytes:
            return b"\xff\xd8\xff\xe0real-jpeg\xff\xd9"

    class Capture:
        def __init__(self, path: str) -> None:
            calls.append(("capture", path))

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            calls.append(("seek", prop, value))
            return True

        def read(self) -> tuple[bool, object]:
            calls.append("read")
            return True, object()

        def get(self, prop: int) -> float:
            calls.append(("position", prop))
            seek = next(
                item for item in reversed(calls) if isinstance(item, tuple) and item[0] == "seek"
            )
            return float(seek[2])

        def release(self) -> None:
            calls.append("release")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture

        @staticmethod
        def cvtColor(_frame: object, _mode: int) -> Array:
            return Array([0, 10, 240, 250] * 16)

        @staticmethod
        def Laplacian(gray: Array, _kind: int) -> Array:
            return gray

        @staticmethod
        def resize(gray: Array, _size: object, **_kwargs: object) -> Array:
            return gray

        @staticmethod
        def imencode(_extension: str, _frame: object, _params: object) -> tuple[bool, Encoded]:
            calls.append("encode")
            return True, Encoded()

    runtime = tmp_path / "runtime"
    proxy = runtime / "runs/scope/run_001/media/proxy.mp4"
    proxy.parent.mkdir(parents=True)
    proxy.write_bytes(b"proxy")
    extractor = OpenCvFrameExtractor(runtime, module_loader=lambda: Cv2, samples_per_window=2)

    result = extractor.extract(
        proxy,
        Path("runs/scope/run_001"),
        (TimeRange(start_ms=0, end_ms=4_000), TimeRange(start_ms=4_000, end_ms=8_000)),
        is_cancel_requested=lambda: False,
    )

    assert sum(len(item.candidates) for item in result) == 4
    assert len([item for item in calls if isinstance(item, tuple) and item[0] == "capture"]) == 1
    assert [item[2] for item in calls if isinstance(item, tuple) and item[0] == "seek"] == [
        1_000.0,
        3_000.0,
        5_000.0,
        7_000.0,
    ]
    assert calls[-1] == "release"
    for group in result:
        for frame in group.candidates:
            path = runtime / frame.relative_path
            assert path.read_bytes().startswith(b"\xff\xd8\xff")
            assert frame.sharpness == 123.0
            assert frame.black_ratio == 0.25
            assert len(frame.perceptual_hash) == 16


def test_opencv_dependency_failure_has_stable_sanitized_error(tmp_path: Path) -> None:
    def missing() -> object:
        raise ImportError("secret-opencv-location")

    with pytest.raises(VideoDemoError) as raised:
        OpenCvFrameExtractor(tmp_path, module_loader=missing).extract(
            tmp_path / "proxy.mp4",
            Path("runs/scope/run_001"),
            (TimeRange(start_ms=0, end_ms=1_000),),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE
    assert raised.value.__cause__ is None


def test_opencv_loader_oserror_is_dependency_unavailable_and_redacted(tmp_path: Path) -> None:
    secret = "/sensitive/dylib/path/libopencv.dylib"

    def broken_loader() -> object:
        raise OSError(secret)

    with pytest.raises(VideoDemoError) as raised:
        OpenCvFrameExtractor(tmp_path, module_loader=broken_loader).extract(
            tmp_path / "proxy.mp4",
            Path("runs/scope/run_001"),
            (TimeRange(start_ms=0, end_ms=1_000),),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize(
    "failure",
    ["unopened", "read", "metrics", "cancelled"],
)
def test_opencv_capture_is_released_exactly_once_on_every_failure_path(
    tmp_path: Path,
    failure: str,
) -> None:
    releases: list[str] = []

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return failure != "unopened"

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            if failure == "read":
                raise RuntimeError("首帧读取失败")
            return True, object()

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            releases.append("release")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        VideoCapture = Capture

        @staticmethod
        def cvtColor(_pixels: object, _mode: int) -> object:
            if failure == "metrics":
                raise RuntimeError("指标计算失败")
            return object()

    cancel_calls = 0

    def cancelled() -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        return failure == "cancelled" and cancel_calls >= 2

    with pytest.raises(VideoDemoError):
        OpenCvFrameExtractor(
            tmp_path,
            module_loader=lambda: Cv2,
            samples_per_window=1,
        ).extract(
            tmp_path / "proxy.mp4",
            Path("runs/scope/run_001"),
            (TimeRange(start_ms=0, end_ms=1_000),),
            is_cancel_requested=cancelled,
        )

    assert releases == ["release"]


def test_opencv_failed_seek_is_skipped_without_reading(tmp_path: Path) -> None:
    calls: list[str] = []

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            calls.append("seek")
            return False

        def get(self, _prop: int) -> float:
            raise AssertionError("seek 失败后不得读取实际时间")

        def read(self) -> tuple[bool, object]:
            calls.append("read")
            raise AssertionError("seek 失败后不得读取帧")

        def release(self) -> None:
            calls.append("release")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        VideoCapture = Capture

    result = OpenCvFrameExtractor(
        tmp_path,
        module_loader=lambda: Cv2,
        samples_per_window=1,
    ).extract(
        tmp_path / "proxy.mp4",
        Path("runs/scope/run_001"),
        (TimeRange(start_ms=0, end_ms=1_000),),
        is_cancel_requested=lambda: False,
    )

    assert result[0].candidates == ()
    assert calls == ["seek", "release"]


@pytest.mark.parametrize(
    ("actual_ms", "expected_timestamps"),
    [(520.0, [520]), (541.0, [])],
)
def test_opencv_uses_actual_timestamp_only_within_frame_tolerance(
    tmp_path: Path,
    actual_ms: float,
    expected_timestamps: list[int],
) -> None:
    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.1

        def var(self) -> float:
            return 1.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return b"\xff\xd8\xffimage\xff\xd9"

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return actual_ms

        def read(self) -> tuple[bool, object]:
            return True, object()

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda _pixels, _mode: Array())
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    runtime = tmp_path / "runtime"
    result = OpenCvFrameExtractor(
        runtime,
        module_loader=lambda: Cv2,
        samples_per_window=1,
    ).extract(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (TimeRange(start_ms=0, end_ms=1_000),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=40,
    )

    assert [item.timestamp_ms for item in result[0].candidates] == expected_timestamps


def test_opencv_deduplicates_repeated_actual_timestamp_before_writing(
    tmp_path: Path,
) -> None:
    encodes: list[bytes] = []

    class Array:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return float(self.marker)

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            values = [0] * 64 if self.marker == 200 else [255] * 64
            return iter(values)

    class Encoded:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    class Capture:
        reads = 0

        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return self.reads < 2

        def read(self) -> tuple[bool, object]:
            self.reads += 1
            return True, Array(200 if self.reads == 1 else 10)

        def get(self, _prop: int) -> float:
            return 167.0

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)

        @staticmethod
        def imencode(_ext: str, pixels: Array, _params: object) -> tuple[bool, Encoded]:
            payload = b"\xff\xd8\xff" + str(pixels.marker).encode() + b"\xff\xd9"
            encodes.append(payload)
            return True, Encoded(payload)

    runtime = tmp_path / "runtime"
    result = OpenCvFrameExtractor(runtime, module_loader=lambda: Cv2).extract(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (TimeRange(start_ms=0, end_ms=1_000),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=100,
    )

    assert len(result[0].candidates) == 1
    candidate = result[0].candidates[0]
    assert candidate.timestamp_ms == 167
    assert candidate.sharpness == 200.0
    assert encodes == [b"\xff\xd8\xff200\xff\xd9"]
    payload = (runtime / candidate.relative_path).read_bytes()
    assert payload == encodes[0]
    assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(encodes[0]).hexdigest()


def test_opencv_actual_timestamp_is_unique_across_windows_in_one_extract(
    tmp_path: Path,
) -> None:
    encoded_markers: list[int] = []

    class Array:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return float(self.marker)

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([self.marker] * 64)

    class Encoded:
        def __init__(self, marker: int) -> None:
            self.marker = marker

        def tobytes(self) -> bytes:
            return b"\xff\xd8\xff" + str(self.marker).encode() + b"\xff\xd9"

    class Capture:
        marker = 0

        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, object]:
            self.marker += 1
            return True, Array(self.marker)

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)

        @staticmethod
        def imencode(_ext: str, pixels: Array, _params: object) -> tuple[bool, Encoded]:
            encoded_markers.append(pixels.marker)
            return True, Encoded(pixels.marker)

    runtime = tmp_path / "runtime"
    result = OpenCvFrameExtractor(
        runtime,
        module_loader=lambda: Cv2,
        samples_per_window=1,
    ).extract(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (TimeRange(start_ms=0, end_ms=1_000), TimeRange(start_ms=0, end_ms=1_000)),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=100,
    )

    assert [len(group.candidates) for group in result] == [1, 0]
    candidate = result[0].candidates[0]
    assert encoded_markers == [1]
    assert (runtime / candidate.relative_path).read_bytes() == b"\xff\xd8\xff1\xff\xd9"


def test_exact_sampling_decodes_equal_timestamp_once_and_rebinds_sample_ids(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    jpeg = b"\xff\xd8\xffexact\xff\xd9"

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 10.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return jpeg

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            calls.append("seek")
            return True

        def read(self) -> tuple[bool, Array]:
            calls.append("read")
            return True, Array()

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            calls.append("release")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    runtime = tmp_path / "runtime"
    artifact_session = _artifact_session(runtime)
    results = OpenCvFrameExtractor(runtime, module_loader=lambda: Cv2).extract_samples(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (
            FrameSample(sample_id="sample_a", timestamp_ms=500),
            FrameSample(sample_id="sample_b", timestamp_ms=500),
        ),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=40,
        artifact_session=artifact_session,
    )
    artifact_session.close()

    assert [item.sample_id for item in results] == ["sample_a", "sample_b"]
    assert [item.status for item in results] == ["SUCCEEDED", "SUCCEEDED"]
    assert results[0].candidate == results[1].candidate
    assert calls.count("read") == calls.count("seek") == 1
    assert results[0].candidate is not None
    assert results[0].candidate.relative_path.name == (f"{hashlib.sha256(jpeg).hexdigest()}.jpg")
    destination = runtime / results[0].candidate.relative_path
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700


def test_exact_sampling_executes_primary_tier_before_earlier_supplement(
    tmp_path: Path,
) -> None:
    seeks: list[int] = []

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 1.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return b"\xff\xd8\xffpriority\xff\xd9"

    class Capture:
        position_ms = 0

        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, value: float) -> bool:
            self.position_ms = round(value)
            seeks.append(self.position_ms)
            return True

        def read(self) -> tuple[bool, Array]:
            return True, Array()

        def get(self, _prop: int) -> float:
            return float(self.position_ms)

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    session = _artifact_session(tmp_path)
    results = OpenCvFrameExtractor(tmp_path, module_loader=lambda: Cv2).extract_samples(
        tmp_path / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (
            FrameSample(
                sample_id="base_supplement",
                timestamp_ms=100,
                admission_tier="BASE_SUPPLEMENT",
            ),
            FrameSample(
                sample_id="semantic_primary",
                timestamp_ms=1_000,
                admission_tier="SEMANTIC_PRIMARY",
            ),
        ),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=20,
        artifact_session=session,
    )
    session.close()

    assert seeks == [1_000, 100]
    assert [result.sample_id for result in results] == ["base_supplement", "semantic_primary"]


def test_exact_sampling_rejects_black_frame_before_encoding_or_budget(
    tmp_path: Path,
) -> None:
    encoded = False

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.96

        def var(self) -> float:
            return 1.0

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, Array]:
            return True, Array()

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            pass

    def encode(*_args: object) -> tuple[bool, object]:
        nonlocal encoded
        encoded = True
        raise AssertionError("黑帧不应编码")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        imencode = staticmethod(encode)

    session = _artifact_session(tmp_path)
    result = OpenCvFrameExtractor(tmp_path, module_loader=lambda: Cv2).extract_samples(
        tmp_path / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (FrameSample(sample_id="black", timestamp_ms=500),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=20,
        artifact_session=session,
    )[0]
    session.close()

    assert result.status == "QUALITY_REJECTED"
    assert result.candidate is None
    assert result.artifact_status is None
    assert encoded is False
    assert session.unique_bytes == 0


def test_exact_sampling_reports_single_image_budget_rejection_without_file(
    tmp_path: Path,
) -> None:
    jpeg = b"\xff\xd8\xfftoo-large\xff\xd9"

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 1.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return jpeg

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, Array]:
            return True, Array()

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    session = CandidateArtifactSession(
        runtime_root=tmp_path,
        max_unique_bytes=1024,
        max_files=100,
        max_file_bytes=len(jpeg) - 1,
    )
    result = OpenCvFrameExtractor(tmp_path, module_loader=lambda: Cv2).extract_samples(
        tmp_path / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (FrameSample(sample_id="too_large", timestamp_ms=500),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=20,
        artifact_session=session,
    )[0]
    session.close()

    assert result.status == "SUCCEEDED"
    assert result.artifact_status == "BUDGET_REJECTED"
    assert result.candidate is None
    candidate_root = tmp_path / "runs/scope/run_001/visual/candidates"
    assert not tuple(candidate_root.iterdir())


def test_exact_sampling_reads_forward_to_nearby_timestamp_without_second_seek(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 10.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return b"\xff\xd8\xffnearby\xff\xd9"

    class Capture:
        position_ms = 0

        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, value: float) -> bool:
            calls.append("seek")
            self.position_ms = round(value) - 40
            return True

        def read(self) -> tuple[bool, Array]:
            calls.append("read")
            self.position_ms += 40
            return True, Array()

        def get(self, _prop: int) -> float:
            return float(self.position_ms)

        def release(self) -> None:
            calls.append("release")

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    runtime = tmp_path / "runtime"
    artifact_session = _artifact_session(runtime)
    results = OpenCvFrameExtractor(runtime, module_loader=lambda: Cv2).extract_samples(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (
            FrameSample(sample_id="sample_first", timestamp_ms=100),
            FrameSample(sample_id="sample_nearby", timestamp_ms=500),
        ),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=20,
        artifact_session=artifact_session,
    )
    artifact_session.close()

    assert [item.status for item in results] == ["SUCCEEDED", "SUCCEEDED"]
    assert [item.candidate.timestamp_ms for item in results if item.candidate] == [100, 500]
    assert calls.count("seek") == 1
    assert calls.count("read") == 11


def test_exact_sampling_reuses_overshot_frame_for_next_nearby_target(tmp_path: Path) -> None:
    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 1.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return b"\xff\xd8\xfflookahead\xff\xd9"

    class Capture:
        timestamps = iter((500.0, 540.0))
        current = 0.0

        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, Array]:
            self.current = next(self.timestamps)
            return True, Array()

        def get(self, _prop: int) -> float:
            return self.current

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    artifact_session = _artifact_session(tmp_path)
    results = OpenCvFrameExtractor(tmp_path, module_loader=lambda: Cv2).extract_samples(
        tmp_path / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (
            FrameSample(sample_id="sample_early", timestamp_ms=100),
            FrameSample(sample_id="sample_next", timestamp_ms=500),
        ),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=20,
        artifact_session=artifact_session,
    )
    artifact_session.close()

    assert [result.status for result in results] == ["OUT_OF_TOLERANCE", "SUCCEEDED"]
    assert results[1].candidate is not None
    assert results[1].candidate.timestamp_ms == 500


def test_exact_sampling_reuses_matching_content_addressed_jpeg(tmp_path: Path) -> None:
    jpeg = b"\xff\xd8\xffexisting\xff\xd9"
    digest = hashlib.sha256(jpeg).hexdigest()
    runtime = tmp_path / "runtime"
    destination = runtime / f"runs/scope/run_001/visual/candidates/{digest}.jpg"
    destination.parent.mkdir(parents=True, mode=0o700)
    destination.write_bytes(jpeg)
    destination.chmod(0o600)

    class Array:
        def __le__(self, _value: int) -> Array:
            return self

        def mean(self) -> float:
            return 0.0

        def var(self) -> float:
            return 1.0

        def reshape(self, _size: int) -> Array:
            return self

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([0, 255] * 32)

    class Encoded:
        def tobytes(self) -> bytes:
            return jpeg

    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, Array]:
            return True, Array()

        def get(self, _prop: int) -> float:
            return 500.0

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        COLOR_BGR2GRAY = 20
        CV_64F = 30
        INTER_AREA = 40
        IMWRITE_JPEG_QUALITY = 50
        VideoCapture = Capture
        cvtColor = staticmethod(lambda pixels, _mode: pixels)
        Laplacian = staticmethod(lambda gray, _kind: gray)
        resize = staticmethod(lambda gray, _size, **_kwargs: gray)
        imencode = staticmethod(lambda _ext, _pixels, _params: (True, Encoded()))

    artifact_session = _artifact_session(runtime)
    result = OpenCvFrameExtractor(runtime, module_loader=lambda: Cv2).extract_samples(
        runtime / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (FrameSample(sample_id="sample_existing", timestamp_ms=500),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=40,
        artifact_session=artifact_session,
    )
    artifact_session.close()

    assert result[0].status == "SUCCEEDED"
    assert result[0].candidate is not None
    assert result[0].candidate.created_by_call is False
    assert destination.read_bytes() == jpeg


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("seek", "SEEK_FAILED"),
        ("decode", "DECODE_FAILED"),
        ("timestamp", "INVALID_TIMESTAMP"),
        ("tolerance", "OUT_OF_TOLERANCE"),
    ],
)
def test_exact_sampling_returns_explicit_recoverable_failure_status(
    tmp_path: Path,
    failure: str,
    expected_status: Literal[
        "SEEK_FAILED",
        "DECODE_FAILED",
        "INVALID_TIMESTAMP",
        "OUT_OF_TOLERANCE",
    ],
) -> None:
    class Capture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            return failure != "seek"

        def read(self) -> tuple[bool, object | None]:
            return failure != "decode", object()

        def get(self, _prop: int) -> float:
            if failure == "timestamp":
                return float("nan")
            if failure == "tolerance":
                return 900.0
            return 500.0

        def release(self) -> None:
            pass

    class Cv2:
        CAP_PROP_POS_MSEC = 10
        VideoCapture = Capture

    result = OpenCvFrameExtractor(tmp_path, module_loader=lambda: Cv2).extract_samples(
        tmp_path / "runs/scope/run_001/media/proxy.mp4",
        Path("runs/scope/run_001"),
        (FrameSample(sample_id="sample_failure", timestamp_ms=500),),
        is_cancel_requested=lambda: False,
        frame_tolerance_ms=40,
    )

    assert result[0].status == expected_status
    assert result[0].candidate is None
