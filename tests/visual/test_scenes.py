from __future__ import annotations

from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual.scenes import PySceneDetectAdapter, RawScene, build_scene_evidence


def test_scene_detection_output_is_marked_as_candidate_evidence() -> None:
    scenes = build_scene_evidence(
        (
            RawScene(start_ms=0, end_ms=5_000, score=0.9, transition="hard_cut"),
            RawScene(start_ms=5_000, end_ms=10_000, score=0.4, transition="gradual"),
        ),
        source_sha256="a" * 64,
    )

    assert [(scene.start_ms, scene.end_ms) for scene in scenes] == [
        (0, 5_000),
        (5_000, 10_000),
    ]
    assert scenes[0].transition == "hard_cut"
    assert scenes[0].evidence_type == "SCENE"
    assert scenes[0].evidence_id.startswith("scene_")


def test_scene_evidence_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="镜头区间不得重叠"):
        build_scene_evidence(
            (
                RawScene(0, 5_000, 0.8, "hard_cut"),
                RawScene(4_000, 7_000, 0.7, "gradual"),
            ),
            source_sha256="a" * 64,
        )


def test_scene_evidence_id_binds_source_digest_and_is_stable() -> None:
    raw = (RawScene(0, 1_000, 1.0, "candidate"),)

    first = build_scene_evidence(raw, source_sha256="a" * 64)
    repeated = build_scene_evidence(raw, source_sha256="a" * 64)
    other_source = build_scene_evidence(raw, source_sha256="b" * 64)

    assert first[0].evidence_id == repeated[0].evidence_id
    assert first[0].evidence_id != other_source[0].evidence_id


def test_pyscenedetect_uses_public_api_and_converts_half_open_milliseconds(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    class Timecode:
        def __init__(self, seconds: float) -> None:
            self._seconds = seconds

        def get_seconds(self) -> float:
            return self._seconds

    class SceneManager:
        def add_detector(self, detector: object) -> None:
            calls.append(("detector", detector.__class__.__name__))

        def detect_scenes(self, *, video: object, show_progress: bool) -> None:
            calls.append(("detect", video, show_progress))

        def get_scene_list(self, *, start_in_scene: bool) -> object:
            calls.append(("list", start_in_scene))
            return ((Timecode(0), Timecode(1.25)), (Timecode(1.25), Timecode(4.0)))

    class ContentDetector:
        pass

    class Module:
        @staticmethod
        def open_video(path: str) -> str:
            calls.append(("open", path))
            return "video"

    class Detectors:
        pass

    Module.SceneManager = SceneManager
    Detectors.ContentDetector = ContentDetector

    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy")
    adapter = PySceneDetectAdapter(module_loader=lambda: (Module, Detectors))

    result = adapter.detect(
        proxy,
        duration_ms=4_000,
        source_sha256="a" * 64,
        frame_tolerance_ms=40,
    )

    assert [(item.start_ms, item.end_ms, item.transition) for item in result] == [
        (0, 1_250, "candidate"),
        (1_250, 4_000, "hard_cut"),
    ]
    assert calls[0] == ("open", str(proxy))
    assert ("detect", "video", False) in calls
    assert ("list", True) in calls


def test_pyscenedetect_no_cut_is_full_candidate_scene(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy")

    class Manager:
        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, **_kwargs: object) -> None:
            pass

        def get_scene_list(self, **_kwargs: object) -> object:
            return ()

    class Module:
        SceneManager = Manager
        open_video = staticmethod(lambda _path: object())

    class Detectors:
        ContentDetector = object

    result = PySceneDetectAdapter(module_loader=lambda: (Module, Detectors)).detect(
        proxy,
        duration_ms=2_000,
        source_sha256="a" * 64,
        frame_tolerance_ms=40,
    )

    assert [(item.start_ms, item.end_ms, item.transition) for item in result] == [
        (0, 2_000, "candidate")
    ]


@pytest.mark.parametrize("detected_end_ms", [3_960, 4_040])
def test_pyscenedetect_normalizes_one_frame_tail_difference(
    tmp_path: Path,
    detected_end_ms: int,
) -> None:
    class Timecode:
        def __init__(self, milliseconds: int) -> None:
            self._seconds = milliseconds / 1_000

        def get_seconds(self) -> float:
            return self._seconds

    class Manager:
        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, **_kwargs: object) -> None:
            pass

        def get_scene_list(self, **_kwargs: object) -> object:
            return ((Timecode(0), Timecode(detected_end_ms)),)

    class Module:
        SceneManager = Manager
        open_video = staticmethod(lambda _path: object())

    class Detectors:
        ContentDetector = object

    result = PySceneDetectAdapter(module_loader=lambda: (Module, Detectors)).detect(
        tmp_path / "proxy.mp4",
        duration_ms=4_000,
        source_sha256="a" * 64,
        frame_tolerance_ms=40,
    )

    assert [(item.start_ms, item.end_ms) for item in result] == [(0, 4_000)]


def test_pyscenedetect_normalizes_audio_led_container_tail_difference(
    tmp_path: Path,
) -> None:
    class Timecode:
        def __init__(self, milliseconds: int) -> None:
            self._seconds = milliseconds / 1_000

        def get_seconds(self) -> float:
            return self._seconds

    class Manager:
        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, **_kwargs: object) -> None:
            pass

        def get_scene_list(self, **_kwargs: object) -> object:
            return ((Timecode(0), Timecode(3_916)),)

    class Module:
        SceneManager = Manager
        open_video = staticmethod(lambda _path: object())

    class Detectors:
        ContentDetector = object

    result = PySceneDetectAdapter(module_loader=lambda: (Module, Detectors)).detect(
        tmp_path / "proxy.mp4",
        duration_ms=4_000,
        source_sha256="a" * 64,
        frame_tolerance_ms=40,
    )

    assert [(item.start_ms, item.end_ms) for item in result] == [(0, 4_000)]


@pytest.mark.parametrize("detected_end_ms", [3_899, 4_041])
def test_pyscenedetect_rejects_tail_difference_beyond_one_frame(
    tmp_path: Path,
    detected_end_ms: int,
) -> None:
    class Timecode:
        def __init__(self, milliseconds: int) -> None:
            self._seconds = milliseconds / 1_000

        def get_seconds(self) -> float:
            return self._seconds

    class Manager:
        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, **_kwargs: object) -> None:
            pass

        def get_scene_list(self, **_kwargs: object) -> object:
            return ((Timecode(0), Timecode(detected_end_ms)),)

    class Module:
        SceneManager = Manager
        open_video = staticmethod(lambda _path: object())

    class Detectors:
        ContentDetector = object

    with pytest.raises(VideoDemoError) as raised:
        PySceneDetectAdapter(module_loader=lambda: (Module, Detectors)).detect(
            tmp_path / "proxy.mp4",
            duration_ms=4_000,
            source_sha256="a" * 64,
            frame_tolerance_ms=40,
        )

    assert raised.value.code == ErrorCode.VISUAL_MEDIA_INVALID


def test_pyscenedetect_dependency_failure_is_sanitized(tmp_path: Path) -> None:
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy")

    def missing() -> object:
        raise ModuleNotFoundError("secret-module-path")

    with pytest.raises(VideoDemoError) as missing_error:
        PySceneDetectAdapter(module_loader=missing).detect(
            proxy,
            duration_ms=1_000,
            source_sha256="a" * 64,
            frame_tolerance_ms=40,
        )
    assert missing_error.value.code == ErrorCode.VISUAL_DEPENDENCY_UNAVAILABLE
    assert missing_error.value.__cause__ is None
