from __future__ import annotations

import tomllib
import traceback
import wave
from array import array
from pathlib import Path

import pytest

from video_demo.audio.yamnet import (
    NativeYamnetBackend,
    NativeYamnetDetector,
    RawAudioEventFrame,
    YamnetThresholds,
    _read_pcm16_mono,
    aggregate_audio_events,
    load_thresholds,
)
from video_demo.errors import ErrorCode, VideoDemoError


def test_audio_events_dependencies_keep_pkg_resources_for_tensorflow_hub() -> None:
    project_root = Path(__file__).parents[2]
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    dependencies = project["project"]["optional-dependencies"]["audio-events"]

    assert "setuptools>=80.9,<82" in dependencies
    assert "tensorflow>=2.20,<2.21" in dependencies


def test_aggregate_filters_speech_and_merges_adjacent_high_signal_events() -> None:
    thresholds = YamnetThresholds(
        schema_version="1.0.0",
        threshold_version="eval-unvalidated-v1",
        default_threshold=0.5,
        class_thresholds={"Music": 0.6},
        excluded_classes=("Speech", "Conversation"),
        normalized_classes={"Music": "音乐", "Applause": "掌声"},
        merge_gap_ms=500,
        calibration_dataset_sha256=None,
    )
    events = aggregate_audio_events(
        (
            RawAudioEventFrame(0, 960, "Speech", 0.99),
            RawAudioEventFrame(0, 960, "Music", 0.70),
            RawAudioEventFrame(480, 1_440, "Music", 0.80),
            RawAudioEventFrame(2_400, 3_360, "Applause", 0.65),
            RawAudioEventFrame(2_880, 3_840, "Applause", 0.45),
        ),
        thresholds,
    )

    assert [(event.start_ms, event.end_ms, event.normalized_event) for event in events] == [
        (0, 1_440, "音乐"),
        (2_400, 3_360, "掌声"),
    ]
    assert events[0].confidence == 0.8
    assert all(event.audioset_class != "Speech" for event in events)


def test_aggregate_does_not_merge_same_class_across_large_gap() -> None:
    thresholds = YamnetThresholds(
        schema_version="1.0.0",
        threshold_version="eval-unvalidated-v1",
        default_threshold=0.5,
        class_thresholds={},
        excluded_classes=(),
        normalized_classes={"Alarm": "警报"},
        merge_gap_ms=300,
        calibration_dataset_sha256=None,
    )

    events = aggregate_audio_events(
        (
            RawAudioEventFrame(0, 960, "Alarm", 0.8),
            RawAudioEventFrame(2_000, 2_960, "Alarm", 0.9),
        ),
        thresholds,
    )

    assert len(events) == 2


def test_threshold_file_is_versioned_and_explicitly_unvalidated() -> None:
    thresholds_path = (
        Path(__file__).parents[2] / "src" / "video_demo" / "audio" / "thresholds.json"
    )

    thresholds = load_thresholds(thresholds_path)

    assert thresholds.schema_version == "1.0.0"
    assert thresholds.threshold_version == "eval-unvalidated-v1"
    assert thresholds.calibration_dataset_sha256 is None
    assert "Speech" in thresholds.excluded_classes


def test_yamnet_frame_timing_uses_official_960ms_window_and_480ms_hop() -> None:
    from video_demo.audio.yamnet import scores_to_frames

    frames = scores_to_frames(
        scores=((0.1, 0.8), (0.7, 0.2)),
        class_names=("Music", "Applause"),
        duration_ms=1_440,
    )

    assert [(frame.start_ms, frame.end_ms) for frame in frames] == [
        (0, 960),
        (480, 1_440),
        (0, 960),
        (480, 1_440),
    ]
    assert [frame.audioset_class for frame in frames] == [
        "Music",
        "Music",
        "Applause",
        "Applause",
    ]


def test_native_yamnet_reads_pcm_wav_and_reuses_deterministic_postprocessing(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 16_000)
    class_map = tmp_path / "class-map.csv"
    class_map.write_text("index,mid,display_name\n0,/m/music,Music\n", encoding="utf-8")
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        """{
          "schema_version":"1.0.0","threshold_version":"test-v1",
          "default_threshold":0.5,"class_thresholds":{},"excluded_classes":[],
          "normalized_classes":{"Music":"音乐"},"merge_gap_ms":0,
          "calibration_dataset_sha256":null
        }""",
        encoding="utf-8",
    )
    calls: list[object] = []

    class Backend:
        def predict(self, samples: object) -> list[list[float]]:
            calls.append(samples)
            return [[0.8]]

    detector = NativeYamnetDetector(
        backend_factory=lambda: Backend(),
        class_map_path=class_map,
        thresholds_path=thresholds_path,
    )
    assert calls == []

    events = detector.detect(audio, duration_ms=1_000)

    assert len(events) == 1
    assert events[0].normalized_event == "音乐"
    assert calls
    assert isinstance(calls[0], array)
    assert calls[0].typecode == "f"
    assert calls[0].itemsize == 4


def test_native_yamnet_rejects_non_16khz_mono_pcm(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44_100)
        stream.writeframes(b"\x00\x00" * 4)
    detector = NativeYamnetDetector(
        backend_factory=lambda: (_ for _ in ()).throw(AssertionError("不应加载模型")),
        class_map_path=tmp_path / "map.csv",
        thresholds_path=tmp_path / "thresholds.json",
    )

    with pytest.raises(VideoDemoError) as raised:
        detector.detect(audio, duration_ms=1_000)

    assert raised.value.code == ErrorCode.SPEECH_AUDIO_INVALID


def test_yamnet_pcm_reader_returns_compact_normalized_float32_buffer(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x80\x00\x00\xff\x7f")

    samples = _read_pcm16_mono(audio)

    assert isinstance(samples, array)
    assert samples.typecode == "f"
    assert samples.itemsize == 4
    assert list(samples) == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_native_yamnet_drops_sensitive_third_party_traceback(tmp_path: Path) -> None:
    secret = "yamnet-import-secret"
    model = tmp_path / "saved_model"
    model.mkdir()
    backend = NativeYamnetBackend(
        model,
        importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError(secret)),
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.predict(array("f", [0.0]))

    assert raised.value.code == ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
