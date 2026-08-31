from __future__ import annotations

import traceback
import wave
from importlib import import_module
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.vad import RawVadSpan, SileroVadAdapter, build_vad_result


def test_vad_merges_short_gaps_and_keeps_long_silence_boundaries() -> None:
    result = build_vad_result(
        duration_ms=10_000,
        raw_spans=(
            RawVadSpan(start_ms=500, end_ms=1_000, confidence=0.8),
            RawVadSpan(start_ms=1_120, end_ms=2_000, confidence=0.9),
            RawVadSpan(start_ms=4_500, end_ms=5_000, confidence=0.7),
        ),
        merge_gap_ms=200,
    )

    assert [(item.start_ms, item.end_ms) for item in result.speech] == [
        (500, 2_000),
        (4_500, 5_000),
    ]
    expected_weighted_confidence = (0.8 * 500 + 0.9 * 880) / (500 + 880)
    assert result.speech[0].confidence == pytest.approx(expected_weighted_confidence)
    assert [(item.start_ms, item.end_ms) for item in result.silence] == [
        (0, 500),
        (2_000, 4_500),
        (5_000, 10_000),
    ]
    assert result.long_silence_boundaries_ms == (2_000, 4_500, 5_000)


def test_vad_empty_speech_covers_whole_audio_as_silence() -> None:
    result = build_vad_result(duration_ms=3_000, raw_spans=(), merge_gap_ms=200)

    assert result.speech == ()
    assert [(item.start_ms, item.end_ms) for item in result.silence] == [(0, 3_000)]
    assert result.warnings == ("NO_SPEECH_DETECTED",)


def test_vad_clamps_only_one_millisecond_quantization_tail() -> None:
    result = build_vad_result(
        duration_ms=1_000,
        raw_spans=(RawVadSpan(start_ms=100, end_ms=1_001, confidence=0.9),),
        merge_gap_ms=200,
    )

    assert [(item.start_ms, item.end_ms) for item in result.speech] == [(100, 1_000)]
    assert [(item.start_ms, item.end_ms) for item in result.silence] == [(0, 100)]


def test_silero_adapter_maps_material_timeline_overrun_to_audio_invalid(
    tmp_path: Path,
) -> None:
    class Backend:
        def load_audio(self, _path: Path, _sampling_rate: int) -> object:
            return object()

        def speech_timestamps(
            self,
            _audio: object,
            *,
            sampling_rate: int,
            threshold: float,
        ) -> list[dict[str, int]]:
            del sampling_rate, threshold
            return [{"start": 0, "end": 16_032}]

        def interval_confidence(self, *_args: object, **_kwargs: object) -> float:
            return 0.9

    with pytest.raises(VideoDemoError) as raised:
        SileroVadAdapter(Backend()).detect(
            tmp_path / "audio.wav",
            duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.SPEECH_AUDIO_INVALID
    assert raised.value.__cause__ is None


def test_silero_adapter_maps_invalid_confidence_to_model_unavailable(
    tmp_path: Path,
) -> None:
    class Backend:
        def load_audio(self, _path: Path, _sampling_rate: int) -> object:
            return object()

        def speech_timestamps(
            self,
            _audio: object,
            *,
            sampling_rate: int,
            threshold: float,
        ) -> list[dict[str, int]]:
            del sampling_rate, threshold
            return [{"start": 0, "end": 8_000}]

        def interval_confidence(self, *_args: object, **_kwargs: object) -> float:
            return 1.1

    with pytest.raises(VideoDemoError) as raised:
        SileroVadAdapter(Backend()).detect(
            tmp_path / "audio.wav",
            duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None


def test_silero_adapter_uses_16khz_and_real_interval_scores(tmp_path: Path) -> None:
    from video_demo.speech.vad import SileroVadAdapter

    calls: list[tuple[str, object]] = []

    class Backend:
        def load_audio(self, path: Path, sampling_rate: int) -> object:
            calls.append(("load_audio", (path, sampling_rate)))
            return [0.0] * 32_000

        def speech_timestamps(
            self,
            audio: object,
            *,
            sampling_rate: int,
            threshold: float,
        ) -> list[dict[str, int]]:
            calls.append(("speech_timestamps", (sampling_rate, threshold)))
            return [{"start": 0, "end": 16_000}]

        def interval_confidence(
            self,
            audio: object,
            *,
            start_sample: int,
            end_sample: int,
            sampling_rate: int,
        ) -> float:
            calls.append(("interval_confidence", (start_sample, end_sample, sampling_rate)))
            return 0.83

    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"wav")
    result = SileroVadAdapter(Backend()).detect(wav, duration_ms=2_000)

    assert result.speech[0].confidence == 0.83
    assert calls == [
        ("load_audio", (wav, 16_000)),
        ("speech_timestamps", (16_000, 0.5)),
        ("interval_confidence", (0, 16_000, 16_000)),
    ]


def test_native_silero_backend_defers_import_and_model_load_until_first_use() -> None:
    from video_demo.speech.vad import NativeSileroBackend

    calls: list[object] = []

    class Module:
        @staticmethod
        def load_silero_vad() -> object:
            calls.append("load_model")
            return object()

        @staticmethod
        def get_speech_timestamps(*args: object, **kwargs: object) -> object:
            calls.append(("timestamps", args, kwargs))
            return []

    def importer(name: str) -> object:
        calls.append(("import", name))
        return Module()

    backend = NativeSileroBackend(importer=importer)
    assert calls == []

    backend.speech_timestamps([], sampling_rate=16_000, threshold=0.5)
    backend.speech_timestamps([], sampling_rate=16_000, threshold=0.5)

    assert calls[:2] == [
        ("import", "silero_vad"),
        "load_model",
    ]
    assert calls.count("load_model") == 1


def test_native_silero_backend_reads_pcm16_without_third_party_audio_backend(
    tmp_path: Path,
) -> None:
    from video_demo.speech.vad import NativeSileroBackend

    pytest.importorskip("torch")

    class Module:
        @staticmethod
        def load_silero_vad() -> object:
            return object()

        @staticmethod
        def read_audio(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("不得调用依赖 TorchAudio/SoX 的第三方文件解码器")

        @staticmethod
        def get_speech_timestamps(*_args: object, **_kwargs: object) -> object:
            return []

    def importer(name: str) -> object:
        return Module() if name == "silero_vad" else import_module(name)

    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x80\x00\x00\xff\x7f")

    waveform = NativeSileroBackend(importer=importer).load_audio(audio, 16_000)

    assert waveform.tolist() == pytest.approx([-1.0, 0.0, 32_767 / 32_768])  # type: ignore[attr-defined]


def test_native_silero_drops_sensitive_third_party_traceback() -> None:
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.vad import NativeSileroBackend

    secret = "silero-import-secret"
    backend = NativeSileroBackend(
        importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError(secret))
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.load_audio(Path("audio.wav"), 16_000)

    assert raised.value.code == ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_native_silero_audio_reader_drops_sensitive_traceback(tmp_path: Path) -> None:
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.vad import NativeSileroBackend

    secret = "silero-reader-secret"

    class Module:
        @staticmethod
        def load_silero_vad() -> object:
            return object()

        @staticmethod
        def get_speech_timestamps(*_args: object, **_kwargs: object) -> object:
            return []

    backend = NativeSileroBackend(importer=lambda _name: Module())
    invalid_audio = tmp_path / f"{secret}.wav"
    invalid_audio.write_bytes(b"invalid wav")

    with pytest.raises(VideoDemoError) as raised:
        backend.load_audio(invalid_audio, 16_000)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_native_silero_padding_drops_sensitive_traceback() -> None:
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.vad import NativeSileroBackend

    secret = "silero-padding-secret"

    class Audio:
        def __getitem__(self, _item: object) -> Audio:
            return self

        def __len__(self) -> int:
            return 1

    class Model:
        def __call__(self, *_args: object) -> object:
            raise AssertionError("pad 失败后不应调用模型")

    class Silero:
        load_silero_vad = staticmethod(lambda: Model())
        read_audio = staticmethod(lambda *_args, **_kwargs: Audio())
        get_speech_timestamps = staticmethod(lambda *_args, **_kwargs: [])

    class Functional:
        @staticmethod
        def pad(*_args: object) -> object:
            raise RuntimeError(secret)

    class Torch:
        class nn:
            functional = Functional()

    def importer(name: str) -> object:
        return Silero() if name == "silero_vad" else Torch()

    backend = NativeSileroBackend(importer=importer)

    with pytest.raises(VideoDemoError) as raised:
        backend.interval_confidence(
            Audio(),
            start_sample=0,
            end_sample=1,
            sampling_rate=16_000,
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_silero_adapter_drops_sensitive_timestamp_traceback(tmp_path: Path) -> None:
    from video_demo.errors import ErrorCode, VideoDemoError
    from video_demo.speech.vad import SileroVadAdapter

    secret = "silero-timestamp-secret"

    class Timestamp(dict[str, int]):
        def __getitem__(self, _key: str) -> int:
            raise KeyError(secret)

    class Backend:
        def load_audio(self, _path: Path, _sampling_rate: int) -> object:
            return object()

        def speech_timestamps(self, *_args: object, **_kwargs: object) -> list[dict[str, int]]:
            return [Timestamp()]

        def interval_confidence(self, *_args: object, **_kwargs: object) -> float:
            return 0.9

    with pytest.raises(VideoDemoError) as raised:
        SileroVadAdapter(Backend()).detect(tmp_path / "audio.wav", duration_ms=1_000)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
