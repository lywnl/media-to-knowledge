from __future__ import annotations

import traceback
import wave
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.diarization import (
    NativePyannoteBackend,
    PyannoteDiarizer,
    RawSpeakerTurn,
    _load_pcm16_waveform,
    normalize_speaker_turns,
)


def test_speaker_ids_follow_first_appearance_and_keep_overlap() -> None:
    turns = normalize_speaker_turns(
        (
            RawSpeakerTurn(start_ms=2_000, end_ms=4_000, raw_speaker="speaker-b"),
            RawSpeakerTurn(start_ms=0, end_ms=3_000, raw_speaker="speaker-a"),
            RawSpeakerTurn(start_ms=5_000, end_ms=6_000, raw_speaker="speaker-b"),
        ),
    )

    assert [(turn.start_ms, turn.end_ms, turn.speaker) for turn in turns] == [
        (0, 3_000, "SPEAKER_01"),
        (2_000, 4_000, "SPEAKER_02"),
        (5_000, 6_000, "SPEAKER_02"),
    ]
    assert turns[0].overlap_speakers == ("SPEAKER_02",)
    assert turns[1].overlap_speakers == ("SPEAKER_01",)
    assert turns[2].overlap_speakers == ()


def test_pyannote_adapter_processes_full_audio_once_and_passes_speaker_bounds(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int | None, int | None]] = []

    class Backend:
        def diarize(
            self,
            audio: Path,
            *,
            min_speakers: int | None,
            max_speakers: int | None,
        ) -> tuple[RawSpeakerTurn, ...]:
            calls.append((audio, min_speakers, max_speakers))
            return (RawSpeakerTurn(start_ms=0, end_ms=1_000, raw_speaker="S0"),)

    audio = tmp_path / "audio.wav"
    diarizer = PyannoteDiarizer(Backend())

    turns = diarizer.diarize(audio, min_speakers=1, max_speakers=3)

    assert calls == [(audio, 1, 3)]
    assert turns[0].speaker == "SPEAKER_01"


def test_pyannote_pcm16_loader_builds_normalized_channel_first_waveform(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x80\x00\x00\xff\x7f")

    payload = _load_pcm16_waveform(audio)
    waveform = payload["waveform"]

    assert payload["sample_rate"] == 16_000
    assert payload["uri"] == "audio"
    assert tuple(waveform.shape) == (1, 3)  # type: ignore[union-attr]
    assert waveform.tolist()[0] == pytest.approx(  # type: ignore[union-attr]
        [-1.0, 0.0, 32_767 / 32_768]
    )


def test_pyannote_pcm16_loader_rejects_non_pcm16_wave(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(1)
        stream.setframerate(16_000)
        stream.writeframes(b"\x80")

    with pytest.raises(ValueError, match="PCM16 WAV"):
        _load_pcm16_waveform(audio)


def test_pyannote_loader_uses_locked_community_model_and_token() -> None:
    from video_demo.speech.diarization import load_pyannote_pipeline

    captured: dict[str, object] = {}

    def factory(
        model_id: str,
        *,
        token: str,
        cache_dir: str,
    ) -> object:
        captured.update(
            {
                "model_id": model_id,
                "token": token,
                "cache_dir": cache_dir,
            }
        )
        return object()

    pipeline = load_pyannote_pipeline(factory, "hf_read_only", Path("runtime/models"))

    assert pipeline is not None
    assert captured == {
        "model_id": "pyannote/speaker-diarization-community-1",
        "token": "hf_read_only",
        "cache_dir": "runtime/models/pyannote",
    }


def test_native_pyannote_unwraps_token_only_when_first_called(tmp_path: Path) -> None:
    calls: list[object] = []

    class Segment:
        start = 0.25
        end = 1.5

    class Output:
        speaker_diarization = object()

    class Pipeline:
        def __call__(self, payload: object, **kwargs: object) -> Output:
            calls.append(("run", payload, kwargs))
            return Output()

    def importer(name: str) -> object:
        calls.append(("import", name))

        class Factory:
            @staticmethod
            def from_pretrained(
                model_id: str,
                *,
                token: str,
                cache_dir: str,
            ) -> Pipeline:
                calls.append(("load", model_id, token, cache_dir))
                return Pipeline()

        class Module:
            Pipeline = Factory

        return Module()

    backend = NativePyannoteBackend(
        "hf_test_token",
        model_root=tmp_path / "models",
        importer=importer,
        audio_loader=lambda path: {
            "waveform": "preloaded-waveform",
            "sample_rate": 16_000,
            "uri": path.stem,
        },
    )
    backend._iter_tracks = lambda _annotation: [  # type: ignore[method-assign]
        (Segment(), None, "raw-speaker")
    ]
    assert calls == []

    turns = backend.diarize(tmp_path / "audio.wav", min_speakers=1, max_speakers=3)

    assert turns == (RawSpeakerTurn(250, 1_500, "raw-speaker"),)
    assert calls[0] == ("import", "pyannote.audio")
    assert calls[1] == (
        "load",
        "pyannote/speaker-diarization-community-1",
        "hf_test_token",
        str(tmp_path / "models/pyannote"),
    )
    run = calls[2]
    assert run[1] == {
        "waveform": "preloaded-waveform",
        "sample_rate": 16_000,
        "uri": "audio",
    }
    assert run[2] == {"min_speakers": 1, "max_speakers": 3}


def test_native_pyannote_missing_token_is_stable_without_import(tmp_path: Path) -> None:
    calls: list[str] = []
    backend = NativePyannoteBackend(
        None,
        model_root=tmp_path / "models",
        importer=lambda name: calls.append(name),  # type: ignore[arg-type]
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.diarize(tmp_path / "audio.wav", min_speakers=None, max_speakers=None)

    assert raised.value.code == ErrorCode.PYANNOTE_AUTHENTICATION_FAILED
    assert calls == []


def test_native_pyannote_never_exposes_token_from_loader_failure(tmp_path: Path) -> None:
    secret = "hf_leak_guard"

    def importer(_name: str) -> object:
        class Factory:
            @staticmethod
            def from_pretrained(
                _model_id: str,
                *,
                token: str,
                cache_dir: str,
            ) -> object:
                raise PermissionError(f"denied {token}")

        class Module:
            Pipeline = Factory

        return Module()

    backend = NativePyannoteBackend(secret, model_root=tmp_path / "models", importer=importer)

    with pytest.raises(VideoDemoError) as raised:
        backend.diarize(tmp_path / "audio.wav", min_speakers=None, max_speakers=None)

    assert raised.value.code == ErrorCode.PYANNOTE_AUTHENTICATION_FAILED
    assert secret not in f"{raised.value.message} {raised.value.details}"
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_native_pyannote_model_loader_failure_is_distinct_from_auth(tmp_path: Path) -> None:
    def importer(_name: str) -> object:
        class Factory:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs: object) -> object:
                raise RuntimeError("model unavailable")

        class Module:
            Pipeline = Factory

        return Module()

    backend = NativePyannoteBackend(
        "hf_test",
        model_root=tmp_path / "models",
        importer=importer,
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.diarize(tmp_path / "audio.wav", min_speakers=None, max_speakers=None)

    assert raised.value.code == ErrorCode.PYANNOTE_MODEL_UNAVAILABLE


def test_native_pyannote_translates_gated_http_status_without_leaking_secret(
    tmp_path: Path,
) -> None:
    secret = "hf_status_secret"

    class Response:
        status_code = 403

    class HubFailure(Exception):
        response = Response()

    def importer(_name: str) -> object:
        class Factory:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs: object) -> object:
                raise HubFailure(secret)

        class Module:
            Pipeline = Factory

        return Module()

    backend = NativePyannoteBackend(
        secret,
        model_root=tmp_path / "models",
        importer=importer,
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.diarize(tmp_path / "audio.wav", min_speakers=None, max_speakers=None)

    assert raised.value.code == ErrorCode.PYANNOTE_AUTHENTICATION_FAILED
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_native_pyannote_inference_failure_drops_sensitive_traceback(tmp_path: Path) -> None:
    secret = "pyannote-inference-secret"

    class Pipeline:
        def __call__(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError(secret)

    def importer(_name: str) -> object:
        class Factory:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs: object) -> Pipeline:
                return Pipeline()

        class Module:
            Pipeline = Factory

        return Module()

    backend = NativePyannoteBackend(
        "hf_test",
        model_root=tmp_path / "models",
        importer=importer,
    )

    with pytest.raises(VideoDemoError) as raised:
        backend.diarize(tmp_path / "audio.wav", min_speakers=None, max_speakers=None)

    assert raised.value.code == ErrorCode.PYANNOTE_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
