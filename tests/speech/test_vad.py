from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.vad import NativeSileroBackend


class _FakeWaveform:
    def __init__(self, channels: int, samples: int) -> None:
        self.shape = (channels, samples)
        self._samples = samples
        self.dtype = "source"

    def numel(self) -> int:
        return self.shape[0] * self._samples

    def __getitem__(self, index: int) -> _FakeWaveform:
        if index != 0:
            raise IndexError(index)
        result = _FakeWaveform(1, self._samples)
        result.shape = (self._samples,)
        return result

    def to(self, *, dtype: object) -> _FakeWaveform:
        self.dtype = dtype
        return self


def _backend(
    *,
    channels: int = 1,
    sample_rate: int = 16_000,
    samples: int = 320,
) -> NativeSileroBackend:
    waveform = _FakeWaveform(channels, samples)
    modules = {
        "silero_vad": SimpleNamespace(
            load_silero_vad=lambda: object(),
            get_speech_timestamps=lambda *_args, **_kwargs: [],
        ),
        "torchaudio": SimpleNamespace(
            load=lambda _path: (waveform, sample_rate),
        ),
        "torch": SimpleNamespace(float32="float32"),
    }
    return NativeSileroBackend(importer=modules.__getitem__)


def test_native_backend_loads_mp3_as_float32_mono_waveform(tmp_path: Path) -> None:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"mp3")

    waveform = _backend().load_audio(path, 16_000)

    assert isinstance(waveform, _FakeWaveform)
    assert waveform.shape == (320,)
    assert waveform.dtype == "float32"


@pytest.mark.parametrize(
    ("channels", "sample_rate", "samples"),
    ((2, 16_000, 320), (1, 8_000, 320), (1, 16_000, 0)),
)
def test_native_backend_rejects_invalid_decoded_mp3(
    tmp_path: Path,
    channels: int,
    sample_rate: int,
    samples: int,
) -> None:
    path = tmp_path / "audio.mp3"
    path.write_bytes(b"mp3")

    with pytest.raises(VideoDemoError) as raised:
        _backend(
            channels=channels,
            sample_rate=sample_rate,
            samples=samples,
        ).load_audio(path, 16_000)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
