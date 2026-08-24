from __future__ import annotations

import tomllib
from pathlib import Path

import video_demo.speech.asr as asr_module
import video_demo.speech.language as language_module
import video_demo.speech.runtime as runtime_module


def test_retired_local_speech_dependencies_and_extra_are_absent() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        optional = tomllib.load(stream)["project"]["optional-dependencies"]

    assert optional["speech"] == [
        "silero-vad>=5.1,<7",
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
    ]
    assert "audio-events" not in optional
    assert optional["evaluation"] == [
        "jiwer>=3.1,<5",
        "psutil>=6.1,<8",
        "rapidfuzz>=3.11,<4",
    ]


def test_production_speech_modules_do_not_expose_retired_local_whisper_symbols() -> None:
    retired_symbols = {
        "FasterWhisperAdapter",
        "NativeFasterWhisperBackend",
        "FasterWhisperLanguageDetector",
        "FasterWhisperModelId",
        "load_faster_whisper_model",
        "faster_whisper_model_directory",
        "is_complete_faster_whisper_model",
    }

    for module in (asr_module, language_module, runtime_module):
        assert retired_symbols.isdisjoint(vars(module))
