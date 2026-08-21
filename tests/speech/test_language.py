from __future__ import annotations

import traceback
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.language import (
    FasterWhisperLanguageDetector,
    LanguageDetection,
    SegmentLanguageIdentifier,
)
from video_demo.speech.vad import SpeechInterval


def test_language_identifier_maps_low_confidence_to_und_and_tracks_changes(
    tmp_path: Path,
) -> None:
    detections = iter(
        (
            LanguageDetection(language="zh", confidence=0.91),
            LanguageDetection(language="en", confidence=0.42),
            LanguageDetection(language="ja", confidence=0.88),
        ),
    )
    received_hints: list[tuple[str, ...]] = []

    def detect(_audio: Path, _span: SpeechInterval, hints: tuple[str, ...]) -> LanguageDetection:
        received_hints.append(hints)
        return next(detections)

    identifier = SegmentLanguageIdentifier(detect, threshold=0.6)
    speech = (
        SpeechInterval(evidence_id="vad_1", start_ms=0, end_ms=1_000, confidence=0.9),
        SpeechInterval(evidence_id="vad_2", start_ms=1_000, end_ms=2_000, confidence=0.9),
        SpeechInterval(evidence_id="vad_3", start_ms=2_000, end_ms=3_000, confidence=0.9),
    )

    result = identifier.identify(tmp_path / "audio.wav", speech, hints=("zh", "en", "ja"))

    assert [span.language for span in result.spans] == ["zh", "und", "ja"]
    assert [span.confidence for span in result.spans] == [0.91, 0.42, 0.88]
    assert result.change_boundaries_ms == (1_000, 2_000)
    assert received_hints == [("zh", "en", "ja")] * 3


def test_language_identifier_marks_other_whisper_language_as_not_fully_evaluated(
    tmp_path: Path,
) -> None:
    identifier = SegmentLanguageIdentifier(
        lambda _audio, _span, _hints: LanguageDetection(language="fr", confidence=0.95),
    )
    speech = (
        SpeechInterval(evidence_id="vad_1", start_ms=0, end_ms=1_000, confidence=0.9),
    )

    result = identifier.identify(tmp_path / "audio.wav", speech, hints=())

    assert result.spans[0].language == "fr"
    assert result.spans[0].is_fully_evaluated_language is False


def test_language_identifier_tracks_all_five_validation_languages(tmp_path: Path) -> None:
    languages = iter(("zh", "en", "ja", "ko", "es"))
    identifier = SegmentLanguageIdentifier(
        lambda _audio, _span, _hints: LanguageDetection(
            language=next(languages),
            confidence=0.95,
        ),
    )
    speech = tuple(
        SpeechInterval(
            evidence_id=f"vad_{index}",
            start_ms=index * 1_000,
            end_ms=(index + 1) * 1_000,
            confidence=0.9,
        )
        for index in range(5)
    )

    result = identifier.identify(tmp_path / "audio.wav", speech, hints=())

    assert tuple(span.language for span in result.spans) == ("zh", "en", "ja", "ko", "es")
    assert result.change_boundaries_ms == (1_000, 2_000, 3_000, 4_000)
    assert all(span.is_fully_evaluated_language for span in result.spans)


def test_faster_whisper_language_detector_uses_slice_and_supported_hint(tmp_path: Path) -> None:
    calls: list[object] = []

    class Info:
        language = "en"
        language_probability = 0.87

    class Backend:
        def transcribe(self, audio: Path, **kwargs: object) -> tuple[object, Info]:
            calls.append((audio, kwargs))
            return ((), Info())

    audio_slice = tmp_path / "speech-slice.wav"
    detector = FasterWhisperLanguageDetector(Backend())
    result = detector(
        audio_slice,
        SpeechInterval(evidence_id="vad_local", start_ms=0, end_ms=1_000, confidence=1.0),
        ("zh", "en"),
    )

    assert result == LanguageDetection(language="en", confidence=0.87)
    assert calls == [
        (
            audio_slice,
            {
                "task": "transcribe",
                "beam_size": 1,
                "vad_filter": False,
                "word_timestamps": False,
                "condition_on_previous_text": False,
            },
        )
    ]


def test_faster_whisper_language_detector_uses_single_supported_hint(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class Info:
        language = "ja"
        language_probability = 0.99

    class Backend:
        def transcribe(self, _audio: Path, **kwargs: object) -> tuple[object, Info]:
            calls.append(kwargs)
            return ((), Info())

    FasterWhisperLanguageDetector(Backend())(
        tmp_path / "slice.wav",
        SpeechInterval(evidence_id="vad_local", start_ms=0, end_ms=1_000, confidence=1.0),
        ("ja",),
    )

    assert calls[0]["language"] == "ja"


def test_faster_whisper_language_detector_rejects_invalid_info(tmp_path: Path) -> None:
    class Backend:
        def transcribe(self, audio: Path, **kwargs: object) -> tuple[object, object]:
            return ((), object())

    with pytest.raises(VideoDemoError) as raised:
        FasterWhisperLanguageDetector(Backend())(
            tmp_path / "slice.wav",
            SpeechInterval(evidence_id="vad_local", start_ms=0, end_ms=1_000, confidence=1.0),
            (),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE


def test_faster_whisper_language_detector_drops_sensitive_info_traceback(
    tmp_path: Path,
) -> None:
    secret = "faster-whisper-info-secret"

    class Info:
        @property
        def language(self) -> str:
            raise RuntimeError(secret)

    class Backend:
        def transcribe(self, _audio: Path, **_kwargs: object) -> tuple[object, Info]:
            return ((), Info())

    with pytest.raises(VideoDemoError) as raised:
        FasterWhisperLanguageDetector(Backend())(
            tmp_path / "slice.wav",
            SpeechInterval(evidence_id="vad_local", start_ms=0, end_ms=1_000, confidence=1.0),
            (),
        )

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))
