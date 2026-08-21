from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from video_demo.domain.base import FrozenModel, LanguageCode, Probability, stable_identifier
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.vad import SpeechInterval

_FULLY_EVALUATED_LANGUAGES = frozenset({"zh", "en", "ja", "ko", "es"})


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    language: str
    confidence: float


class LanguageSpan(TimeRange):
    evidence_id: str
    language: LanguageCode
    confidence: Probability
    is_fully_evaluated_language: bool


class LanguageIdentificationResult(FrozenModel):
    spans: tuple[LanguageSpan, ...]
    change_boundaries_ms: tuple[int, ...]


LanguageDetector = Callable[[Path, SpeechInterval, tuple[str, ...]], LanguageDetection]


class LanguageWhisperBackend(Protocol):
    def transcribe(self, audio: Path, **kwargs: object) -> tuple[object, object]: ...


class FasterWhisperLanguageDetector:
    """仅在已切出的 speech interval 上执行 faster-whisper 语言探测。"""

    def __init__(self, backend: LanguageWhisperBackend) -> None:
        self._backend = backend

    def __call__(
        self,
        audio: Path,
        speech: SpeechInterval,
        hints: tuple[str, ...],
    ) -> LanguageDetection:
        if speech.start_ms != 0:
            raise VideoDemoError(
                ErrorCode.SPEECH_AUDIO_INVALID,
                "语言探测必须接收切片局部时间窗",
            )
        supported_hints = tuple(item for item in hints if item != "und")
        optional_language = {"language": supported_hints[0]} if len(supported_hints) == 1 else {}
        _segments, info = self._backend.transcribe(
            audio,
            **optional_language,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        try:
            language = getattr(info, "language", None)
            confidence = getattr(info, "language_probability", None)
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "faster-whisper 未返回有效语言信息",
            ) from None
        if not isinstance(language, str) or isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "faster-whisper 未返回有效语言信息",
            )
        return LanguageDetection(language=language, confidence=float(confidence))


class SegmentLanguageIdentifier:
    def __init__(self, detector: LanguageDetector, *, threshold: float = 0.6) -> None:
        self._detector = detector
        self._threshold = threshold

    def identify(
        self,
        audio: Path,
        speech: Sequence[SpeechInterval],
        hints: tuple[str, ...],
    ) -> LanguageIdentificationResult:
        spans: list[LanguageSpan] = []
        for speech_interval in speech:
            detection = self._detector(audio, speech_interval, hints)
            if not 0 <= detection.confidence <= 1:
                raise ValueError("语言识别置信度必须在 0 到 1 之间")
            normalized = detection.language.lower().strip()
            language = normalized if detection.confidence >= self._threshold else "und"
            fully_evaluated = language in _FULLY_EVALUATED_LANGUAGES
            spans.append(
                LanguageSpan(
                    evidence_id=stable_identifier(
                        "lid",
                        {
                            "speech_evidence_id": speech_interval.evidence_id,
                            "language": language,
                        },
                    ),
                    start_ms=speech_interval.start_ms,
                    end_ms=speech_interval.end_ms,
                    language=language,
                    confidence=detection.confidence,
                    is_fully_evaluated_language=fully_evaluated,
                ),
            )
        boundaries = tuple(
            current.start_ms
            for previous, current in pairwise(spans)
            if previous.language != current.language
        )
        return LanguageIdentificationResult(spans=tuple(spans), change_boundaries_ms=boundaries)
