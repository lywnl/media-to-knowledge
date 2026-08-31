from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from video_demo.domain.base import FrozenModel, LanguageCode, Probability, stable_identifier
from video_demo.domain.run import TimeRange

if TYPE_CHECKING:
    from video_demo.speech.vad import SpeechInterval

_FULLY_EVALUATED_LANGUAGES = frozenset({"zh", "en", "ja", "ko", "es"})


@dataclass(frozen=True, slots=True)
class LanguageDetection:
    language: str
    confidence: float


class LanguageSpan(TimeRange):
    evidence_id: str
    language: LanguageCode
    confidence: Probability | None = None
    detection_source: Literal["MODEL", "HINT"] = "MODEL"
    is_fully_evaluated_language: bool


class LanguageIdentificationResult(FrozenModel):
    spans: tuple[LanguageSpan, ...]
    change_boundaries_ms: tuple[int, ...]


LanguageDetector = Callable[[Path, "SpeechInterval", tuple[str, ...]], LanguageDetection]


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
