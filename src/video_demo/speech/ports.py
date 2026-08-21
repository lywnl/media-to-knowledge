from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_demo.domain.evidence import SpeechSegment
from video_demo.speech.language import LanguageIdentificationResult
from video_demo.speech.vad import VadResult


class VoiceActivityDetector(Protocol):
    def detect(self, audio: Path, *, duration_ms: int) -> VadResult: ...


class LanguageIdentifier(Protocol):
    def identify(
        self,
        audio: Path,
        speech: object,
        hints: tuple[str, ...],
    ) -> LanguageIdentificationResult: ...


class SpeechRecognizer(Protocol):
    def transcribe(
        self,
        audio: Path,
        languages: LanguageIdentificationResult,
    ) -> tuple[SpeechSegment, ...]: ...
