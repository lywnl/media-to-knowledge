"""云端语音识别的中性数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from video_demo.domain.evidence import SpeechSegment
from video_demo.speech.language import LanguageSpan


@dataclass(frozen=True, slots=True)
class RawAsrSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CloudAsrWindowProjection:
    """单个云端 ASR 窗口的绝对时间投影结果。"""

    language_span: LanguageSpan
    segments: tuple[SpeechSegment, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WindowTranscriptionResult:
    language: str
    segments: tuple[RawAsrSegment, ...]
    warnings: tuple[str, ...] = ()


class WindowRecognizerPort(Protocol):
    def transcribe_window(
        self,
        audio_slice: Path,
        *,
        language_hint: str | None,
        prompt: str | None,
        chunk_index: int | None = None,
    ) -> WindowTranscriptionResult: ...
