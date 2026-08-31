"""云端语音识别的中性数据契约。

该模块只描述 ASR 窗口、原始片段和识别器端口，不包含视频或音频业务编排。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.language import LanguageSpan

if TYPE_CHECKING:
    from video_demo.speech.vad import SpeechInterval


@dataclass(frozen=True, slots=True)
class RawAsrSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CloudAsrWindow:
    upload_range: TimeRange
    owned_range: TimeRange
    speech_interval: SpeechInterval
    source_intervals: tuple[SpeechInterval, ...] = ()


@dataclass(frozen=True, slots=True)
class CloudAsrWindowProjection:
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
    ) -> WindowTranscriptionResult: ...
