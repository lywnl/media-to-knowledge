from __future__ import annotations

import importlib
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from video_demo.domain.base import FrozenModel, Probability, stable_identifier
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError


@dataclass(frozen=True, slots=True)
class RawVadSpan:
    start_ms: int
    end_ms: int
    confidence: float


class SpeechInterval(TimeRange):
    evidence_id: str
    confidence: Probability


class SilenceInterval(TimeRange):
    pass


class VadResult(FrozenModel):
    speech: tuple[SpeechInterval, ...]
    silence: tuple[SilenceInterval, ...]
    long_silence_boundaries_ms: tuple[int, ...]
    warnings: tuple[str, ...] = ()


class SileroBackend(Protocol):
    def load_audio(self, path: Path, sampling_rate: int) -> object: ...

    def speech_timestamps(
        self,
        audio: object,
        *,
        sampling_rate: int,
        threshold: float,
    ) -> list[dict[str, int]]: ...

    def interval_confidence(
        self,
        audio: object,
        *,
        start_sample: int,
        end_sample: int,
        sampling_rate: int,
    ) -> float: ...


class SileroVadAdapter:
    def __init__(
        self,
        backend: SileroBackend,
        *,
        threshold: float = 0.5,
        merge_gap_ms: int = 200,
    ) -> None:
        self._backend = backend
        self._threshold = threshold
        self._merge_gap_ms = merge_gap_ms

    def detect(self, audio: Path, *, duration_ms: int) -> VadResult:
        sampling_rate = 16_000
        samples = self._backend.load_audio(audio, sampling_rate)
        timestamps = self._backend.speech_timestamps(
            samples,
            sampling_rate=sampling_rate,
            threshold=self._threshold,
        )
        try:
            spans = tuple(
                RawVadSpan(
                    start_ms=_sample_to_ms(item["start"], sampling_rate),
                    end_ms=_sample_to_ms(item["end"], sampling_rate),
                    confidence=self._backend.interval_confidence(
                        samples,
                        start_sample=item["start"],
                        end_sample=item["end"],
                        sampling_rate=sampling_rate,
                    ),
                )
                for item in timestamps
            )
        except VideoDemoError:
            raise
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "Silero VAD 返回结构非法",
            ) from None
        return build_vad_result(
            duration_ms=duration_ms,
            raw_spans=spans,
            merge_gap_ms=self._merge_gap_ms,
        )


class NativeSileroBackend:
    """对 `silero-vad` 官方 API 的懒加载封装。"""

    def __init__(
        self,
        *,
        importer: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._importer = importer
        self._model: Any | None = None
        self._get_speech_timestamps: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            silero: Any = self._importer("silero_vad")
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "未安装 silero-vad 可选依赖",
            ) from None
        try:
            self._model = silero.load_silero_vad()
            self._get_speech_timestamps = silero.get_speech_timestamps
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "Silero VAD 模型不可用",
            ) from None

    def load_audio(self, path: Path, sampling_rate: int) -> object:
        self._load()
        try:
            with wave.open(str(path), "rb") as stream:
                channels = stream.getnchannels()
                source_rate = stream.getframerate()
                sample_width = stream.getsampwidth()
                compression = stream.getcomptype()
                frames = stream.readframes(stream.getnframes())
            if (
                channels != 1
                or source_rate != sampling_rate
                or sample_width != 2
                or compression != "NONE"
                or not frames
                or len(frames) % sample_width != 0
            ):
                raise ValueError("Silero VAD 仅接受非空单声道 PCM16 WAV")
            torch: Any = self._importer("torch")
            waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16)
            return waveform.to(dtype=torch.float32).div_(32_768)
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "Silero VAD 音频读取失败",
            ) from None

    def speech_timestamps(
        self,
        audio: object,
        *,
        sampling_rate: int,
        threshold: float,
    ) -> list[dict[str, int]]:
        self._load()
        assert self._get_speech_timestamps is not None
        try:
            result = self._get_speech_timestamps(
                audio,
                self._model,
                sampling_rate=sampling_rate,
                threshold=threshold,
                return_seconds=False,
            )
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "Silero VAD 推理失败",
            ) from None
        return cast(list[dict[str, int]], result)

    def interval_confidence(
        self,
        audio: object,
        *,
        start_sample: int,
        end_sample: int,
        sampling_rate: int,
    ) -> float:
        self._load()
        assert self._model is not None
        scores: list[float] = []
        dynamic_audio: Any = audio
        for offset in range(start_sample, end_sample, 512):
            chunk = dynamic_audio[offset : min(offset + 512, end_sample)]
            if len(chunk) == 0:
                continue
            if len(chunk) < 512:
                try:
                    torch = self._importer("torch")
                except ModuleNotFoundError:
                    raise VideoDemoError(
                        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                        "未安装 speech 可选依赖 torch",
                    ) from None
                try:
                    dynamic_torch: Any = torch
                    chunk = dynamic_torch.nn.functional.pad(chunk, (0, 512 - len(chunk)))
                except Exception:
                    raise VideoDemoError(
                        ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                        "Silero VAD 音频补齐失败",
                    ) from None
            try:
                scores.append(float(self._model(chunk, sampling_rate).item()))
            except Exception:
                raise VideoDemoError(
                    ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                    "Silero VAD 置信度计算失败",
                ) from None
        return sum(scores) / len(scores) if scores else 0.0


def build_vad_result(
    *,
    duration_ms: int,
    raw_spans: tuple[RawVadSpan, ...],
    merge_gap_ms: int,
    long_silence_ms: int = 1_000,
) -> VadResult:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    ordered = sorted(raw_spans, key=lambda item: (item.start_ms, item.end_ms))
    merged: list[RawVadSpan] = []
    for span in ordered:
        if span.start_ms < 0 or span.end_ms <= span.start_ms or span.end_ms > duration_ms:
            raise ValueError("VAD 区间越界或为空")
        if not 0 <= span.confidence <= 1:
            raise ValueError("VAD 置信度必须在 0 到 1 之间")
        if merged and span.start_ms - merged[-1].end_ms <= merge_gap_ms:
            previous = merged[-1]
            previous_duration = previous.end_ms - previous.start_ms
            current_duration = span.end_ms - span.start_ms
            confidence = (
                previous.confidence * previous_duration + span.confidence * current_duration
            ) / (previous_duration + current_duration)
            merged[-1] = RawVadSpan(
                previous.start_ms,
                max(previous.end_ms, span.end_ms),
                confidence,
            )
        else:
            merged.append(span)

    speech = tuple(
        SpeechInterval(
            evidence_id=stable_identifier(
                "vad",
                {"start_ms": item.start_ms, "end_ms": item.end_ms},
            ),
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            confidence=item.confidence,
        )
        for item in merged
    )
    silence = _silence_intervals(duration_ms, merged)
    long_boundaries: list[int] = []
    for item in silence:
        if item.duration_ms < long_silence_ms:
            continue
        if item.start_ms > 0:
            long_boundaries.append(item.start_ms)
        if item.end_ms < duration_ms:
            long_boundaries.append(item.end_ms)
    warnings = () if speech else ("NO_SPEECH_DETECTED",)
    return VadResult(
        speech=speech,
        silence=silence,
        long_silence_boundaries_ms=tuple(long_boundaries),
        warnings=warnings,
    )


def _silence_intervals(
    duration_ms: int,
    speech: list[RawVadSpan],
) -> tuple[SilenceInterval, ...]:
    silence: list[SilenceInterval] = []
    cursor = 0
    for item in speech:
        if item.start_ms > cursor:
            silence.append(SilenceInterval(start_ms=cursor, end_ms=item.start_ms))
        cursor = max(cursor, item.end_ms)
    if cursor < duration_ms:
        silence.append(SilenceInterval(start_ms=cursor, end_ms=duration_ms))
    return tuple(silence)


def _sample_to_ms(sample: int, sampling_rate: int) -> int:
    return round(sample * 1000 / sampling_rate)
