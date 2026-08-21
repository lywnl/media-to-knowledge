from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.language import LanguageSpan
from video_demo.storage.workspace import reject_symlink_components

_FASTER_WHISPER_MODEL_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
# faster-whisper 的 segment 时间戳固定在 0.02 秒网格，切片末端可能向上量化一格。
_WHISPER_TIMESTAMP_PRECISION_MS = 20


@dataclass(frozen=True, slots=True)
class RawAsrSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float


class TranscriptionSegment(Protocol):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


class WhisperBackend(Protocol):
    def transcribe(
        self,
        audio: Path,
        **kwargs: object,
    ) -> tuple[Iterable[TranscriptionSegment], object]: ...


class FasterWhisperAdapter:
    def __init__(self, backend: WhisperBackend) -> None:
        self._backend = backend

    def transcribe_slice(
        self,
        audio_slice: Path,
        language_span: LanguageSpan,
    ) -> tuple[RawAsrSegment, ...]:
        language = None if language_span.language == "und" else language_span.language
        segments, _info = self._backend.transcribe(
            audio_slice,
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        try:
            return tuple(
                RawAsrSegment(
                    start_ms=round(item.start * 1000),
                    end_ms=round(item.end * 1000),
                    text=item.text,
                    confidence=_derived_confidence(item.avg_logprob, item.no_speech_prob),
                )
                for item in segments
                if item.text.strip()
            )
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "faster-whisper 返回结构非法",
            ) from None


class NativeFasterWhisperBackend:
    """faster-whisper 的懒加载边界，基础导入不要求 speech 可选依赖。"""

    def __init__(
        self,
        model_root: Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        importer: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._model_root = model_root
        self._device = device
        self._compute_type = compute_type
        self._importer = importer
        self._model: Any | None = None

    def transcribe(
        self,
        audio: Path,
        **kwargs: object,
    ) -> tuple[Iterable[TranscriptionSegment], object]:
        model = self._load()
        try:
            result = model.transcribe(str(audio), **kwargs)
        except VideoDemoError:
            raise
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "faster-whisper 推理失败",
            ) from None
        return result  # type: ignore[no-any-return]

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            module: Any = self._importer("faster_whisper")
            utils_module: Any = self._importer("faster_whisper.utils")
        except (ModuleNotFoundError, ImportError):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "未安装 faster-whisper 可选依赖",
            ) from None
        try:
            self._model = load_faster_whisper_model(
                module.WhisperModel,
                self._model_root,
                device=self._device,
                compute_type=self._compute_type,
                downloader=utils_module.download_model,
            )
        except Exception:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "faster-whisper 模型不可用",
            ) from None
        return self._model


def build_speech_segments(
    language_span: LanguageSpan,
    raw_segments: Sequence[RawAsrSegment],
) -> tuple[SpeechSegment, ...]:
    built: list[SpeechSegment] = []
    for item in raw_segments:
        if item.start_ms < 0 or item.end_ms <= item.start_ms:
            raise ValueError("ASR 片段时间非法")
        aligned_duration_ms = (
            language_span.duration_ms + _WHISPER_TIMESTAMP_PRECISION_MS - 1
        ) // _WHISPER_TIMESTAMP_PRECISION_MS * _WHISPER_TIMESTAMP_PRECISION_MS
        maximum_end_ms = aligned_duration_ms + _WHISPER_TIMESTAMP_PRECISION_MS
        if item.end_ms > maximum_end_ms:
            raise ValueError("ASR 片段超出语言窗口")
        bounded_end_ms = min(item.end_ms, language_span.duration_ms)
        if bounded_end_ms <= item.start_ms:
            raise ValueError("ASR 片段超出语言窗口")
        absolute = TimeRange(
            start_ms=language_span.start_ms + item.start_ms,
            end_ms=language_span.start_ms + bounded_end_ms,
        )
        text = item.text.strip()
        if not text:
            continue
        built.append(
            SpeechSegment(
                evidence_id=stable_identifier(
                    "asr",
                    {
                        "language_evidence_id": language_span.evidence_id,
                        "start_ms": absolute.start_ms,
                        "end_ms": absolute.end_ms,
                        "text": text,
                    },
                ),
                start_ms=absolute.start_ms,
                end_ms=absolute.end_ms,
                text=text,
                language=language_span.language,
                confidence=item.confidence,
                is_fully_evaluated_language=language_span.is_fully_evaluated_language,
            ),
        )
    return tuple(built)


def load_faster_whisper_model(
    factory: Callable[..., object],
    model_root: Path,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    downloader: Callable[..., str],
) -> object:
    model_dir = model_root / "faster-whisper"
    cache_dir = model_root.parent / "cache" / "huggingface"
    model_root = reject_symlink_components(
        model_root.parent,
        model_root,
        message="faster-whisper 模型目录不能包含符号链接",
    )
    model_dir = reject_symlink_components(
        model_root.parent,
        model_dir,
        message="faster-whisper 模型目录不能包含符号链接",
    )
    cache_dir = reject_symlink_components(
        model_root.parent,
        cache_dir,
        message="faster-whisper 下载缓存不能包含符号链接",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not is_complete_faster_whisper_model(model_dir):
        downloader(
            "large-v3",
            output_dir=str(model_dir),
            cache_dir=str(cache_dir),
        )
    return factory(
        str(model_dir),
        device=device,
        compute_type=compute_type,
        local_files_only=True,
    )


def is_complete_faster_whisper_model(model_dir: Path) -> bool:
    return all(
        (model_dir / filename).is_file()
        and not (model_dir / filename).is_symlink()
        and (model_dir / filename).stat().st_size > 0
        for filename in _FASTER_WHISPER_MODEL_FILES
    )


def _derived_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    probability = math.exp(min(0.0, avg_logprob)) * (1 - no_speech_prob)
    return min(1.0, max(0.0, probability))
