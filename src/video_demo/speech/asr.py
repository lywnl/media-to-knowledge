from __future__ import annotations

import importlib
import math
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.language import LanguageSpan
from video_demo.speech.vad import SpeechInterval
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


@dataclass(frozen=True, slots=True)
class CloudAsrWindow:
    upload_range: TimeRange
    owned_range: TimeRange
    speech_interval: SpeechInterval


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


def build_cloud_asr_windows(
    speech_intervals: Sequence[SpeechInterval],
    *,
    max_window_ms: int,
    overlap_ms: int,
) -> tuple[CloudAsrWindow, ...]:
    """按 VAD 区间建立串行上传窗口，并为重叠区域分配唯一所有权。"""

    _validate_cloud_asr_window_parameters(max_window_ms, overlap_ms)
    _validate_ordered_speech_intervals(speech_intervals)
    windows: list[CloudAsrWindow] = []
    for speech_interval in speech_intervals:
        if speech_interval.duration_ms <= max_window_ms:
            windows.append(
                CloudAsrWindow(
                    upload_range=speech_interval,
                    owned_range=speech_interval,
                    speech_interval=speech_interval,
                )
            )
            continue
        windows.extend(
            _split_cloud_asr_interval(
                speech_interval,
                max_window_ms=max_window_ms,
                overlap_ms=overlap_ms,
            )
        )
    return tuple(windows)


def _validate_cloud_asr_window_parameters(max_window_ms: int, overlap_ms: int) -> None:
    if isinstance(max_window_ms, bool) or not isinstance(max_window_ms, int):
        raise ValueError("max_window_ms 必须是整数")
    if isinstance(overlap_ms, bool) or not isinstance(overlap_ms, int):
        raise ValueError("overlap_ms 必须是整数")
    if max_window_ms < 1:
        raise ValueError("max_window_ms 必须大于 0")
    if not 0 <= overlap_ms < max_window_ms:
        raise ValueError("overlap_ms 必须大于等于 0 且小于 max_window_ms")


def _validate_ordered_speech_intervals(
    speech_intervals: Sequence[SpeechInterval],
) -> None:
    previous: SpeechInterval | None = None
    for current in speech_intervals:
        if previous is not None and current.start_ms < previous.end_ms:
            raise ValueError("语音区间必须有序且不能重叠")
        previous = current


def _split_cloud_asr_interval(
    speech_interval: SpeechInterval,
    *,
    max_window_ms: int,
    overlap_ms: int,
) -> tuple[CloudAsrWindow, ...]:
    minimum_count = math.ceil(
        speech_interval.duration_ms / (max_window_ms - overlap_ms)
    )
    window_count = max(2, minimum_count)
    while True:
        owned_ranges = _balanced_owned_ranges(speech_interval, window_count)
        upload_ranges = _overlapping_upload_ranges(owned_ranges, overlap_ms)
        if all(item.duration_ms <= max_window_ms for item in upload_ranges):
            return tuple(
                CloudAsrWindow(
                    upload_range=upload_range,
                    owned_range=owned_range,
                    speech_interval=speech_interval,
                )
                for upload_range, owned_range in zip(
                    upload_ranges,
                    owned_ranges,
                    strict=True,
                )
            )
        window_count += 1


def _balanced_owned_ranges(
    speech_interval: SpeechInterval,
    window_count: int,
) -> tuple[TimeRange, ...]:
    base_duration, longer_range_count = divmod(
        speech_interval.duration_ms,
        window_count,
    )
    ranges: list[TimeRange] = []
    start_ms = speech_interval.start_ms
    for index in range(window_count):
        duration_ms = base_duration + (1 if index < longer_range_count else 0)
        end_ms = start_ms + duration_ms
        ranges.append(TimeRange(start_ms=start_ms, end_ms=end_ms))
        start_ms = end_ms
    return tuple(ranges)


def _overlapping_upload_ranges(
    owned_ranges: Sequence[TimeRange],
    overlap_ms: int,
) -> tuple[TimeRange, ...]:
    left_overlap_ms = overlap_ms // 2
    right_overlap_ms = overlap_ms - left_overlap_ms
    last_index = len(owned_ranges) - 1
    return tuple(
        TimeRange(
            start_ms=(
                owned.start_ms if index == 0 else owned.start_ms - left_overlap_ms
            ),
            end_ms=(
                owned.end_ms if index == last_index else owned.end_ms + right_overlap_ms
            ),
        )
        for index, owned in enumerate(owned_ranges)
    )


def project_cloud_asr_window(
    window: CloudAsrWindow,
    *,
    language: str,
    raw_segments: Sequence[RawAsrSegment],
    warnings: Sequence[str] = (),
) -> CloudAsrWindowProjection:
    """将云端相对时间投影到窗口唯一所有权范围。"""

    _validate_cloud_asr_window(window)
    language_span = LanguageSpan(
        evidence_id=stable_identifier(
            "lid",
            {
                "speech_evidence_id": window.speech_interval.evidence_id,
                "owned_start_ms": window.owned_range.start_ms,
                "owned_end_ms": window.owned_range.end_ms,
                "language": language,
            },
        ),
        start_ms=window.owned_range.start_ms,
        end_ms=window.owned_range.end_ms,
        language=language,
        confidence=None,
        is_fully_evaluated_language=language in {"zh", "en", "ja", "ko", "es"},
    )
    projected_warnings = list(dict.fromkeys(warnings))
    owned_segments: list[RawAsrSegment] = []
    for item in raw_segments:
        projected = _project_raw_segment_to_owned_range(
            item,
            window,
            projected_warnings,
        )
        if projected is not None:
            owned_segments.append(projected)
    return CloudAsrWindowProjection(
        language_span=language_span,
        segments=build_speech_segments(language_span, owned_segments),
        warnings=tuple(projected_warnings),
    )


def _validate_cloud_asr_window(window: CloudAsrWindow) -> None:
    if not window.speech_interval.contains(window.upload_range):
        raise ValueError("云端上传窗口必须位于原语音区间内")
    if not window.upload_range.contains(window.owned_range):
        raise ValueError("云端所有权窗口必须位于上传窗口内")


def _project_raw_segment_to_owned_range(
    item: RawAsrSegment,
    window: CloudAsrWindow,
    warnings: list[str],
) -> RawAsrSegment | None:
    if not _is_cloud_asr_timestamp(item.start_ms) or not _is_cloud_asr_timestamp(
        item.end_ms
    ):
        raise ValueError("ASR 片段时间非法")
    text = item.text.strip()
    if not text:
        return None
    absolute_start_ms = window.upload_range.start_ms + item.start_ms
    absolute_end_ms = window.upload_range.start_ms + item.end_ms
    bounded_start_ms = max(window.upload_range.start_ms, absolute_start_ms)
    bounded_end_ms = min(window.upload_range.end_ms, absolute_end_ms)
    if bounded_end_ms <= bounded_start_ms:
        _append_cloud_asr_warning(warnings, "ASR_TIMESTAMP_DROPPED")
        return None
    if bounded_start_ms != absolute_start_ms or bounded_end_ms != absolute_end_ms:
        _append_cloud_asr_warning(warnings, "ASR_TIMESTAMP_CLAMPED")
    midpoint_twice = bounded_start_ms + bounded_end_ms
    if not (
        2 * window.owned_range.start_ms
        <= midpoint_twice
        < 2 * window.owned_range.end_ms
    ):
        return None
    owned_start_ms = max(window.owned_range.start_ms, bounded_start_ms)
    owned_end_ms = min(window.owned_range.end_ms, bounded_end_ms)
    if owned_start_ms != bounded_start_ms or owned_end_ms != bounded_end_ms:
        _append_cloud_asr_warning(warnings, "ASR_OVERLAP_TIMESTAMP_CLAMPED")
    if owned_end_ms <= owned_start_ms:
        _append_cloud_asr_warning(warnings, "ASR_TIMESTAMP_DROPPED")
        return None
    return RawAsrSegment(
        start_ms=owned_start_ms - window.owned_range.start_ms,
        end_ms=owned_end_ms - window.owned_range.start_ms,
        text=text,
        confidence=item.confidence,
    )


def remove_adjacent_cloud_asr_duplicates(
    segments: Sequence[SpeechSegment],
) -> tuple[SpeechSegment, ...]:
    """只删除窗口边界处规范化文本完全相同的低置信度副本。"""

    deduplicated: list[SpeechSegment] = []
    for current in segments:
        if not deduplicated:
            deduplicated.append(current)
            continue
        previous = deduplicated[-1]
        is_boundary_duplicate = (
            previous.end_ms == current.start_ms
            and _normalize_cloud_asr_text(previous.text)
            == _normalize_cloud_asr_text(current.text)
        )
        if not is_boundary_duplicate:
            deduplicated.append(current)
        elif current.confidence > previous.confidence:
            deduplicated[-1] = current
    return tuple(deduplicated)


def _normalize_cloud_asr_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _is_cloud_asr_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _append_cloud_asr_warning(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


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
        *,
        hotwords: tuple[str, ...] = (),
        core_context: str | None = None,
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
            hotwords=" ".join(hotwords) or None,
            initial_prompt=core_context or None,
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
