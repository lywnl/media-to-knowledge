"""音频链路固定时间 ASR 分块。

音频使用与视频语音阶段相同的固定时间窗口算法，但保留独立的音频
契约和稳定标识，避免音频业务反向依赖视频流水线。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.asr_contracts import RawAsrSegment
from video_demo.speech.language import LanguageSpan

AUDIO_ASR_CHUNK_DURATION_MS = 600_000
# 与最新视频 ASR 路径保持一致：单个媒体 Run 内固定串行上传窗口；
# 跨 Run 的并发仍由音频阶段调度器控制。
AUDIO_ASR_CONCURRENCY = 1


@dataclass(frozen=True, slots=True)
class AudioFixedAsrWindow:
    chunk_index: int
    upload_range: TimeRange
    owned_range: TimeRange


@dataclass(frozen=True, slots=True)
class AudioFixedAsrProjection:
    """单个固定音频块的绝对时间转写结果。"""

    language_span: LanguageSpan
    segments: tuple[SpeechSegment, ...]
    warnings: tuple[str, ...] = ()


def build_fixed_audio_asr_windows(
    duration_ms: int,
    *,
    chunk_duration_ms: int = AUDIO_ASR_CHUNK_DURATION_MS,
) -> tuple[AudioFixedAsrWindow, ...]:
    if type(duration_ms) is not int or duration_ms < 1:
        raise ValueError("音频时长必须是正整数")
    if type(chunk_duration_ms) is not int or chunk_duration_ms < 1:
        raise ValueError("音频 ASR 固定分块时长必须是正整数")
    windows: list[AudioFixedAsrWindow] = []
    start_ms = 0
    chunk_index = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + chunk_duration_ms)
        time_range = TimeRange(start_ms=start_ms, end_ms=end_ms)
        windows.append(
            AudioFixedAsrWindow(
                chunk_index=chunk_index,
                upload_range=time_range,
                owned_range=time_range,
            ),
        )
        start_ms = end_ms
        chunk_index += 1
    return tuple(windows)


def project_fixed_audio_asr_window(
    window: AudioFixedAsrWindow,
    *,
    language: str,
    raw_segments: Sequence[RawAsrSegment],
    warnings: Sequence[str] = (),
) -> AudioFixedAsrProjection:
    """把窗口内的相对时间戳转换成当前音频的绝对时间戳。"""

    if window.upload_range != window.owned_range:
        raise ValueError("音频 ASR 固定窗口不得包含重叠范围")
    projected_warnings = list(dict.fromkeys(warnings))
    language_span = LanguageSpan(
        evidence_id=stable_identifier(
            "audio-asr-language",
            {
                "chunk_index": window.chunk_index,
                "start_ms": window.owned_range.start_ms,
                "end_ms": window.owned_range.end_ms,
                "language": language,
            },
        ),
        start_ms=window.owned_range.start_ms,
        end_ms=window.owned_range.end_ms,
        language=language,
        confidence=None,
        is_fully_evaluated_language=language in {"zh", "en", "ja", "ko", "es"},
    )
    segments: list[SpeechSegment] = []
    for item in raw_segments:
        if not _is_integer_timestamp(item.start_ms) or not _is_integer_timestamp(item.end_ms):
            raise ValueError("ASR 片段时间非法")
        text = item.text.strip()
        if not text:
            continue
        bounded_start_ms = max(0, item.start_ms)
        bounded_end_ms = min(window.owned_range.duration_ms, item.end_ms)
        if bounded_end_ms <= bounded_start_ms:
            _append_warning(projected_warnings, "ASR_TIMESTAMP_DROPPED")
            continue
        if bounded_start_ms != item.start_ms or bounded_end_ms != item.end_ms:
            _append_warning(projected_warnings, "ASR_TIMESTAMP_CLAMPED")
        start_ms = window.owned_range.start_ms + bounded_start_ms
        end_ms = window.owned_range.start_ms + bounded_end_ms
        segments.append(
            SpeechSegment(
                evidence_id=stable_identifier(
                    "audio-asr",
                    {
                        "language_evidence_id": language_span.evidence_id,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                    },
                ),
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                language=language,
                confidence=item.confidence,
                is_fully_evaluated_language=language_span.is_fully_evaluated_language,
            ),
        )
    return AudioFixedAsrProjection(
        language_span=language_span,
        segments=tuple(segments),
        warnings=tuple(projected_warnings),
    )


def _is_integer_timestamp(value: object) -> bool:
    return type(value) is int


def _append_warning(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)


def remove_adjacent_audio_asr_duplicates(
    segments: Sequence[SpeechSegment],
) -> tuple[SpeechSegment, ...]:
    """按固定块边界去除同文案重复片段，保留置信度更高的一条。"""

    deduplicated: list[SpeechSegment] = []
    for current in segments:
        if not deduplicated:
            deduplicated.append(current)
            continue
        previous = deduplicated[-1]
        is_boundary_duplicate = (
            previous.end_ms == current.start_ms
            and _normalize_audio_asr_text(previous.text)
            == _normalize_audio_asr_text(current.text)
        )
        if not is_boundary_duplicate:
            deduplicated.append(current)
        elif current.confidence > previous.confidence:
            deduplicated[-1] = current
    return tuple(deduplicated)


def _normalize_audio_asr_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
