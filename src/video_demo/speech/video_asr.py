"""视频链路的固定时间 ASR 分块，不依赖 VAD。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.asr_contracts import (
    CloudAsrWindowProjection,
    RawAsrSegment,
)
from video_demo.speech.language import LanguageSpan

VIDEO_ASR_CHUNK_DURATION_MS = 600_000
VIDEO_ASR_CONCURRENCY = 2


@dataclass(frozen=True, slots=True)
class FixedAsrWindow:
    chunk_index: int
    upload_range: TimeRange
    owned_range: TimeRange


def build_fixed_asr_windows(
    duration_ms: int,
    *,
    chunk_duration_ms: int = VIDEO_ASR_CHUNK_DURATION_MS,
) -> tuple[FixedAsrWindow, ...]:
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 1:
        raise ValueError("视频时长必须是正整数")
    if (
        not isinstance(chunk_duration_ms, int)
        or isinstance(chunk_duration_ms, bool)
        or chunk_duration_ms < 1
    ):
        raise ValueError("ASR 固定分块时长必须是正整数")
    windows: list[FixedAsrWindow] = []
    start_ms = 0
    chunk_index = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + chunk_duration_ms)
        time_range = TimeRange(start_ms=start_ms, end_ms=end_ms)
        windows.append(
            FixedAsrWindow(
                chunk_index=chunk_index,
                upload_range=time_range,
                owned_range=time_range,
            )
        )
        start_ms = end_ms
        chunk_index += 1
    return tuple(windows)


def project_fixed_asr_window(
    window: FixedAsrWindow,
    *,
    language: str,
    raw_segments: Sequence[RawAsrSegment],
    warnings: Sequence[str] = (),
) -> CloudAsrWindowProjection:
    """把固定窗口内的相对时间戳转换为全视频绝对时间戳。"""

    if window.upload_range != window.owned_range:
        raise ValueError("视频 ASR 固定窗口不得包含重叠范围")
    projected_warnings = list(dict.fromkeys(warnings))
    language_span = LanguageSpan(
        evidence_id=stable_identifier(
            "video-asr-language",
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
                    "asr",
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
            )
        )
    return CloudAsrWindowProjection(
        language_span=language_span,
        segments=tuple(segments),
        warnings=tuple(projected_warnings),
    )


def _is_integer_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _append_warning(warnings: list[str], code: str) -> None:
    if code not in warnings:
        warnings.append(code)
