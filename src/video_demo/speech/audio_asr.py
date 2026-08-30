from __future__ import annotations

import logging
import math
import unicodedata
from collections.abc import Sequence

from video_demo.domain.base import stable_identifier
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.asr_contracts import (
    CloudAsrWindow,
    CloudAsrWindowProjection,
    RawAsrSegment,
)
from video_demo.speech.language import LanguageSpan
from video_demo.speech.vad import SpeechInterval

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "CloudAsrWindow",
    "CloudAsrWindowProjection",
    "RawAsrSegment",
    "build_cloud_asr_windows",
    "build_speech_segments",
    "project_cloud_asr_window",
    "remove_adjacent_cloud_asr_duplicates",
]


def build_cloud_asr_windows(
    speech_intervals: Sequence[SpeechInterval],
    *,
    max_window_ms: int,
    overlap_ms: int,
    merge_gap_ms: int = 2_000,
    max_upload_bytes: int = 25 * 1024 * 1024,
) -> tuple[CloudAsrWindow, ...]:
    """合并相邻 VAD 区间，建立串行上传窗口并分配唯一所有权。"""

    _validate_cloud_asr_window_parameters(
        max_window_ms,
        overlap_ms,
        merge_gap_ms,
        max_upload_bytes,
    )
    _validate_ordered_speech_intervals(speech_intervals)
    windows: list[CloudAsrWindow] = []
    max_duration_ms = min(max_window_ms, (max_upload_bytes - 44) // 32)
    if max_duration_ms < 1:
        raise ValueError("max_upload_bytes 不足以容纳有效音频")
    for merged, sources in _merge_speech_intervals(
        speech_intervals,
        merge_gap_ms,
        max_duration_ms=max_duration_ms,
        max_upload_bytes=max_upload_bytes,
    ):
        windows.extend(
            _build_windows_for_interval(
                merged,
                sources,
                max_window_ms=max_window_ms,
                overlap_ms=overlap_ms,
                max_upload_bytes=max_upload_bytes,
            )
        )
    return tuple(windows)


def _validate_cloud_asr_window_parameters(
    max_window_ms: int,
    overlap_ms: int,
    merge_gap_ms: int,
    max_upload_bytes: int,
) -> None:
    if isinstance(max_window_ms, bool) or not isinstance(max_window_ms, int):
        raise ValueError("max_window_ms 必须是整数")
    if isinstance(overlap_ms, bool) or not isinstance(overlap_ms, int):
        raise ValueError("overlap_ms 必须是整数")
    if isinstance(merge_gap_ms, bool) or not isinstance(merge_gap_ms, int):
        raise ValueError("merge_gap_ms 必须是整数")
    if isinstance(max_upload_bytes, bool) or not isinstance(max_upload_bytes, int):
        raise ValueError("max_upload_bytes 必须是整数")
    if max_window_ms < 1:
        raise ValueError("max_window_ms 必须大于 0")
    if not 0 <= overlap_ms < max_window_ms:
        raise ValueError("overlap_ms 必须大于等于 0 且小于 max_window_ms")
    if merge_gap_ms < 0:
        raise ValueError("merge_gap_ms 必须大于等于 0")
    if max_upload_bytes <= 44:
        raise ValueError("max_upload_bytes 必须大于 WAV 头大小")


def _merge_speech_intervals(
    speech_intervals: Sequence[SpeechInterval],
    merge_gap_ms: int,
    *,
    max_duration_ms: int,
    max_upload_bytes: int,
) -> tuple[tuple[SpeechInterval, tuple[SpeechInterval, ...]], ...]:
    merged: list[tuple[SpeechInterval, tuple[SpeechInterval, ...]]] = []
    current_sources: list[SpeechInterval] = []
    for interval in speech_intervals:
        if not current_sources:
            current_sources.append(interval)
            continue
        current = current_sources[-1]
        candidate_start = current_sources[0].start_ms
        candidate_end = interval.end_ms
        candidate_duration = candidate_end - candidate_start
        if (
            interval.start_ms - current.end_ms <= merge_gap_ms
            and candidate_duration <= max_duration_ms
            and _estimate_pcm16_wav_bytes(candidate_duration) <= max_upload_bytes
        ):
            current_sources.append(interval)
            continue
        merged.append(_merged_speech_interval(tuple(current_sources)))
        current_sources = [interval]
    if current_sources:
        merged.append(_merged_speech_interval(tuple(current_sources)))
    return tuple(merged)


def _merged_speech_interval(
    source_intervals: tuple[SpeechInterval, ...],
) -> tuple[SpeechInterval, tuple[SpeechInterval, ...]]:
    if len(source_intervals) == 1:
        return source_intervals[0], source_intervals
    start_ms = source_intervals[0].start_ms
    end_ms = source_intervals[-1].end_ms
    total_duration = sum(item.duration_ms for item in source_intervals)
    confidence = sum(
        item.confidence * item.duration_ms for item in source_intervals
    ) / total_duration
    speech_interval = SpeechInterval(
        evidence_id=stable_identifier(
            "vad-window",
            {
                "source_evidence_ids": tuple(item.evidence_id for item in source_intervals),
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
        ),
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
    )
    return speech_interval, source_intervals


def _build_windows_for_interval(
    speech_interval: SpeechInterval,
    source_intervals: tuple[SpeechInterval, ...],
    *,
    max_window_ms: int,
    overlap_ms: int,
    max_upload_bytes: int,
) -> tuple[CloudAsrWindow, ...]:
    max_duration_ms = min(
        max_window_ms,
        (max_upload_bytes - 44) // 32,
    )
    if max_duration_ms < 1:
        raise ValueError("max_upload_bytes 不足以容纳有效音频")
    if speech_interval.duration_ms <= max_duration_ms:
        return (
            CloudAsrWindow(
                upload_range=speech_interval,
                owned_range=speech_interval,
                speech_interval=speech_interval,
                source_intervals=source_intervals,
            ),
        )
    effective_overlap_ms = min(overlap_ms, max_duration_ms - 1)
    return _split_cloud_asr_interval(
        speech_interval,
        source_intervals=source_intervals,
        max_window_ms=max_duration_ms,
        overlap_ms=effective_overlap_ms,
        max_upload_bytes=max_upload_bytes,
    )


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
    source_intervals: tuple[SpeechInterval, ...],
    max_window_ms: int,
    overlap_ms: int,
    max_upload_bytes: int,
) -> tuple[CloudAsrWindow, ...]:
    minimum_count = math.ceil(
        speech_interval.duration_ms / (max_window_ms - overlap_ms)
    )
    window_count = max(2, minimum_count)
    while True:
        owned_ranges = _balanced_owned_ranges(speech_interval, window_count)
        upload_ranges = _overlapping_upload_ranges(owned_ranges, overlap_ms)
        if all(
            item.duration_ms <= max_window_ms
            and _estimate_pcm16_wav_bytes(item.duration_ms) <= max_upload_bytes
            for item in upload_ranges
        ):
            return tuple(
                CloudAsrWindow(
                    upload_range=upload_range,
                    owned_range=owned_range,
                    speech_interval=speech_interval,
                    source_intervals=tuple(
                        source
                        for source in source_intervals
                        if source.overlaps(upload_range)
                    ),
                )
                for upload_range, owned_range in zip(
                    upload_ranges,
                    owned_ranges,
                    strict=True,
                )
            )
        window_count += 1


def _estimate_pcm16_wav_bytes(duration_ms: int) -> int:
    return 44 + math.ceil(duration_ms * 32)


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


def build_speech_segments(
    language_span: LanguageSpan,
    raw_segments: Sequence[RawAsrSegment],
    *,
    warnings: list[str] | None = None,
) -> tuple[SpeechSegment, ...]:
    built: list[SpeechSegment] = []
    for item in raw_segments:
        if not _is_integer_timestamp(item.start_ms) or not _is_integer_timestamp(item.end_ms):
            raise ValueError("ASR 片段时间非法")
        text = item.text.strip()
        if not text:
            continue
        bounded_start_ms = max(0, item.start_ms)
        bounded_end_ms = min(language_span.duration_ms, item.end_ms)
        if bounded_end_ms <= bounded_start_ms:
            _append_asr_warning(warnings, "ASR_TIMESTAMP_DROPPED")
            _LOGGER.warning(
                "ASR 片段没有落在语言窗口内，已丢弃",
                extra={
                    "warning_code": "ASR_TIMESTAMP_DROPPED",
                    "language_evidence_id": language_span.evidence_id,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "duration_ms": language_span.duration_ms,
                },
            )
            continue
        if bounded_start_ms != item.start_ms or bounded_end_ms != item.end_ms:
            _append_asr_warning(warnings, "ASR_TIMESTAMP_CLAMPED")
            _LOGGER.warning(
                "ASR 片段时间戳已夹紧到语言窗口",
                extra={
                    "warning_code": "ASR_TIMESTAMP_CLAMPED",
                    "language_evidence_id": language_span.evidence_id,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "bounded_start_ms": bounded_start_ms,
                    "bounded_end_ms": bounded_end_ms,
                    "duration_ms": language_span.duration_ms,
                },
            )
        absolute = TimeRange(
            start_ms=language_span.start_ms + bounded_start_ms,
            end_ms=language_span.start_ms + bounded_end_ms,
        )
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


def _is_integer_timestamp(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _append_asr_warning(warnings: list[str] | None, code: str) -> None:
    if warnings is not None and code not in warnings:
        warnings.append(code)
