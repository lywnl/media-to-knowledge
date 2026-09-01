from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from video_demo.domain.evidence import SpeechSegment
from video_demo.speech.asr_contracts import (
    RawAsrSegment,
    WindowRecognizerPort,
    WindowTranscriptionResult,
)

__all__ = [
    "RawAsrSegment",
    "WindowRecognizerPort",
    "WindowTranscriptionResult",
    "remove_adjacent_cloud_asr_duplicates",
]


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
