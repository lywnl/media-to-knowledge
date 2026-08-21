from __future__ import annotations

import hashlib
from collections.abc import Sequence

from video_demo.domain.result import (
    SummaryChapter,
    SummaryUnderstanding,
    VideoSegment,
    VideoSummary,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.retrieval_text import render_summary_fields


def build_video_summary(
    understanding: SummaryUnderstanding,
    *,
    duration_ms: int,
    segments: Sequence[VideoSegment],
    chapters: Sequence[SummaryChapter],
) -> VideoSummary:
    if duration_ms <= 0:
        raise ValueError("duration_ms 必须大于 0")
    if not segments:
        raise ValueError("视频摘要至少需要一个片段")
    ordered_segments = tuple(
        sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id)),
    )
    segment_by_id = {item.segment_id: item for item in ordered_segments}
    if len(segment_by_id) != len(ordered_segments):
        raise VideoDemoError(ErrorCode.DUPLICATE_SEGMENT_ID, "片段 ID 重复")
    ordered_chapters = tuple(
        sorted(chapters, key=lambda item: (item.start_ms, item.end_ms, item.title)),
    )
    for chapter in ordered_chapters:
        _validate_chapter(chapter, segment_by_id, duration_ms)
    retrieval_text = render_summary_fields(
        duration_ms=duration_ms,
        semantics=understanding,
        chapters=ordered_chapters,
    )
    return VideoSummary(
        **understanding.model_dump(),
        duration_ms=duration_ms,
        chapters=ordered_chapters,
        retrieval_text=retrieval_text,
        retrieval_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
    )


def _validate_chapter(
    chapter: SummaryChapter,
    segment_by_id: dict[str, VideoSegment],
    duration_ms: int,
) -> None:
    if chapter.end_ms > duration_ms:
        raise ValueError("章节时间不得超过视频时长")
    for segment_id in chapter.segment_ids:
        segment = segment_by_id.get(segment_id)
        if segment is None:
            raise VideoDemoError(
                ErrorCode.UNKNOWN_SEGMENT_REFERENCE,
                "摘要章节引用了不存在的片段",
                {"segment_id": segment_id},
            )
        if not chapter.contains(segment):
            raise ValueError("章节必须覆盖其引用的片段")
