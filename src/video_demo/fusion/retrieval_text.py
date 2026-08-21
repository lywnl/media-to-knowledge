from __future__ import annotations

import re
from collections.abc import Sequence

from video_demo.domain.result import (
    SemanticFields,
    SummaryChapter,
    VideoSegment,
    VideoSummary,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_retrieval_value(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def render_segment_fields(
    *,
    start_ms: int,
    end_ms: int,
    semantics: SemanticFields,
) -> str:
    return "\n".join(
        (
            f"标题：{normalize_retrieval_value(semantics.title)}",
            f"时间范围(毫秒)：[{start_ms}, {end_ms})",
            f"中文摘要：{normalize_retrieval_value(semantics.summary_zh)}",
            *_semantic_lines_after_summary(semantics),
        ),
    )


def render_segment_retrieval_text(segment: VideoSegment) -> str:
    return render_segment_fields(
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        semantics=segment,
    )


def render_summary_fields(
    *,
    duration_ms: int,
    semantics: SemanticFields,
    chapters: Sequence[SummaryChapter],
) -> str:
    chapter_text = "；".join(
        f"{normalize_retrieval_value(chapter.title)} "
        f"[{chapter.start_ms}, {chapter.end_ms}) -> {_join(chapter.segment_ids)}"
        for chapter in chapters
    )
    return "\n".join(
        (
            f"标题：{normalize_retrieval_value(semantics.title)}",
            f"视频时长(毫秒)：{duration_ms}",
            f"中文摘要：{normalize_retrieval_value(semantics.summary_zh)}",
            f"章节：{chapter_text or '无'}",
            *_semantic_lines_after_summary(semantics),
        ),
    )


def render_summary_retrieval_text(summary: VideoSummary) -> str:
    return render_summary_fields(
        duration_ms=summary.duration_ms,
        semantics=summary,
        chapters=summary.chapters,
    )


def _semantic_lines_after_summary(semantics: SemanticFields) -> tuple[str, ...]:
    return (
        f"说话人：{_join(semantics.speakers)}",
        f"语言：{_join(semantics.languages)}",
        f"主题：{_join(semantics.topics)}",
        f"实体：{_join(semantics.entities)}",
        f"动作：{_join(semantics.actions)}",
        f"关键词：{_join(semantics.keywords)}",
        f"原语言关键词：{_join(semantics.original_keywords)}",
    )


def _join(values: Sequence[str]) -> str:
    normalized = tuple(normalize_retrieval_value(value) for value in values)
    return "、".join(value for value in normalized if value) or "无"
