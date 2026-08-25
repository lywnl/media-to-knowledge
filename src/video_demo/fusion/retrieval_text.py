from __future__ import annotations

import re
from collections.abc import Sequence

from video_demo.domain.result import (
    SemanticFields,
    SummaryChapter,
    VideoSegment,
    VideoSummary,
    normalize_keyword_fields,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_retrieval_value(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _keyword_lines(
    label: str,
    keywords: Sequence[str],
    original_keywords: Sequence[str],
) -> tuple[str, ...]:
    normalized_keywords, distinct_original = normalize_keyword_fields(
        keywords,
        original_keywords,
    )
    lines = [f"{label}：{_join(normalized_keywords)}"]
    if distinct_original:
        lines.append(f"原语言关键词：{_join(distinct_original)}")
    return tuple(lines)


def render_segment_fields(
    *,
    start_ms: int,
    end_ms: int,
    semantics: SemanticFields,
    video_title: str = "",
    transcript_text: str = "",
    ocr_text: Sequence[str] = (),
    visual_facts: Sequence[str] = (),
    transcript_source: str = "NONE",
) -> str:
    """渲染可直接用于片段向量化的文本。

    原始字幕/ASR 和 OCR 独立成段，避免模型摘要覆盖可检索的专业原文。
    ``video_title`` 等参数保留默认值，兼容尚未完成证据投影的调用方。
    """
    lines = [
        "文档类型：VIDEO_SEGMENT",
        f"视频标题：{normalize_retrieval_value(video_title) or '未提供'}",
        f"片段标题：{normalize_retrieval_value(semantics.title)}",
        f"时间范围：[{start_ms}, {end_ms})",
        f"文本来源：{normalize_retrieval_value(transcript_source) or 'NONE'}",
        f"语言：{_join(semantics.languages)}",
        "",
        f"局部摘要：{normalize_retrieval_value(semantics.summary_zh)}",
        f"原始文本：{normalize_retrieval_value(transcript_text) or '无'}",
        f"画面事实：{_join(visual_facts)}",
        f"OCR文字：{_join(ocr_text)}",
        f"主题：{_join(semantics.topics)}",
        f"实体：{_join(semantics.entities)}",
        f"动作：{_join(semantics.actions)}",
        *_keyword_lines("关键词", semantics.keywords, semantics.original_keywords),
    ]
    return "\n".join(lines)


def render_segment_retrieval_text(segment: VideoSegment) -> str:
    return render_segment_fields(
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        semantics=segment,
        video_title=segment.video_title,
        transcript_text=segment.transcript_text,
        ocr_text=segment.ocr_text,
        visual_facts=segment.visual_facts,
        transcript_source=segment.transcript_source,
    )


def render_summary_fields(
    *,
    duration_ms: int,
    semantics: SemanticFields,
    chapters: Sequence[SummaryChapter],
) -> str:
    chapter_text = "；".join(
        f"{normalize_retrieval_value(chapter.title)} "
        f"[{chapter.start_ms}, {chapter.end_ms})"
        for chapter in chapters
    )
    lines = [
        "文档类型：VIDEO_SUMMARY",
        f"视频标题：{normalize_retrieval_value(semantics.title)}",
        f"视频时长：{duration_ms}",
        "",
        f"整体摘要：{normalize_retrieval_value(semantics.summary_zh)}",
        f"核心主题：{_join(semantics.topics)}",
        f"核心实体：{_join(semantics.entities)}",
        f"核心动作：{_join(semantics.actions)}",
        *_keyword_lines("核心关键词", semantics.keywords, semantics.original_keywords),
        f"章节概览：{chapter_text or '无'}",
    ]
    return "\n".join(lines)


def render_summary_retrieval_text(summary: VideoSummary) -> str:
    return render_summary_fields(
        duration_ms=summary.duration_ms,
        semantics=summary,
        chapters=summary.chapters,
    )


def _join(values: Sequence[str]) -> str:
    normalized = tuple(normalize_retrieval_value(value) for value in values)
    return "、".join(value for value in normalized if value) or "无"
