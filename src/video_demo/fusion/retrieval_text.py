from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from video_demo.domain.document import (
    BulletListBlock,
    CodeBlock,
    FormulaBlock,
    ParagraphBlock,
    QuoteBlock,
    SemanticChapter,
    TableBlock,
    VideoDocumentSummary,
    VisualBlock,
)
from video_demo.domain.evidence import VisualObservationEvidence
from video_demo.domain.result import (
    SemanticFields,
    SummaryChapter,
    VideoSegment,
    VideoSummary,
    normalize_keyword_fields,
)

_WHITESPACE = re.compile(r"\s+")
_MAX_DOCUMENT_CHAPTER_RETRIEVAL_CHARS = 32_000
_MAX_DOCUMENT_SUMMARY_RETRIEVAL_CHARS = 8_000


def render_document_chapter_retrieval_text(
    chapter: SemanticChapter,
    visual_observations: Sequence[VisualObservationEvidence],
) -> str:
    """字段级投影 3.0 章节，不泄露内部引用或重复转写原文。"""

    if chapter.content_status == "NO_SEMANTIC_EVIDENCE":
        return ""
    observation_by_id = {item.evidence_id: item for item in visual_observations}
    body = _join_plain_text(_body_block_text(block) for block in chapter.body_blocks)
    visual_captions = _join_plain_text(
        item.caption
        for item in visual_observations
        if item.evidence_id in chapter.evidence_refs
        and item.relation_to_transcript != "DUPLICATE"
    )
    uncertainties = _join(
        tuple(
            uncertainty
            for item in visual_observations
            if item.evidence_id in chapter.evidence_refs
            for uncertainty in item.uncertainties
        ),
    )
    lines = [
        "文档类型：SEMANTIC_CHAPTER",
        f"章节标题：{normalize_retrieval_value(chapter.title)}",
        f"时间范围：[{chapter.start_ms}, {chapter.end_ms})",
        f"章节摘要：{normalize_retrieval_value(chapter.summary_zh)}",
        f"正文：{body or '无'}",
        f"关键结论：{_join(tuple(claim.text for claim in chapter.claims))}",
        f"视觉补充：{visual_captions or '无'}",
        f"不确定性：{uncertainties}",
    ]
    # 显式访问映射，防止将来 VISUAL block 的未知引用被静默投影。
    for block in chapter.body_blocks:
        if isinstance(block, VisualBlock) and block.visual_observation_ref not in observation_by_id:
            raise ValueError("VISUAL block 引用了未提供的视觉观察")
    return _fit_retrieval_lines(lines, _MAX_DOCUMENT_CHAPTER_RETRIEVAL_CHARS)


def render_document_summary_retrieval_text(summary: VideoDocumentSummary) -> str:
    """字段级投影 3.0 全局摘要。"""

    if not summary.overview_zh and not summary.key_points:
        return ""
    return _fit_retrieval_lines(
        (
            "文档类型：VIDEO_DOCUMENT_SUMMARY",
            f"视频标题：{normalize_retrieval_value(summary.title)}",
            f"视频时长：{summary.duration_ms}",
            f"核心概览：{normalize_retrieval_value(summary.overview_zh) or '无'}",
            f"关键结论：{_join(tuple(point.text for point in summary.key_points))}",
        ),
        _MAX_DOCUMENT_SUMMARY_RETRIEVAL_CHARS,
    )


def _body_block_text(block: object) -> str:
    if isinstance(block, (ParagraphBlock, QuoteBlock)):
        return block.text
    if isinstance(block, BulletListBlock):
        return " ".join(block.items)
    if isinstance(block, CodeBlock):
        return block.code
    if isinstance(block, TableBlock):
        return " ".join((*block.columns, *(cell for row in block.rows for cell in row)))
    if isinstance(block, FormulaBlock):
        return " ".join(value for value in (block.latex, block.explanation) if value)
    if isinstance(block, VisualBlock):
        return block.caption
    raise TypeError(f"不支持的章节正文块：{type(block).__name__}")


def _join_plain_text(values: Iterable[str]) -> str:
    normalized = (normalize_retrieval_value(value) for value in values)
    return " ".join(value for value in normalized if value)


def _fit_retrieval_lines(lines: Sequence[str], maximum: int) -> str:
    """保留全部字段标签，按相同比例确定性裁剪字段值。"""

    rendered = "\n".join(lines)
    if len(rendered) <= maximum:
        return rendered
    split_lines = tuple(line.split("：", maxsplit=1) for line in lines)
    labels_size = sum(len(parts[0]) + 1 for parts in split_lines) + len(lines) - 1
    value_budget = maximum - labels_size
    if value_budget < len(lines):
        raise ValueError("检索文本字段标签超过字符预算")
    value_lengths = tuple(len(parts[1]) if len(parts) == 2 else 0 for parts in split_lines)
    total_values = sum(value_lengths)
    fitted: list[str] = []
    remaining = value_budget
    for index, parts in enumerate(split_lines):
        value = parts[1] if len(parts) == 2 else ""
        if index == len(split_lines) - 1:
            allowance = remaining
        else:
            allowance = max(1, value_budget * len(value) // max(total_values, 1))
            allowance = min(allowance, remaining - (len(split_lines) - index - 1))
        fitted.append(f"{parts[0]}：{value[:allowance]}")
        remaining -= min(len(value), allowance)
    return "\n".join(fitted)


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
