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
    selected_visual_blocks = tuple(
        block for block in chapter.body_blocks if isinstance(block, VisualBlock)
    )
    selected_observations = tuple(
        observation_by_id[block.visual_observation_ref]
        for block in selected_visual_blocks
        if block.visual_observation_ref in observation_by_id
    )
    body = _join_plain_text(_body_block_text(block) for block in chapter.body_blocks)
    visual_captions = _join_plain_text(
        dict.fromkeys(
            normalize_retrieval_value(block.caption)
            for block in selected_visual_blocks
        ),
    )
    uncertainties = _join(
        tuple(
            dict.fromkeys(
                uncertainty
                for item in selected_observations
                for uncertainty in item.uncertainties
            ),
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
    for block in chapter.body_blocks:
        if (
            isinstance(block, VisualBlock)
            and block.visual_observation_ref not in observation_by_id
        ):
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
        return ""
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


def _join(values: Sequence[str]) -> str:
    normalized = tuple(normalize_retrieval_value(value) for value in values)
    return "、".join(value for value in normalized if value) or "无"
