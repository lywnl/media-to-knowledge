from __future__ import annotations

import hashlib
import html
import re
from typing import Literal

from pydantic import Field

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.document import (
    BulletListBlock,
    CodeBlock,
    FormulaBlock,
    ParagraphBlock,
    QuoteBlock,
    SemanticChapter,
    TableBlock,
    VideoUnderstandingResult,
    VisualBlock,
    validate_evidence_references,
)
from video_demo.domain.evidence import DocumentEvidenceItem

_MARKDOWN_SPECIAL_PATTERN = re.compile(r"([\\`*_\[\]{}()#|>!])")
_LEADING_BLOCK_PATTERN = re.compile(r"^(\s*)([-+*]\s|\d+[.)]\s)")
_BACKTICK_RUN_PATTERN = re.compile(r"`+")
_SAFE_LANGUAGE_PATTERN = re.compile(r"[^A-Za-z0-9_+.-]")


class RenderedDocument(FrozenModel):
    content: bytes = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    media_type: Literal["text/markdown; charset=utf-8"] = "text/markdown; charset=utf-8"


def render_markdown(
    result: VideoUnderstandingResult,
    evidence: tuple[DocumentEvidenceItem, ...],
) -> RenderedDocument:
    """把已闭包校验的结构化结果投影为唯一、可下载的 Markdown。"""

    validate_evidence_references(result, evidence)
    lines: list[str] = []

    _heading(lines, 1, result.summary.title)
    _heading(lines, 2, "核心概览")
    _paragraph(lines, result.summary.overview_zh or "未提供核心概览。")

    _heading(lines, 2, "目录")
    for chapter_index, chapter in enumerate(result.chapters, start=1):
        lines.append(
            f"- 第{_chinese_ordinal(chapter_index)}章："
            f"{_escape_markdown(chapter.title)}"
            f" \N{FULLWIDTH LEFT PARENTHESIS}{_format_time(chapter.start_ms)} - "
            f"{_format_time(chapter.end_ms)}\N{FULLWIDTH RIGHT PARENTHESIS}",
        )
    lines.append("")

    for chapter_index, chapter in enumerate(result.chapters, start=1):
        _heading(lines, 2, f"第{_chinese_ordinal(chapter_index)}章：{chapter.title}")
        _render_chapter(lines, chapter)

    encoded = ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")
    return RenderedDocument(
        content=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _render_chapter(
    lines: list[str],
    chapter: SemanticChapter,
) -> None:
    _paragraph(
        lines,
        f"时间：{_format_time(chapter.start_ms)} - "
        f"{_format_time(chapter.end_ms)}",
    )
    if chapter.content_status == "GROUNDED":
        if chapter.summary_zh and not _summary_repeats_body(chapter):
            _paragraph(lines, chapter.summary_zh)
        for block in chapter.body_blocks:
            _render_body_block(lines, block)
        if chapter.claims:
            _heading(lines, 4, "本章结论")
            for claim in chapter.claims:
                lines.append(f"- {_escape_markdown(claim.text)}")
            lines.append("")


def _summary_repeats_body(chapter: SemanticChapter) -> bool:
    body_text = " ".join(
        block.text
        for block in chapter.body_blocks
        if isinstance(block, (ParagraphBlock, QuoteBlock))
    )
    return bool(body_text) and " ".join(chapter.summary_zh.split()) == " ".join(body_text.split())


def _render_body_block(lines: list[str], block: object) -> None:
    if isinstance(block, ParagraphBlock):
        _paragraph(lines, block.text)
    elif isinstance(block, BulletListBlock):
        for item in block.items:
            escaped = _escape_markdown(item).replace("\n", "\n  ")
            lines.append(f"- {escaped}")
        lines.append("")
    elif isinstance(block, QuoteBlock):
        lines.extend(f"> {line}" for line in _escape_markdown(block.text).splitlines())
        lines.append("")
    elif isinstance(block, CodeBlock):
        longest_run = max(
            (len(value) for value in _BACKTICK_RUN_PATTERN.findall(block.code)),
            default=0,
        )
        fence = "`" * max(3, longest_run + 1)
        language = _SAFE_LANGUAGE_PATTERN.sub("", block.language or "")
        lines.extend((f"{fence}{language}", block.code, fence, ""))
    elif isinstance(block, TableBlock):
        _render_table(lines, block)
    elif isinstance(block, FormulaBlock):
        lines.extend(("$$", _escape_html(block.latex), "$$"))
        if block.explanation:
            lines.append(_escape_markdown(block.explanation))
        lines.append("")
    elif isinstance(block, VisualBlock):
        lines.extend((f"**视觉补充：** {_escape_markdown(block.caption)}", ""))
    else:  # pragma: no cover - 领域联合类型新增成员时的失败关闭门禁。
        raise TypeError(f"不支持的章节正文块：{type(block).__name__}")


def _render_table(lines: list[str], block: TableBlock) -> None:
    columns = tuple(_escape_markdown(value).replace("\n", " / ") for value in block.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in block.rows:
        if len(row) != len(columns):
            raise ValueError("表格每行列数必须与 columns 一致")
        cells = tuple(_escape_markdown(value).replace("\n", " / ") for value in row)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")


def _heading(lines: list[str], level: int, title: str) -> None:
    lines.extend((f"{'#' * level} {_escape_markdown(title)}", ""))


def _paragraph(lines: list[str], text: str) -> None:
    lines.extend((_escape_markdown(text), ""))


def _escape_markdown(value: str) -> str:
    escaped = _MARKDOWN_SPECIAL_PATTERN.sub(r"\\\1", _escape_html(value))
    return "\n".join(_LEADING_BLOCK_PATTERN.sub(r"\1\\\2", line) for line in escaped.split("\n"))


def _escape_html(value: str) -> str:
    return html.escape(value, quote=True)


def _format_time(milliseconds: int) -> str:
    total_seconds = milliseconds // 1_000
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _chinese_ordinal(value: int) -> str:
    numerals = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
    if not 1 <= value <= 999:
        raise ValueError("中文序号只支持 1 到 999")
    if value < 10:
        return numerals[value]
    if value < 100:
        return _chinese_under_one_hundred(value, numerals)
    hundreds, remainder = divmod(value, 100)
    prefix = f"{numerals[hundreds]}百"
    if remainder == 0:
        return prefix
    if remainder < 10:
        return f"{prefix}零{numerals[remainder]}"
    return f"{prefix}{_chinese_under_one_hundred(remainder, numerals)}"


def _chinese_under_one_hundred(value: int, numerals: tuple[str, ...]) -> str:
    tens, ones = divmod(value, 10)
    prefix = "十" if tens == 1 else f"{numerals[tens]}十"
    return f"{prefix}{numerals[ones] if ones else ''}"
