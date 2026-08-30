from __future__ import annotations

import hashlib
import html
import re

from pydantic import Field

from video_demo.domain.audio_document import AudioChapter, AudioUnderstandingResult
from video_demo.domain.audio_plan import (
    AudioBulletListBlock,
    AudioCodeBlock,
    AudioFormulaBlock,
    AudioParagraphBlock,
    AudioQuoteBlock,
    AudioTableBlock,
)
from video_demo.domain.base import FrozenModel, Sha256

_MARKDOWN_SPECIAL_PATTERN = re.compile(r"([\\`*_\[\]{}()#|>!])")
_LEADING_BLOCK_PATTERN = re.compile(r"^(\s*)([-+*]\s|\d+[.)]\s)")
_EVIDENCE_MARKER_PATTERN = re.compile(
    r"(?:\\?\[\s*(?:asr|subtitle)_[A-Za-z0-9_-]{3,}\s*\\?\]"
    r"|(?:asr|subtitle)_[A-Za-z0-9_-]{3,})",
    re.IGNORECASE,
)


class RenderedAudioDocument(FrozenModel):
    content: bytes = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    media_type: str = "text/markdown; charset=utf-8"


def render_audio_markdown(result: AudioUnderstandingResult) -> RenderedAudioDocument:
    lines = [
        f"# {_escape(result.summary.title)}",
        "",
        "## 核心概览",
        "",
        _escape(result.summary.overview_zh),
        "",
        "## 目录",
        "",
    ]
    for index, chapter in enumerate(result.chapters, start=1):
        lines.append(
            f"- 第{_chinese_ordinal(index)}章：{_escape(chapter.title)} "
            f"({_format_time(chapter.start_ms)} - {_format_time(chapter.end_ms)})",
        )
    lines.append("")
    for index, chapter in enumerate(result.chapters, start=1):
        lines.extend((f"## 第{_chinese_ordinal(index)}章：{_escape(chapter.title)}", ""))
        lines.extend(
            (f"时间：{_format_time(chapter.start_ms)} - {_format_time(chapter.end_ms)}", ""),
        )
        if chapter.summary_zh and not _summary_repeats_body(chapter):
            lines.extend((_escape(chapter.summary_zh), ""))
        for block in chapter.body_blocks:
            _render_block(lines, block)
        if chapter.claims:
            lines.extend(("#### 本章结论", ""))
            lines.extend(f"- {_escape(claim.text)}" for claim in chapter.claims)
            lines.append("")
    content = ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")
    return RenderedAudioDocument(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _render_block(lines: list[str], block: object) -> None:
    if isinstance(block, AudioParagraphBlock):
        lines.extend((_escape(block.text), ""))
    elif isinstance(block, AudioBulletListBlock):
        lines.extend(f"- {_escape(item)}" for item in block.items)
        lines.append("")
    elif isinstance(block, AudioQuoteBlock):
        lines.extend(f"> {_escape(line)}" for line in block.text.splitlines())
        lines.append("")
    elif isinstance(block, AudioCodeBlock):
        language = re.sub(r"[^A-Za-z0-9_+.-]", "", block.language or "")
        lines.extend((f"```{language}", block.code, "```", ""))
    elif isinstance(block, AudioTableBlock):
        columns = tuple(_escape(value) for value in block.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in block.rows:
            lines.append("| " + " | ".join(_escape(value) for value in row) + " |")
        lines.append("")
    elif isinstance(block, AudioFormulaBlock):
        lines.extend(("$$", block.latex, "$$", ""))


def _escape(value: str) -> str:
    cleaned = clean_audio_text(value)
    escaped = _MARKDOWN_SPECIAL_PATTERN.sub(r"\\\1", html.escape(cleaned, quote=True))
    return "\n".join(_LEADING_BLOCK_PATTERN.sub(r"\1\\\2", line) for line in escaped.splitlines())


def clean_audio_text(value: str) -> str:
    """清除模型误写进自然语言的内部证据标识。"""

    cleaned = _EVIDENCE_MARKER_PATTERN.sub("", value)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([，。；：、,.!?\uFF01\uFF1F])", r"\1", cleaned)
    return cleaned.strip()


def _summary_repeats_body(chapter: AudioChapter) -> bool:
    body_text = " ".join(
        block.text
        for block in chapter.body_blocks
        if isinstance(block, (AudioParagraphBlock, AudioQuoteBlock))
    )
    return bool(body_text) and _normalize_text(chapter.summary_zh) == _normalize_text(body_text)


def _normalize_text(value: str) -> str:
    return " ".join(clean_audio_text(value).split())


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
    if value < 20:
        return "十" + (numerals[value - 10] if value > 10 else "")
    tens, ones = divmod(value, 10)
    if value < 100:
        return f"{numerals[tens]}十{numerals[ones] if ones else ''}"
    hundreds, remainder = divmod(value, 100)
    if remainder == 0:
        return f"{numerals[hundreds]}百"
    if remainder < 10:
        return f"{numerals[hundreds]}百零{numerals[remainder]}"
    return f"{numerals[hundreds]}百{_chinese_ordinal(remainder)}"
