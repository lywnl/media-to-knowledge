from __future__ import annotations

import hashlib

from video_demo.application.document_rendering import RenderedDocument
from video_demo.domain.image_document import ImageUnderstandingResult


def render_image_markdown(result: ImageUnderstandingResult) -> RenderedDocument:
    lines = [
        f"# {result.document.title}",
        "",
        "## 核心概览",
        "",
        result.document.overview_zh or "未提供核心概览。",
        "",
        "## 图片内容",
        "",
    ]
    for block in result.document.content_blocks:
        if block.content_type == "DESCRIPTION":
            lines.extend((block.text, ""))
        else:
            lines.extend((f"### {block.content_type}", "", block.text, ""))
    lines.extend(("## 关键结论", ""))
    for claim in result.document.claims:
        lines.append(f"- {claim.text}")
    lines.append("")
    content = ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")
    return RenderedDocument(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
