"""媒体文档共用的安全标题规范化。"""

from __future__ import annotations

import re
import unicodedata

TITLE_MAX_LENGTH = 200
_TITLE_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_document_title(
    explicit_title: str | None,
    original_filename: str | None = None,
) -> str | None:
    """生成安全标题；没有显式标题时从原始文件名回退。"""

    candidate = explicit_title
    if candidate is None or not candidate.strip():
        if original_filename is None:
            return None
        filename = original_filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        extension_separator = filename.rfind(".")
        candidate = filename[:extension_separator] if extension_separator > 0 else filename
    cleaned = "".join(
        " " if character in "/\\" or unicodedata.category(character).startswith("C") else character
        for character in candidate
    )
    normalized = _TITLE_WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return normalized[:TITLE_MAX_LENGTH] or None
