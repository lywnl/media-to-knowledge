from __future__ import annotations

import re
from collections.abc import Sequence

_WHITESPACE = re.compile(r"\s+")
_MAX_HOTWORDS = 50
_MAX_HOTWORD_CHARS = 64
_MAX_HOTWORDS_UTF8_BYTES = 2_048
_MAX_CORE_CONTEXT_CHARS = 1_000


def normalize_hotwords(values: Sequence[str]) -> tuple[str, ...]:
    if len(values) > _MAX_HOTWORDS:
        raise ValueError("热词数量不能超过 50 个")
    normalized = tuple(_normalize_text(value, "热词") for value in values)
    if any(len(value) > _MAX_HOTWORD_CHARS for value in normalized):
        raise ValueError("单个热词不能超过 64 个字符")
    if len(normalized) != len(set(normalized)):
        raise ValueError("规范化后的热词不得重复")
    if sum(len(value.encode("utf-8")) for value in normalized) > _MAX_HOTWORDS_UTF8_BYTES:
        raise ValueError("热词 UTF-8 总长度不能超过 2048 字节")
    return normalized


def normalize_core_context(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value, "核心上下文")
    if len(normalized) > _MAX_CORE_CONTEXT_CHARS:
        raise ValueError("核心上下文不能超过 1000 个字符")
    return normalized


def _normalize_text(value: str, field_name: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name}不能包含控制字符")
    normalized = _WHITESPACE.sub(" ", value).strip()
    if not normalized:
        raise ValueError(f"{field_name}不能为空")
    return normalized
