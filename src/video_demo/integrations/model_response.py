"""模型响应的宽松边界解析工具。

模型供应商经常会在文本外包一层 Markdown、思考标签或内容块；这些包装不
改变文档契约，因此在适配器边界统一剥离。解析后的对象仍由各业务模型和
引用校验负责验证，不能把这里的宽松解析当成业务校验的替代品。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, cast

_REMOVED_DOCUMENT_FIELDS = frozenset(
    {
        "sections",
        "quality_flags",
        "uncertainties",
        "uncertainty_policy",
        "key_points",
        "retrieval_text",
        "retrieval_hash",
    },
)


def extract_model_message_content(body: object) -> str:
    """兼容常见 OpenAI/Responses/Anthropic 响应形态，提取文本内容。"""

    if not isinstance(body, dict):
        return ""

    for key in ("content", "output_text", "response", "text"):
        candidate = extract_text_from_content_blocks(body.get(key))
        if candidate:
            return candidate

    output = body.get("output")
    text = extract_text_from_content_blocks(output)
    if text:
        return text

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    first = choices[0]
    text = extract_text_from_content_blocks(first.get("text"))
    if text:
        return text
    message = first.get("message")
    if isinstance(message, dict):
        return extract_text_from_content_blocks(message.get("content"))
    return ""


def extract_text_from_content_blocks(content: object) -> str:
    """把字符串或嵌套文本内容块合并成模型消息文本。"""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "").strip().lower()
        if block_type in {"thinking", "reasoning", "redacted_thinking"}:
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue
        nested = extract_text_from_content_blocks(item.get("content"))
        if nested:
            parts.append(nested)
    return "\n".join(parts).strip()


def parse_json_content(text: str) -> object:
    """从模型文本中提取 JSON，接受围栏、思考标签和前后解释文字。"""

    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    candidates = [cleaned.strip()]
    fenced = _strip_markdown_fence(cleaned)
    if fenced not in candidates:
        candidates.append(fenced)

    for candidate in candidates:
        if not candidate:
            continue
        parsed = _try_load_json(candidate)
        if parsed is not None:
            return parsed
        # 以 JSON 起始符开头时，说明模型返回的是一个损坏的完整对象；
        # 不从其中捞出嵌套数组，避免 NaN 等非法常量被误解析成看似合法的片段。
        if candidate[:1] in {"{", "["}:
            continue
        for fragment in _json_fragments(candidate):
            parsed = _try_load_json(fragment)
            if parsed is not None:
                return parsed
    raise ValueError("模型消息中没有可解析的 JSON")


def unwrap_single_response_envelope(value: object) -> object:
    """解包一层供应商常见的 result/data/response 外壳。"""

    if isinstance(value, dict) and len(value) == 1:
        for key in ("result", "data", "response"):
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                return nested
    return value


def strip_removed_document_fields(value: object) -> object:
    """丢弃已从 4.1 契约删除的模型展示字段，保留其余结构校验。"""

    if isinstance(value, dict):
        return {
            key: strip_removed_document_fields(nested)
            for key, nested in value.items()
            if key not in _REMOVED_DOCUMENT_FIELDS
        }
    if isinstance(value, list):
        return [strip_removed_document_fields(item) for item in value]
    return value


def _strip_markdown_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_load_json(value: str) -> object | None:
    for strict in (True, False):
        try:
            parsed: Any = json.loads(
                value,
                strict=strict,
                parse_constant=_reject_json_constant,
            )
            return cast(object, parsed)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _json_fragments(text: str) -> Iterator[str]:
    for start, opener in enumerate(text):
        if opener not in "[{":
            continue
        stack = [opener]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                mismatched = (
                    (char == "]" and stack[-1] != "[")
                    or (char == "}" and stack[-1] != "{")
                )
                if not stack or mismatched:
                    break
                stack.pop()
                if not stack:
                    yield text[start : index + 1]
                    break


def _reject_json_constant(_value: str) -> None:
    raise ValueError("JSON 常量非法")
