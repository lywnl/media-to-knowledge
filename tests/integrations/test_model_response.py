from __future__ import annotations

from video_demo.integrations.model_response import (
    parse_json_content,
    strip_removed_document_fields,
    unwrap_single_response_envelope,
)


def test_parse_json_content_removes_think_and_markdown_fence() -> None:
    assert parse_json_content("<think>忽略</think>\n```json\n{\"ok\":1}\n```") == {
        "ok": 1,
    }


def test_unwrap_single_response_envelope_only_once() -> None:
    assert unwrap_single_response_envelope({"result": {"value": 1}}) == {"value": 1}
    nested = {"result": {"data": {"value": 1}}}
    assert unwrap_single_response_envelope(nested) == {"data": {"value": 1}}


def test_strip_removed_document_fields_is_recursive() -> None:
    value = {
        "overview_zh": "概览",
        "key_points": ["旧字段"],
        "chapter": {"retrieval_text": "旧检索", "claims": []},
    }

    assert strip_removed_document_fields(value) == {
        "overview_zh": "概览",
        "chapter": {"claims": []},
    }
