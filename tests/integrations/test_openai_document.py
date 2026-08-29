from __future__ import annotations

import hashlib
import json
import logging

import httpx
import pytest
from pydantic import ValidationError

from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import BaseSegment, ChapterPlan
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanRepairRequest,
    ChapterWritingRepairRequest,
    ChapterWritingRequest,
    GlobalChapterInput,
    GlobalWritingRepairRequest,
    GlobalWritingRequest,
    InvalidModelResponse,
    ModelResponseValidationError,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import (
    prompt_for_writing,
    prompt_for_writing_repair,
)
from video_demo.integrations.openai_document import OpenAIDocumentClient


def _segment() -> BaseSegment:
    return BaseSegment(
        segment_id="segment_001",
        start_ms=0,
        end_ms=10_000,
        evidence_refs=("asr_001",),
        transcript_source="ASR",
    )


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="请打开设置页面",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _context() -> DocumentWritingContext:
    return DocumentWritingContext(
        run_id="run_001",
        asset_sha256="a" * 64,
        title_hint="测试视频",
        duration_ms=10_000,
        transcript_source="ASR",
        document_config=DocumentGenerationConfig(),
    )


def _planning_request() -> ChapterPlanningRequest:
    return ChapterPlanningRequest(
        title_hint="测试视频",
        duration_ms=10_000,
        segments=(_segment(),),
        transcript_evidence=(_speech(),),
        document_config=DocumentGenerationConfig(),
        prompt_version="chapter-planner-v1",
    )


def _chapter() -> ChapterPlan:
    return ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="SINGLE",
        semantic_targets=(),
        base_coverage_targets=(),
    )


def _client(
    handler: httpx.MockTransport,
    *,
    max_attempts: int = 1,
    max_response_bytes: int = 2 * 1024 * 1024,
) -> OpenAIDocumentClient:
    return OpenAIDocumentClient(
        httpx.Client(transport=handler),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        max_attempts=max_attempts,
        max_response_bytes=max_response_bytes,
        sleeper=lambda _delay: None,
    )


def _provider_response(request: httpx.Request, body: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}]},
        request=request,
    )


def test_plan_request_uses_main_schema_prompt_and_temperature_zero() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "segment_refs": ["segment_001"],
                        "title_hint": "设置页面",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    result = _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    assert result.chapter_drafts[0].segment_refs == ("segment_001",)
    assert payloads[0]["temperature"] == 0
    assert payloads[0]["max_tokens"] == 8_192
    assert payloads[0]["response_format"]["json_schema"]["name"] == (  # type: ignore[index]
        "chapter_planning_v1"
    )
    assert "chapter-planner-v1" in payloads[0]["messages"][0]["content"]  # type: ignore[index]
    assert "完整分区" in payloads[0]["messages"][0]["content"]  # type: ignore[index]
    assert "1~3" in payloads[0]["messages"][0]["content"]  # type: ignore[index]
    assert "30 秒" in payloads[0]["messages"][0]["content"]  # type: ignore[index]
    assert "fine 60~120 秒" in payloads[0]["messages"][0]["content"]  # type: ignore[index]
    assert "text-secret" not in json.dumps(payloads)


def test_plan_accepts_content_blocks_and_markdown_wrapped_json() -> None:
    body = {
        "chapter_drafts": [
            {
                "segment_refs": ["segment_001"],
                "title_hint": "设置页面",
                "visual_mode": "NONE",
                "semantic_targets": [],
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        message = "结果如下：\n```json\n" + json.dumps(body, ensure_ascii=False) + "\n```"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "thinking", "text": "内部思考"},
                                {"type": "text", "text": message},
                            ],
                        },
                    },
                ],
            },
            request=request,
        )

    result = _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    assert result.chapter_drafts[0].title_hint == "设置页面"


def test_writing_requests_disable_thinking_for_latency() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        schema = payload["response_format"]["json_schema"]["name"]
        body = _writing_body() if schema == "chapter_writing_v2" else _global_body()
        return _provider_response(request, body)

    writing = ChapterWritingRequest(
        context=_context(),
        chapter=_chapter(),
        transcript_evidence=(_speech(),),
        visual_observations=(),
        prompt_version="chapter-writer-v1",
    )
    global_request = GlobalWritingRequest(
        context=_context(),
        chapters=(
            GlobalChapterInput(
                start_ms=0,
                end_ms=10_000,
                chapter_id="chapter_001",
                title="章节",
                summary_zh="摘要",
                content_status="GROUNDED",
            ),
        ),
        prompt_version="global-editor-v1",
    )

    client = _client(httpx.MockTransport(handler))
    client.write_chapter(writing)
    client.organize_document(global_request)

    assert [payload["thinking"] for payload in payloads] == [
        {"type": "disabled"},
        {"type": "disabled"},
    ]


def test_writing_prompts_separate_observation_and_content_references() -> None:
    writing = ChapterWritingRequest(
        context=_context(),
        chapter=_chapter(),
        transcript_evidence=(_speech(),),
        visual_observations=(),
        prompt_version="chapter-writer-v1",
    )
    repair = ChapterWritingRepairRequest(
        request=writing,
        invalid_response=InvalidModelResponse(
            content_sha256="b" * 64,
            validation_errors=("claims.evidence_refs:unknown_reference",),
        ),
        allowed_evidence_ids=("asr_001",),
        prompt_version="chapter-writer-repair-v1",
    )

    main_instruction = prompt_for_writing(writing)[1]
    repair_instruction = prompt_for_writing_repair(repair)[1]

    for instruction in (main_instruction, repair_instruction):
        assert (
            "title_evidence_refs、summary_evidence_refs、普通正文 evidence_refs 和 "
            "claims.evidence_refs"
        ) in instruction
        assert (
            "只能引用 ASR/字幕 evidence_id 或 visual_observation.evidence_id"
        ) in instruction
        assert (
            "keyframe_refs、keyframe_id、visual_content_id、visual_fact_id、"
            "source_keyframe_refs"
        ) in instruction
        assert "只有 VisualBlock.visual_content_refs 才能引用视觉内容 ID" in instruction


def test_writing_prompts_require_discriminated_body_block_type() -> None:
    writing = ChapterWritingRequest(
        context=_context(),
        chapter=_chapter(),
        transcript_evidence=(_speech(),),
        visual_observations=(),
        prompt_version="chapter-writer-v1",
    )
    repair = ChapterWritingRepairRequest(
        request=writing,
        invalid_response=InvalidModelResponse(
            content_sha256="b" * 64,
            validation_errors=("body_blocks.0:union_tag_not_found",),
        ),
        allowed_evidence_ids=("asr_001",),
        prompt_version="chapter-writer-repair-v1",
    )

    for instruction in (prompt_for_writing(writing)[1], prompt_for_writing_repair(repair)[1]):
        assert "每个 body_blocks 项都必须包含 block_type" in instruction
        assert "PARAGRAPH、BULLET_LIST、QUOTE、CODE、TABLE、FORMULA 或 VISUAL" in instruction
        assert "所有 evidence_id 必须从输入白名单逐字复制" in instruction
        assert "输出前逐项检查每个引用都在上述白名单中" in instruction


def test_compact_planning_request_maps_indexes_back_to_stable_ids() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 1,
                        "title_hint": "设置页面",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [
                            {
                                "query_zh": "页面设置",
                                "anchor_transcript_indexes": [0],
                            },
                        ],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    result = client.plan_chapters(_planning_request())

    assert result.chapter_drafts[0].segment_refs == ("segment_001",)
    assert result.chapter_drafts[0].semantic_targets[0].anchor_evidence_refs == ("asr_001",)
    assert payloads[0]["response_format"]["json_schema"]["name"] == (  # type: ignore[index]
        "chapter_planning_compact_v1"
    )
    assert payloads[0]["max_tokens"] == 1_024
    assert payloads[0]["thinking"] == {"type": "disabled"}
    sent = json.loads(payloads[0]["messages"][1]["content"].split("\n", 1)[1])  # type: ignore[index]
    assert sent["segments"][0] == [0, 0, 10_000, [0]]
    assert sent["segment_transcript_index_ranges"] == [[0, 0]]
    assert sent["transcript_evidence"][0] == [0, 1_000, 2_000, "请打开设置页面"]
    assert "segment_001" not in json.dumps(sent, ensure_ascii=False)
    assert "asr_001" not in json.dumps(sent, ensure_ascii=False)


def test_compact_planning_output_budget_scales_with_batch_size() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 1,
                        "title_hint": "设置页面",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    client.plan_chapters(_planning_request())

    assert payloads[0]["max_tokens"] == 1_024


def test_compact_planning_large_batch_keeps_full_output_budget() -> None:
    payloads: list[dict[str, object]] = []
    segments = tuple(
        BaseSegment(
            segment_id=f"segment_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            evidence_refs=("asr_001",),
            transcript_source="ASR",
        )
        for index in range(48)
    )
    request = _planning_request().model_copy(
        update={"duration_ms": 480_000, "segments": segments},
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(http_request.content))
        return _provider_response(
            http_request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 48,
                        "title_hint": "设置页面",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    client.plan_chapters(request)

    assert payloads[0]["max_tokens"] == 8_192


def test_compact_planning_drops_anchor_from_another_chapter_without_repair() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 1,
                        "title_hint": "第一章",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [],
                    },
                    {
                        "start_segment_index": 1,
                        "end_segment_index": 2,
                        "title_hint": "第二章",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [
                            {
                                "query_zh": "错误锚点",
                                "anchor_transcript_indexes": [0],
                            },
                        ],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    result = client.plan_chapters(_planning_request().model_copy(
        update={
            "segments": (
                _segment(),
                _segment().model_copy(
                    update={
                        "segment_id": "segment_002",
                        "start_ms": 10_000,
                        "end_ms": 20_000,
                        "evidence_refs": ("asr_002",),
                    },
                ),
            ),
            "transcript_evidence": (
                _speech(),
                _speech().model_copy(
                    update={
                        "evidence_id": "asr_002",
                        "start_ms": 11_000,
                        "end_ms": 12_000,
                    },
                ),
            ),
            "duration_ms": 20_000,
        },
    ))

    assert result.chapter_drafts[1].visual_mode == "NONE"
    assert result.chapter_drafts[1].semantic_targets == ()


def test_compact_planning_drops_invalid_anchor_span_without_repair() -> None:
    first_segment = _segment()
    second_segment = first_segment.model_copy(
        update={
            "segment_id": "segment_002",
            "start_ms": 10_000,
            "end_ms": 60_000,
            "evidence_refs": ("asr_002",),
        },
    )
    request = _planning_request().model_copy(
        update={
            "segments": (first_segment, second_segment),
            "transcript_evidence": (
                _speech(),
                _speech().model_copy(
                    update={
                        "evidence_id": "asr_002",
                        "start_ms": 50_000,
                        "end_ms": 51_000,
                    },
                ),
            ),
            "duration_ms": 60_000,
        },
    )
    payloads: list[dict[str, object]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(http_request.content))
        return _provider_response(
            http_request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 2,
                        "title_hint": "章节",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [
                            {
                                "query_zh": "跨度过大的目标",
                                "anchor_transcript_indexes": [0, 1],
                            },
                        ],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    result = client.plan_chapters(request)

    assert len(payloads) == 1
    assert result.chapter_drafts[0].visual_mode == "NONE"
    assert result.chapter_drafts[0].semantic_targets == ()


def test_compact_planning_ignores_trailing_empty_draft_without_repair() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 1,
                        "title_hint": "章节",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                    {
                        "start_segment_index": 1,
                        "end_segment_index": 1,
                        "title_hint": "多余空章",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    result = client.plan_chapters(_planning_request())

    assert len(payloads) == 1
    assert len(result.chapter_drafts) == 1


def test_compact_planning_normalizes_last_end_index_one_past_segment_count() -> None:
    segments = tuple(
        _segment().model_copy(
            update={
                "segment_id": f"segment_{index:03d}",
                "start_ms": index * 10_000,
                "end_ms": (index + 1) * 10_000,
            },
        )
        for index in range(3)
    )
    request = _planning_request().model_copy(
        update={"segments": segments, "duration_ms": 30_000},
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        return _provider_response(
            http_request,
            {
                "chapter_drafts": [
                    {
                        "start_segment_index": 0,
                        "end_segment_index": 4,
                        "title_hint": "完整章节",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    client = OpenAIDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="text-secret",
        model_id="text-model",
        compact_planning=True,
        sleeper=lambda _delay: None,
    )

    result = client.plan_chapters(request)

    assert result.chapter_drafts[0].segment_refs == tuple(
        f"segment_{index:03d}" for index in range(3)
    )


def test_plan_request_allows_large_repair_budget_for_provider_validation() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(request, _planning_body())

    oversized = _planning_request().model_copy(
        update={
            "transcript_evidence": (
                _speech().model_copy(update={"text": "长" * 70_000}),
            ),
        },
    )

    result = _client(httpx.MockTransport(handler)).plan_chapters(oversized)

    assert result.chapter_drafts[0].segment_refs == ("segment_001",)
    assert len(payloads) == 1


def test_invalid_model_json_raises_bounded_repair_context_without_leaking_content() -> None:
    raw_message = '{"chapter_drafts":[{"segment_refs":[]}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw_message}}]},
            request=request,
        )

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    error = raised.value
    assert error.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID
    assert error.invalid_response.content_sha256 == hashlib.sha256(
        raw_message.encode("utf-8"),
    ).hexdigest()
    assert error.invalid_response.validation_errors
    assert error.invalid_response.safe_json_excerpt is not None
    assert raw_message not in str(error)
    assert raw_message not in json.dumps(error.details, ensure_ascii=False)


def test_invalid_model_response_logs_finish_reason_length_and_safe_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_message = '{"chapter_drafts":[]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": raw_message},
                    },
                ],
            },
            request=request,
        )

    caplog.set_level(logging.WARNING, logger="video_demo.integrations.openai_document")
    with pytest.raises(ModelResponseValidationError):
        _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "finish_reason=length" in message
    assert "response_bytes=" in message
    assert "chapter_drafts" in message
    assert raw_message not in message


def test_unknown_reference_raises_model_response_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "segment_refs": ["segment_unknown"],
                        "title_hint": "非法",
                        "visual_mode": "SINGLE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    assert raised.value.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID
    assert raised.value.invalid_response.validation_errors == (
        "chapter_drafts.segment_refs:unknown_reference",
    )


def test_repair_request_keeps_original_request_and_uses_repair_prompt() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            request,
            {
                "chapter_drafts": [
                    {
                        "segment_refs": ["segment_001"],
                        "title_hint": "修复章节",
                        "visual_mode": "NONE",
                        "semantic_targets": [],
                    },
                ],
            },
        )

    repair = ChapterPlanRepairRequest(
        request=_planning_request(),
        invalid_response=InvalidModelResponse(
            content_sha256="b" * 64,
            validation_errors=("chapter_drafts:too_short",),
            safe_json_excerpt='{"chapter_drafts":[]}',
        ),
        allowed_segment_ids=("segment_001",),
        allowed_transcript_ids=("asr_001",),
        prompt_version="chapter-planner-repair-v1",
    )

    _client(httpx.MockTransport(handler)).repair_chapter_plan(repair)

    payload = payloads[0]
    assert payload["response_format"]["json_schema"]["name"] == (  # type: ignore[index]
        "chapter_planning_repair_v1"
    )
    assert "chapter-planner-repair-v1" in payload["messages"][0]["content"]  # type: ignore[index]
    sent = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])  # type: ignore[index]
    assert sent["request"]["segments"][0] == [
        "segment_001",
        10_000,
        [0],
    ]
    assert "confidence" not in json.dumps(sent["request"], ensure_ascii=False)
    assert sent["invalid_response"]["content_sha256"] == "b" * 64
    assert "original" not in sent


def test_repair_request_rejects_a_silently_reduced_allowed_id_set() -> None:
    with pytest.raises(ValidationError, match="allowed_transcript_ids"):
        ChapterPlanRepairRequest(
            request=_planning_request(),
            invalid_response=InvalidModelResponse(
                content_sha256="b" * 64,
                validation_errors=("response:invalid",),
            ),
            allowed_segment_ids=("segment_001",),
            allowed_transcript_ids=(),
            prompt_version="chapter-planner-repair-v1",
        )


def test_invalid_model_excerpt_drops_secrets_urls_and_base64() -> None:
    unsafe_messages = (
        '{"api_key":"secret-value"}',
        '{"image":"https://example.test/private.jpg"}',
        '{"image":"data:image/jpeg;base64,' + "A" * 100 + '"}',
    )

    for raw_message in unsafe_messages:
        def handler(request: httpx.Request, message: str = raw_message) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": message}}]},
                request=request,
            )

        with pytest.raises(ModelResponseValidationError) as raised:
            _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())
        assert raised.value.invalid_response.safe_json_excerpt is None
        assert raw_message not in str(raised.value)


def test_non_json_message_hashes_message_content_instead_of_provider_envelope() -> None:
    raw_message = "not-json-model-content"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw_message}}]},
            request=request,
        )

    with pytest.raises(ModelResponseValidationError) as raised:
        _client(httpx.MockTransport(handler)).plan_chapters(_planning_request())

    assert raised.value.invalid_response.content_sha256 == hashlib.sha256(
        raw_message.encode("utf-8"),
    ).hexdigest()
    assert raised.value.invalid_response.safe_json_excerpt is None


def test_invalid_model_response_normalizes_control_characters_before_repair_context() -> None:
    invalid = invalid_model_response(
        b"raw",
        ("field:\x00bad\nvalue",),
        parsed_json={"text": "line\nvalue"},
    )

    assert invalid.validation_errors == ("field: bad value",)
    assert invalid.safe_json_excerpt is None

    surrogate = invalid_model_response(
        b"raw",
        ("field:invalid",),
        parsed_json={"text": "\ud800"},
    )
    assert surrogate.safe_json_excerpt is None

    emoji_joiner = invalid_model_response(
        b"raw",
        ("field:invalid",),
        parsed_json={"text": "程序员\u200d💻"},
    )
    assert emoji_joiner.safe_json_excerpt is None

    unsafe_key = invalid_model_response(
        b"raw",
        ("field:invalid",),
        parsed_json={"字段\u200d名": "value"},
    )
    assert unsafe_key.safe_json_excerpt is None

    suspicious_key = invalid_model_response(
        b"raw",
        ("field:invalid",),
        parsed_json={"https://example.invalid": "value"},
    )
    assert suspicious_key.safe_json_excerpt is None


@pytest.mark.parametrize(
    "suspicious_value",
    (
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "YWJjZGVmZ2hpamtsbW5vcA==",
        "A" * 80,
    ),
)
def test_invalid_model_response_excludes_common_tokens_and_short_base64(
    suspicious_value: str,
) -> None:
    invalid = invalid_model_response(
        b"raw",
        ("field:invalid",),
        parsed_json={"text": suspicious_value},
    )

    assert invalid.safe_json_excerpt is None


@pytest.mark.parametrize("field", ["validation_errors", "safe_json_excerpt"])
def test_invalid_model_response_model_rejects_directly_injected_sensitive_context(
    field: str,
) -> None:
    values: dict[str, object] = {
        "content_sha256": "f" * 64,
        "validation_errors": ("response:invalid",),
        "safe_json_excerpt": None,
    }
    values[field] = (
        ("sk-proj-abcdefghijklmnopqrstuvwxyz123456",)
        if field == "validation_errors"
        else '{"text":"https://example.invalid/secret"}'
    )

    with pytest.raises(ValidationError, match="疑似敏感信息"):
        InvalidModelResponse.model_validate(values)


def test_all_six_operations_use_distinct_schema_and_prompt_versions() -> None:
    payloads: list[dict[str, object]] = []
    response_by_schema = {
        "chapter_planning_v1": _planning_body(),
        "chapter_planning_repair_v1": _planning_body(),
        "chapter_writing_v2": _writing_body(),
        "chapter_writing_repair_v2": _writing_body(),
        "global_writing_v1": _global_body(),
        "global_writing_repair_v1": _global_body(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        schema = payload["response_format"]["json_schema"]["name"]
        return _provider_response(request, response_by_schema[schema])

    invalid = InvalidModelResponse(
        content_sha256="c" * 64,
        validation_errors=("root:invalid",),
    )
    planning = _planning_request()
    writing = ChapterWritingRequest(
        context=_context(),
        chapter=_chapter(),
        transcript_evidence=(_speech(),),
        visual_observations=(),
        prompt_version="chapter-writer-v1",
    )
    global_request = GlobalWritingRequest(
        context=_context(),
        chapters=(
            GlobalChapterInput(
                start_ms=0,
                end_ms=10_000,
                chapter_id="chapter_001",
                title="章节",
                summary_zh="摘要",
                content_status="GROUNDED",
            ),
        ),
        prompt_version="global-editor-v1",
    )
    client = _client(httpx.MockTransport(handler))

    client.plan_chapters(planning)
    client.repair_chapter_plan(
        ChapterPlanRepairRequest(
            request=planning,
            invalid_response=invalid,
            allowed_segment_ids=("segment_001",),
            allowed_transcript_ids=("asr_001",),
            prompt_version="chapter-planner-repair-v1",
        ),
    )
    client.write_chapter(writing)
    client.repair_chapter_writing(
        ChapterWritingRepairRequest(
            request=writing,
            invalid_response=invalid,
            allowed_evidence_ids=("asr_001",),
            prompt_version="chapter-writer-repair-v1",
        ),
    )
    client.organize_document(global_request)
    client.repair_global_writing(
        GlobalWritingRepairRequest(
            request=global_request,
            invalid_response=invalid,
            allowed_chapter_ids=("chapter_001",),
            prompt_version="global-editor-repair-v1",
        ),
    )

    assert [
        payload["response_format"]["json_schema"]["name"] for payload in payloads  # type: ignore[index]
    ] == list(response_by_schema)
    expected_prompts = (
        "chapter-planner-v1",
        "chapter-planner-repair-v1",
        "chapter-writer-v1",
        "chapter-writer-repair-v1",
        "global-editor-v1",
        "global-editor-repair-v1",
    )
    assert all(
        prompt in payload["messages"][0]["content"]  # type: ignore[index]
        for prompt, payload in zip(expected_prompts, payloads, strict=True)
    )


def test_text_client_retries_temporary_failure_but_not_capability_error() -> None:
    attempts = 0
    reported_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return _provider_response(request, _planning_body())

    def report_attempt() -> None:
        nonlocal reported_attempts
        reported_attempts += 1

    _client(httpx.MockTransport(handler), max_attempts=2).plan_chapters(
        _planning_request(),
        on_provider_attempt=report_attempt,
    )
    assert attempts == 2
    assert reported_attempts == 2

    attempts = 0

    def missing(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, request=request)

    with pytest.raises(VideoDemoError) as raised:
        _client(httpx.MockTransport(missing), max_attempts=3).plan_chapters(
            _planning_request(),
        )
    assert raised.value.code == ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE
    assert attempts == 1


def test_text_response_is_streamed_with_configured_byte_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _provider_response(request, {"chapter_drafts": []})

    with pytest.raises(VideoDemoError) as raised:
        _client(
            httpx.MockTransport(handler),
            max_response_bytes=10,
        ).plan_chapters(_planning_request())
    assert raised.value.code == ErrorCode.TEXT_LLM_RESPONSE_INVALID


def test_authentication_status_is_not_masked_by_an_oversized_error_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"x" * 100, request=request)

    with pytest.raises(VideoDemoError) as raised:
        _client(
            httpx.MockTransport(handler),
            max_response_bytes=10,
        ).plan_chapters(_planning_request())
    assert raised.value.code == ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED


def _planning_body() -> dict[str, object]:
    return {
        "chapter_drafts": [
            {
                "segment_refs": ["segment_001"],
                "title_hint": "章节",
                "visual_mode": "NONE",
                "semantic_targets": [],
            },
        ],
    }


def _writing_body() -> dict[str, object]:
    return {
        "title": "章节",
        "title_evidence_refs": ["asr_001"],
        "summary_zh": "摘要",
        "summary_evidence_refs": ["asr_001"],
        "body_blocks": [],
        "claims": [],
    }


def _global_body() -> dict[str, object]:
    return {
        "overview_zh": "概览",
        "key_points": [],
    }
