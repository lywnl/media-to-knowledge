from __future__ import annotations

import json

import httpx
import pytest

from video_demo.domain.audio_plan import AudioBaseSegment, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.audio_document_client import (
    AudioDocumentClient,
    _pydantic_error_summary,
)
from video_demo.integrations.audio_document_port import (
    AudioChapterBoundaryCoordinationRequest,
    AudioChapterPlanningRequest,
    AudioModelResponseValidationError,
)


def test_audio_client_expands_compact_ranges_and_uses_audio_schema() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        body = {
            "chapter_drafts": [
                {
                    "start_segment_index": 0,
                    "end_segment_index": 3,
                    "title_hint": "主题",
                },
            ],
            "key_points": ["旧版本字段"],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
            request=request,
        )

    segments = tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(f"asr_{index}",),
            transcript_source="ASR",
        )
        for index in range(3)
    )
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            text="内容",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(3)
    )
    request = AudioChapterPlanningRequest(
        title_hint="音频",
        duration_ms=90_000,
        segments=segments,
        transcript_evidence=evidence,
        document_config=AudioDocumentConfig(),
        prompt_version="audio-chapter-planner-v1",
    )
    client = AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=1,
        sleeper=lambda _delay: None,
    )

    response = client.plan_chapters(request)

    assert response.chapter_drafts[0].segment_refs == tuple(item.segment_id for item in segments)
    assert captured[0]["response_format"]["json_schema"]["name"] == "audio_chapter_planning_v1"  # type: ignore[index]
    encoded = json.dumps(captured[0], ensure_ascii=False)
    assert "visual_mode" not in encoded
    assert "keyframe" not in encoded.lower()


def test_audio_client_copies_video_latency_payload_settings() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "chapter_drafts": [
                                        {
                                            "start_segment_index": 0,
                                            "end_segment_index": 3,
                                            "title_hint": "主题",
                                        },
                                    ],
                                },
                            ),
                        },
                    },
                ],
            },
            request=request,
        )

    client = AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=1,
        sleeper=lambda _delay: None,
    )
    client.plan_chapters(_planning_request())

    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["max_tokens"] == 1_024


def _client_for_body(body: dict[str, object]) -> AudioDocumentClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(body)}}]},
            request=request,
        )

    return AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=1,
        sleeper=lambda _delay: None,
    )


def _planning_request() -> AudioChapterPlanningRequest:
    segments = tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(f"asr_{index}",),
            transcript_source="ASR",
        )
        for index in range(3)
    )
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            text="内容",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(3)
    )
    return AudioChapterPlanningRequest(
        title_hint="音频",
        duration_ms=90_000,
        segments=segments,
        transcript_evidence=evidence,
        document_config=AudioDocumentConfig(),
        prompt_version="audio-chapter-planner-v1",
    )


def test_audio_client_normalizes_final_segment_count_plus_one() -> None:
    client = _client_for_body(
        {
            "chapter_drafts": [
                {"start_segment_index": 0, "end_segment_index": 4, "title_hint": "主题"},
            ],
        },
    )

    response = client.plan_chapters(_planning_request())

    assert response.chapter_drafts[0].segment_refs == (
        "audio_segment_0",
        "audio_segment_1",
        "audio_segment_2",
    )


def test_audio_client_wraps_invalid_normalization_as_model_response_error() -> None:
    client = _client_for_body(
        {
            "chapter_drafts": [
                {"start_segment_index": 0, "end_segment_index": 5, "title_hint": "越界"},
            ],
        },
    )

    with pytest.raises(AudioModelResponseValidationError) as raised:
        client.plan_chapters(_planning_request())

    assert raised.value.invalid_response.validation_errors == ("音频章节片段范围越界",)


def test_audio_client_unwraps_result_envelope_and_top_level_draft_array() -> None:
    client = _client_for_body(
        {
            "result": [
                {
                    "start_segment_index": "0",
                    "end_segment_index": "3",
                    "title": "主题",
                    "key_points": ["旧字段"],
                },
            ],
        },
    )

    response = client.plan_chapters(_planning_request())

    assert response.chapter_drafts[0].title_hint == "主题"
    assert response.chapter_drafts[0].segment_refs == tuple(
        item.segment_id for item in _planning_request().segments
    )


def test_audio_client_retries_temporary_failure_before_parsing() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "chapter_drafts": [
                                        {
                                            "start_segment_index": 0,
                                            "end_segment_index": 3,
                                            "title_hint": "主题",
                                        },
                                    ],
                                },
                            ),
                        },
                    },
                ],
            },
            request=request,
        )

    client = AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=2,
        sleeper=sleeps.append,
    )

    response = client.plan_chapters(_planning_request())

    assert response.chapter_drafts[0].title_hint == "主题"
    assert attempts == 2
    assert sleeps == [1]


def test_audio_client_does_not_retry_capability_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, request=request)

    client = AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=3,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.plan_chapters(_planning_request())

    assert raised.value.code == ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE
    assert attempts == 1


def test_audio_client_validation_summary_keeps_only_safe_provider_reason() -> None:
    assert _pydantic_error_summary(
        {
            "loc": ("chapter_drafts", 0, "title_hint"),
            "type": "value_error",
            "ctx": {"error": "标题内容不符合约束"},
        },
    ) == "chapter_drafts.0.title_hint:value_error:标题内容不符合约束"
    assert _pydantic_error_summary(
        {
            "loc": ("chapter_drafts",),
            "type": "value_error",
            "ctx": {"error": "https://example.invalid/secret"},
        },
    ) == "chapter_drafts:value_error"


def test_audio_client_normalizes_boundary_coordination_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "result": {
                                        "decisions": [
                                            {
                                                "boundary_index": "0",
                                                "decision": "KEEP",
                                            },
                                        ],
                                    },
                                },
                            ),
                        },
                    },
                ],
            },
            request=request,
        )

    request = AudioChapterBoundaryCoordinationRequest(
        boundaries=(
            {
                "boundary_index": 0,
                "left_title_hint": "左",
                "right_title_hint": "右",
                "left_duration_ms": 60_000,
                "right_duration_ms": 60_000,
                "left_tail_evidence": (),
                "right_head_evidence": (),
            },
        ),
        prompt_version="audio-chapter-boundary-coordinator-v1",
    )
    client = AudioDocumentClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://text.example.test/v1",
        api_key="secret",
        model_id="text-model",
        max_attempts=1,
        sleeper=lambda _delay: None,
    )

    response = client.coordinate_chapter_boundaries(request)

    assert response.decisions[0].boundary_index == 0
    assert response.decisions[0].decision == "KEEP"


def test_audio_client_ignores_trailing_empty_draft_without_repair() -> None:
    client = _client_for_body(
        {
            "chapter_drafts": [
                {
                    "start_segment_index": 0,
                    "end_segment_index": 3,
                    "title_hint": "章节",
                },
                {
                    "start_segment_index": 3,
                    "end_segment_index": 3,
                    "title_hint": "多余空章",
                },
            ],
        },
    )

    response = client.plan_chapters(_planning_request())

    assert len(response.chapter_drafts) == 1
    assert response.chapter_drafts[0].segment_refs == tuple(
        item.segment_id for item in _planning_request().segments
    )


@pytest.mark.parametrize(
    ("drafts", "message"),
    (
        (
            [
                {"start_segment_index": 0, "end_segment_index": 4, "title_hint": "第一"},
                {"start_segment_index": 2, "end_segment_index": 3, "title_hint": "第二"},
            ],
            "范围越界",
        ),
        (
            [{"start_segment_index": 0, "end_segment_index": 5, "title_hint": "主题"}],
            "范围越界",
        ),
        (
            [{"start_segment_index": 1, "end_segment_index": 1, "title_hint": "主题"}],
            "范围为空",
        ),
        (
            [
                {"start_segment_index": 0, "end_segment_index": 1, "title_hint": "第一"},
                {"start_segment_index": 2, "end_segment_index": 3, "title_hint": "第二"},
            ],
            "不连续",
        ),
        (
            [
                {"start_segment_index": 0, "end_segment_index": 2, "title_hint": "第一"},
                {"start_segment_index": 1, "end_segment_index": 3, "title_hint": "第二"},
            ],
            "不连续",
        ),
        (
            [{"start_segment_index": 0, "end_segment_index": 2, "title_hint": "主题"}],
            "完整覆盖",
        ),
    ),
)
def test_audio_client_rejects_invalid_segment_boundaries(
    drafts: list[dict[str, object]],
    message: str,
) -> None:
    client = _client_for_body({"chapter_drafts": drafts})

    with pytest.raises(VideoDemoError, match=message):
        client.plan_chapters(_planning_request())
