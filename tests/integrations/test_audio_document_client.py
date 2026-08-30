from __future__ import annotations

import json

import httpx
import pytest

from video_demo.domain.audio_plan import AudioBaseSegment, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import VideoDemoError
from video_demo.integrations.audio_document_client import AudioDocumentClient
from video_demo.integrations.audio_document_port import AudioChapterPlanningRequest


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
