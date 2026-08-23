from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

import video_demo.integrations.qwen as qwen_module
from video_demo.domain.evidence import (
    AlignedWord,
    BoundingBox,
    OcrEvidence,
    OcrLine,
    SceneBoundary,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.qwen import QwenVideoClient
from video_demo.integrations.video_port import (
    SegmentSummaryInput,
    SegmentUnderstandingRequest,
    SummaryUnderstandingRequest,
    VideoClipInput,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowInput,
)


def _provider_response(content: object, *, model: str = "qwen3-vl-plus") -> httpx.Response:
    serialized = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_001",
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": serialized}}],
        },
    )


def _is_probe_request(request: httpx.Request) -> bool:
    payload = json.loads(request.content)
    response_format = payload.get("response_format", {})
    return response_format.get("json_schema", {}).get("name") == "qwen_capability_probe"


def _valid_segment(evidence_refs: tuple[str, ...] = ("asr_001",)) -> dict[str, object]:
    return {
        "title": "问候",
        "summary_zh": "讲者向观众问好。",
        "speakers": ["SPEAKER_01"],
        "languages": ["en"],
        "topics": ["问候"],
        "entities": [],
        "actions": ["问好"],
        "keywords": ["问候"],
        "original_keywords": ["Hello"],
        "evidence_refs": list(evidence_refs),
    }


def _segment_request(tmp_path: Path, *, text: str = "Hello") -> SegmentUnderstandingRequest:
    video = b"video-bytes"
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(video)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=100,
        end_ms=400,
        text=text,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    return SegmentUnderstandingRequest(
        clip=VideoClipInput(
            clip_id="clip_001",
            start_ms=0,
            end_ms=500,
            path=clip_path,
            mime_type="video/mp4",
            sha256=hashlib.sha256(video).hexdigest(),
        ),
        window=TimeRange(start_ms=0, end_ms=500),
        timeline=build_timeline((speech,)),
        evidence=(speech,),
    )


def _whole_request(remote_url: str) -> WholeVideoUnderstandingRequest:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=100,
        end_ms=400,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    return WholeVideoUnderstandingRequest(
        video=VideoClipInput(
            clip_id="full_video_001",
            start_ms=0,
            end_ms=500,
            source_url=remote_url,
            mime_type="video/mp4",
            sha256="a" * 64,
        ),
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=0,
                end_ms=500,
                timeline=build_timeline((speech,)),
                evidence=(speech,),
            ),
        ),
    )


def _multi_window_whole_request(
    remote_url: str,
    *,
    window_count: int,
) -> WholeVideoUnderstandingRequest:
    windows: list[WholeVideoWindowInput] = []
    for index in range(window_count):
        start_ms = index * 500
        speech = SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=start_ms + 100,
            end_ms=start_ms + 400,
            text=f"第 {index + 1} 个本地窗口",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        windows.append(
            WholeVideoWindowInput(
                window_id=f"window_{index:03d}",
                start_ms=start_ms,
                end_ms=start_ms + 500,
                timeline=build_timeline((speech,)),
                evidence=(speech,),
            ),
        )
    return WholeVideoUnderstandingRequest(
        video=VideoClipInput(
            clip_id="full_video_001",
            start_ms=0,
            end_ms=window_count * 500,
            source_url=remote_url,
            mime_type="video/mp4",
            sha256="a" * 64,
        ),
        windows=tuple(windows),
    )


def test_flash_maps_variable_coarse_summaries_back_to_every_local_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            {
                "group_summaries": [f"第 {index + 1} 组视觉语义" for index in range(8)],
                "title": "全片摘要",
                "summary_zh": "完整视频包含三十二个本地证据窗口。",
                "topics": ["完整视频理解"],
                "keywords": ["画面", "语音", "OCR"],
            },
            model="qwen3-vl-flash",
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    result = client.understand_video(
        _multi_window_whole_request(remote_url, window_count=32),
    )

    assert [item.window_id for item in result.windows] == [
        f"window_{index:03d}" for index in range(32)
    ]
    assert [item.understanding.summary_zh for item in result.windows] == [
        f"第 {index + 1} 个本地窗口" for index in range(32)
    ]
    assert [item.understanding.evidence_refs for item in result.windows] == [
        (f"asr_{index:03d}",) for index in range(32)
    ]
    assert result.summary.summary_zh == (
        "完整视频包含三十二个本地证据窗口。；"
        "第 1 组视觉语义；第 2 组视觉语义；第 3 组视觉语义；第 4 组视觉语义；"
        "第 5 组视觉语义；第 6 组视觉语义；第 7 组视觉语义；第 8 组视觉语义"
    )
    assert len(payloads) == 1
    evidence_text = payloads[0]["messages"][1]["content"][1]["text"]  # type: ignore[index]
    evidence_document = json.loads(
        str(evidence_text).removeprefix("UNTRUSTED_WHOLE_VIDEO_EVIDENCE_JSON\n"),
    )
    assert len(evidence_document["windows"]) == 32
    assert evidence_document["local_window_count"] == 32
    assert "groups" not in evidence_document


def test_whole_video_understanding_uses_one_video_url_and_one_http_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            {
                "group_summaries": ["讲者向观众问好。"],
                "title": "摘要",
                "summary_zh": "完整视频包含一段问候。",
                "topics": ["问候"],
                "keywords": ["问候"],
            },
            model="qwen3-vl-flash",
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    result = client.understand_video(_whole_request(remote_url))

    assert result.summary.title == "摘要"
    assert result.summary.summary_zh == "完整视频包含一段问候。；讲者向观众问好。"
    assert result.summary.topics == ("问候",)
    assert result.summary.keywords == ("问候",)
    assert result.windows[0].understanding.title == "Hello"
    assert result.windows[0].understanding.summary_zh == "Hello"
    assert result.windows[0].understanding.evidence_refs == ("asr_001",)
    assert len(payloads) == 1
    assert payloads[0]["response_format"] == {"type": "json_object"}
    serialized = json.dumps(payloads[0], ensure_ascii=False)
    assert serialized.count(remote_url) == 1
    assert "qwen_capability_probe" not in serialized
    assert "group_summaries" in serialized
    assert "自行选择合理的非空粗分组数量" in serialized
    assert "按时间顺序连续覆盖全片" in serialized
    assert "本地证据引用由程序绑定" in serialized
    assert "JSON Schema" not in serialized


def test_flash_group_summaries_only_enrich_video_summary_and_deduplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "group_summaries": ["重复分组", "  重复分组 ", "新增分组"],
                        "title": "摘要",
                        "summary_zh": "  紧凑摘要  ",
                    },
                    model="qwen3-vl-flash",
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
    )
    request = _multi_window_whole_request(remote_url, window_count=3)

    result = client.understand_video(request)

    assert result.summary.summary_zh == "紧凑摘要；重复分组；新增分组"
    assert all(
        "重复分组" not in item.understanding.summary_zh
        and "新增分组" not in item.understanding.summary_zh
        for item in result.windows
    )


def test_normal_whole_video_preserves_visual_facts_in_window_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    request = _whole_request(remote_url)

    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "windows": [
                            {
                                "window_id": "window_001",
                                "understanding": {
                                    "title": "窗口",
                                    "summary_zh": "局部摘要",
                                    "visual_facts": ["画面中的软件界面"],
                                    "evidence_refs": ["asr_001"],
                                },
                            },
                        ],
                        "summary": {"title": "视频", "summary_zh": "整体摘要"},
                    },
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
    )

    result = client.understand_video(request)

    assert result.windows[0].understanding.visual_facts == ("画面中的软件界面",)


def test_whole_video_binds_all_local_evidence_even_when_prompt_projects_a_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    base_request = _whole_request(remote_url)
    speech = base_request.windows[0].evidence[0]
    aligned_word = AlignedWord(
        evidence_id="word_001",
        start_ms=100,
        end_ms=200,
        text="H",
        language="en",
        probability=0.9,
    )
    evidence = (speech, aligned_word)
    request = base_request.model_copy(
        update={
            "windows": (
                base_request.windows[0].model_copy(
                    update={
                        "timeline": build_timeline(evidence),
                        "evidence": evidence,
                    },
                ),
            ),
        },
    )
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "group_summaries": ["问候"],
                        "title": "摘要",
                        "summary_zh": "摘要。",
                    },
                    model="qwen3-vl-flash",
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    result = client.understand_video(request)

    assert result.windows[0].understanding.evidence_refs == ("asr_001", "word_001")
    assert result.windows[0].understanding.title == "Hello"
    assert result.windows[0].understanding.keywords == ("Hello",)


def test_whole_video_transport_failure_never_retries_full_video_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    transport_calls = 0
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )

    def fail_transport(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(503)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(fail_transport)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        max_attempts=3,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_video(_whole_request(remote_url))

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert transport_calls == 1


def test_whole_video_local_semantics_sample_text_but_keep_all_evidence_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    speech = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 50,
            end_ms=index * 50 + 40,
            text=f"句子 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(10)
    )
    request = WholeVideoUnderstandingRequest(
        video=VideoClipInput(
            clip_id="full_video_001",
            start_ms=0,
            end_ms=500,
            source_url=remote_url,
            mime_type="video/mp4",
        ),
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=0,
                end_ms=500,
                timeline=build_timeline(speech),
                evidence=speech,
            ),
        ),
    )
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "group_summaries": ["完整窗口"],
                        "title": "摘要",
                        "summary_zh": "摘要。",
                    },
                    model="qwen3-vl-flash",
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    understanding = client.understand_video(request).windows[0].understanding

    assert understanding.keywords == ("句子 0", "句子 4", "句子 9")
    assert understanding.evidence_refs == tuple(item.evidence_id for item in speech)


@pytest.mark.parametrize("returned_group_count", [0, 2])
def test_whole_video_understanding_rejects_invalid_group_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returned_group_count: int,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "group_summaries": ["问候"] * returned_group_count,
                        "title": "摘要",
                        "summary_zh": "摘要。",
                    },
                    model="qwen3-vl-flash",
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_video(
            _whole_request(remote_url)
            if returned_group_count == 2
            else _multi_window_whole_request(remote_url, window_count=2),
        )

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID


def test_whole_video_understanding_rejects_extra_model_generated_binding_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = "https://private-bucket.oss-cn-hangzhou.aliyuncs.com/full.mp4?signed=1"
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {
                        "group_summaries": ["问候"],
                        "title": "摘要",
                        "summary_zh": "摘要。",
                        "evidence_index": 0,
                    },
                    model="qwen3-vl-flash",
                ),
            ),
        ),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        allowed_remote_video_hosts=frozenset(
            {"private-bucket.oss-cn-hangzhou.aliyuncs.com"},
        ),
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_video(_whole_request(remote_url))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID


def test_demo_fallback_builds_deterministic_semantics_when_qwen_is_unavailable(
    tmp_path: Path,
) -> None:
    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500))),
        base_url=None,
        api_key=None,
        model_id=None,
        allowed_video_root=tmp_path,
    )
    fallback_type = getattr(qwen_module, "DemoFallbackVideoUnderstanding", None)
    assert fallback_type is not None
    fallback = fallback_type(client)
    request = _segment_request(tmp_path, text="Hello demo")

    result = fallback.understand_segment(request)

    assert result.title == "Hello demo"
    assert result.summary_zh == "Hello demo"
    assert result.evidence_refs == ("asr_001",)
    assert result.original_keywords == ()
    assert fallback.degraded_warnings == ("DEMO_DEGRADED_QWEN",)

    summary = fallback.summarize_video(
        SummaryUnderstandingRequest(
            segments=(SegmentSummaryInput(segment_ref="clip_001", understanding=result),),
        )
    )
    assert summary.summary_zh == "Hello demo"


def test_subtitle_is_projected_separately_and_preferred_by_demo_fallback(
    tmp_path: Path,
) -> None:
    request = _segment_request(tmp_path, text="ASR 文本")
    subtitle = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=100,
        end_ms=400,
        text="字幕文本",
        language="zh",
        stream_index=2,
    )
    mixed = request.model_copy(
        update={
            "timeline": build_timeline((*request.evidence, subtitle)),
            "evidence": (*request.evidence, subtitle),
        },
    )
    whole_request = WholeVideoUnderstandingRequest(
        video=mixed.clip,
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=mixed.window.start_ms,
                end_ms=mixed.window.end_ms,
                timeline=mixed.timeline,
                evidence=mixed.evidence,
            ),
        ),
    )
    projected = json.loads(
        qwen_module.render_whole_video_evidence(whole_request).removeprefix(
            "UNTRUSTED_WHOLE_VIDEO_EVIDENCE_JSON\n"
        ),
    )

    assert projected["windows"][0]["evidence"]["subtitles"]["texts"] == [
        "字幕文本"
    ]
    assert projected["windows"][0]["evidence"]["asr"]["texts"] == ["ASR 文本"]
    assert qwen_module._fallback_segment_understanding(mixed).summary_zh == "字幕文本"
    assert "字幕" in qwen_module.WHOLE_VIDEO_SYSTEM_INSTRUCTION


def test_demo_fallback_builds_all_whole_video_windows_and_summary(
    tmp_path: Path,
) -> None:
    segment_request = _segment_request(tmp_path, text="Hello full video")
    request = WholeVideoUnderstandingRequest(
        video=segment_request.clip,
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=segment_request.window.start_ms,
                end_ms=segment_request.window.end_ms,
                timeline=segment_request.timeline,
                evidence=segment_request.evidence,
            ),
        ),
    )

    class Unavailable:
        def understand_video(
            self,
            _request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "服务暂时不可用")

    fallback = qwen_module.DemoFallbackVideoUnderstanding(Unavailable())

    result = fallback.understand_video(request)

    assert [item.window_id for item in result.windows] == ["window_001"]
    assert result.windows[0].understanding.title == "Hello full video"
    assert result.summary.summary_zh == "Hello full video"
    assert fallback.degraded_warnings == ("DEMO_DEGRADED_QWEN",)


def test_demo_fallback_keeps_ocr_out_of_visual_facts(tmp_path: Path) -> None:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=100,
        end_ms=400,
        text="讲解画面",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    ocr = OcrEvidence(
        evidence_id="ocr_001",
        start_ms=100,
        end_ms=400,
        keyframe_id="keyframe_001",
        timestamp_ms=200,
        language="zh",
        lines=(
            OcrLine(
                text="页面标题",
                bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
                confidence=0.9,
            ),
        ),
        provider_request_id="ocr-request-001",
    )
    scene = SceneBoundary(
        evidence_id="scene_001",
        start_ms=100,
        end_ms=400,
        transition="candidate",
        score=0.8,
    )
    request = _segment_request(tmp_path).model_copy(
        update={
            "timeline": build_timeline((speech, ocr, scene)),
            "evidence": (speech, ocr, scene),
        },
    )

    result = qwen_module._fallback_segment_understanding(request)

    assert result.visual_facts == ("画面场景",)
    assert "画面文字：页面标题" not in result.visual_facts


def test_merge_summary_units_deduplicates_and_caps_video_summary() -> None:
    units = ("重复摘要", "重复摘要", "新的摘要" * 2_000)

    result = qwen_module._merge_summary_units(units, "主摘要")

    assert result.startswith("主摘要；重复摘要；新的摘要")
    assert len(result) == 4_000
    assert result.endswith("…")


def test_first_segment_uses_current_verified_clip_for_probe_then_understanding(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert result.title == "问候"
    assert len(payloads) == 2
    assert payloads[0]["response_format"]["json_schema"]["name"] == (  # type: ignore[index]
        "qwen_capability_probe"
    )
    probe_video = payloads[0]["messages"][0]["content"][0]["video_url"]["url"]  # type: ignore[index]
    segment_video = payloads[1]["messages"][1]["content"][0]["video_url"]["url"]  # type: ignore[index]
    assert probe_video == segment_video
    assert str(probe_video).endswith("dmlkZW8tYnl0ZXM=")


def test_qwen_normalizes_and_deduplicates_keywords_before_returning_semantics(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        return _provider_response(
            {
                **_valid_segment(),
                "keywords": [" AI  共创社群 ", "Codex"],
                "original_keywords": ["ai 共创社群", "codex", "HNSW"],
            },
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert result.keywords == ("AI 共创社群", "Codex")
    assert result.original_keywords == ("HNSW",)


def test_diagnostic_calls_expose_only_validated_provider_receipts(
    tmp_path: Path,
) -> None:
    response_ids = iter(("probe-response-001", "segment-response-001"))

    def handler(request: httpx.Request) -> httpx.Response:
        content = {"supported": True} if _is_probe_request(request) else _valid_segment()
        response = _provider_response(content)
        payload = response.json()
        payload["id"] = next(response_ids)
        return httpx.Response(200, json=payload)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    request = _segment_request(tmp_path)

    capabilities, probe_receipt = client.probe_capabilities_with_receipt(request.clip)
    result, segment_receipt = client.understand_segment_with_receipt(request)

    assert capabilities.model_id == "qwen3-vl-plus"
    assert result.title == "问候"
    assert (probe_receipt.response_id, probe_receipt.http_status) == (
        "probe-response-001",
        200,
    )
    assert (segment_receipt.response_id, segment_receipt.http_status) == (
        "segment-response-001",
        200,
    )


def test_lazy_api_key_provider_is_called_only_on_first_http_operation(
    tmp_path: Path,
) -> None:
    provider_calls = 0
    authorizations: list[str] = []

    def api_key_provider() -> SecretStr:
        nonlocal provider_calls
        provider_calls += 1
        return SecretStr("  qwen-secret  ")

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["Authorization"])
        return httpx.Response(200)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key_provider=api_key_provider,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    assert provider_calls == 0
    client._post_with_retry({})  # type: ignore[attr-defined]
    client._post_with_retry({})  # type: ignore[attr-defined]

    assert provider_calls == 1
    assert authorizations == ["Bearer qwen-secret", "Bearer qwen-secret"]


@pytest.mark.parametrize(
    "invalid_secret",
    [
        "\rstatic-secret-marker",
        "static-secret-marker\r",
        "static\rsecret-marker",
        "\nstatic-secret-marker",
        "static-secret-marker\n",
        "static\nsecret-marker",
        "\r\nstatic-secret-marker\r\n",
        "static\r\nsecret-marker",
        "\tstatic-secret-marker\t",
        "static\tsecret-marker",
        "\x0bstatic-secret-marker\x0b",
        "static\x0bsecret-marker",
        "\x0cstatic-secret-marker\x0c",
        "static\x0csecret-marker",
        "\x00static-secret-marker",
        "static-secret-marker\x00",
        "static\x00secret-marker",
        "\x7fstatic-secret-marker",
        "static-secret-marker\x7f",
        "static\x7fsecret-marker",
        "\x85static-secret-marker",
        "static-secret-marker\x85",
        "static\x85secret-marker",
        "\u00a0static-secret-marker\u00a0",
        "static\u00a0secret-marker",
        "\u3000static-secret-marker\u3000",
        "static\u3000secret-marker",
    ],
    ids=[
        "leading-cr",
        "trailing-cr",
        "middle-cr",
        "leading-lf",
        "trailing-lf",
        "middle-lf",
        "edge-crlf",
        "middle-crlf",
        "edge-tab",
        "middle-tab",
        "edge-vt",
        "middle-vt",
        "edge-ff",
        "middle-ff",
        "leading-nul",
        "trailing-nul",
        "middle-nul",
        "leading-del",
        "trailing-del",
        "middle-del",
        "leading-c1-nel",
        "trailing-c1-nel",
        "middle-c1-nel",
        "edge-no-break-space",
        "middle-no-break-space",
        "edge-ideographic-space",
        "middle-ideographic-space",
    ],
)
def test_static_api_key_rejects_non_visible_ascii_before_transport_without_leakage(
    tmp_path: Path,
    invalid_secret: str,
) -> None:
    marker = "static-secret-marker"
    transport_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("非法静态 Qwen 凭据不得进入 transport")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key=invalid_secret,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client._post_with_retry({})  # type: ignore[attr-defined]

    error = raised.value
    rendered = "".join(traceback.format_exception(error))
    assert error.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert transport_calls == 0
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in repr(error.args)
    assert marker not in rendered
    assert error.__cause__ is None


def test_provider_rejects_secret_str_subclass_with_hostile_str_without_leakage(
    tmp_path: Path,
) -> None:
    marker = "qwen-hostile-secret-marker"
    provider_calls = 0
    transport_calls = 0

    class HostileStr(str):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError(marker)

    class HostileSecretStr(SecretStr):
        def get_secret_value(self) -> str:
            return HostileStr("qwen-secret")

    def api_key_provider() -> SecretStr:
        nonlocal provider_calls
        provider_calls += 1
        return HostileSecretStr("unused")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("恶意 Qwen 凭据不得进入 transport")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key_provider=api_key_provider,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client._post_with_retry({})  # type: ignore[attr-defined]

    error = raised.value
    rendered = "".join(traceback.format_exception(error))
    assert error.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert provider_calls == 1
    assert transport_calls == 0
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in repr(error.args)
    assert marker not in rendered
    assert error.__cause__ is None


def test_provider_rejects_non_ascii_secret_before_transport_without_leakage(
    tmp_path: Path,
) -> None:
    marker = "qwen-secret-marker-密钥"
    provider_calls = 0
    transport_calls = 0

    def api_key_provider() -> SecretStr:
        nonlocal provider_calls
        provider_calls += 1
        return SecretStr(marker)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("非法 Qwen 凭据不得进入 transport")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key_provider=api_key_provider,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client._post_with_retry({})  # type: ignore[attr-defined]

    error = raised.value
    rendered = "".join(traceback.format_exception(error))
    assert error.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert provider_calls == 1
    assert transport_calls == 0
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in repr(error.args)
    assert marker not in rendered
    assert error.__cause__ is None


@pytest.mark.parametrize(
    "invalid_secret",
    ["   ", "qwen-secret-marker-密钥"],
)
def test_invalid_provider_secret_is_not_cached_and_can_recover(
    tmp_path: Path,
    invalid_secret: str,
) -> None:
    provider_calls = 0
    authorizations: list[str] = []
    values = iter((SecretStr(invalid_secret), SecretStr("  qwen-secret  ")))

    def api_key_provider() -> SecretStr:
        nonlocal provider_calls
        provider_calls += 1
        return next(values)

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["Authorization"])
        return httpx.Response(200)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key_provider=api_key_provider,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client._post_with_retry({})  # type: ignore[attr-defined]
    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE

    client._post_with_retry({})  # type: ignore[attr-defined]

    assert provider_calls == 2
    assert authorizations == ["Bearer qwen-secret"]


def test_successful_capability_probe_is_cached_once_across_concurrent_windows(
    tmp_path: Path,
) -> None:
    first_probe_entered = threading.Event()
    second_probe_entered = threading.Event()
    release_first_probe = threading.Event()
    counter_lock = threading.Lock()
    probe_calls = 0
    segment_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal probe_calls, segment_calls
        if _is_probe_request(request):
            with counter_lock:
                probe_calls += 1
                current_probe = probe_calls
            if current_probe == 1:
                first_probe_entered.set()
                assert release_first_probe.wait(timeout=2)
            else:
                second_probe_entered.set()
            return _provider_response({"supported": True})
        with counter_lock:
            segment_calls += 1
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    first_request = _segment_request(tmp_path)
    second_request = _segment_request(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.understand_segment, first_request)
        assert first_probe_entered.wait(timeout=1)
        second = executor.submit(client.understand_segment, second_request)
        second_probe_entered.wait(timeout=0.2)
        release_first_probe.set()
        assert first.result(timeout=2).title == "问候"
        assert second.result(timeout=2).title == "问候"

    assert probe_calls == 1
    assert segment_calls == 2


def test_concurrent_first_probe_serializes_clip_read_with_capability_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_probe_entered = threading.Event()
    release_first_probe = threading.Event()
    read_lock = threading.Lock()
    read_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            first_probe_entered.set()
            assert release_first_probe.wait(timeout=2)
            return _provider_response({"supported": True})
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    request = _segment_request(tmp_path)
    original_read = client._read_verified_clip  # type: ignore[attr-defined]

    def counted_read(clip: VideoClipInput) -> bytes:
        nonlocal read_calls
        with read_lock:
            read_calls += 1
        return original_read(clip)

    monkeypatch.setattr(client, "_read_verified_clip", counted_read)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.understand_segment, request)
        assert first_probe_entered.wait(timeout=1)
        second = executor.submit(client.understand_segment, request)
        threading.Event().wait(0.1)
        with read_lock:
            reads_while_probe_blocked = read_calls
        release_first_probe.set()
        assert first.result(timeout=2).title == "问候"
        assert second.result(timeout=2).title == "问候"

    assert reads_while_probe_blocked == 1


@pytest.mark.parametrize(
    ("base_url", "api_key", "model_id"),
    [
        (None, "qwen-secret", "qwen3-vl-plus"),
        ("https://dashscope.example/compatible-mode/v1", None, "qwen3-vl-plus"),
        ("https://dashscope.example/compatible-mode/v1", "qwen-secret", None),
        ("   ", "qwen-secret", "qwen3-vl-plus"),
        ("https://dashscope.example/compatible-mode/v1", "   ", "qwen3-vl-plus"),
        ("https://dashscope.example/compatible-mode/v1", "qwen-secret", "   "),
    ],
)
def test_missing_qwen_configuration_fails_only_on_first_real_clip_call(
    tmp_path: Path,
    base_url: str | None,
    api_key: str | None,
    model_id: str | None,
) -> None:
    network_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("缺配置不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert network_calls == 0
    assert "qwen-secret" not in str(raised.value)


def test_clip_path_rejects_symlink_component_before_network(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    request = _segment_request(real_directory)
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    linked_clip = request.clip.model_copy(
        update={"path": linked_directory / request.clip.path.name},
    )
    linked_request = request.model_copy(update={"clip": linked_clip})
    network_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("符号链接 clip 不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(linked_request)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert network_calls == 0


def test_clip_must_be_regular_file_before_network(tmp_path: Path) -> None:
    request = _segment_request(tmp_path)
    directory = tmp_path / "clip-directory"
    directory.mkdir()
    directory_clip = request.clip.model_copy(update={"path": directory})
    directory_request = request.model_copy(update={"clip": directory_clip})
    network_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("目录不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(directory_request)

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert network_calls == 0


def test_clip_streaming_limit_stops_growth_after_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _segment_request(tmp_path)
    original_stat = Path.stat
    original_open = Path.open
    read_sizes: list[int] = []

    def fake_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        actual = original_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == request.clip.path:
            values = list(actual)
            values[6] = 1
            return os.stat_result(values)
        return actual

    class BoundedReader:
        def __init__(self) -> None:
            self._stream = original_open(request.clip.path, "rb")

        def __enter__(self) -> BoundedReader:
            return self

        def __exit__(self, *_args: object) -> None:
            self._stream.close()

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            assert size > 0, "Qwen clip 禁止无界读取"
            return self._stream.read(size)

    def fake_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == request.clip.path:
            return BoundedReader()
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(Path, "open", fake_open)
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: (_ for _ in ()).throw(
                    AssertionError("超限 clip 不得外发"),
                ),
            ),
        ),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_video_bytes=5,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert read_sizes and all(size == 1024 * 1024 for size in read_sizes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"max_attempts": 0},
        {"retry_backoff_seconds": float("nan")},
        {"retry_backoff_seconds": float("inf")},
        {"retry_backoff_seconds": -1.0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("-inf")},
    ],
)
def test_qwen_constructor_rejects_non_finite_or_invalid_retry_values(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        QwenVideoClient(
            httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(500))),
            base_url="https://dashscope.example/compatible-mode/v1",
            api_key="qwen-secret",
            model_id="qwen3-vl-plus",
            allowed_video_root=tmp_path,
            sleeper=lambda _delay: None,
            **overrides,  # type: ignore[arg-type]
        )


def test_timeout_traceback_does_not_retain_request_or_sensitive_cause(
    tmp_path: Path,
) -> None:
    secret = "qwen-secret-trace"
    supplier_body = "supplier-private-body"
    request = _segment_request(tmp_path)
    absolute_path = str(request.clip.path)

    def handler(http_request: httpx.Request) -> httpx.Response:
        if _is_probe_request(http_request):
            return _provider_response({"supported": True})
        raise httpx.ReadTimeout(
            f"{secret} {supplier_body} {absolute_path} data:video/mp4;base64,LEAK",
            request=http_request,
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key=secret,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_attempts=1,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert raised.value.__cause__ is None
    assert secret not in rendered
    assert supplier_body not in rendered
    assert absolute_path not in rendered
    assert "data:video/mp4;base64" not in rendered


@pytest.mark.parametrize(
    "transport_error",
    [httpx.RemoteProtocolError, httpx.ProxyError, httpx.UnsupportedProtocol],
)
def test_all_httpx_transport_errors_are_retried_without_sensitive_traceback(
    tmp_path: Path,
    transport_error: type[httpx.TransportError],
) -> None:
    secret = "qwen-secret-transport"
    supplier_body = "supplier-private-transport-body"
    request = _segment_request(tmp_path)
    absolute_path = str(request.clip.path)

    def handler(http_request: httpx.Request) -> httpx.Response:
        if _is_probe_request(http_request):
            return _provider_response({"supported": True})
        raise transport_error(
            f"{secret} {supplier_body} {absolute_path} data:video/mp4;base64,LEAK",
            request=http_request,
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key=secret,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_attempts=1,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert raised.value.__cause__ is None
    assert secret not in rendered
    assert supplier_body not in rendered
    assert absolute_path not in rendered
    assert "data:video/mp4;base64" not in rendered


@pytest.mark.parametrize(
    "response_kind",
    ["malformed_json", "extra_field", "unknown_ref"],
)
def test_invalid_response_traceback_discards_supplier_content_and_validation_input(
    tmp_path: Path,
    response_kind: str,
) -> None:
    supplier_body = "supplier-private-validation-body"
    request = _segment_request(tmp_path)

    def handler(http_request: httpx.Request) -> httpx.Response:
        if _is_probe_request(http_request):
            return _provider_response({"supported": True})
        if response_kind == "malformed_json":
            return _provider_response(f"{{broken-{supplier_body}")
        if response_kind == "extra_field":
            return _provider_response(
                {**_valid_segment(), "supplier_private": supplier_body},
            )
        return _provider_response(_valid_segment((supplier_body,)))

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert raised.value.__cause__ is None
    assert supplier_body not in rendered
    assert str(request.clip.path) not in rendered
    assert "data:video/mp4;base64" not in rendered


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("size", ErrorCode.QWEN_CAPABILITY_UNAVAILABLE),
        ("duration", ErrorCode.QWEN_CAPABILITY_UNAVAILABLE),
        ("digest", ErrorCode.VIDEO_DIGEST_MISMATCH),
    ],
)
def test_clip_size_duration_and_digest_fail_before_first_network_call(
    tmp_path: Path,
    mutation: str,
    expected_code: ErrorCode,
) -> None:
    request = _segment_request(tmp_path)
    max_video_bytes = 5 if mutation == "size" else 64 * 1024 * 1024
    if mutation == "duration":
        clip = request.clip.model_copy(update={"end_ms": 30_001})
        request = request.model_copy(update={"clip": clip})
    elif mutation == "digest":
        clip = request.clip.model_copy(update={"sha256": "0" * 64})
        request = request.model_copy(update={"clip": clip})
    network_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("非法 clip 不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_video_bytes=max_video_bytes,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    assert raised.value.code == expected_code
    assert network_calls == 0


def test_cached_explicit_probe_still_validates_new_clip_before_returning_capabilities(
    tmp_path: Path,
) -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return _provider_response({"supported": True})

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    clip = _segment_request(tmp_path).clip
    client.probe_capabilities(clip)
    invalid_clip = clip.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(VideoDemoError) as raised:
        client.probe_capabilities(invalid_clip)

    assert raised.value.code == ErrorCode.VIDEO_DIGEST_MISMATCH
    assert requests == 1


def test_probe_verifies_configured_model_video_and_json_schema(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    clip = _segment_request(tmp_path).clip

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return _provider_response({"supported": True})

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    capabilities = client.probe_capabilities(clip)

    assert capabilities.model_id == "qwen3-vl-plus"
    assert capabilities.video_input == "data_url"
    assert capabilities.json_schema is True
    assert capabilities.protocol == "chat_completions"
    assert capabilities.max_video_bytes == 64 * 1024 * 1024
    assert capabilities.max_video_duration_ms == 30_000
    assert capabilities.timeout_seconds == 300.0
    assert requests[0]["temperature"] == 0
    assert requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "qwen_capability_probe",
            "strict": True,
            "schema": {
                "additionalProperties": False,
                "properties": {
                    "supported": {"const": True, "title": "Supported", "type": "boolean"},
                },
                "required": ["supported"],
                "title": "CapabilityProbeResponse",
                "type": "object",
            },
        },
    }


def test_probe_rejects_unexpected_model_identity(tmp_path: Path) -> None:
    clip = _segment_request(tmp_path).clip
    client = QwenVideoClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: _provider_response(
                    {"supported": True},
                    model="different-model",
                ),
            ),
        ),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.probe_capabilities(clip)

    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE


def test_failed_capability_probe_is_not_cached_and_next_segment_retries(
    tmp_path: Path,
) -> None:
    probe_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal probe_calls
        if _is_probe_request(request):
            probe_calls += 1
            return _provider_response(
                {"supported": True},
                model="different-model" if probe_calls == 1 else "qwen3-vl-plus",
            )
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    request = _segment_request(tmp_path)

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    assert raised.value.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE
    assert client.understand_segment(request).title == "问候"
    assert probe_calls == 2


def test_segment_call_uses_video_data_url_temperature_zero_and_strict_schema(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        assert request.headers["authorization"] == "Bearer qwen-secret"
        payloads.append(json.loads(request.content))
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1/",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert isinstance(result, SegmentUnderstanding)
    assert result.evidence_refs == ("asr_001",)
    payload = payloads[0]
    assert payload["model"] == "qwen3-vl-plus"
    assert payload["temperature"] == 0
    response_format = payload["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True  # type: ignore[index]
    messages = payload["messages"]
    assert isinstance(messages, list)
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "video_url"
    assert user_content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert str(user_content[0]["video_url"]["url"]).endswith("dmlkZW8tYnl0ZXM=")


def test_segment_call_for_remote_clip_forwards_allowlisted_url_without_reading_video(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []
    remote_url = (
        "https://test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com/"
        "%E6%B5%8B%E8%AF%95demo2.mp4"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        return _provider_response(_valid_segment())

    request = _segment_request(tmp_path).model_copy(
        update={
            "clip": _segment_request(tmp_path).clip.model_copy(
                update={"path": None, "sha256": None, "source_url": remote_url},
            ),
        },
    )
    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(request)

    assert result.title == "问候"
    assert len(payloads) == 2
    for payload in payloads:
        messages = payload["messages"]
        assert isinstance(messages, list)
        content = messages[0]["content"] if _is_probe_request(  # type: ignore[index]
            httpx.Request("POST", "https://provider.invalid", content=json.dumps(payload)),
        ) else messages[1]["content"]
        assert content[0]["video_url"]["url"] == remote_url  # type: ignore[index]


@pytest.mark.parametrize(
    "remote_url",
    [
        "http://test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com/video.mp4",
        "https://localhost/video.mp4",
        "https://127.0.0.1/video.mp4",
        "https://example.com/video.mp4",
    ],
)
def test_remote_clip_url_is_rejected_before_provider_request(
    tmp_path: Path,
    remote_url: str,
) -> None:
    request = _segment_request(tmp_path).model_copy(
        update={
            "clip": _segment_request(tmp_path).clip.model_copy(
                update={"path": None, "sha256": None, "source_url": remote_url},
            ),
        },
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("非法公网视频不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert calls == 0


def test_remote_clip_rejects_allowlisted_host_that_resolves_to_private_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = (
        "https://test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com/video.mp4"
    )
    request = _segment_request(tmp_path).model_copy(
        update={
            "clip": _segment_request(tmp_path).clip.model_copy(
                update={"path": None, "sha256": None, "source_url": remote_url},
            ),
        },
    )
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("10.0.0.8", 443))],
    )
    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200))),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.probe_capabilities(request.clip)

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID


def test_segment_call_rejects_video_outside_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    request = _segment_request(tmp_path)
    called = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called += 1
        if _is_probe_request(http_request):
            return _provider_response({"supported": True})
        raise AssertionError("越界视频不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=allowed_root,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(request)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert called == 0


def test_unknown_evidence_reference_gets_one_schema_repair(tmp_path: Path) -> None:
    responses = iter(
        [
            _provider_response(_valid_segment(("fabricated_001",))),
            _provider_response(_valid_segment()),
        ],
    )
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        payloads.append(json.loads(request.content))
        return next(responses)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert result.evidence_refs == ("asr_001",)
    assert len(payloads) == 2
    assert "UNKNOWN_EVIDENCE_REFERENCE" in json.dumps(payloads[1], ensure_ascii=False)


def test_segment_understanding_falls_back_to_plain_json_after_strict_schema_failure(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []
    strict_failures = iter(
        (
            _provider_response(
                "视觉理解结果不是合法结构化 JSON",
                model="qwen3-vl-flash",
            ),
            _provider_response(
                "仍然不是合法结构化 JSON",
                model="qwen3-vl-flash",
            ),
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if _is_probe_request(request):
            return _provider_response({"supported": True}, model="qwen3-vl-flash")
        if "response_format" in payload:
            return next(strict_failures)
        return _provider_response(_valid_segment(), model="qwen3-vl-flash")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert result.title == "问候"
    assert len(payloads) == 4
    assert "response_format" in payloads[1]
    assert "response_format" in payloads[2]
    assert "response_format" not in payloads[3]
    fallback_messages = payloads[3]["messages"]
    assert isinstance(fallback_messages, list)
    assert "只输出一个合法 JSON 对象" in json.dumps(
        fallback_messages,
        ensure_ascii=False,
    )


def test_plain_json_fallback_is_limited_to_verified_qwen_flash_model(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if _is_probe_request(request):
            return _provider_response({"supported": True}, model="unverified-vl-flash")
        if "response_format" in payload:
            return _provider_response("不是合法 JSON", model="unverified-vl-flash")
        return _provider_response(_valid_segment(), model="unverified-vl-flash")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="unverified-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert len(payloads) == 3
    assert all("response_format" in payload for payload in payloads)


def test_plain_json_fallback_rejects_malformed_json(tmp_path: Path) -> None:
    non_probe_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal non_probe_attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True}, model="qwen3-vl-flash")
        non_probe_attempts += 1
        return _provider_response("{broken", model="qwen3-vl-flash")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert non_probe_attempts == 3


def test_plain_json_fallback_rejects_unknown_evidence_reference(tmp_path: Path) -> None:
    non_probe_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal non_probe_attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True}, model="qwen3-vl-flash")
        non_probe_attempts += 1
        if non_probe_attempts < 3:
            return _provider_response("不是合法 JSON", model="qwen3-vl-flash")
        return _provider_response(
            _valid_segment(("fabricated_001",)),
            model="qwen3-vl-flash",
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert non_probe_attempts == 3


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_attempts"),
    [
        ("authentication", ErrorCode.QWEN_AUTHENTICATION_FAILED, 1),
        ("network", ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 2),
    ],
)
def test_plain_json_fallback_does_not_mask_transport_or_authentication_failure(
    tmp_path: Path,
    failure: str,
    expected_code: ErrorCode,
    expected_attempts: int,
) -> None:
    non_probe_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal non_probe_attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True}, model="qwen3-vl-flash")
        non_probe_attempts += 1
        if failure == "authentication":
            return httpx.Response(401)
        raise httpx.ReadTimeout("timeout", request=request)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        max_attempts=2,
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == expected_code
    assert non_probe_attempts == expected_attempts


def test_extra_output_field_is_rejected_after_only_one_repair(tmp_path: Path) -> None:
    invalid = {**_valid_segment(), "start_ms": 0}
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        attempts += 1
        return _provider_response(invalid)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert attempts == 2


def test_transient_network_failures_use_finite_exponential_backoff(tmp_path: Path) -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        if attempts == 2:
            return httpx.Response(429)
        return _provider_response(_valid_segment())

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_attempts=3,
        retry_backoff_seconds=0.25,
        sleeper=delays.append,
    )

    result = client.understand_segment(_segment_request(tmp_path))

    assert result.title == "问候"
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_qwen_5xx_stops_after_finite_attempts_and_is_retryable(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        attempts += 1
        return httpx.Response(503)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_attempts=2,
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )
    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert attempts == 2


def test_qwen_timeout_stops_after_finite_attempts_and_is_retryable(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        max_attempts=2,
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )
    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert attempts == 2


def test_qwen_malformed_json_fails_closed_after_one_schema_repair(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        attempts += 1
        return httpx.Response(
            200,
            json={
                "model": "qwen3-vl-plus",
                "choices": [{"message": {"content": "{broken"}}],
            },
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    assert raised.value.code == ErrorCode.QWEN_RESPONSE_INVALID
    assert attempts == 2


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_error_does_not_leak_secret(
    tmp_path: Path,
    status_code: int,
) -> None:
    secret = "qwen-secret"
    supplier_body = "supplier-private-auth-body"

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        return httpx.Response(status_code, text=f"{secret} {supplier_body}")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key=secret,
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.understand_segment(_segment_request(tmp_path))

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.code == ErrorCode.QWEN_AUTHENTICATION_FAILED
    assert raised.value.__cause__ is None
    assert secret not in rendered
    assert supplier_body not in rendered


def test_video_summary_call_uses_structured_segment_inputs_without_time_fields(
    tmp_path: Path,
) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_probe_request(request):
            return _provider_response({"supported": True})
        payload = json.loads(request.content)
        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "segment_understanding":
            return _provider_response(_valid_segment())
        payloads.append(payload)
        return _provider_response(
            {
                "title": "摘要",
                "summary_zh": "视频包含一段问候。",
                "speakers": ["SPEAKER_01"],
                "languages": ["en"],
                "topics": ["问候"],
                "entities": [],
                "actions": ["问好"],
                "keywords": ["问候"],
                "original_keywords": ["Hello"],
            },
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-plus",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )
    request = SummaryUnderstandingRequest(
        segments=(
            SegmentSummaryInput(
                segment_ref="timeline_001",
                understanding=SegmentUnderstanding.model_validate(_valid_segment()),
            ),
        ),
    )

    client.understand_segment(_segment_request(tmp_path))
    result = client.summarize_video(request)

    assert isinstance(result, SummaryUnderstanding)
    assert result.title == "摘要"
    serialized = json.dumps(payloads[0], ensure_ascii=False)
    assert "timeline_001" in serialized
    assert "start_ms" not in serialized
    assert "end_ms" not in serialized


def test_remote_video_summary_uses_plain_text_protocol_and_forwards_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = (
        "https://test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com/"
        "%E6%B5%8B%E8%AF%95demo2.mp4"
    )
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            "视频主要讲解如何使用 AI 工具批量生产小红书内容。",
            model="qwen3-vl-flash",
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    summary = client.summarize_video(
        VideoClipInput(
            clip_id="remote_full_video",
            start_ms=0,
            end_ms=500,
            source_url=remote_url,
            mime_type="video/mp4",
        ),
        instruction="请概括完整视频的主要内容，并列出三个关键点。",
        max_tokens=500,
    )

    assert summary.startswith("视频主要讲解")
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["model"] == "qwen3-vl-flash"
    assert payload["max_tokens"] == 500
    assert "response_format" not in payload
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["type"] == "video_url"  # type: ignore[index]
    assert content[0]["video_url"]["url"] == remote_url  # type: ignore[index]


def test_remote_video_default_task_requests_full_visual_understanding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_url = (
        "https://test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com/"
        "%E6%B5%8B%E8%AF%95demo2.mp4"
    )
    payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        qwen_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response("完整视频视觉理解结果", model="qwen3-vl-flash")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    client.summarize_video(
        VideoClipInput(
            clip_id="remote_full_visual_video",
            start_ms=0,
            end_ms=921_484,
            source_url=remote_url,
            mime_type="video/mp4",
        ),
    )

    messages = payloads[0]["messages"]
    assert isinstance(messages, list)
    instruction = messages[0]["content"][1]["text"]  # type: ignore[index]
    assert "视觉理解" in instruction
    assert "整体摘要" in instruction
    assert "关键事件" in instruction
    assert "画面文字" in instruction
    assert "视觉事实" in instruction
    assert "visual(画面可见内容)" in instruction
    assert "speech(语音提到内容)" in instruction
    assert "按以下五段输出" in instruction
    assert "每项保持精简" in instruction
    assert "恰好 10" in instruction
    assert "恰好 12" in instruction
    assert "不要逐条抄录贯穿全片的口播字幕" in instruction
    assert "恰好 6" in instruction
    assert "恰好 5" in instruction
    assert payloads[0]["max_tokens"] == 3_000


def test_remote_video_summary_rejects_invalid_url_before_network(tmp_path: Path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("非法公网 URL 不得外发")

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.summarize_video(
            VideoClipInput(
                clip_id="invalid_remote_video",
                start_ms=0,
                end_ms=500,
                source_url="https://localhost/video.mp4",
                mime_type="video/mp4",
            ),
            instruction="请总结完整视频。",
        )

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert calls == 0


def test_local_video_summary_uses_data_url_without_strict_schema(
    tmp_path: Path,
) -> None:
    video = b"local-full-video"
    video_path = tmp_path / "full.mp4"
    video_path.write_bytes(video)
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return _provider_response(
            "完整视频总结：视频展示了一个产品演示过程。",
            model="qwen3-vl-flash",
        )

    client = QwenVideoClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        base_url="https://ai-proxy.example/v1",
        api_key="qwen-secret",
        model_id="qwen3-vl-flash",
        allowed_video_root=tmp_path,
        max_video_bytes=1024,
        sleeper=lambda _delay: None,
    )

    summary = client.summarize_video(
        VideoClipInput(
            clip_id="local_full_video",
            start_ms=0,
            end_ms=90_000,
            path=video_path,
            mime_type="video/mp4",
            sha256=hashlib.sha256(video).hexdigest(),
        ),
        instruction="请总结完整视频。",
    )

    assert summary.startswith("完整视频总结")
    assert len(payloads) == 1
    payload = payloads[0]
    assert "response_format" not in payload
    messages = payload["messages"]
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert content[0]["video_url"]["url"].startswith(  # type: ignore[index]
        "data:video/mp4;base64,"
    )
