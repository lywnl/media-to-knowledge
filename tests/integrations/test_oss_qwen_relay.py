from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.oss import PublishedVideo, PublishedVideoUnderstanding
from video_demo.integrations.video_port import (
    SegmentSummaryInput,
    SegmentUnderstandingRequest,
    SummaryUnderstandingRequest,
    VideoClipInput,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowInput,
    WholeVideoWindowUnderstanding,
)


def _request(tmp_path: Path) -> SegmentUnderstandingRequest:
    payload = b"local-window-video"
    path = tmp_path / "clip.mp4"
    path.write_bytes(payload)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="上传图片到 ChatGPT",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    return SegmentUnderstandingRequest(
        clip=VideoClipInput(
            clip_id="clip_001",
            start_ms=1_000,
            end_ms=6_000,
            path=path,
            mime_type="video/mp4",
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        window=TimeRange(start_ms=1_000, end_ms=6_000),
        timeline=build_timeline((speech,)),
        evidence=(speech,),
    )


def _understanding() -> SegmentUnderstanding:
    return SegmentUnderstanding(
        title="ChatGPT 图片处理",
        summary_zh="画面显示 ChatGPT 界面，语音提到上传图片。",
        languages=("zh",),
        topics=("AI 工具",),
        entities=("ChatGPT",),
        actions=("上传图片",),
        keywords=("ChatGPT",),
        original_keywords=("ChatGPT",),
        evidence_refs=("asr_001",),
    )


def test_relay_publishes_local_clip_and_delegates_same_window_and_digest(
    tmp_path: Path,
) -> None:
    original = _request(tmp_path)
    published_clips: list[VideoClipInput] = []
    discarded_keys: list[str] = []
    delegated_requests: list[SegmentUnderstandingRequest] = []

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, clip: VideoClipInput) -> PublishedVideo:
            published_clips.append(clip)
            return PublishedVideo(
                published_clip=VideoClipInput(
                    clip_id=clip.clip_id,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                    source_url="https://private-video-bucket.oss-cn-hangzhou.aliyuncs.com/clip.mp4?signature=redacted",
                    mime_type=clip.mime_type,
                    sha256=clip.sha256,
                ),
                object_key="video-demo/qwen-clips/owner/publish-clip.mp4",
            )

        def delete(self, object_key: str) -> None:
            discarded_keys.append(object_key)

    class Delegate:
        degraded_warnings = ("DELEGATE_WARNING",)

        def understand_segment(
            self,
            request: SegmentUnderstandingRequest,
        ) -> SegmentUnderstanding:
            delegated_requests.append(request)
            return _understanding()

        def summarize_video(
            self,
            _request: SummaryUnderstandingRequest,
        ) -> SummaryUnderstanding:
            return SummaryUnderstanding(title="摘要", summary_zh="测试摘要。")

    relay = PublishedVideoUnderstanding(Delegate(), Publisher())

    result = relay.understand_segment(original)

    assert result == _understanding()
    assert published_clips == [original.clip]
    assert discarded_keys == ["video-demo/qwen-clips/owner/publish-clip.mp4"]
    assert len(delegated_requests) == 1
    delegated = delegated_requests[0]
    assert delegated.clip.path is None
    assert delegated.clip.source_url is not None
    assert delegated.clip.sha256 == original.clip.sha256
    assert delegated.clip.clip_id == original.clip.clip_id
    assert delegated.window == original.window
    assert delegated.timeline == original.timeline
    assert delegated.evidence == original.evidence
    assert relay.degraded_warnings == ("DELEGATE_WARNING",)
    assert "signature=redacted" not in result.model_dump_json()


def test_relay_summary_does_not_publish_another_video(tmp_path: Path) -> None:
    segment = _understanding()
    summary_request = SummaryUnderstandingRequest(
        segments=(SegmentSummaryInput(segment_ref="clip_001", understanding=segment),),
    )
    published = 0
    summarized: list[SummaryUnderstandingRequest] = []

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, _clip: VideoClipInput) -> PublishedVideo:
            nonlocal published
            published += 1
            raise AssertionError("摘要阶段不应再次发布视频")

        def delete(self, _object_key: str) -> None:
            raise AssertionError("摘要阶段不应删除视频")

    class Delegate:
        def understand_segment(
            self,
            _request: SegmentUnderstandingRequest,
        ) -> SegmentUnderstanding:
            raise AssertionError("测试不调用片段理解")

        def summarize_video(
            self,
            request: SummaryUnderstandingRequest,
        ) -> SummaryUnderstanding:
            summarized.append(request)
            return SummaryUnderstanding(title="摘要", summary_zh="视频摘要。")

    result = PublishedVideoUnderstanding(Delegate(), Publisher()).summarize_video(
        summary_request,
    )

    assert result.title == "摘要"
    assert summarized == [summary_request]
    assert published == 0


def test_relay_publishes_full_video_exactly_once(tmp_path: Path) -> None:
    segment = _request(tmp_path)
    whole_request = WholeVideoUnderstandingRequest(
        video=segment.clip.model_copy(update={"start_ms": 0, "end_ms": 6_000}),
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=0,
                end_ms=6_000,
                timeline=segment.timeline,
                evidence=segment.evidence,
            ),
        ),
    )
    published: list[VideoClipInput] = []
    discarded_keys: list[str] = []
    delegated: list[WholeVideoUnderstandingRequest] = []

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, video: VideoClipInput) -> PublishedVideo:
            published.append(video)
            return PublishedVideo(
                published_clip=video.model_copy(
                    update={
                        "path": None,
                        "source_url": (
                            "https://private-video-bucket.oss-cn-hangzhou.aliyuncs.com/"
                            "full.mp4?signature=redacted"
                        ),
                    },
                ),
                object_key="video-demo/qwen-clips/owner/publish-full.mp4",
            )

        def delete(self, object_key: str) -> None:
            discarded_keys.append(object_key)

    class Delegate:
        def understand_video(
            self,
            request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            delegated.append(request)
            return WholeVideoUnderstanding(
                windows=(
                    WholeVideoWindowUnderstanding(
                        window_id="window_001",
                        understanding=_understanding(),
                    ),
                ),
                summary=SummaryUnderstanding(title="摘要", summary_zh="全片摘要。"),
            )

    result = PublishedVideoUnderstanding(Delegate(), Publisher()).understand_video(
        whole_request,
    )

    assert published == [whole_request.video]
    assert discarded_keys == ["video-demo/qwen-clips/owner/publish-full.mp4"]
    assert len(delegated) == 1
    assert delegated[0].video.path is None
    assert delegated[0].video.sha256 == whole_request.video.sha256
    assert delegated[0].windows == whole_request.windows
    assert "signature=redacted" not in result.model_dump_json()


def test_relay_discards_after_qwen_failure_without_replacing_original_error(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    discarded_keys: list[str] = []
    original_error = VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 返回非法")

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, clip: VideoClipInput) -> PublishedVideo:
            return PublishedVideo(published_clip=clip, object_key="owner/publish.mp4")

        def delete(self, object_key: str) -> None:
            discarded_keys.append(object_key)

    class Delegate:
        def understand_segment(
            self,
            _request: SegmentUnderstandingRequest,
        ) -> SegmentUnderstanding:
            raise original_error

        def summarize_video(
            self,
            _request: SummaryUnderstandingRequest,
        ) -> SummaryUnderstanding:
            raise AssertionError("测试不调用摘要")

    with pytest.raises(VideoDemoError) as raised:
        PublishedVideoUnderstanding(Delegate(), Publisher()).understand_segment(request)

    assert raised.value is original_error
    assert discarded_keys == ["owner/publish.mp4"]


def test_relay_discards_best_effort_when_cleanup_fails(tmp_path: Path) -> None:
    request = _request(tmp_path)
    original_error = VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 返回非法")

    class Publisher:
        @property
        def remote_host(self) -> str:
            return "private-video-bucket.oss-cn-hangzhou.aliyuncs.com"

        def publish(self, clip: VideoClipInput) -> PublishedVideo:
            return PublishedVideo(published_clip=clip, object_key="owner/publish.mp4")

        def delete(self, _object_key: str) -> None:
            raise RuntimeError("OSS 删除失败")

    class Delegate:
        def understand_segment(
            self,
            _request: SegmentUnderstandingRequest,
        ) -> SegmentUnderstanding:
            raise original_error

        def summarize_video(
            self,
            _request: SummaryUnderstandingRequest,
        ) -> SummaryUnderstanding:
            raise AssertionError("测试不调用摘要")

    with pytest.raises(VideoDemoError) as raised:
        PublishedVideoUnderstanding(Delegate(), Publisher()).understand_segment(request)

    assert raised.value is original_error
