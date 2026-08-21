from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import TimeRange
from video_demo.fusion.timeline import build_timeline
from video_demo.integrations.video_port import (
    SegmentSummaryInput,
    SegmentUnderstandingRequest,
    SummaryUnderstandingRequest,
    VideoClipInput,
    VideoUnderstandingPort,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingPort,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowInput,
    WholeVideoWindowUnderstanding,
)


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=100,
        end_ms=400,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _segment_understanding() -> SegmentUnderstanding:
    return SegmentUnderstanding(
        title="问候",
        summary_zh="讲者向观众问好。",
        speakers=("SPEAKER_01",),
        languages=("en",),
        topics=("问候",),
        actions=("问好",),
        keywords=("问候",),
        original_keywords=("Hello",),
        evidence_refs=("asr_001",),
    )


def test_segment_request_accepts_video_clip_and_freezes_deduplicated_evidence(
    tmp_path: Path,
) -> None:
    speech = _speech()
    request = SegmentUnderstandingRequest(
        clip=VideoClipInput(
            clip_id="clip_001",
            start_ms=0,
            end_ms=500,
            path=tmp_path / "clip.mp4",
            mime_type="video/mp4",
            sha256="a" * 64,
        ),
        window=TimeRange(start_ms=0, end_ms=500),
        timeline=build_timeline((speech,)),
        evidence=(speech, speech),
    )

    assert request.evidence == (speech,)
    with pytest.raises(ValidationError, match="frozen"):
        request.evidence[0].text = "Changed"  # type: ignore[misc]


def test_video_port_contract_rejects_image_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        VideoClipInput(
            clip_id="clip_001",
            start_ms=0,
            end_ms=500,
            path=tmp_path / "frame.jpg",
            mime_type="image/jpeg",
            sha256="a" * 64,
        )


def test_video_port_accepts_remote_video_url_without_local_materialization() -> None:
    clip = VideoClipInput(
        clip_id="clip_remote_001",
        start_ms=0,
        end_ms=500,
        source_url="https://media.example.test/video.mp4",
        mime_type="video/mp4",
    )

    assert clip.path is None
    assert clip.source_url == "https://media.example.test/video.mp4"
    assert clip.sha256 is None


def test_video_port_accepts_remote_video_url_bound_to_local_digest() -> None:
    clip = VideoClipInput(
        clip_id="clip_remote_001",
        start_ms=0,
        end_ms=500,
        source_url="https://media.example.test/video.mp4",
        mime_type="video/mp4",
        sha256="a" * 64,
    )

    assert clip.path is None
    assert clip.sha256 == "a" * 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_url": None},
        {"path": Path("clip.mp4"), "source_url": "https://media.example.test/video.mp4"},
        {"path": Path("clip.mp4"), "sha256": None},
    ],
)
def test_video_port_requires_exactly_one_complete_video_source(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "clip_id": "clip_remote_001",
        "start_ms": 0,
        "end_ms": 500,
        "mime_type": "video/mp4",
        "source_url": "https://media.example.test/video.mp4",
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        VideoClipInput(**values)  # type: ignore[arg-type]


def test_segment_request_rejects_timeline_outside_window(tmp_path: Path) -> None:
    speech = _speech()

    with pytest.raises(ValidationError, match="时间轴条目必须位于理解窗口内"):
        SegmentUnderstandingRequest(
            clip=VideoClipInput(
                clip_id="clip_001",
                start_ms=0,
                end_ms=300,
                path=tmp_path / "clip.mp4",
                mime_type="video/mp4",
                sha256="a" * 64,
            ),
            window=TimeRange(start_ms=0, end_ms=300),
            timeline=build_timeline((speech,)),
            evidence=(speech,),
        )


def test_segment_request_serialization_is_stable_across_evidence_order(
    tmp_path: Path,
) -> None:
    first = _speech()
    second = first.model_copy(
        update={"evidence_id": "asr_002", "start_ms": 400, "end_ms": 500},
    )
    timeline = build_timeline((first, second))
    clip = VideoClipInput(
        clip_id="clip_001",
        start_ms=0,
        end_ms=500,
        path=tmp_path / "clip.mp4",
        mime_type="video/mp4",
        sha256="a" * 64,
    )

    left = SegmentUnderstandingRequest(
        clip=clip,
        window=TimeRange(start_ms=0, end_ms=500),
        timeline=tuple(reversed(timeline)),
        evidence=(second, first),
    )
    right = SegmentUnderstandingRequest(
        clip=clip,
        window=TimeRange(start_ms=0, end_ms=500),
        timeline=timeline,
        evidence=(first, second),
    )

    assert left.model_dump_json() == right.model_dump_json()


def test_qwen_output_contracts_do_not_contain_time_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SummaryUnderstanding(
            title="摘要",
            summary_zh="视频包含一段问候。",
            languages=("en",),
            start_ms=0,
            end_ms=500,
        )


def test_video_understanding_protocol_has_segment_and_summary_operations() -> None:
    class Port:
        def understand_segment(
            self,
            request: SegmentUnderstandingRequest,
        ) -> SegmentUnderstanding:
            return _segment_understanding()

        def summarize_video(
            self,
            request: SummaryUnderstandingRequest,
        ) -> SummaryUnderstanding:
            return SummaryUnderstanding(
                title="摘要",
                summary_zh="视频包含一段问候。",
                languages=("en",),
            )

    summary_request = SummaryUnderstandingRequest(
        segments=(
            SegmentSummaryInput(
                segment_ref="timeline_001",
                understanding=_segment_understanding(),
            ),
        ),
    )

    assert isinstance(Port(), VideoUnderstandingPort)
    assert Port().summarize_video(summary_request).title == "摘要"


def _whole_video_request(tmp_path: Path) -> WholeVideoUnderstandingRequest:
    first = _speech().model_copy(update={"start_ms": 100, "end_ms": 400})
    second = _speech().model_copy(
        update={"evidence_id": "asr_002", "start_ms": 30_100, "end_ms": 30_400},
    )
    return WholeVideoUnderstandingRequest(
        video=VideoClipInput(
            clip_id="video_full_001",
            start_ms=0,
            end_ms=60_000,
            path=tmp_path / "proxy.mp4",
            mime_type="video/mp4",
            sha256="a" * 64,
        ),
        windows=(
            WholeVideoWindowInput(
                window_id="window_001",
                start_ms=0,
                end_ms=30_000,
                timeline=build_timeline((first,)),
                evidence=(first,),
            ),
            WholeVideoWindowInput(
                window_id="window_002",
                start_ms=30_000,
                end_ms=60_000,
                timeline=build_timeline((second,)),
                evidence=(second,),
            ),
        ),
    )


def test_whole_video_request_requires_contiguous_complete_windows(tmp_path: Path) -> None:
    request = _whole_video_request(tmp_path)

    assert request.windows[0].start_ms == request.video.start_ms
    assert request.windows[-1].end_ms == request.video.end_ms

    with pytest.raises(ValidationError, match="连续覆盖完整视频"):
        WholeVideoUnderstandingRequest(
            video=request.video,
            windows=(
                request.windows[0],
                WholeVideoWindowInput(
                    window_id=request.windows[1].window_id,
                    start_ms=30_001,
                    end_ms=request.windows[1].end_ms,
                    timeline=request.windows[1].timeline,
                    evidence=request.windows[1].evidence,
                ),
            ),
        )


def test_whole_video_request_rejects_window_over_thirty_seconds(tmp_path: Path) -> None:
    request = _whole_video_request(tmp_path)

    with pytest.raises(ValidationError, match="30 秒"):
        WholeVideoUnderstandingRequest(
            video=request.video,
            windows=(
                request.windows[0].model_copy(update={"end_ms": 30_001}),
                request.windows[1].model_copy(update={"start_ms": 30_001}),
            ),
        )


def test_whole_video_response_rejects_duplicate_window_ids(tmp_path: Path) -> None:
    request = _whole_video_request(tmp_path)
    item = WholeVideoWindowUnderstanding(
        window_id=request.windows[0].window_id,
        understanding=_segment_understanding(),
    )

    with pytest.raises(ValidationError, match="window_id 不得重复"):
        WholeVideoUnderstanding(
            windows=(item, item),
            summary=SummaryUnderstanding(title="摘要", summary_zh="全片摘要。"),
        )


def test_whole_video_understanding_protocol_has_single_operation(tmp_path: Path) -> None:
    request = _whole_video_request(tmp_path)

    class Port:
        def understand_video(
            self,
            _request: WholeVideoUnderstandingRequest,
        ) -> WholeVideoUnderstanding:
            return WholeVideoUnderstanding(
                windows=tuple(
                    WholeVideoWindowUnderstanding(
                        window_id=window.window_id,
                        understanding=_segment_understanding(),
                    )
                    for window in request.windows
                ),
                summary=SummaryUnderstanding(title="摘要", summary_zh="全片摘要。"),
            )

    assert isinstance(Port(), WholeVideoUnderstandingPort)
    assert len(Port().understand_video(request).windows) == 2
