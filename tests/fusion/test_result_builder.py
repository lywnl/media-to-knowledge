from __future__ import annotations

import hashlib

import pytest

from video_demo.domain.result import SegmentUnderstanding, SummaryChapter, SummaryUnderstanding
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.merge import BoundaryPoint, WindowUnderstanding, merge_segment_understandings
from video_demo.fusion.result_builder import build_video_summary
from video_demo.fusion.retrieval_text import render_summary_retrieval_text


def _segments():
    understanding = SegmentUnderstanding(
        title="问候",
        summary_zh="讲者问好。",
        speakers=("SPEAKER_01",),
        languages=("en",),
        topics=("问候",),
        actions=("问好",),
        keywords=("问候",),
        original_keywords=("Hello",),
        evidence_refs=("asr_001",),
    )
    return merge_segment_understandings(
        (
            WindowUnderstanding(
                window_id="window_001",
                start_ms=0,
                end_ms=500,
                understanding=understanding,
            ),
        ),
        boundaries=(
            BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
            BoundaryPoint(timestamp_ms=500, sources=("video_end",)),
        ),
    )


def _summary_understanding() -> SummaryUnderstanding:
    return SummaryUnderstanding(
        title="测试视频",
        summary_zh="视频包含一段问候。",
        speakers=("SPEAKER_01",),
        languages=("en",),
        topics=("问候",),
        actions=("问好",),
        keywords=("问候",),
        original_keywords=("Hello",),
    )


def test_summary_chapters_can_only_reference_existing_segments() -> None:
    with pytest.raises(VideoDemoError) as raised:
        build_video_summary(
            _summary_understanding(),
            duration_ms=500,
            segments=_segments(),
            chapters=(
                SummaryChapter(
                    title="伪造章节",
                    start_ms=0,
                    end_ms=500,
                    segment_ids=("segment_missing",),
                ),
            ),
        )

    assert raised.value.code == ErrorCode.UNKNOWN_SEGMENT_REFERENCE


def test_summary_retrieval_text_and_hash_are_deterministic() -> None:
    segments = _segments()
    chapter = SummaryChapter(
        title="问候",
        start_ms=0,
        end_ms=500,
        segment_ids=(segments[0].segment_id,),
    )

    summary = build_video_summary(
        _summary_understanding(),
        duration_ms=500,
        segments=segments,
        chapters=(chapter,),
    )

    assert summary.retrieval_text.splitlines() == [
        "标题：测试视频",
        "视频时长(毫秒)：500",
        "中文摘要：视频包含一段问候。",
        f"章节：问候 [0, 500) -> {segments[0].segment_id}",
        "说话人：SPEAKER_01",
        "语言：en",
        "主题：问候",
        "实体：无",
        "动作：问好",
        "关键词：问候",
        "原语言关键词：Hello",
    ]
    assert render_summary_retrieval_text(summary) == summary.retrieval_text
    assert summary.retrieval_hash == hashlib.sha256(
        summary.retrieval_text.encode("utf-8"),
    ).hexdigest()
