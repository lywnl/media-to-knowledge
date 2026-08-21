from __future__ import annotations

import hashlib

from video_demo.domain.result import SegmentUnderstanding
from video_demo.fusion.merge import BoundaryPoint, WindowUnderstanding, merge_segment_understandings
from video_demo.fusion.retrieval_text import render_segment_retrieval_text


def test_segment_retrieval_text_has_fixed_chinese_field_order_and_stable_hash() -> None:
    understanding = SegmentUnderstanding(
        title="  产品   演示 ",
        summary_zh="展示   视频理解能力。\n生成检索文本。",
        speakers=("SPEAKER_01",),
        languages=("zh", "en"),
        topics=("视频检索",),
        entities=("Qwen",),
        actions=("演示",),
        keywords=("视频理解",),
        original_keywords=("retrieval_text",),
        evidence_refs=("asr_001",),
    )
    segments = merge_segment_understandings(
        (
            WindowUnderstanding(
                window_id="window_001",
                start_ms=0,
                end_ms=1_000,
                understanding=understanding,
            ),
        ),
        boundaries=(
            BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
            BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
        ),
    )
    segment = segments[0]

    assert segment.retrieval_text.splitlines() == [
        "标题：产品 演示",
        "时间范围(毫秒)：[0, 1000)",
        "中文摘要：展示 视频理解能力。 生成检索文本。",
        "说话人：SPEAKER_01",
        "语言：zh、en",
        "主题：视频检索",
        "实体：Qwen",
        "动作：演示",
        "关键词：视频理解",
        "原语言关键词：retrieval_text",
    ]
    assert render_segment_retrieval_text(segment) == segment.retrieval_text
    assert segment.retrieval_hash == hashlib.sha256(
        segment.retrieval_text.encode("utf-8"),
    ).hexdigest()


def test_same_semantics_produce_identical_segment_serialization() -> None:
    understanding = SegmentUnderstanding(
        title="演示",
        summary_zh="生成检索文本。",
        languages=("zh",),
        topics=("视频检索",),
        keywords=("视频理解",),
        original_keywords=("retrieval",),
        evidence_refs=("asr_001",),
    )
    boundaries = (
        BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
        BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
    )
    window = WindowUnderstanding(
        window_id="window_001",
        start_ms=0,
        end_ms=1_000,
        understanding=understanding,
    )

    left = merge_segment_understandings((window,), boundaries=boundaries)[0]
    right = merge_segment_understandings((window,), boundaries=tuple(reversed(boundaries)))[0]

    assert left.model_dump_json() == right.model_dump_json()
