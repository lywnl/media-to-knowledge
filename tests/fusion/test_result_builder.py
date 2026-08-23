from __future__ import annotations

import hashlib

import pytest

from video_demo.domain.evidence import BoundingBox, OcrEvidence, OcrLine, SpeechSegment
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
        "文档类型：VIDEO_SUMMARY",
        "视频标题：测试视频",
        "视频时长：500",
        "",
        "整体摘要：视频包含一段问候。",
        "核心主题：问候",
        "核心实体：无",
        "核心动作：问好",
        "核心关键词：问候",
        "原语言关键词：Hello",
        "章节概览：问候 [0, 500)",
    ]
    assert "segment_id" not in summary.retrieval_text
    assert render_summary_retrieval_text(summary) == summary.retrieval_text
    assert summary.retrieval_hash == hashlib.sha256(
        summary.retrieval_text.encode("utf-8"),
    ).hexdigest()


def test_retrieval_ready_fixture_keeps_key_evidence_without_internal_identifiers() -> None:
    """检索文本保留回答所需事实，同时隔离运行与供应商内部标识。"""
    speech = SpeechSegment(
        evidence_id="asr_technical_001",
        start_ms=0,
        end_ms=1_000,
        text="Milvus 使用 HNSW 索引完成向量近邻检索。",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    ocr = OcrEvidence(
        evidence_id="ocr_screen_001",
        start_ms=0,
        end_ms=1_000,
        keyframe_id="keyframe_internal_001",
        timestamp_ms=500,
        language="zh",
        lines=(
            OcrLine(
                text="Milvus 2.4 / 1536 维",
                bounding_box=BoundingBox(x=1, y=2, width=100, height=20),
                confidence=0.99,
            ),
        ),
        provider_request_id="provider-request-internal-001",
    )
    local_summary = "演示 HNSW 参数与向量维度配置。"
    global_summary = "视频讲解 Milvus 向量检索的索引配置与查询流程。"
    segments = merge_segment_understandings(
        (
            WindowUnderstanding(
                window_id="window_technical_001",
                start_ms=0,
                end_ms=1_000,
                understanding=SegmentUnderstanding(
                    title="Milvus 索引配置",
                    summary_zh=local_summary,
                    languages=("zh",),
                    topics=("向量检索",),
                    entities=("Milvus",),
                    actions=("配置 HNSW",),
                    keywords=("HNSW", "1536 维"),
                    original_keywords=("Milvus",),
                    visual_facts=("画面展示 Milvus 控制台的索引参数面板。",),
                    evidence_refs=(speech.evidence_id, ocr.evidence_id),
                ),
            ),
        ),
        boundaries=(
            BoundaryPoint(timestamp_ms=0, sources=("video_start",)),
            BoundaryPoint(timestamp_ms=1_000, sources=("video_end",)),
        ),
        evidence=(speech, ocr),
        video_title="Milvus 检索课程",
    )
    summary = build_video_summary(
        SummaryUnderstanding(
            title="Milvus 检索课程",
            summary_zh=global_summary,
            topics=("向量检索",),
            entities=("Milvus",),
            actions=("配置 HNSW",),
            keywords=("HNSW",),
            original_keywords=("Milvus",),
        ),
        duration_ms=1_000,
        segments=segments,
        chapters=(
            SummaryChapter(
                title="索引配置",
                start_ms=0,
                end_ms=1_000,
                segment_ids=(segments[0].segment_id,),
            ),
        ),
    )

    segment_text = segments[0].retrieval_text
    assert "Milvus 使用 HNSW 索引完成向量近邻检索。" in segment_text
    assert "OCR文字：Milvus 2.4 / 1536 维" in segment_text
    assert "画面事实：画面展示 Milvus 控制台的索引参数面板。" in segment_text
    assert local_summary in segment_text
    assert global_summary not in segment_text
    assert global_summary in summary.retrieval_text
    for internal_identifier in (
        "run_id",
        "segment_id",
        "evidence_refs",
        "chapter_id",
        "provider-request-internal-001",
        "asr_technical_001",
        "ocr_screen_001",
    ):
        assert internal_identifier not in segment_text
        assert internal_identifier not in summary.retrieval_text
