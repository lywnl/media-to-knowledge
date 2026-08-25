from __future__ import annotations

import hashlib

from video_demo.domain.document import (
    GroundedClaim,
    ParagraphBlock,
    SemanticChapter,
    VideoDocumentSummary,
    VisualBlock,
)
from video_demo.domain.evidence import VisualObservationEvidence
from video_demo.domain.result import SegmentUnderstanding
from video_demo.fusion.merge import BoundaryPoint, WindowUnderstanding, merge_segment_understandings
from video_demo.fusion.retrieval_text import (
    render_document_chapter_retrieval_text,
    render_document_summary_retrieval_text,
    render_segment_retrieval_text,
)


def test_duplicate_keyword_fields_are_rendered_only_once() -> None:
    understanding = SegmentUnderstanding(
        title="演示",
        summary_zh="展示关键词去重。",
        keywords=(" AI  共创社群 ", "Codex"),
        original_keywords=("ai 共创社群", "codex", "HNSW"),
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

    lines = segments[0].retrieval_text.splitlines()

    assert "关键词：AI 共创社群、Codex" in lines
    assert "原语言关键词：HNSW" in lines


def test_all_duplicate_original_keywords_are_omitted_from_retrieval_text() -> None:
    understanding = SegmentUnderstanding(
        title="演示",
        summary_zh="展示关键词去重。",
        keywords=("AI共创社群", "Codex"),
        original_keywords=("AI共创社群", "codex"),
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

    assert "关键词：AI共创社群、Codex" in segments[0].retrieval_text
    assert "原语言关键词：" not in segments[0].retrieval_text


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
        "文档类型：VIDEO_SEGMENT",
        "视频标题：未提供",
        "片段标题：产品 演示",
        "时间范围：[0, 1000)",
        "文本来源：NONE",
        "语言：zh、en",
        "",
        "局部摘要：展示 视频理解能力。 生成检索文本。",
        "原始文本：无",
        "画面事实：无",
        "OCR文字：无",
        "主题：视频检索",
        "实体：Qwen",
        "动作：演示",
        "关键词：视频理解",
        "原语言关键词：retrieval_text",
    ]
    assert "segment_id" not in segment.retrieval_text
    assert "evidence_refs" not in segment.retrieval_text
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


def test_document_chapter_retrieval_projection_excludes_internal_fields() -> None:
    observation = VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="chapter_001",
        start_ms=100,
        end_ms=200,
        target_ids=("target_001",),
        keyframe_refs=("keyframe_001",),
        transcript_evidence_refs=(),
        visual_type="TEXT",
        caption="界面显示参数 42。",
        relation_to_transcript="INDEPENDENT",
        certainty=0.8,
        uncertainties=("小字号可能识别有误",),
    )
    chapter = SemanticChapter.model_construct(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="参数设置",
        summary_zh="介绍参数配置。",
        body_blocks=(
            ParagraphBlock(text="打开设置页面。", evidence_refs=("asr_001",)),
            VisualBlock(
                visual_observation_ref="visual_001",
                visual_content_refs=(),
                caption="参数为 42。",
                evidence_refs=("visual_001",),
            ),
        ),
        claims=(GroundedClaim(text="参数可以调整。", evidence_refs=("asr_001",), certainty=0.9),),
        content_status="GROUNDED",
        evidence_refs=("asr_001", "visual_001"),
        selected_keyframe_refs=("keyframe_001",),
        transcript_source="ASR",
        retrieval_text="",
        retrieval_hash="0" * 64,
    )

    rendered = render_document_chapter_retrieval_text(chapter, (observation,))

    assert rendered.splitlines() == [
        "文档类型：SEMANTIC_CHAPTER",
        "章节标题：参数设置",
        "时间范围：[0, 1000)",
        "章节摘要：介绍参数配置。",
        "正文：打开设置页面。",
        "关键结论：参数可以调整。",
        "视觉补充：参数为 42。",
        "不确定性：小字号可能识别有误",
    ]
    assert "chapter_001" not in rendered
    assert "visual_001" not in rendered
    assert "keyframe_001" not in rendered
    assert "model" not in rendered.lower()


def test_document_retrieval_projection_preserves_labels_under_character_limits() -> None:
    summary = VideoDocumentSummary(
        title="测试",
        duration_ms=1_000,
        overview_zh="概" * 8_000,
        key_points=(),
        retrieval_text="",
        retrieval_hash=hashlib.sha256(b"").hexdigest(),
    )

    rendered = render_document_summary_retrieval_text(summary)

    assert len(rendered) <= 8_000
    assert "文档类型：" in rendered
    assert "视频标题：" in rendered
    assert "视频时长：" in rendered
    assert "核心概览：" in rendered
    assert "关键结论：" in rendered


def test_document_retrieval_uses_only_selected_visual_blocks_without_duplicate_caption() -> None:
    selected = VisualObservationEvidence(
        evidence_id="visual_selected",
        chapter_id="chapter_001",
        start_ms=100,
        end_ms=200,
        target_ids=("target_selected",),
        keyframe_refs=("keyframe_selected",),
        transcript_evidence_refs=(),
        visual_type="TEXT",
        caption="观察原始描述。",
        relation_to_transcript="INDEPENDENT",
        certainty=0.9,
        uncertainties=("选中观察的不确定性",),
    )
    unused = selected.model_copy(
        update={
            "evidence_id": "visual_unused",
            "target_ids": ("target_unused",),
            "keyframe_refs": ("keyframe_unused",),
            "caption": "未使用的视觉事实不得进入检索。",
            "uncertainties": ("未使用观察的不确定性",),
        },
    )
    chapter = SemanticChapter.model_construct(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="视觉选择",
        summary_zh="只投影正文实际选择的视觉观察。",
        body_blocks=(
            VisualBlock(
                visual_observation_ref=selected.evidence_id,
                visual_content_refs=(),
                caption="最终视觉描述。",
                evidence_refs=(selected.evidence_id,),
            ),
            VisualBlock(
                visual_observation_ref=selected.evidence_id,
                visual_content_refs=(),
                caption="最终视觉描述。",
                evidence_refs=(selected.evidence_id,),
            ),
        ),
        claims=(),
        content_status="GROUNDED",
        evidence_refs=(selected.evidence_id, unused.evidence_id),
        selected_keyframe_refs=("keyframe_selected",),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash="0" * 64,
    )

    rendered = render_document_chapter_retrieval_text(chapter, (selected, unused))

    assert rendered.count("最终视觉描述。") == 1
    assert "正文：无" in rendered
    assert "视觉补充：最终视觉描述。" in rendered
    assert "选中观察的不确定性" in rendered
    assert "未使用的视觉事实不得进入检索" not in rendered
    assert "未使用观察的不确定性" not in rendered
