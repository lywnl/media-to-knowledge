from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from video_demo.application.document_rendering import render_markdown
from video_demo.application.document_retrieval_text import render_document_chapter_retrieval_text
from video_demo.application.document_writing import _validate_global_response
from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    SummaryPoint,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.evidence import SpeechSegment
from video_demo.integrations.document_port import (
    GlobalChapterInput,
    GlobalWritingRequest,
    GlobalWritingResponse,
)


def _metadata() -> DocumentGenerationMetadata:
    return DocumentGenerationMetadata(
        document_config=DocumentGenerationConfig(),
        text_model_id="text-model",
        vlm_model_id="qwen3-vl-flash",
        prompt_versions=PromptVersions(
            chapter_planner="chapter-planner-v1",
            chapter_planner_repair="chapter-planner-repair-v1",
            chapter_vlm="chapter-vlm-v1",
            chapter_vlm_repair="chapter-vlm-repair-v1",
            chapter_writer="chapter-writer-v1",
            chapter_writer_repair="chapter-writer-repair-v1",
            global_editor="global-editor-v1",
            global_editor_repair="global-editor-repair-v1",
        ),
    )


def _fixture() -> tuple[VideoUnderstandingResult, tuple[SpeechSegment, ...]]:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="讲解安装步骤。",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    retrieval_text = "章节标题：安装步骤\n正文：讲解安装步骤。\n关键结论：完成安装。"
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        title="安装步骤",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="讲解安装步骤。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(ParagraphBlock(text="讲解安装步骤。", evidence_refs=(speech.evidence_id,)),),
        claims=(),
        evidence_refs=(speech.evidence_id,),
        selected_keyframe_refs=(),
        transcript_source="ASR",
        retrieval_text=retrieval_text,
        retrieval_hash=hashlib.sha256(retrieval_text.encode()).hexdigest(),
    )
    summary_retrieval = "视频标题：测试视频\n核心概览：全文介绍安装步骤。"
    result = VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="测试视频",
            duration_ms=10_000,
            overview_zh="全文介绍安装步骤。",
            key_points=(SummaryPoint(text="完成安装。", chapter_refs=(chapter.chapter_id,)),),
            retrieval_text=summary_retrieval,
            retrieval_hash=hashlib.sha256(summary_retrieval.encode()).hexdigest(),
        ),
        chapters=(chapter,),
        generation=_metadata(),
    )
    return result, (speech,)


def test_v4_result_has_only_chapters_and_document_has_no_removed_blocks() -> None:
    result, evidence = _fixture()

    assert result.schema_version == "4.0.0"
    assert "sections" not in result.model_dump(mode="json")
    markdown = render_markdown(result, evidence).content.decode("utf-8")
    assert "## 第一章：安装步骤" in markdown
    assert "## 目录" in markdown
    assert "第一部分" not in markdown
    assert "信息边界" not in markdown
    assert "关键画面引用" not in markdown
    assert markdown.count("安装步骤") == 4

    retrieval = render_document_chapter_retrieval_text(result.chapters[0], ())
    assert "章节标题：安装步骤" in retrieval
    assert "信息边界" not in retrieval


def test_global_editor_request_uses_chapter_fact_projection() -> None:
    result, _ = _fixture()
    context = DocumentWritingContext(
        run_id=result.run_id,
        asset_sha256=result.asset_sha256,
        title_hint=result.summary.title,
        duration_ms=result.summary.duration_ms,
        transcript_source="ASR",
        document_config=DocumentGenerationConfig(),
    )
    request = GlobalWritingRequest(
        context=context,
        chapters=(
            GlobalChapterInput(
                chapter_id="chapter_001",
                start_ms=0,
                end_ms=10_000,
                title="安装步骤",
                summary_zh="讲解安装步骤。",
                key_conclusions=("完成安装。",),
                content_status="GROUNDED",
            ),
        ),
        prompt_version="global-editor-v1",
    )

    payload = request.model_dump(mode="json")
    assert "sections" not in payload
    assert payload["chapters"][0]["key_conclusions"] == ["完成安装。"]


def test_removed_visual_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        from video_demo.domain.evidence import VisualObservationEvidence

        VisualObservationEvidence(
            evidence_id="visual_001",
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=1_000,
            target_ids=("target_001",),
            keyframe_refs=("frame_001",),
            transcript_evidence_refs=("asr_001",),
            visual_type="GENERAL",
            caption="画面",
            relation_to_transcript="SUPPORTING",
            certainty=0.9,
            quality_flags=(),
            uncertainties=(),
        )


def test_global_editor_rejects_empty_overview() -> None:
    result, _ = _fixture()
    context = DocumentWritingContext(
        run_id=result.run_id,
        asset_sha256=result.asset_sha256,
        title_hint=result.summary.title,
        duration_ms=result.summary.duration_ms,
        transcript_source="ASR",
        document_config=DocumentGenerationConfig(),
    )
    request = GlobalWritingRequest(
        context=context,
        chapters=(
            GlobalChapterInput(
                chapter_id="chapter_001",
                start_ms=0,
                end_ms=10_000,
                title="安装步骤",
                summary_zh="讲解安装步骤。",
                key_conclusions=(),
                content_status="GROUNDED",
            ),
        ),
        prompt_version="global-editor-v1",
    )

    with pytest.raises(ValueError, match="核心概览"):
        _validate_global_response(
            GlobalWritingResponse(overview_zh="", key_points=()),
            request,
            result.chapters,
        )
