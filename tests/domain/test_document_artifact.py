from __future__ import annotations

import hashlib

from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    PromptVersions,
    SemanticChapter,
    SemanticSection,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import DocumentArtifactPayload


def _result() -> VideoUnderstandingResult:
    text = ""
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="无语义",
        summary_zh="本时段未提取到可验证语义内容",
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
        retrieval_text=text,
        retrieval_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    return VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="测试",
            duration_ms=1_000,
            overview_zh="无可验证语义",
            key_points=(),
            retrieval_text=text,
            retrieval_hash=hashlib.sha256(text.encode()).hexdigest(),
        ),
        sections=(
            SemanticSection(
                section_id="section_001",
                title="全部内容",
                summary_zh="无可验证语义",
                chapter_refs=(chapter.chapter_id,),
            ),
        ),
        chapters=(chapter,),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text",
            vlm_model_id="vlm",
            prompt_versions=PromptVersions(
                chapter_planner="chapter-planner-v1",
                chapter_vlm="chapter-vlm-v1",
                chapter_writer="chapter-writer-v1",
                global_editor="global-editor-v1",
            ),
        ),
    )


def test_document_artifact_requires_3_schema_and_document_digest() -> None:
    artifact = DocumentArtifactPayload(
        result=_result(),
        evidence=(),
        stage_metrics={"RESULT": 1},
        status="SUCCEEDED",
        warnings=(),
        transcript_source="NONE",
        document_sha256="b" * 64,
        document_size_bytes=10,
    )

    assert artifact.artifact_schema_version == "3.0.0"
    assert artifact.result.schema_version == "3.0.0"
