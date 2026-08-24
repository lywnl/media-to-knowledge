from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    PromptVersions,
    SemanticChapter,
    SemanticSection,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    section_id_for,
)
from video_demo.domain.document_artifact import (
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
    DocumentArtifactPayload,
)


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
                section_id=section_id_for("a" * 64, (chapter.chapter_id,)),
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
                chapter_planner_repair="chapter-planner-repair-v1",
                chapter_vlm="chapter-vlm-v1",
                chapter_vlm_repair="chapter-vlm-repair-v1",
                chapter_writer="chapter-writer-v1",
                chapter_writer_repair="chapter-writer-repair-v1",
                global_editor="global-editor-v1",
                global_editor_repair="global-editor-repair-v1",
            ),
        ),
    )


def _artifact_payload() -> dict[str, object]:
    return {
        "result": _result(),
        "evidence": (),
        "stage_metrics": dict.fromkeys(RESULT_STAGE_NAMES, 0),
        "model_metrics": dict.fromkeys(MODEL_METRIC_NAMES, 0),
        "status": "SUCCEEDED",
        "warnings": (),
        "transcript_source": "NONE",
        "document_sha256": "b" * 64,
        "document_size_bytes": 10,
    }


def test_document_artifact_requires_3_schema_and_document_digest() -> None:
    artifact = DocumentArtifactPayload.model_validate(_artifact_payload())

    assert artifact.artifact_schema_version == "3.0.0"
    assert artifact.result.schema_version == "3.0.0"


@pytest.mark.parametrize("field,names", [
    ("stage_metrics", RESULT_STAGE_NAMES),
    ("model_metrics", MODEL_METRIC_NAMES),
])
def test_artifact_metrics_require_the_exact_whitelist(
    field: str,
    names: frozenset[str],
) -> None:
    missing = _artifact_payload()
    missing[field] = dict.fromkeys(tuple(names)[1:], 0)
    with pytest.raises(ValidationError, match=r"缺失|白名单|完整"):
        DocumentArtifactPayload.model_validate(missing)

    unknown = _artifact_payload()
    unknown[field] = {**dict.fromkeys(names, 0), "unknown": 0}
    with pytest.raises(ValidationError, match=r"未知|白名单"):
        DocumentArtifactPayload.model_validate(unknown)


@pytest.mark.parametrize("invalid", [True, 1.0, -1, 2**63])
@pytest.mark.parametrize("field,names", [
    ("stage_metrics", RESULT_STAGE_NAMES),
    ("model_metrics", MODEL_METRIC_NAMES),
])
def test_artifact_metrics_require_bounded_strict_non_negative_integers(
    field: str,
    names: frozenset[str],
    invalid: object,
) -> None:
    payload = _artifact_payload()
    metrics = dict.fromkeys(names, 0)
    metrics[next(iter(names))] = invalid  # type: ignore[assignment]
    payload[field] = metrics

    with pytest.raises(ValidationError):
        DocumentArtifactPayload.model_validate(payload)
