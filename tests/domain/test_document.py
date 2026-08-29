from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    sanitize_document_title,
)


def test_document_generation_config_allows_zero_visuals() -> None:
    configuration = DocumentGenerationConfig(max_visuals_per_chapter=0)

    assert configuration.max_visuals_per_chapter == 0


def test_document_title_is_normalized_without_treating_explicit_title_as_filename() -> None:
    configuration = DocumentGenerationConfig(
        document_title="  faster-whisper v1.0\n/ 入门\\教程  ",
    )

    assert configuration.document_title == "faster-whisper v1.0 入门 教程"


def test_document_title_falls_back_to_cross_platform_filename_stem() -> None:
    assert sanitize_document_title(None, r"C:\uploads\faster-whisper.v1.mp4") == (
        "faster-whisper.v1"
    )
    assert sanitize_document_title(None, "/uploads/课程.mov") == "课程"


def test_document_title_drops_control_characters_and_is_bounded() -> None:
    assert sanitize_document_title("\x00\t\n") is None
    assert sanitize_document_title("标题\u202e伪装") == "标题 伪装"
    assert sanitize_document_title("标题" * 150) == ("标题" * 100)


def test_document_title_keeps_hidden_filename_name_when_falling_back() -> None:
    assert sanitize_document_title(None, "/uploads/.env") == ".env"

ASSET_SHA = "a" * 64


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _chapter(
    chapter_id: str,
    start_ms: int,
    end_ms: int,
    *,
    grounded: bool = True,
) -> SemanticChapter:
    if grounded:
        text = f"章节 {chapter_id} 的正文"
        return SemanticChapter(
            chapter_id=chapter_id,
            start_ms=start_ms,
            end_ms=end_ms,
            title=f"章节 {chapter_id}",
            title_evidence_refs=("asr_001",),
            summary_zh="这是章节摘要。",
            summary_evidence_refs=("asr_001",),
            body_blocks=(ParagraphBlock(text=text, evidence_refs=("asr_001",)),),
            claims=(GroundedClaim(text="可验证结论", evidence_refs=("asr_001",), certainty=0.9),),
            content_status="GROUNDED",
            evidence_refs=("asr_001",),
            selected_keyframe_refs=(),
            transcript_source="ASR",
        )
    return SemanticChapter(
        chapter_id=chapter_id,
        start_ms=start_ms,
        end_ms=end_ms,
        title="本时段未提取到可验证语义内容",
        title_evidence_refs=(),
        summary_zh="本时段未提取到可验证语义内容",
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        selected_keyframe_refs=(),
        transcript_source="NONE",
    )


def _result(
    chapters: tuple[SemanticChapter, ...],
) -> VideoUnderstandingResult:
    summary = VideoDocumentSummary(
        title="测试视频",
        duration_ms=chapters[-1].end_ms,
        overview_zh="视频摘要。",
    )
    return VideoUnderstandingResult(
        run_id="run_document_001",
        asset_sha256=ASSET_SHA,
        summary=summary,
        chapters=chapters,
        generation=_metadata(),
    )


def test_document_accepts_contiguous_chapters_without_retired_fields() -> None:
    result = _result((_chapter("ch_001", 0, 1_000), _chapter("ch_002", 1_000, 2_000)))

    assert result.schema_version == "4.1.0"
    assert result.summary.duration_ms == result.chapters[-1].end_ms


def test_document_rejects_non_contiguous_or_overlapping_chapters() -> None:
    with pytest.raises(ValidationError, match="连续"):
        _result((_chapter("ch_001", 0, 1_000), _chapter("ch_002", 1_100, 2_000)))


def test_document_rejects_chapter_longer_than_five_minutes() -> None:
    with pytest.raises(ValidationError, match="5 分钟"):
        _result((_chapter("ch_001", 0, 300_001),))


@pytest.mark.parametrize("field", ["title_evidence_refs", "summary_evidence_refs"])
def test_grounded_chapter_requires_header_evidence_references(field: str) -> None:
    payload = _chapter("ch_001", 0, 1_000).model_dump(exclude={"duration_ms"})
    payload[field] = ()

    with pytest.raises(ValidationError, match="标题和摘要至少需要一个证据引用"):
        SemanticChapter.model_validate(payload)


def test_document_rejects_non_4_schema() -> None:
    payload = _result((_chapter("ch_001", 0, 1_000),)).model_dump()
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError):
        VideoUnderstandingResult.model_validate(payload)


def test_prompt_versions_requires_main_and_repair_versions() -> None:
    with pytest.raises(ValidationError):
        PromptVersions(
            chapter_planner="chapter-planner-v1",
            chapter_vlm="chapter-vlm-v1",
            chapter_writer="chapter-writer-v1",
            global_editor="global-editor-v1",
        )


def test_generation_metadata_rejects_endpoint_secret_and_timestamp_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DocumentGenerationMetadata(
            **_metadata().model_dump(),
            endpoint="https://example.invalid",
            generated_at="2026-08-25T00:00:00Z",
            api_key="secret",
        )


def test_no_semantic_evidence_chapter_has_no_claims() -> None:
    payload = _chapter("ch_001", 0, 1_000, grounded=False).model_dump(
        exclude={"duration_ms"},
    )
    payload["claims"] = (GroundedClaim(text="伪造", evidence_refs=("asr_001",), certainty=0.8),)
    with pytest.raises(ValidationError, match=r"NO_SEMANTIC_EVIDENCE|语义"):
        SemanticChapter.model_validate(payload)
