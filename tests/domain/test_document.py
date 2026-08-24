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
    SemanticSection,
    SummaryPoint,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    section_id_for,
)

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
            chapter_vlm="chapter-vlm-v1",
            chapter_writer="chapter-writer-v1",
            global_editor="global-editor-v1",
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
            summary_zh="这是章节摘要。",
            body_blocks=(ParagraphBlock(text=text, evidence_refs=("asr_001",)),),
            claims=(GroundedClaim(text="可验证结论", evidence_refs=("asr_001",), certainty=0.9),),
            content_status="GROUNDED",
            evidence_refs=("asr_001",),
            selected_keyframe_refs=(),
            transcript_source="ASR",
            retrieval_text=text,
            retrieval_hash=_hash(text),
        )
    return SemanticChapter(
        chapter_id=chapter_id,
        start_ms=start_ms,
        end_ms=end_ms,
        title="本时段未提取到可验证语义内容",
        summary_zh="本时段未提取到可验证语义内容",
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        selected_keyframe_refs=(),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash=_hash(""),
    )


def _result(
    chapters: tuple[SemanticChapter, ...],
    *,
    sections: tuple[SemanticSection, ...] | None = None,
    key_points: tuple[SummaryPoint, ...] = (),
) -> VideoUnderstandingResult:
    section_values = sections or (
        SemanticSection(
            section_id=section_id_for(ASSET_SHA, tuple(item.chapter_id for item in chapters)),
            title="全部内容",
            summary_zh="章节汇总。",
            chapter_refs=tuple(item.chapter_id for item in chapters),
        ),
    )
    summary_text = "视频摘要"
    summary = VideoDocumentSummary(
        title="测试视频",
        duration_ms=chapters[-1].end_ms,
        overview_zh="视频摘要。",
        key_points=key_points,
        retrieval_text=summary_text,
        retrieval_hash=_hash(summary_text),
    )
    return VideoUnderstandingResult(
        schema_version="3.0.0",
        run_id="run_document_001",
        asset_sha256=ASSET_SHA,
        summary=summary,
        sections=section_values,
        chapters=chapters,
        generation=_metadata(),
    )


def test_document_accepts_contiguous_chapters_and_valid_retrieval_hashes() -> None:
    result = _result((_chapter("ch_001", 0, 1_000), _chapter("ch_002", 1_000, 2_000)))

    assert result.schema_version == "3.0.0"
    assert result.summary.duration_ms == result.chapters[-1].end_ms


def test_document_rejects_non_contiguous_or_overlapping_chapters() -> None:
    with pytest.raises(ValidationError, match="连续"):
        _result((_chapter("ch_001", 0, 1_000), _chapter("ch_002", 1_100, 2_000)))


def test_document_rejects_chapter_longer_than_five_minutes() -> None:
    with pytest.raises(ValidationError, match="5 分钟"):
        _result((_chapter("ch_001", 0, 300_001),))


def test_document_rejects_non_3_schema() -> None:
    payload = _result((_chapter("ch_001", 0, 1_000),)).model_dump()
    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError):
        VideoUnderstandingResult.model_validate(payload)


def test_section_refs_must_cover_each_chapter_exactly_once() -> None:
    chapters = (_chapter("ch_001", 0, 1_000), _chapter("ch_002", 1_000, 2_000))
    duplicate = SemanticSection(
        section_id="section_bad",
        title="重复章节",
        summary_zh="非法。",
        chapter_refs=("ch_001", "ch_001"),
    )

    with pytest.raises(ValidationError, match=r"Section|章节"):
        _result(chapters, sections=(duplicate,))


def test_section_id_is_stable_and_does_not_depend_on_title() -> None:
    first = section_id_for(ASSET_SHA, ("ch_001", "ch_002"))
    second = section_id_for(ASSET_SHA, ("ch_001", "ch_002"))
    different_order = section_id_for(ASSET_SHA, ("ch_002", "ch_001"))

    assert first == second
    assert first != different_order


def test_summary_point_must_reference_grounded_existing_chapter() -> None:
    chapters = (_chapter("ch_001", 0, 1_000, grounded=False),)
    point = SummaryPoint(text="不能引用占位章节", chapter_refs=("ch_001",))

    with pytest.raises(ValidationError, match=r"SummaryPoint|关键结论|GROUNDED"):
        _result(chapters, key_points=(point,))


def test_generation_metadata_rejects_endpoint_secret_and_timestamp_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DocumentGenerationMetadata(
            **_metadata().model_dump(),
            endpoint="https://example.invalid",
            generated_at="2026-08-25T00:00:00Z",
            api_key="secret",
        )


def test_no_semantic_evidence_chapter_has_no_claims_or_retrieval_text() -> None:
    payload = _chapter("ch_001", 0, 1_000, grounded=False).model_dump(
        exclude={"duration_ms"},
    )
    payload["claims"] = (GroundedClaim(text="伪造", evidence_refs=("asr_001",), certainty=0.8),)
    with pytest.raises(ValidationError, match=r"NO_SEMANTIC_EVIDENCE|语义"):
        SemanticChapter.model_validate(payload)


def test_summary_retrieval_text_is_limited_to_eight_thousand_characters() -> None:
    text = "a" * 8_001
    with pytest.raises(ValidationError, match="string_too_long"):
        VideoDocumentSummary(
            title="测试视频",
            duration_ms=1_000,
            overview_zh="摘要",
            key_points=(),
            retrieval_text=text,
            retrieval_hash=_hash(text),
        )


def test_chapter_retrieval_text_is_limited_to_thirty_two_thousand_characters() -> None:
    text = "a" * 32_001
    with pytest.raises(ValidationError, match="string_too_long"):
        SemanticChapter(
            chapter_id="ch_001",
            start_ms=0,
            end_ms=1_000,
            title="章节",
            summary_zh="摘要",
            body_blocks=(ParagraphBlock(text="正文", evidence_refs=("asr_001",)),),
            claims=(GroundedClaim(text="结论", evidence_refs=("asr_001",), certainty=0.9),),
            content_status="GROUNDED",
            evidence_refs=("asr_001",),
            transcript_source="ASR",
            retrieval_text=text,
            retrieval_hash=_hash(text),
        )
