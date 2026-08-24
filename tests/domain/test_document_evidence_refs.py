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
    VideoDocumentSummary,
    VideoUnderstandingResult,
    section_id_for,
    validate_evidence_references,
)
from video_demo.domain.evidence import (
    KeyframeEvidence,
    SpeechSegment,
    VisualFrameRelation,
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.errors import ErrorCode, VideoDemoError


def _chapter() -> SemanticChapter:
    text = "章节正文"
    return SemanticChapter(
        chapter_id="ch_001",
        start_ms=0,
        end_ms=1_000,
        title="章节",
        summary_zh="摘要",
        body_blocks=(ParagraphBlock(text=text, evidence_refs=("asr_001",)),),
        claims=(GroundedClaim(text="结论", evidence_refs=("asr_001",), certainty=0.9),),
        evidence_refs=("asr_001",),
        transcript_source="ASR",
        retrieval_text=text,
        retrieval_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _result(chapter: SemanticChapter | None = None) -> VideoUnderstandingResult:
    actual = chapter or _chapter()
    section = SemanticSection(
        section_id=section_id_for("a" * 64, (actual.chapter_id,)),
        title="章节",
        summary_zh="摘要",
        chapter_refs=(actual.chapter_id,),
    )
    summary_text = "视频摘要"
    return VideoUnderstandingResult(
        schema_version="3.0.0",
        run_id="run_document_002",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="测试",
            duration_ms=1_000,
            overview_zh="摘要",
            key_points=(),
            retrieval_text=summary_text,
            retrieval_hash=hashlib.sha256(summary_text.encode()).hexdigest(),
        ),
        sections=(section,),
        chapters=(actual,),
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


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=100,
        end_ms=500,
        text="屏幕上显示一个数字。",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _keyframe(keyframe_id: str = "frame_001", *, timestamp_ms: int = 600) -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id=f"keyframe_{keyframe_id}",
        start_ms=500,
        end_ms=700,
        keyframe_id=keyframe_id,
        timestamp_ms=timestamp_ms,
        relative_path=f"visual/keyframes/{keyframe_id}.jpg",
        mime_type="image/jpeg",
        sha256="b" * 64,
        perceptual_hash="0123456789abcdef",
        size_bytes=123,
    )


def _observation() -> VisualObservationEvidence:
    return VisualObservationEvidence(
        evidence_id="visual_001",
        chapter_id="ch_001",
        start_ms=500,
        end_ms=700,
        target_ids=("target_001",),
        keyframe_refs=("keyframe_frame_001",),
        transcript_evidence_refs=("asr_001",),
        visual_type="TEXT",
        caption="画面显示一个数字。",
        content_blocks=(
            VisualTextContent(
                source_keyframe_refs=("keyframe_frame_001",),
                text="42",
            ),
        ),
        relation_to_transcript="COMPLEMENTARY",
        certainty=0.9,
    )


def test_visual_observation_rejects_unknown_local_frame_reference() -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload["content_blocks"] = (
        VisualTextContent(source_keyframe_refs=("frame_unknown",), text="42"),
    )
    with pytest.raises(ValidationError, match="source_keyframe_refs"):
        VisualObservationEvidence.model_validate(payload)


def test_visual_observation_requires_uncertainty_for_conflict() -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload["relation_to_transcript"] = "CONFLICTING"
    payload["uncertainties"] = ()
    with pytest.raises(ValidationError, match=r"uncertainties|冲突"):
        VisualObservationEvidence.model_validate(payload)


def test_cross_object_validation_requires_frame_and_transcript_membership() -> None:
    result = _result()

    with pytest.raises(VideoDemoError) as missing_frame:
        validate_evidence_references(result, (_speech(), _observation()))
    assert missing_frame.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE

    with pytest.raises(VideoDemoError) as missing_transcript:
        validate_evidence_references(result, (_keyframe(), _observation()))
    assert missing_transcript.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


def test_cross_object_validation_accepts_visual_observation_and_body_refs() -> None:
    result = _result()
    validate_evidence_references(result, (_speech(), _keyframe(), _observation()))


def test_cross_object_validation_requires_selected_keyframes_in_chapter_evidence() -> None:
    chapter = _chapter().model_copy(update={"selected_keyframe_refs": ("missing_frame",)})
    result = _result(chapter)

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(result, (_speech(),))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


def test_cross_object_validation_rejects_observation_outside_chapter() -> None:
    observation = _observation().model_copy(update={"start_ms": 1_200, "end_ms": 1_300})

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_speech(), _keyframe(), observation))

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_CHAPTER


def test_cross_object_validation_rejects_reverse_frame_relation() -> None:
    first = _keyframe("frame_001", timestamp_ms=600)
    second = _keyframe("frame_002", timestamp_ms=650)
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload["keyframe_refs"] = (first.evidence_id, second.evidence_id)
    payload["content_blocks"] = ()
    payload["frame_relations"] = (
        VisualFrameRelation(
            relation_type="BEFORE_AFTER",
            from_keyframe_ref=second.evidence_id,
            to_keyframe_ref=first.evidence_id,
            description="时间顺序反向",
        ),
    )
    observation = VisualObservationEvidence.model_validate(payload)

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_speech(), first, second, observation))

    assert raised.value.code == ErrorCode.EVIDENCE_RELATION_INVALID
