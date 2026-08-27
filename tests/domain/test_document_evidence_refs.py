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
    VisualBlock,
    section_id_for,
    validate_evidence_references,
    visual_caption_for_policy,
)
from video_demo.domain.evidence import (
    ChapterVisualObservation,
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
        title_evidence_refs=("asr_001",),
        summary_zh="摘要",
        summary_evidence_refs=("asr_001",),
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


def _keyframe(
    keyframe_id: str = "frame_001",
    *,
    timestamp_ms: int = 600,
    end_ms: int | None = None,
) -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id=f"keyframe_{keyframe_id}",
        start_ms=timestamp_ms,
        end_ms=end_ms or timestamp_ms + 1,
        keyframe_id=keyframe_id,
        timestamp_ms=timestamp_ms,
        relative_path=f"visual/keyframes/{'b' * 64}.jpg",
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
                visual_content_id="visual_content_001",
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
        VisualTextContent(
            visual_content_id="visual_content_001",
            source_keyframe_refs=("frame_unknown",),
            text="42",
        ),
    )
    with pytest.raises(ValidationError, match="source_keyframe_refs"):
        VisualObservationEvidence.model_validate(payload)


def test_visual_observation_requires_uncertainty_for_conflict() -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload["relation_to_transcript"] = "CONFLICTING"
    payload["uncertainties"] = ()
    with pytest.raises(ValidationError, match=r"uncertainties|冲突"):
        VisualObservationEvidence.model_validate(payload)


@pytest.mark.parametrize("uncertainty", ["", "   ", "\n\t"])
def test_conflicting_visual_models_reject_blank_uncertainty(uncertainty: str) -> None:
    evidence_payload = _observation().model_dump(exclude={"duration_ms"})
    evidence_payload.update(
        {
            "relation_to_transcript": "CONFLICTING",
            "uncertainties": (uncertainty,),
        },
    )
    with pytest.raises(ValidationError, match="uncertainties"):
        VisualObservationEvidence.model_validate(evidence_payload)

    draft_payload = {
        "target_ids": ("target_001",),
        "selected_frame_ids": ("frame_001",),
        "transcript_evidence_refs": ("asr_001",),
        "visual_type": "TEXT",
        "caption": "画面显示参数为 42。",
        "relation_to_transcript": "CONFLICTING",
        "certainty": 0.9,
        "uncertainties": (uncertainty,),
    }
    with pytest.raises(ValidationError, match="uncertainties"):
        ChapterVisualObservation.model_validate(draft_payload)


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


def test_cross_object_validation_rejects_keyframe_without_visual_observation() -> None:
    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_speech(), _keyframe()))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


@pytest.mark.parametrize(
    ("relation", "certainty", "policy"),
    (
        ("DUPLICATE", 0.9, "explicit"),
        ("CONFLICTING", 0.9, "explicit"),
        ("COMPLEMENTARY", 0.6, "conservative"),
    ),
)
@pytest.mark.parametrize("surface", ["title", "summary", "paragraph", "claim"])
def test_cross_object_validation_reapplies_visual_policy_to_attributed_text(
    relation: str,
    certainty: float,
    policy: str,
    surface: str,
) -> None:
    payload = _observation().model_dump(exclude={"duration_ms"})
    payload.update(
        {
            "relation_to_transcript": relation,
            "certainty": certainty,
            "uncertainties": ("画面信息需要审慎处理",),
        },
    )
    observation = VisualObservationEvidence.model_validate(payload)
    chapter_updates: dict[str, object] = {
        "evidence_refs": ("asr_001", observation.evidence_id),
    }
    if surface == "title":
        chapter_updates["title_evidence_refs"] = (observation.evidence_id,)
    elif surface == "summary":
        chapter_updates["summary_evidence_refs"] = (observation.evidence_id,)
    elif surface == "paragraph":
        chapter_updates["body_blocks"] = (
            ParagraphBlock(
                text="把视觉观察当作普通正文。",
                evidence_refs=(observation.evidence_id,),
            ),
        )
    else:
        chapter_updates["claims"] = (
            GroundedClaim(
                text="把视觉观察当作确定结论。",
                evidence_refs=(observation.evidence_id,),
                certainty=1.0,
            ),
        )
    chapter = _chapter().model_copy(update=chapter_updates)
    result = _result(chapter).model_copy(
        update={
            "generation": _result(chapter).generation.model_copy(
                update={
                    "document_config": DocumentGenerationConfig(
                        uncertainty_policy=policy,
                    ),
                },
            ),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(
            result,
            (_speech(), _keyframe(), observation),
        )

    assert raised.value.code == ErrorCode.EVIDENCE_RELATION_INVALID


def test_cross_object_validation_requires_every_body_block_to_have_evidence() -> None:
    chapter = _chapter().model_copy(
        update={
            "body_blocks": (
                ParagraphBlock(text="没有来源的正文。", evidence_refs=()),
            ),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(chapter), (_speech(),))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


@pytest.mark.parametrize("surface", ["paragraph", "claim"])
def test_cross_object_validation_requires_body_and_claim_refs_in_chapter_closure(
    surface: str,
) -> None:
    extra_speech = _speech().model_copy(update={"evidence_id": "asr_002"})
    updates: dict[str, object] = {}
    if surface == "paragraph":
        updates["body_blocks"] = (
            ParagraphBlock(text="引用未列入章节闭包的证据。", evidence_refs=("asr_002",)),
        )
    else:
        updates["claims"] = (
            GroundedClaim(
                text="引用未列入章节闭包的结论。",
                evidence_refs=("asr_002",),
                certainty=0.9,
            ),
        )
    chapter = _chapter().model_copy(update=updates)

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(chapter), (_speech(), extra_speech))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


@pytest.mark.parametrize("surface", ["title", "summary", "paragraph", "claim"])
def test_cross_object_validation_rejects_direct_keyframe_attribution(surface: str) -> None:
    keyframe = _keyframe()
    chapter_updates: dict[str, object] = {
        "evidence_refs": ("asr_001", keyframe.evidence_id),
    }
    if surface == "title":
        chapter_updates["title_evidence_refs"] = (keyframe.evidence_id,)
    elif surface == "summary":
        chapter_updates["summary_evidence_refs"] = (keyframe.evidence_id,)
    elif surface == "paragraph":
        chapter_updates["body_blocks"] = (
            ParagraphBlock(text="直接解读关键帧。", evidence_refs=(keyframe.evidence_id,)),
        )
    else:
        chapter_updates["claims"] = (
            GroundedClaim(
                text="直接把关键帧写成结论。",
                evidence_refs=(keyframe.evidence_id,),
                certainty=1.0,
            ),
        )
    chapter = _chapter().model_copy(update=chapter_updates)

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(chapter), (_speech(), keyframe))

    assert raised.value.code == ErrorCode.EVIDENCE_TYPE_MISMATCH


@pytest.mark.parametrize("policy", ["explicit", "conservative"])
def test_cross_object_validation_rejects_unmarked_conflicting_visual_caption(
    policy: str,
) -> None:
    observation = _observation().model_copy(
        update={
            "relation_to_transcript": "CONFLICTING",
            "uncertainties": ("屏幕数字与口述不一致",),
        },
    )
    chapter = _chapter().model_copy(
        update={
            "body_blocks": (
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_content_001",),
                    caption="参数确定是 42。",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
            "evidence_refs": ("asr_001", observation.evidence_id, _keyframe().evidence_id),
            "selected_keyframe_refs": (_keyframe().evidence_id,),
        },
    )
    result = _result(chapter).model_copy(
        update={
            "generation": _result(chapter).generation.model_copy(
                update={
                    "document_config": DocumentGenerationConfig(
                        uncertainty_policy=policy,
                    ),
                },
            ),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(
            result,
            (_speech(), _keyframe(), observation),
        )

    assert raised.value.code == ErrorCode.EVIDENCE_RELATION_INVALID


def test_conservative_conflict_caption_preserves_both_visual_side_and_uncertainty() -> None:
    observation = _observation().model_copy(
        update={
            "caption": "画面显示参数为 42。",
            "relation_to_transcript": "CONFLICTING",
            "uncertainties": ("口述参数为 7，与画面不一致",),
        },
    )

    caption = visual_caption_for_policy(
        observation,
        DocumentGenerationConfig(uncertainty_policy="conservative"),
    )

    assert observation.caption in caption
    assert observation.uncertainties[0] in caption
    assert "未采纳为确定事实" in caption


def test_cross_object_validation_requires_visual_block_to_reference_only_its_observation() -> None:
    observation = _observation()
    chapter = _chapter().model_copy(
        update={
            "body_blocks": (
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_content_001",),
                    caption="视觉正文混入了转写引用。",
                    evidence_refs=(observation.evidence_id, "asr_001"),
                ),
            ),
            "evidence_refs": ("asr_001", observation.evidence_id),
            "selected_keyframe_refs": (_keyframe().evidence_id,),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(
            _result(chapter),
            (_speech(), _keyframe(), observation),
        )

    assert raised.value.code == ErrorCode.EVIDENCE_RELATION_INVALID


def test_visual_block_rejects_unknown_or_cross_observation_content_id() -> None:
    observation = _observation()
    chapter = _chapter().model_copy(
        update={
            "body_blocks": (
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_content_other",),
                    caption="非法子内容",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
            "evidence_refs": ("asr_001", observation.evidence_id),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(chapter), (_keyframe(), observation, _speech()))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


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


def test_cross_object_validation_requires_observation_to_cover_its_keyframes() -> None:
    observation = _observation().model_copy(update={"start_ms": 500, "end_ms": 550})

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(_result(), (_speech(), _keyframe(), observation))

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_CHAPTER


def test_keyframe_contract_rejects_retired_shape_at_model_boundary() -> None:
    payload = _keyframe().model_dump(exclude={"duration_ms"})

    for field, invalid in (
        ("mime_type", "image/png"),
        ("relative_path", "../visual/keyframes/" + "b" * 64 + ".jpg"),
        ("relative_path", "visual/keyframes/not-the-digest.jpg"),
        ("end_ms", payload["end_ms"] + 1),
    ):
        invalid_payload = {**payload, field: invalid}
        with pytest.raises((ValidationError, VideoDemoError)):
            legacy_compatible = KeyframeEvidence.model_validate(invalid_payload)
            validate_evidence_references(_result(), (_speech(), legacy_compatible))


def test_keyframe_at_chapter_end_is_clipped_to_a_valid_half_open_range() -> None:
    keyframe = _keyframe(timestamp_ms=999, end_ms=1_000)

    assert keyframe.start_ms == 999
    assert keyframe.end_ms == 1_000


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


def test_cross_object_validation_rebuilds_selected_keyframes_from_visual_blocks() -> None:
    first = _keyframe("frame_001", timestamp_ms=600)
    second = _keyframe("frame_002", timestamp_ms=650)
    observation = _observation().model_copy(
        update={
            "keyframe_refs": (first.evidence_id, second.evidence_id),
            "content_blocks": (
                VisualTextContent(
                    visual_content_id="visual_content_first",
                    source_keyframe_refs=(first.evidence_id,),
                    text="第一帧内容",
                ),
                VisualTextContent(
                    visual_content_id="visual_content_second",
                    source_keyframe_refs=(second.evidence_id,),
                    text="第二帧内容",
                ),
            ),
        },
    )
    chapter = _chapter().model_copy(
        update={
            "body_blocks": (
                VisualBlock(
                    visual_observation_ref=observation.evidence_id,
                    visual_content_refs=("visual_content_first",),
                    caption="只展示第一帧",
                    evidence_refs=(observation.evidence_id,),
                ),
            ),
            "evidence_refs": (
                "asr_001",
                observation.evidence_id,
                second.evidence_id,
            ),
            "selected_keyframe_refs": (second.evidence_id,),
        },
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_evidence_references(
            _result(chapter),
            (_speech(), first, second, observation),
        )

    assert raised.value.code == ErrorCode.EVIDENCE_RELATION_INVALID
