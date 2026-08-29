from __future__ import annotations

from video_demo.domain.document import ParagraphBlock, VisualBlock
from video_demo.domain.evidence import (
    VisualObservationEvidence,
    VisualTextContent,
)
from video_demo.integrations.document_port import ChapterWritingResponse
from video_demo.integrations.document_writing_normalization import (
    normalize_optional_visual_blocks,
)


def _observation_without_content(evidence_id: str) -> VisualObservationEvidence:
    return VisualObservationEvidence(
        evidence_id=evidence_id,
        chapter_id="chapter_001",
        start_ms=1_000,
        end_ms=1_001,
        target_ids=("target_001",),
        keyframe_refs=("keyframe_evidence_001",),
        visual_type="GENERAL",
        caption="画面观察",
        relation_to_transcript="SUPPORTING",
        transcript_evidence_refs=("asr_001",),
        certainty=0.9,
    )


def _observation_with_content(
    evidence_id: str,
    content_id: str,
) -> VisualObservationEvidence:
    return _observation_without_content(evidence_id).model_copy(
        update={
            "content_blocks": (
                VisualTextContent(
                    visual_content_id=content_id,
                    source_keyframe_refs=("keyframe_evidence_001",),
                    text="画面文字",
                ),
            ),
        },
    )


def _response_with_blocks(
    visual: VisualBlock,
) -> ChapterWritingResponse:
    return ChapterWritingResponse(
        title="章节标题",
        title_evidence_refs=("asr_001",),
        summary_zh="章节摘要",
        summary_evidence_refs=("asr_001",),
        body_blocks=(
            ParagraphBlock(text="模型细化后的 ASR 正文", evidence_refs=("asr_001",)),
            visual,
        ),
        claims=(),
    )


def _visual_block(*, observation_id: str, content_refs: tuple[str, ...]) -> VisualBlock:
    return VisualBlock(
        visual_observation_ref=observation_id,
        visual_content_refs=content_refs,
        caption="画面补充",
        evidence_refs=(observation_id,),
    )


def test_empty_visual_observation_clears_fabricated_content_refs() -> None:
    response = _response_with_blocks(
        _visual_block(
            observation_id="visual_001",
            content_refs=("visual_content_fabricated",),
        ),
    )

    normalized = normalize_optional_visual_blocks(
        response,
        (_observation_without_content("visual_001"),),
    )

    visual = next(item for item in normalized.body_blocks if item.block_type == "VISUAL")
    assert visual.visual_content_refs == ()
    assert normalized.body_blocks[0].text == "模型细化后的 ASR 正文"


def test_unknown_visual_block_is_removed_without_dropping_asr_blocks() -> None:
    response = _response_with_blocks(
        _visual_block(
            observation_id="visual_unknown",
            content_refs=("visual_content_unknown",),
        ),
    )

    normalized = normalize_optional_visual_blocks(response, ())

    assert len(normalized.body_blocks) == 1
    assert normalized.body_blocks[0].block_type == "PARAGRAPH"


def test_cross_observation_visual_content_block_is_removed() -> None:
    response = _response_with_blocks(
        _visual_block(
            observation_id="visual_001",
            content_refs=("visual_content_from_other_observation",),
        ),
    )

    normalized = normalize_optional_visual_blocks(
        response,
        (_observation_with_content("visual_001", "visual_content_001"),),
    )

    assert all(item.block_type != "VISUAL" for item in normalized.body_blocks)


def test_valid_visual_block_is_unchanged_and_normalization_is_idempotent() -> None:
    response = _response_with_blocks(
        _visual_block(
            observation_id="visual_001",
            content_refs=("visual_content_001",),
        ),
    )
    observations = (_observation_with_content("visual_001", "visual_content_001"),)

    once = normalize_optional_visual_blocks(response, observations)
    twice = normalize_optional_visual_blocks(once, observations)

    assert once == response
    assert twice == once
