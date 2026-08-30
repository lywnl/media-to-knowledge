from __future__ import annotations

import pytest

from video_demo.domain.audio_document import AudioChapter
from video_demo.domain.audio_plan import (
    AudioBodyBlock,
    AudioChapterPlan,
    AudioDocumentConfig,
    AudioGroundedClaim,
    AudioParagraphBlock,
)


def test_audio_document_config_has_no_visual_configuration() -> None:
    config = AudioDocumentConfig(document_title="访谈", detail_level="standard")

    assert config.model_dump(mode="json") == {
        "document_title": "访谈",
        "detail_level": "standard",
        "chapter_granularity": "standard",
        "include_verbatim_quotes": True,
    }
    assert not {"max_visuals_per_chapter", "visual_mode"}.intersection(
        config.model_dump(mode="json"),
    )


def test_audio_body_block_union_rejects_visual_block() -> None:
    with pytest.raises(ValueError):
        AudioChapter(
            chapter_id="audio_chapter_001",
            start_ms=0,
            end_ms=1_000,
            title="访谈",
            title_evidence_refs=("asr_001",),
            summary_zh="摘要",
            summary_evidence_refs=("asr_001",),
            body_blocks=(
                {  # type: ignore[arg-type]
                    "block_type": "VISUAL",
                    "visual_observation_ref": "visual_001",
                    "visual_content_refs": (),
                    "caption": "画面",
                    "evidence_refs": (),
                },
            ),
            claims=(),
            evidence_refs=("asr_001",),
            transcript_source="ASR",
        )


def test_audio_plan_keeps_text_claims_only() -> None:
    body: AudioBodyBlock = AudioParagraphBlock(
        text="语音正文",
        evidence_refs=("asr_001",),
    )
    claim = AudioGroundedClaim(
        text="关键结论",
        evidence_refs=("asr_001",),
        certainty=0.9,
    )
    chapter = AudioChapter(
        chapter_id="audio_chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="访谈",
        title_evidence_refs=("asr_001",),
        summary_zh="摘要",
        summary_evidence_refs=("asr_001",),
        body_blocks=(body,),
        claims=(claim,),
        evidence_refs=("asr_001",),
        transcript_source="ASR",
    )

    assert chapter.body_blocks[0].block_type == "PARAGRAPH"
    assert chapter.claims[0].text == "关键结论"


def test_audio_chapter_plan_contains_only_audio_partition_fields() -> None:
    plan = AudioChapterPlan(
        chapter_id="audio_chapter_001",
        start_ms=0,
        end_ms=1_000,
        segment_refs=("audio_segment_001",),
        title_hint="访谈",
    )

    assert not {"visual_mode", "semantic_targets", "scene_refs"}.intersection(
        plan.model_dump(mode="json"),
    )
