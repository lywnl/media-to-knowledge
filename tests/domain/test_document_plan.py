from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_demo.domain.document_plan import (
    BaseSegment,
    ChapterDraft,
    ChapterPlan,
    FrameCandidateArtifact,
    VisualSearchTarget,
)
from video_demo.domain.evidence import ChapterVisualObservation


def test_chapter_plan_contains_only_program_owned_time_and_ids() -> None:
    segment = BaseSegment(
        segment_id="segment_001",
        start_ms=0,
        end_ms=10_000,
        evidence_refs=("asr_001",),
        transcript_source="ASR",
    )
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="屏幕上的配置参数",
        anchor_evidence_refs=("asr_001",),
    )
    plan = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=(segment.segment_id,),
        title_hint="配置参数",
        visual_mode="SINGLE",
        semantic_targets=(target,),
        base_coverage_targets=(),
    )

    assert plan.segment_refs == ("segment_001",)
    assert "start_ms" not in ChapterDraft.model_fields
    assert "chapter_id" not in ChapterDraft.model_fields


def test_base_segment_without_transcript_allows_empty_evidence_refs() -> None:
    segment = BaseSegment(
        segment_id="segment_001",
        start_ms=0,
        end_ms=10_000,
        evidence_refs=(),
        transcript_source="NONE",
    )

    assert segment.evidence_refs == ()


def test_semantic_target_requires_one_to_three_transcript_anchors() -> None:
    with pytest.raises(ValidationError):
        VisualSearchTarget(
            target_id="target_001",
            purpose="SEMANTIC",
            query_zh="参数",
            anchor_evidence_refs=(),
        )


def test_base_coverage_target_cannot_use_transcript_anchor() -> None:
    with pytest.raises(ValidationError, match=r"scene_refs|锚点"):
        VisualSearchTarget(
            target_id="target_001",
            purpose="BASE_COVERAGE",
            query_zh="代表性画面",
            anchor_evidence_refs=("asr_001",),
            scene_refs=(),
        )


def test_frame_candidate_has_bounded_identity_and_positive_size() -> None:
    digest = "a" * 64
    candidate = FrameCandidateArtifact(
        frame_id="frame_001",
        timestamp_ms=2_000,
        sha256=digest,
        size_bytes=128,
        relative_path=f"visual/candidates/{digest}.jpg",
        perceptual_hash="0123456789abcdef",
        target_ids=("target_001",),
    )

    assert candidate.size_bytes == 128


def test_frame_candidate_is_jpeg_only_and_uses_content_addressed_run_path() -> None:
    valid = {
        "frame_id": "frame_001",
        "timestamp_ms": 2_000,
        "sha256": "a" * 64,
        "size_bytes": 128,
        "relative_path": f"visual/candidates/{'a' * 64}.jpg",
        "mime_type": "image/jpeg",
        "perceptual_hash": "0123456789abcdef",
        "target_ids": ("target_001",),
    }

    for field, invalid in (
        ("mime_type", "image/png"),
        ("relative_path", f"runs/scope/run/visual/candidates/{'a' * 64}.jpg"),
        ("relative_path", "visual/candidates/wrong.jpg"),
    ):
        with pytest.raises(ValidationError):
            FrameCandidateArtifact.model_validate({**valid, field: invalid})


def test_candidate_and_observation_accept_at_most_six_target_ids() -> None:
    target_ids = tuple(f"target_{index}" for index in range(6))
    candidate = FrameCandidateArtifact(
        frame_id="frame_001",
        timestamp_ms=2_000,
        sha256="a" * 64,
        size_bytes=128,
        relative_path=f"visual/candidates/{'a' * 64}.jpg",
        perceptual_hash="0123456789abcdef",
        target_ids=target_ids,
    )
    observation = ChapterVisualObservation(
        target_ids=target_ids,
        selected_frame_ids=("frame_001",),
        visual_type="GENERAL",
        caption="代表性画面",
        relation_to_transcript="INDEPENDENT",
        certainty=0.9,
    )

    assert len(candidate.target_ids) == 6
    assert len(observation.target_ids) == 6

    with pytest.raises(ValidationError):
        FrameCandidateArtifact.model_validate(
            {**candidate.model_dump(), "target_ids": (*target_ids, "target_6")},
        )
    with pytest.raises(ValidationError):
        ChapterVisualObservation.model_validate(
            {**observation.model_dump(), "target_ids": (*target_ids, "target_6")},
        )


def test_chapter_models_accept_twenty_thousand_segment_refs_but_no_more() -> None:
    segment_refs = tuple(f"segment_{index}" for index in range(20_000))
    plan = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=segment_refs,
        title_hint="章节",
        visual_mode="NONE",
        semantic_targets=(),
        base_coverage_targets=(),
    )
    draft = ChapterDraft(
        segment_refs=segment_refs,
        title_hint="章节",
        visual_mode="NONE",
        semantic_targets=(),
    )

    assert len(plan.segment_refs) == 20_000
    assert len(draft.segment_refs) == 20_000

    too_many = (*segment_refs, "segment_20000")
    with pytest.raises(ValidationError):
        ChapterPlan.model_validate({**plan.model_dump(), "segment_refs": too_many})
    with pytest.raises(ValidationError):
        ChapterDraft.model_validate({**draft.model_dump(), "segment_refs": too_many})


def test_base_coverage_without_scenes_uses_program_owned_sample_timestamp() -> None:
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="BASE_COVERAGE",
        query_zh="章节代表性画面",
        sample_timestamps_ms=(5_000,),
    )
    plan = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="章节",
        visual_mode="SINGLE",
        semantic_targets=(),
        base_coverage_targets=(target,),
    )

    assert plan.base_coverage_targets[0].sample_timestamps_ms == (5_000,)


def test_base_coverage_sample_timestamp_must_be_inside_chapter() -> None:
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="BASE_COVERAGE",
        query_zh="章节代表性画面",
        sample_timestamps_ms=(10_000,),
    )
    with pytest.raises(ValidationError, match="章节范围"):
        ChapterPlan(
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=10_000,
            segment_refs=("segment_001",),
            title_hint="章节",
            visual_mode="SINGLE",
            semantic_targets=(),
            base_coverage_targets=(target,),
        )


def test_none_visual_mode_rejects_semantic_targets() -> None:
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="参数",
        anchor_evidence_refs=("asr_001",),
    )
    with pytest.raises(ValidationError, match="NONE"):
        ChapterPlan(
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=10_000,
            segment_refs=("segment_001",),
            title_hint="章节",
            visual_mode="NONE",
            semantic_targets=(target,),
            base_coverage_targets=(),
        )


def test_complex_visual_mode_requires_two_disjoint_anchor_groups() -> None:
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="参数",
        anchor_evidence_refs=("asr_001",),
    )
    with pytest.raises(ValidationError, match="两个"):
        ChapterPlan(
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=10_000,
            segment_refs=("segment_001",),
            title_hint="章节",
            visual_mode="COMPARISON",
            semantic_targets=(target,),
            base_coverage_targets=(),
        )

    overlapping = VisualSearchTarget(
        target_id="target_002",
        purpose="SEMANTIC",
        query_zh="另一个参数",
        anchor_evidence_refs=("asr_001", "asr_002"),
    )
    with pytest.raises(ValidationError, match="不重叠"):
        ChapterPlan(
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=10_000,
            segment_refs=("segment_001",),
            title_hint="章节",
            visual_mode="MULTI_STEP",
            semantic_targets=(target, overlapping),
            base_coverage_targets=(),
        )
