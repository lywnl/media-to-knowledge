from __future__ import annotations

import pytest

from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import FrameCandidateArtifact, VisualSearchTarget
from video_demo.domain.evidence import (
    ChapterVisualObservation,
    SpeechSegment,
    VisualFrameRelationDraft,
    VisualTextContentDraft,
)
from video_demo.integrations.document_port import ChapterVisionRequest, ChapterVisionResponse
from video_demo.integrations.document_validation import (
    normalize_chapter_vision_response_data,
    validate_chapter_vision_response,
)


def _request() -> ChapterVisionRequest:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="查看画面",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="画面文字",
        anchor_evidence_refs=(speech.evidence_id,),
    )
    frames = tuple(
        FrameCandidateArtifact(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            sha256=f"{frame_id[-1]}" * 64,
            size_bytes=10,
            relative_path=f"visual/candidates/{frame_id[-1] * 64}.jpg",
            perceptual_hash="0123456789abcdef",
            target_ids=(target.target_id,),
        )
        for frame_id, timestamp_ms in (("frame_a", 1_000), ("frame_b", 2_000))
    )
    return ChapterVisionRequest(
        chapter_id="chapter_001",
        targets=(target,),
        frames=frames,
        transcript_evidence=(speech,),
        document_config=DocumentGenerationConfig(max_visuals_per_chapter=2),
        prompt_version="chapter-vlm-v1",
    )


def _observation(*, selected_frame_ids: tuple[str, ...] = ("frame_a",)) -> ChapterVisualObservation:
    return ChapterVisualObservation(
        target_ids=("target_001",),
        selected_frame_ids=selected_frame_ids,
        transcript_evidence_refs=("asr_001",),
        visual_type="TEXT",
        caption="画面文字",
        relation_to_transcript="COMPLEMENTARY",
        certainty=0.9,
    )


def test_shared_validation_accepts_valid_response_and_enforces_budget() -> None:
    request = _request()

    validate_chapter_vision_response(
        ChapterVisionResponse(
            observations=(_observation(selected_frame_ids=("frame_a", "frame_b")),)
        ),
        request,
        max_selected_frames=2,
    )

    with pytest.raises(ValueError, match="budget_exceeded"):
        validate_chapter_vision_response(
            ChapterVisionResponse(
                observations=(_observation(selected_frame_ids=("frame_a", "frame_b")),)
            ),
            request,
            max_selected_frames=1,
        )


def test_shared_validation_rejects_unknown_reference_and_wrong_frame_binding() -> None:
    request = _request()

    with pytest.raises(ValueError, match="unknown_reference"):
        validate_chapter_vision_response(
            ChapterVisionResponse(
                observations=(_observation(selected_frame_ids=("frame_unknown",)),)
            ),
            request,
            max_selected_frames=2,
        )

    second_target = VisualSearchTarget(
        target_id="target_002",
        purpose="SEMANTIC",
        query_zh="另一处画面",
        anchor_evidence_refs=("asr_001",),
    )
    with pytest.raises(ValueError, match="frame_binding_mismatch"):
        validate_chapter_vision_response(
            ChapterVisionResponse(
                observations=(
                    _observation().model_copy(update={"target_ids": ("target_002",)}),
                )
            ),
            request.model_copy(update={"targets": (*request.targets, second_target)}),
            max_selected_frames=2,
        )


def test_shared_validation_allows_context_frame_with_additional_target_binding() -> None:
    request = _request()
    second_target = VisualSearchTarget(
        target_id="target_002",
        purpose="SEMANTIC",
        query_zh="辅助画面",
        anchor_evidence_refs=("asr_001",),
    )
    context_frame = request.frames[1].model_copy(update={"target_ids": ("target_002",)})
    request = request.model_copy(
        update={
            "targets": (*request.targets, second_target),
            "frames": (request.frames[0], context_frame),
        },
    )

    validate_chapter_vision_response(
        ChapterVisionResponse(
            observations=(_observation(selected_frame_ids=("frame_a", "frame_b")),),
        ),
        request,
        max_selected_frames=2,
    )

def test_shared_validation_rejects_reverse_frame_relation_and_duplicate_observation() -> None:
    request = _request()
    relation = VisualFrameRelationDraft(
        relation_type="BEFORE_AFTER",
        from_frame_id="frame_b",
        to_frame_id="frame_a",
        description="顺序错误",
    )
    response = ChapterVisionResponse(
        observations=(
            _observation(selected_frame_ids=("frame_a", "frame_b")).model_copy(
                update={"frame_relations": (relation,)}
            ),
        )
    )

    with pytest.raises(ValueError, match="time_order_invalid"):
        validate_chapter_vision_response(response, request, max_selected_frames=2)

    duplicate = ChapterVisionResponse(observations=(_observation(), _observation()))
    with pytest.raises(ValueError, match="duplicate_equivalent"):
        validate_chapter_vision_response(duplicate, request, max_selected_frames=2)


def test_shared_validation_revalidates_model_copy_nested_references() -> None:
    request = _request()
    invalid_block = VisualTextContentDraft(
        source_frame_ids=("frame_a",),
        text="画面文字",
    ).model_copy(update={"source_frame_ids": ("frame_unknown",)})
    observation_data = _observation().model_dump()
    observation_data["content_blocks"] = (invalid_block,)
    invalid_observation = _observation().model_construct(**observation_data)
    response = ChapterVisionResponse.model_construct(observations=(invalid_observation,))

    with pytest.raises(ValueError, match="response:invalid"):
        validate_chapter_vision_response(response, request, max_selected_frames=2)


def test_normalization_drops_one_malformed_observation_and_keeps_valid_one() -> None:
    request = _request()
    valid = _observation().model_dump(mode="json")
    malformed = {
        "target_ids": ["target_001"],
        "selected_frame_ids": ["frame_a"],
        "visual_type": "TEXT",
    }

    normalized = normalize_chapter_vision_response_data(
        {"observations": [malformed, valid]},
        request,
    )

    assert normalized == {"observations": [valid]}
    validate_chapter_vision_response(
        ChapterVisionResponse.model_validate(normalized),
        request,
        max_selected_frames=2,
    )


def test_normalization_drops_all_structurally_invalid_observations_to_no_value() -> None:
    request = _request()
    normalized = normalize_chapter_vision_response_data(
        {
            "observations": [
                {"target_ids": ["target_001"]},
                "不是对象",
            ],
        },
        request,
    )

    assert normalized == {"observations": []}
    assert ChapterVisionResponse.model_validate(normalized).observations == ()


def test_normalization_preserves_unknown_reference_for_hard_validation() -> None:
    request = _request()
    unknown = _observation().model_dump(mode="json")
    unknown["selected_frame_ids"] = ["frame_unknown"]

    normalized = normalize_chapter_vision_response_data(
        {"observations": [unknown]},
        request,
    )

    response = ChapterVisionResponse.model_validate(normalized)
    with pytest.raises(ValueError, match="unknown_reference"):
        validate_chapter_vision_response(response, request, max_selected_frames=2)


def test_normalization_drops_malformed_block_for_unselected_known_frame() -> None:
    request = _request()
    observation = _observation().model_dump(mode="json")
    observation["content_blocks"] = [
        {
            "content_type": "TEXT",
            "source_frame_ids": ["frame_b"],
            "text": "",
        },
    ]

    normalized = normalize_chapter_vision_response_data(
        {"observations": [observation]},
        request,
    )

    assert normalized["observations"][0]["content_blocks"] == []  # type: ignore[index]
    assert len(ChapterVisionResponse.model_validate(normalized).observations) == 1


def test_shared_validation_rejects_duplicate_request_ids_and_incomplete_whitelist() -> None:
    request = _request()
    duplicate_frames = request.model_copy(
        update={"frames": (request.frames[0], request.frames[0])},
    )
    with pytest.raises(ValueError, match=r"request\.frames\.frame_id"):
        validate_chapter_vision_response(
            ChapterVisionResponse(observations=()),
            duplicate_frames,
            max_selected_frames=2,
        )

    with pytest.raises(ValueError, match="allowed_frame_ids"):
        validate_chapter_vision_response(
            ChapterVisionResponse(observations=()),
            request,
            max_selected_frames=2,
            allowed_frames=("frame_a",),
        )


def test_shared_validation_requires_ordered_complete_repair_whitelists() -> None:
    request = _request()

    with pytest.raises(ValueError, match="allowed_frame_ids"):
        validate_chapter_vision_response(
            ChapterVisionResponse(observations=()),
            request,
            max_selected_frames=2,
            allowed_frames=("frame_b", "frame_a"),
            allowed_targets=("target_001",),
            allowed_transcripts=("asr_001",),
        )
