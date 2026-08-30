from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.document_plan import frame_candidate_id
from video_demo.domain.evidence import (
    ChapterVisualObservation,
    VisualFormulaContentDraft,
    VisualTextContentDraft,
)
from video_demo.domain.manifest import Rational
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.evaluation.annotations import EvaluationAnnotation, VerifiedAnnotation
from video_demo.evaluation.chapter_vlm_input import (
    ChapterVlmInputFrame,
    ChapterVlmInputManifest,
    ValidatedChapterVlmInputContext,
    base_coverage_target_id,
    evaluation_run_id_for_input,
)
from video_demo.evaluation.chapter_vlm_live import (
    build_visual_text_score_fact,
    execute_chapter_vlm_live,
)
from video_demo.integrations.document_port import ChapterVisionRequest, ChapterVisionResponse
from video_demo.integrations.qwen_vl import QwenVisionProviderReceipt

_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
_JPEG_A = b"\xff\xd8\xffimage-a\xff\xd9"
_JPEG_B = b"\xff\xd8\xffimage-b\xff\xd9"


def _sha(value: bytes | str) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _input_fixture(
    tmp_path: Path,
) -> tuple[ChapterVlmInputManifest, ValidatedChapterVlmInputContext, VerifiedAnnotation]:
    run_root = tmp_path / "runs/evaluation/run_001"
    candidate_root = run_root / "visual/candidates"
    candidate_root.mkdir(parents=True)
    (run_root / "visual").chmod(0o700)
    candidate_root.chmod(0o700)
    proxy_path = run_root / "media/source.mp4"
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_bytes(_MP4)
    proxy_path.chmod(0o600)
    source_sha = _sha("source")
    annotation_sha = _sha("annotation")
    requested_ids = ("reference_a", "reference_b")
    evaluation_run_id = evaluation_run_id_for_input(
        "parent_001",
        "sample_001",
        source_sha,
        annotation_sha,
        1280,
        90,
        requested_ids,
    )
    image_sha_a = _sha(_JPEG_A)
    image_sha_b = _sha(_JPEG_B)
    provisional = ChapterVlmInputManifest.model_construct(
        schema_version="1.0.0",
        parent_evaluation_run_id="parent_001",
        evaluation_run_id=evaluation_run_id,
        sample_id="sample_001",
        source_media_sha256=source_sha,
        source_duration_ms=10_000,
        annotation_sha256=annotation_sha,
        proxy_max_edge=1280,
        proxy_width=1280,
        proxy_height=720,
        proxy_frame_rate=Rational(numerator=30, denominator=1),
        proxy_is_variable_frame_rate=False,
        proxy_duration_ms=10_000,
        proxy_relative_path="media/source.mp4",
        duration_tolerance_ms=100,
        jpeg_quality=90,
        proxy_sha256=_sha(_MP4),
        proxy_size_bytes=len(_MP4),
        frame_tolerance_ms=34,
        requested_reference_frame_ids=requested_ids,
        requested_image_sha256s=(image_sha_a, image_sha_b),
        retained_reference_frame_ids=requested_ids,
        duplicate_frame_count=0,
        frames=(),
    )
    target_id = base_coverage_target_id(provisional)
    frames: list[ChapterVlmInputFrame] = []
    for reference_id, timestamp_ms, payload in (
        ("reference_a", 1_000, _JPEG_A),
        ("reference_b", 2_000, _JPEG_B),
    ):
        digest = _sha(payload)
        path = candidate_root / f"{digest}.jpg"
        path.write_bytes(payload)
        path.chmod(0o600)
        frames.append(
            ChapterVlmInputFrame(
                reference_frame_id=reference_id,
                frame_id=frame_candidate_id(
                    source_sha,
                    actual_timestamp_ms=timestamp_ms,
                    image_sha256=digest,
                ),
                requested_timestamp_ms=timestamp_ms,
                actual_timestamp_ms=timestamp_ms,
                relative_path=f"visual/candidates/{digest}.jpg",
                sha256=digest,
                size_bytes=len(payload),
                perceptual_hash="0123456789abcdef",
                target_ids=(target_id,),
            )
        )
    manifest = ChapterVlmInputManifest.model_validate(
        provisional.model_copy(update={"frames": tuple(frames)}).model_dump(mode="python")
    )
    context = ValidatedChapterVlmInputContext(
        parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        evaluation_run_id=manifest.evaluation_run_id,
        sample_id=manifest.sample_id,
        source_media_sha256=manifest.source_media_sha256,
        annotation_sha256=manifest.annotation_sha256,
        source_duration_ms=manifest.source_duration_ms,
        source_display_width=1920,
        source_display_height=1080,
        allowed_run_root=run_root,
        proxy_relative_path=manifest.proxy_relative_path,
        proxy_sha256=manifest.proxy_sha256,
        proxy_size_bytes=manifest.proxy_size_bytes,
        proxy_max_edge=manifest.proxy_max_edge,
        proxy_width=manifest.proxy_width,
        proxy_height=manifest.proxy_height,
        proxy_frame_rate=manifest.proxy_frame_rate,
        proxy_is_variable_frame_rate=manifest.proxy_is_variable_frame_rate,
        proxy_duration_ms=manifest.proxy_duration_ms,
        duration_tolerance_ms=manifest.duration_tolerance_ms,
        frame_tolerance_ms=manifest.frame_tolerance_ms,
        jpeg_quality=manifest.jpeg_quality,
        vlm_max_image_bytes=1024,
        max_candidate_frame_bytes_per_run=1024 * 1024,
        max_candidate_frame_files_per_run=10,
    )
    annotation = EvaluationAnnotation.model_validate(
        {
            "schema_version": "2.0.0",
            "sample_id": manifest.sample_id,
            "media_sha256": manifest.source_media_sha256,
            "duration_ms": manifest.source_duration_ms,
            "language": "zh",
            "reference_text": "Alpha Beta",
            # 人工标注故意反序，评分顺序必须仍由 Manifest 决定。
            "visual_frames": [
                {
                    "frame_id": "reference_b",
                    "timestamp_ms": 2_000,
                    "text_lines": ["Beta"],
                    "quality_categories": ["UI_SMALL_TEXT"],
                    "key_fields": ["Beta"],
                },
                {
                    "frame_id": "reference_a",
                    "timestamp_ms": 1_000,
                    "text_lines": ["Alpha"],
                    "quality_categories": ["FORMULA"],
                    "key_fields": ["Alpha"],
                },
            ],
            "scene_boundaries_ms": [5_000],
            "semantic_boundaries_ms": [5_000],
            "supported_facts": [{"fact_id": "fact_001", "canonical_text": "事实"}],
            "key_fact_ids": ["fact_001"],
        }
    )
    return manifest, context, VerifiedAnnotation(annotation=annotation, sha256=annotation_sha)


def _response(manifest: ChapterVlmInputManifest) -> ChapterVisionResponse:
    frame_a, frame_b = manifest.frames
    return ChapterVisionResponse(
        observations=(
            ChapterVisualObservation(
                target_ids=(base_coverage_target_id(manifest),),
                selected_frame_ids=(frame_b.frame_id, frame_a.frame_id),
                transcript_evidence_refs=(),
                visual_type="FORMULA",
                caption="该描述不应进入逐字评分",
                content_blocks=(
                    VisualTextContentDraft(
                        source_frame_ids=(frame_b.frame_id,),
                        text="Beta",
                    ),
                    VisualFormulaContentDraft(
                        source_frame_ids=(frame_a.frame_id,),
                        latex="Alpha",
                        explanation="该解释不应进入逐字评分",
                    ),
                ),
                visual_facts=(),
                frame_relations=(),
                relation_to_transcript="INDEPENDENT",
                certainty=0.9,
            ),
        )
    )


class _RecordingVisionClient:
    def __init__(self, response: ChapterVisionResponse):
        self.response = response
        self.requests: list[ChapterVisionRequest] = []

    def analyze_chapter_with_receipt(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
    ) -> tuple[ChapterVisionResponse, QwenVisionProviderReceipt]:
        assert allowed_run_root.name == "run_001"
        self.requests.append(request)
        return self.response, QwenVisionProviderReceipt(
            provider_attempt_count=1,
            final_http_status=200,
            provider_response_sha256="d" * 64,
            request_json_bytes=123,
            encoded_request_bytes=123,
            elapsed_ms=7,
        )


def test_execute_live_builds_one_answer_free_production_request(tmp_path: Path) -> None:
    manifest, context, annotation = _input_fixture(tmp_path)
    client = _RecordingVisionClient(_response(manifest))

    response, receipt = execute_chapter_vlm_live(
        manifest,
        context=context,
        expected_parent_evaluation_run_id=manifest.parent_evaluation_run_id,
        expected_evaluation_run_id=manifest.evaluation_run_id,
        vision_client=client,  # type: ignore[arg-type]
    )

    assert response == client.response
    assert len(client.requests) == 1
    request = client.requests[0]
    assert tuple(frame.frame_id for frame in request.frames) == tuple(
        frame.frame_id for frame in manifest.frames
    )
    assert request.targets[0].sample_timestamps_ms == (1_000, 2_000)
    assert request.transcript_evidence == ()
    assert request.document_config.max_visuals_per_chapter == 2
    payload = request.model_dump_json()
    for secret in ("Alpha", "Beta", "FORMULA", "UI_SMALL_TEXT"):
        assert secret not in payload
    assert receipt.request_json_bytes == receipt.encoded_request_bytes == 123
    assert receipt.vlm_elapsed_ms == 7
    assert annotation.annotation.visual_frames


def test_visual_text_score_uses_manifest_order_and_only_typed_projection(
    tmp_path: Path,
) -> None:
    manifest, _context, annotation = _input_fixture(tmp_path)
    response = _response(manifest)
    from video_demo.integrations.document_validation import chapter_vision_response_sha256

    fact = build_visual_text_score_fact(
        manifest,
        annotation,
        response,
        response_sha256=chapter_vision_response_sha256(response),
    )

    assert fact.errors == 0
    assert fact.reference_units == len("Alpha\nBeta")
    assert fact.key_field_matches == fact.key_field_reference_units == 2
    assert fact.quality_categories == ("FORMULA", "UI_SMALL_TEXT")
    assert fact.selected_reference_frame_count == 2


def test_visual_text_score_counts_key_fields_from_all_requested_references(
    tmp_path: Path,
) -> None:
    manifest, _context, annotation = _input_fixture(tmp_path)
    original = _response(manifest).observations[0]
    response = _response(manifest).model_copy(
        update={
            "observations": (
                original.model_copy(
                    update={
                        "selected_frame_ids": (manifest.frames[0].frame_id,),
                        "content_blocks": (original.content_blocks[1],),
                    }
                ),
            )
        }
    )
    from video_demo.integrations.document_validation import chapter_vision_response_sha256

    fact = build_visual_text_score_fact(
        manifest,
        annotation,
        response,
        response_sha256=chapter_vision_response_sha256(response),
    )

    assert fact.key_field_reference_units == 2
    assert fact.key_field_matches == 1


def test_visual_text_score_rejects_unknown_response_reference(tmp_path: Path) -> None:
    manifest, _context, annotation = _input_fixture(tmp_path)
    response = _response(manifest)
    forged = response.model_construct(
        observations=(
            response.observations[0].model_copy(
                update={"selected_frame_ids": ("frame_unknown",)}
            ),
        )
    )

    from video_demo.integrations.document_validation import chapter_vision_response_sha256

    with pytest.raises(VideoDemoError) as raised:
        build_visual_text_score_fact(
            manifest,
            annotation,
            forged,
            response_sha256=chapter_vision_response_sha256(forged),
        )

    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
