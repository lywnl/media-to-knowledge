from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_demo.application.chapter_frames import ChapterFrameSearchBatch
from video_demo.application.chapter_vision import ChapterVisionService
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import (
    ChapterFrameSet,
    ChapterPlan,
    FrameCandidateArtifact,
    VisualSearchTarget,
    frame_candidate_id,
)
from video_demo.domain.evidence import (
    ChapterVisualObservation,
    SpeechSegment,
    VisualFactDraft,
    VisualFrameRelationDraft,
    VisualTextContentDraft,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterVisionRepairRequest,
    ChapterVisionRequest,
    ChapterVisionResponse,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_ASSET_SHA256 = "a" * 64
_FIXTURE_DIGEST = hashlib.sha256(b"\xff\xd8\xffchapter-vision\xff\xd9").hexdigest()


def _frame_id(timestamp_ms: int, image_sha256: str) -> str:
    return frame_candidate_id(
        _ASSET_SHA256,
        actual_timestamp_ms=timestamp_ms,
        image_sha256=image_sha256,
    )


def _updated_frame(
    frame: FrameCandidateArtifact,
    **updates: object,
) -> FrameCandidateArtifact:
    values = frame.model_dump(mode="python")
    values.update(updates)
    values["frame_id"] = frame_candidate_id(
        _ASSET_SHA256,
        actual_timestamp_ms=values["timestamp_ms"],
        image_sha256=values["sha256"],
    )
    return FrameCandidateArtifact.model_validate(values)


class _VisionPort:
    def __init__(
        self,
        response: ChapterVisionResponse | VideoDemoError,
    ) -> None:
        self._response = response
        self.requests: list[ChapterVisionRequest] = []

    def analyze_chapter(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        assert allowed_run_root.is_absolute()
        self.requests.append(request)
        if on_provider_attempt is not None:
            on_provider_attempt()
        if isinstance(self._response, VideoDemoError):
            raise self._response
        return self._response

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        raise AssertionError("合法主响应不应调用修复")


def _identity() -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation="chapter_vision",
        provider_config_fingerprint="a" * 64,
        model_id="qwen3-vl-flash",
        generation_config=(("temperature", "0"),),
        main_response_schema_name="chapter_vlm_v2",
        main_prompt_version="chapter-vlm-v1",
        repair_response_schema_name="chapter_vlm_repair_v2",
        repair_prompt_version="chapter-vlm-repair-v1",
    )


def _fixture(
    tmp_path: Path,
) -> tuple[Path, ChapterPlan, ChapterFrameSearchBatch, SpeechSegment, bytes]:
    run_root = tmp_path / "runs/scope_001/run_001"
    run_root.mkdir(parents=True)
    payload = b"\xff\xd8\xffchapter-vision\xff\xd9"
    digest = hashlib.sha256(payload).hexdigest()
    candidate = run_root / "visual/candidates" / f"{digest}.jpg"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    candidate.chmod(0o600)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="画面显示并发数为 2",
        language="zh",
        confidence=0.99,
        is_fully_evaluated_language=True,
    )
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="读取并发配置",
        anchor_evidence_refs=(speech.evidence_id,),
    )
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="并发配置",
        visual_mode="SINGLE",
        semantic_targets=(target,),
        base_coverage_targets=(),
    )
    frame = FrameCandidateArtifact(
        frame_id=_frame_id(1_500, digest),
        timestamp_ms=1_500,
        sha256=digest,
        size_bytes=len(payload),
        relative_path=f"visual/candidates/{digest}.jpg",
        perceptual_hash="0123456789abcdef",
        target_ids=(target.target_id,),
    )
    frame_batch = ChapterFrameSearchBatch(
        asset_sha256=_ASSET_SHA256,
        allowed_run_root=run_root,
        frame_tolerance_ms=50,
        frame_sets=(ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=(frame,)),),
        chapter_status=((chapter.chapter_id, "SUCCEEDED"),),
        metrics={},
    )
    return run_root, chapter, frame_batch, speech, payload


def _response() -> ChapterVisionResponse:
    return ChapterVisionResponse(
        observations=(
            ChapterVisualObservation(
                target_ids=("target_001",),
                selected_frame_ids=(_frame_id(1_500, _FIXTURE_DIGEST),),
                transcript_evidence_refs=("asr_001",),
                visual_type="TEXT",
                caption="配置将并发数设置为 2",
                content_blocks=(
                    VisualTextContentDraft(
                        source_frame_ids=(_frame_id(1_500, _FIXTURE_DIGEST),),
                        text="vlm_concurrency = 2",
                    ),
                ),
                relation_to_transcript="COMPLEMENTARY",
                certainty=0.95,
            ),
        ),
    )


def _service(
    tmp_path: Path,
    port: _VisionPort,
    *,
    invocation_wait_timeout_seconds: float = 2,
) -> ChapterVisionService:
    return ChapterVisionService(
        port,
        _identity(),
        runtime_root=tmp_path,
        concurrency=2,
        max_image_bytes=1024,
        max_request_image_bytes=4096,
        max_encoded_request_bytes=64 * 1024,
        max_published_keyframe_bytes=4096,
        max_published_keyframe_files=10,
        invocation_wait_timeout_seconds=invocation_wait_timeout_seconds,
        candidate_lock_timeout_seconds=2,
    )


@pytest.mark.parametrize(
    ("identity_field", "invalid_schema"),
    (
        ("main_response_schema_name", "chapter_vlm_v1"),
        ("repair_response_schema_name", "chapter_vlm_repair_v1"),
    ),
)
def test_chapter_vision_rejects_stale_cache_schema_identity(
    tmp_path: Path,
    identity_field: str,
    invalid_schema: str,
) -> None:
    identity = _identity().model_copy(update={identity_field: invalid_schema})

    with pytest.raises(ValueError, match=r"章节视觉缓存身份与.*Schema 不一致"):
        ChapterVisionService(
            _VisionPort(_response()),
            identity,
            runtime_root=tmp_path,
            concurrency=2,
            max_image_bytes=1024,
            max_request_image_bytes=4096,
            max_encoded_request_bytes=64 * 1024,
            max_published_keyframe_bytes=4096,
            max_published_keyframe_files=10,
            invocation_wait_timeout_seconds=2,
            candidate_lock_timeout_seconds=2,
        )


def _replace_frame_batch(
    frame_batch: ChapterFrameSearchBatch,
    *,
    chapter: ChapterPlan,
    candidates: tuple[FrameCandidateArtifact, ...],
    status: str = "SUCCEEDED",
) -> ChapterFrameSearchBatch:
    return ChapterFrameSearchBatch(
        asset_sha256=frame_batch.asset_sha256,
        allowed_run_root=frame_batch.allowed_run_root,
        frame_tolerance_ms=frame_batch.frame_tolerance_ms,
        frame_sets=(ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=candidates),),
        chapter_status=((chapter.chapter_id, status),),
        status="PARTIAL_SUCCEEDED" if status == "DEGRADED" else "SUCCEEDED",
        metrics={},
    )


def _cache(run_root: Path) -> DocumentModelCache:
    return DocumentModelCache(run_root, max_entry_bytes=64 * 1024, max_run_bytes=256 * 1024)


def test_chapter_vision_maps_response_and_publishes_selected_keyframe(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, payload = _fixture(tmp_path)
    port = _VisionPort(_response())

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert len(port.requests) == 1
    assert port.requests[0].transcript_evidence == (speech,)
    assert result.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert result.status == "SUCCEEDED"
    assert result.warnings == ()
    assert len(result.observations) == len(result.keyframe_evidence) == 1
    keyframe = result.keyframe_evidence[0]
    observation = result.observations[0]
    assert observation.keyframe_refs == (keyframe.evidence_id,)
    assert result.evidence == (keyframe, observation)
    assert (run_root / keyframe.relative_path).read_bytes() == payload
    assert result.metrics == {
        "vlm_logical_analyses": 1,
        "vlm_provider_attempts": 1,
        "vlm_structure_repairs": 0,
        "vlm_cache_hits": 0,
        "vlm_no_value_chapters": 0,
        "vlm_fallback_chapters": 0,
        "visual_published_budget_degraded_chapters": 0,
    }


def test_chapter_vision_generates_stable_unique_content_and_fact_ids(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    base = _response().observations[0]
    response = ChapterVisionResponse(
        observations=(
            base.model_copy(
                update={
                    "content_blocks": (
                        VisualTextContentDraft(
                            source_frame_ids=(frame_batch.frame_sets[0].candidates[0].frame_id,),
                            text="vlm_concurrency = 2",
                        ),
                        VisualTextContentDraft(
                            source_frame_ids=(frame_batch.frame_sets[0].candidates[0].frame_id,),
                            text="worker_concurrency = 2",
                        ),
                    ),
                    "visual_facts": (
                        VisualFactDraft(
                            text="界面显示两个并发配置",
                            source_frame_ids=(frame_batch.frame_sets[0].candidates[0].frame_id,),
                        ),
                    ),
                },
            ),
        ),
    )
    port = _VisionPort(response)
    service = _service(tmp_path, port)
    cache = _cache(run_root)
    arguments = (
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
    )

    first = service.analyze_all(
        *arguments,
        cache=cache,
        is_cancel_requested=lambda: False,
    )
    second = service.analyze_all(
        *arguments,
        cache=cache,
        is_cancel_requested=lambda: False,
    )

    first_observation = first.observations[0]
    second_observation = second.observations[0]
    first_ids = (
        *(item.visual_content_id for item in first_observation.content_blocks),
        *(item.visual_fact_id for item in first_observation.visual_facts),
    )
    second_ids = (
        *(item.visual_content_id for item in second_observation.content_blocks),
        *(item.visual_fact_id for item in second_observation.visual_facts),
    )
    assert len(first_ids) == len(set(first_ids)) == 3
    assert second_ids == first_ids
    assert len(port.requests) == 1
    assert second.metrics["vlm_cache_hits"] == 1


def test_chapter_vision_draft_schema_does_not_expose_final_content_ids() -> None:
    schema_json = json.dumps(ChapterVisionResponse.model_json_schema(), ensure_ascii=False)

    assert "visual_content_id" not in schema_json
    assert "visual_fact_id" not in schema_json


def test_chapter_vision_cache_hit_still_revalidates_candidate_file(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    port = _VisionPort(_response())
    service = _service(tmp_path, port)
    cache = _cache(run_root)
    arguments = (
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
    )
    service.analyze_all(*arguments, cache=cache, is_cancel_requested=lambda: False)
    candidate = run_root / frame_batch.frame_sets[0].candidates[0].relative_path
    candidate.chmod(0o644)

    with pytest.raises(VideoDemoError) as raised:
        service.analyze_all(*arguments, cache=cache, is_cancel_requested=lambda: False)

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert len(port.requests) == 1


def test_chapter_vision_degrades_only_temporary_failed_chapter(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    port = _VisionPort(
        VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "视觉模型暂时失败"),
    )

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.observations == result.keyframe_evidence == result.evidence == ()
    assert result.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert result.status == "PARTIAL_SUCCEEDED"
    assert result.warnings == (f"CHAPTER_VLM_DEGRADED:{chapter.chapter_id}",)
    assert result.metrics["vlm_fallback_chapters"] == 1


def test_oversized_candidate_is_rejected_during_batch_file_verification(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, payload = _fixture(tmp_path)
    first = frame_batch.frame_sets[0].candidates[0]
    large_payload = b"\xff\xd8\xff" + b"x" * 1_100 + b"\xff\xd9"
    large_digest = hashlib.sha256(large_payload).hexdigest()
    large_path = run_root / f"visual/candidates/{large_digest}.jpg"
    large_path.write_bytes(large_payload)
    large_path.chmod(0o600)
    oversized = _updated_frame(
        first,
        timestamp_ms=1_400,
        sha256=large_digest,
        size_bytes=len(large_payload),
        relative_path=f"visual/candidates/{large_digest}.jpg",
    )
    batch = _replace_frame_batch(
        frame_batch,
        chapter=chapter,
        candidates=(oversized, first),
    )
    port = _VisionPort(_response())

    with pytest.raises(VideoDemoError) as raised:
        _service(tmp_path, port).analyze_all(
            (chapter,),
            batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert len(large_payload) > 1_024 > len(payload)
    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert port.requests == []


def test_only_target_covering_candidate_over_image_limit_fails_batch_verification(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    original = frame_batch.frame_sets[0].candidates[0]
    large_payload = b"\xff\xd8\xff" + b"y" * 1_100 + b"\xff\xd9"
    large_digest = hashlib.sha256(large_payload).hexdigest()
    large_path = run_root / f"visual/candidates/{large_digest}.jpg"
    large_path.write_bytes(large_payload)
    large_path.chmod(0o600)
    oversized = _updated_frame(
        original,
        sha256=large_digest,
        size_bytes=len(large_payload),
        relative_path=f"visual/candidates/{large_digest}.jpg",
    )
    batch = _replace_frame_batch(frame_batch, chapter=chapter, candidates=(oversized,))
    port = _VisionPort(_response())

    with pytest.raises(VideoDemoError) as raised:
        _service(tmp_path, port).analyze_all(
            (chapter,),
            batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert port.requests == []
    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_visual_disabled_short_circuits_before_lease_and_candidate_read(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    candidate_path = run_root / frame_batch.frame_sets[0].candidates[0].relative_path
    candidate_path.chmod(0o644)
    port = _VisionPort(_response())

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(max_visuals_per_chapter=0),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert port.requests == []
    assert result.chapter_status == ((chapter.chapter_id, "DISABLED"),)
    assert result.status == "SUCCEEDED"
    assert result.metrics == {
        "vlm_logical_analyses": 0,
        "vlm_provider_attempts": 0,
        "vlm_structure_repairs": 0,
        "vlm_cache_hits": 0,
        "vlm_no_value_chapters": 0,
        "vlm_fallback_chapters": 0,
        "visual_published_budget_degraded_chapters": 0,
    }
    assert not (run_root / "visual/keyframes").exists()


@pytest.mark.parametrize("observations", [(), _response().observations])
def test_degraded_frame_chapter_with_candidates_stays_degraded_without_copying_frame_events(
    tmp_path: Path,
    observations: tuple[ChapterVisualObservation, ...],
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    degraded_batch = ChapterFrameSearchBatch(
        asset_sha256=frame_batch.asset_sha256,
        allowed_run_root=run_root,
        frame_tolerance_ms=frame_batch.frame_tolerance_ms,
        frame_sets=frame_batch.frame_sets,
        chapter_status=((chapter.chapter_id, "DEGRADED"),),
        warnings=(f"CHAPTER_FRAME_DEGRADED:{chapter.chapter_id}",),
        status="PARTIAL_SUCCEEDED",
        metrics={"visual_frame_degraded_chapters": 1},
    )

    port = _VisionPort(ChapterVisionResponse(observations=observations))
    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        degraded_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert result.status == "PARTIAL_SUCCEEDED"
    assert result.warnings == ()
    assert "visual_frame_degraded_chapters" not in result.metrics
    assert result.metrics["vlm_no_value_chapters"] == (0 if observations else 1)


def test_observation_time_uses_frame_tolerance_and_clamps_to_chapter(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)

    result = _service(tmp_path, _VisionPort(_response())).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert (result.observations[0].start_ms, result.observations[0].end_ms) == (1_450, 1_551)


def test_nonempty_no_candidate_batch_is_rejected_before_reading_candidate(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    candidate_path = run_root / frame_batch.frame_sets[0].candidates[0].relative_path
    candidate_path.chmod(0o644)
    inconsistent = frame_batch.model_copy(
        update={"chapter_status": ((chapter.chapter_id, "NO_CANDIDATE"),)},
    )

    with pytest.raises(VideoDemoError) as raised:
        _service(tmp_path, _VisionPort(_response())).analyze_all(
            (chapter,),
            inconsistent,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


@pytest.mark.parametrize(
    "update",
    [
        {"timestamp_ms": 10_000},
        {"target_ids": ("target_unknown",)},
    ],
)
def test_candidate_must_belong_to_its_chapter_before_file_read(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    candidate = frame_batch.frame_sets[0].candidates[0].model_copy(update=update)
    batch = _replace_frame_batch(frame_batch, chapter=chapter, candidates=(candidate,))

    with pytest.raises(VideoDemoError) as raised:
        _service(tmp_path, _VisionPort(_response())).analyze_all(
            (chapter,),
            batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_same_sha_with_distinct_frame_ids_publishes_one_file_and_two_evidence_items(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    first = frame_batch.frame_sets[0].candidates[0]
    second = _updated_frame(first, timestamp_ms=1_700)
    batch = _replace_frame_batch(
        frame_batch,
        chapter=chapter,
        candidates=(first, second),
    )
    response = ChapterVisionResponse(
        observations=(
            _response().observations[0].model_copy(
                update={"selected_frame_ids": (first.frame_id, second.frame_id)},
            ),
        ),
    )

    result = _service(tmp_path, _VisionPort(response)).analyze_all(
        (chapter,),
        batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert len(result.keyframe_evidence) == 2
    assert len({item.relative_path for item in result.keyframe_evidence}) == 1
    assert len(tuple((run_root / "visual/keyframes").iterdir())) == 1


class _RepairingVisionPort(_VisionPort):
    def __init__(self, response: ChapterVisionResponse) -> None:
        super().__init__(response)
        self.invalid = InvalidModelResponse(
            content_sha256="b" * 64,
            validation_errors=("observations.0.selected_frame_ids:unknown_reference",),
        )
        self.repair_requests: list[ChapterVisionRepairRequest] = []

    def analyze_chapter(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        self.requests.append(request)
        if on_provider_attempt is not None:
            on_provider_attempt()
        raise ModelResponseValidationError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "主响应结构非法",
            self.invalid,
        )

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        self.repair_requests.append(request)
        if on_provider_attempt is not None:
            on_provider_attempt()
        return self._response  # type: ignore[return-value]


def test_structure_repair_reuses_original_ordered_frames_once_and_counts_real_attempts(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    first = frame_batch.frame_sets[0].candidates[0]
    second = _updated_frame(first, timestamp_ms=1_700)
    batch = _replace_frame_batch(
        frame_batch,
        chapter=chapter,
        candidates=(second, first),
    )
    port = _RepairingVisionPort(_response())

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert len(port.repair_requests) == 1
    assert port.repair_requests[0].invalid_response is port.invalid
    assert port.repair_requests[0].request.frames == port.requests[0].frames
    assert tuple(frame.frame_id for frame in port.requests[0].frames) == (
        first.frame_id,
        second.frame_id,
    )
    assert result.metrics["vlm_logical_analyses"] == 1
    assert result.metrics["vlm_provider_attempts"] == 2
    assert result.metrics["vlm_structure_repairs"] == 1


def test_reverse_time_frame_relation_is_rejected_as_local_invalid_response(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    first = frame_batch.frame_sets[0].candidates[0]
    second = _updated_frame(first, timestamp_ms=1_700)
    batch = _replace_frame_batch(frame_batch, chapter=chapter, candidates=(first, second))
    response = ChapterVisionResponse(
        observations=(
            _response().observations[0].model_copy(
                update={
                    "selected_frame_ids": (first.frame_id, second.frame_id),
                    "frame_relations": (
                        VisualFrameRelationDraft(
                            relation_type="BEFORE_AFTER",
                            from_frame_id=second.frame_id,
                            to_frame_id=first.frame_id,
                            description="错误地把后帧声明为前帧",
                        ),
                    ),
                },
            ),
        ),
    )

    class RelationshipRepairPort(_VisionPort):
        def __init__(self) -> None:
            super().__init__(response)
            self.repair_requests: list[ChapterVisionRepairRequest] = []

        def repair_chapter(
            self,
            request: ChapterVisionRepairRequest,
            *,
            allowed_run_root: Path,
            on_provider_attempt: Callable[[], None] | None = None,
        ) -> ChapterVisionResponse:
            del allowed_run_root
            self.repair_requests.append(request)
            if on_provider_attempt is not None:
                on_provider_attempt()
            return _response()

    port = RelationshipRepairPort()
    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert result.metrics["vlm_structure_repairs"] == 1
    assert result.metrics["vlm_fallback_chapters"] == 0
    assert len(port.repair_requests) == 1
    assert port.repair_requests[0].request.frames == port.requests[0].frames
    assert "frame_relations" in port.repair_requests[0].invalid_response.validation_errors[0]


def test_local_type_error_from_port_is_not_masked_as_chapter_fallback(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)

    class BrokenPort(_VisionPort):
        def analyze_chapter(
            self,
            request: ChapterVisionRequest,
            *,
            allowed_run_root: Path,
            on_provider_attempt: Callable[[], None] | None = None,
        ) -> ChapterVisionResponse:
            del request, allowed_run_root, on_provider_attempt
            raise TypeError("模拟本地程序错误")

    with pytest.raises(TypeError, match="模拟本地程序错误"):
        _service(tmp_path, BrokenPort(_response())).analyze_all(
            (chapter,),
            frame_batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )


class _SemanticRepairPort(_VisionPort):
    def __init__(
        self,
        main_response: ChapterVisionResponse,
        repair_response: ChapterVisionResponse,
    ) -> None:
        super().__init__(main_response)
        self._repair_response = repair_response
        self.repair_requests: list[ChapterVisionRepairRequest] = []

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        del allowed_run_root
        self.repair_requests.append(request)
        if on_provider_attempt is not None:
            on_provider_attempt()
        return self._repair_response


def test_duplicate_equivalent_observations_are_rejected_before_publication(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    duplicate = _response().observations[0]
    response = ChapterVisionResponse(observations=(duplicate, duplicate))
    port = _SemanticRepairPort(response, _response())

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)
    assert result.metrics["vlm_structure_repairs"] == 1
    assert result.metrics["vlm_provider_attempts"] == 2
    assert len(port.repair_requests) == 1
    assert port.repair_requests[0].request.frames == port.requests[0].frames


def test_duplicate_equivalent_observations_degrade_when_repair_is_still_duplicate(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    duplicate = _response().observations[0]
    response = ChapterVisionResponse(observations=(duplicate, duplicate))
    port = _SemanticRepairPort(response, response)

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.observations == ()
    assert result.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert result.metrics["vlm_structure_repairs"] == 1
    assert result.metrics["vlm_fallback_chapters"] == 1
    assert not (run_root / "visual/keyframes").exists()


def test_published_budget_removes_complete_observation_and_degrades_only_its_chapter(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, payload = _fixture(tmp_path)
    port = _VisionPort(_response())
    service = ChapterVisionService(
        port,
        _identity(),
        runtime_root=tmp_path,
        concurrency=1,
        max_image_bytes=1024,
        max_request_image_bytes=4096,
        max_encoded_request_bytes=64 * 1024,
        max_published_keyframe_bytes=len(payload) - 1,
        max_published_keyframe_files=10,
        invocation_wait_timeout_seconds=2,
        candidate_lock_timeout_seconds=2,
    )

    result = service.analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.observations == result.keyframe_evidence == result.evidence == ()
    assert result.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert result.warnings == (f"VISUAL_PUBLISHED_BUDGET_DEGRADED:{chapter.chapter_id}",)
    assert result.metrics["visual_published_budget_degraded_chapters"] == 1
    assert not (run_root / "visual/keyframes").exists()


def test_macos_publish_budget_counts_staging_and_formal_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root, chapter, frame_batch, speech, payload = _fixture(tmp_path)
    monkeypatch.setattr(keyframe_artifacts.platform, "system", lambda: "Darwin")
    service = ChapterVisionService(
        _VisionPort(_response()),
        _identity(),
        runtime_root=tmp_path,
        concurrency=1,
        max_image_bytes=1024,
        max_request_image_bytes=4096,
        max_encoded_request_bytes=64 * 1024,
        max_published_keyframe_bytes=len(payload),
        max_published_keyframe_files=1,
        invocation_wait_timeout_seconds=2,
        candidate_lock_timeout_seconds=2,
    )

    result = service.analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert result.observations == result.keyframe_evidence == result.evidence == ()
    assert result.chapter_status == ((chapter.chapter_id, "DEGRADED"),)
    assert result.warnings == (f"VISUAL_PUBLISHED_BUDGET_DEGRADED:{chapter.chapter_id}",)
    assert not (run_root / "visual/keyframes").exists()


def test_keyframe_snapshot_rejects_dangling_directory_symlink(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    keyframes = run_root / "visual/keyframes"
    keyframes.symlink_to(run_root / "missing-keyframes", target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        _service(tmp_path, _VisionPort(ChapterVisionResponse(observations=()))).analyze_all(
            (chapter,),
            frame_batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_publish_keeps_verified_file_from_current_batch_on_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root, chapter, frame_batch, speech, payload = _fixture(tmp_path)
    first = frame_batch.frame_sets[0].candidates[0]
    second_payload = b"\xff\xd8\xffsecond-publish\xff\xd9"
    second_digest = hashlib.sha256(second_payload).hexdigest()
    second_path = run_root / f"visual/candidates/{second_digest}.jpg"
    second_path.write_bytes(second_payload)
    second_path.chmod(0o600)
    second = _updated_frame(
        first,
        timestamp_ms=1_700,
        sha256=second_digest,
        size_bytes=len(second_payload),
        relative_path=f"visual/candidates/{second_digest}.jpg",
    )
    batch = _replace_frame_batch(frame_batch, chapter=chapter, candidates=(first, second))
    response = ChapterVisionResponse(
        observations=(
            _response().observations[0].model_copy(
                update={"selected_frame_ids": (first.frame_id, second.frame_id)},
            ),
        ),
    )
    real_write = keyframe_artifacts.KeyframeArtifactSession._write_private_jpeg
    writes = 0

    def fail_second(
        session: keyframe_artifacts.KeyframeArtifactSession,
        name: str,
        content: bytes,
    ) -> tuple[int, int, int]:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "模拟第二张复制失败")
        return real_write(session, name, content)

    monkeypatch.setattr(
        keyframe_artifacts.KeyframeArtifactSession,
        "_write_private_jpeg",
        fail_second,
    )

    with pytest.raises(VideoDemoError):
        _service(tmp_path, _VisionPort(response)).analyze_all(
            (chapter,),
            batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=lambda: False,
        )

    assert payload != second_payload
    first_leaf = run_root / "visual/keyframes" / f"{first.sha256}.jpg"
    assert first_leaf.read_bytes() == payload
    expected = {first.sha256: len(payload)}
    if platform.system() == "Darwin":
        staging = next((run_root / "visual/.keyframe-staging").iterdir())
        expected[f"pending:{staging.name}"] = len(payload)
    with keyframe_artifacts.KeyframeArtifactSession(
        run_root,
        max_files=10,
        max_bytes=10_000,
    ) as session:
        assert session.snapshot() == expected


def test_private_jpeg_partial_write_failure_only_leaves_budgeted_staging_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_demo.visual import keyframe_artifacts

    run_root, _chapter, frame_batch, _speech, _payload = _fixture(tmp_path)
    frame = frame_batch.frame_sets[0].candidates[0]
    (run_root / "visual").chmod(0o700)
    real_open = keyframe_artifacts.os.open
    real_write = keyframe_artifacts.os.write
    created_descriptor = -1
    calls = 0

    def remember_exclusive_leaf(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal created_descriptor
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        temporary_flag = getattr(os, "O_TMPFILE", 0)
        if (
            flags & os.O_EXCL and os.fspath(path).endswith(".pending")
        ) or (temporary_flag and flags & temporary_flag == temporary_flag):
            created_descriptor = descriptor
        return descriptor

    def fail_after_first_byte(descriptor: int, payload: bytes | memoryview) -> int:
        nonlocal calls
        if descriptor != created_descriptor:
            return real_write(descriptor, payload)
        calls += 1
        if calls == 1:
            return real_write(descriptor, bytes(payload[:1]))
        raise OSError("模拟写入中途失败")

    monkeypatch.setattr(keyframe_artifacts.os, "open", remember_exclusive_leaf)
    monkeypatch.setattr(keyframe_artifacts.os, "write", fail_after_first_byte)

    with (
        keyframe_artifacts.KeyframeArtifactSession(
            run_root,
            max_files=10,
            max_bytes=10_000,
        ) as session,
        pytest.raises(VideoDemoError) as raised,
    ):
        session._open_keyframe_directory(create=True)
        session._open_staging_directory(create=True)
        session._write_private_jpeg(f"{frame.sha256}.jpg", _payload)

    assert calls == 2, str(raised.value)
    assert raised.value.code == ErrorCode.VISUAL_RESULT_INVALID
    assert tuple((run_root / "visual/keyframes").iterdir()) == ()
    staging = tuple((run_root / "visual/.keyframe-staging").iterdir())
    assert len(staging) == (1 if platform.system() == "Darwin" else 0)
    if staging:
        assert staging[0].read_bytes() == b"\xff"
    with keyframe_artifacts.KeyframeArtifactSession(
        run_root,
        max_files=10,
        max_bytes=10_000,
    ) as session:
        existing = session.snapshot()
    assert len(existing) == len(staging)
    assert sum(existing.values()) == len(staging)


class _ConcurrentVisionPort(_VisionPort):
    def __init__(self) -> None:
        super().__init__(ChapterVisionResponse(observations=()))
        self._lock = threading.Lock()
        self.current = 0
        self.maximum = 0
        self.completed: list[str] = []

    def analyze_chapter(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        if on_provider_attempt is not None:
            on_provider_attempt()
        with self._lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)
        delays = {"chapter_001": 0.06, "chapter_002": 0.03, "chapter_003": 0.01}
        time.sleep(delays[request.chapter_id])
        with self._lock:
            self.current -= 1
            self.completed.append(request.chapter_id)
        return ChapterVisionResponse(observations=())


def test_bounded_concurrency_preserves_chapter_order(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    chapter_speeches = tuple(
        speech.model_copy(
            update={
                "evidence_id": f"asr_{index:03d}",
                "start_ms": (index - 1) * 10_000 + 1_000,
                "end_ms": (index - 1) * 10_000 + 2_000,
            },
        )
        for index in range(1, 4)
    )
    chapters = tuple(
        chapter.model_copy(
            update={
                "chapter_id": f"chapter_{index:03d}",
                "start_ms": (index - 1) * 10_000,
                "end_ms": index * 10_000,
                "semantic_targets": (
                    chapter.semantic_targets[0].model_copy(
                        update={
                            "anchor_evidence_refs": (chapter_speeches[index - 1].evidence_id,),
                        },
                    ),
                ),
            },
        )
        for index in range(1, 4)
    )
    base_frame = frame_batch.frame_sets[0].candidates[0]
    frames = tuple(
        _updated_frame(
            base_frame,
            timestamp_ms=(index - 1) * 10_000 + 1_500,
        )
        for index in range(1, 4)
    )
    batch = ChapterFrameSearchBatch(
        asset_sha256=frame_batch.asset_sha256,
        allowed_run_root=run_root,
        frame_tolerance_ms=50,
        frame_sets=tuple(
            ChapterFrameSet(chapter_id=item.chapter_id, candidates=(frame,))
            for item, frame in zip(chapters, frames, strict=True)
        ),
        chapter_status=tuple((item.chapter_id, "SUCCEEDED") for item in chapters),
        metrics={},
    )
    port = _ConcurrentVisionPort()

    result = _service(tmp_path, port).analyze_all(
        chapters,
        batch,
        chapter_speeches,
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert port.maximum == 2
    assert port.completed != [chapter.chapter_id for chapter in chapters]
    assert result.chapter_status == tuple((item.chapter_id, "NO_VALUE") for item in chapters)


def test_chapter_vision_limits_candidates_to_selected_frame_budget(
    tmp_path: Path,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    base_frame = frame_batch.frame_sets[0].candidates[0]
    candidates = tuple(
        _updated_frame(base_frame, timestamp_ms=1_500 + index * 1_000)
        for index in range(3)
    )
    batch = frame_batch.model_copy(
        update={
            "frame_sets": (
                ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=candidates),
            ),
        },
    )
    port = _VisionPort(_response())

    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=_cache(run_root),
        is_cancel_requested=lambda: False,
    )

    assert len(port.requests) == 1
    assert len(port.requests[0].frames) == 2
    assert result.chapter_status == ((chapter.chapter_id, "SUCCEEDED"),)


def test_cancellation_stops_submitting_and_waits_for_inflight_chapters(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    chapter_speeches = tuple(
        speech.model_copy(
            update={
                "evidence_id": f"asr_{index:03d}",
                "start_ms": (index - 1) * 10_000 + 1_000,
                "end_ms": (index - 1) * 10_000 + 2_000,
            },
        )
        for index in range(1, 4)
    )
    chapters = tuple(
        chapter.model_copy(
            update={
                "chapter_id": f"chapter_{index:03d}",
                "start_ms": (index - 1) * 10_000,
                "end_ms": index * 10_000,
                "semantic_targets": (
                    chapter.semantic_targets[0].model_copy(
                        update={
                            "anchor_evidence_refs": (chapter_speeches[index - 1].evidence_id,),
                        },
                    ),
                ),
            },
        )
        for index in range(1, 4)
    )
    base_frame = frame_batch.frame_sets[0].candidates[0]
    frames = tuple(
        _updated_frame(
            base_frame,
            timestamp_ms=(index - 1) * 10_000 + 1_500,
        )
        for index in range(1, 4)
    )
    batch = ChapterFrameSearchBatch(
        asset_sha256=frame_batch.asset_sha256,
        allowed_run_root=run_root,
        frame_tolerance_ms=50,
        frame_sets=tuple(
            ChapterFrameSet(chapter_id=item.chapter_id, candidates=(frame,))
            for item, frame in zip(chapters, frames, strict=True)
        ),
        chapter_status=tuple((item.chapter_id, "SUCCEEDED") for item in chapters),
        metrics={},
    )
    started = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    entered: list[str] = []
    lock = threading.Lock()

    class BlockingPort(_VisionPort):
        def analyze_chapter(
            self,
            request: ChapterVisionRequest,
            *,
            allowed_run_root: Path,
            on_provider_attempt: Callable[[], None] | None = None,
        ) -> ChapterVisionResponse:
            with lock:
                entered.append(request.chapter_id)
                if len(entered) == 2:
                    started.set()
            assert release.wait(timeout=2)
            return ChapterVisionResponse(observations=())

    service = _service(tmp_path, BlockingPort(ChapterVisionResponse(observations=())))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            service.analyze_all,
            chapters,
            batch,
            chapter_speeches,
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=cancelled.is_set,
        )
        assert started.wait(timeout=2)
        cancelled.set()
        time.sleep(0.08)
        assert not future.done()
        release.set()
        with pytest.raises(VideoDemoError) as raised:
            future.result(timeout=2)

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert set(entered) == {"chapter_001", "chapter_002"}
    assert "chapter_003" not in entered


def test_cancellation_returns_after_finite_wait_but_keeps_lease_until_port_finishes(
    tmp_path: Path,
) -> None:
    from video_demo.visual.candidate_artifacts import CandidateDirectoryLease

    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    started = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    class StuckPort(_VisionPort):
        def analyze_chapter(
            self,
            request: ChapterVisionRequest,
            *,
            allowed_run_root: Path,
            on_provider_attempt: Callable[[], None] | None = None,
        ) -> ChapterVisionResponse:
            del request, allowed_run_root, on_provider_attempt
            started.set()
            release.wait()
            return ChapterVisionResponse(observations=())

    service = _service(
        tmp_path,
        StuckPort(ChapterVisionResponse(observations=())),
        invocation_wait_timeout_seconds=0.05,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            service.analyze_all,
            (chapter,),
            frame_batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=cancelled.is_set,
        )
        assert started.wait(timeout=1)
        cancelled.set()
        with pytest.raises(VideoDemoError) as raised:
            future.result(timeout=0.5)
        assert raised.value.code == ErrorCode.JOB_CANCELLED

        held_lease = CandidateDirectoryLease.from_allowed_run_root(
            runtime_root=tmp_path,
            allowed_run_root=run_root,
            mode="EXCLUSIVE",
            wait_timeout_seconds=0,
        )
        with pytest.raises(VideoDemoError) as held:
            held_lease.acquire()
        assert held.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE

        release.set()
        lease = CandidateDirectoryLease.from_allowed_run_root(
            runtime_root=tmp_path,
            allowed_run_root=run_root,
            mode="EXCLUSIVE",
            wait_timeout_seconds=1,
        )
        with lease:
            assert lease.is_acquired
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_cleanup_handoff_start_failure_happens_before_port_and_returns_promptly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, chapter, frame_batch, speech, _payload = _fixture(tmp_path)
    port_started = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()
    real_thread_start = threading.Thread.start

    class StuckPort(_VisionPort):
        def analyze_chapter(
            self,
            request: ChapterVisionRequest,
            *,
            allowed_run_root: Path,
            on_provider_attempt: Callable[[], None] | None = None,
        ) -> ChapterVisionResponse:
            del request, allowed_run_root, on_provider_attempt
            port_started.set()
            assert release.wait(timeout=2)
            return ChapterVisionResponse(observations=())

    def fail_cleanup_handoff_start(thread: threading.Thread) -> None:
        if thread.name == "chapter-vision-deferred-cleanup":
            raise RuntimeError("模拟收尾交接线程启动失败")
        real_thread_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_cleanup_handoff_start)
    service = _service(
        tmp_path,
        StuckPort(ChapterVisionResponse(observations=())),
        invocation_wait_timeout_seconds=0.05,
    )
    release_timer = threading.Timer(0.8, release.set)
    release_timer.start()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            service.analyze_all,
            (chapter,),
            frame_batch,
            (speech,),
            DocumentGenerationConfig(),
            cache=_cache(run_root),
            is_cancel_requested=cancelled.is_set,
        )
        port_started.wait(timeout=0.1)
        cancelled.set()
        with pytest.raises(RuntimeError, match="收尾交接线程启动失败"):
            future.result(timeout=0.5)
    finally:
        release.set()
        release_timer.cancel()
        executor.shutdown(wait=True)

    assert not port_started.is_set()
