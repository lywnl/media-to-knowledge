from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

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
from video_demo.domain.evidence import SpeechSegment
from video_demo.integrations.document_port import ChapterVisionPort, ChapterVisionResponse
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity


def _fixture(tmp_path: Path) -> tuple[Path, ChapterPlan, ChapterFrameSearchBatch, SpeechSegment]:
    run_root = tmp_path / "runs/scope/run"
    candidate_root = run_root / "visual/candidates"
    candidate_root.mkdir(parents=True)
    payload = b"\xff\xd8\xffvalid-jpeg\xff\xd9"
    digest = hashlib.sha256(payload).hexdigest()
    (candidate_root / f"{digest}.jpg").write_bytes(payload)
    candidate_root.chmod(0o700)
    (candidate_root / f"{digest}.jpg").chmod(0o600)
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=1_000,
        end_ms=2_000,
        text="画面显示参数",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    target = VisualSearchTarget(
        target_id="target_001",
        purpose="SEMANTIC",
        query_zh="读取画面参数",
        anchor_evidence_refs=(speech.evidence_id,),
    )
    chapter = ChapterPlan(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=10_000,
        segment_refs=("segment_001",),
        title_hint="参数",
        visual_mode="SINGLE",
        semantic_targets=(target,),
        base_coverage_targets=(),
    )
    frame = FrameCandidateArtifact(
        frame_id=frame_candidate_id("a" * 64, 1_500, digest),
        timestamp_ms=1_500,
        sha256=digest,
        size_bytes=len(payload),
        relative_path=f"visual/candidates/{digest}.jpg",
        target_ids=(target.target_id,),
    )
    batch = ChapterFrameSearchBatch(
        asset_sha256="a" * 64,
        allowed_run_root=run_root,
        frame_sets=(ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=(frame,)),),
        chapter_status=((chapter.chapter_id, "SUCCEEDED"),),
        metrics={},
    )
    return run_root, chapter, batch, speech


class _VisionPort:
    def __init__(self, response: ChapterVisionResponse) -> None:
        self.response = response
        self.requests: list[object] = []

    def analyze_chapter(self, request: object, **_kwargs: object) -> ChapterVisionResponse:
        self.requests.append(request)
        return self.response

    def repair_chapter_vision(self, request: object, **_kwargs: object) -> ChapterVisionResponse:
        del request
        return self.response


def _service(tmp_path: Path, port: _VisionPort) -> ChapterVisionService:
    identity = ModelInvocationIdentity(
        logical_operation="chapter_vision",
        provider_config_fingerprint="b" * 64,
        model_id="vlm",
        generation_config=(),
        main_response_schema_name="chapter_vlm_v2",
        main_prompt_version="chapter-vlm-v1",
        repair_response_schema_name="chapter_vlm_repair_v2",
        repair_prompt_version="chapter-vlm-repair-v1",
    )
    return ChapterVisionService(
        cast(ChapterVisionPort, port),
        identity,
        runtime_root=tmp_path,
        concurrency=1,
        max_image_bytes=1_024,
        max_request_image_bytes=4_096,
        max_encoded_request_bytes=64_000,
        max_published_keyframe_bytes=4_096,
        max_published_keyframe_files=10,
        invocation_wait_timeout_seconds=2,
        candidate_lock_timeout_seconds=2,
    )


def test_successful_vlm_analysis_publishes_keyframe_evidence(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech = _fixture(tmp_path)
    frame_id = frame_batch.frame_sets[0].candidates[0].frame_id
    response = ChapterVisionResponse(
        observations=(
            {
                "target_ids": ["target_001"],
                "selected_frame_ids": [frame_id],
                "transcript_evidence_refs": [speech.evidence_id],
                "visual_type": "TEXT",
                "caption": "画面参数",
                "content_blocks": [],
                "visual_facts": [],
                "frame_relations": [],
                "relation_to_transcript": "COMPLEMENTARY",
                "certainty": 0.9,
            },
        )
    )
    result = _service(tmp_path, _VisionPort(response)).analyze_all(
        (chapter,),
        frame_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=DocumentModelCache(run_root, max_entry_bytes=64_000, max_run_bytes=256_000),
        is_cancel_requested=lambda: False,
    )
    assert result.status == "SUCCEEDED"
    assert result.observations[0].keyframe_refs
    assert len(result.keyframe_evidence) == 1


def test_no_candidate_chapter_skips_vlm_and_keeps_text_path(tmp_path: Path) -> None:
    run_root, chapter, frame_batch, speech = _fixture(tmp_path)
    empty_batch = frame_batch.model_copy(
        update={
            "frame_sets": (ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=()),),
            "chapter_status": ((chapter.chapter_id, "NO_CANDIDATE"),),
        }
    )
    port = _VisionPort(ChapterVisionResponse(observations=()))
    result = _service(tmp_path, port).analyze_all(
        (chapter,),
        empty_batch,
        (speech,),
        DocumentGenerationConfig(),
        cache=DocumentModelCache(run_root, max_entry_bytes=64_000, max_run_bytes=256_000),
        is_cancel_requested=lambda: False,
    )
    assert port.requests == []
    assert result.chapter_status == ((chapter.chapter_id, "NO_CANDIDATE"),)
