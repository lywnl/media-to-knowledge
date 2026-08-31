from __future__ import annotations

from pathlib import Path
from typing import cast

from video_demo.application.chapter_planning import ChapterPlanner
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import BaseSegment, ChapterDraft, VisualTargetDraft
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode
from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanningResponse,
    DocumentTextPort,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_ASSET_SHA256 = "a" * 64


def _speech(evidence_id: str, start_ms: int, end_ms: int) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="讲解设置步骤",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


class _TextPort:
    def __init__(self, response: ChapterPlanningResponse | BaseException) -> None:
        self.response = response
        self.requests: list[ChapterPlanningRequest] = []

    def plan_chapters(
        self, request: ChapterPlanningRequest, **_kwargs: object
    ) -> ChapterPlanningResponse:
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def repair_chapter_plan(self, request: object, **_kwargs: object) -> ChapterPlanningResponse:
        del request
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _identity() -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation="chapter_planning",
        provider_config_fingerprint="b" * 64,
        model_id="text-model",
        generation_config=(("temperature", "0"),),
        main_response_schema_name="chapter_planning_v1",
        main_prompt_version="chapter-planner-v1",
        repair_response_schema_name="chapter_planning_repair_v1",
        repair_prompt_version="chapter-planner-repair-v1",
    )


def _planner(port: _TextPort) -> ChapterPlanner:
    return ChapterPlanner(
        cast(DocumentTextPort, port),
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=2,
        compact_planning=True,
    )


def _segments() -> tuple[BaseSegment, ...]:
    return tuple(
        BaseSegment(
            segment_id=f"segment_{i}",
            start_ms=i * 60_000,
            end_ms=(i + 1) * 60_000,
            evidence_refs=(f"asr_{i}",),
            transcript_source="ASR",
        )
        for i in range(2)
    )


def _plan(
    planner: ChapterPlanner,
    tmp_path: Path,
    segments: tuple[BaseSegment, ...],
    transcript: tuple[SpeechSegment, ...],
):
    return planner.plan(
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_048_576, max_run_bytes=4_194_304),
        asset_sha256=_ASSET_SHA256,
        title_hint="测试视频",
        duration_ms=segments[-1].end_ms,
        segments=segments,
        transcript_evidence=transcript,
        document_config=DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )


def test_planner_materializes_midpoint_base_target_and_no_scene_fields(tmp_path: Path) -> None:
    segments = _segments()
    transcript = tuple(_speech(f"asr_{i}", i * 60_000, i * 60_000 + 10_000) for i in range(2))
    response = ChapterPlanningResponse(
        chapter_drafts=(
            ChapterDraft(
                segment_refs=("segment_0", "segment_1"),
                title_hint="设置流程",
                visual_mode="SINGLE",
                semantic_targets=(
                    VisualTargetDraft(query_zh="关键参数", anchor_evidence_refs=("asr_0",)),
                ),
            ),
        )
    )
    batch = _plan(_planner(_TextPort(response)), tmp_path, segments, transcript)
    chapter = batch.plans[0]
    assert chapter.base_coverage_targets[0].sample_timestamps_ms == (60_000,)
    assert chapter.base_coverage_targets[0].anchor_evidence_refs == ()
    assert "scene_refs" not in chapter.model_dump(mode="json")


def test_planner_falls_back_when_model_response_is_invalid(tmp_path: Path) -> None:
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "响应非法",
        InvalidModelResponse(content_sha256="c" * 64, validation_errors=("invalid",)),
    )
    segments = _segments()
    transcript = tuple(_speech(f"asr_{i}", i * 60_000, i * 60_000 + 10_000) for i in range(2))
    batch = _plan(_planner(_TextPort(invalid)), tmp_path, segments, transcript)
    assert batch.status == "PARTIAL_SUCCEEDED"
    assert batch.warnings
    assert batch.metrics["chapter_planner_fallback_chapters"] >= 1
