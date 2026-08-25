from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

from video_demo.application.chapter_planning import ChapterPlanner, ChapterPlanningBatch
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import BaseSegment, ChapterDraft, VisualTargetDraft
from video_demo.domain.evidence import SceneBoundary, SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanningResponse,
    DocumentTextPort,
    InvalidModelResponse,
    ModelResponseValidationError,
)
from video_demo.integrations.document_prompts import prompt_for_planning
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity

_ASSET_SHA256 = "a" * 64


def _scene(index: int, start_ms: int, end_ms: int) -> SceneBoundary:
    return SceneBoundary(
        evidence_id=f"scene_{index:03d}",
        start_ms=start_ms,
        end_ms=end_ms,
        transition="candidate" if index == 0 else "hard_cut",
        score=0.9,
    )


def _speech(evidence_id: str, start_ms: int, end_ms: int, text: str = "语音") -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


class _PlanningTextPort:
    def __init__(
        self,
        main: Callable[[ChapterPlanningRequest], ChapterPlanningResponse | BaseException],
        repair: Callable[[ChapterPlanningRequest], ChapterPlanningResponse | BaseException],
    ) -> None:
        self._main = main
        self._repair = repair
        self.main_requests: list[ChapterPlanningRequest] = []
        self.repair_requests: list[ChapterPlanningRequest] = []

    def plan_chapters(
        self,
        request: ChapterPlanningRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse:
        self.main_requests.append(request)
        if on_provider_attempt is not None:
            on_provider_attempt()
        return _raise_or_return(self._main(request))

    def repair_chapter_plan(
        self,
        request: object,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse:
        original = request.request  # type: ignore[attr-defined]
        self.repair_requests.append(original)
        if on_provider_attempt is not None:
            on_provider_attempt()
        return _raise_or_return(self._repair(original))


def _raise_or_return(
    value: ChapterPlanningResponse | BaseException,
) -> ChapterPlanningResponse:
    if isinstance(value, BaseException):
        raise value
    return value


def _planner(
    port: _PlanningTextPort,
    *,
    max_input_chars: int = 60_000,
    max_input_bytes: int = 1_048_576,
    max_chapters: int = 240,
    max_planning_batches: int = 64,
) -> ChapterPlanner:
    return ChapterPlanner(
        cast(DocumentTextPort, port),
        _planning_identity(),
        max_input_chars=max_input_chars,
        max_input_bytes=max_input_bytes,
        max_chapters=max_chapters,
        max_planning_batches=max_planning_batches,
        invocation_wait_timeout_seconds=2,
    )


def _planning_identity() -> ModelInvocationIdentity:
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_chapters": 241},
        {"max_planning_batches": 65},
    ],
)
def test_chapter_planner_rejects_budget_above_hard_contract(
    overrides: dict[str, int],
) -> None:
    port = _PlanningTextPort(
        lambda _request: AssertionError("构造失败后不得调用模型"),
        lambda _request: AssertionError("构造失败后不得调用修复模型"),
    )

    with pytest.raises(ValueError, match="预算"):
        _planner(port, **overrides)  # type: ignore[arg-type]


def _planning_fixture(
    count: int = 2,
    *,
    text: str = "讲解设置步骤",
    interval_ms: int = 60_000,
) -> tuple[tuple[BaseSegment, ...], tuple[SpeechSegment, ...], tuple[SceneBoundary, ...]]:
    segments: list[BaseSegment] = []
    transcript: list[SpeechSegment] = []
    for index in range(count):
        start_ms = index * interval_ms
        end_ms = (index + 1) * interval_ms
        evidence_id = f"asr_{index:03d}"
        transcript.append(
            _speech(evidence_id, start_ms, min(start_ms + 10_000, end_ms), text),
        )
        segments.append(
            BaseSegment(
                segment_id=f"segment_{index:03d}",
                start_ms=start_ms,
                end_ms=end_ms,
                evidence_refs=(evidence_id,),
                scene_refs=(f"scene_{index:03d}",),
                transcript_source="ASR",
            ),
        )
    scenes = tuple(
        _scene(index, index * interval_ms, (index + 1) * interval_ms)
        for index in range(count)
    )
    return tuple(segments), tuple(transcript), scenes


def _plan(
    planner: ChapterPlanner,
    tmp_path: Path,
    segments: tuple[BaseSegment, ...],
    transcript: tuple[SpeechSegment, ...],
    scenes: tuple[SceneBoundary, ...],
    document_config: DocumentGenerationConfig | None = None,
) -> ChapterPlanningBatch:
    return planner.plan(
        cache=DocumentModelCache(
            tmp_path,
            max_entry_bytes=1_048_576,
            max_run_bytes=4_194_304,
        ),
        asset_sha256=_ASSET_SHA256,
        title_hint="测试视频",
        duration_ms=segments[-1].end_ms,
        segments=segments,
        transcript_evidence=transcript,
        scenes=scenes,
        document_config=document_config or DocumentGenerationConfig(),
        is_cancel_requested=lambda: False,
    )


def test_chapter_planner_repairs_once_caches_result_and_owns_ids_and_time(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture()
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "响应非法",
        InvalidModelResponse(
            content_sha256="c" * 64,
            validation_errors=("chapter_drafts:invalid",),
        ),
    )
    repaired = ChapterPlanningResponse(
        chapter_drafts=(
            ChapterDraft(
                segment_refs=("segment_000", "segment_001"),
                title_hint="设置流程",
                visual_mode="SINGLE",
                semantic_targets=(
                    VisualTargetDraft(
                        query_zh="设置页面中的关键参数是什么",
                        anchor_evidence_refs=("asr_000",),
                    ),
                ),
            ),
        ),
    )
    port = _PlanningTextPort(lambda _request: invalid, lambda _request: repaired)
    planner = _planner(port)

    first = _plan(planner, tmp_path, segments, transcript, scenes)
    second = _plan(planner, tmp_path, segments, transcript, scenes)

    assert first.plans == second.plans
    chapter = first.plans[0]
    assert (chapter.start_ms, chapter.end_ms) == (0, 120_000)
    assert chapter.chapter_id.startswith("chapter_")
    assert chapter.semantic_targets[0].target_id.startswith("visual_target_")
    assert chapter.base_coverage_targets[0].scene_refs == ("scene_000", "scene_001")
    assert first.metrics == {
        "chapter_planner_logical_calls": 1,
        "chapter_planner_provider_attempts": 2,
        "chapter_planner_structure_repairs": 1,
        "chapter_planner_cache_hits": 0,
        "chapter_planner_fallback_chapters": 0,
    }
    assert second.metrics["chapter_planner_cache_hits"] == 1
    assert second.metrics["chapter_planner_provider_attempts"] == 0
    assert len(port.main_requests) == len(port.repair_requests) == 1


def test_chapter_planner_invalid_main_and_repair_fall_back_deterministically(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture()
    invalid_partition = ChapterPlanningResponse(
        chapter_drafts=(
            ChapterDraft(
                segment_refs=("segment_001",),
                title_hint="越界章节",
                visual_mode="NONE",
                semantic_targets=(),
            ),
        ),
    )
    port = _PlanningTextPort(
        lambda _request: invalid_partition,
        lambda _request: invalid_partition,
    )

    batch = _plan(_planner(port), tmp_path, segments, transcript, scenes)

    assert batch.status == "PARTIAL_SUCCEEDED"
    assert tuple(ref for plan in batch.plans for ref in plan.segment_refs) == (
        "segment_000",
        "segment_001",
    )
    assert batch.plans[0].semantic_targets == ()
    assert batch.metrics["chapter_planner_structure_repairs"] == 1
    assert batch.metrics["chapter_planner_fallback_chapters"] == len(batch.plans)
    assert batch.warnings == tuple(
        f"CHAPTER_PLANNING_FALLBACK:{plan.chapter_id}" for plan in batch.plans
    )


def test_chapter_planner_does_not_hide_unexpected_repair_programming_error(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture()
    invalid = ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "响应非法",
        InvalidModelResponse(
            content_sha256="c" * 64,
            validation_errors=("chapter_drafts:invalid",),
        ),
    )
    port = _PlanningTextPort(
        lambda _request: invalid,
        lambda _request: TypeError("修复适配器内部错误"),
    )

    with pytest.raises(TypeError, match="修复适配器内部错误"):
        _plan(_planner(port), tmp_path, segments, transcript, scenes)


def test_chapter_planner_textless_video_uses_successful_rule_path_without_model(
    tmp_path: Path,
) -> None:
    segments = tuple(
        BaseSegment(
            segment_id=f"segment_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(),
            scene_refs=("scene_000",),
            transcript_source="NONE",
        )
        for index in range(4)
    )
    scenes = (_scene(0, 0, 120_000),)
    port = _PlanningTextPort(
        lambda _request: AssertionError("无文本不得调用规划模型"),
        lambda _request: AssertionError("无文本不得调用修复模型"),
    )

    batch = _plan(_planner(port), tmp_path, segments, (), scenes)

    assert batch.status == "SUCCEEDED"
    assert batch.warnings == ()
    assert len(batch.plans) == 1
    assert batch.plans[0].visual_mode == "SINGLE"
    assert batch.plans[0].semantic_targets == ()
    assert batch.plans[0].base_coverage_targets
    assert batch.metrics["chapter_planner_logical_calls"] == 0
    assert not port.main_requests


def test_chapter_planner_applies_configured_granularity_to_rule_chapters(
    tmp_path: Path,
) -> None:
    segments = tuple(
        BaseSegment(
            segment_id=f"segment_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(),
            scene_refs=("scene_000",),
            transcript_source="NONE",
        )
        for index in range(10)
    )
    scenes = (_scene(0, 0, 300_000),)
    port = _PlanningTextPort(
        lambda _request: AssertionError("无文本不得调用规划模型"),
        lambda _request: AssertionError("无文本不得调用修复模型"),
    )

    counts = {
        granularity: len(
            _plan(
                _planner(port),
                tmp_path / granularity,
                segments,
                (),
                scenes,
                DocumentGenerationConfig(chapter_granularity=granularity),
            ).plans
        )
        for granularity in ("fine", "standard", "coarse")
    }

    assert counts == {"fine": 3, "standard": 2, "coarse": 1}
    assert not port.main_requests


def test_rule_chapters_choose_the_safe_boundary_nearest_target_duration(
    tmp_path: Path,
) -> None:
    boundaries = (0, 100_000, 190_000, 300_000)
    segments = tuple(
        BaseSegment(
            segment_id=f"segment_{index:03d}",
            start_ms=start_ms,
            end_ms=end_ms,
            evidence_refs=(),
            scene_refs=(f"scene_{index:03d}",),
            transcript_source="NONE",
        )
        for index, (start_ms, end_ms) in enumerate(
            pairwise(boundaries),
        )
    )
    scenes = tuple(
        _scene(index, segment.start_ms, segment.end_ms)
        for index, segment in enumerate(segments)
    )
    port = _PlanningTextPort(
        lambda _request: AssertionError("无文本不得调用规划模型"),
        lambda _request: AssertionError("无文本不得调用修复模型"),
    )

    batch = _plan(_planner(port), tmp_path, segments, (), scenes)

    assert [(plan.start_ms, plan.end_ms) for plan in batch.plans] == [
        (0, 190_000),
        (190_000, 300_000),
    ]


def test_chapter_planner_rejects_transcript_outside_referencing_segment(
    tmp_path: Path,
) -> None:
    segments, _transcript, scenes = _planning_fixture()
    misplaced = (
        _speech("asr_000", 60_000, 90_000),
        _speech("asr_001", 90_000, 120_000),
    )
    port = _PlanningTextPort(
        lambda _request: AssertionError("非法时间归属不得调用模型"),
        lambda _request: AssertionError("非法时间归属不得调用修复模型"),
    )

    with pytest.raises(VideoDemoError) as raised:
        _plan(_planner(port), tmp_path, segments, misplaced, scenes)

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_SEGMENT
    assert not port.main_requests


def test_chapter_planner_splits_oversized_input_on_segment_boundaries(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(4, text="长文本" * 200)
    config = DocumentGenerationConfig()
    single_sizes = []
    for segment, evidence in zip(segments, transcript, strict=True):
        request = ChapterPlanningRequest(
            title_hint="测试视频",
            duration_ms=segments[-1].end_ms,
            segments=(segment,),
            transcript_evidence=(evidence,),
            document_config=config,
            prompt_version="chapter-planner-v1",
        )
        data = prompt_for_planning(request)[2]
        single_sizes.append(len(data))

    def response(request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        return ChapterPlanningResponse(
            chapter_drafts=(
                ChapterDraft(
                    segment_refs=tuple(item.segment_id for item in request.segments),
                    title_hint="分批章节",
                    visual_mode="NONE",
                    semantic_targets=(),
                ),
            ),
        )

    port = _PlanningTextPort(response, response)
    planner = _planner(port, max_input_chars=max(single_sizes), max_input_bytes=1_048_576)

    batch = _plan(planner, tmp_path, segments, transcript, scenes)

    assert len(port.main_requests) == 4
    assert tuple(ref for plan in batch.plans for ref in plan.segment_refs) == tuple(
        item.segment_id for item in segments
    )
    assert batch.metrics["chapter_planner_logical_calls"] == 4


def test_chapter_planner_rejects_excessive_batches_before_provider_calls(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(3, text="长文本" * 200)
    config = DocumentGenerationConfig()
    single_request_sizes = tuple(
        len(
            prompt_for_planning(
                ChapterPlanningRequest(
                    title_hint="测试视频",
                    duration_ms=segments[-1].end_ms,
                    segments=(segment,),
                    transcript_evidence=(evidence,),
                    document_config=config,
                    prompt_version="chapter-planner-v1",
                ),
            )[2],
        )
        for segment, evidence in zip(segments, transcript, strict=True)
    )

    def response(request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        return ChapterPlanningResponse(
            chapter_drafts=(
                ChapterDraft(
                    segment_refs=tuple(item.segment_id for item in request.segments),
                    title_hint="分批章节",
                    visual_mode="NONE",
                    semantic_targets=(),
                ),
            ),
        )

    port = _PlanningTextPort(response, response)
    planner = _planner(
        port,
        max_input_chars=max(single_request_sizes),
        max_planning_batches=2,
    )

    with pytest.raises(VideoDemoError) as raised:
        _plan(planner, tmp_path, segments, transcript, scenes)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert not port.main_requests


def test_chapter_planner_rejects_impossible_chapter_budget_before_provider_calls(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(3, interval_ms=240_000)
    port = _PlanningTextPort(
        lambda _request: AssertionError("不可能满足的章节预算不得调用模型"),
        lambda _request: AssertionError("不可能满足的章节预算不得调用修复模型"),
    )

    with pytest.raises(VideoDemoError) as raised:
        _plan(
            _planner(port, max_chapters=2),
            tmp_path,
            segments,
            transcript,
            scenes,
        )

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert not port.main_requests


def test_chapter_planner_rejects_oversized_base_segment_before_provider_calls(
    tmp_path: Path,
) -> None:
    segment = BaseSegment(
        segment_id="segment_000",
        start_ms=0,
        end_ms=300_001,
        evidence_refs=("asr_000",),
        scene_refs=("scene_000",),
        transcript_source="ASR",
    )
    transcript = (_speech("asr_000", 0, 10_000),)
    scenes = (_scene(0, 0, 300_001),)
    port = _PlanningTextPort(
        lambda _request: AssertionError("超长基础片段不得调用模型"),
        lambda _request: AssertionError("超长基础片段不得调用修复模型"),
    )

    with pytest.raises(VideoDemoError) as raised:
        _plan(_planner(port), tmp_path, (segment,), transcript, scenes)

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert not port.main_requests


def test_chapter_planner_authentication_error_never_uses_rule_fallback(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture()
    port = _PlanningTextPort(
        lambda _request: VideoDemoError(
            ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED,
            "鉴权失败",
        ),
        lambda _request: AssertionError("鉴权失败不得修复"),
    )

    with pytest.raises(VideoDemoError) as raised:
        _plan(_planner(port), tmp_path, segments, transcript, scenes)

    assert raised.value.code == ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED
    assert not port.repair_requests


def test_chapter_planner_merges_short_model_chapters_before_count_budget(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(3, interval_ms=30_000)

    def fragmented(request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        return ChapterPlanningResponse(
            chapter_drafts=tuple(
                ChapterDraft(
                    segment_refs=(segment.segment_id,),
                    title_hint=f"碎片 {index + 1}",
                    visual_mode="NONE",
                    semantic_targets=(),
                )
                for index, segment in enumerate(request.segments)
            ),
        )

    batch = _plan(
        _planner(_PlanningTextPort(fragmented, fragmented)),
        tmp_path,
        segments,
        transcript,
        scenes,
    )

    assert len(batch.plans) == 1
    assert batch.plans[0].segment_refs == tuple(item.segment_id for item in segments)
    assert (batch.plans[0].start_ms, batch.plans[0].end_ms) == (0, 90_000)


def test_chapter_planner_does_not_invent_comparison_when_merging_short_chapters(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(2, interval_ms=30_000)

    def independent_targets(_request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        return ChapterPlanningResponse(
            chapter_drafts=tuple(
                ChapterDraft(
                    segment_refs=(f"segment_{index:03d}",),
                    title_hint=f"独立主题 {index + 1}",
                    visual_mode="SINGLE",
                    semantic_targets=(
                        VisualTargetDraft(
                            query_zh=f"独立画面 {index + 1}",
                            anchor_evidence_refs=(f"asr_{index:03d}",),
                        ),
                    ),
                )
                for index in range(2)
            ),
        )

    batch = _plan(
        _planner(_PlanningTextPort(independent_targets, independent_targets)),
        tmp_path,
        segments,
        transcript,
        scenes,
    )

    assert len(batch.plans) == 1
    assert batch.plans[0].visual_mode == "SINGLE"
    assert len(batch.plans[0].semantic_targets) == 2


def test_chapter_planner_repairs_semantic_anchor_span_over_thirty_seconds(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(2, interval_ms=30_000)

    def oversized_span(_request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        return ChapterPlanningResponse(
            chapter_drafts=(
                ChapterDraft(
                    segment_refs=("segment_000", "segment_001"),
                    title_hint="跨度过大的视觉目标",
                    visual_mode="SINGLE",
                    semantic_targets=(
                        VisualTargetDraft(
                            query_zh="比较相距过远的画面",
                            anchor_evidence_refs=("asr_000", "asr_001"),
                        ),
                    ),
                ),
            ),
        )

    batch = _plan(
        _planner(_PlanningTextPort(oversized_span, oversized_span)),
        tmp_path,
        segments,
        transcript,
        scenes,
    )

    assert batch.status == "PARTIAL_SUCCEEDED"
    assert batch.metrics["chapter_planner_structure_repairs"] == 1
    assert batch.plans[0].semantic_targets == ()


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {"unknown_metric": 1},
        {"chapter_planner_logical_calls": True},
        {"chapter_planner_logical_calls": -1},
        {"chapter_planner_logical_calls": 2**63},
    ],
)
def test_chapter_planning_batch_rejects_invalid_stage_metrics(
    tmp_path: Path,
    invalid_metrics: dict[str, object],
) -> None:
    segments, transcript, scenes = _planning_fixture()
    source = _plan(
        _planner(
            _PlanningTextPort(
                lambda request: ChapterPlanningResponse(
                    chapter_drafts=(
                        ChapterDraft(
                            segment_refs=tuple(
                                item.segment_id for item in request.segments
                            ),
                            title_hint="合法章节",
                            visual_mode="NONE",
                            semantic_targets=(),
                        ),
                    ),
                ),
                lambda _request: AssertionError("合法响应不应修复"),
            ),
        ),
        tmp_path,
        segments,
        transcript,
        scenes,
    )

    with pytest.raises(ValueError):
        ChapterPlanningBatch(
            plans=source.plans,
            metrics=invalid_metrics,  # type: ignore[arg-type]
        )


def test_chapter_planner_keeps_short_chapters_when_semantic_targets_conflict(
    tmp_path: Path,
) -> None:
    segments, transcript, scenes = _planning_fixture(2, interval_ms=30_000)

    def conflicting(_request: ChapterPlanningRequest) -> ChapterPlanningResponse:
        target = VisualTargetDraft(
            query_zh="同一锚点画面",
            anchor_evidence_refs=("asr_000",),
        )
        return ChapterPlanningResponse(
            chapter_drafts=(
                ChapterDraft(
                    segment_refs=("segment_000",),
                    title_hint="前章",
                    visual_mode="SINGLE",
                    semantic_targets=(target,),
                ),
                ChapterDraft(
                    segment_refs=("segment_001",),
                    title_hint="后章",
                    visual_mode="SINGLE",
                    semantic_targets=(
                        VisualTargetDraft(
                            query_zh="后章仍错误引用前章锚点",
                            anchor_evidence_refs=("asr_000",),
                        ),
                    ),
                ),
            ),
        )

    batch = _plan(
        _planner(_PlanningTextPort(conflicting, conflicting)),
        tmp_path,
        segments,
        transcript,
        scenes,
    )

    assert batch.status == "PARTIAL_SUCCEEDED"
    assert batch.metrics["chapter_planner_structure_repairs"] == 1
    assert batch.metrics["chapter_planner_fallback_chapters"] == 1
