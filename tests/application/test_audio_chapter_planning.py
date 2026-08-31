from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from video_demo.application.audio_chapter_planning import (
    AudioChapterPlanner,
    _materialize,
    _merge_short_drafts,
    _requests,
)
from video_demo.domain.audio_plan import AudioBaseSegment, AudioChapterDraft, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.audio_document_port import (
    AudioChapterBoundaryCoordinationResponse,
    AudioChapterBoundaryDecision,
    AudioChapterPlanningRequest,
    AudioChapterPlanningResponse,
)
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity


def _segments() -> tuple[AudioBaseSegment, ...]:
    return tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(f"asr_{index:03d}",),
            transcript_source="ASR",
        )
        for index in range(2)
    )


def _evidence() -> tuple[SpeechSegment, ...]:
    return tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            text=f"主题 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(2)
    )


def _identity() -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation="audio_chapter_planning",
        provider_config_fingerprint="a" * 64,
        model_id="text-model",
        generation_config=(("temperature", "0"),),
        main_response_schema_name="audio_chapter_planning_v1",
        main_prompt_version="audio-chapter-planner-v1",
        repair_response_schema_name="audio_chapter_planning_repair_v1",
        repair_prompt_version="audio-chapter-planner-repair-v1",
    )


class _Port:
    def plan_chapters(self, request, *, on_provider_attempt=None):
        from video_demo.domain.audio_plan import AudioChapterDraft
        from video_demo.integrations.audio_document_port import AudioChapterPlanningResponse

        return AudioChapterPlanningResponse(
            chapter_drafts=(
                AudioChapterDraft(
                    segment_refs=tuple(item.segment_id for item in request.segments),
                    title_hint="统一主题",
                ),
            ),
        )

    def repair_chapter_plan(self, request, *, on_provider_attempt=None):
        return self.plan_chapters(request.request, on_provider_attempt=on_provider_attempt)

    def coordinate_chapter_boundaries(self, request, *, on_provider_attempt=None):
        return AudioChapterBoundaryCoordinationResponse(
            decisions=tuple(
                AudioChapterBoundaryDecision(
                    boundary_index=item.boundary_index,
                    decision="KEEP",
                )
                for item in request.boundaries
            ),
        )


def _many_segments(count: int) -> tuple[AudioBaseSegment, ...]:
    return tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            evidence_refs=(f"asr_{index:03d}",),
            transcript_source="ASR",
        )
        for index in range(count)
    )


def _many_evidence(count: int) -> tuple[SpeechSegment, ...]:
    return tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            text=f"主题内容 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(count)
    )


def _planner(port: _Port) -> AudioChapterPlanner:
    return AudioChapterPlanner(
        port,
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=1,
    )


def _cache(tmp_path: Path) -> DocumentModelCache:
    return DocumentModelCache(
        tmp_path,
        max_entry_bytes=1_000_000,
        max_run_bytes=2_000_000,
    )


@pytest.mark.parametrize("wait_timeout_seconds", (0, -1))
def test_audio_planner_rejects_non_positive_invocation_wait_timeout(
    wait_timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="模型调用锁等待时间必须大于 0"):
        AudioChapterPlanner(
            _Port(),
            _identity(),
            max_input_chars=60_000,
            max_input_bytes=1_048_576,
            max_chapters=240,
            invocation_wait_timeout_seconds=wait_timeout_seconds,
        )


def test_audio_short_chapter_prefers_merging_with_previous_chapter() -> None:
    segments = (
        AudioBaseSegment(
            segment_id="audio_segment_000",
            start_ms=0,
            end_ms=100_000,
            evidence_refs=("asr_000",),
            transcript_source="ASR",
        ),
        AudioBaseSegment(
            segment_id="audio_segment_001",
            start_ms=100_000,
            end_ms=130_000,
            evidence_refs=("asr_001",),
            transcript_source="ASR",
        ),
        AudioBaseSegment(
            segment_id="audio_segment_002",
            start_ms=130_000,
            end_ms=230_000,
            evidence_refs=("asr_002",),
            transcript_source="ASR",
        ),
    )
    drafts = (
        AudioChapterDraft(segment_refs=(segments[0].segment_id,), title_hint="前一章"),
        AudioChapterDraft(segment_refs=(segments[1].segment_id,), title_hint="短章"),
        AudioChapterDraft(segment_refs=(segments[2].segment_id,), title_hint="后一章"),
    )

    normalized = _merge_short_drafts(drafts, segments, "standard")

    assert tuple(item.segment_refs for item in normalized) == (
        ("audio_segment_000", "audio_segment_001"),
        ("audio_segment_002",),
    )


def test_audio_planner_materializes_contiguous_audio_chapters(tmp_path: Path) -> None:
    planner = AudioChapterPlanner(
        _Port(),
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=2,
    )
    cache = DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000)

    result = planner.plan(
        cache=cache,
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=60_000,
        segments=_segments(),
        transcript_evidence=_evidence(),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert len(result.plans) == 1
    assert result.plans[0].start_ms == 0
    assert result.plans[0].end_ms == 60_000
    assert result.plans[0].segment_refs == tuple(item.segment_id for item in _segments())


def test_audio_planner_accepts_boundary_candidates_without_visual_inputs(tmp_path: Path) -> None:
    planner = AudioChapterPlanner(
        _Port(),
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=1,
    )
    cache = DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000)

    result = planner.plan(
        cache=cache,
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=60_000,
        segments=_segments(),
        transcript_evidence=_evidence(),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert result.plans[0].end_ms == 60_000


def test_audio_planner_rejects_evidence_outside_its_segment(tmp_path: Path) -> None:
    segments = _segments()
    evidence = _evidence()
    invalid_evidence = evidence[0].model_copy(update={"end_ms": 40_000})

    with pytest.raises(VideoDemoError) as raised:
        _planner(_Port()).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=60_000,
            segments=segments,
            transcript_evidence=(invalid_evidence, evidence[1]),
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_SEGMENT


def test_audio_planner_propagates_invocation_lock_timeout_without_fallback(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path)
    request = AudioChapterPlanningRequest(
        title_hint="音频",
        duration_ms=60_000,
        segments=_segments(),
        transcript_evidence=_evidence(),
        document_config=AudioDocumentConfig(),
        prompt_version="audio-chapter-planner-v1",
    )
    entered = Event()
    release = Event()

    def hold_lock() -> None:
        with cache.invocation_lock(
            _identity(),
            request,
            wait_timeout_seconds=2,
            is_cancel_requested=lambda: False,
            poll_interval_seconds=0.005,
        ):
            entered.set()
            release.wait(timeout=2)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(hold_lock)
            assert entered.wait(timeout=1)
            planner = AudioChapterPlanner(
                _Port(),
                _identity(),
                max_input_chars=60_000,
                max_input_bytes=1_048_576,
                max_chapters=240,
                invocation_wait_timeout_seconds=0.01,
                concurrency=1,
            )
            with pytest.raises(VideoDemoError) as raised:
                planner.plan(
                    cache=cache,
                    asset_sha256="b" * 64,
                    title_hint="音频",
                    duration_ms=60_000,
                    segments=_segments(),
                    transcript_evidence=_evidence(),
                    document_config=AudioDocumentConfig(),
                    is_cancel_requested=lambda: False,
                )
            assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
            assert future.done() is False
    finally:
        release.set()


def test_audio_planner_rejects_impossible_chapter_budget_before_provider_calls(
    tmp_path: Path,
) -> None:
    segments = _many_segments(3)
    segments = tuple(
        item.model_copy(
            update={
                "start_ms": index * 240_000,
                "end_ms": (index + 1) * 240_000,
            },
        )
        for index, item in enumerate(segments)
    )
    evidence = tuple(
        item.model_copy(
            update={
                "start_ms": index * 240_000,
                "end_ms": index * 240_000 + 10_000,
            },
        )
        for index, item in enumerate(_many_evidence(3))
    )

    class UnexpectedPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            raise AssertionError("不可能满足的章节预算不得调用模型")

    with pytest.raises(VideoDemoError) as raised:
        AudioChapterPlanner(
            UnexpectedPort(),
            _identity(),
            max_input_chars=60_000,
            max_input_bytes=1_048_576,
            max_chapters=2,
            invocation_wait_timeout_seconds=2,
        ).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=720_000,
            segments=segments,
            transcript_evidence=evidence,
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_audio_planner_rejects_oversized_base_segment_before_provider_calls(
    tmp_path: Path,
) -> None:
    segment = AudioBaseSegment(
        segment_id="audio_segment_000",
        start_ms=0,
        end_ms=300_001,
        evidence_refs=("asr_000",),
        transcript_source="ASR",
    )
    evidence = (
        SpeechSegment(
            evidence_id="asr_000",
            start_ms=0,
            end_ms=10_000,
            text="内容",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        ),
    )

    class UnexpectedPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            raise AssertionError("超长基础片段不得调用模型")

    with pytest.raises(VideoDemoError) as raised:
        AudioChapterPlanner(
            UnexpectedPort(),
            _identity(),
            max_input_chars=60_000,
            max_input_bytes=1_048_576,
            max_chapters=240,
            invocation_wait_timeout_seconds=2,
        ).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=300_001,
            segments=(segment,),
            transcript_evidence=evidence,
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_audio_request_builder_rejects_batch_limit_during_construction() -> None:
    with pytest.raises(VideoDemoError) as raised:
        _requests(
            "音频",
            490_000,
            _many_segments(49),
            _many_evidence(49),
            AudioDocumentConfig(),
            60_000,
            1_048_576,
            max_batches=2,
        )

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED


def test_audio_planner_does_not_fallback_on_capability_error(tmp_path: Path) -> None:
    class CapabilityPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            raise VideoDemoError(
                ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE,
                "结构化输出能力不可用",
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            raise AssertionError("能力错误不得调用修复模型")

    with pytest.raises(VideoDemoError) as raised:
        _planner(CapabilityPort()).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=60_000,
            segments=_segments(),
            transcript_evidence=_evidence(),
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )

    assert raised.value.code == ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE


def test_audio_planner_propagates_port_programming_type_error(tmp_path: Path) -> None:
    class BrokenPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            raise TypeError("音频端口内部参数错误")

    with pytest.raises(TypeError, match="音频端口内部参数错误"):
        _planner(BrokenPort()).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=60_000,
            segments=_segments(),
            transcript_evidence=_evidence(),
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )


def test_audio_planner_falls_back_on_plain_invalid_provider_error(tmp_path: Path) -> None:
    class InvalidPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            raise VideoDemoError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                "文本模型响应非法",
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            raise AssertionError("端口直接返回非法响应时不应再次修复")

    result = _planner(InvalidPort()).plan(
        cache=_cache(tmp_path),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=60_000,
        segments=_segments(),
        transcript_evidence=_evidence(),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert result.status == "PARTIAL_SUCCEEDED"
    assert result.warnings
    assert result.metrics["audio_planner_repairs"] == 0


def test_audio_planner_marks_repaired_response_in_cache(tmp_path: Path, caplog) -> None:
    from video_demo.domain.audio_plan import AudioChapterDraft
    from video_demo.integrations.audio_document_port import AudioChapterPlanningResponse

    class RepairPort(_Port):
        def __init__(self) -> None:
            self.repair_errors: tuple[str, ...] = ()
            self.repair_excerpt: str | None = None

        def plan_chapters(self, request, *, on_provider_attempt=None):
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=(request.segments[0].segment_id,) * 2,
                        title_hint="非法范围",
                    ),
                ),
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            self.repair_errors = request.invalid_response.validation_errors
            self.repair_excerpt = request.invalid_response.safe_json_excerpt
            return super().plan_chapters(request.request, on_provider_attempt=on_provider_attempt)

    port = RepairPort()
    planner = AudioChapterPlanner(
        port,
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=1,
    )
    cache = DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000)

    result = planner.plan(
        cache=cache,
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=60_000,
        segments=_segments(),
        transcript_evidence=_evidence(),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert result.plans
    assert "segment_range:incomplete_coverage" in port.repair_errors
    assert port.repair_errors
    assert port.repair_excerpt is not None
    assert "audio_segment_000" in port.repair_excerpt
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "音频章节规划响应校验失败" in message
        and "segment_range:incomplete_coverage" in message
        for message in messages
    )


def test_audio_planner_propagates_repair_port_programming_type_error(tmp_path: Path) -> None:
    class BrokenRepairPort(_Port):
        def plan_chapters(self, request, *, on_provider_attempt=None):
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=(request.segments[0].segment_id,) * 2,
                        title_hint="非法范围",
                    ),
                ),
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            raise TypeError("音频修复端口内部参数错误")

    with pytest.raises(TypeError, match="音频修复端口内部参数错误"):
        _planner(BrokenRepairPort()).plan(
            cache=_cache(tmp_path),
            asset_sha256="b" * 64,
            title_hint="音频",
            duration_ms=60_000,
            segments=_segments(),
            transcript_evidence=_evidence(),
            document_config=AudioDocumentConfig(),
            is_cancel_requested=lambda: False,
        )


def test_audio_materialize_rejects_missing_segment_references() -> None:
    segments = _many_segments(3)
    drafts = (
        AudioChapterDraft(
            segment_refs=(segments[0].segment_id, segments[2].segment_id),
            title_hint="跨越片段",
        ),
    )

    with pytest.raises(VideoDemoError, match="完整覆盖"):
        _materialize("b" * 64, drafts, segments)


def test_audio_planner_splits_input_into_batches_of_at_most_24_segments(
    tmp_path: Path,
) -> None:
    from video_demo.domain.audio_plan import AudioChapterDraft
    from video_demo.integrations.audio_document_port import AudioChapterPlanningResponse

    segments = tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            evidence_refs=(f"asr_{index:03d}",),
            transcript_source="ASR",
        )
        for index in range(25)
    )
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            text=f"主题 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(25)
    )

    class RecordingPort(_Port):
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def plan_chapters(self, request, *, on_provider_attempt=None):
            self.batch_sizes.append(len(request.segments))
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=tuple(item.segment_id for item in request.segments),
                        title_hint="统一主题",
                    ),
                ),
            )

    port = RecordingPort()
    planner = AudioChapterPlanner(
        port,
        _identity(),
        max_input_chars=60_000,
        max_input_bytes=1_048_576,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=1,
    )

    result = planner.plan(
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=250_000,
        segments=segments,
        transcript_evidence=evidence,
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert port.batch_sizes == [24, 1]
    assert result.plans[0].start_ms == 0
    assert result.plans[-1].end_ms == 250_000


def test_audio_planner_splits_again_when_actual_prompt_budget_is_smaller(
    tmp_path: Path,
) -> None:
    """预算不足以容纳 24 段时，应按真实音频 Prompt 动态拆批。"""

    from video_demo.domain.audio_plan import AudioChapterDraft
    from video_demo.integrations.audio_document_port import AudioChapterPlanningResponse

    segments = tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            evidence_refs=(f"asr_{index:03d}",),
            transcript_source="ASR",
        )
        for index in range(25)
    )
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 10_000,
            end_ms=(index + 1) * 10_000,
            text=f"这是一段用于章节规划预算测试的语音内容 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(25)
    )

    class RecordingPort(_Port):
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def plan_chapters(self, request, *, on_provider_attempt=None):
            self.batch_sizes.append(len(request.segments))
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=tuple(item.segment_id for item in request.segments),
                        title_hint="统一主题",
                    ),
                ),
            )

    port = RecordingPort()
    planner = AudioChapterPlanner(
        port,
        _identity(),
        max_input_chars=500,
        max_input_bytes=1_000,
        max_chapters=240,
        invocation_wait_timeout_seconds=2,
        concurrency=1,
    )

    result = planner.plan(
        cache=DocumentModelCache(tmp_path, max_entry_bytes=1_000_000, max_run_bytes=2_000_000),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=250_000,
        segments=segments,
        transcript_evidence=evidence,
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert len(port.batch_sizes) > 2
    assert max(port.batch_sizes) < 24
    assert sum(port.batch_sizes) == len(segments)
    assert result.plans[0].start_ms == 0
    assert result.plans[-1].end_ms == 250_000


def test_audio_planner_merges_same_topic_across_batch_boundary(tmp_path: Path) -> None:
    class MergePort(_Port):
        def __init__(self) -> None:
            self.boundary_requests = []

        def coordinate_chapter_boundaries(self, request, *, on_provider_attempt=None):
            self.boundary_requests.append(request)
            return AudioChapterBoundaryCoordinationResponse(
                decisions=(
                    AudioChapterBoundaryDecision(
                        boundary_index=request.boundaries[0].boundary_index,
                        decision="MERGE",
                        merged_title_hint="统一主题",
                    ),
                ),
            )

    port = MergePort()
    segments = _many_segments(30)
    result = _planner(port).plan(
        cache=_cache(tmp_path),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=300_000,
        segments=segments,
        transcript_evidence=_many_evidence(30),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert len(port.boundary_requests) == 1
    assert len(result.plans) == 1
    assert result.plans[0].segment_refs == tuple(item.segment_id for item in segments)
    assert result.plans[0].title_hint == "统一主题"


def test_audio_planner_keeps_batches_when_boundary_coordinator_fails(
    tmp_path: Path,
) -> None:
    class FailingBoundaryPort(_Port):
        def __init__(self) -> None:
            self.boundary_calls = 0

        def coordinate_chapter_boundaries(self, request, *, on_provider_attempt=None):
            self.boundary_calls += 1
            raise VideoDemoError(
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                "边界协调器暂时不可用",
            )

    port = FailingBoundaryPort()
    result = _planner(port).plan(
        cache=_cache(tmp_path),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=300_000,
        segments=_many_segments(30),
        transcript_evidence=_many_evidence(30),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert port.boundary_calls == 1
    assert len(result.plans) == 2
    assert result.plans[0].end_ms == result.plans[1].start_ms


def test_audio_planner_coordinates_boundary_after_batch_repair(tmp_path: Path) -> None:
    class RepairBoundaryPort(_Port):
        def __init__(self) -> None:
            self.boundary_calls = 0

        def plan_chapters(self, request, *, on_provider_attempt=None):
            if request.segments[0].start_ms == 0:
                return AudioChapterPlanningResponse(
                    chapter_drafts=(
                        AudioChapterDraft(
                            segment_refs=(request.segments[0].segment_id,) * 2,
                            title_hint="非法范围",
                        ),
                    ),
                )
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=tuple(item.segment_id for item in request.segments),
                        title_hint="第二主题",
                    ),
                ),
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=tuple(
                            item.segment_id for item in request.request.segments
                        ),
                        title_hint="第一主题",
                    ),
                ),
            )

        def coordinate_chapter_boundaries(self, request, *, on_provider_attempt=None):
            self.boundary_calls += 1
            return super().coordinate_chapter_boundaries(
                request,
                on_provider_attempt=on_provider_attempt,
            )

    port = RepairBoundaryPort()
    result = _planner(port).plan(
        cache=_cache(tmp_path),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=300_000,
        segments=_many_segments(30),
        transcript_evidence=_many_evidence(30),
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert port.boundary_calls == 1
    assert result.metrics["audio_planner_repairs"] == 1
    assert len(result.plans) == 2


def test_audio_planner_repairs_chapter_that_exceeds_five_minutes(tmp_path: Path) -> None:
    segments = tuple(
        AudioBaseSegment(
            segment_id=f"audio_segment_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            evidence_refs=(f"asr_{index:03d}",),
            transcript_source="ASR",
        )
        for index in range(11)
    )
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            text=f"主题内容 {index}",
            language="zh",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(11)
    )

    class OversizedChapterPort(_Port):
        def __init__(self) -> None:
            self.repair_calls = 0

        def plan_chapters(self, request, *, on_provider_attempt=None):
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=tuple(item.segment_id for item in request.segments),
                        title_hint="超长章节",
                    ),
                ),
            )

        def repair_chapter_plan(self, request, *, on_provider_attempt=None):
            self.repair_calls += 1
            segment_ids = tuple(item.segment_id for item in request.request.segments)
            return AudioChapterPlanningResponse(
                chapter_drafts=(
                    AudioChapterDraft(
                        segment_refs=segment_ids[:10],
                        title_hint="第一章",
                    ),
                    AudioChapterDraft(
                        segment_refs=segment_ids[10:],
                        title_hint="第二章",
                    ),
                ),
            )

    port = OversizedChapterPort()
    result = _planner(port).plan(
        cache=_cache(tmp_path),
        asset_sha256="b" * 64,
        title_hint="音频",
        duration_ms=330_000,
        segments=segments,
        transcript_evidence=evidence,
        document_config=AudioDocumentConfig(),
        is_cancel_requested=lambda: False,
    )

    assert port.repair_calls == 1
    assert len(result.plans) == 2
    assert all(plan.duration_ms <= 300_000 for plan in result.plans)
