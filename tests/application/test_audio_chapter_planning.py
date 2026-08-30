from __future__ import annotations

from pathlib import Path

from video_demo.application.audio_chapter_planning import AudioChapterPlanner
from video_demo.domain.audio_plan import AudioBaseSegment, AudioDocumentConfig
from video_demo.domain.evidence import SpeechSegment
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


def test_audio_planner_marks_repaired_response_in_cache(tmp_path: Path) -> None:
    from video_demo.domain.audio_plan import AudioChapterDraft
    from video_demo.integrations.audio_document_port import AudioChapterPlanningResponse

    class RepairPort(_Port):
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
            return super().plan_chapters(request.request, on_provider_attempt=on_provider_attempt)

    planner = AudioChapterPlanner(
        RepairPort(),
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
