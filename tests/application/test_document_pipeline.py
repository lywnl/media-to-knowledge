from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import pytest

from video_demo.application.chapter_frames import ChapterFrameSearchBatch
from video_demo.application.chapter_planning import ChapterPlanningBatch
from video_demo.application.chapter_vision import ChapterVisionBatch
from video_demo.application.document_pipeline import VideoUnderstandingPipeline
from video_demo.application.document_rendering import render_markdown
from video_demo.application.document_writing import WrittenDocument
from video_demo.application.pipeline_contracts import (
    EvidencePreparationLimits,
    PipelineContext,
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
    SceneIndex,
    SpeechAnalysis,
    StageMetric,
    scene_index_sha256,
)
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.domain.document_plan import (
    ChapterFrameSet,
    ChapterPlan,
    FrameCandidateArtifact,
)
from video_demo.domain.evidence import (
    KeyframeEvidence,
    SceneBoundary,
    SpeechSegment,
    VisualObservationEvidence,
)
from video_demo.domain.manifest import Rational, VideoAssetManifest, VideoStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.persistence.repositories import Scope
from video_demo.storage.document_cache import DocumentModelCache

_OUTER_STAGES = [
    "REGISTER",
    "PROBE",
    "TRANSCODE",
    "EVIDENCE_PREP",
    "CHAPTER_PLAN",
    "FRAME_SEARCH",
    "VISUAL_EVIDENCE",
    "CHAPTER_WRITE",
    "DOCUMENT_ASSEMBLY",
    "RESULT",
]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="讲解知识文档流水线",
        language="zh",
        confidence=0.99,
        is_fully_evaluated_language=True,
    )


def _result(run_id: str, asset_sha256: str, speech: SpeechSegment) -> VideoUnderstandingResult:
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="流水线",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="讲解知识文档流水线。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(
            ParagraphBlock(
                text="讲解知识文档流水线。",
                evidence_refs=(speech.evidence_id,),
            ),
        ),
        claims=(),
        evidence_refs=(speech.evidence_id,),
        transcript_source="ASR",
    )
    return VideoUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        summary=VideoDocumentSummary(
            title="测试知识文档",
            duration_ms=1_000,
            overview_zh="流水线概览",
        ),
        chapters=(chapter,),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text-model",
            vlm_model_id="vlm-model",
            prompt_versions=PromptVersions(
                chapter_planner="chapter-planner-v1",
                chapter_planner_repair="chapter-planner-repair-v1",
                chapter_vlm="chapter-vlm-v1",
                chapter_vlm_repair="chapter-vlm-repair-v1",
                chapter_writer="chapter-writer-v1",
                chapter_writer_repair="chapter-writer-repair-v1",
                global_editor="global-editor-v1",
                global_editor_repair="global-editor-repair-v1",
            ),
        ),
    )


def _empty_result(run_id: str, asset_sha256: str) -> VideoUnderstandingResult:
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="无语义内容",
        title_evidence_refs=(),
        summary_zh="本时段未提取到可验证语义内容。",
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
    )
    return VideoUnderstandingResult(
        run_id=run_id,
        asset_sha256=asset_sha256,
        summary=VideoDocumentSummary(
            title="测试知识文档",
            duration_ms=1_000,
            overview_zh="未提取到可验证语义内容。",
        ),
        chapters=(chapter,),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text-model",
            vlm_model_id="vlm-model",
            prompt_versions=PromptVersions(
                chapter_planner="chapter-planner-v1",
                chapter_planner_repair="chapter-planner-repair-v1",
                chapter_vlm="chapter-vlm-v1",
                chapter_vlm_repair="chapter-vlm-repair-v1",
                chapter_writer="chapter-writer-v1",
                chapter_writer_repair="chapter-writer-repair-v1",
                global_editor="global-editor-v1",
                global_editor_repair="global-editor-repair-v1",
            ),
        ),
    )


class _RunCache:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root


class _PipelineFixture:
    def __init__(
        self,
        runtime_root: Path,
        *,
        visual_evidence: bool = False,
        speech_error: VideoDemoError | None = None,
        speech_analysis: SpeechAnalysis | None = None,
        planning_status: str = "SUCCEEDED",
        visual_status: str = "SUCCEEDED",
        writing_status: str = "SUCCEEDED",
        has_frame_candidate: bool = True,
        run_root_factory: Callable[[str], Path] | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.speech_error = speech_error
        self.speech_analysis = speech_analysis
        self.planning_status = planning_status
        self.visual_status = visual_status
        self.writing_status = writing_status
        self.has_frame_candidate = has_frame_candidate
        self.run_root_factory = run_root_factory or (
            lambda run_id: Path("private-runs") / run_id
        )
        self.speech_started = threading.Event()
        self.scene_started = threading.Event()
        self.speech_finished = threading.Event()
        self.scene_finished = threading.Event()
        self.planning_finished = threading.Event()
        self.frame_finished = threading.Event()
        self.visual_finished = threading.Event()
        self.writing_finished = threading.Event()
        self.calls: list[str] = []
        self.cache_roots: list[Path] = []
        self.planner_caches: list[object] = []
        self.vision_caches: list[object] = []
        self.writer_caches: list[object] = []
        self.writer_calls = 0
        self._visual_evidence = visual_evidence

    def register(self, context: PipelineContext) -> RegisteredAsset:
        self.calls.append("REGISTER_CALL")
        source = self.runtime_root / f"{context.run_id}.mp4"
        return RegisteredAsset(
            source_path=source,
            source_sha256="a" * 64,
            object_ref="object_001",
            source_size_bytes=1,
            source_mime="video/mp4",
            run_relative_root=self.run_root_factory(context.run_id),
            config=PipelineRunConfig(),
        )

    def probe(self, asset: RegisteredAsset) -> ProbedAsset:
        self.calls.append("PROBE_CALL")
        return ProbedAsset(
            asset=asset,
            manifest=VideoAssetManifest(
                object_ref=asset.object_ref,
                source_sha256=asset.source_sha256,
                source_size_bytes=asset.source_size_bytes,
                source_mime=asset.source_mime,
                duration_ms=1_000,
                video_stream=VideoStream(
                    index=0,
                    codec_name="h264",
                    width=640,
                    height=360,
                    average_frame_rate=Rational(numerator=25, denominator=1),
                ),
                format_name="mov,mp4",
                ffprobe_version="ffprobe test",
            ),
            limits=ProbeLimits(),
        )

    def transcode(
        self,
        probed: ProbedAsset,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> PreparedMedia:
        assert not is_cancel_requested()
        self.calls.append("TRANSCODE_CALL")
        return PreparedMedia(
            source=probed,
            proxy_path=self.runtime_root / probed.asset.run_relative_root / "proxy.mp4",
            proxy_sha256="b" * 64,
            proxy_size_bytes=1,
            audio_path=self.runtime_root / probed.asset.run_relative_root / "audio.wav",
            audio_sha256="c" * 64,
        )

    def analyze(
        self,
        _media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis:
        assert not is_cancel_requested()
        self.calls.append("SPEECH_CALL")
        self.speech_started.set()
        assert self.scene_started.wait(timeout=1), "场景分支未与语音分支并发"
        if self.speech_error is not None:
            raise self.speech_error
        self.speech_finished.set()
        if self.speech_analysis is not None:
            return self.speech_analysis
        return SpeechAnalysis(
            transcript_source="ASR",
            evidence=(_speech(),),
            warnings=("SPEECH_WARNING",),
            stage_metrics=(StageMetric("SPEECH_ASR", 0),),
            stage_cache_hits=("SPEECH_ASR",),
        )

    def prepare_scene_index(
        self,
        media: PreparedMedia,
        *,
        limits: EvidencePreparationLimits,
        is_cancel_requested: Callable[[], bool],
    ) -> SceneIndex:
        assert limits.max_scene_boundaries >= 1
        assert not is_cancel_requested()
        self.calls.append("SCENE_CALL")
        self.scene_started.set()
        assert self.speech_started.wait(timeout=1), "语音分支未与场景分支并发"
        self.scene_finished.set()
        scene = SceneBoundary(
            evidence_id="scene_001",
            start_ms=0,
            end_ms=media.source.duration_ms,
            transition="candidate",
            score=0.9,
        )
        return SceneIndex(
            proxy_sha256=media.proxy_sha256,
            duration_ms=media.source.duration_ms,
            frame_tolerance_ms=40,
            scenes=(scene,),
            index_sha256=scene_index_sha256(
                proxy_sha256=media.proxy_sha256,
                duration_ms=media.source.duration_ms,
                frame_tolerance_ms=40,
                scenes=(scene,),
            ),
        )

    def plan(self, **kwargs: object) -> ChapterPlanningBatch:
        assert self.speech_finished.is_set() and self.scene_finished.is_set()
        self.calls.append("PLAN_CALL")
        self.planner_caches.append(kwargs["cache"])
        segments = kwargs["segments"]
        segment = cast(tuple[object, ...], segments)[0]
        plan = ChapterPlan(
            chapter_id="chapter_001",
            start_ms=0,
            end_ms=1_000,
            segment_refs=(segment.segment_id,),  # type: ignore[attr-defined]
            title_hint="流水线",
            visual_mode="NONE",
            semantic_targets=(),
            base_coverage_targets=(),
        )
        self.planning_finished.set()
        return ChapterPlanningBatch(
            plans=(plan,),
            warnings=("PLANNING_WARNING", "SHARED_WARNING"),
            status=cast(Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"], self.planning_status),
            metrics={"chapter_planner_logical_calls": 1},
        )

    def search(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        _transcript_by_id: object,
        scene_index: SceneIndex,
        _document_config: DocumentGenerationConfig,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterFrameSearchBatch:
        assert self.planning_finished.is_set()
        assert not is_cancel_requested()
        self.calls.append("FRAME_CALL")
        candidates = ()
        if self.has_frame_candidate:
            candidates = (
                FrameCandidateArtifact(
                    frame_id="candidate_001",
                    timestamp_ms=500,
                    sha256="d" * 64,
                    size_bytes=1,
                    relative_path=f"visual/candidates/{'d' * 64}.jpg",
                    perceptual_hash="0123456789abcdef",
                    target_ids=("temporary_target",),
                ),
            )
        self.frame_finished.set()
        return ChapterFrameSearchBatch(
            asset_sha256=media.source.asset.source_sha256,
            allowed_run_root=self.runtime_root / media.source.asset.run_relative_root,
            frame_tolerance_ms=scene_index.frame_tolerance_ms,
            frame_sets=(
                ChapterFrameSet(
                    chapter_id=chapters[0].chapter_id,
                    candidates=candidates,
                ),
            ),
            chapter_status=(
                (
                    chapters[0].chapter_id,
                    "SUCCEEDED" if candidates else "NO_CANDIDATE",
                ),
            ),
            warnings=("FRAME_WARNING",),
            metrics={"visual_collapsed_same_frame_chapters": 1},
        )

    def analyze_all(
        self,
        chapters: tuple[ChapterPlan, ...],
        frame_batch: ChapterFrameSearchBatch,
        _transcript_evidence: tuple[SpeechSegment, ...],
        _document_config: DocumentGenerationConfig,
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterVisionBatch:
        assert self.frame_finished.is_set()
        assert frame_batch.allowed_run_root.is_absolute()
        assert frame_batch.allowed_run_root.is_relative_to(self.runtime_root)
        assert not is_cancel_requested()
        self.calls.append("VISION_CALL")
        self.vision_caches.append(cache)
        self.visual_finished.set()
        if not self._visual_evidence:
            chapter_status = (
                "DEGRADED"
                if self.visual_status == "PARTIAL_SUCCEEDED"
                else "NO_VALUE"
            )
            return ChapterVisionBatch(
                observations=(),
                evidence=(),
                keyframe_evidence=(),
                chapter_status=(
                    (
                        chapters[0].chapter_id,
                        cast(Literal["DEGRADED", "NO_VALUE"], chapter_status),
                    ),
                ),
                warnings=("VISION_WARNING", "SHARED_WARNING"),
                status=cast(Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"], self.visual_status),
                metrics={"vlm_no_value_chapters": 1},
            )
        keyframe = KeyframeEvidence(
            evidence_id="keyframe_001",
            start_ms=500,
            end_ms=501,
            keyframe_id="frame_001",
            timestamp_ms=500,
            relative_path=f"visual/keyframes/{'e' * 64}.jpg",
            mime_type="image/jpeg",
            sha256="e" * 64,
            perceptual_hash="0123456789abcdef",
            size_bytes=1,
        )
        observation = VisualObservationEvidence(
            evidence_id="observation_001",
            start_ms=0,
            end_ms=1_000,
            chapter_id=chapters[0].chapter_id,
            target_ids=("target_001",),
            keyframe_refs=(keyframe.evidence_id,),
            transcript_evidence_refs=(_speech().evidence_id,),
            visual_type="GENERAL",
            caption="画面展示流水线",
            relation_to_transcript="SUPPORTING",
            certainty=0.9,
        )
        chapter_status = (
            "DEGRADED" if self.visual_status == "PARTIAL_SUCCEEDED" else "SUCCEEDED"
        )
        return ChapterVisionBatch(
            observations=(observation,),
            evidence=(keyframe, observation),
            keyframe_evidence=(keyframe,),
            chapter_status=(
                (
                    chapters[0].chapter_id,
                    cast(
                        Literal["SUCCEEDED", "DEGRADED"],
                        chapter_status,
                    ),
                ),
            ),
            warnings=("VISION_WARNING", "SHARED_WARNING"),
            status=cast(Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"], self.visual_status),
            metrics={"vlm_logical_analyses": 1},
        )

    def write(
        self,
        context: object,
        _plans: tuple[ChapterPlan, ...],
        transcript_evidence: tuple[SpeechSegment, ...],
        _visual_evidence: tuple[VisualObservationEvidence, ...],
        _keyframes: tuple[KeyframeEvidence, ...],
        *,
        cache: DocumentModelCache,
        is_cancel_requested: Callable[[], bool],
    ) -> WrittenDocument:
        assert self.visual_finished.is_set()
        assert not is_cancel_requested()
        self.calls.append("WRITE_CALL")
        self.writer_calls += 1
        self.writer_caches.append(cache)
        self.writing_finished.set()
        if not transcript_evidence:
            return WrittenDocument(
                result=_empty_result(
                    context.run_id,  # type: ignore[attr-defined]
                    context.asset_sha256,  # type: ignore[attr-defined]
                ),
                warnings=("WRITING_WARNING", "SHARED_WARNING"),
                status=cast(
                    Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"],
                    self.writing_status,
                ),
                metrics={"chapter_writer_logical_calls": 1},
            )
        return WrittenDocument(
            result=_result(
                context.run_id,  # type: ignore[attr-defined]
                context.asset_sha256,  # type: ignore[attr-defined]
                transcript_evidence[0],
            ),
            warnings=("WRITING_WARNING", "SHARED_WARNING"),
            status=cast(Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"], self.writing_status),
            metrics={"chapter_writer_logical_calls": 1},
        )

    def cache_factory(self, run_root: Path) -> DocumentModelCache:
        self.cache_roots.append(run_root)
        return cast(DocumentModelCache, _RunCache(run_root))


def _pipeline(
    fixture: _PipelineFixture,
    *,
    max_result_evidence_items: int = 25_000,
) -> VideoUnderstandingPipeline:
    return VideoUnderstandingPipeline(
        fixture,
        fixture,
        fixture,
        fixture,
        fixture,
        fixture,
        fixture,
        fixture,
        fixture,
        fixture.cache_factory,
        runtime_root=fixture.runtime_root,
        evidence_preparation_limits=EvidencePreparationLimits(
            max_transcript_evidence_items=20_000,
            max_transcript_chars=1_000_000,
            max_scene_boundaries=20_000,
            max_base_segments=20_000,
        ),
        max_result_evidence_items=max_result_evidence_items,
    )


def _context(
    run_id: str,
    *,
    is_cancel_requested: Callable[[], bool] = lambda: False,
    on_stage_start: Callable[[str], None] = lambda _stage: None,
) -> PipelineContext:
    return PipelineContext(
        run_id=run_id,
        scope=Scope("tenant-a", "app-a", "kb-a"),
        title_hint="测试知识文档",
        document_config=DocumentGenerationConfig(),
        is_cancel_requested=is_cancel_requested,
        on_stage_start=on_stage_start,
    )


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_document_pipeline_rejects_invalid_result_evidence_budget(
    tmp_path: Path,
    invalid: object,
) -> None:
    fixture = _PipelineFixture(tmp_path)

    with pytest.raises(ValueError, match="正整数"):
        _pipeline(fixture, max_result_evidence_items=invalid)  # type: ignore[arg-type]


def test_document_pipeline_runs_fixed_dependency_order_and_closes_output(
    tmp_path: Path,
) -> None:
    fixture = _PipelineFixture(tmp_path)
    stages: list[str] = []

    def on_stage_start(stage: str) -> None:
        if stage == "DOCUMENT_ASSEMBLY":
            assert fixture.writing_finished.is_set(), "文档组装不得早于全部章节写作"
        stages.append(stage)

    outcome = _pipeline(fixture).run(_context("run_001", on_stage_start=on_stage_start))

    assert stages == _OUTER_STAGES
    assert outcome.status == "SUCCEEDED"
    assert outcome.evidence == (_speech(),)
    assert all(item.evidence_type != "SCENE" for item in outcome.evidence)
    assert outcome.document == render_markdown(outcome.result, outcome.evidence)
    assert outcome.warnings == (
        "SPEECH_WARNING",
        "PLANNING_WARNING",
        "SHARED_WARNING",
        "FRAME_WARNING",
        "VISION_WARNING",
        "WRITING_WARNING",
    )
    assert set(outcome.stage_metrics) == RESULT_STAGE_NAMES
    assert set(outcome.model_metrics) == MODEL_METRIC_NAMES
    assert outcome.model_metrics["chapter_planner_logical_calls"] == 1
    assert outcome.model_metrics["visual_collapsed_same_frame_chapters"] == 1
    assert outcome.model_metrics["vlm_no_value_chapters"] == 1
    assert outcome.model_metrics["chapter_writer_logical_calls"] == 1
    assert outcome.stage_cache_hits == ("SPEECH_ASR",)
    assert outcome.stage_metrics["SPEECH_ASR"] == 0
    assert outcome.transcript_source == "ASR"
    assert outcome.frame_batch.frame_sets[0].candidates
    assert outcome.visual_batch.evidence == ()


@pytest.mark.parametrize(
    "partial_source",
    ["planning", "visual", "writing"],
)
def test_document_pipeline_merges_only_planning_visual_and_written_status(
    tmp_path: Path,
    partial_source: str,
) -> None:
    fixture = _PipelineFixture(
        tmp_path,
        planning_status=(
            "PARTIAL_SUCCEEDED" if partial_source == "planning" else "SUCCEEDED"
        ),
        visual_status=(
            "PARTIAL_SUCCEEDED" if partial_source == "visual" else "SUCCEEDED"
        ),
        writing_status=(
            "PARTIAL_SUCCEEDED" if partial_source == "writing" else "SUCCEEDED"
        ),
    )

    outcome = _pipeline(fixture).run(_context("run_001"))

    assert outcome.status == "PARTIAL_SUCCEEDED"
    assert outcome.warnings.count("SHARED_WARNING") == 1
    assert outcome.warnings.index("FRAME_WARNING") < outcome.warnings.index("VISION_WARNING")


def test_document_pipeline_treats_normal_no_frame_and_no_visual_as_succeeded(
    tmp_path: Path,
) -> None:
    fixture = _PipelineFixture(tmp_path, has_frame_candidate=False)

    outcome = _pipeline(fixture).run(_context("run_001"))

    assert outcome.frame_batch.chapter_status == (("chapter_001", "NO_CANDIDATE"),)
    assert outcome.visual_batch.chapter_status == (("chapter_001", "NO_VALUE"),)
    assert outcome.status == "SUCCEEDED"


@pytest.mark.parametrize(
    ("transcript_source", "warning"),
    [
        ("NONE", "NO_AUDIO_TRACK"),
        ("ASR", "NO_SPEECH_DETECTED"),
    ],
)
def test_no_audio_or_no_speech_keeps_visual_pipeline_running(
    tmp_path: Path,
    transcript_source: Literal["NONE", "ASR"],
    warning: str,
) -> None:
    fixture = _PipelineFixture(
        tmp_path,
        speech_analysis=SpeechAnalysis(
            transcript_source=transcript_source,
            evidence=(),
            warnings=(warning,),
        ),
    )

    outcome = _pipeline(fixture).run(_context("run_001"))

    assert outcome.transcript_source == transcript_source
    assert warning in outcome.warnings
    assert "FRAME_CALL" in fixture.calls
    assert "VISION_CALL" in fixture.calls
    assert outcome.result.chapters[0].content_status == "NO_SEMANTIC_EVIDENCE"


def test_document_pipeline_outcome_metrics_are_read_only(tmp_path: Path) -> None:
    outcome = _pipeline(_PipelineFixture(tmp_path)).run(_context("run_001"))

    with pytest.raises(TypeError):
        outcome.stage_metrics["RESULT"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        outcome.model_metrics["vlm_logical_analyses"] = 99  # type: ignore[index]


def test_document_pipeline_builds_one_exact_isolated_cache_per_run(tmp_path: Path) -> None:
    fixture = _PipelineFixture(tmp_path)
    pipeline = _pipeline(fixture)

    pipeline.run(_context("run_001"))
    pipeline.run(_context("run_002"))

    assert fixture.cache_roots == [
        (tmp_path / "private-runs/run_001").resolve(),
        (tmp_path / "private-runs/run_002").resolve(),
    ]
    assert fixture.planner_caches[0] is fixture.vision_caches[0] is fixture.writer_caches[0]
    assert fixture.planner_caches[1] is fixture.vision_caches[1] is fixture.writer_caches[1]
    assert fixture.planner_caches[0] is not fixture.planner_caches[1]


def test_document_pipeline_rejects_registered_run_root_escape_before_cache(
    tmp_path: Path,
) -> None:
    fixture = _PipelineFixture(
        tmp_path,
        run_root_factory=lambda _run_id: Path("../outside-run"),
    )

    with pytest.raises(VideoDemoError) as raised:
        _pipeline(fixture).run(_context("run_001"))

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert fixture.cache_roots == []


def test_document_pipeline_checks_actual_evidence_budget_before_writer(tmp_path: Path) -> None:
    fixture = _PipelineFixture(tmp_path, visual_evidence=True)

    with pytest.raises(VideoDemoError) as raised:
        _pipeline(fixture, max_result_evidence_items=2).run(_context("run_001"))

    assert raised.value.code == ErrorCode.INPUT_BUDGET_EXCEEDED
    assert fixture.writer_calls == 0


def test_document_pipeline_stops_stage_body_when_stage_callback_requests_cancel(
    tmp_path: Path,
) -> None:
    fixture = _PipelineFixture(tmp_path)
    cancelled = False

    def cancel() -> bool:
        return cancelled

    def on_stage_start(stage: str) -> None:
        nonlocal cancelled
        if stage == "CHAPTER_PLAN":
            cancelled = True

    with pytest.raises(VideoDemoError) as raised:
        _pipeline(fixture).run(
            _context(
                "run_001",
                is_cancel_requested=cancel,
                on_stage_start=on_stage_start,
            ),
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert "PLAN_CALL" not in fixture.calls


def test_document_pipeline_propagates_evidence_branch_error_without_thread_leak(
    tmp_path: Path,
) -> None:
    error = VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "模拟语音失败")
    fixture = _PipelineFixture(tmp_path, speech_error=error)

    with pytest.raises(VideoDemoError) as raised:
        _pipeline(fixture).run(_context("run_001"))

    assert raised.value is error
    assert fixture.scene_finished.is_set()
    assert not any(
        thread.name.startswith("document-evidence-prep")
        for thread in threading.enumerate()
    )


@pytest.mark.parametrize("failing_branch", ["speech", "scene"])
def test_document_pipeline_cancels_evidence_sibling_and_preserves_original_error(
    tmp_path: Path,
    failing_branch: str,
) -> None:
    original = VideoDemoError(ErrorCode.SPEECH_AUDIO_INVALID, "模拟证据分支失败")
    sibling_saw_cancel = threading.Event()

    class Fixture(_PipelineFixture):
        def analyze(
            self,
            media: PreparedMedia,
            *,
            is_cancel_requested: Callable[[], bool],
        ) -> SpeechAnalysis:
            if failing_branch == "speech":
                self.speech_started.set()
                assert self.scene_started.wait(timeout=1)
                raise original
            analysis = super().analyze(
                media,
                is_cancel_requested=is_cancel_requested,
            )
            _wait_for_sibling_cancel(is_cancel_requested, sibling_saw_cancel)
            return analysis

        def prepare_scene_index(
            self,
            media: PreparedMedia,
            *,
            limits: EvidencePreparationLimits,
            is_cancel_requested: Callable[[], bool],
        ) -> SceneIndex:
            scene_index = super().prepare_scene_index(
                media,
                limits=limits,
                is_cancel_requested=is_cancel_requested,
            )
            if failing_branch == "scene":
                raise original
            _wait_for_sibling_cancel(is_cancel_requested, sibling_saw_cancel)
            return scene_index

    started_at = time.monotonic()
    with pytest.raises(VideoDemoError) as raised:
        _pipeline(Fixture(tmp_path)).run(_context("run_001"))

    assert raised.value is original
    assert sibling_saw_cancel.is_set()
    assert time.monotonic() - started_at < 0.2
    assert not any(
        thread.name.startswith("document-evidence-prep")
        for thread in threading.enumerate()
    )


def _wait_for_sibling_cancel(
    is_cancel_requested: Callable[[], bool],
    observed: threading.Event,
) -> None:
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
        if is_cancel_requested():
            observed.set()
            return
        time.sleep(0.005)
