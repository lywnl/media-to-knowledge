from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.application.chapter_frames import ChapterFrameSearchBatch
from video_demo.application.chapter_vision import ChapterVisionBatch
from video_demo.application.document_rendering import render_markdown
from video_demo.application.pipeline import PipelineJobHandler
from video_demo.application.pipeline_contracts import (
    PipelineContext,
    PipelineOutcome,
    PipelineRunConfig,
)
from video_demo.application.queries import ResultQueryService
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    PromptVersions,
    SemanticChapter,
    SemanticSection,
    VideoDocumentSummary,
    VideoUnderstandingResult,
    section_id_for,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus, RunStatusValue
from video_demo.persistence.repositories import (
    ClaimedJob,
    JobRepository,
    Scope,
    VideoObjectRepository,
    VideoRunRepository,
)
from video_demo.storage.artifacts import AtomicArtifactStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _claim(
    tmp_path: Path,
    *,
    config_snapshot: dict[str, object] | None = None,
) -> tuple[Database, Scope, ClaimedJob, Path]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'handler.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        VideoObjectRepository(session).add_ready(
            scope=scope,
            object_ref="object_001",
            original_filename="faster-whisper 教程.mp4",
            declared_mime="video/mp4",
            detected_mime="video/mp4",
            size_bytes=1,
            sha256="a" * 64,
            relative_path="objects/source.mp4",
        )
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="object_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="object_001",
            idempotency_key="handler-001",
            config_snapshot=(
                config_snapshot
                if config_snapshot is not None
                else PipelineRunConfig().model_dump(mode="json")
            ),
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    return database, scope, claimed, runtime_root


def _outcome() -> PipelineOutcome:
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="无语义",
        title_evidence_refs=(),
        summary_zh="未提取到可验证语义内容",
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash=_digest(""),
    )
    result = VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="faster-whisper 教程",
            duration_ms=1_000,
            overview_zh="未提取到可验证语义内容",
            key_points=(),
            retrieval_text="摘要",
            retrieval_hash=_digest("摘要"),
        ),
        sections=(
            SemanticSection(
                section_id=section_id_for("a" * 64, (chapter.chapter_id,)),
                title="全部内容",
                summary_zh="未提取到可验证语义内容",
                chapter_refs=(chapter.chapter_id,),
            ),
        ),
        chapters=(chapter,),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text-model",
            vlm_model_id="qwen3-vl-flash",
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
    frame_batch = ChapterFrameSearchBatch(
        asset_sha256="a" * 64,
        allowed_run_root=Path("runs/test-scope/run_001"),
        frame_tolerance_ms=40,
        frame_sets=(),
        chapter_status=(),
        metrics={},
    )
    visual_batch = ChapterVisionBatch(
        observations=(),
        evidence=(),
        keyframe_evidence=(),
        chapter_status=(),
        metrics={},
    )
    return PipelineOutcome(
        status="SUCCEEDED",
        result=result,
        evidence=(),
        document=render_markdown(result, ()),
        warnings=(),
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        stage_cache_hits=(),
        transcript_source="NONE",
        frame_batch=frame_batch,
        visual_batch=visual_batch,
    )


def test_handler_uses_snapshot_and_filename_then_publishes_same_3_outcome(
    tmp_path: Path,
) -> None:
    database, scope, claimed, runtime_root = _claim(tmp_path)
    captured: list[PipelineContext] = []
    outcome = _outcome()

    class Pipeline:
        def run(self, context: PipelineContext) -> PipelineOutcome:
            captured.append(context)
            return outcome

    queries = ResultQueryService(database, AtomicArtifactStore(runtime_root))
    PipelineJobHandler(database, Pipeline(), queries)(claimed)  # type: ignore[arg-type]

    assert captured[0].title_hint == "faster-whisper 教程"
    assert captured[0].document_config == DocumentGenerationConfig()
    assert queries.get_result(scope, "run_001") == outcome.result
    assert queries.get_document(scope, "run_001") == outcome.document.content
    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        job = JobRepository(session).get(scope, "job_001")
        assert run is not None and run.status == RunStatusValue.SUCCEEDED
        assert job is not None and job.status == JobStatus.SUCCEEDED


def test_handler_rejects_old_snapshot_before_running_pipeline_or_publishing(
    tmp_path: Path,
) -> None:
    database, scope, claimed, runtime_root = _claim(
        tmp_path,
        config_snapshot={"language_hints": []},
    )
    called = False

    class Pipeline:
        def run(self, _context: PipelineContext) -> PipelineOutcome:
            nonlocal called
            called = True
            return _outcome()

    queries = ResultQueryService(database, AtomicArtifactStore(runtime_root))
    with pytest.raises(VideoDemoError) as raised:
        PipelineJobHandler(database, Pipeline(), queries)(claimed)  # type: ignore[arg-type]

    assert raised.value.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED
    assert called is False
    with pytest.raises(VideoDemoError) as not_ready:
        queries.get_result(scope, "run_001")
    assert not_ready.value.code == ErrorCode.VIDEO_RESULT_NOT_READY


def test_handler_records_current_stage_on_unexpected_failure(tmp_path: Path) -> None:
    database, scope, claimed, runtime_root = _claim(tmp_path)

    class Pipeline:
        def run(self, context: PipelineContext) -> PipelineOutcome:
            context.on_stage_start("PROBE")
            raise RuntimeError("内部错误正文不得落库")

    handler = PipelineJobHandler(
        database,
        Pipeline(),  # type: ignore[arg-type]
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
    )
    with pytest.raises(RuntimeError):
        handler(claimed)

    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        assert run is not None
        assert run.status == RunStatusValue.FAILED
        assert run.current_stage == "PROBE"
        assert run.error_code == ErrorCode.SYSTEM_FAILURE
