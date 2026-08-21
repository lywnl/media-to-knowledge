from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from video_demo.application.pipeline import (
    PipelineContext,
    PipelineJobHandler,
    PipelineOutcome,
    StageMetric,
)
from video_demo.application.queries import ResultQueryService
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.result import (
    SummaryChapter,
    VideoSegment,
    VideoSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.run import RunStatus
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import JobStatus, RunStatusValue
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.worker.runtime import ReliableWorker


def test_pipeline_job_handler_only_builds_context_persists_outcome_and_status(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'worker-pipeline.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="pipeline-worker-0001",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    captured: list[PipelineContext] = []
    outcome = _outcome()

    class Pipeline:
        def run(self, context: PipelineContext) -> PipelineOutcome:
            captured.append(context)
            assert context.is_cancel_requested() is False
            return outcome

    queries = ResultQueryService(database, AtomicArtifactStore(runtime_root))
    handler = PipelineJobHandler(database, Pipeline(), queries)  # type: ignore[arg-type]

    handler(claimed)

    assert [context.run_id for context in captured] == ["run_001"]
    assert queries.get_result(scope, "run_001") == outcome.result
    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        job = JobRepository(session).get(scope, "job_001")
        assert run is not None
        assert job is not None
        assert run.status == RunStatusValue.SUCCEEDED
        assert run.current_stage == "RESULT"
        assert job.status == JobStatus.SUCCEEDED
        assert job.worker_id is None


def test_pipeline_job_handler_tracks_stage_and_closes_unexpected_failure(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'worker-failure.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="pipeline-worker-failure",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None

    class Pipeline:
        def run(self, context: PipelineContext) -> PipelineOutcome:
            context.on_stage_start("PROBE")
            raise RuntimeError("不得持久化的内部错误正文")

    handler = PipelineJobHandler(
        database,
        Pipeline(),  # type: ignore[arg-type]
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
    )

    try:
        handler(claimed)
    except RuntimeError:
        pass
    else:
        raise AssertionError("未分类异常必须继续抛给 ReliableWorker")

    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        assert run is not None
        assert run.status == RunStatusValue.FAILED
        assert run.current_stage == "PROBE"
        assert run.error_code == ErrorCode.SYSTEM_FAILURE


def test_stale_pipeline_handler_cannot_overwrite_replacement_run_status(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'worker-stale.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="pipeline-worker-stale",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    now = datetime.now(UTC)
    with database.session() as session:
        stale = JobRepository(session).claim("worker-old", lease_seconds=1, now=now)
    assert stale is not None
    with database.session() as session:
        replacement = JobRepository(session).claim(
            "worker-new",
            lease_seconds=60,
            now=now + timedelta(seconds=2),
        )
        run = VideoRunRepository(session).get(scope, "run_001")
        assert run is not None
        run.status = RunStatusValue.RUNNING
        run.current_stage = "PROBE"
    assert replacement is not None

    class Pipeline:
        def run(self, _context: PipelineContext) -> PipelineOutcome:
            raise VideoDemoError(ErrorCode.JOB_LEASE_LOST, "旧租约失效")

    handler = PipelineJobHandler(
        database,
        Pipeline(),  # type: ignore[arg-type]
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
    )

    try:
        handler(stale)
    except VideoDemoError as error:
        assert error.code == ErrorCode.JOB_LEASE_LOST
    else:
        raise AssertionError("旧 Worker 必须感知租约丢失")

    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        assert run is not None
        assert run.status == RunStatusValue.RUNNING
        assert run.current_stage == "PROBE"
        assert run.error_code is None


def test_cancellation_after_pipeline_result_closes_job_and_run_without_bundle(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'worker-cancel.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="pipeline-worker-cancel",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None

    class Pipeline:
        def run(self, _context: PipelineContext) -> PipelineOutcome:
            with database.session() as session:
                assert JobRepository(session).request_cancel(scope, "job_001") is True
            return _outcome()

    handler = PipelineJobHandler(
        database,
        Pipeline(),  # type: ignore[arg-type]
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
    )

    try:
        handler(claimed)
    except VideoDemoError as error:
        assert error.code == ErrorCode.JOB_CANCELLED
    else:
        raise AssertionError("取消必须阻止结果发布")

    with database.session() as session:
        run = VideoRunRepository(session).get(scope, "run_001")
        job = JobRepository(session).get(scope, "job_001")
        assert run is not None
        assert job is not None
        assert run.status == RunStatusValue.CANCELLED
        assert run.error_code == ErrorCode.JOB_CANCELLED
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.JOB_CANCELLED
    assert list(runtime_root.rglob("bundle-*.json")) == []


@pytest.mark.parametrize(
    "retryable_code",
    (
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.SPEECH_SUBPROCESS_TIMEOUT,
        ErrorCode.SPEECH_SUBPROCESS_CRASHED,
    ),
)
def test_worker_failure_cleanup_is_noop_when_cancel_wins_after_pipeline_retry(
    tmp_path: Path,
    retryable_code: ErrorCode,
) -> None:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    runtime_root.mkdir(parents=True)
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'worker-retry-cancel.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="pipeline-worker-retry-cancel",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )

    class Pipeline:
        def run(self, _context: PipelineContext) -> PipelineOutcome:
            raise VideoDemoError(
                retryable_code,
                "依赖暂时不可用",
            )

    pipeline_handler = PipelineJobHandler(
        database,
        Pipeline(),  # type: ignore[arg-type]
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
    )

    def cancel_after_pipeline_retry(job: object) -> None:
        try:
            pipeline_handler(job)  # type: ignore[arg-type]
        except VideoDemoError:
            with database.session() as session:
                retry_wait = JobRepository(session).get(scope, "job_001")
                assert retry_wait is not None
                assert retry_wait.status == JobStatus.RETRY_WAIT
                assert JobRepository(session).request_cancel(scope, "job_001") is True
            raise

    worker = ReliableWorker(database, "worker-a", cancel_after_pipeline_retry)

    assert worker.run_once() is True

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        run = VideoRunRepository(session).get(scope, "run_001")
        assert job is not None
        assert run is not None
        assert job.status == JobStatus.CANCELLED
        assert job.error_code == ErrorCode.JOB_CANCELLED
        assert run.status == RunStatusValue.CANCELLED
        assert run.error_code == ErrorCode.JOB_CANCELLED


def _outcome() -> PipelineOutcome:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    segment_text = "类型：VIDEO_SEGMENT"
    segment = VideoSegment(
        segment_id="segment_001",
        start_ms=0,
        end_ms=1_000,
        title="问候",
        summary_zh="讲者问好。",
        languages=("en",),
        topics=("问候",),
        keywords=("问候",),
        original_keywords=("Hello",),
        evidence_refs=(speech.evidence_id,),
        retrieval_text=segment_text,
        retrieval_hash=hashlib.sha256(segment_text.encode("utf-8")).hexdigest(),
    )
    summary_text = "类型：VIDEO_SUMMARY"
    summary = VideoSummary(
        title="测试视频",
        summary_zh="视频包含问候。",
        duration_ms=1_000,
        chapters=(
            SummaryChapter(
                title="问候",
                start_ms=0,
                end_ms=1_000,
                segment_ids=(segment.segment_id,),
            ),
        ),
        languages=("en",),
        topics=("问候",),
        keywords=("问候",),
        original_keywords=("Hello",),
        retrieval_text=summary_text,
        retrieval_hash=hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
    )
    return PipelineOutcome(
        status=RunStatus.SUCCEEDED,
        transcript_source="ASR",
        result=VideoUnderstandingResult(
            run_id="run_001",
            asset_sha256="a" * 64,
            segments=(segment,),
            summary=summary,
        ),
        evidence=(speech,),
        warnings=(),
        stage_metrics=(StageMetric(stage="RESULT", duration_ms=1),),
    )
