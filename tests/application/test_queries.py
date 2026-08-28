from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_demo.application.document_publication import ResultWriteFence, scope_key
from video_demo.application.document_rendering import render_markdown
from video_demo.application.queries import ResultQueryService
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    GroundedClaim,
    ParagraphBlock,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.domain.evidence import SpeechSegment
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.models import VideoSegmentModel, VideoSummaryModel
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository
from video_demo.storage.artifacts import AtomicArtifactStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result() -> tuple[VideoUnderstandingResult, SpeechSegment]:
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="faster-whisper 提供高效语音识别。",
        language="zh",
        confidence=0.99,
        is_fully_evaluated_language=True,
    )
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="模型概览",
        title_evidence_refs=(speech.evidence_id,),
        summary_zh="介绍 faster-whisper。",
        summary_evidence_refs=(speech.evidence_id,),
        body_blocks=(
            ParagraphBlock(
                text="faster-whisper 提供高效语音识别。",
                evidence_refs=(speech.evidence_id,),
            ),
        ),
        claims=(
            GroundedClaim(
                text="faster-whisper 用于语音识别。",
                evidence_refs=(speech.evidence_id,),
                certainty=0.99,
            ),
        ),
        evidence_refs=(speech.evidence_id,),
        transcript_source="ASR",
        retrieval_text="faster-whisper 语音识别",
        retrieval_hash=_digest("faster-whisper 语音识别"),
    )
    result = VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="faster-whisper 教程",
            duration_ms=1_000,
            overview_zh="介绍模型用途。",
            key_points=(),
            retrieval_text="模型用途",
            retrieval_hash=_digest("模型用途"),
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
    return result, speech


@pytest.fixture
def service(
    tmp_path: Path,
) -> tuple[ResultQueryService, Database, Scope, ResultWriteFence, Path]:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'queries.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
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
            idempotency_key="query-001",
            config_snapshot={"result_schema_version": "3.0.0"},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    return (
        ResultQueryService(database, AtomicArtifactStore(runtime_root)),
        database,
        scope,
        ResultWriteFence(claimed.id, claimed.worker_id, claimed.attempt_count),
        runtime_root,
    )


def _publish(
    service: ResultQueryService,
    scope: Scope,
    fence: ResultWriteFence,
) -> tuple[VideoUnderstandingResult, SpeechSegment]:
    result, speech = _result()
    service.persist(
        scope,
        result,
        evidence=(speech,),
        document=render_markdown(result, (speech,)),
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        status="SUCCEEDED",
        transcript_source="ASR",
        fence=fence,
    )
    return result, speech


def test_query_service_round_trips_result_document_evidence_and_public_artifact(
    service: tuple[ResultQueryService, Database, Scope, ResultWriteFence, Path],
) -> None:
    queries, _database, scope, fence, _runtime_root = service
    result, speech = _publish(queries, scope, fence)

    artifact, document = queries.get_artifact(scope, result.run_id)

    assert queries.get_result(scope, result.run_id) == result
    assert queries.get_document(scope, result.run_id) == document
    assert queries.get_evidence(scope, result.run_id).items == (speech,)
    assert artifact.result == result
    assert artifact.evidence == (speech,)
    assert artifact.document_sha256 == hashlib.sha256(document).hexdigest()
    assert artifact.document_size_bytes == len(document)


def test_query_service_rejects_old_uniform_result_rows_as_unsupported(
    service: tuple[ResultQueryService, Database, Scope, ResultWriteFence, Path],
) -> None:
    queries, database, scope, fence, _runtime_root = service
    _publish(queries, scope, fence)
    with database.session() as session:
        session.query(VideoSummaryModel).one().schema_version = "2.0.0"
        session.query(VideoSegmentModel).one().schema_version = "2.0.0"

    for query in (
        lambda: queries.get_result(scope, "run_001"),
        lambda: queries.get_document(scope, "run_001"),
        lambda: queries.get_evidence(scope, "run_001"),
        lambda: queries.get_keyframe(scope, "run_001", "missing"),
    ):
        with pytest.raises(VideoDemoError) as raised:
            query()
        assert raised.value.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED


@pytest.mark.parametrize("tamper", ["mixed_rows", "bundle_payload"])
def test_query_service_rejects_mixed_or_corrupted_bundle_as_artifact_invalid(
    service: tuple[ResultQueryService, Database, Scope, ResultWriteFence, Path],
    tamper: str,
) -> None:
    queries, database, scope, fence, runtime_root = service
    _publish(queries, scope, fence)
    if tamper == "mixed_rows":
        with database.session() as session:
            session.query(VideoSegmentModel).one().schema_version = "2.0.0"
    else:
        with database.session() as session:
            run = VideoRunRepository(session).get(scope, "run_001")
            assert run is not None and run.artifact_manifest_relative_path
            path = runtime_root / run.artifact_manifest_relative_path
        envelope = json.loads(path.read_bytes())
        envelope["payload"]["artifact_schema_version"] = "2.0.0"
        path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(VideoDemoError) as raised:
        queries.get_artifact(scope, "run_001")
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_query_service_isolates_scope_and_run(
    service: tuple[ResultQueryService, Database, Scope, ResultWriteFence, Path],
) -> None:
    queries, _database, scope, fence, _runtime_root = service
    _publish(queries, scope, fence)

    for foreign_scope, run_id in (
        (Scope("tenant-b", "app-a", "kb-a"), "run_001"),
        (scope, "run_other"),
    ):
        with pytest.raises(VideoDemoError):
            queries.get_artifact(foreign_scope, run_id)

    assert scope_key(scope) not in queries.get_document(scope, "run_001").decode("utf-8")
