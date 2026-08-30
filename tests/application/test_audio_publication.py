from __future__ import annotations

from video_demo.application.audio_publication import AudioPublicationService
from video_demo.application.audio_rendering import render_audio_markdown
from video_demo.application.document_publication import ResultWriteFence
from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import AudioUnderstandingRunModel
from video_demo.persistence.repositories import JobRepository
from video_demo.persistence.scope import Scope
from video_demo.storage.artifacts import AtomicArtifactStore


def test_audio_publication_service_is_explicit_audio_type(tmp_path) -> None:
    service = AudioPublicationService(
        Database(f"sqlite+pysqlite:///{tmp_path / 'audio.db'}"),
        AtomicArtifactStore(tmp_path),
    )

    assert service.database.engine is not None


def test_audio_publication_round_trips_bundle_markdown_and_audio_rows(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'audio.db'}")
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    result = AudioUnderstandingResult(
        run_id="run_audio_001",
        asset_sha256="a" * 64,
        summary=AudioDocumentSummary(title="音频", duration_ms=1_000, overview_zh="概览"),
        chapters=(
            AudioChapter(
                chapter_id="audio_chapter_001",
                start_ms=0,
                end_ms=1_000,
                title="章节",
                title_evidence_refs=("asr_001",),
                summary_zh="摘要",
                summary_evidence_refs=("asr_001",),
                body_blocks=(),
                claims=(),
                evidence_refs=("asr_001",),
                transcript_source="ASR",
            ),
        ),
    )
    with database.session() as session:
        MediaRunRepository(session, AudioUnderstandingRunModel).add(
            scope=scope,
            run_id=result.run_id,
            object_ref="obj_audio_001",
            idempotency_key="idempotency-audio-001",
            config_snapshot={"result_schema_version": "1.0.0"},
        )
        JobRepository(session).enqueue_media_run(
            scope=scope,
            job_id="job_audio_001",
            resource_id=result.run_id,
            job_type="AUDIO_UNDERSTANDING",
            resource_type="AUDIO_UNDERSTANDING_RUN",
        )
    with database.session() as session:
        claimed = JobRepository(session).claim(
            "audio-worker",
            lease_seconds=60,
            job_type="AUDIO_UNDERSTANDING",
        )
    assert claimed is not None
    service = AudioPublicationService(database, AtomicArtifactStore(runtime_root))

    service.persist(
        scope,
        result,
        document=render_audio_markdown(result),
        status="PARTIAL_SUCCEEDED",
        warnings=("AUDIO_GLOBAL_WRITING_FALLBACK",),
        fence=ResultWriteFence(claimed.id, claimed.worker_id, claimed.attempt_count),
    )

    publication = service.get(scope, result.run_id)
    assert publication.result == result
    assert publication.document == render_audio_markdown(result).content
