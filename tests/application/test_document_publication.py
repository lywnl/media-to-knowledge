from __future__ import annotations

import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_demo.application.document_publication import DocumentPublicationService
from video_demo.application.queries import ResultWriteFence
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
from video_demo.persistence.document_repository import DocumentResultRepository
from video_demo.persistence.models import (
    VideoSegmentModel,
    VideoSummaryModel,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository
from video_demo.storage.artifacts import AtomicArtifactStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result() -> VideoUnderstandingResult:
    chapter = SemanticChapter(
        chapter_id="chapter_001",
        start_ms=0,
        end_ms=1_000,
        title="无语义",
        title_evidence_refs=(),
        summary_zh="本时段未提取到可验证语义内容",
        summary_evidence_refs=(),
        body_blocks=(),
        claims=(),
        content_status="NO_SEMANTIC_EVIDENCE",
        evidence_refs=(),
        transcript_source="NONE",
        retrieval_text="",
        retrieval_hash=_digest(""),
    )
    return VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="测试知识文档",
            duration_ms=1_000,
            overview_zh="无可验证语义",
            key_points=(),
            retrieval_text="摘要检索",
            retrieval_hash=_digest("摘要检索"),
        ),
        sections=(
            SemanticSection(
                section_id=section_id_for("a" * 64, (chapter.chapter_id,)),
                title="全部内容",
                summary_zh="无可验证语义",
                chapter_refs=(chapter.chapter_id,),
            ),
        ),
        chapters=(chapter,),
        generation=DocumentGenerationMetadata(
            document_config=DocumentGenerationConfig(),
            text_model_id="text",
            vlm_model_id="vlm",
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


@pytest.fixture
def publication(
    tmp_path: Path,
) -> tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path]:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'document.db'}")
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=scope, asset_id="asset_001", object_ref="object_001", source_sha256="a" * 64
        )
        runs.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="object_001",
            idempotency_key="idempotency-001",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(scope=scope, job_id="job_001", run_id="run_001")
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    fence = ResultWriteFence(claimed.id, claimed.worker_id, claimed.attempt_count)
    return (
        DocumentPublicationService(database, AtomicArtifactStore(runtime_root)),
        database,
        scope,
        fence,
        runtime_root,
    )


def _publish(service: DocumentPublicationService, scope: Scope, fence: ResultWriteFence) -> None:
    service.persist_document(
        scope,
        _result(),
        evidence=(),
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        status="SUCCEEDED",
        transcript_source="NONE",
        fence=fence,
    )


@pytest.mark.parametrize(
    ("summary_version", "segment_version", "expected"),
    (
        ("1.0.0", "1.0.0", ErrorCode.RESULT_SCHEMA_UNSUPPORTED),
        ("2.0.0", "2.0.0", ErrorCode.RESULT_SCHEMA_UNSUPPORTED),
        ("1.0.0", "2.0.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("2.0.0", "3.0.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("3.0.0", "1.0.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("", "", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("4.0.0", "4.0.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
    ),
)
def test_document_repository_rejects_unsupported_or_mixed_version_matrix(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    summary_version: str,
    segment_version: str,
    expected: ErrorCode,
) -> None:
    _, database, scope, _, _ = publication
    with database.session() as session:
        repository = DocumentResultRepository(session)
        repository.replace(scope, _result())
        session.query(VideoSummaryModel).one().schema_version = summary_version
        session.query(VideoSegmentModel).one().schema_version = segment_version
    with database.session() as session, pytest.raises(VideoDemoError) as raised:
        DocumentResultRepository(session).get(scope, "run_001", "a" * 64)
    assert raised.value.code == expected


def test_document_repository_round_trips_3_result(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    _, database, scope, _, _ = publication
    with database.session() as session:
        repository = DocumentResultRepository(session)
        repository.replace(scope, _result())
        assert repository.get(scope, "run_001", "a" * 64) == _result()


def test_document_repository_validates_before_deleting_existing_rows(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    _, database, scope, _, _ = publication
    original = _result()
    with database.session() as session:
        DocumentResultRepository(session).replace(scope, original)
    forged_summary = original.summary.model_copy(update={"retrieval_hash": "b" * 64})
    forged = original.model_copy(update={"summary": forged_summary})

    with database.session() as session, pytest.raises(ValueError):
        DocumentResultRepository(session).replace(scope, forged)

    with database.session() as session:
        assert DocumentResultRepository(session).get(scope, "run_001", "a" * 64) == original


def test_document_publication_atomically_publishes_bundle_markdown_rows_and_job(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)

    document = service.get_document(scope, "run_001")
    assert document.startswith(b"# ")
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        job = JobRepository(session).get(scope, "job_001")
        assert run.artifact_manifest_relative_path and run.artifact_manifest_relative_path.endswith(
            ".json"
        )
        assert run.document_relative_path and run.document_relative_path.endswith(".md")
        assert run.document_sha256 == hashlib.sha256(document).hexdigest()
        assert run.document_size_bytes == len(document)
        assert job is not None and job.status.value == "SUCCEEDED"
    assert len(tuple((runtime_root / "runs").rglob("*.json"))) == 1
    assert len(tuple((runtime_root / "runs").rglob("*.md"))) == 1


@pytest.mark.parametrize("tamper", ["summary", "segment", "bundle", "markdown"])
def test_document_read_rejects_any_tampered_fact(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    tamper: str,
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        if tamper == "summary":
            session.query(VideoSummaryModel).one().retrieval_text = "被篡改"
        elif tamper == "segment":
            session.query(VideoSegmentModel).one().end_ms = 999
        elif tamper == "bundle":
            assert run.artifact_manifest_relative_path
            (runtime_root / run.artifact_manifest_relative_path).write_bytes(b"{}")
        else:
            assert run.document_relative_path
            (runtime_root / run.document_relative_path).write_bytes(b"tampered")

    with pytest.raises(VideoDemoError) as raised:
        service.get_document(scope, "run_001")
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_stale_worker_cannot_overwrite_current_document_publication(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, stale_fence, _ = publication
    with database.session() as session:
        stale = session.get(VideoUnderstandingRunModel, 1)
        assert stale is not None
        job = JobRepository(session).get(scope, "job_001")
        assert job is not None
        job.lease_expires_at = job.lease_expires_at.replace(year=2020)  # type: ignore[union-attr]
    with database.session() as session:
        current = JobRepository(session).claim("worker-b", lease_seconds=60)
    assert current is not None
    current_fence = ResultWriteFence(current.id, current.worker_id, current.attempt_count)
    _publish(service, scope, current_fence)
    before = service.get_document(scope, "run_001")

    with pytest.raises(VideoDemoError) as raised:
        _publish(service, scope, stale_fence)
    assert raised.value.code == ErrorCode.JOB_LEASE_LOST
    assert service.get_document(scope, "run_001") == before


def test_two_real_publishers_compete_and_exactly_one_wins_transaction(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, scope, fence, runtime_root = publication
    barrier = threading.Barrier(2)
    original = DocumentPublicationService._require_active_fence

    def synchronize(self: DocumentPublicationService, candidate: ResultWriteFence) -> None:
        original(self, candidate)
        barrier.wait(timeout=5)

    monkeypatch.setattr(DocumentPublicationService, "_require_active_fence", synchronize)

    def attempt() -> ErrorCode | None:
        try:
            _publish(service, scope, fence)
        except VideoDemoError as error:
            return error.code
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(2)))

    assert sorted(str(item) for item in outcomes) == sorted(
        (str(None), str(ErrorCode.JOB_LEASE_LOST))
    )
    assert service.get_document(scope, "run_001").startswith(b"# ")
    assert len(tuple((runtime_root / "runs").rglob("*.json"))) == 1
    assert len(tuple((runtime_root / "runs").rglob("*.md"))) == 1
    with database.session() as session:
        assert session.query(VideoSummaryModel).count() == 1
        assert session.query(VideoSegmentModel).count() == 1


def test_current_run_orphan_recovery_is_explicit_bounded_and_preserves_current_closure(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        assert run.artifact_manifest_relative_path and run.document_relative_path
        bundle = Path(run.artifact_manifest_relative_path)
        document = Path(run.document_relative_path)
    orphan_bundle = bundle.with_name(bundle.stem.rsplit("-", 1)[0] + "-" + "b" * 32 + ".json")
    orphan_document = document.with_name(document.stem.rsplit("-", 1)[0] + "-" + "b" * 32 + ".md")
    shutil.copyfile(runtime_root / bundle, runtime_root / orphan_bundle)
    shutil.copyfile(runtime_root / document, runtime_root / orphan_document)

    with pytest.raises(ValueError, match="停止"):
        service.recover_current_run_orphans(scope, "run_001", publishers_stopped=False)
    assert (
        service.recover_current_run_orphans(
            scope,
            "run_001",
            publishers_stopped=True,
            max_entries=4,
        )
        == 2
    )
    assert service.get_document(scope, "run_001").startswith(b"# ")
    assert (runtime_root / bundle).is_file()
    assert (runtime_root / document).is_file()


def test_current_run_orphan_recovery_fails_closed_on_unknown_or_excess_entries(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        assert run.document_relative_path
        result_root = (runtime_root / run.document_relative_path).parent
    unknown = result_root / "unknown.tmp"
    unknown.write_bytes(b"unknown")

    with pytest.raises(VideoDemoError) as unknown_error:
        service.recover_current_run_orphans(scope, "run_001", publishers_stopped=True)
    assert unknown_error.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert unknown.is_file()
    with pytest.raises(VideoDemoError) as bounded_error:
        service.recover_current_run_orphans(
            scope,
            "run_001",
            publishers_stopped=True,
            max_entries=2,
        )
    assert bounded_error.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_published_attempt_accepts_worker_outer_idempotent_completion(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, fence, _ = publication
    _publish(service, scope, fence)

    with database.session() as session:
        assert (
            JobRepository(session).mark_succeeded(
                fence.job_pk,
                fence.worker_id,
                attempt_count=fence.attempt_count,
            )
            is False
        )


def test_document_read_rejects_pointer_escape_symlink_and_invalid_size(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    tmp_path: Path,
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside")
    symlink = runtime_root / "linked.md"
    symlink.symlink_to(outside)

    invalid_values: tuple[tuple[str, object], ...] = (
        ("document_relative_path", "../outside.md"),
        ("document_relative_path", "linked.md"),
        ("document_size_bytes", 0),
        ("document_size_bytes", 16 * 1024 * 1024 + 1),
        ("document_sha256", "invalid"),
    )
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        original = {field: getattr(run, field) for field, _ in invalid_values}

    for field, value in invalid_values:
        with database.session() as session:
            run = session.query(VideoUnderstandingRunModel).one()
            setattr(run, field, value)
        with pytest.raises(VideoDemoError) as raised:
            service.get_document(scope, "run_001")
        assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
        with database.session() as session:
            run = session.query(VideoUnderstandingRunModel).one()
            setattr(run, field, original[field])


@pytest.mark.parametrize("kind", ["bundle", "document"])
def test_document_read_rejects_forged_publication_filename_digest(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    kind: str,
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        if kind == "bundle":
            assert run.artifact_manifest_relative_path
            source = Path(run.artifact_manifest_relative_path)
            forged = source.with_name(f"bundle-{'b' * 64}-{'c' * 32}.json")
            shutil.copyfile(runtime_root / source, runtime_root / forged)
            run.artifact_manifest_relative_path = forged.as_posix()
        else:
            assert run.document_relative_path
            source = Path(run.document_relative_path)
            forged = source.with_name(f"knowledge-note-{'b' * 64}-{'c' * 32}.md")
            shutil.copyfile(runtime_root / source, runtime_root / forged)
            run.document_relative_path = forged.as_posix()

    with pytest.raises(VideoDemoError) as raised:
        service.get_document(scope, "run_001")
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_document_read_rejects_run_status_or_warnings_tampering(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    service, database, scope, fence, _ = publication
    _publish(service, scope, fence)

    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        run.warning_codes = ["TAMPERED"]

    with pytest.raises(VideoDemoError) as raised:
        service.get_document(scope, "run_001")
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
