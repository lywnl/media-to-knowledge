from __future__ import annotations

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_demo.application.document_publication import (
    DocumentPublicationService,
    ResultWriteFence,
    scope_key,
)
from video_demo.application.document_rendering import render_markdown
from video_demo.domain.document import (
    DocumentGenerationConfig,
    DocumentGenerationMetadata,
    PromptVersions,
    SemanticChapter,
    VideoDocumentSummary,
    VideoUnderstandingResult,
)
from video_demo.domain.document_artifact import MODEL_METRIC_NAMES, RESULT_STAGE_NAMES
from video_demo.domain.evidence import KeyframeEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.database import Database
from video_demo.persistence.document_repository import ResultRepository
from video_demo.persistence.models import (
    VideoSegmentModel,
    VideoSummaryModel,
    VideoUnderstandingRunModel,
)
from video_demo.persistence.repositories import JobRepository, Scope, VideoRunRepository
from video_demo.storage.artifacts import ArtifactBytesReceipt, ArtifactReceipt, AtomicArtifactStore


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
    )
    return VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        summary=VideoDocumentSummary(
            title="测试知识文档",
            duration_ms=1_000,
            overview_zh="无可验证语义",
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
    result = _result()
    service.persist(
        scope,
        result,
        evidence=(),
        document=render_markdown(result, ()),
        stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
        model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
        status="SUCCEEDED",
        transcript_source="NONE",
        fence=fence,
    )


class _RecordingVisualCleaner:
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        self.fail_cleanup = fail_cleanup
        self.cleanup_calls: list[tuple[Path, tuple[KeyframeEvidence, ...]]] = []
        self.pending_calls: list[Path] = []

    def cleanup(
        self,
        run_relative_root: Path,
        keyframes: tuple[KeyframeEvidence, ...],
    ) -> bool:
        self.cleanup_calls.append((run_relative_root, keyframes))
        if self.fail_cleanup:
            raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模拟清理失败")
        return True

    def mark_pending(self, run_relative_root: Path) -> None:
        self.pending_calls.append(run_relative_root)


def _keyframe() -> KeyframeEvidence:
    return KeyframeEvidence(
        evidence_id="keyframe_evidence_001",
        start_ms=500,
        end_ms=501,
        keyframe_id="keyframe_001",
        timestamp_ms=500,
        relative_path=f"visual/keyframes/{'b' * 64}.jpg",
        mime_type="image/jpeg",
        sha256="b" * 64,
        size_bytes=1_024,
    )


def test_document_publication_rejects_precommit_visual_closure_mismatch_without_cleanup(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    _, database, scope, fence, runtime_root = publication
    cleaner = _RecordingVisualCleaner()
    service = DocumentPublicationService(
        database,
        AtomicArtifactStore(runtime_root),
        visual_cleaner=cleaner,
    )
    result = _result()

    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            scope,
            result,
            evidence=(),
            document=render_markdown(result, ()),
            stage_metrics=dict.fromkeys(RESULT_STAGE_NAMES, 0),
            model_metrics=dict.fromkeys(MODEL_METRIC_NAMES, 0),
            status="SUCCEEDED",
            transcript_source="NONE",
            fence=fence,
            published_keyframes=(_keyframe(),),
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert cleaner.cleanup_calls == []
    assert cleaner.pending_calls == []
    with database.session() as session:
        assert session.query(VideoSummaryModel).count() == 0


def test_document_publication_marks_pending_when_committed_bundle_reread_fails(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, database, scope, fence, runtime_root = publication
    cleaner = _RecordingVisualCleaner()
    service = DocumentPublicationService(
        database,
        AtomicArtifactStore(runtime_root),
        visual_cleaner=cleaner,
    )

    def fail_reread(_scope: Scope, _run_id: str) -> object:
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "模拟提交后重读失败")

    monkeypatch.setattr(service, "get_artifact", fail_reread)
    _publish(service, scope, fence)

    expected_root = Path("runs") / scope_key(scope) / "run_001"
    assert cleaner.cleanup_calls == []
    assert cleaner.pending_calls == [expected_root]
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        assert run.status.value == "SUCCEEDED"


@pytest.mark.parametrize(
    ("summary_version", "segment_version", "expected"),
    (
        ("1.0.0", "1.0.0", ErrorCode.RESULT_SCHEMA_UNSUPPORTED),
        ("2.0.0", "2.0.0", ErrorCode.RESULT_SCHEMA_UNSUPPORTED),
        ("1.0.0", "2.0.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("2.0.0", "4.2.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("3.0.0", "4.2.0", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("", "", ErrorCode.ARTIFACT_SCHEMA_INVALID),
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
        repository = ResultRepository(session)
        repository.replace(scope, _result())
        session.query(VideoSummaryModel).one().schema_version = summary_version
        session.query(VideoSegmentModel).one().schema_version = segment_version
    with database.session() as session, pytest.raises(VideoDemoError) as raised:
        ResultRepository(session).get(scope, "run_001", "a" * 64)
    assert raised.value.code == expected


def test_document_repository_round_trips_4_result(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    _, database, scope, _, _ = publication
    with database.session() as session:
        repository = ResultRepository(session)
        repository.replace(scope, _result())
        assert repository.get(scope, "run_001", "a" * 64) == _result()


@pytest.mark.parametrize(
    ("remaining", "expected"),
    (
        ("NONE", ErrorCode.VIDEO_RESULT_NOT_READY),
        ("SUMMARY", ErrorCode.ARTIFACT_SCHEMA_INVALID),
        ("SEGMENT", ErrorCode.ARTIFACT_SCHEMA_INVALID),
    ),
)
def test_document_repository_distinguishes_empty_and_partial_rows(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    remaining: str,
    expected: ErrorCode,
) -> None:
    _, database, scope, _, _ = publication
    with database.session() as session:
        ResultRepository(session).replace(scope, _result())
        if remaining in {"NONE", "SUMMARY"}:
            session.query(VideoSegmentModel).delete()
        if remaining in {"NONE", "SEGMENT"}:
            session.query(VideoSummaryModel).delete()
    with database.session() as session, pytest.raises(VideoDemoError) as raised:
        ResultRepository(session).get(scope, "run_001", "a" * 64)
    assert raised.value.code == expected


def test_document_repository_validates_before_deleting_existing_rows(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
) -> None:
    _, database, scope, _, _ = publication
    original = _result()
    with database.session() as session:
        ResultRepository(session).replace(scope, original)
    forged_summary = original.summary.model_copy(update={"title": ""})
    forged = original.model_copy(update={"summary": forged_summary})

    with database.session() as session, pytest.raises(ValueError):
        ResultRepository(session).replace(scope, forged)

    with database.session() as session:
        assert ResultRepository(session).get(scope, "run_001", "a" * 64) == original


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
            session.query(VideoSummaryModel).one().payload_json = {"summary": {"bad": "payload"}}
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


def test_stale_lease_owner_cannot_overwrite_current_document_publication(
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
    original = DocumentPublicationService._commit
    written: list[tuple[str, str]] = []
    reached_commit: set[int] = set()
    written_lock = threading.Lock()

    def synchronize(
        self: DocumentPublicationService,
        candidate_scope: Scope,
        result: VideoUnderstandingResult,
        candidate: ResultWriteFence,
        status: str,
        warnings: tuple[str, ...],
        document: ArtifactBytesReceipt,
        bundle: ArtifactReceipt,
    ) -> None:
        with written_lock:
            written.append((bundle.relative_path, document.relative_path))
            reached_commit.add(threading.get_ident())
        barrier.wait(timeout=5)
        original(self, candidate_scope, result, candidate, status, warnings, document, bundle)

    monkeypatch.setattr(DocumentPublicationService, "_commit", synchronize)

    def attempt() -> tuple[ErrorCode | None, str | None]:
        try:
            _publish(service, scope, fence)
        except VideoDemoError as error:
            return error.code, None
        except Exception as error:
            return None, type(error).__name__
        return None, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(2)))

    assert len(reached_commit) == 2
    assert sorted(outcomes, key=lambda value: value[0] is not None) == [
        (None, None),
        (ErrorCode.JOB_LEASE_LOST, None),
    ]
    assert len(written) == 2
    assert service.get_document(scope, "run_001").startswith(b"# ")
    assert len(tuple((runtime_root / "runs").rglob("*.json"))) == 1
    assert len(tuple((runtime_root / "runs").rglob("*.md"))) == 1
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        winning = (run.artifact_manifest_relative_path, run.document_relative_path)
        assert session.query(VideoSummaryModel).count() == 1
        assert session.query(VideoSegmentModel).count() == 1
    assert winning in written
    losing = next(pair for pair in written if pair != winning)
    assert all((runtime_root / path).is_file() for path in winning if path is not None)
    assert all(not (runtime_root / path).exists() for path in losing)


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


@pytest.mark.parametrize("tamper", ["envelope", "schema", "upstream", "run", "asset", "evidence"])
def test_current_run_orphan_recovery_rejects_invalid_or_cross_target_bundle(
    publication: tuple[DocumentPublicationService, Database, Scope, ResultWriteFence, Path],
    tamper: str,
) -> None:
    service, database, scope, fence, runtime_root = publication
    _publish(service, scope, fence)
    with database.session() as session:
        run = session.query(VideoUnderstandingRunModel).one()
        assert run.artifact_manifest_relative_path
        bundle = runtime_root / run.artifact_manifest_relative_path
    envelope = json.loads(bundle.read_bytes())
    payload = envelope["payload"]
    if tamper == "envelope":
        envelope["unknown"] = True
    elif tamper == "schema":
        envelope["schema_version"] = "2.0.0"
    elif tamper == "upstream":
        envelope["upstream_sha256"] = "b" * 64
    elif tamper == "run":
        payload["result"]["run_id"] = "run_other"
    elif tamper == "asset":
        payload["result"]["asset_sha256"] = "b" * 64
    else:
        payload["result"]["chapters"][0]["evidence_refs"] = ["missing_evidence"]
    payload_digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    orphan = bundle.with_name(f"bundle-{payload_digest}-{'b' * 32}.json")
    orphan.write_bytes(
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    with pytest.raises(VideoDemoError) as raised:
        service.recover_current_run_orphans(
            scope,
            "run_001",
            publishers_stopped=True,
            max_entries=3,
        )
    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID
    assert orphan.is_file()


def test_published_attempt_accepts_lease_owner_idempotent_completion(
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
