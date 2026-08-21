from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Thread
from typing import cast

import pytest
from sqlalchemy.orm import Session

from video_demo.application.queries import ResultQueryService, ResultWriteFence
from video_demo.domain.evidence import KeyframeEvidence, SpeechSegment
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


def _result(evidence_refs: tuple[str, ...] = ("asr_001",)) -> VideoUnderstandingResult:
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
        evidence_refs=evidence_refs,
        retrieval_text=segment_text,
        retrieval_hash=hashlib.sha256(segment_text.encode("utf-8")).hexdigest(),
    )
    summary_text = "类型：VIDEO_SUMMARY"
    summary = VideoSummary(
        title="测试视频",
        summary_zh="视频包含一段问候。",
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
    return VideoUnderstandingResult(
        run_id="run_001",
        asset_sha256="a" * 64,
        segments=(segment,),
        summary=summary,
    )


def _claim_fence(runtime_root: Path, worker_id: str = "worker-a") -> ResultWriteFence:
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    with database.session() as session:
        claimed = JobRepository(session).claim(worker_id, lease_seconds=60)
    assert claimed is not None
    return ResultWriteFence(
        job_pk=claimed.id,
        worker_id=claimed.worker_id,
        attempt_count=claimed.attempt_count,
    )


@pytest.fixture
def result_service(tmp_path: Path) -> tuple[ResultQueryService, Scope, Path]:
    runtime_root = tmp_path / ".codex" / "video-rag-demo"
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    runtime_root.mkdir(parents=True)
    database.create_schema()
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        run_repository = VideoRunRepository(session)
        run_repository.get_or_create_asset(
            scope=scope,
            asset_id="asset_001",
            object_ref="obj_001",
            source_sha256="a" * 64,
        )
        run_repository.add(
            scope=scope,
            run_id="run_001",
            asset_id="asset_001",
            object_ref="obj_001",
            idempotency_key="idempotency-key-0001",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_001",
            run_id="run_001",
        )
    return (
        ResultQueryService(
            database,
            AtomicArtifactStore(runtime_root),
            max_evidence_items=3,
        ),
        scope,
        runtime_root,
    )


def test_result_bundle_persists_segments_summary_evidence_and_status(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    service.persist(
        scope,
        _result(),
        evidence=(speech,),
        stage_metrics={"RESULT": 12},
        status=RunStatus.PARTIAL_SUCCEEDED,
        fence=_claim_fence(runtime_root),
        warnings=("WINDOW_FAILED",),
    )

    assert service.get_result(scope, "run_001") == _result()
    page = service.get_evidence(scope, "run_001", limit=10)
    assert page.items == (speech,)
    assert page.next_cursor is None
    assert service.get_run_metadata(scope, "run_001").status == RunStatus.PARTIAL_SUCCEEDED


def test_evidence_cursor_is_stable_and_filter_aware(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 100,
            end_ms=(index + 1) * 100,
            text=f"word-{index}",
            language="en",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(3)
    )
    service.persist(
        scope,
        _result(),
        evidence=evidence,
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=_claim_fence(runtime_root),
    )

    first = service.get_evidence(
        scope,
        "run_001",
        evidence_type="ASR_SEGMENT",
        start_ms=100,
        limit=1,
    )
    second = service.get_evidence(
        scope,
        "run_001",
        evidence_type="ASR_SEGMENT",
        start_ms=100,
        limit=1,
        cursor=first.next_cursor,
    )

    assert [item.evidence_id for item in first.items] == ["asr_001"]
    assert [item.evidence_id for item in second.items] == ["asr_002"]
    assert second.next_cursor is None


def test_keyframe_bytes_are_digest_checked_and_return_correct_mime(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    keyframe_bytes = b"\x89PNG\r\n\x1a\ncontent"
    relative_path = Path("runs") / service.scope_key(scope) / "run_001" / "keyframes" / "kf.png"
    keyframe_path = runtime_root / relative_path
    keyframe_path.parent.mkdir(parents=True)
    keyframe_path.write_bytes(keyframe_bytes)
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_ev_001",
        start_ms=0,
        end_ms=1_000,
        keyframe_id="keyframe_001",
        timestamp_ms=500,
        relative_path=relative_path.as_posix(),
        mime_type="image/png",
        sha256=hashlib.sha256(keyframe_bytes).hexdigest(),
        perceptual_hash="abcdef12",
    )
    service.persist(
        scope,
        _result(("keyframe_ev_001",)),
        evidence=(keyframe,),
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=_claim_fence(runtime_root),
    )

    content = service.get_keyframe(scope, "run_001", "keyframe_001")

    assert content.content == keyframe_bytes
    assert content.mime_type == "image/png"


def test_keyframe_path_cannot_escape_to_another_run(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    content = b"\x89PNG\r\n\x1a\nother-run"
    scope_root = Path("runs") / service.scope_key(scope)
    other_path = runtime_root / scope_root / "run_other" / "keyframes" / "kf.png"
    other_path.parent.mkdir(parents=True)
    other_path.write_bytes(content)
    traversal = scope_root / "run_001" / ".." / "run_other" / "keyframes" / "kf.png"
    keyframe = KeyframeEvidence(
        evidence_id="keyframe_ev_001",
        start_ms=0,
        end_ms=1_000,
        keyframe_id="keyframe_001",
        timestamp_ms=500,
        relative_path=traversal.as_posix(),
        mime_type="image/png",
        sha256=hashlib.sha256(content).hexdigest(),
        perceptual_hash="abcdef12",
    )
    service.persist(
        scope,
        _result((keyframe.evidence_id,)),
        evidence=(keyframe,),
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=_claim_fence(runtime_root),
    )

    with pytest.raises(VideoDemoError) as raised:
        service.get_keyframe(scope, "run_001", "keyframe_001")

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


def test_invalid_asset_digest_cannot_overwrite_existing_result_bundle(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    fence = _claim_fence(runtime_root)
    service.persist(
        scope,
        _result(),
        evidence=(speech,),
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=fence,
    )
    valid_result = service.get_result(scope, "run_001")
    invalid = _result().model_copy(update={"asset_sha256": "f" * 64})

    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            scope,
            invalid,
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=fence,
        )

    assert raised.value.code == ErrorCode.VIDEO_DIGEST_MISMATCH
    assert service.get_result(scope, "run_001") == valid_result
    assert service.get_evidence(scope, "run_001").items == (speech,)


def test_evidence_persistence_has_explicit_count_limit(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:05d}",
            start_ms=index,
            end_ms=index + 1,
            text="x",
            language="en",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(4)
    )
    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            scope,
            _result(),
            evidence=evidence,
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=_claim_fence(runtime_root),
        )

    assert raised.value.code == ErrorCode.ARTIFACT_SCHEMA_INVALID


def test_evidence_cursor_rejects_tampering_and_filter_reuse(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    evidence = tuple(
        SpeechSegment(
            evidence_id=f"asr_{index:03d}",
            start_ms=index * 100,
            end_ms=(index + 1) * 100,
            text=f"word-{index}",
            language="en",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        for index in range(2)
    )
    service.persist(
        scope,
        _result(),
        evidence=evidence,
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=_claim_fence(runtime_root),
    )
    cursor = service.get_evidence(scope, "run_001", limit=1).next_cursor
    assert cursor is not None

    for invalid_cursor, evidence_type in (
        (cursor[:-1] + ("0" if cursor[-1] != "0" else "1"), None),
        (cursor, "ASR_SEGMENT"),
    ):
        with pytest.raises(VideoDemoError) as raised:
            service.get_evidence(
                scope,
                "run_001",
                cursor=invalid_cursor,
                evidence_type=evidence_type,
                limit=1,
            )
        assert raised.value.code == ErrorCode.INVALID_EVIDENCE_CURSOR


def test_stale_worker_fence_cannot_publish_result(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    now = datetime.now(UTC)
    with database.session() as session:
        first = JobRepository(session).claim(
            "worker-old",
            lease_seconds=1,
            now=now,
        )
    assert first is not None
    with database.session() as session:
        replacement = JobRepository(session).claim(
            "worker-new",
            lease_seconds=60,
            now=now + timedelta(seconds=2),
        )
    assert replacement is not None
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            scope,
            _result(),
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=ResultWriteFence(
                job_pk=first.id,
                worker_id=first.worker_id,
                attempt_count=first.attempt_count,
            ),
        )

    assert raised.value.code == ErrorCode.JOB_LEASE_LOST
    with pytest.raises(VideoDemoError) as not_ready:
        service.get_result(scope, "run_001")
    assert not_ready.value.code == ErrorCode.VIDEO_RESULT_NOT_READY
    assert list(runtime_root.rglob("bundle-*.json")) == []


def test_same_fence_can_publish_only_one_result_bundle(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    _service, scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    both_bundles_written = Barrier(2)

    class SynchronizedArtifactStore(AtomicArtifactStore):
        def write_json(
            self,
            relative_path: Path,
            payload: dict[str, object] | list[object],
            *,
            schema_version: str,
            upstream_sha256: str,
        ) -> object:
            receipt = super().write_json(
                relative_path,
                payload,
                schema_version=schema_version,
                upstream_sha256=upstream_sha256,
            )
            both_bundles_written.wait(timeout=5)
            return receipt

    service = ResultQueryService(database, SynchronizedArtifactStore(runtime_root))
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    fence = ResultWriteFence(
        job_pk=claimed.id,
        worker_id=claimed.worker_id,
        attempt_count=claimed.attempt_count,
    )
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            service.persist(
                scope,
                _result(),
                evidence=(speech,),
                stage_metrics={"RESULT": 1},
                status=RunStatus.SUCCEEDED,
                fence=fence,
            )
        except BaseException as error:
            errors.append(error)

    publishers = [Thread(target=publish) for _ in range(2)]
    for publisher in publishers:
        publisher.start()
    for publisher in publishers:
        publisher.join(timeout=10)

    assert all(not publisher.is_alive() for publisher in publishers)
    assert len(errors) == 1
    assert isinstance(errors[0], VideoDemoError)
    assert errors[0].code == ErrorCode.JOB_LEASE_LOST
    assert service.get_result(scope, "run_001") == _result()
    assert len(list(runtime_root.rglob("bundle-*.json"))) == 1


def test_result_publish_rejects_runtime_none_fence_before_writing_bundle(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    with pytest.raises(ValueError, match="fence"):
        service.persist(
            scope,
            _result(),
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=cast(ResultWriteFence, None),
        )

    assert list(runtime_root.rglob("bundle-*.json")) == []


@pytest.mark.parametrize(
    "target_scope",
    (
        Scope("tenant-a", "app-a", "kb-a"),
        Scope("tenant-b", "app-a", "kb-a"),
    ),
    ids=("other-run", "other-scope"),
)
def test_result_publish_fence_must_belong_to_target_scope_and_run(
    result_service: tuple[ResultQueryService, Scope, Path],
    target_scope: Scope,
) -> None:
    _service, source_scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    target_run_id = "run_002"
    with database.session() as session:
        runs = VideoRunRepository(session)
        runs.get_or_create_asset(
            scope=target_scope,
            asset_id="asset_002",
            object_ref="obj_002",
            source_sha256="b" * 64,
        )
        runs.add(
            scope=target_scope,
            run_id=target_run_id,
            asset_id="asset_002",
            object_ref="obj_002",
            idempotency_key="idempotency-key-0002",
            config_snapshot={},
        )
        JobRepository(session).enqueue_video_run(
            scope=target_scope,
            job_id="job_002",
            run_id=target_run_id,
        )
    with database.session() as session:
        source_job = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert source_job is not None
    assert source_job.scope == source_scope
    service = ResultQueryService(database, AtomicArtifactStore(runtime_root))
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    target_result = _result().model_copy(
        update={"run_id": target_run_id, "asset_sha256": "b" * 64},
    )

    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            target_scope,
            target_result,
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=ResultWriteFence(
                job_pk=source_job.id,
                worker_id=source_job.worker_id,
                attempt_count=source_job.attempt_count,
            ),
        )

    assert raised.value.code == ErrorCode.JOB_LEASE_LOST
    with database.session() as session:
        source = JobRepository(session).get(source_scope, "job_001")
        target = JobRepository(session).get(target_scope, "job_002")
        target_run = VideoRunRepository(session).get(target_scope, target_run_id)
        assert source is not None
        assert target is not None
        assert target_run is not None
        assert source.status == JobStatus.RUNNING
        assert target.status == JobStatus.PENDING
        assert target_run.status == RunStatusValue.PENDING
    with pytest.raises(VideoDemoError) as not_ready:
        service.get_result(target_scope, target_run_id)
    assert not_ready.value.code == ErrorCode.VIDEO_RESULT_NOT_READY
    assert list(runtime_root.rglob("bundle-*.json")) == []


def test_cancellation_after_bundle_write_rolls_back_and_removes_unpublished_bundle(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    _service, scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None

    class CancellingArtifactStore(AtomicArtifactStore):
        def write_json(
            self,
            relative_path: Path,
            payload: dict[str, object] | list[object],
            *,
            schema_version: str,
            upstream_sha256: str,
        ) -> object:
            receipt = super().write_json(
                relative_path,
                payload,
                schema_version=schema_version,
                upstream_sha256=upstream_sha256,
            )
            with database.session() as session:
                assert JobRepository(session).request_cancel(scope, "job_001") is True
            return receipt

    service = ResultQueryService(database, CancellingArtifactStore(runtime_root))
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    with pytest.raises(VideoDemoError) as raised:
        service.persist(
            scope,
            _result(),
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=ResultWriteFence(
                job_pk=claimed.id,
                worker_id=claimed.worker_id,
                attempt_count=claimed.attempt_count,
            ),
        )

    assert raised.value.code == ErrorCode.JOB_CANCELLED
    assert list(runtime_root.rglob("bundle-*.json")) == []


def test_late_cancellation_cannot_overwrite_published_success(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    speech = SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="Hello",
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )
    service.persist(
        scope,
        _result(),
        evidence=(speech,),
        stage_metrics={},
        status=RunStatus.SUCCEEDED,
        fence=ResultWriteFence(
            job_pk=claimed.id,
            worker_id=claimed.worker_id,
            attempt_count=claimed.attempt_count,
        ),
    )

    with database.session() as session:
        assert JobRepository(session).request_cancel(scope, "job_001") is True

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        run = VideoRunRepository(session).get(scope, "run_001")
        assert job is not None
        assert run is not None
        assert job.status == JobStatus.SUCCEEDED
        assert job.cancel_requested is False
        assert run.status == RunStatusValue.SUCCEEDED
    assert service.get_result(scope, "run_001") == _result()
    assert len(list(runtime_root.rglob("bundle-*.json"))) == 1


def test_stale_cancel_reader_cannot_overwrite_published_success(
    result_service: tuple[ResultQueryService, Scope, Path],
) -> None:
    service, scope, runtime_root = result_service
    database = Database(f"sqlite+pysqlite:///{runtime_root / 'result.db'}")
    with database.session() as session:
        claimed = JobRepository(session).claim("worker-a", lease_seconds=60)
    assert claimed is not None
    stale_session = Session(bind=database.engine, expire_on_commit=False)
    try:
        stale_repository = JobRepository(stale_session)
        stale = stale_repository.get(scope, "job_001")
        assert stale is not None
        assert stale.status == JobStatus.RUNNING
        stale_session.commit()

        speech = SpeechSegment(
            evidence_id="asr_001",
            start_ms=0,
            end_ms=1_000,
            text="Hello",
            language="en",
            confidence=0.9,
            is_fully_evaluated_language=True,
        )
        service.persist(
            scope,
            _result(),
            evidence=(speech,),
            stage_metrics={},
            status=RunStatus.SUCCEEDED,
            fence=ResultWriteFence(
                job_pk=claimed.id,
                worker_id=claimed.worker_id,
                attempt_count=claimed.attempt_count,
            ),
        )

        assert stale_repository.request_cancel(scope, "job_001") is True
        stale_session.commit()
    finally:
        stale_session.close()

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_001")
        run = VideoRunRepository(session).get(scope, "run_001")
        assert job is not None
        assert run is not None
        assert job.status == JobStatus.SUCCEEDED
        assert job.cancel_requested is False
        assert run.status == RunStatusValue.SUCCEEDED
