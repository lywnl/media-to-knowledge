from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from video_demo.persistence.database import Database
from video_demo.persistence.models import VideoObjectModel, VideoObjectStatus
from video_demo.persistence.repositories import JobRepository, Scope, VideoObjectRepository


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    database.create_schema()
    return database


def test_video_object_lookup_is_limited_to_full_scope(database: Database) -> None:
    scope = Scope("tenant-a", "app-a", "kb-a")
    other_scope = Scope("tenant-b", "app-a", "kb-a")
    with database.session() as session:
        repository = VideoObjectRepository(session)
        repository.add_ready(
            scope=scope,
            object_ref="obj_001",
            original_filename="lesson.mp4",
            declared_mime="video/mp4",
            detected_mime="video/mp4",
            size_bytes=1024,
            sha256="a" * 64,
            relative_path="objects/obj_001/source.mp4",
        )

    with database.session() as session:
        repository = VideoObjectRepository(session)
        assert repository.get_ready(scope, "obj_001") is not None
        assert repository.get_ready(other_scope, "obj_001") is None


def test_same_object_ref_can_exist_in_different_scope(database: Database) -> None:
    scope_a = Scope("tenant-a", "app-a", "kb-a")
    scope_b = Scope("tenant-b", "app-a", "kb-a")
    with database.session() as session:
        repository = VideoObjectRepository(session)
        for scope in (scope_a, scope_b):
            repository.add_ready(
                scope=scope,
                object_ref="obj_shared",
                original_filename="lesson.mp4",
                declared_mime="video/mp4",
                detected_mime="video/mp4",
                size_bytes=1024,
                sha256="a" * 64,
                relative_path=f"objects/{scope.tenant_id}/source.mp4",
            )

    with database.session() as session:
        count = len(session.scalars(select(VideoObjectModel)).all())
        assert count == 2


def test_duplicate_object_ref_inside_same_scope_is_rejected(database: Database) -> None:
    scope = Scope("tenant-a", "app-a", "kb-a")
    with pytest.raises(IntegrityError), database.session() as session:
        for relative_path in ("objects/one.mp4", "objects/two.mp4"):
            VideoObjectRepository(session).add_ready(
                scope=scope,
                object_ref="obj_duplicate",
                original_filename="lesson.mp4",
                declared_mime="video/mp4",
                detected_mime="video/mp4",
                size_bytes=1024,
                sha256="a" * 64,
                relative_path=relative_path,
            )


def test_session_rolls_back_after_exception(database: Database) -> None:
    with pytest.raises(RuntimeError), database.session() as session:
        session.add(
            VideoObjectModel(
                tenant_id="tenant-a",
                application_id="app-a",
                knowledge_base_id="kb-a",
                object_ref="obj_rollback",
                original_filename="lesson.mp4",
                declared_mime="video/mp4",
                detected_mime="video/mp4",
                size_bytes=1024,
                sha256="a" * 64,
                relative_path="objects/rollback.mp4",
                status=VideoObjectStatus.READY,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            ),
        )
        raise RuntimeError("force rollback")

    with database.session() as session:
        assert session.scalar(
            select(VideoObjectModel).where(VideoObjectModel.object_ref == "obj_rollback"),
        ) is None


def test_persisted_json_fields_reject_secret_like_keys(database: Database) -> None:
    with database.session() as session:
        repository = VideoObjectRepository(session)
        with pytest.raises(ValueError, match="敏感字段"):
            repository.update_scan_details(
                scope=Scope("tenant-a", "app-a", "kb-a"),
                object_ref="obj_missing",
                details={"api_key": "should-not-persist"},
            )


def test_sqlite_datetime_round_trip_restores_utc_timezone(database: Database) -> None:
    now = datetime(2026, 8, 17, tzinfo=UTC)
    scope = Scope("tenant-a", "app-a", "kb-a")
    with database.session() as session:
        JobRepository(session).enqueue_video_run(
            scope=scope,
            job_id="job_time",
            run_id="run_time",
            now=now,
        )

    with database.session() as session:
        job = JobRepository(session).get(scope, "job_time")
        assert job is not None
        assert job.next_attempt_at == now
        assert job.next_attempt_at.tzinfo is UTC
