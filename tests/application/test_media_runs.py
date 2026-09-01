from __future__ import annotations

from pathlib import Path

from sqlalchemy import update

from video_demo.application.media_runs import MediaRunService
from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository
from video_demo.persistence.models import (
    ImageObjectModel,
    ImageUnderstandingRunModel,
    JobModel,
    JobStatus,
    RunStatusValue,
)
from video_demo.persistence.scope import Scope


class _Scheduler:
    def __init__(self) -> None:
        self.submissions: list[tuple[Scope, str]] = []

    def submit(self, scope: Scope, run_id: str) -> str:
        self.submissions.append((scope, run_id))
        return "accepted"


def test_idempotent_pending_image_run_is_submitted_again(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'media-runs.db'}")
    database.create_schema()
    scope = Scope("tenant", "application", "kb")
    object_ref = "obj_image_idempotent"
    with database.session() as session:
        MediaObjectRepository(session, ImageObjectModel).add_ready(
            scope=scope,
            object_ref=object_ref,
            original_filename="pixel.png",
            declared_mime="image/png",
            detected_mime="image/png",
            size_bytes=1,
            sha256="a" * 64,
            relative_path="image_object/object/source.png",
        )

    scheduler = _Scheduler()
    service = MediaRunService(database, scheduler)
    first = service.create(
        scope=scope,
        object_ref=object_ref,
        idempotency_key="image-request-001",
    )
    scheduler.submissions.clear()

    replay = service.create(
        scope=scope,
        object_ref=object_ref,
        idempotency_key="image-request-001",
    )

    assert replay == first
    assert scheduler.submissions == [(scope, first.run_id)]


def test_retry_failed_image_run_resets_job_and_submits_scheduler(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'media-runs-retry.db'}")
    database.create_schema()
    scope = Scope("tenant", "application", "kb")
    object_ref = "obj_image_retry"
    with database.session() as session:
        MediaObjectRepository(session, ImageObjectModel).add_ready(
            scope=scope,
            object_ref=object_ref,
            original_filename="pixel.png",
            declared_mime="image/png",
            detected_mime="image/png",
            size_bytes=1,
            sha256="a" * 64,
            relative_path="image_object/object/source.png",
        )

    scheduler = _Scheduler()
    service = MediaRunService(database, scheduler)
    created = service.create(
        scope=scope,
        object_ref=object_ref,
        idempotency_key="image-request-retry",
    )
    scheduler.submissions.clear()

    with database.session() as session:
        session.execute(
            update(JobModel)
            .where(JobModel.job_id == created.job_id)
            .values(
                status=JobStatus.FAILED,
                error_code="IMAGE_VLM_UNAVAILABLE",
            ),
        )
        session.execute(
            update(ImageUnderstandingRunModel)
            .where(ImageUnderstandingRunModel.run_id == created.run_id)
            .values(
                status=RunStatusValue.FAILED,
                error_code="IMAGE_VLM_UNAVAILABLE",
            ),
        )

    retried = service.retry_job(scope, created.job_id)

    assert retried.job_id == created.job_id
    assert retried.resource_id == created.run_id
    assert retried.status == JobStatus.PENDING
    assert retried.attempt_count == 0
    assert retried.error_code is None
    assert scheduler.submissions == [(scope, created.run_id)]
