from __future__ import annotations

from pathlib import Path

from video_demo.application.audio_runs import AudioRunService
from video_demo.domain.audio_plan import AudioDocumentConfig
from video_demo.persistence.database import Database
from video_demo.persistence.models import AudioObjectModel, AudioStageName, VideoObjectStatus
from video_demo.persistence.scope import Scope


class _Scheduler:
    def __init__(self) -> None:
        self.submissions: list[tuple[Scope, str, AudioStageName]] = []

    def submit(self, scope: Scope, run_id: str, stage: AudioStageName) -> str:
        self.submissions.append((scope, run_id, stage))
        return "accepted"


def test_audio_run_initializes_and_submits_transcription_stage(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'audio-runs.db'}")
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session:
        session.add(
            AudioObjectModel(
                tenant_id=scope.tenant_id,
                application_id=scope.application_id,
                knowledge_base_id=scope.knowledge_base_id,
                object_ref="obj_" + "a" * 32,
                original_filename="sample.mp3",
                declared_mime="audio/mpeg",
                detected_mime="audio/mpeg",
                size_bytes=1,
                sha256="b" * 64,
                relative_path="audio_object/object/source.mp3",
                status=VideoObjectStatus.READY,
                scan_details={},
            ),
        )

    scheduler = _Scheduler()
    service = AudioRunService(database, scheduler)
    view = service.create(
        scope=scope,
        object_ref="obj_" + "a" * 32,
        idempotency_key="audio-run-scheduler-001",
        document_config=AudioDocumentConfig(),
    )

    assert scheduler.submissions == [(scope, view.run_id, AudioStageName.TRANSCRIPTION)]
