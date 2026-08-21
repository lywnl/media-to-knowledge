from __future__ import annotations

import io
from pathlib import Path

from video_demo.application.uploads import UploadService
from video_demo.persistence.database import Database
from video_demo.persistence.repositories import Scope, VideoObjectRepository
from video_demo.storage.object_store import LocalVideoObjectStore


def test_upload_service_persists_ready_object_in_request_scope(tmp_path: Path) -> None:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'upload.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(tmp_path / "runtime", max_video_bytes=1024)
    service = UploadService(database, object_store)
    scope = Scope("tenant-a", "app-a", "kb-a")
    mime = "video/mp4"
    content = b"\x00\x00\x00\x18ftypisom" + b"m" * 128

    record = service.upload(io.BytesIO(content), "lesson.mp4", mime, scope)

    with database.session() as session:
        persisted = VideoObjectRepository(session).get_ready(scope, record.object_ref)
        assert persisted is not None
        assert persisted.sha256 == record.sha256
        assert persisted.relative_path == record.relative_path
