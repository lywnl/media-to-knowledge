from __future__ import annotations

from typing import BinaryIO

from video_demo.persistence.database import Database
from video_demo.persistence.repositories import Scope, VideoObjectRepository
from video_demo.storage.object_store import LocalVideoObjectStore, VideoObjectRecord


class UploadService:
    """协调隔离写入与作用域元数据登记。"""

    def __init__(self, database: Database, object_store: LocalVideoObjectStore) -> None:
        self._database = database
        self._object_store = object_store

    def upload(
        self,
        stream: BinaryIO,
        filename: str,
        declared_mime: str,
        scope: Scope,
    ) -> VideoObjectRecord:
        record = self._object_store.ingest(stream, filename, declared_mime, scope)
        with self._database.session() as session:
            VideoObjectRepository(session).add_ready(
                scope=scope,
                object_ref=record.object_ref,
                original_filename=record.original_filename,
                declared_mime=record.declared_mime,
                detected_mime=record.detected_mime,
                size_bytes=record.size_bytes,
                sha256=record.sha256,
                relative_path=record.relative_path,
            )
        return record
