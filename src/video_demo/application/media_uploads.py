from __future__ import annotations

from typing import BinaryIO

from video_demo.persistence.database import Database
from video_demo.persistence.media_repositories import MediaObjectRepository
from video_demo.persistence.models import AudioObjectModel, ImageObjectModel
from video_demo.persistence.scope import Scope
from video_demo.storage.media_object_store import BinaryMediaObjectStore, MediaObjectRecord


class MediaUploadService:
    """登记音频或图片对象；视频上传仍由原有 UploadService 处理。"""

    def __init__(
        self,
        database: Database,
        object_store: BinaryMediaObjectStore,
        model: type[AudioObjectModel] | type[ImageObjectModel],
    ) -> None:
        self._database = database
        self._object_store = object_store
        self._model = model

    def upload(
        self,
        stream: BinaryIO,
        filename: str,
        declared_mime: str,
        scope: Scope,
    ) -> MediaObjectRecord:
        record = self._object_store.ingest(stream, filename, declared_mime, scope)
        with self._database.session() as session:
            MediaObjectRepository(session, self._model).add_ready(
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
