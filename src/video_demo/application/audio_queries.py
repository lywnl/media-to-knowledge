from __future__ import annotations

from sqlalchemy import select

from video_demo.application.audio_publication import (
    AudioPublicationService,
    AudioResultPublication,
)
from video_demo.domain.audio_document import AudioUnderstandingResult
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_document_repository import AudioResultRepository
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import (
    AudioAssetModel,
    AudioUnderstandingRunModel,
)
from video_demo.persistence.scope import Scope


class AudioQueryService:
    """从 audio_* 结果行读取音频结果，并校验已发布制品。"""

    def __init__(self, publication: AudioPublicationService) -> None:
        self.publication = publication

    def get_result(self, scope: Scope, run_id: str) -> AudioUnderstandingResult:
        return self._consistent_publication(scope, run_id).result

    def _consistent_publication(
        self,
        scope: Scope,
        run_id: str,
    ) -> AudioResultPublication:
        published = self.publication.get(scope, run_id)
        with self.publication.database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            asset = session.scalar(
                select(AudioAssetModel).where(
                    AudioAssetModel.tenant_id == scope.tenant_id,
                    AudioAssetModel.application_id == scope.application_id,
                    AudioAssetModel.knowledge_base_id == scope.knowledge_base_id,
                    AudioAssetModel.run_id == run.run_id,
                ),
            )
            if asset is None:
                raise VideoDemoError(ErrorCode.RESULT_SCHEMA_UNSUPPORTED, "音频结果需要重新运行")
            stored = AudioResultRepository(session).get(scope, run_id, asset.source_sha256)
        if stored != published.result:
            raise VideoDemoError(
                ErrorCode.ARTIFACT_SCHEMA_INVALID,
                "音频数据库结果与已发布制品不一致",
            )
        return published

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        publication = self._consistent_publication(scope, run_id)
        return publication.document
