from __future__ import annotations

from typing import Any

from sqlalchemy import select

from video_demo.application.media_publication import MediaPublicationService
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_document_repository import AudioResultRepository
from video_demo.persistence.media_repositories import MediaRunRepository
from video_demo.persistence.models import (
    AudioAssetModel,
    AudioUnderstandingRunModel,
    RunStatusValue,
)
from video_demo.persistence.repositories import Scope


class AudioQueryService:
    """从 audio_* 结果行读取音频结果，并校验已发布制品。"""

    def __init__(self, publication: MediaPublicationService) -> None:
        self.publication = publication

    def get_result(self, scope: Scope, run_id: str) -> Any:
        with self.publication.database.session() as session:
            run = MediaRunRepository(session, AudioUnderstandingRunModel).get(scope, run_id)
            if run is None:
                raise VideoDemoError(ErrorCode.AUDIO_RUN_NOT_FOUND, "音频运行不存在")
            if run.status not in {RunStatusValue.SUCCEEDED, RunStatusValue.PARTIAL_SUCCEEDED}:
                raise VideoDemoError(ErrorCode.AUDIO_RESULT_NOT_READY, "音频理解结果尚未就绪")
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
            return AudioResultRepository(session).get(scope, run_id, asset.source_sha256)

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        return self.publication.get(scope, run_id).document
