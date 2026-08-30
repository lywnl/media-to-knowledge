from __future__ import annotations

from video_demo.application.audio_publication import (
    AudioPublicationService,
    AudioResultPublication,
)
from video_demo.domain.audio_document import AudioUnderstandingResult
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
        return self.publication.get(scope, run_id)

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        publication = self._consistent_publication(scope, run_id)
        return publication.document
