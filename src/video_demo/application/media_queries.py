from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from video_demo.application.media_publication import MediaPublicationService
from video_demo.persistence.repositories import Scope


@dataclass(frozen=True, slots=True)
class MediaQueryService:
    publication: MediaPublicationService

    def get_result(self, scope: Scope, run_id: str) -> Any:
        return self.publication.get(scope, run_id).result

    def get_document(self, scope: Scope, run_id: str) -> bytes:
        return self.publication.get(scope, run_id).document
