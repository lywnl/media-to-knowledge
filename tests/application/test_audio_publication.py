from __future__ import annotations

from video_demo.application.audio_publication import AudioPublicationService
from video_demo.application.audio_rendering import render_audio_markdown
from video_demo.domain.audio_document import (
    AudioUnderstandingResult,
)
from video_demo.errors import ErrorCode
from video_demo.persistence.database import Database
from video_demo.persistence.models import AudioUnderstandingRunModel
from video_demo.storage.artifacts import AtomicArtifactStore


def test_audio_publication_service_is_explicit_audio_type(tmp_path) -> None:
    service = AudioPublicationService(
        Database(f"sqlite+pysqlite:///{tmp_path / 'audio.db'}"),
        AtomicArtifactStore(tmp_path),
        run_model=AudioUnderstandingRunModel,
        result_type=AudioUnderstandingResult,
        render=render_audio_markdown,
        resource_type="AUDIO_UNDERSTANDING_RUN",
        not_found_code=ErrorCode.AUDIO_RUN_NOT_FOUND,
    )

    assert service.database.engine is not None
