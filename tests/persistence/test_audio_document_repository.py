from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, select

from video_demo.domain.audio_document import (
    AudioChapter,
    AudioDocumentSummary,
    AudioUnderstandingResult,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.persistence.audio_document_repository import AudioResultRepository
from video_demo.persistence.database import Database
from video_demo.persistence.models import (
    AudioAssetModel,
    AudioSegmentModel,
    AudioSummaryModel,
    VideoSegmentModel,
    VideoSummaryModel,
)
from video_demo.persistence.repositories import Scope


def _result() -> AudioUnderstandingResult:
    return AudioUnderstandingResult(
        run_id="run_audio_001",
        asset_sha256="a" * 64,
        summary=AudioDocumentSummary(title="音频", duration_ms=1_000, overview_zh="概览"),
        chapters=(
            AudioChapter(
                chapter_id="audio_chapter_001",
                start_ms=0,
                end_ms=1_000,
                title="章节",
                title_evidence_refs=("asr_001",),
                summary_zh="摘要",
                summary_evidence_refs=("asr_001",),
                body_blocks=(),
                claims=(),
                evidence_refs=("asr_001",),
                transcript_source="ASR",
            ),
        ),
    )


def test_audio_repository_round_trips_audio_tables_only(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'audio.db'}"
    database = Database(database_url)
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session:
        repository = AudioResultRepository(session)
        repository.replace(scope, _result(), object_ref="obj_audio_001")
        loaded = repository.get(scope, "run_audio_001", "a" * 64)

    assert loaded == _result()
    tables = set(inspect(create_engine(database_url)).get_table_names())
    assert {"audio_asset", "audio_segment", "audio_summary"}.isdisjoint(tables) is False


def test_audio_repository_writes_only_audio_rows_and_replaces_idempotently(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'audio-replace.db'}"
    database = Database(database_url)
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session:
        repository = AudioResultRepository(session)
        repository.replace(scope, _result(), object_ref="obj_audio_001")
        repository.replace(scope, _result(), object_ref="obj_audio_001")

        assert session.scalar(
            select(AudioAssetModel).where(AudioAssetModel.run_id == "run_audio_001"),
        ) is not None
        assert len(session.scalars(
            select(AudioSegmentModel).where(AudioSegmentModel.run_id == "run_audio_001"),
        ).all()) == 1
        assert len(session.scalars(
            select(AudioSummaryModel).where(AudioSummaryModel.run_id == "run_audio_001"),
        ).all()) == 1
        assert session.scalars(select(VideoSegmentModel)).all() == []
        assert session.scalars(select(VideoSummaryModel)).all() == []


def test_audio_repository_reports_missing_rows_as_not_ready(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'audio-missing.db'}"
    database = Database(database_url)
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session, pytest.raises(VideoDemoError) as raised:
        AudioResultRepository(session).get(scope, "run_audio_missing", "a" * 64)

    assert raised.value.code == ErrorCode.AUDIO_RESULT_NOT_READY


def test_audio_repository_rejects_requested_digest_different_from_asset_row(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'audio-digest.db'}"
    database = Database(database_url)
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session:
        repository = AudioResultRepository(session)
        repository.replace(scope, _result(), object_ref="obj_audio_001")
        with pytest.raises(VideoDemoError) as raised:
            repository.get(scope, "run_audio_001", "b" * 64)

    assert raised.value.code == ErrorCode.AUDIO_DIGEST_MISMATCH


def test_audio_repository_rejects_asset_row_with_unsupported_schema(tmp_path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'audio-schema.db'}"
    database = Database(database_url)
    database.create_schema()
    scope = Scope("tenant", "app", "kb")
    with database.session() as session:
        repository = AudioResultRepository(session)
        repository.replace(scope, _result(), object_ref="obj_audio_001")
        asset = session.scalar(
            select(AudioAssetModel).where(AudioAssetModel.run_id == "run_audio_001"),
        )
        assert asset is not None
        asset.schema_version = "0.9.0"
        with pytest.raises(VideoDemoError) as raised:
            repository.get(scope, "run_audio_001", "a" * 64)

    assert raised.value.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED
