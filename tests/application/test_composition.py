from __future__ import annotations

import inspect
from pathlib import Path

import video_demo.application.composition as composition
from video_demo.application.chapter_frames import ChapterFrameSearcher
from video_demo.application.chapter_vision import ChapterVisionService
from video_demo.application.document_pipeline import VideoUnderstandingPipeline
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
)
from video_demo.config import Settings
from video_demo.storage.object_store import LocalVideoObjectStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace_root=tmp_path,
        openai_base_url="https://asr.example.test/v1",
        openai_api_key="asr-key",
        openai_model="openai/whisper",
        text_llm_base_url="https://text.example.test/v1",
        text_llm_api_key="text-key",
        text_llm_model_id="text-model",
        vlm_base_url="https://vlm.example.test/v1",
        vlm_api_key="vlm-key",
        _env_file=None,
    )


def test_composition_has_no_legacy_scene_imports() -> None:
    source = inspect.getsource(composition)
    assert "production_scene" not in source
    assert "scenedetect" not in source.lower()
    assert "cv2" not in source


def test_production_pipeline_uses_ffmpeg_frame_searcher(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    from video_demo.persistence.database import Database

    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'composition.db'}")
    object_store = LocalVideoObjectStore(
        settings.runtime_root, max_video_bytes=settings.max_video_bytes
    )
    pipeline = composition.build_production_pipeline(settings, database, object_store)
    try:
        assert isinstance(pipeline, VideoUnderstandingPipeline)
        assert isinstance(pipeline.registrar, ProductionAssetRegistrar)
        assert isinstance(pipeline.probe, ProductionAssetProbe)
        assert isinstance(pipeline.transcoder, ProductionMediaTranscoder)
        assert isinstance(pipeline.frame_searcher, ChapterFrameSearcher)
        assert isinstance(pipeline.chapter_vision, ChapterVisionService)
        assert not hasattr(pipeline, "scene_index_provider")
    finally:
        pipeline.close()
