from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

import video_demo.application.composition as composition
from video_demo.application.chapter_vision import ChapterVisionService
from video_demo.application.document_pipeline import VideoUnderstandingPipeline
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
)
from video_demo.application.production_scene import ProductionSceneIndexProvider
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.object_store import LocalVideoObjectStore


def _model_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "workspace_root": tmp_path,
        "openai_base_url": "https://asr.example.test/v1",
        "openai_api_key": "asr-key",
        "openai_model": "openai/whisper",
        "text_llm_base_url": "https://text.example.test/v1",
        "text_llm_api_key": "text-key",
        "text_llm_model_id": "text-model",
        "vlm_base_url": "https://vlm.example.test/v1",
        "vlm_api_key": "vlm-key",
    }
    values.update(overrides)
    return Settings(**values, _env_file=None)


def _copy_migrations(tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "migrations", tmp_path / "migrations")


def test_production_composition_import_surface_contains_no_legacy_chain() -> None:
    source = inspect.getsource(composition)

    assert "legacy_composition" not in source
    assert "production_visual" not in source
    assert "integrations.oss" not in source
    assert "integrations.qwen import" not in source
    assert "WholeVideoUnderstanding" not in source


def test_production_pipeline_uses_only_3_components(tmp_path: Path) -> None:
    settings = _model_settings(tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    from video_demo.persistence.database import Database

    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'composition.db'}")
    object_store = LocalVideoObjectStore(
        settings.runtime_root,
        max_video_bytes=settings.max_video_bytes,
    )

    pipeline = composition.build_production_pipeline(settings, database, object_store)
    try:
        assert isinstance(pipeline, VideoUnderstandingPipeline)
        assert isinstance(pipeline.registrar, ProductionAssetRegistrar)
        assert isinstance(pipeline.probe, ProductionAssetProbe)
        assert isinstance(pipeline.transcoder, ProductionMediaTranscoder)
        assert isinstance(pipeline.scene_index_provider, ProductionSceneIndexProvider)
        assert isinstance(pipeline.chapter_vision, ChapterVisionService)
        assert not hasattr(pipeline, "understanding")
        assert not hasattr(pipeline, "visual_analyzer")
    finally:
        pipeline.close()


def test_production_identity_contains_asr_text_vlm_and_scene_only(tmp_path: Path) -> None:
    report = composition.build_production_model_identity_report(_model_settings(tmp_path))

    assert report.schema_version == "3.0.0"
    assert {item.component for item in report.models} == {
        "silero_vad",
        "cloud_whisper",
        "document_text_llm",
        "chapter_vlm",
        "scene_detect",
    }
    assert "key" not in report.model_dump_json().lower()


def test_text_and_vlm_fingerprints_are_component_isolated(tmp_path: Path) -> None:
    base = _model_settings(tmp_path)
    text_changed = _model_settings(tmp_path, text_llm_model_id="text-model-v2")
    vision_changed = _model_settings(tmp_path, vlm_model_id="qwen3-vl-plus")

    assert composition._component_fingerprint(
        {"model": base.require_text_llm_configuration().model_id}
    ) != composition._component_fingerprint(
        {"model": text_changed.require_text_llm_configuration().model_id}
    )
    assert composition._component_fingerprint(
        {"model": base.require_vlm_configuration().model_id}
    ) == composition._component_fingerprint(
        {"model": text_changed.require_vlm_configuration().model_id}
    )
    assert composition._component_fingerprint(
        {"model": base.require_text_llm_configuration().model_id}
    ) == composition._component_fingerprint(
        {"model": vision_changed.require_text_llm_configuration().model_id}
    )


@pytest.mark.parametrize(
    "missing",
    ["openai_api_key", "text_llm_api_key", "vlm_api_key"],
)
def test_worker_migrates_and_recovers_before_rejecting_model_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    _copy_migrations(tmp_path)
    settings = _model_settings(tmp_path, **{missing: None})
    calls: list[str] = []
    original_upgrade = composition.upgrade_runtime_database

    def upgrade(*args: object, **kwargs: object) -> None:
        calls.append("migrate")
        original_upgrade(*args, **kwargs)

    class Recovery:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def recover(self) -> None:
            calls.append("recover")

    monkeypatch.setattr(composition, "upgrade_runtime_database", upgrade)
    monkeypatch.setattr(composition, "PublishedVisualCleanupRecovery", Recovery)
    monkeypatch.setattr(
        composition,
        "build_production_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("配置失败时不得构造 Pipeline")
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        composition.build_worker(settings, worker_id="worker-test")

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert calls == ["migrate", "recover"]


def test_worker_startup_order_is_migrate_recover_require_then_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_migrations(tmp_path)
    settings = _model_settings(tmp_path)
    calls: list[str] = []
    original_upgrade = composition.upgrade_runtime_database

    def upgrade(*args: object, **kwargs: object) -> None:
        calls.append("migrate")
        original_upgrade(*args, **kwargs)

    class Recovery:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def recover(self) -> None:
            calls.append("recover")

    for name, label in (
        ("require_cloud_asr_configuration", "asr"),
        ("require_text_llm_configuration", "text"),
        ("require_vlm_configuration", "vlm"),
    ):
        original = getattr(Settings, name)

        def wrapped(self: Settings, _original=original, _label=label):
            calls.append(_label)
            return _original(self)

        monkeypatch.setattr(Settings, name, wrapped)

    class Pipeline:
        def close(self) -> None:
            calls.append("pipeline-close")

    monkeypatch.setattr(composition, "upgrade_runtime_database", upgrade)
    monkeypatch.setattr(composition, "PublishedVisualCleanupRecovery", Recovery)
    monkeypatch.setattr(
        composition,
        "build_production_pipeline",
        lambda *_args, **_kwargs: calls.append("pipeline") or Pipeline(),
    )
    worker = composition.build_worker(settings, worker_id="worker-order")
    try:
        assert calls[:6] == ["migrate", "recover", "asr", "text", "vlm", "pipeline"]
    finally:
        worker.close()
