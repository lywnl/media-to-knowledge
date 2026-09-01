from __future__ import annotations

import re
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def test_audio_business_modules_do_not_import_visual_or_video_core() -> None:
    roots = (
        Path("src/video_demo/application/audio_pipeline.py"),
        Path("src/video_demo/application/audio_segments.py"),
        Path("src/video_demo/application/audio_chapter_planning.py"),
        Path("src/video_demo/application/audio_document_writing.py"),
        Path("src/video_demo/application/audio_composition.py"),
        Path("src/video_demo/application/audio_publication.py"),
        Path("src/video_demo/application/audio_queries.py"),
        Path("src/video_demo/application/audio_speech.py"),
        Path("src/video_demo/application/audio_transcode.py"),
        Path("src/video_demo/application/audio_runs.py"),
        Path("src/video_demo/application/audio_uploads.py"),
        Path("src/video_demo/media/audio_transcode.py"),
        Path("src/video_demo/media/audio_probe.py"),
        Path("src/video_demo/application/publication_contracts.py"),
        Path("src/video_demo/application/audio_run_config.py"),
        Path("src/video_demo/domain/audio_plan.py"),
        Path("src/video_demo/domain/audio_document.py"),
        Path("src/video_demo/integrations/audio_document_port.py"),
        Path("src/video_demo/integrations/audio_document_prompts.py"),
        Path("src/video_demo/integrations/audio_document_client.py"),
        Path("src/video_demo/persistence/audio_document_repository.py"),
    )
    forbidden = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"videounderstandingpipeline",
            r"(?<!audio)chapterplanner",
            r"(?<!audio)documentwriter",
            r"(?<!audio)chapterplan",
            r"(?<!audio)chapterdraft",
            r"chapterbodyblock",
            r"videounderstandingresult",
            r"visualblock",
            r"visual_mode",
            r"semantic_targets",
            r"base_coverage_targets",
            r"keyframe",
            r"vlm",
            r"\"silence\"",
            r"from video_demo\.domain\.document import",
            r"from video_demo\.application\.pipeline_contracts import",
            r"from video_demo\.application\.composition import",
            r"from video_demo\.application\.media_publication import",
        )
    )
    for path in roots:
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker.search(text) is None, f"{path} 包含禁止音频耦合字段 {marker.pattern}"


def test_audio_transcode_uses_audio_kernel_without_video_transcode_import() -> None:
    source = Path("src/video_demo/application/audio_transcode.py").read_text(encoding="utf-8")
    assert "video_demo.media.transcode" not in source
    assert "video_demo.media.audio_transcode" in source


def test_audio_routes_use_audio_specific_services() -> None:
    source = Path("src/video_demo/api/audio_routes.py").read_text(encoding="utf-8")
    assert "MediaRunService" not in source
    assert "media_run_services" not in source
    assert "media_upload_services" not in source
    assert "audio_run_service" in source
    assert "audio_upload_service" in source


def test_generic_image_services_do_not_keep_audio_dispatch_branches() -> None:
    run_source = Path("src/video_demo/application/media_runs.py").read_text(encoding="utf-8")
    upload_source = Path("src/video_demo/application/media_uploads.py").read_text(encoding="utf-8")
    assert "AudioRunConfig" not in run_source
    assert "AudioDocumentConfig" not in run_source
    assert "AudioUnderstandingRunModel" not in run_source
    assert 'kind == "AUDIO"' not in run_source
    assert "AudioObjectModel" not in upload_source


def test_audio_pipeline_does_not_name_video_cancellation_code() -> None:
    source = Path("src/video_demo/application/audio_pipeline.py").read_text(encoding="utf-8")
    assert "VIDEO_PROCESS_CANCELLED" not in source


def test_audio_asr_uses_audio_owned_algorithm_module() -> None:
    pipeline_source = Path("src/video_demo/application/audio_pipeline.py").read_text(
        encoding="utf-8",
    )
    whisper_source = Path("src/video_demo/integrations/cloud_whisper.py").read_text(
        encoding="utf-8",
    )
    assert "video_demo.speech.asr import" not in pipeline_source
    assert "video_demo.speech.audio_fixed_asr" in pipeline_source
    assert "video_demo.speech.asr import" not in whisper_source


def test_retired_audio_asr_module_is_removed() -> None:
    assert not Path("src/video_demo/speech/audio_asr.py").exists()


def test_audio_query_uses_published_audio_consistency_path() -> None:
    source = Path("src/video_demo/application/audio_queries.py").read_text(encoding="utf-8")
    assert "self.publication.get(scope, run_id)" in source
    assert "AudioResultRepository" not in source


def test_audio_api_routes_are_defined_in_an_independent_module() -> None:
    path = Path("src/video_demo/api/audio_routes.py")
    assert path.is_file()
    source = path.read_text(encoding="utf-8")
    assert "from video_demo.api.media_routes import" not in source
    assert "CreateAudioRunRequest" in source
    assert "PublicAudioUnderstandingResult" in source


def test_audio_publication_does_not_require_generic_media_constructor() -> None:
    source = Path("src/video_demo/application/audio_publication.py").read_text(encoding="utf-8")
    assert "AudioUnderstandingRunModel" in source
    assert "AudioResultRepository" in source
    assert "class AudioPublicationService" in source
