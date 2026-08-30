from __future__ import annotations

import ast
import re
from pathlib import Path

from video_demo.implementation import implementation_import_closure

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def test_audio_business_modules_do_not_import_visual_or_video_core() -> None:
    roots = (
        Path("src/video_demo/application/audio_pipeline.py"),
        Path("src/video_demo/application/audio_segments.py"),
        Path("src/video_demo/application/audio_chapter_planning.py"),
        Path("src/video_demo/application/audio_document_writing.py"),
        Path("src/video_demo/application/audio_composition.py"),
        Path("src/video_demo/application/audio_workers.py"),
        Path("src/video_demo/application/audio_publication.py"),
        Path("src/video_demo/application/audio_queries.py"),
        Path("src/video_demo/application/audio_speech.py"),
        Path("src/video_demo/application/audio_transcode.py"),
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
            r"from video_demo\.domain\.document import",
            r"from video_demo\.application\.pipeline_contracts import",
            r"from video_demo\.application\.composition import",
            r"from video_demo\.application\.media_workers import",
            r"from video_demo\.application\.media_publication import",
        )
    )
    for path in roots:
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker.search(text) is None, f"{path} 包含禁止音频耦合字段 {marker.pattern}"


def test_audio_worker_does_not_import_video_or_visual_assembly_modules() -> None:
    source = Path("src/video_demo/application/audio_workers.py").read_text(encoding="utf-8")
    forbidden_modules = (
        "video_demo.application.production_media",
        "video_demo.application.production_speech",
        "video_demo.media.transcode",
    )
    assert not any(module in source for module in forbidden_modules)


def test_audio_worker_import_closure_excludes_video_and_visual_business_modules() -> None:
    closure = set(
        implementation_import_closure(
            _WORKSPACE_ROOT,
            (Path("src/video_demo/audio_worker_main.py"),),
            extra_files=(),
        ),
    )
    forbidden = {
        Path("src/video_demo/application/chapter_planning.py"),
        Path("src/video_demo/application/chapter_frames.py"),
        Path("src/video_demo/application/chapter_vision.py"),
        Path("src/video_demo/application/document_pipeline.py"),
        Path("src/video_demo/application/document_writing.py"),
        Path("src/video_demo/application/composition.py"),
        Path("src/video_demo/integrations/qwen_vl.py"),
        Path("src/video_demo/visual/keyframes.py"),
        Path("src/video_demo/visual/scenes.py"),
    }
    assert closure.isdisjoint(forbidden)


def test_audio_worker_uses_audio_publication_and_neutral_title_helper() -> None:
    source = Path("src/video_demo/application/audio_workers.py").read_text(encoding="utf-8")
    assert (
        "from video_demo.application.audio_publication import AudioPublicationService"
        in source
    )
    assert "from video_demo.application.media_publication import" not in source
    assert "from video_demo.domain.title import sanitize_document_title" in source


def test_audio_pipeline_does_not_name_video_cancellation_code() -> None:
    source = Path("src/video_demo/application/audio_pipeline.py").read_text(encoding="utf-8")
    assert "VIDEO_PROCESS_CANCELLED" not in source


def test_audio_query_uses_published_audio_consistency_path() -> None:
    source = Path("src/video_demo/application/audio_queries.py").read_text(encoding="utf-8")
    assert "self.publication.get(scope, run_id)" in source
    assert "音频数据库结果与已发布制品不一致" in source


def test_audio_worker_imports_are_parseable_without_video_domain_import() -> None:
    source = Path("src/video_demo/application/audio_workers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "video_demo.domain.document" not in imported_modules


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


def test_video_worker_has_dedicated_composition_entrypoint() -> None:
    source = Path("src/video_demo/worker_main.py").read_text(encoding="utf-8")
    assert "video_demo.application.video_composition import build_worker" in source
    assert Path("src/video_demo/application/video_composition.py").is_file()
