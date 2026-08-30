from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from video_demo.api.app import create_app
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError


def test_api_starts_without_model_configuration(
    tmp_path: Path,
) -> None:
    shutil.copytree(Path.cwd() / "migrations", tmp_path / "migrations")
    settings = Settings(workspace_root=tmp_path, _env_file=None)
    assert settings.runtime_root is not None

    app = create_app(settings)

    assert app.state.container.runtime_root == settings.runtime_root
    assert not hasattr(app.state.container, "settings")
    assert settings.runtime_root.exists()


def test_create_app_runs_migration_before_database_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.api.app as app_module

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ai-proxy.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai/whisper")
    events: list[str] = []

    def migrate(*_args: object) -> None:
        events.append("迁移")
        raise VideoDemoError(ErrorCode.ARTIFACT_SCHEMA_INVALID, "停止于迁移")

    class ForbiddenDatabase:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            events.append("数据库")

    monkeypatch.setattr(app_module, "upgrade_runtime_database", migrate)
    monkeypatch.setattr(app_module, "Database", ForbiddenDatabase)

    with pytest.raises(VideoDemoError, match="停止于迁移"):
        app_module.create_app(Settings(workspace_root=tmp_path))
    assert events == ["迁移"]


def test_create_app_upgrades_real_unversioned_0001_database(
    tmp_path: Path,
    cloud_asr_environment: None,
) -> None:
    settings = Settings(workspace_root=tmp_path)
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True)
    database_url = f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}"
    config = Config()
    config.attributes["configure_logging"] = False
    config.set_main_option("script_location", str(tmp_path / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0001_video_demo")
    with create_engine(database_url).begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))

    app = create_app(settings)

    with create_engine(database_url).connect() as connection:
        assert (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == "0006_audio_document_result"
        )
    assert app.state.container.database is not None


def test_frontend_page_exposes_local_video_file_workflow(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert '<html lang="zh-CN">' in html
    assert "本地媒体理解" in html
    assert 'id="video-file"' in html
    assert 'type="file"' in html
    assert 'accept=".mp4,.mov,.mkv,.webm"' in html
    assert 'href="/static/styles.css?v=media-pipelines-2"' in html
    assert 'src="/static/app.js?v=media-pipelines-2"' in html
    assert "video-url" not in html
    assert "tenant-id" not in html
    assert "application-id" not in html
    assert "knowledge-base-id" not in html
    assert 'id="history-panel"' in html
    assert 'id="history-list"' in html
    assert response.headers["cache-control"] == "no-store"


def test_frontend_static_resources_are_available(client: TestClient) -> None:
    stylesheet = client.get("/static/styles.css")
    script = client.get("/static/app.js")

    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]


def test_frontend_only_accepts_one_local_file(client: TestClient) -> None:
    html = client.get("/").text
    file_input = re.search(r'<input\s+[^>]*id="video-file"[^>]*>', html)

    assert file_input is not None
    assert "multiple" not in file_input.group(0)
    assert "拖拽" not in html
    assert 'type="url"' not in html


def test_frontend_script_uses_existing_async_api_contract(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-objects" in script
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs" in script
    assert 'X-Tenant-Id' in script
    assert 'X-Application-Id' in script
    assert "const POLL_INTERVAL_MS = 2000" in script
    assert 'new Set(["SUCCEEDED", "PARTIAL_SUCCEEDED"])' in script
    assert 'new Set(["FAILED", "CANCELLED"])' in script
    assert "AbortController" in script
    assert "crypto.randomUUID()" in script
    assert '".mkv": "video/x-matroska"' in script
    assert "new File([file], file.name, { type: declaredMime })" in script
    assert "this.httpStatus = httpStatus" in script
    assert "class TerminalRunError extends Error" in script
    assert "const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 502, 503, 504])" in script
    assert "isRetryablePollingError(error)" in script
    assert 'signal.removeEventListener("abort", rejectOnAbort)' in script
    assert 'updateStatus("处理未完成"' in script
    assert "loadHistory()" in script
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/video-understanding-runs" in script
    assert "original_filename" in script
    assert "min_speakers" not in script
    assert "max_speakers" not in script
    assert "speech_enrichment_mode" not in script


def test_frontend_exposes_independent_video_audio_image_workflows(client: TestClient) -> None:
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert 'data-media-kind="VIDEO"' in html
    assert 'data-media-kind="AUDIO"' in html
    assert 'data-media-kind="IMAGE"' in html
    assert "音频" in html
    assert "图片" in html
    assert 'accept=".mp4,.mov,.mkv,.webm"' in html
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/audio-objects" in script
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/image-objects" in script
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/audio-understanding-runs" in script
    assert "/api/kb/knowledge-bases/${SCOPE.knowledgeBaseId}/image-understanding-runs" in script
    assert "renderAudioResult" in script
    assert "renderImageResult" in script


def test_frontend_script_renders_untrusted_result_as_text(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert ".textContent" in script
    assert ".innerHTML" not in script
    assert 'behavior: "smooth"' not in script


def test_frontend_script_exposes_structured_document_reading_contract(
    client: TestClient,
) -> None:
    html = client.get("/").text
    script = client.get("/static/app.js").text

    assert 'id="document-overview"' in html
    assert 'id="document-key-points"' not in html
    assert "documentKeyPoints" not in script
    assert 'id="document-toc"' in html
    assert 'id="download-status"' in html
    assert 'summary.overview_zh || "未提供核心概览。"' in script
    assert "fetchEvidence" not in script
    assert "/keyframes/" not in script
    assert "renderKeyframeFigure" not in script
    assert "AbortController" in script
    assert "下载 Markdown失败" in script


def test_frontend_script_renders_chapter_claims(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert "本章结论" in script
    assert "chapter.claims" in script


def test_frontend_does_not_render_description_field_label(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert 'block.content_type === "DESCRIPTION"' in script
    assert 'element("h4", null, block.content_type)' in script
    assert 'element("p", "chapter-body", block.text)' in script
    assert (
        'content.append(element("h4", null, block.content_type), '
        'element("p", "chapter-body", block.text));'
    ) not in script


def test_frontend_does_not_map_keyframe_evidence_refs_to_images(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert "keyframe_id" not in script
    assert "keyframeIdForEvidenceRef" not in script
    assert "renderKeyframeFigure" not in script


def test_frontend_styles_include_document_reading_states(client: TestClient) -> None:
    stylesheet = client.get("/static/styles.css").text

    assert ".document-reader" in stylesheet
    assert ".document-toc" in stylesheet
    assert ".chapter-body--quote" in stylesheet
    assert ".chapter-keyframe" not in stylesheet
    assert ".retrieval-text" not in stylesheet
