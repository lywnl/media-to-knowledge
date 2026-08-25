from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.config import Settings
from video_demo.errors import ErrorCode, VideoDemoError


def test_api_rejects_missing_cloud_asr_configuration_before_writing_runtime(
    tmp_path: Path,
) -> None:
    settings = Settings(workspace_root=tmp_path, _env_file=None)
    assert settings.runtime_root is not None

    with pytest.raises(VideoDemoError) as raised:
        create_app(settings)

    assert raised.value.code == ErrorCode.INVALID_CONFIGURATION
    assert not settings.runtime_root.exists()


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


def test_frontend_page_exposes_local_video_file_workflow(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert '<html lang="zh-CN">' in html
    assert "本地视频理解" in html
    assert 'id="video-file"' in html
    assert 'type="file"' in html
    assert 'accept=".mp4,.mov,.mkv,.webm"' in html
    assert 'href="/static/styles.css"' in html
    assert 'src="/static/app.js"' in html
    assert "video-url" not in html
    assert "tenant-id" not in html
    assert "application-id" not in html
    assert "knowledge-base-id" not in html
    assert 'id="history-panel"' in html
    assert 'id="history-list"' in html


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


def test_frontend_script_renders_untrusted_result_as_text(client: TestClient) -> None:
    script = client.get("/static/app.js").text

    assert ".textContent" in script
    assert ".innerHTML" not in script
    assert 'behavior: "smooth"' not in script
