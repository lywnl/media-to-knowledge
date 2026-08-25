"""测试环境与开发机 `.env` 隔离，避免本地凭据改变默认配置断言。"""

from __future__ import annotations

import os

import pytest

from video_demo.config import Settings


def pytest_configure() -> None:
    for name in (
        "VIDEO_DEMO_HUGGINGFACE_TOKEN",
        "VIDEO_DEMO_WHISPER_MODEL_ID",
        "VIDEO_DEMO_WHISPER_COMPUTE_TYPE",
        "VIDEO_DEMO_INFERENCE_DEVICE",
        "VIDEO_DEMO_SPEECH_ENRICHMENT_TIMEOUT_SECONDS",
        "VIDEO_DEMO_BAIDU_API_KEY",
        "VIDEO_DEMO_BAIDU_SECRET_KEY",
        "VIDEO_DEMO_QWEN_API_KEY",
        "VIDEO_DEMO_QWEN_BASE_URL",
        "VIDEO_DEMO_QWEN_MODEL_ID",
        "VIDEO_DEMO_OSS_ENDPOINT",
        "VIDEO_DEMO_OSS_BUCKET",
        "VIDEO_DEMO_OSS_ACCESS_KEY_ID",
        "VIDEO_DEMO_OSS_ACCESS_KEY_SECRET",
    ):
        os.environ[name] = ""
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_ASR_TIMEOUT_SECONDS",
        "OPENAI_ASR_MAX_ATTEMPTS",
    ):
        os.environ.pop(name, None)
    os.environ["VIDEO_DEMO_BAIDU_OCR_ENDPOINT"] = (
        "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    )


@pytest.fixture
def cloud_asr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """只供显式启动生产组合根的测试注入非真实凭据。"""

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ai-proxy.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai/whisper")
    monkeypatch.setenv("OPENAI_ASR_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("OPENAI_ASR_MAX_ATTEMPTS", "3")


@pytest.fixture(autouse=True)
def isolate_workspace_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有测试默认禁用开发机 `.env`，需要环境值的用例必须显式注入。"""

    monkeypatch.setitem(Settings.model_config, "env_file", None)
