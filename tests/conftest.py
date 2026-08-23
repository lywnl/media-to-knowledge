"""测试环境与开发机 `.env` 隔离，避免本地凭据改变默认配置断言。"""

from __future__ import annotations

import os


def pytest_configure() -> None:
    for name in (
        "VIDEO_DEMO_HUGGINGFACE_TOKEN",
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
    os.environ["VIDEO_DEMO_BAIDU_OCR_ENDPOINT"] = (
        "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    )
