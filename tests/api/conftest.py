from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from video_demo.api.app import create_app
from video_demo.config import Settings


@pytest.fixture
def client(tmp_path: Path, cloud_asr_environment: None) -> Iterator[TestClient]:
    settings = Settings(workspace_root=tmp_path, max_video_bytes=1024 * 1024)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def scope_headers() -> dict[str, str]:
    return {"X-Tenant-Id": "tenant-a", "X-Application-Id": "app-a"}


@pytest.fixture
def mp4_content() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"m" * 128
