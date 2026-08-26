from __future__ import annotations

from pathlib import Path

import pytest

from video_demo.application.legacy_composition import (
    build_production_diagnostic_components,
)
from video_demo.config import Settings
from video_demo.errors import VideoDemoError


@pytest.mark.parametrize(
    "overrides",
    [
        {"oss_endpoint": "oss.example.test"},
        {"qwen_base_url": "https://qwen.example.test/v1"},
        {"baidu_api_key": "baidu-key"},
    ],
)
def test_legacy_diagnostic_builder_owns_retired_group_validation(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    settings = Settings(
        workspace_root=tmp_path,
        openai_base_url="https://asr.example.test/v1",
        openai_api_key="asr-key",
        openai_model="openai/whisper",
        **overrides,
        _env_file=None,
    )

    with pytest.raises((VideoDemoError, ValueError)):
        build_production_diagnostic_components(settings)
