from __future__ import annotations

import video_demo.application.pipeline_contracts as contracts
from video_demo.application.pipeline_contracts import PreparedMedia
from video_demo.domain.document_artifact import RESULT_STAGE_NAMES
from video_demo.errors import ErrorCode, VideoDemoError


def test_result_stages_no_longer_include_scene_detection() -> None:
    assert "SCENE_DETECT" not in RESULT_STAGE_NAMES
    assert "FRAME_SEARCH" in RESULT_STAGE_NAMES


def test_pipeline_config_requires_4_2_snapshot() -> None:
    config = contracts.PipelineRunConfig(result_schema_version="4.2.0")
    assert config.result_schema_version == "4.2.0"
    try:
        contracts.pipeline_run_config_from_snapshot({"result_schema_version": "4.1.0"})
    except VideoDemoError as error:
        assert error.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED
    else:
        raise AssertionError("旧结果版本必须被拒绝")


def test_prepared_media_keeps_proxy_name_as_visual_input() -> None:
    assert "proxy_path" in PreparedMedia.__dataclass_fields__
