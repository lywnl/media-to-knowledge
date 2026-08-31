from __future__ import annotations

import inspect

import pytest

import video_demo.application.pipeline as pipeline
from video_demo.application.document_pipeline import VideoUnderstandingPipeline
from video_demo.application.pipeline import (
    PipelineContext,
    PipelineOutcome,
    PipelineRunConfig,
    pipeline_run_config_from_snapshot,
)
from video_demo.errors import ErrorCode, VideoDemoError


def test_pipeline_public_surface_is_the_single_4_pipeline() -> None:
    assert pipeline.VideoUnderstandingPipeline is VideoUnderstandingPipeline
    assert pipeline.PipelineContext is PipelineContext
    assert pipeline.PipelineOutcome is PipelineOutcome
    assert not hasattr(pipeline, "VisualAnalysis")
    assert not hasattr(pipeline, "VisualPreparation")
    assert not hasattr(pipeline, "WholeVideoUnderstandingPort")
    source = inspect.getsource(pipeline)
    assert "legacy_result" not in source
    assert "production_visual" not in source
    assert "fusion." not in source


def test_pipeline_run_config_parses_only_4_snapshot() -> None:
    config = pipeline_run_config_from_snapshot(
        {
            "language_hints": ["zh", "en"],
            "hotwords": ["Qwen3-VL"],
            "core_context": "视频知识文档",
            "document_config": {"detail_level": "detailed"},
            "result_schema_version": "4.2.0",
        }
    )

    assert isinstance(config, PipelineRunConfig)
    assert config.language_hints == ("zh", "en")
    assert config.document_config.detail_level == "detailed"


@pytest.mark.parametrize("version", [None, "1.0.0", "2.0.0", "3.0.0"])
def test_pipeline_run_config_rejects_missing_or_legacy_schema(version: str | None) -> None:
    snapshot: dict[str, object] = {}
    if version is not None:
        snapshot["result_schema_version"] = version

    with pytest.raises(VideoDemoError) as raised:
        pipeline_run_config_from_snapshot(snapshot)

    assert raised.value.code == ErrorCode.RESULT_SCHEMA_UNSUPPORTED
    assert raised.value.details == {"supported_schema_version": "4.2.0"}
