from __future__ import annotations

import inspect

from video_demo.application.document_pipeline import (
    ChapterFrameSearcherPort,
    ChapterPlannerPort,
    VideoUnderstandingPipeline,
)
from video_demo.domain.document_artifact import RESULT_STAGE_NAMES


def test_pipeline_does_not_reference_scene_index_runtime() -> None:
    source = inspect.getsource(VideoUnderstandingPipeline)
    assert "SceneIndex" not in source
    assert "scene_index" not in source


def test_pipeline_protocols_use_time_point_frame_search() -> None:
    assert "scenes" not in inspect.signature(ChapterPlannerPort.plan).parameters
    assert "scenes" not in inspect.signature(ChapterFrameSearcherPort.search).parameters
    assert "SCENE_DETECT" not in RESULT_STAGE_NAMES
