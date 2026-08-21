from __future__ import annotations

from typing import Literal

from pydantic import StrictInt, field_validator

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import EvidenceItem
from video_demo.domain.result import VideoUnderstandingResult


class ResultArtifactPayload(FrozenModel):
    """`ResultQueryService.persist` 唯一允许写入的生产结果阶段 payload。"""

    result: VideoUnderstandingResult
    evidence: tuple[EvidenceItem, ...]
    stage_metrics: dict[str, StrictInt]
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    warnings: tuple[str, ...]

    @field_validator("stage_metrics")
    @classmethod
    def validate_stage_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or type(metric) is not int for key, metric in value.items()):
            raise ValueError("阶段指标必须是非空名称和整数值")
        return value

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not warning for warning in value) or len(value) != len(set(value)):
            raise ValueError("运行警告不得为空或重复")
        return value
