from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import ConfigDict, Field, computed_field, model_validator

from video_demo.domain.base import FrozenModel, StableId


class TimeRange(FrozenModel):
    """整数毫秒半开时间区间 `[start_ms, end_ms)`。"""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_non_empty_range(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms 必须大于 start_ms")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def contains(self, other: TimeRange) -> bool:
        return self.start_ms <= other.start_ms and other.end_ms <= self.end_ms

    def overlaps(self, other: TimeRange) -> bool:
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCEEDED = "PARTIAL_SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStage(StrEnum):
    REGISTER = "REGISTER"
    PROBE = "PROBE"
    TRANSCODE = "TRANSCODE"
    SPEECH = "SPEECH"
    VISUAL = "VISUAL"
    FUSION = "FUSION"
    UNDERSTANDING = "UNDERSTANDING"
    RESULT = "RESULT"


class ModelIdentity(FrozenModel):
    model_config = ConfigDict(revalidate_instances="always")

    component: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    revision: str | None = Field(default=None, max_length=256)
    device: str | None = Field(default=None, max_length=32)


class RunWarning(FrozenModel):
    code: str = Field(min_length=3, max_length=128)
    stage: RunStage
    message: str = Field(min_length=1, max_length=500)


class RunSnapshot(FrozenModel):
    run_id: StableId
    status: RunStatus
    current_stage: RunStage
    warnings: tuple[RunWarning, ...] = ()
    models: tuple[ModelIdentity, ...] = ()
