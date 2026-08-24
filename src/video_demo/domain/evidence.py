from __future__ import annotations

from typing import Literal, Self, TypeAlias

from pydantic import Field, field_validator, model_validator

from video_demo.domain.base import FrozenModel, LanguageCode, Probability, StableId
from video_demo.domain.run import TimeRange

SpeakerId = Literal[
    "SPEAKER_UNKNOWN",
    "SPEAKER_01",
    "SPEAKER_02",
    "SPEAKER_03",
    "SPEAKER_04",
    "SPEAKER_05",
    "SPEAKER_06",
    "SPEAKER_07",
    "SPEAKER_08",
    "SPEAKER_09",
    "SPEAKER_10",
]


class TimedEvidence(TimeRange):
    evidence_id: StableId


class SpeechSegment(TimedEvidence):
    evidence_type: Literal["ASR_SEGMENT"] = "ASR_SEGMENT"
    text: str = Field(min_length=1)
    language: LanguageCode
    confidence: Probability
    is_fully_evaluated_language: bool


class SubtitleCue(TimedEvidence):
    evidence_type: Literal["SUBTITLE_CUE"] = "SUBTITLE_CUE"
    text: str = Field(min_length=1, max_length=4000)
    language: LanguageCode
    stream_index: int = Field(ge=0)


class SceneBoundary(TimedEvidence):
    evidence_type: Literal["SCENE"] = "SCENE"
    transition: Literal["hard_cut", "gradual", "candidate"]
    score: Probability


class KeyframeEvidence(TimedEvidence):
    evidence_type: Literal["KEYFRAME"] = "KEYFRAME"
    keyframe_id: StableId
    timestamp_ms: int = Field(ge=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    mime_type: Literal["image/jpeg", "image/png"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    perceptual_hash: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def validate_timestamp_inside_range(self) -> Self:
        if not self.start_ms <= self.timestamp_ms < self.end_ms:
            raise ValueError("timestamp_ms 必须位于关键帧证据区间内")
        return self


class BoundingBox(FrozenModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class OcrLine(FrozenModel):
    text: str = Field(min_length=1)
    bounding_box: BoundingBox
    confidence: Probability


class OcrEvidence(TimedEvidence):
    evidence_type: Literal["OCR"] = "OCR"
    keyframe_id: StableId
    timestamp_ms: int = Field(ge=0)
    language: LanguageCode
    lines: tuple[OcrLine, ...]
    provider_request_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_timestamp_inside_range(self) -> Self:
        if not self.start_ms <= self.timestamp_ms < self.end_ms:
            raise ValueError("timestamp_ms 必须位于 OCR 证据区间内")
        return self


class TimelineEvidence(TimeRange):
    timeline_id: StableId
    evidence_refs: tuple[StableId, ...] = Field(min_length=1)

    @field_validator("evidence_refs")
    @classmethod
    def reject_duplicate_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_refs 不得重复")
        return value


EvidenceItem: TypeAlias = (
    SpeechSegment
    | SubtitleCue
    | SceneBoundary
    | KeyframeEvidence
    | OcrEvidence
)
