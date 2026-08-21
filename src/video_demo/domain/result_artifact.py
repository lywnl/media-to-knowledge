from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import StrictInt, field_validator, model_validator

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    EvidenceItem,
    SpeakerTurn,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import VideoUnderstandingResult

TranscriptSource: TypeAlias = Literal["SUBTITLE", "ASR", "NONE"]


class ResultArtifactPayload(FrozenModel):
    """`ResultQueryService.persist` 唯一允许写入的生产结果阶段 payload。"""

    result: VideoUnderstandingResult
    evidence: tuple[EvidenceItem, ...]
    stage_metrics: dict[str, StrictInt]
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    warnings: tuple[str, ...]
    transcript_source: TranscriptSource

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

    @model_validator(mode="after")
    def validate_transcript_source_evidence(self) -> ResultArtifactPayload:
        subtitle_cues = tuple(item for item in self.evidence if isinstance(item, SubtitleCue))
        asr_evidence = tuple(
            item
            for item in self.evidence
            if isinstance(item, (SpeechSegment, AlignedWord, SpeakerTurn, AudioEvent))
        )
        if self.transcript_source == "SUBTITLE":
            if not subtitle_cues or asr_evidence:
                raise ValueError("字幕来源必须包含字幕且不得包含 ASR 语音证据")
        elif self.transcript_source == "ASR":
            if subtitle_cues:
                raise ValueError("ASR 来源不得包含字幕证据")
        elif subtitle_cues or asr_evidence:
            raise ValueError("无文本来源不得包含字幕或 ASR 语音证据")
        return self
