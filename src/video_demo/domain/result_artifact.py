from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import StrictInt, field_validator, model_validator

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import (
    EvidenceItem,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import VideoUnderstandingResult

TranscriptSource: TypeAlias = Literal["SUBTITLE", "ASR", "NONE"]
ARTIFACT_ENVELOPE_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
RESULT_STAGE_NAMES = frozenset(
    {
        "REGISTER",
        "PROBE",
        "TRANSCODE",
        "SPEECH",
        "SPEECH_ASR",
        "VISUAL",
        "VISUAL_SCENE_DETECT",
        "VISUAL_FRAME_EXTRACT",
        "VISUAL_KEYFRAME_SELECT",
        "VISUAL_OCR",
        "VISUAL_WAIT_SPEECH",
        "VISUAL_FUSION",
        "FUSION",
        "UNDERSTANDING",
        "RESULT",
    },
)
SPEECH_CACHE_HIT_STAGE_NAMES = frozenset({"SPEECH_ASR"})


class ResultArtifactPayload(FrozenModel):
    """`ResultQueryService.persist` 唯一允许写入的生产结果阶段 payload。"""

    result: VideoUnderstandingResult
    evidence: tuple[EvidenceItem, ...]
    stage_metrics: dict[str, StrictInt]
    stage_cache_hits: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    warnings: tuple[str, ...]
    transcript_source: TranscriptSource

    @field_validator("stage_metrics")
    @classmethod
    def validate_stage_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if any(
            not key or type(metric) is not int or metric < 0
            for key, metric in value.items()
        ):
            raise ValueError("阶段指标必须是非空名称和整数值")
        if set(value).difference(RESULT_STAGE_NAMES):
            raise ValueError("阶段名称不在允许白名单中")
        return value

    @field_validator("stage_cache_hits")
    @classmethod
    def validate_stage_cache_hits(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value) or len(value) != len(set(value)):
            raise ValueError("缓存命中阶段不得为空或重复")
        if set(value).difference(SPEECH_CACHE_HIT_STAGE_NAMES):
            raise ValueError("缓存命中只能记录语音子阶段")
        return value

    @model_validator(mode="after")
    def validate_stage_cache_contract(self) -> ResultArtifactPayload:
        cache_hits = set(self.stage_cache_hits)
        if not cache_hits.issubset(self.stage_metrics):
            raise ValueError("缓存命中阶段必须是阶段指标的子集")
        if any(self.stage_metrics[stage] != 0 for stage in cache_hits):
            raise ValueError("缓存命中阶段耗时必须为 0")
        return self

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not warning for warning in value) or len(value) != len(set(value)):
            raise ValueError("运行警告不得为空或重复")
        return value

    @model_validator(mode="after")
    def validate_transcript_source_evidence(self) -> ResultArtifactPayload:
        subtitle_cues = tuple(item for item in self.evidence if isinstance(item, SubtitleCue))
        asr_evidence = tuple(item for item in self.evidence if isinstance(item, SpeechSegment))
        if self.transcript_source == "SUBTITLE":
            if not subtitle_cues or asr_evidence:
                raise ValueError("字幕来源必须包含字幕且不得包含 ASR 语音证据")
        elif self.transcript_source == "ASR":
            if subtitle_cues:
                raise ValueError("ASR 来源不得包含字幕证据")
        elif subtitle_cues or asr_evidence:
            raise ValueError("无文本来源不得包含字幕或 ASR 语音证据")
        return self
