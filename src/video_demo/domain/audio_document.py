from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from video_demo.domain.audio_plan import (
    AudioBodyBlock,
    AudioGroundedClaim,
    AudioTranscriptSource,
)
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.run import TimeRange


class AudioDocumentSummary(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    overview_zh: str = Field(max_length=8_000)


class AudioChapter(TimeRange):
    result_type: Literal["AUDIO_CHAPTER"] = "AUDIO_CHAPTER"
    chapter_id: StableId
    title: str = Field(min_length=1, max_length=200)
    title_evidence_refs: tuple[StableId, ...] = Field(max_length=32)
    summary_zh: str = Field(max_length=4_000)
    summary_evidence_refs: tuple[StableId, ...] = Field(max_length=32)
    body_blocks: tuple[AudioBodyBlock, ...] = Field(max_length=128)
    claims: tuple[AudioGroundedClaim, ...] = Field(max_length=128)
    content_status: Literal["GROUNDED", "NO_SEMANTIC_EVIDENCE"] = "GROUNDED"
    evidence_refs: tuple[StableId, ...] = Field(max_length=256)
    transcript_source: AudioTranscriptSource

    @model_validator(mode="after")
    def validate_content_boundary(self) -> AudioChapter:
        if self.end_ms <= self.start_ms:
            raise ValueError("音频章节时间范围非法")
        if self.duration_ms > 300_000:
            raise ValueError("单个音频章节不得超过 5 分钟")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("音频章节 evidence_refs 不得重复")
        if not set((*self.title_evidence_refs, *self.summary_evidence_refs)).issubset(
            self.evidence_refs,
        ):
            raise ValueError("音频章节标题和摘要引用必须属于证据闭包")
        if self.content_status == "NO_SEMANTIC_EVIDENCE":
            if self.transcript_source != "NONE" or self.body_blocks or self.claims:
                raise ValueError("无语义音频章节不得包含转写事实")
        elif (
            not self.evidence_refs
            or not self.title_evidence_refs
            or not self.summary_evidence_refs
        ):
            raise ValueError("有语义音频章节必须包含证据")
        return self


class AudioUnderstandingResult(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: StableId
    asset_sha256: Sha256
    summary: AudioDocumentSummary
    chapters: tuple[AudioChapter, ...] = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_timeline(self) -> AudioUnderstandingResult:
        if self.chapters[0].start_ms != 0:
            raise ValueError("音频章节时间轴必须从 0 开始")
        if self.chapters[-1].end_ms != self.summary.duration_ms:
            raise ValueError("音频摘要时长必须等于最后章节终点")
        if any(
            left.end_ms != right.start_ms
            for left, right in zip(self.chapters, self.chapters[1:], strict=False)
        ):
            raise ValueError("音频章节必须连续且无重叠")
        return self
