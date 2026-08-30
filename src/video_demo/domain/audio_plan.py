"""音频章节规划和写作共用的中性领域契约。

这些模型只描述时间轴、转写证据和文字内容。音频结果不需要也不允许携带
画面、帧或场景相关信息。
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Probability, Sha256, StableId
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import TimeRange

AudioTranscriptEvidence: TypeAlias = SpeechSegment | SubtitleCue
AudioTranscriptSource: TypeAlias = Literal["SUBTITLE", "ASR", "NONE"]


class AudioDocumentConfig(FrozenModel):
    document_title: str | None = Field(default=None, max_length=200)
    detail_level: Literal["concise", "standard", "detailed"] = "standard"
    chapter_granularity: Literal["fine", "standard", "coarse"] = "standard"
    include_verbatim_quotes: bool = True


class AudioBaseSegment(TimeRange):
    segment_id: StableId
    evidence_refs: tuple[StableId, ...] = Field(max_length=256)
    transcript_source: AudioTranscriptSource

    @model_validator(mode="after")
    def validate_evidence_source(self) -> AudioBaseSegment:
        if self.transcript_source == "NONE" and self.evidence_refs:
            raise ValueError("无转写来源的音频片段不得包含证据")
        if self.transcript_source != "NONE" and not self.evidence_refs:
            raise ValueError("有转写来源的音频片段必须包含证据")
        return self


class AudioChapterDraft(FrozenModel):
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    title_hint: str = Field(min_length=1, max_length=200)


class AudioChapterPlan(TimeRange):
    chapter_id: StableId
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    title_hint: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_duration(self) -> AudioChapterPlan:
        if self.duration_ms > 300_000:
            raise ValueError("单个音频章节不得超过 5 分钟")
        return self


class AudioGroundedClaim(FrozenModel):
    text: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    certainty: Probability

    @model_validator(mode="after")
    def reject_duplicate_refs(self) -> AudioGroundedClaim:
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("音频结论 evidence_refs 不得重复")
        return self


class AudioParagraphBlock(FrozenModel):
    block_type: Literal["PARAGRAPH"] = "PARAGRAPH"
    text: str = Field(min_length=1, max_length=16_000)
    evidence_refs: tuple[StableId, ...] = Field(max_length=32)


class AudioBulletListBlock(FrozenModel):
    block_type: Literal["BULLET_LIST"] = "BULLET_LIST"
    items: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_refs: tuple[StableId, ...] = Field(max_length=32)


class AudioQuoteBlock(FrozenModel):
    block_type: Literal["QUOTE"] = "QUOTE"
    text: str = Field(min_length=1, max_length=8_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


class AudioCodeBlock(FrozenModel):
    block_type: Literal["CODE"] = "CODE"
    language: str | None = Field(default=None, max_length=32)
    code: str = Field(min_length=1, max_length=32_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


class AudioTableBlock(FrozenModel):
    block_type: Literal["TABLE"] = "TABLE"
    columns: tuple[str, ...] = Field(min_length=1, max_length=32)
    rows: tuple[tuple[str, ...], ...] = Field(max_length=256)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_row_widths(self) -> AudioTableBlock:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("音频表格每行列数必须与 columns 一致")
        return self


class AudioFormulaBlock(FrozenModel):
    block_type: Literal["FORMULA"] = "FORMULA"
    latex: str = Field(min_length=1, max_length=4_000)
    explanation: str = Field(max_length=4_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


AudioBodyBlock: TypeAlias = Annotated[
    AudioParagraphBlock
    | AudioBulletListBlock
    | AudioQuoteBlock
    | AudioCodeBlock
    | AudioTableBlock
    | AudioFormulaBlock,
    Field(discriminator="block_type"),
]
