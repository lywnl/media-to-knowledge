from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

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
VisualUncertainty = Annotated[str, Field(min_length=1)]


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
    size_bytes: int = Field(ge=1)

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


class VisualTextContent(FrozenModel):
    content_type: Literal["TEXT"] = "TEXT"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    text: str = Field(min_length=1, max_length=16_000)


class VisualCodeContent(FrozenModel):
    content_type: Literal["CODE"] = "CODE"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    language: str | None = Field(default=None, max_length=32)
    code: str = Field(min_length=1, max_length=32_000)


class VisualTableContent(FrozenModel):
    content_type: Literal["TABLE"] = "TABLE"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    columns: tuple[str, ...] = Field(min_length=1, max_length=32)
    rows: tuple[tuple[str, ...], ...] = Field(max_length=256)

    @model_validator(mode="after")
    def validate_row_widths(self) -> Self:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("视觉表格每行列数必须与 columns 一致")
        return self


class VisualFormulaContent(FrozenModel):
    content_type: Literal["FORMULA"] = "FORMULA"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    latex: str = Field(min_length=1, max_length=8_000)
    explanation: str | None = Field(default=None, max_length=4_000)


class VisualDiagramContent(FrozenModel):
    content_type: Literal["DIAGRAM"] = "DIAGRAM"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    description: str = Field(min_length=1, max_length=8_000)
    labels: tuple[str, ...] = Field(max_length=64)
    relations: tuple[str, ...] = Field(max_length=64)


class VisualStateContent(FrozenModel):
    content_type: Literal["STATE"] = "STATE"
    visual_content_id: StableId
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    state_type: Literal["UI_CONTROL", "TERMINAL", "GENERAL"]
    description: str = Field(min_length=1, max_length=8_000)
    key_values: tuple[tuple[str, str], ...] = Field(max_length=64)


VisualContentBlock: TypeAlias = Annotated[
    VisualTextContent
    | VisualCodeContent
    | VisualTableContent
    | VisualFormulaContent
    | VisualDiagramContent
    | VisualStateContent,
    Field(discriminator="content_type"),
]


class GroundedVisualFact(FrozenModel):
    visual_fact_id: StableId
    text: str = Field(min_length=1, max_length=2_000)
    source_keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)


class VisualFieldChange(FrozenModel):
    field: str = Field(min_length=1, max_length=200)
    before: str | None = Field(default=None, max_length=1_000)
    after: str | None = Field(default=None, max_length=1_000)


class VisualFrameRelation(FrozenModel):
    relation_type: Literal[
        "SAME_STATE", "BEFORE_AFTER", "STEP_SEQUENCE",
        "PARAMETER_CHANGE", "VIEW_CHANGE",
    ]
    from_keyframe_ref: StableId
    to_keyframe_ref: StableId
    description: str = Field(min_length=1, max_length=2_000)
    changes: tuple[VisualFieldChange, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_change_contract(self) -> Self:
        if self.relation_type == "SAME_STATE" and self.changes:
            raise ValueError("SAME_STATE 不得包含 changes")
        if self.relation_type == "PARAMETER_CHANGE" and not self.changes:
            raise ValueError("PARAMETER_CHANGE 至少需要一个 change")
        return self


class VisualObservationEvidence(TimedEvidence):
    evidence_type: Literal["VISUAL_OBSERVATION"] = "VISUAL_OBSERVATION"
    chapter_id: StableId
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)
    keyframe_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    transcript_evidence_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    visual_type: Literal[
        "TEXT", "CODE", "TABLE", "FORMULA",
        "DIAGRAM", "UI_CONTROL", "TERMINAL", "GENERAL",
    ]
    caption: str = Field(min_length=1, max_length=2_000)
    content_blocks: tuple[VisualContentBlock, ...] = Field(default=(), max_length=16)
    visual_facts: tuple[GroundedVisualFact, ...] = Field(default=(), max_length=32)
    frame_relations: tuple[VisualFrameRelation, ...] = Field(default=(), max_length=8)
    relation_to_transcript: Literal[
        "SUPPORTING", "COMPLEMENTARY", "DUPLICATE",
        "CONFLICTING", "INDEPENDENT",
    ]
    certainty: Probability
    quality_flags: tuple[str, ...] = Field(default=(), max_length=16)
    uncertainties: tuple[VisualUncertainty, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_local_references(self) -> Self:
        _reject_duplicate_ids(self.target_ids, "target_ids")
        _reject_duplicate_ids(self.keyframe_refs, "keyframe_refs")
        _reject_duplicate_ids(self.transcript_evidence_refs, "transcript_evidence_refs")
        keyframes = set(self.keyframe_refs)
        content_ids = tuple(block.visual_content_id for block in self.content_blocks)
        fact_ids = tuple(fact.visual_fact_id for fact in self.visual_facts)
        _reject_duplicate_ids(content_ids, "visual_content_id")
        _reject_duplicate_ids(fact_ids, "visual_fact_id")
        if set(content_ids) & set(fact_ids):
            raise ValueError("视觉内容与事实 ID 不得重复")
        for block in self.content_blocks:
            if not set(block.source_keyframe_refs).issubset(keyframes):
                raise ValueError("source_keyframe_refs 必须属于当前观察的 keyframe_refs")
        for fact in self.visual_facts:
            if not set(fact.source_keyframe_refs).issubset(keyframes):
                raise ValueError("视觉事实的 source_keyframe_refs 必须属于当前观察")
        for relation in self.frame_relations:
            if (
                relation.from_keyframe_ref not in keyframes
                or relation.to_keyframe_ref not in keyframes
            ):
                raise ValueError("帧关系两端必须属于当前观察")
        if self.relation_to_transcript == "INDEPENDENT" and self.transcript_evidence_refs:
            raise ValueError("INDEPENDENT 观察不得引用转写证据")
        if self.relation_to_transcript != "INDEPENDENT" and not self.transcript_evidence_refs:
            raise ValueError("音画关系需要至少一个转写证据引用")
        if self.relation_to_transcript == "CONFLICTING" and not self.uncertainties:
            raise ValueError("CONFLICTING 观察必须说明 uncertainties")
        return self


class VisualTextContentDraft(FrozenModel):
    content_type: Literal["TEXT"] = "TEXT"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    text: str = Field(min_length=1, max_length=8_000)


class VisualCodeContentDraft(FrozenModel):
    content_type: Literal["CODE"] = "CODE"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    language: str | None = Field(default=None, max_length=32)
    code: str = Field(min_length=1, max_length=16_000)


class VisualTableContentDraft(FrozenModel):
    content_type: Literal["TABLE"] = "TABLE"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    columns: tuple[str, ...] = Field(min_length=1, max_length=32)
    rows: tuple[tuple[str, ...], ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def validate_row_widths(self) -> Self:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("视觉表格草稿每行列数必须与 columns 一致")
        return self


class VisualFormulaContentDraft(FrozenModel):
    content_type: Literal["FORMULA"] = "FORMULA"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    latex: str = Field(min_length=1, max_length=4_000)
    explanation: str | None = Field(default=None, max_length=4_000)


class VisualDiagramContentDraft(FrozenModel):
    content_type: Literal["DIAGRAM"] = "DIAGRAM"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    description: str = Field(min_length=1, max_length=8_000)
    labels: tuple[str, ...] = Field(default=(), max_length=64)
    relations: tuple[str, ...] = Field(default=(), max_length=64)


class VisualStateContentDraft(FrozenModel):
    content_type: Literal["STATE"] = "STATE"
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    state_type: Literal["UI_CONTROL", "TERMINAL", "GENERAL"]
    description: str = Field(min_length=1, max_length=8_000)
    key_values: tuple[tuple[str, str], ...] = Field(default=(), max_length=64)


VisualContentBlockDraft: TypeAlias = Annotated[
    VisualTextContentDraft
    | VisualCodeContentDraft
    | VisualTableContentDraft
    | VisualFormulaContentDraft
    | VisualDiagramContentDraft
    | VisualStateContentDraft,
    Field(discriminator="content_type"),
]


class VisualFactDraft(FrozenModel):
    text: str = Field(min_length=1, max_length=2_000)
    source_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)


class VisualFieldChangeDraft(FrozenModel):
    field: str = Field(min_length=1, max_length=200)
    before: str | None = Field(default=None, max_length=1_000)
    after: str | None = Field(default=None, max_length=1_000)


class VisualFrameRelationDraft(FrozenModel):
    relation_type: Literal[
        "SAME_STATE", "BEFORE_AFTER", "STEP_SEQUENCE",
        "PARAMETER_CHANGE", "VIEW_CHANGE",
    ]
    from_frame_id: StableId
    to_frame_id: StableId
    description: str = Field(min_length=1, max_length=2_000)
    changes: tuple[VisualFieldChangeDraft, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_change_contract(self) -> Self:
        if self.relation_type == "SAME_STATE" and self.changes:
            raise ValueError("SAME_STATE 草稿不得包含 changes")
        if self.relation_type == "PARAMETER_CHANGE" and not self.changes:
            raise ValueError("PARAMETER_CHANGE 草稿至少需要一个 change")
        return self


class ChapterVisualObservation(FrozenModel):
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)
    selected_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=3)
    transcript_evidence_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    visual_type: Literal[
        "TEXT", "CODE", "TABLE", "FORMULA",
        "DIAGRAM", "UI_CONTROL", "TERMINAL", "GENERAL",
    ]
    caption: str = Field(min_length=1, max_length=2_000)
    content_blocks: tuple[VisualContentBlockDraft, ...] = Field(default=(), max_length=16)
    visual_facts: tuple[VisualFactDraft, ...] = Field(default=(), max_length=32)
    frame_relations: tuple[VisualFrameRelationDraft, ...] = Field(default=(), max_length=8)
    relation_to_transcript: Literal[
        "SUPPORTING", "COMPLEMENTARY", "DUPLICATE",
        "CONFLICTING", "INDEPENDENT",
    ]
    certainty: Probability
    quality_flags: tuple[str, ...] = Field(default=(), max_length=16)
    uncertainties: tuple[VisualUncertainty, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_local_references(self) -> Self:
        _reject_duplicate_ids(self.target_ids, "target_ids")
        _reject_duplicate_ids(self.selected_frame_ids, "selected_frame_ids")
        _reject_duplicate_ids(self.transcript_evidence_refs, "transcript_evidence_refs")
        frame_ids = set(self.selected_frame_ids)
        for block in self.content_blocks:
            if not set(block.source_frame_ids).issubset(frame_ids):
                raise ValueError("source_frame_ids 必须属于当前观察的 selected_frame_ids")
        for fact in self.visual_facts:
            if not set(fact.source_frame_ids).issubset(frame_ids):
                raise ValueError("视觉事实的 source_frame_ids 必须属于当前观察")
        for relation in self.frame_relations:
            if relation.from_frame_id not in frame_ids or relation.to_frame_id not in frame_ids:
                raise ValueError("帧关系两端必须属于当前观察")
        if self.relation_to_transcript == "INDEPENDENT" and self.transcript_evidence_refs:
            raise ValueError("INDEPENDENT 草稿不得引用转写证据")
        if self.relation_to_transcript != "INDEPENDENT" and not self.transcript_evidence_refs:
            raise ValueError("音画关系需要至少一个转写证据引用")
        if self.relation_to_transcript == "CONFLICTING" and not self.uncertainties:
            raise ValueError("CONFLICTING 草稿必须说明 uncertainties")
        return self


def _reject_duplicate_ids(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} 不得重复")


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

DocumentEvidenceItem: TypeAlias = Annotated[
    SpeechSegment | SubtitleCue | KeyframeEvidence | VisualObservationEvidence,
    Field(discriminator="evidence_type"),
]
