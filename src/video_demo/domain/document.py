from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Probability, Sha256, StableId, stable_identifier
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    TimedEvidence,
    VisualObservationEvidence,
)
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError

RESULT_SCHEMA_VERSION: Literal["3.0.0"] = "3.0.0"
TranscriptSource: TypeAlias = Literal["SUBTITLE", "ASR", "NONE"]


class DocumentGenerationConfig(FrozenModel):
    detail_level: Literal["concise", "standard", "detailed"] = "standard"
    chapter_granularity: Literal["fine", "standard", "coarse"] = "standard"
    include_verbatim_quotes: bool = True
    max_visuals_per_chapter: int = Field(default=2, ge=1, le=3)
    uncertainty_policy: Literal["explicit", "conservative"] = "explicit"


class SummaryPoint(FrozenModel):
    text: str = Field(min_length=1, max_length=1_000)
    chapter_refs: tuple[StableId, ...] = Field(min_length=1, max_length=240)


class VideoDocumentSummary(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    overview_zh: str = Field(max_length=8_000)
    key_points: tuple[SummaryPoint, ...] = Field(max_length=64)
    retrieval_text: str = Field(max_length=64_000)
    retrieval_hash: Sha256

    @model_validator(mode="after")
    def validate_retrieval_hash(self) -> VideoDocumentSummary:
        _validate_retrieval_hash(self.retrieval_text, self.retrieval_hash)
        return self


class SemanticSection(FrozenModel):
    section_id: StableId
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=4_000)
    chapter_refs: tuple[StableId, ...] = Field(min_length=1, max_length=240)


class SectionDraft(FrozenModel):
    """全局编辑器返回的草稿；最终 section_id 由程序生成。"""

    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=4_000)
    chapter_refs: tuple[StableId, ...] = Field(min_length=1, max_length=240)


class GroundedClaim(FrozenModel):
    text: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    certainty: Probability

    @model_validator(mode="after")
    def reject_duplicate_refs(self) -> GroundedClaim:
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("Claim 的 evidence_refs 不得重复")
        return self


class ParagraphBlock(FrozenModel):
    block_type: Literal["PARAGRAPH"] = "PARAGRAPH"
    text: str = Field(min_length=1, max_length=16_000)
    evidence_refs: tuple[StableId, ...] = Field(max_length=32)


class BulletListBlock(FrozenModel):
    block_type: Literal["BULLET_LIST"] = "BULLET_LIST"
    items: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_refs: tuple[StableId, ...] = Field(max_length=32)


class QuoteBlock(FrozenModel):
    block_type: Literal["QUOTE"] = "QUOTE"
    text: str = Field(min_length=1, max_length=8_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


class CodeBlock(FrozenModel):
    block_type: Literal["CODE"] = "CODE"
    language: str | None = Field(default=None, max_length=32)
    code: str = Field(min_length=1, max_length=32_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


class TableBlock(FrozenModel):
    block_type: Literal["TABLE"] = "TABLE"
    columns: tuple[str, ...] = Field(min_length=1, max_length=32)
    rows: tuple[tuple[str, ...], ...] = Field(max_length=256)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_row_widths(self) -> TableBlock:
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("表格每行列数必须与 columns 一致")
        return self


class FormulaBlock(FrozenModel):
    block_type: Literal["FORMULA"] = "FORMULA"
    latex: str = Field(min_length=1, max_length=4_000)
    explanation: str = Field(max_length=4_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


class VisualBlock(FrozenModel):
    block_type: Literal["VISUAL"] = "VISUAL"
    visual_observation_ref: StableId
    caption: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)


ChapterBodyBlock: TypeAlias = Annotated[
    ParagraphBlock
    | BulletListBlock
    | QuoteBlock
    | CodeBlock
    | TableBlock
    | FormulaBlock
    | VisualBlock,
    Field(discriminator="block_type"),
]


class SemanticChapter(TimeRange):
    result_type: Literal["SEMANTIC_CHAPTER"] = "SEMANTIC_CHAPTER"
    chapter_id: StableId
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=4_000)
    body_blocks: tuple[ChapterBodyBlock, ...] = Field(max_length=128)
    claims: tuple[GroundedClaim, ...] = Field(max_length=128)
    content_status: Literal["GROUNDED", "NO_SEMANTIC_EVIDENCE"] = "GROUNDED"
    evidence_refs: tuple[StableId, ...] = Field(max_length=256)
    selected_keyframe_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    transcript_source: TranscriptSource
    retrieval_text: str = Field(max_length=64_000)
    retrieval_hash: Sha256

    @model_validator(mode="after")
    def validate_content_boundary(self) -> SemanticChapter:
        if self.duration_ms > 300_000:
            raise ValueError("单章时长不得超过 5 分钟")
        _validate_retrieval_hash(self.retrieval_text, self.retrieval_hash)
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("章节 evidence_refs 不得重复")
        if len(self.selected_keyframe_refs) != len(set(self.selected_keyframe_refs)):
            raise ValueError("章节 selected_keyframe_refs 不得重复")
        if self.content_status == "NO_SEMANTIC_EVIDENCE":
            if self.transcript_source != "NONE":
                raise ValueError("NO_SEMANTIC_EVIDENCE 章节必须没有转写来源")
            if self.body_blocks or self.claims or self.evidence_refs or self.selected_keyframe_refs:
                raise ValueError("NO_SEMANTIC_EVIDENCE 章节不得包含事实或证据")
            if self.retrieval_text:
                raise ValueError("NO_SEMANTIC_EVIDENCE 章节检索文本必须为空")
        elif not self.evidence_refs:
            raise ValueError("GROUNDED 章节至少需要一个证据引用")
        return self


class DocumentGenerationMetadata(FrozenModel):
    document_config: DocumentGenerationConfig
    text_model_id: str = Field(min_length=1, max_length=256)
    vlm_model_id: str = Field(min_length=1, max_length=256)
    prompt_versions: PromptVersions


class PromptVersions(FrozenModel):
    chapter_planner: Literal["chapter-planner-v1"]
    chapter_vlm: Literal["chapter-vlm-v1"]
    chapter_writer: Literal["chapter-writer-v1"]
    global_editor: Literal["global-editor-v1"]


class VideoUnderstandingResult(FrozenModel):
    schema_version: Literal["3.0.0"] = RESULT_SCHEMA_VERSION
    run_id: StableId
    asset_sha256: Sha256
    summary: VideoDocumentSummary
    sections: tuple[SemanticSection, ...] = Field(min_length=1, max_length=240)
    chapters: tuple[SemanticChapter, ...] = Field(min_length=1, max_length=240)
    generation: DocumentGenerationMetadata

    @model_validator(mode="after")
    def validate_result_timeline_and_sections(self) -> VideoUnderstandingResult:
        if self.chapters[0].start_ms != 0:
            raise ValueError("章节时间轴必须从 0 开始")
        if self.chapters[-1].end_ms != self.summary.duration_ms:
            raise ValueError("Summary duration 必须等于最后章节终点")
        chapter_ids = tuple(chapter.chapter_id for chapter in self.chapters)
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("chapter_id 不得重复")
        for previous, current in zip(self.chapters[:-1], self.chapters[1:], strict=True):
            if previous.end_ms != current.start_ms:
                raise ValueError("章节必须连续且无重叠")
        chapter_set = set(chapter_ids)
        section_refs: list[str] = []
        for section in self.sections:
            if len(section.chapter_refs) != len(set(section.chapter_refs)):
                raise ValueError("Section chapter_refs 不得重复")
            if any(ref not in chapter_set for ref in section.chapter_refs):
                raise ValueError("Section 引用了不存在的章节")
            section_refs.extend(section.chapter_refs)
        if tuple(section_refs) != chapter_ids:
            raise ValueError("Section 必须按顺序完整覆盖每个章节一次")
        grounded = {
            chapter.chapter_id
            for chapter in self.chapters
            if chapter.content_status == "GROUNDED"
        }
        for point in self.summary.key_points:
            if any(ref not in grounded for ref in point.chapter_refs):
                raise ValueError("SummaryPoint 只能引用已有的 GROUNDED 章节")
        return self


def section_id_for(asset_sha256: str, ordered_chapter_refs: tuple[str, ...]) -> str:
    return stable_identifier(
        "section",
        {"asset_sha256": asset_sha256, "chapter_refs": ordered_chapter_refs},
    )


def validate_evidence_references(
    result: VideoUnderstandingResult,
    evidence: Iterable[DocumentEvidenceItem],
) -> None:
    evidence_by_id: dict[str, TimedEvidence] = {}
    for item in evidence:
        if item.evidence_id in evidence_by_id:
            raise VideoDemoError(
                ErrorCode.DUPLICATE_EVIDENCE_ID,
                "证据 ID 重复",
                {"evidence_id": item.evidence_id},
            )
        evidence_by_id[item.evidence_id] = item

    chapters_by_id = {chapter.chapter_id: chapter for chapter in result.chapters}
    for chapter in result.chapters:
        for evidence_ref in chapter.evidence_refs:
            _require_chapter_evidence(chapter, evidence_ref, evidence_by_id)
        for keyframe_ref in chapter.selected_keyframe_refs:
            referenced = evidence_by_id.get(keyframe_ref)
            if referenced is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "章节引用了不存在的关键帧",
                    {"chapter_id": chapter.chapter_id, "evidence_id": keyframe_ref},
                )
            if not isinstance(referenced, KeyframeEvidence):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_TYPE_MISMATCH,
                    "章节 selected_keyframe_refs 只能引用关键帧证据",
                )
            if not chapter.contains(referenced):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                    "章节引用的关键帧超出自身时间范围",
                )
        for block in chapter.body_blocks:
            for evidence_ref in block.evidence_refs:
                referenced = _require_chapter_evidence(chapter, evidence_ref, evidence_by_id)
                if isinstance(block, VisualBlock) and not isinstance(
                    referenced,
                    VisualObservationEvidence,
                ):
                    raise VideoDemoError(
                        ErrorCode.EVIDENCE_TYPE_MISMATCH,
                        "VISUAL block 只能引用视觉观察证据",
                    )
            if (
                isinstance(block, VisualBlock)
                and block.visual_observation_ref not in block.evidence_refs
            ):
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "VISUAL block 的观察引用必须属于 evidence_refs",
                )
        for claim in chapter.claims:
            for evidence_ref in claim.evidence_refs:
                _require_chapter_evidence(chapter, evidence_ref, evidence_by_id)

    for evidence_item in evidence_by_id.values():
        if not isinstance(evidence_item, VisualObservationEvidence):
            continue
        observation_chapter = chapters_by_id.get(evidence_item.chapter_id)
        if observation_chapter is None:
            raise VideoDemoError(
                ErrorCode.UNKNOWN_CHAPTER_REFERENCE,
                "视觉观察引用了不存在的章节",
            )
        if not observation_chapter.contains(evidence_item):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                "视觉观察时间范围超出所属章节",
                {"evidence_id": evidence_item.evidence_id},
            )
        for frame_ref in evidence_item.keyframe_refs:
            frame = evidence_by_id.get(frame_ref)
            if frame is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "视觉观察引用了不存在的关键帧",
                    {"evidence_id": frame_ref},
                )
            if not isinstance(frame, KeyframeEvidence):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_TYPE_MISMATCH,
                    "视觉观察只能引用关键帧证据",
                )
            if not observation_chapter.contains(frame):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                    "关键帧不属于视觉观察所在章节",
                )
        keyframe_times = {
            frame_ref: _keyframe_timestamp(evidence_by_id, frame_ref)
            for frame_ref in evidence_item.keyframe_refs
        }
        for relation in evidence_item.frame_relations:
            if (
                keyframe_times[relation.from_keyframe_ref]
                >= keyframe_times[relation.to_keyframe_ref]
            ):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_RELATION_INVALID,
                    "帧关系必须按真实时间从早到晚",
                    {"evidence_id": evidence_item.evidence_id},
                )
        for transcript_ref in evidence_item.transcript_evidence_refs:
            transcript = evidence_by_id.get(transcript_ref)
            if transcript is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "视觉观察引用了不存在的转写证据",
                    {"evidence_id": transcript_ref},
                )
            if not isinstance(transcript, (SpeechSegment, SubtitleCue)):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_TYPE_MISMATCH,
                    "视觉观察的音频引用必须是 ASR 或字幕证据",
                )
            if not observation_chapter.contains(transcript):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                    "视觉观察的转写证据不属于所在章节",
                )


def _require_chapter_evidence(
    chapter: SemanticChapter,
    evidence_ref: str,
    evidence_by_id: dict[str, TimedEvidence],
) -> TimedEvidence:
    referenced = evidence_by_id.get(evidence_ref)
    if referenced is None:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
            "章节引用了不存在的证据",
            {"chapter_id": chapter.chapter_id, "evidence_id": evidence_ref},
        )
    if not chapter.contains(referenced):
        raise VideoDemoError(
            ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
            "章节引用的证据超出自身时间范围",
            {"chapter_id": chapter.chapter_id, "evidence_id": evidence_ref},
        )
    return referenced


def _keyframe_timestamp(
    evidence_by_id: dict[str, TimedEvidence],
    evidence_id: str,
) -> int:
    item = evidence_by_id[evidence_id]
    if not isinstance(item, KeyframeEvidence):
        raise VideoDemoError(
            ErrorCode.EVIDENCE_TYPE_MISMATCH,
            "帧关系只能引用关键帧证据",
        )
    return item.timestamp_ms


def _validate_retrieval_hash(text: str, digest: str) -> None:
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError("retrieval_hash 必须等于 retrieval_text 的 SHA-256")
