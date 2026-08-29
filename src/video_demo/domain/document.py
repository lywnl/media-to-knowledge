from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from video_demo.domain.base import FrozenModel, Probability, Sha256, StableId
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

RESULT_SCHEMA_VERSION: Literal["4.1.0"] = "4.1.0"
TranscriptSource: TypeAlias = Literal["SUBTITLE", "ASR", "NONE"]
_TITLE_MAX_LENGTH = 200
_VISUAL_CAPTION_MAX_LENGTH = 2_000
_TITLE_WHITESPACE_PATTERN = re.compile(r"\s+")
_DOCUMENT_KEYFRAME_PATH_PATTERN = re.compile(r"^visual/keyframes/([0-9a-f]{64})\.jpg$")


def sanitize_document_title(
    explicit_title: str | None,
    original_filename: str | None = None,
) -> str | None:
    """生成唯一的安全标题；仅文件名回退分支移除扩展名。"""

    candidate = explicit_title
    if candidate is None or not candidate.strip():
        if original_filename is None:
            return None
        filename = original_filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        extension_separator = filename.rfind(".")
        candidate = (
            filename[:extension_separator]
            if extension_separator > 0
            else filename
        )
    cleaned = "".join(
        " " if character in "/\\" or unicodedata.category(character).startswith("C") else character
        for character in candidate
    )
    normalized = _TITLE_WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return normalized[:_TITLE_MAX_LENGTH] or None


class DocumentGenerationConfig(FrozenModel):
    document_title: str | None = Field(default=None, max_length=_TITLE_MAX_LENGTH)
    detail_level: Literal["concise", "standard", "detailed"] = "standard"
    chapter_granularity: Literal["fine", "standard", "coarse"] = "standard"
    include_verbatim_quotes: bool = True
    max_visuals_per_chapter: int = Field(default=2, ge=0, le=3)

    @field_validator("document_title", mode="before")
    @classmethod
    def normalize_document_title(cls, value: object) -> object:
        if value is None or isinstance(value, str):
            return sanitize_document_title(value)
        return value


class VideoDocumentSummary(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    overview_zh: str = Field(max_length=8_000)

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
    visual_content_refs: tuple[StableId, ...] = Field(max_length=48)
    caption: str = Field(min_length=1, max_length=2_000)
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)

    @field_validator("visual_content_refs")
    @classmethod
    def reject_duplicate_content_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("visual_content_refs 不得重复")
        return value


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
    title_evidence_refs: tuple[StableId, ...] = Field(max_length=32)
    summary_zh: str = Field(max_length=4_000)
    summary_evidence_refs: tuple[StableId, ...] = Field(max_length=32)
    body_blocks: tuple[ChapterBodyBlock, ...] = Field(max_length=128)
    claims: tuple[GroundedClaim, ...] = Field(max_length=128)
    content_status: Literal["GROUNDED", "NO_SEMANTIC_EVIDENCE"] = "GROUNDED"
    evidence_refs: tuple[StableId, ...] = Field(max_length=256)
    selected_keyframe_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    transcript_source: TranscriptSource

    @model_validator(mode="after")
    def validate_content_boundary(self) -> SemanticChapter:
        if self.duration_ms > 300_000:
            raise ValueError("单章时长不得超过 5 分钟")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("章节 evidence_refs 不得重复")
        if len(self.title_evidence_refs) != len(set(self.title_evidence_refs)):
            raise ValueError("章节 title_evidence_refs 不得重复")
        if len(self.summary_evidence_refs) != len(set(self.summary_evidence_refs)):
            raise ValueError("章节 summary_evidence_refs 不得重复")
        if not set((*self.title_evidence_refs, *self.summary_evidence_refs)).issubset(
            self.evidence_refs,
        ):
            raise ValueError("章节标题和摘要引用必须属于章节证据闭包")
        if len(self.selected_keyframe_refs) != len(set(self.selected_keyframe_refs)):
            raise ValueError("章节 selected_keyframe_refs 不得重复")
        if self.content_status == "NO_SEMANTIC_EVIDENCE":
            if self.transcript_source != "NONE":
                raise ValueError("NO_SEMANTIC_EVIDENCE 章节必须没有转写来源")
            if (
                self.body_blocks
                or self.claims
                or self.evidence_refs
                or self.title_evidence_refs
                or self.summary_evidence_refs
                or self.selected_keyframe_refs
            ):
                raise ValueError("NO_SEMANTIC_EVIDENCE 章节不得包含事实或证据")
        else:
            if not self.evidence_refs:
                raise ValueError("GROUNDED 章节至少需要一个证据引用")
            if not self.title_evidence_refs or not self.summary_evidence_refs:
                raise ValueError("GROUNDED 章节的标题和摘要至少需要一个证据引用")
        return self


class PromptVersions(FrozenModel):
    chapter_planner: Literal["chapter-planner-v1"]
    chapter_planner_repair: Literal["chapter-planner-repair-v1"]
    chapter_vlm: Literal["chapter-vlm-v1"]
    chapter_vlm_repair: Literal["chapter-vlm-repair-v1"]
    chapter_writer: Literal["chapter-writer-v1"]
    chapter_writer_repair: Literal["chapter-writer-repair-v1"]
    global_editor: Literal["global-editor-v1"]
    global_editor_repair: Literal["global-editor-repair-v1"]


class DocumentGenerationMetadata(FrozenModel):
    document_config: DocumentGenerationConfig
    text_model_id: str = Field(min_length=1, max_length=256)
    vlm_model_id: str = Field(min_length=1, max_length=256)
    prompt_versions: PromptVersions


class VideoUnderstandingResult(FrozenModel):
    schema_version: Literal["4.1.0"] = RESULT_SCHEMA_VERSION
    run_id: StableId
    asset_sha256: Sha256
    summary: VideoDocumentSummary
    chapters: tuple[SemanticChapter, ...] = Field(min_length=1, max_length=240)
    generation: DocumentGenerationMetadata

    @model_validator(mode="after")
    def validate_result_timeline(self) -> VideoUnderstandingResult:
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
        return self


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
    referenced_keyframes: set[str] = set()
    for evidence_item in evidence_by_id.values():
        if isinstance(evidence_item, KeyframeEvidence):
            _validate_document_keyframe(evidence_item)
        elif isinstance(evidence_item, VisualObservationEvidence):
            referenced_keyframes.update(evidence_item.keyframe_refs)
    orphan_keyframes = {
        evidence_id
        for evidence_id, evidence_item in evidence_by_id.items()
        if isinstance(evidence_item, KeyframeEvidence)
        and evidence_id not in referenced_keyframes
    }
    for chapter in result.chapters:
        _validate_attributed_evidence_refs(
            chapter,
            chapter.title_evidence_refs,
            evidence_by_id,
            allow_typed_visual=False,
        )
        _validate_attributed_evidence_refs(
            chapter,
            chapter.summary_evidence_refs,
            evidence_by_id,
            allow_typed_visual=False,
        )
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
            if not block.evidence_refs:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "章节正文块至少需要一个证据引用",
                )
            if isinstance(block, VisualBlock) and block.evidence_refs != (
                block.visual_observation_ref,
            ):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_RELATION_INVALID,
                    "VISUAL block 只能绑定其视觉观察",
                )
            _validate_attributed_evidence_refs(
                chapter,
                block.evidence_refs,
                evidence_by_id,
                allow_typed_visual=isinstance(block, VisualBlock),
            )
            if isinstance(block, VisualBlock):
                observation = evidence_by_id.get(block.visual_observation_ref)
                if not isinstance(observation, VisualObservationEvidence):
                    raise VideoDemoError(
                        ErrorCode.EVIDENCE_TYPE_MISMATCH,
                        "VISUAL block 必须绑定视觉观察",
                    )
                allowed_content_ids = {
                    *(item.visual_content_id for item in observation.content_blocks),
                    *(item.visual_fact_id for item in observation.visual_facts),
                }
                if allowed_content_ids:
                    if not block.visual_content_refs:
                        raise VideoDemoError(
                            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                            "有内容的视觉观察至少选择一个子内容",
                        )
                    if not set(block.visual_content_refs).issubset(allowed_content_ids):
                        raise VideoDemoError(
                            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                            "VISUAL block 引用了未知或跨观察子内容",
                        )
                elif block.visual_content_refs:
                    raise VideoDemoError(
                        ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                        "空内容视觉观察不得选择子内容",
                    )
                if (
                    observation.relation_to_transcript == "CONFLICTING"
                    and block.caption
                    != observation.caption
                ):
                    raise VideoDemoError(
                        ErrorCode.EVIDENCE_RELATION_INVALID,
                        "冲突视觉文案必须是视觉观察的确定性策略投影",
                    )
        for claim in chapter.claims:
            _validate_attributed_evidence_refs(
                chapter,
                claim.evidence_refs,
                evidence_by_id,
                allow_typed_visual=False,
            )
        rebuilt_keyframes = _selected_keyframes_from_body(
            chapter,
            evidence_by_id,
        )
        if rebuilt_keyframes != chapter.selected_keyframe_refs:
            raise VideoDemoError(
                ErrorCode.EVIDENCE_RELATION_INVALID,
                "章节展示关键帧必须严格来自正文所选视觉子内容",
                {"chapter_id": chapter.chapter_id},
            )
        if any(ref not in chapter.evidence_refs for ref in rebuilt_keyframes):
            raise VideoDemoError(
                ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                "章节展示关键帧必须属于章节证据闭包",
                {"chapter_id": chapter.chapter_id},
            )

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
            if not evidence_item.contains(frame):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_CHAPTER,
                    "视觉观察时间范围未覆盖所引用的关键帧",
                    {"evidence_id": evidence_item.evidence_id},
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
    if orphan_keyframes:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
            "最终证据不得包含未被视觉观察引用的孤立关键帧",
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


def _validate_attributed_evidence_refs(
    chapter: SemanticChapter,
    evidence_refs: tuple[str, ...],
    evidence_by_id: dict[str, TimedEvidence],
    *,
    allow_typed_visual: bool,
) -> None:
    for evidence_ref in evidence_refs:
        if evidence_ref not in chapter.evidence_refs:
            raise VideoDemoError(
                ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                "标题、摘要、正文和 Claim 的引用必须属于章节证据闭包",
                {"chapter_id": chapter.chapter_id, "evidence_id": evidence_ref},
            )
        referenced = _require_chapter_evidence(chapter, evidence_ref, evidence_by_id)
        if isinstance(referenced, KeyframeEvidence):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_TYPE_MISMATCH,
                "标题、摘要、正文和 Claim 不能直接引用关键帧证据",
            )
        if not isinstance(referenced, VisualObservationEvidence):
            continue
        relation = referenced.relation_to_transcript
        if relation == "DUPLICATE":
            raise VideoDemoError(
                ErrorCode.EVIDENCE_RELATION_INVALID,
                "DUPLICATE 视觉观察不得重复进入最终正文",
            )
        if relation == "CONFLICTING" and not allow_typed_visual:
            raise VideoDemoError(
                ErrorCode.EVIDENCE_RELATION_INVALID,
                "CONFLICTING 视觉观察只能通过类型化视觉块表达",
            )


def _selected_keyframes_from_body(
    chapter: SemanticChapter,
    evidence_by_id: dict[str, TimedEvidence],
) -> tuple[str, ...]:
    selected: list[str] = []
    for block in chapter.body_blocks:
        if not isinstance(block, VisualBlock):
            continue
        observation = evidence_by_id.get(block.visual_observation_ref)
        if not isinstance(observation, VisualObservationEvidence):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_TYPE_MISMATCH,
                "VISUAL block 必须绑定视觉观察",
            )
        source_by_content = {
            item.visual_content_id: item.source_keyframe_refs
            for item in observation.content_blocks
        }
        source_by_content.update(
            {
                item.visual_fact_id: item.source_keyframe_refs
                for item in observation.visual_facts
            },
        )
        sources = tuple(
            ref
            for content_ref in block.visual_content_refs
            for ref in source_by_content[content_ref]
        ) or observation.keyframe_refs
        sources = tuple(dict.fromkeys(sources))
        if not _selected_frames_are_related(observation, sources):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_RELATION_INVALID,
                "多图视觉正文缺少所选帧之间的对应关系",
                {"evidence_id": observation.evidence_id},
            )
        for ref in sources:
            if ref not in selected:
                selected.append(ref)
    return tuple(selected)


def _selected_frames_are_related(
    observation: VisualObservationEvidence,
    sources: tuple[str, ...],
) -> bool:
    selected = set(sources)
    if len(selected) <= 1:
        return True
    relation_edges = {
        frozenset((relation.from_keyframe_ref, relation.to_keyframe_ref))
        for relation in observation.frame_relations
        if (
            relation.from_keyframe_ref in selected
            and relation.to_keyframe_ref in selected
        )
    }
    if len(selected) == 2:
        return frozenset(selected) in relation_edges
    adjacency: dict[str, set[str]] = {source: set() for source in selected}
    for relation in observation.frame_relations:
        if (
            relation.from_keyframe_ref in selected
            and relation.to_keyframe_ref in selected
        ):
            adjacency[relation.from_keyframe_ref].add(relation.to_keyframe_ref)
            adjacency[relation.to_keyframe_ref].add(relation.from_keyframe_ref)
    visited: set[str] = set()
    stack = [sources[0]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    return visited == selected


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


def _validate_document_keyframe(keyframe: KeyframeEvidence) -> None:
    path_match = _DOCUMENT_KEYFRAME_PATH_PATTERN.fullmatch(keyframe.relative_path)
    if keyframe.mime_type != "image/jpeg":
        raise VideoDemoError(
            ErrorCode.EVIDENCE_RELATION_INVALID,
            "3.0 文档关键帧只允许 JPEG",
            {"evidence_id": keyframe.evidence_id},
        )
    if path_match is None or path_match.group(1) != keyframe.sha256:
        raise VideoDemoError(
            ErrorCode.EVIDENCE_RELATION_INVALID,
            "3.0 文档关键帧必须使用 Run 相对内容寻址路径",
            {"evidence_id": keyframe.evidence_id},
        )
    if (
        keyframe.start_ms != keyframe.timestamp_ms
        or keyframe.end_ms > keyframe.timestamp_ms + 1
    ):
        raise VideoDemoError(
            ErrorCode.EVIDENCE_RELATION_INVALID,
            "3.0 文档关键帧必须使用实际帧时间起始且最长 1ms 的半开区间",
            {"evidence_id": keyframe.evidence_id},
        )
