from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import Field, model_validator

from video_demo.domain.base import (
    FrozenModel,
    LanguageCode,
    Sha256,
    StableId,
    UniqueStringTuplesMixin,
)
from video_demo.domain.evidence import SpeakerId, TimedEvidence
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError


class SemanticFields(UniqueStringTuplesMixin, FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(min_length=1, max_length=4000)
    speakers: tuple[SpeakerId, ...] = ()
    languages: tuple[LanguageCode, ...] = ()
    topics: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    original_keywords: tuple[str, ...] = ()


class SegmentUnderstanding(SemanticFields):
    """Qwen 可返回的片段语义; 此契约故意不包含时间字段。"""

    evidence_refs: tuple[StableId, ...] = Field(min_length=1)


class SummaryUnderstanding(SemanticFields):
    """Qwen 可返回的视频级语义；时间和章节边界由程序生成。"""


class VideoSegment(TimeRange, SemanticFields):
    result_type: Literal["VIDEO_SEGMENT"] = "VIDEO_SEGMENT"
    segment_id: StableId
    evidence_refs: tuple[StableId, ...] = Field(min_length=1)
    retrieval_text: str = Field(min_length=1)
    retrieval_hash: Sha256

    @model_validator(mode="after")
    def validate_retrieval_hash(self) -> Self:
        _validate_retrieval_hash(self.retrieval_text, self.retrieval_hash)
        return self


class SummaryChapter(TimeRange, UniqueStringTuplesMixin, FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    segment_ids: tuple[StableId, ...] = Field(min_length=1)


class VideoSummary(SemanticFields):
    result_type: Literal["VIDEO_SUMMARY"] = "VIDEO_SUMMARY"
    duration_ms: int = Field(gt=0, le=1_800_000)
    chapters: tuple[SummaryChapter, ...]
    retrieval_text: str = Field(min_length=1)
    retrieval_hash: Sha256

    @model_validator(mode="after")
    def validate_retrieval_hash(self) -> Self:
        _validate_retrieval_hash(self.retrieval_text, self.retrieval_hash)
        return self


class VideoUnderstandingResult(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: StableId
    asset_sha256: Sha256
    segments: tuple[VideoSegment, ...] = Field(min_length=1)
    summary: VideoSummary

    @model_validator(mode="after")
    def validate_result_timeline(self) -> Self:
        segment_by_id = {segment.segment_id: segment for segment in self.segments}
        if len(segment_by_id) != len(self.segments):
            raise ValueError("segment_id 不得重复")
        for segment in self.segments:
            if segment.end_ms > self.summary.duration_ms:
                raise ValueError("片段时间不得超过视频时长")
        for chapter in self.summary.chapters:
            if chapter.end_ms > self.summary.duration_ms:
                raise ValueError("章节时间不得超过视频时长")
            for segment_id in chapter.segment_ids:
                referenced_segment = segment_by_id.get(segment_id)
                if referenced_segment is not None and not chapter.contains(referenced_segment):
                    raise ValueError("章节必须覆盖其引用的片段")
        return self


def _validate_retrieval_hash(retrieval_text: str, retrieval_hash: str) -> None:
    expected = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
    if retrieval_hash != expected:
        raise ValueError("retrieval_hash 必须等于 retrieval_text 的 SHA-256")


def validate_evidence_references(
    result: VideoUnderstandingResult,
    evidence: Iterable[TimedEvidence],
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

    segment_ids = {segment.segment_id for segment in result.segments}
    if len(segment_ids) != len(result.segments):
        raise VideoDemoError(ErrorCode.DUPLICATE_SEGMENT_ID, "片段 ID 重复")

    for segment in result.segments:
        for evidence_ref in segment.evidence_refs:
            referenced_evidence = evidence_by_id.get(evidence_ref)
            if referenced_evidence is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
                    "片段引用了不存在的证据",
                    {"segment_id": segment.segment_id, "evidence_id": evidence_ref},
                )
            if not segment.contains(referenced_evidence):
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_SEGMENT,
                    "片段引用的证据超出自身时间范围",
                    {"segment_id": segment.segment_id, "evidence_id": evidence_ref},
                )

    for chapter in result.summary.chapters:
        for segment_id in chapter.segment_ids:
            if segment_id not in segment_ids:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_SEGMENT_REFERENCE,
                    "摘要章节引用了不存在的片段",
                    {"segment_id": segment_id},
                )
