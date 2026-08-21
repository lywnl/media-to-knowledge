from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.evidence import EvidenceItem, TimelineEvidence
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import TimeRange
from video_demo.fusion.timeline import canonicalize_evidence, validate_timeline


class VideoClipInput(TimeRange):
    clip_id: StableId
    path: Path | None = None
    source_url: str | None = Field(default=None, min_length=1, max_length=4096)
    mime_type: Literal["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"]
    sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        has_path = self.path is not None
        has_url = self.source_url is not None
        if has_path == has_url:
            raise ValueError("视频片段必须且只能提供 path 或 source_url")
        if has_path and self.sha256 is None:
            raise ValueError("本地视频片段必须同时提供 path 和 sha256")
        return self


class SegmentUnderstandingRequest(FrozenModel):
    clip: VideoClipInput
    window: TimeRange
    timeline: tuple[TimelineEvidence, ...] = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        canonical_evidence = canonicalize_evidence(self.evidence)
        canonical_timeline = _canonicalize_timeline(self.timeline)
        object.__setattr__(self, "evidence", canonical_evidence)
        object.__setattr__(self, "timeline", canonical_timeline)
        if not self.clip.contains(self.window):
            raise ValueError("视频片段必须覆盖理解窗口")
        if any(not self.window.contains(item) for item in canonical_timeline):
            raise ValueError("时间轴条目必须位于理解窗口内")
        if any(not self.window.contains(item) for item in canonical_evidence):
            raise ValueError("输入证据必须位于理解窗口内")
        validate_timeline(canonical_timeline, canonical_evidence)
        return self


class SegmentSummaryInput(FrozenModel):
    segment_ref: StableId
    understanding: SegmentUnderstanding


class SummaryUnderstandingRequest(FrozenModel):
    segments: tuple[SegmentSummaryInput, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_segment_refs(self) -> Self:
        refs = tuple(item.segment_ref for item in self.segments)
        if len(refs) != len(set(refs)):
            raise ValueError("segment_ref 不得重复")
        return self


class WholeVideoWindowInput(TimeRange):
    """由本地证据冻结的全片理解窗口，模型不得改写其时间。"""

    window_id: StableId
    timeline: tuple[TimelineEvidence, ...] = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonicalize_and_validate(self) -> Self:
        canonical_evidence = canonicalize_evidence(self.evidence)
        canonical_timeline = _canonicalize_timeline(self.timeline)
        object.__setattr__(self, "evidence", canonical_evidence)
        object.__setattr__(self, "timeline", canonical_timeline)
        if self.duration_ms > 30_000:
            raise ValueError("全片理解窗口不得超过 30 秒")
        if any(not self.contains(item) for item in canonical_evidence):
            raise ValueError("输入证据必须位于全片理解窗口内")
        if any(not self.contains(item) for item in canonical_timeline):
            raise ValueError("时间轴条目必须位于全片理解窗口内")
        validate_timeline(canonical_timeline, canonical_evidence)
        return self


class WholeVideoUnderstandingRequest(FrozenModel):
    video: VideoClipInput
    windows: tuple[WholeVideoWindowInput, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_complete_timeline(self) -> Self:
        if self.video.start_ms != 0 or self.video.duration_ms > 1_800_000:
            raise ValueError("全片视频必须从 0 开始且不得超过 30 分钟")
        ids = tuple(item.window_id for item in self.windows)
        if len(ids) != len(set(ids)):
            raise ValueError("window_id 不得重复")
        cursor = self.video.start_ms
        for window in self.windows:
            if window.start_ms != cursor:
                raise ValueError("全片理解窗口必须连续覆盖完整视频")
            cursor = window.end_ms
        if cursor != self.video.end_ms:
            raise ValueError("全片理解窗口必须连续覆盖完整视频")
        return self


class WholeVideoWindowUnderstanding(FrozenModel):
    window_id: StableId
    understanding: SegmentUnderstanding


class WholeVideoUnderstanding(FrozenModel):
    windows: tuple[WholeVideoWindowUnderstanding, ...] = Field(min_length=1)
    summary: SummaryUnderstanding

    @model_validator(mode="after")
    def reject_duplicate_window_ids(self) -> Self:
        ids = tuple(item.window_id for item in self.windows)
        if len(ids) != len(set(ids)):
            raise ValueError("window_id 不得重复")
        return self


@runtime_checkable
class VideoUnderstandingPort(Protocol):
    def understand_segment(
        self,
        request: SegmentUnderstandingRequest,
    ) -> SegmentUnderstanding: ...

    def summarize_video(
        self,
        request: SummaryUnderstandingRequest,
    ) -> SummaryUnderstanding: ...


@runtime_checkable
class WholeVideoUnderstandingPort(Protocol):
    def understand_video(
        self,
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding: ...


def _canonicalize_timeline(
    timeline: tuple[TimelineEvidence, ...],
) -> tuple[TimelineEvidence, ...]:
    by_id: dict[str, TimelineEvidence] = {}
    for item in timeline:
        existing = by_id.get(item.timeline_id)
        if existing is not None and existing != item:
            raise ValueError("同一 timeline_id 对应了不同内容")
        by_id[item.timeline_id] = item
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (
                item.start_ms,
                item.end_ms,
                item.timeline_id,
            ),
        ),
    )
