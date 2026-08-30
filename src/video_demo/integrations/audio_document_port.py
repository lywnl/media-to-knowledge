"""音频文字模型的独立协议。

这里的请求只携带音频时间片和转写证据，避免把其他媒体的领域契约带入音频。
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Literal, Protocol

from pydantic import Field, model_validator

from video_demo.domain.audio_plan import (
    AudioBaseSegment,
    AudioBodyBlock,
    AudioChapterDraft,
    AudioChapterPlan,
    AudioDocumentConfig,
    AudioGroundedClaim,
    AudioTranscriptEvidence,
    AudioTranscriptSource,
)
from video_demo.domain.base import FrozenModel, Sha256, StableId


class AudioChapterPlanningRequest(FrozenModel):
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    segments: tuple[AudioBaseSegment, ...] = Field(min_length=1, max_length=20_000)
    transcript_evidence: tuple[AudioTranscriptEvidence, ...] = Field(max_length=20_000)
    document_config: AudioDocumentConfig
    prompt_version: Literal["audio-chapter-planner-v1"]


class AudioChapterPlanningResponse(FrozenModel):
    chapter_drafts: tuple[AudioChapterDraft, ...] = Field(min_length=1, max_length=240)


class AudioInvalidModelResponse(FrozenModel):
    content_sha256: Sha256
    validation_errors: tuple[str, ...] = Field(min_length=1, max_length=32)
    safe_json_excerpt: str | None = Field(default=None, max_length=8_000)


class AudioChapterPlanRepairRequest(FrozenModel):
    request: AudioChapterPlanningRequest
    invalid_response: AudioInvalidModelResponse
    allowed_segment_ids: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    prompt_version: Literal["audio-chapter-planner-repair-v1"]

    @model_validator(mode="after")
    def validate_allowed_ids(self) -> AudioChapterPlanRepairRequest:
        expected = tuple(item.segment_id for item in self.request.segments)
        if self.allowed_segment_ids != expected:
            raise ValueError("allowed_segment_ids 必须与请求片段一致")
        return self


class AudioTextPort(Protocol):
    def plan_chapters(
        self,
        request: AudioChapterPlanningRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterPlanningResponse: ...

    def repair_chapter_plan(
        self,
        request: AudioChapterPlanRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterPlanningResponse: ...


class AudioChapterWritingRequest(FrozenModel):
    run_id: StableId
    asset_sha256: Sha256
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    transcript_source: AudioTranscriptSource
    document_config: AudioDocumentConfig
    chapter: AudioChapterPlan
    transcript_evidence: tuple[AudioTranscriptEvidence, ...] = Field(max_length=20_000)
    prompt_version: Literal["audio-chapter-writer-v1"]


class AudioChapterWritingResponse(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    title_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    summary_zh: str = Field(max_length=4_000)
    summary_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    body_blocks: tuple[AudioBodyBlock, ...] = Field(max_length=128)
    claims: tuple[AudioGroundedClaim, ...] = Field(max_length=128)


class AudioChapterWritingRepairRequest(FrozenModel):
    request: AudioChapterWritingRequest
    invalid_response: AudioInvalidModelResponse
    allowed_evidence_ids: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    prompt_version: Literal["audio-chapter-writer-repair-v1"]


class AudioGlobalChapterInput(FrozenModel):
    chapter_id: StableId
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0, le=7_200_000)
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=4_000)

    @model_validator(mode="after")
    def validate_range(self) -> AudioGlobalChapterInput:
        if self.end_ms <= self.start_ms:
            raise ValueError("全局音频章节时间范围非法")
        return self


class AudioGlobalWritingRequest(FrozenModel):
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    chapters: tuple[AudioGlobalChapterInput, ...] = Field(min_length=1, max_length=240)
    prompt_version: Literal["audio-global-editor-v1"]

    @model_validator(mode="after")
    def validate_timeline(self) -> AudioGlobalWritingRequest:
        if self.chapters[0].start_ms != 0 or self.chapters[-1].end_ms != self.duration_ms:
            raise ValueError("全局音频章节必须覆盖完整时长")
        if any(
            left.end_ms != right.start_ms for left, right in pairwise(self.chapters)
        ):
            raise ValueError("全局音频章节必须连续")
        return self


class AudioGlobalWritingResponse(FrozenModel):
    overview_zh: str = Field(max_length=8_000)


class AudioDocumentTextPort(AudioTextPort, Protocol):
    def write_chapter(
        self,
        request: AudioChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterWritingResponse: ...

    def repair_chapter_writing(
        self,
        request: AudioChapterWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterWritingResponse: ...

    def organize_document(
        self,
        request: AudioGlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioGlobalWritingResponse: ...
