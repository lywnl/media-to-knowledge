from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.document import TranscriptSource
from video_demo.domain.run import TimeRange


class BaseSegment(TimeRange):
    segment_id: StableId
    evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=256)
    scene_refs: tuple[StableId, ...] = Field(default=(), max_length=8)
    transcript_source: TranscriptSource


class VisualSearchTarget(FrozenModel):
    target_id: StableId
    purpose: Literal["SEMANTIC", "BASE_COVERAGE"]
    query_zh: str = Field(min_length=1, max_length=500)
    anchor_evidence_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    scene_refs: tuple[StableId, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_target_bindings(self) -> VisualSearchTarget:
        if self.purpose == "SEMANTIC":
            if not 1 <= len(self.anchor_evidence_refs) <= 3:
                raise ValueError("SEMANTIC 目标必须绑定 1~3 个转写锚点")
            if self.scene_refs:
                raise ValueError("SEMANTIC 目标不能绑定 scene_refs")
        elif self.anchor_evidence_refs or not self.scene_refs:
            raise ValueError("BASE_COVERAGE 目标必须绑定 scene_refs 且不能绑定转写锚点")
        return self


class ChapterPlan(TimeRange):
    chapter_id: StableId
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=240)
    title_hint: str = Field(min_length=1, max_length=200)
    visual_mode: Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]
    semantic_targets: tuple[VisualSearchTarget, ...] = Field(max_length=4)
    base_coverage_targets: tuple[VisualSearchTarget, ...] = Field(max_length=2)

    @model_validator(mode="after")
    def validate_visual_budget(self) -> ChapterPlan:
        if self.duration_ms > 300_000:
            raise ValueError("单章时长不得超过 5 分钟")
        semantic_count = len(self.semantic_targets)
        maximum = 4 if self.visual_mode in {"COMPARISON", "MULTI_STEP"} else 2
        if semantic_count > maximum:
            raise ValueError("章节语义视觉目标超过复杂度预算")
        if any(target.purpose != "SEMANTIC" for target in self.semantic_targets):
            raise ValueError("semantic_targets 只能包含 SEMANTIC 目标")
        if any(target.purpose != "BASE_COVERAGE" for target in self.base_coverage_targets):
            raise ValueError("base_coverage_targets 只能包含 BASE_COVERAGE 目标")
        return self


class VisualTargetDraft(FrozenModel):
    query_zh: str = Field(min_length=1, max_length=500)
    anchor_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)


class ChapterDraft(FrozenModel):
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=240)
    title_hint: str = Field(min_length=1, max_length=200)
    visual_mode: Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]
    semantic_targets: tuple[VisualTargetDraft, ...] = Field(max_length=4)


class FrameCandidateArtifact(FrozenModel):
    frame_id: StableId
    timestamp_ms: int = Field(ge=0)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    perceptual_hash: str = Field(min_length=8, max_length=128)
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=4)


class ChapterFrameSet(FrozenModel):
    chapter_id: StableId
    candidates: tuple[FrameCandidateArtifact, ...] = Field(max_length=6)
    degraded: bool = False
