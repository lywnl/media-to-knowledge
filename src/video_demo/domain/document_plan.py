from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.document import TranscriptSource
from video_demo.domain.run import TimeRange

_CANDIDATE_PATH_PATTERN = re.compile(r"^visual/candidates/([0-9a-f]{64})\.jpg$")


class BaseSegment(TimeRange):
    segment_id: StableId
    evidence_refs: tuple[StableId, ...] = Field(max_length=256)
    scene_refs: tuple[StableId, ...] = Field(default=(), max_length=8)
    transcript_source: TranscriptSource

    @model_validator(mode="after")
    def validate_transcript_source(self) -> BaseSegment:
        if self.transcript_source == "NONE" and self.evidence_refs:
            raise ValueError("NONE 片段不得包含转写证据")
        if self.transcript_source != "NONE" and not self.evidence_refs:
            raise ValueError("有转写来源的片段必须包含证据")
        return self


class VisualSearchTarget(FrozenModel):
    target_id: StableId
    purpose: Literal["SEMANTIC", "BASE_COVERAGE"]
    query_zh: str = Field(min_length=1, max_length=500)
    anchor_evidence_refs: tuple[StableId, ...] = Field(default=(), max_length=3)
    scene_refs: tuple[StableId, ...] = Field(default=(), max_length=8)
    sample_timestamps_ms: tuple[int, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def validate_target_bindings(self) -> VisualSearchTarget:
        if self.purpose == "SEMANTIC":
            if not 1 <= len(self.anchor_evidence_refs) <= 3:
                raise ValueError("SEMANTIC 目标必须绑定 1~3 个转写锚点")
            if self.scene_refs or self.sample_timestamps_ms:
                raise ValueError("SEMANTIC 目标不能绑定场景或程序采样时间")
        elif self.anchor_evidence_refs:
            raise ValueError("BASE_COVERAGE 目标不能绑定转写锚点")
        elif bool(self.scene_refs) == bool(self.sample_timestamps_ms):
            raise ValueError(
                "BASE_COVERAGE 目标必须绑定 scene_refs 或 sample_timestamps_ms (二选一)",
            )
        elif self.scene_refs and not 1 <= len(self.scene_refs) <= 2:
            raise ValueError("BASE_COVERAGE 目标必须绑定 1~2 个 scene_refs")
        elif self.sample_timestamps_ms and not 1 <= len(self.sample_timestamps_ms) <= 2:
            raise ValueError("BASE_COVERAGE 目标必须绑定 1~2 个 sample_timestamps_ms")
        return self


class ChapterPlan(TimeRange):
    chapter_id: StableId
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
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
        if self.visual_mode == "NONE" and semantic_count:
            raise ValueError("visual_mode=NONE 不得包含语义视觉目标")
        if self.visual_mode in {"COMPARISON", "MULTI_STEP"}:
            if semantic_count < 2:
                raise ValueError("复杂视觉模式至少需要两个语义目标")
            anchor_groups = [set(target.anchor_evidence_refs) for target in self.semantic_targets]
            for index, group in enumerate(anchor_groups):
                if any(group & other for other in anchor_groups[index + 1 :]):
                    raise ValueError("复杂视觉模式的锚点组必须不重叠")
        if any(target.purpose != "SEMANTIC" for target in self.semantic_targets):
            raise ValueError("semantic_targets 只能包含 SEMANTIC 目标")
        if any(target.purpose != "BASE_COVERAGE" for target in self.base_coverage_targets):
            raise ValueError("base_coverage_targets 只能包含 BASE_COVERAGE 目标")
        for target in self.base_coverage_targets:
            if any(
                timestamp < self.start_ms or timestamp >= self.end_ms
                for timestamp in target.sample_timestamps_ms
            ):
                raise ValueError("sample_timestamps_ms 必须位于章节范围")
        return self


class VisualTargetDraft(FrozenModel):
    query_zh: str = Field(min_length=1, max_length=500)
    anchor_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=3)


class ChapterDraft(FrozenModel):
    segment_refs: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    title_hint: str = Field(min_length=1, max_length=200)
    visual_mode: Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]
    semantic_targets: tuple[VisualTargetDraft, ...] = Field(max_length=4)


class FrameCandidateArtifact(FrozenModel):
    frame_id: StableId
    timestamp_ms: int = Field(ge=0)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    mime_type: Literal["image/jpeg"] = "image/jpeg"
    perceptual_hash: str = Field(min_length=8, max_length=128)
    target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_content_addressed_path(self) -> FrameCandidateArtifact:
        match = _CANDIDATE_PATH_PATTERN.fullmatch(self.relative_path)
        if match is None or match.group(1) != self.sha256:
            raise ValueError("候选帧路径必须是当前 Run 根下的内容寻址 JPEG")
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("候选帧 target_ids 不得重复")
        return self


class ChapterFrameSet(FrozenModel):
    chapter_id: StableId
    candidates: tuple[FrameCandidateArtifact, ...] = Field(max_length=6)
