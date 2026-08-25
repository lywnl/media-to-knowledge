from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from video_demo.application.pipeline_contracts import (
    PreparedMedia,
    SceneIndex,
    scene_index_sha256,
)
from video_demo.domain.base import FrozenModel, StableId, stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_artifact import MAX_METRIC_VALUE
from video_demo.domain.document_plan import (
    ChapterFrameSet,
    ChapterPlan,
    FrameCandidateArtifact,
    VisualSearchTarget,
)
from video_demo.domain.evidence import SceneBoundary, SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components, safe_runtime_path
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.keyframes import (
    ExactFrameSampleResult,
    FrameAdmissionTier,
    FrameCandidate,
    FrameExtractor,
    FrameSample,
)

ChapterFrameStatus = Literal["SUCCEEDED", "DEGRADED", "NO_CANDIDATE", "DISABLED"]

_SEARCH_PADDING_MS = 2_000
_DEFAULT_MAX_CANDIDATE_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_CANDIDATE_FILES = 20_000
_DEFAULT_MAX_CANDIDATE_FILE_BYTES = 5 * 1024 * 1024
_DEFAULT_CANDIDATE_LOCK_TIMEOUT_SECONDS = 300.0
_METRIC_NAMES = frozenset(
    {
        "visual_disabled_chapters",
        "visual_no_candidate_chapters",
        "visual_frame_degraded_chapters",
        "visual_collapsed_same_frame_chapters",
        "visual_candidate_budget_degraded_chapters",
    },
)
_FAILED_SAMPLE_STATUSES = frozenset(
    {"SEEK_FAILED", "DECODE_FAILED", "INVALID_TIMESTAMP", "OUT_OF_TOLERANCE"},
)
_NO_ARTIFACT_SAMPLE_STATUSES = _FAILED_SAMPLE_STATUSES | {"QUALITY_REJECTED"}


class ChapterFrameSearchBatch(FrozenModel):
    allowed_run_root: Path
    frame_tolerance_ms: int = Field(ge=0, le=100)
    frame_sets: tuple[ChapterFrameSet, ...] = Field(max_length=240)
    chapter_status: tuple[tuple[StableId, ChapterFrameStatus], ...] = Field(max_length=240)
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) - _METRIC_NAMES:
            raise ValueError("章节抽帧指标包含未知白名单键")
        if any(
            type(metric) is not int or not 0 <= metric <= MAX_METRIC_VALUE
            for metric in value.values()
        ):
            raise ValueError("章节抽帧指标必须是非负严格整数")
        return value

    @model_validator(mode="after")
    def validate_alignment(self) -> ChapterFrameSearchBatch:
        frame_ids = tuple(item.chapter_id for item in self.frame_sets)
        status_ids = tuple(chapter_id for chapter_id, _status in self.chapter_status)
        if frame_ids != status_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("章节抽帧结果和状态必须一一有序对应")
        degraded = any(chapter_status == "DEGRADED" for _, chapter_status in self.chapter_status)
        if degraded != (self.status == "PARTIAL_SUCCEEDED"):
            raise ValueError("章节抽帧批次状态与章节降级状态不一致")
        return self


@dataclass(frozen=True, slots=True)
class _SampleBinding:
    chapter_id: str
    target_id: str
    window_start_ms: int
    window_end_ms: int


@dataclass(frozen=True, slots=True)
class _TargetSamplingPlan:
    target_id: str
    purpose: Literal["SEMANTIC", "BASE_COVERAGE"]
    proposed_timestamps_ms: tuple[int, ...]
    window_start_ms: int
    window_end_ms: int


@dataclass(frozen=True, slots=True)
class _SelectedSamplePoint:
    target_id: str
    timestamp_ms: int
    admission_tier: FrameAdmissionTier


@dataclass(frozen=True, slots=True)
class _CandidateTargetBinding:
    target_id: str
    window_start_ms: int
    window_end_ms: int


@dataclass(frozen=True, slots=True)
class _InternalCandidate:
    chapter_id: str
    frame: FrameCandidate
    sha256: str
    size_bytes: int
    run_relative_path: str
    target_bindings: tuple[_CandidateTargetBinding, ...]
    scene_id: str
    created_by_call: bool

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(binding.target_id for binding in self.target_bindings))


@dataclass(frozen=True, slots=True)
class _ChapterSelection:
    chapter: ChapterPlan
    candidates: tuple[_InternalCandidate, ...]
    sample_results: tuple[ExactFrameSampleResult, ...]


class ChapterFrameSearcher:
    """把章节视觉目标转换为一次全局精确采样，再按章节回绑、过滤和去重。"""

    def __init__(
        self,
        runtime_root: Path,
        extractor: FrameExtractor,
        *,
        max_candidate_bytes: int = _DEFAULT_MAX_CANDIDATE_BYTES,
        max_candidate_files: int = _DEFAULT_MAX_CANDIDATE_FILES,
        max_candidate_file_bytes: int = _DEFAULT_MAX_CANDIDATE_FILE_BYTES,
        candidate_lock_timeout_seconds: float = _DEFAULT_CANDIDATE_LOCK_TIMEOUT_SECONDS,
        maximum_black_ratio: float = 0.95,
        max_hash_distance_for_duplicate: int = 4,
    ) -> None:
        if (
            max_candidate_bytes < 1
            or max_candidate_files < 1
            or max_candidate_file_bytes < 1
            or not math.isfinite(candidate_lock_timeout_seconds)
            or candidate_lock_timeout_seconds <= 0
            or max_hash_distance_for_duplicate < 0
        ):
            raise ValueError("候选帧预算必须为正数且感知哈希距离不得为负数")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._extractor = extractor
        self._max_candidate_bytes = max_candidate_bytes
        self._max_candidate_files = max_candidate_files
        self._max_candidate_file_bytes = max_candidate_file_bytes
        self._candidate_lock_timeout_seconds = candidate_lock_timeout_seconds
        self._maximum_black_ratio = maximum_black_ratio
        self._max_hash_distance_for_duplicate = max_hash_distance_for_duplicate

    def search(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        scene_index: SceneIndex,
        document_config: DocumentGenerationConfig,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterFrameSearchBatch:
        allowed_run_root = self._validate_inputs(media, chapters, transcript_by_id, scene_index)
        if document_config.max_visuals_per_chapter == 0:
            return self._disabled_batch(chapters, allowed_run_root, scene_index.frame_tolerance_ms)
        samples, bindings = self._build_samples(chapters, transcript_by_id, scene_index)
        artifact_session = CandidateArtifactSession(
            runtime_root=self._runtime_root,
            max_unique_bytes=self._max_candidate_bytes,
            max_files=self._max_candidate_files,
            max_file_bytes=self._max_candidate_file_bytes,
            is_cancel_requested=is_cancel_requested,
            lock_timeout_seconds=self._candidate_lock_timeout_seconds,
        )
        try:
            results = self._extractor.extract_samples(
                media.proxy_path,
                media.source.asset.run_relative_root,
                samples,
                is_cancel_requested=is_cancel_requested,
                frame_tolerance_ms=scene_index.frame_tolerance_ms,
                artifact_session=artifact_session,
            )
            self._validate_results(samples, results, scene_index.frame_tolerance_ms)
            candidates = self._collect_candidates(
                results,
                bindings,
                media,
                scene_index.scenes,
                allowed_run_root,
            )
            batch = self._assemble_batch(
                chapters,
                results,
                bindings,
                candidates,
                allowed_run_root,
                scene_index.frame_tolerance_ms,
                media.source.asset.source_sha256,
            )
            artifact_session.cleanup_unretained(
                frozenset(
                    candidate.sha256
                    for frame_set in batch.frame_sets
                    for candidate in frame_set.candidates
                ),
            )
            return batch
        except Exception:
            artifact_session.cleanup_unretained(frozenset())
            raise
        finally:
            artifact_session.close()

    def _validate_inputs(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        scene_index: SceneIndex,
    ) -> Path:
        if scene_index.proxy_sha256 != media.proxy_sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "场景索引与代理视频摘要不一致")
        if scene_index.duration_ms != media.source.duration_ms:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景索引与规范媒体时长不一致")
        if scene_index.index_sha256 != scene_index_sha256(
            proxy_sha256=scene_index.proxy_sha256,
            duration_ms=scene_index.duration_ms,
            frame_tolerance_ms=scene_index.frame_tolerance_ms,
            scenes=scene_index.scenes,
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景索引摘要与内容不一致")
        scene_ids = tuple(scene.evidence_id for scene in scene_index.scenes)
        ordered_scenes = tuple(
            sorted(
                scene_index.scenes,
                key=lambda scene: (scene.start_ms, scene.end_ms, scene.evidence_id),
            ),
        )
        if (
            not scene_index.scenes
            or len(scene_ids) != len(set(scene_ids))
            or scene_index.scenes != ordered_scenes
            or scene_index.scenes[0].start_ms != 0
            or scene_index.scenes[-1].end_ms != scene_index.duration_ms
            or any(
                left.end_ms != right.start_ms
                for left, right in pairwise(scene_index.scenes)
            )
        ):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "场景索引未连续覆盖视频")
        if not chapters:
            raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划不能为空")
        if (
            chapters[0].start_ms != 0
            or chapters[-1].end_ms != media.source.duration_ms
            or any(left.end_ms != right.start_ms for left, right in pairwise(chapters))
        ):
            raise VideoDemoError(
                ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划必须连续覆盖规范时间轴"
            )
        chapter_ids = tuple(chapter.chapter_id for chapter in chapters)
        if len(chapter_ids) != len(set(chapter_ids)):
            raise VideoDemoError(ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节标识不得重复")
        for evidence_id, evidence in transcript_by_id.items():
            if evidence_id != evidence.evidence_id:
                raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "转写证据映射键值不一致")
        allowed_run_root = safe_runtime_path(
            self._runtime_root,
            media.source.asset.run_relative_root,
        )
        reject_symlink_components(
            self._runtime_root,
            allowed_run_root,
            message="章节候选帧 Run 根不能包含符号链接",
        )
        proxy = reject_symlink_components(
            self._runtime_root,
            media.proxy_path,
            message="代理视频必须位于当前 Run 根",
        )
        if not proxy.is_relative_to(allowed_run_root) or not proxy.is_file():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "代理视频必须位于当前 Run 根")
        return allowed_run_root

    def _build_samples(
        self,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        scene_index: SceneIndex,
    ) -> tuple[tuple[FrameSample, ...], dict[str, _SampleBinding]]:
        scenes_by_id = {scene.evidence_id: scene for scene in scene_index.scenes}
        samples: list[FrameSample] = []
        bindings: dict[str, _SampleBinding] = {}
        for chapter in chapters:
            targets = (*chapter.semantic_targets, *chapter.base_coverage_targets)
            target_ids = tuple(target.target_id for target in targets)
            if len(target_ids) != len(set(target_ids)):
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "章节视觉目标标识不得重复")
            target_plans: list[_TargetSamplingPlan] = []
            for target in targets:
                timestamps, window = self._target_timestamps(
                    chapter,
                    target,
                    transcript_by_id,
                    scenes_by_id,
                    scene_index.scenes,
                )
                target_plans.append(
                    _TargetSamplingPlan(
                        target_id=target.target_id,
                        purpose=target.purpose,
                        proposed_timestamps_ms=timestamps,
                        window_start_ms=window[0],
                        window_end_ms=window[1],
                    ),
                )
            selected_points = _select_chapter_sample_points(
                chapter,
                target_plans,
                scene_index.scenes,
            )
            plans_by_id = {plan.target_id: plan for plan in target_plans}
            for point in selected_points:
                plan = plans_by_id[point.target_id]
                sample_id = stable_identifier(
                    "sample",
                    {
                        "chapter_id": chapter.chapter_id,
                        "target_id": point.target_id,
                        "timestamp_ms": point.timestamp_ms,
                    },
                )
                samples.append(
                    FrameSample(
                        sample_id=sample_id,
                        timestamp_ms=point.timestamp_ms,
                        admission_tier=point.admission_tier,
                    ),
                )
                bindings[sample_id] = _SampleBinding(
                    chapter_id=chapter.chapter_id,
                    target_id=point.target_id,
                    window_start_ms=plan.window_start_ms,
                    window_end_ms=plan.window_end_ms,
                )
            unique_timestamps = {point.timestamp_ms for point in selected_points}
            if len(unique_timestamps) > 6:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "章节采样点超过硬上限")
        return tuple(samples), bindings

    def _target_timestamps(
        self,
        chapter: ChapterPlan,
        target: VisualSearchTarget,
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        scenes_by_id: Mapping[str, SceneBoundary],
        scenes: tuple[SceneBoundary, ...],
    ) -> tuple[tuple[int, ...], tuple[int, int]]:
        if target.purpose == "SEMANTIC":
            anchors = self._semantic_anchors(chapter, target, transcript_by_id)
            anchor_start = min(anchor.start_ms for anchor in anchors)
            anchor_end = max(anchor.end_ms for anchor in anchors)
            search_start = max(chapter.start_ms, anchor_start - _SEARCH_PADDING_MS)
            search_end = min(chapter.end_ms, anchor_end + _SEARCH_PADDING_MS)
            if search_end <= search_start:
                raise VideoDemoError(ErrorCode.EVIDENCE_OUTSIDE_CHAPTER, "语义视觉搜索窗口为空")
            return (
                _semantic_sample_timestamps(search_start, search_end, anchor_start, anchor_end),
                (search_start, search_end),
            )
        ranges: list[tuple[int, int]] = []
        for scene_id in target.scene_refs:
            scene = scenes_by_id.get(scene_id)
            if scene is None:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "基础覆盖引用未知场景")
            start_ms = max(scene.start_ms, chapter.start_ms)
            end_ms = min(scene.end_ms, chapter.end_ms)
            if end_ms <= start_ms:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "基础覆盖场景不属于目标章节")
            ranges.append((start_ms, end_ms))
        timestamps = list(_base_coverage_timestamps(ranges))
        timestamps.extend(target.sample_timestamps_ms)
        referenced_scene_ids = set(target.scene_refs)
        timestamps.extend(
            scene.start_ms + (scene.end_ms - scene.start_ms) // 2
            for scene in scenes
            if (
                scene.evidence_id not in referenced_scene_ids
                and scene.start_ms < chapter.end_ms
                and chapter.start_ms < scene.end_ms
            )
        )
        timestamps.extend(_chapter_fallback_timestamps(chapter.start_ms, chapter.end_ms))
        if not timestamps:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "基础覆盖目标缺少采样位置")
        return (
            tuple(dict.fromkeys(min(chapter.end_ms - 1, timestamp) for timestamp in timestamps)),
            (chapter.start_ms, chapter.end_ms),
        )

    @staticmethod
    def _semantic_anchors(
        chapter: ChapterPlan,
        target: VisualSearchTarget,
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
    ) -> tuple[SpeechSegment | SubtitleCue, ...]:
        anchors: list[SpeechSegment | SubtitleCue] = []
        for evidence_id in target.anchor_evidence_refs:
            anchor = transcript_by_id.get(evidence_id)
            if anchor is None:
                raise VideoDemoError(
                    ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "视觉目标引用未知转写锚点"
                )
            if anchor.start_ms < chapter.start_ms or anchor.end_ms > chapter.end_ms:
                raise VideoDemoError(
                    ErrorCode.EVIDENCE_OUTSIDE_CHAPTER, "视觉目标锚点不属于目标章节"
                )
            anchors.append(anchor)
        if tuple(anchors) != tuple(
            sorted(anchors, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
        ):
            raise VideoDemoError(ErrorCode.EVIDENCE_RELATION_INVALID, "视觉目标锚点必须按时间排序")
        if anchors[-1].end_ms - anchors[0].start_ms > 30_000:
            raise VideoDemoError(ErrorCode.EVIDENCE_RELATION_INVALID, "视觉目标锚点跨度超过 30 秒")
        return tuple(anchors)

    @staticmethod
    def _validate_results(
        samples: tuple[FrameSample, ...],
        results: tuple[ExactFrameSampleResult, ...],
        frame_tolerance_ms: int,
    ) -> None:
        expected = Counter(sample.sample_id for sample in samples)
        actual = Counter(result.sample_id for result in results)
        if expected != actual:
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "精确抽帧结果未完整回绑采样计划")
        requested = {sample.sample_id: sample.timestamp_ms for sample in samples}
        if any(requested[result.sample_id] != result.requested_timestamp_ms for result in results):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "精确抽帧返回的请求时间不一致")
        for result in results:
            if not _has_valid_result_state(result):
                raise VideoDemoError(
                    ErrorCode.VISUAL_RESULT_INVALID,
                    "精确抽帧状态、候选和制品状态组合非法",
                )
            if result.candidate is None:
                continue
            candidate = result.candidate
            if (
                abs(candidate.timestamp_ms - result.requested_timestamp_ms) > frame_tolerance_ms
                or not math.isfinite(candidate.sharpness)
                or candidate.sharpness < 0
                or not math.isfinite(candidate.black_ratio)
                or not 0.0 <= candidate.black_ratio <= 1.0
                or re.fullmatch(r"[0-9a-f]{16}", candidate.perceptual_hash) is None
            ):
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "精确抽帧候选指标非法")

    def _collect_candidates(
        self,
        results: tuple[ExactFrameSampleResult, ...],
        bindings: Mapping[str, _SampleBinding],
        media: PreparedMedia,
        scenes: tuple[SceneBoundary, ...],
        allowed_run_root: Path,
    ) -> tuple[_InternalCandidate, ...]:
        candidates: list[_InternalCandidate] = []
        content_cache: dict[Path, bytes] = {}
        for result in results:
            if result.status != "SUCCEEDED" or result.artifact_status != "PUBLISHED":
                continue
            if result.candidate is None:
                raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "已发布采样缺少候选帧")
            candidate = result.candidate
            binding = bindings[result.sample_id]
            if not binding.window_start_ms <= candidate.timestamp_ms < binding.window_end_ms:
                continue
            relative_path = self._candidate_path_in_run(candidate, allowed_run_root)
            payload = content_cache.get(relative_path)
            if payload is None:
                payload = self._read_candidate(relative_path, allowed_run_root)
                content_cache[relative_path] = payload
            digest = hashlib.sha256(payload).hexdigest()
            candidates.append(
                _InternalCandidate(
                    chapter_id=binding.chapter_id,
                    frame=candidate,
                    sha256=digest,
                    size_bytes=len(payload),
                    run_relative_path=relative_path.as_posix(),
                    target_bindings=(
                        _CandidateTargetBinding(
                            target_id=binding.target_id,
                            window_start_ms=binding.window_start_ms,
                            window_end_ms=binding.window_end_ms,
                        ),
                    ),
                    scene_id=_scene_at(scenes, candidate.timestamp_ms).evidence_id,
                    created_by_call=candidate.created_by_call,
                ),
            )
        return tuple(candidates)

    def _candidate_path_in_run(
        self,
        candidate: FrameCandidate,
        allowed_run_root: Path,
    ) -> Path:
        run_relative_root = allowed_run_root.relative_to(self._runtime_root)
        try:
            path_in_run = candidate.relative_path.relative_to(run_relative_root)
        except ValueError:
            path_in_run = candidate.relative_path
        if path_in_run.is_absolute() or path_in_run.parent != Path("visual/candidates"):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧路径不在内容寻址目录")
        return path_in_run

    def _read_candidate(self, path_in_run: Path, allowed_run_root: Path) -> bytes:
        absolute_path = reject_symlink_components(
            self._runtime_root,
            allowed_run_root / path_in_run,
            message="候选帧路径不能包含符号链接",
        )
        payload = _read_regular_file(absolute_path, self._max_candidate_file_bytes)
        digest = hashlib.sha256(payload).hexdigest()
        if path_in_run.name != f"{digest}.jpg" or not _is_jpeg(payload):
            raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧内容地址或 JPEG 格式非法")
        return payload

    def _assemble_batch(
        self,
        chapters: tuple[ChapterPlan, ...],
        results: tuple[ExactFrameSampleResult, ...],
        bindings: Mapping[str, _SampleBinding],
        candidates: tuple[_InternalCandidate, ...],
        allowed_run_root: Path,
        frame_tolerance_ms: int,
        asset_sha256: str,
    ) -> ChapterFrameSearchBatch:
        selections: list[_ChapterSelection] = []
        budget_degraded: set[str] = set()
        for chapter in chapters:
            chapter_candidates = tuple(
                item for item in candidates if item.chapter_id == chapter.chapter_id
            )
            selected = self._select_chapter_candidates(
                chapter,
                chapter_candidates,
                asset_sha256,
            )
            chapter_results = tuple(
                result
                for result in results
                if bindings[result.sample_id].chapter_id == chapter.chapter_id
            )
            required_semantic_ids = {target.target_id for target in chapter.semantic_targets}
            retained_target_ids = {
                target_id for candidate in selected for target_id in candidate.target_ids
            }
            rejected_target_ids = {
                bindings[result.sample_id].target_id
                for result in chapter_results
                if result.artifact_status == "BUDGET_REJECTED"
            }
            if rejected_target_ids - retained_target_ids:
                budget_degraded.add(chapter.chapter_id)
                selected = ()
            if (
                chapter.visual_mode in {"COMPARISON", "MULTI_STEP"}
                and not required_semantic_ids.issubset(retained_target_ids)
            ):
                selected = ()
            selections.append(_ChapterSelection(chapter, selected, chapter_results))
        metrics: Counter[str] = Counter()
        warnings: list[str] = []
        frame_sets: list[ChapterFrameSet] = []
        statuses: list[tuple[str, ChapterFrameStatus]] = []
        for selection in selections:
            chapter = selection.chapter
            selected = selection.candidates
            semantic_target_ids = {target.target_id for target in chapter.semantic_targets}
            collapsed_same_frame = (
                chapter.visual_mode in {"COMPARISON", "MULTI_STEP"}
                and len(selected) == 1
                and semantic_target_ids.issubset(selected[0].target_ids)
            )
            if collapsed_same_frame:
                metrics["visual_collapsed_same_frame_chapters"] += 1
            budget_was_degraded = chapter.chapter_id in budget_degraded
            status = self._chapter_status(selected, selection.sample_results)
            if budget_was_degraded:
                status = "DEGRADED"
                metrics["visual_candidate_budget_degraded_chapters"] += 1
                warnings.append(f"VISUAL_CANDIDATE_BUDGET_DEGRADED:{chapter.chapter_id}")
            if status == "NO_CANDIDATE":
                metrics["visual_no_candidate_chapters"] += 1
            elif status == "DEGRADED" and not budget_was_degraded:
                metrics["visual_frame_degraded_chapters"] += 1
                warnings.append(f"CHAPTER_FRAME_DEGRADED:{chapter.chapter_id}")
            frame_sets.append(
                ChapterFrameSet(
                    chapter_id=chapter.chapter_id,
                    candidates=tuple(
                        self._to_artifact(asset_sha256=asset_sha256, item=item) for item in selected
                    ),
                ),
            )
            statuses.append((chapter.chapter_id, status))
        batch_status = (
            "PARTIAL_SUCCEEDED"
            if any(status == "DEGRADED" for _, status in statuses)
            else "SUCCEEDED"
        )
        return ChapterFrameSearchBatch(
            allowed_run_root=allowed_run_root,
            frame_tolerance_ms=frame_tolerance_ms,
            frame_sets=tuple(frame_sets),
            chapter_status=tuple(statuses),
            warnings=tuple(warnings),
            status=batch_status,
            metrics=dict(metrics),
        )

    def _select_chapter_candidates(
        self,
        chapter: ChapterPlan,
        candidates: tuple[_InternalCandidate, ...],
        asset_sha256: str,
    ) -> tuple[_InternalCandidate, ...]:
        eligible = tuple(
            item for item in candidates if item.frame.black_ratio <= self._maximum_black_ratio
        )
        merged = _merge_same_sha(eligible, _target_order(chapter))
        unique = _deduplicate_phash(
            merged,
            self._max_hash_distance_for_duplicate,
            _target_order(chapter),
        )
        limit = 6 if chapter.visual_mode in {"COMPARISON", "MULTI_STEP"} else 4
        window_best = _best_candidate_per_window(unique)
        ranked = sorted(
            window_best,
            key=lambda item: _candidate_rank(item, chapter, asset_sha256),
        )
        protected = _protected_candidates(ranked, _target_order(chapter))
        selected = list(protected)
        selected.extend(item for item in ranked if item not in selected and len(selected) < limit)
        selected = selected[:limit]
        protected_set = set(protected)
        selected.sort(
            key=lambda item: (
                0 if item in protected_set else 1,
                *_candidate_rank(item, chapter, asset_sha256),
            ),
        )
        return tuple(selected)

    @staticmethod
    def _chapter_status(
        selected: tuple[_InternalCandidate, ...],
        results: tuple[ExactFrameSampleResult, ...],
    ) -> ChapterFrameStatus:
        if selected:
            return "SUCCEEDED"
        if results and all(
            result.status in {"SUCCEEDED", "QUALITY_REJECTED"} for result in results
        ):
            return "NO_CANDIDATE"
        if any(result.status in _FAILED_SAMPLE_STATUSES for result in results):
            return "DEGRADED"
        return "NO_CANDIDATE"

    @staticmethod
    def _to_artifact(asset_sha256: str, item: _InternalCandidate) -> FrameCandidateArtifact:
        return FrameCandidateArtifact(
            frame_id=_frame_identifier(asset_sha256, item),
            timestamp_ms=item.frame.timestamp_ms,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
            relative_path=item.run_relative_path,
            perceptual_hash=item.frame.perceptual_hash,
            target_ids=item.target_ids,
        )

    @staticmethod
    def _disabled_batch(
        chapters: tuple[ChapterPlan, ...],
        allowed_run_root: Path,
        frame_tolerance_ms: int,
    ) -> ChapterFrameSearchBatch:
        return ChapterFrameSearchBatch(
            allowed_run_root=allowed_run_root,
            frame_tolerance_ms=frame_tolerance_ms,
            frame_sets=tuple(
                ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=())
                for chapter in chapters
            ),
            chapter_status=tuple((chapter.chapter_id, "DISABLED") for chapter in chapters),
            metrics={"visual_disabled_chapters": len(chapters)},
        )


def _semantic_sample_timestamps(
    search_start: int,
    search_end: int,
    anchor_start: int,
    anchor_end: int,
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            (
                max(search_start, anchor_start - 1_000),
                (anchor_start + anchor_end) // 2,
                min(search_end - 1, anchor_end + 1_000),
            ),
        ),
    )


def _base_coverage_timestamps(ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    timestamps: list[int] = []
    for start_ms, end_ms in ranges:
        duration_ms = end_ms - start_ms
        timestamps.extend(
            (
                start_ms + duration_ms // 4,
                start_ms + duration_ms // 2,
                min(end_ms - 1, start_ms + (duration_ms * 3) // 4),
            ),
        )
    return tuple(dict.fromkeys(timestamps))


def _chapter_fallback_timestamps(start_ms: int, end_ms: int) -> tuple[int, ...]:
    duration_ms = end_ms - start_ms
    return tuple(
        dict.fromkeys(
            (
                start_ms + duration_ms // 2,
                start_ms + duration_ms // 4,
                min(end_ms - 1, start_ms + (duration_ms * 3) // 4),
            ),
        ),
    )


def _select_chapter_sample_points(
    chapter: ChapterPlan,
    target_plans: Sequence[_TargetSamplingPlan],
    scenes: Sequence[SceneBoundary],
) -> tuple[_SelectedSamplePoint, ...]:
    scene_count = sum(
        1 for scene in scenes if scene.start_ms < chapter.end_ms and chapter.start_ms < scene.end_ms
    )
    base_budget = 3 if scene_count <= 1 else 4 if scene_count <= 3 else 6
    primary: list[_SelectedSamplePoint] = []
    for plan in target_plans:
        target_midpoint = plan.window_start_ms + (plan.window_end_ms - plan.window_start_ms) // 2
        nearest = min(
            plan.proposed_timestamps_ms,
            key=lambda timestamp_ms: (abs(timestamp_ms - target_midpoint), timestamp_ms),
        )
        primary.append(
            _SelectedSamplePoint(
                target_id=plan.target_id,
                timestamp_ms=nearest,
                admission_tier=(
                    "SEMANTIC_PRIMARY" if plan.purpose == "SEMANTIC" else "BASE_PRIMARY"
                ),
            ),
        )
    unique_primary_timestamps = {point.timestamp_ms for point in primary}
    budget = min(6, max(base_budget, len(unique_primary_timestamps)))
    if len(unique_primary_timestamps) > 6:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "章节采样预算无法覆盖全部视觉目标")
    selected = list(primary)
    selected_timestamps = set(unique_primary_timestamps)
    supplement = sorted(
        (
            _SelectedSamplePoint(
                target_id=plan.target_id,
                timestamp_ms=timestamp_ms,
                admission_tier=(
                    "SEMANTIC_SUPPLEMENT"
                    if plan.purpose == "SEMANTIC"
                    else "BASE_SUPPLEMENT"
                ),
            )
            for plan in target_plans
            for timestamp_ms in plan.proposed_timestamps_ms
            if timestamp_ms not in selected_timestamps
        ),
        key=lambda point: (
            0 if point.admission_tier == "SEMANTIC_SUPPLEMENT" else 1,
            point.timestamp_ms,
            point.target_id,
        ),
    )
    for point in supplement:
        if len(selected_timestamps) >= budget:
            break
        selected.append(point)
        selected_timestamps.add(point.timestamp_ms)
    covered_targets = {point.target_id for point in selected}
    expected_targets = {plan.target_id for plan in target_plans}
    if covered_targets != expected_targets:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "章节采样计划未覆盖全部视觉目标")
    return tuple(selected)


def _scene_at(scenes: tuple[SceneBoundary, ...], timestamp_ms: int) -> SceneBoundary:
    for scene in scenes:
        if scene.start_ms <= timestamp_ms < scene.end_ms:
            return scene
    raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧真实时间不在场景索引内")


def _target_order(chapter: ChapterPlan) -> tuple[str, ...]:
    return tuple(
        target.target_id for target in (*chapter.semantic_targets, *chapter.base_coverage_targets)
    )


def _merge_same_sha(
    candidates: Sequence[_InternalCandidate],
    target_order: tuple[str, ...],
) -> tuple[_InternalCandidate, ...]:
    grouped: dict[str, list[_InternalCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.sha256, []).append(candidate)
    merged: list[_InternalCandidate] = []
    for group in grouped.values():
        representative = min(
            group, key=lambda item: (item.frame.timestamp_ms, item.run_relative_path)
        )
        merged.append(
            _InternalCandidate(
                chapter_id=representative.chapter_id,
                frame=representative.frame,
                sha256=representative.sha256,
                size_bytes=representative.size_bytes,
                run_relative_path=representative.run_relative_path,
                target_bindings=tuple(
                    dict.fromkeys(
                        binding
                        for target_id in target_order
                        for item in group
                        for binding in item.target_bindings
                        if binding.target_id == target_id
                    ),
                ),
                scene_id=representative.scene_id,
                created_by_call=any(item.created_by_call for item in group),
            ),
        )
    return tuple(merged)


def _deduplicate_phash(
    candidates: Sequence[_InternalCandidate],
    maximum_distance: int,
    target_order: tuple[str, ...],
) -> tuple[_InternalCandidate, ...]:
    protected = {
        min(
            (item for item in candidates if target_id in item.target_ids),
            key=lambda item: (-item.frame.sharpness, item.frame.timestamp_ms, item.sha256),
        )
        for target_id in target_order
        if any(target_id in item.target_ids for item in candidates)
    }
    ranked = sorted(
        candidates, key=lambda item: (-item.frame.sharpness, item.frame.timestamp_ms, item.sha256)
    )
    kept: list[_InternalCandidate] = []
    for candidate in ranked:
        duplicate = any(
            candidate.scene_id == existing.scene_id
            and _hamming_distance(
                candidate.frame.perceptual_hash,
                existing.frame.perceptual_hash,
            )
            <= maximum_distance
            for existing in kept
        )
        if duplicate and candidate not in protected:
            continue
        kept.append(candidate)
    kept.sort(
        key=lambda item: tuple(target_order.index(target_id) for target_id in item.target_ids)
    )
    return tuple(kept)


def _protected_candidates(
    ranked: Sequence[_InternalCandidate],
    target_order: tuple[str, ...],
) -> tuple[_InternalCandidate, ...]:
    selected: list[_InternalCandidate] = []
    for target_id in target_order:
        candidate = next((item for item in ranked if target_id in item.target_ids), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def _best_candidate_per_window(
    candidates: Sequence[_InternalCandidate],
) -> tuple[_InternalCandidate, ...]:
    grouped: dict[tuple[str, int, int], list[_InternalCandidate]] = {}
    for candidate in candidates:
        for binding in candidate.target_bindings:
            grouped.setdefault(
                (binding.target_id, binding.window_start_ms, binding.window_end_ms),
                [],
            ).append(candidate)
    selected: list[_InternalCandidate] = []
    for group in grouped.values():
        best = min(
            group,
            key=lambda item: (
                -item.frame.sharpness,
                item.frame.black_ratio,
                item.frame.timestamp_ms,
                item.sha256,
            ),
        )
        if best not in selected:
            selected.append(best)
    return tuple(selected)


def _candidate_rank(
    item: _InternalCandidate,
    chapter: ChapterPlan,
    asset_sha256: str,
) -> tuple[int, float, int, int, str]:
    semantic_ids = {target.target_id for target in chapter.semantic_targets}
    base_ids = {target.target_id for target in chapter.base_coverage_targets}
    semantic_hits = len(semantic_ids.intersection(item.target_ids))
    base_hits = len(base_ids.intersection(item.target_ids))
    return (
        -semantic_hits,
        -item.frame.sharpness,
        -base_hits,
        item.frame.timestamp_ms,
        _frame_identifier(asset_sha256, item),
    )


def _frame_identifier(asset_sha256: str, item: _InternalCandidate) -> str:
    return stable_identifier(
        "keyframe",
        {
            "asset_sha256": asset_sha256,
            "timestamp_ms": item.frame.timestamp_ms,
            "sha256": item.sha256,
        },
    )


def _has_valid_result_state(result: ExactFrameSampleResult) -> bool:
    if result.status == "SUCCEEDED":
        return (result.candidate is not None) == (result.artifact_status == "PUBLISHED") and (
            result.candidate is None
        ) == (result.artifact_status == "BUDGET_REJECTED")
    return (
        result.status in _NO_ARTIFACT_SAMPLE_STATUSES
        and result.candidate is None
        and result.artifact_status is None
    )


def _hamming_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧感知哈希非法") from error


def _read_regular_file(path: Path, max_bytes: int) -> bytes:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise VideoDemoError(ErrorCode.INVALID_CONFIGURATION, "当前平台不支持安全候选帧读取")
    descriptor = -1
    try:
        before_path = os.lstat(path)
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
            or _file_identity(before_path) != _file_identity(before)
        ):
            raise OSError
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total)):
            total += len(chunk)
            if total > max_bytes:
                raise OSError
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(after_path)
        ):
            raise OSError
        return b"".join(chunks)
    except OSError:
        raise VideoDemoError(ErrorCode.VISUAL_RESULT_INVALID, "候选帧安全读取失败") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_jpeg(payload: bytes) -> bool:
    return payload.startswith(b"\xff\xd8\xff") and payload.endswith(b"\xff\xd9")
