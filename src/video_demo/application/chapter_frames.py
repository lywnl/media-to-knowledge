from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from video_demo.application.pipeline_contracts import PreparedMedia
from video_demo.domain.base import FrozenModel, Sha256, StableId, stable_identifier
from video_demo.domain.document import DocumentGenerationConfig
from video_demo.domain.document_plan import (
    ChapterFrameSet,
    ChapterPlan,
    FrameCandidateArtifact,
    VisualSearchTarget,
    frame_candidate_id,
)
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.storage.workspace import reject_symlink_components, safe_runtime_path
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.ffmpeg_frames import (
    ExactFrameSampleResult,
    FrameAdmissionTier,
    FrameCandidate,
    FrameExtractor,
    FrameSample,
)

ChapterFrameStatus = Literal["SUCCEEDED", "DEGRADED", "NO_CANDIDATE", "DISABLED"]
_SEARCH_PADDING_MS = 2_000
_FAILED_SAMPLE_STATUSES = frozenset({"SEEK_FAILED", "DECODE_FAILED", "INVALID_TIMESTAMP"})
_METRIC_NAMES = frozenset(
    {
        "visual_disabled_chapters",
        "visual_no_candidate_chapters",
        "visual_frame_degraded_chapters",
        "visual_collapsed_same_frame_chapters",
        "visual_candidate_budget_degraded_chapters",
    }
)
_LOGGER = logging.getLogger(__name__)


class ChapterFrameSearchBatch(FrozenModel):
    asset_sha256: Sha256
    allowed_run_root: Path
    frame_sets: tuple[ChapterFrameSet, ...] = Field(max_length=240)
    chapter_status: tuple[tuple[StableId, ChapterFrameStatus], ...] = Field(max_length=240)
    warnings: tuple[str, ...] = ()
    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"] = "SUCCEEDED"
    metrics: dict[str, StrictInt]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) - _METRIC_NAMES or any(
            type(item) is not int or item < 0 for item in value.values()
        ):
            raise ValueError("章节抽帧指标非法")
        return value

    @model_validator(mode="after")
    def validate_alignment(self) -> ChapterFrameSearchBatch:
        frame_ids = tuple(item.chapter_id for item in self.frame_sets)
        status_ids = tuple(chapter_id for chapter_id, _status in self.chapter_status)
        if frame_ids != status_ids or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("章节抽帧结果和状态必须一一有序对应")
        degraded = any(status == "DEGRADED" for _, status in self.chapter_status)
        if degraded != (self.status == "PARTIAL_SUCCEEDED"):
            raise ValueError("章节抽帧批次状态与章节降级状态不一致")
        return self


class ChapterFrameSearcher:
    """依据语义锚点和章节中点生成采样点，并委托 FFmpeg 抽帧。"""

    def __init__(
        self,
        runtime_root: Path,
        extractor: FrameExtractor,
        *,
        max_candidate_bytes: int = 512 * 1024 * 1024,
        max_candidate_files: int = 20_000,
        max_candidate_file_bytes: int = 8 * 1024 * 1024,
        candidate_lock_timeout_seconds: float = 300.0,
        **_legacy: object,
    ) -> None:
        if min(max_candidate_bytes, max_candidate_files, max_candidate_file_bytes) < 1:
            raise ValueError("候选帧预算必须为正数")
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._extractor = extractor
        self._max_candidate_bytes = max_candidate_bytes
        self._max_candidate_files = max_candidate_files
        self._max_candidate_file_bytes = max_candidate_file_bytes
        self._candidate_lock_timeout_seconds = candidate_lock_timeout_seconds

    def search(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
        document_config: DocumentGenerationConfig,
        *,
        is_cancel_requested: Callable[[], bool],
    ) -> ChapterFrameSearchBatch:
        allowed_root = self._validate_inputs(media, chapters, transcript_by_id)
        if document_config.max_visuals_per_chapter == 0:
            return self._disabled(chapters, allowed_root, media.proxy_sha256)
        samples, bindings = self._build_samples(chapters, transcript_by_id)
        session = CandidateArtifactSession(
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
                artifact_session=session,
            )
            batch = self._assemble(chapters, results, bindings, allowed_root, media.proxy_sha256)
            session.cleanup_unretained(
                frozenset(frame.sha256 for item in batch.frame_sets for frame in item.candidates)
            )
            return batch
        finally:
            session.close()

    def _validate_inputs(
        self,
        media: PreparedMedia,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
    ) -> Path:
        if (
            not chapters
            or chapters[0].start_ms != 0
            or chapters[-1].end_ms != media.source.duration_ms
            or any(left.end_ms != right.start_ms for left, right in pairwise(chapters))
        ):
            raise VideoDemoError(
                ErrorCode.UNKNOWN_CHAPTER_REFERENCE, "章节计划必须连续覆盖规范时间轴"
            )
        allowed_root = safe_runtime_path(self._runtime_root, media.source.asset.run_relative_root)
        reject_symlink_components(
            self._runtime_root, allowed_root, message="章节候选帧 Run 根不能包含符号链接"
        )
        source = reject_symlink_components(
            self._runtime_root, media.proxy_path, message="视觉输入不能包含符号链接"
        )
        if not source.is_relative_to(allowed_root) or not source.is_file():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "视觉输入必须位于当前 Run 根")
        for key, evidence in transcript_by_id.items():
            if key != evidence.evidence_id:
                raise VideoDemoError(ErrorCode.UNKNOWN_EVIDENCE_REFERENCE, "转写证据映射键值不一致")
        return allowed_root

    def _build_samples(
        self,
        chapters: tuple[ChapterPlan, ...],
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
    ) -> tuple[tuple[FrameSample, ...], dict[str, tuple[str, str]]]:
        samples: list[FrameSample] = []
        bindings: dict[str, tuple[str, str]] = {}
        for chapter in chapters:
            targets = (*chapter.semantic_targets, *chapter.base_coverage_targets)
            points: list[tuple[str, int, FrameAdmissionTier]] = []
            has_base_target = False
            for target in targets:
                timestamp = self._target_timestamp(chapter, target, transcript_by_id)
                tier: FrameAdmissionTier = (
                    "SEMANTIC_PRIMARY" if target.purpose == "SEMANTIC" else "BASE_PRIMARY"
                )
                has_base_target = has_base_target or target.purpose == "BASE_COVERAGE"
                points.append((target.target_id, timestamp, tier))
            midpoint = chapter.start_ms + (chapter.end_ms - chapter.start_ms) // 2
            if not has_base_target:
                points.append(("__chapter_midpoint__", midpoint, "BASE_PRIMARY"))
            seen: set[int] = set()
            for target_id, timestamp, tier in points:
                if timestamp in seen or timestamp >= chapter.end_ms:
                    continue
                seen.add(timestamp)
                sample_id = stable_identifier(
                    "sample",
                    {
                        "chapter_id": chapter.chapter_id,
                        "target_id": target_id,
                        "timestamp_ms": timestamp,
                    },
                )
                samples.append(FrameSample(sample_id, timestamp, tier))
                bindings[sample_id] = (chapter.chapter_id, target_id)
        return tuple(samples), bindings

    @staticmethod
    def _target_timestamp(
        chapter: ChapterPlan,
        target: VisualSearchTarget,
        transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue],
    ) -> int:
        if target.purpose == "BASE_COVERAGE":
            return target.sample_timestamps_ms[0]
        anchors = [transcript_by_id[item] for item in target.anchor_evidence_refs]
        start = max(chapter.start_ms, min(item.start_ms for item in anchors) - _SEARCH_PADDING_MS)
        end = min(chapter.end_ms - 1, max(item.end_ms for item in anchors) + _SEARCH_PADDING_MS)
        return (start + end) // 2

    def _assemble(
        self,
        chapters: tuple[ChapterPlan, ...],
        results: tuple[ExactFrameSampleResult, ...],
        bindings: Mapping[str, tuple[str, str]],
        allowed_root: Path,
        asset_sha256: str,
    ) -> ChapterFrameSearchBatch:
        grouped: dict[str, dict[str, tuple[FrameCandidate, set[str]]]] = {
            chapter.chapter_id: {} for chapter in chapters
        }
        failed: set[str] = set()
        for result in results:
            binding = bindings.get(result.sample_id)
            if binding is None:
                continue
            chapter_id, target_id = binding
            if result.status != "SUCCEEDED":
                if result.status in _FAILED_SAMPLE_STATUSES:
                    failed.add(chapter_id)
                continue
            candidate = result.candidate
            if candidate is None:
                continue
            existing = grouped[chapter_id].get(candidate.sha256)
            if existing is None:
                grouped[chapter_id][candidate.sha256] = (candidate, {target_id})
            else:
                existing[1].add(target_id)
        frame_sets: list[ChapterFrameSet] = []
        statuses: list[tuple[str, ChapterFrameStatus]] = []
        warnings: list[str] = []
        metrics: Counter[str] = Counter()
        for chapter in chapters:
            artifacts = tuple(
                sorted(
                    (
                        FrameCandidateArtifact(
                            frame_id=frame_candidate_id(
                                asset_sha256, item[0].timestamp_ms, item[0].sha256
                            ),
                            timestamp_ms=item[0].timestamp_ms,
                            sha256=item[0].sha256,
                            size_bytes=item[0].size_bytes,
                            relative_path=str(item[0].relative_path),
                            target_ids=tuple(sorted(item[1])),
                        )
                        for item in grouped[chapter.chapter_id].values()
                    ),
                    key=lambda item: (item.timestamp_ms, item.frame_id),
                )
            )[:6]
            # 只要本章有采样点失败，就必须保留成功帧并标记视觉降级；
            # 这样不会把单帧解码故障误报为整章成功，也不会阻断文字链路。
            status: ChapterFrameStatus = (
                "DEGRADED"
                if chapter.chapter_id in failed
                else ("SUCCEEDED" if artifacts else "NO_CANDIDATE")
            )
            if status == "DEGRADED":
                metrics["visual_frame_degraded_chapters"] += 1
                warnings.append(f"CHAPTER_FRAME_DEGRADED:{chapter.chapter_id}")
            elif status == "NO_CANDIDATE":
                metrics["visual_no_candidate_chapters"] += 1
            frame_sets.append(ChapterFrameSet(chapter_id=chapter.chapter_id, candidates=artifacts))
            statuses.append((chapter.chapter_id, status))
        return ChapterFrameSearchBatch(
            asset_sha256=asset_sha256,
            allowed_run_root=allowed_root,
            frame_sets=tuple(frame_sets),
            chapter_status=tuple(statuses),
            warnings=tuple(warnings),
            status="PARTIAL_SUCCEEDED"
            if any(status == "DEGRADED" for _, status in statuses)
            else "SUCCEEDED",
            metrics=dict(metrics),
        )

    @staticmethod
    def _disabled(
        chapters: tuple[ChapterPlan, ...], allowed_root: Path, asset_sha256: str
    ) -> ChapterFrameSearchBatch:
        return ChapterFrameSearchBatch(
            asset_sha256=asset_sha256,
            allowed_run_root=allowed_root,
            frame_sets=tuple(
                ChapterFrameSet(chapter_id=item.chapter_id, candidates=()) for item in chapters
            ),
            chapter_status=tuple((item.chapter_id, "DISABLED") for item in chapters),
            metrics={"visual_disabled_chapters": len(chapters)},
        )
