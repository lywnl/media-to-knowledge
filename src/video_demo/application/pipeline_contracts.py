from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, StrictInt, model_validator

from video_demo.domain.base import FrozenModel, LanguageCode, Sha256, StableId
from video_demo.domain.document import DocumentGenerationConfig, TranscriptSource
from video_demo.domain.document_artifact import (
    MAX_METRIC_VALUE,
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
)
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SceneBoundary,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.domain.manifest import VideoAssetManifest
from video_demo.domain.speech_config import normalize_core_context, normalize_hotwords
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits, SupportedMime

if TYPE_CHECKING:
    from video_demo.application.chapter_frames import ChapterFrameSearchBatch
    from video_demo.application.chapter_vision import ChapterVisionBatch
    from video_demo.application.document_rendering import RenderedDocument
    from video_demo.domain.document import VideoUnderstandingResult
    from video_demo.media.subtitles import ParsedSubtitle
    from video_demo.persistence.repositories import Scope


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """4.1 生产流水线一次运行所需的全部显式输入。"""

    run_id: str
    scope: Scope
    title_hint: str
    document_config: DocumentGenerationConfig
    is_cancel_requested: Callable[[], bool] = lambda: False
    on_stage_start: Callable[[str], None] = lambda _stage: None

    def __post_init__(self) -> None:
        if not 1 <= len(self.title_hint) <= 200 or self.title_hint != self.title_hint.strip():
            raise ValueError("title_hint 必须是已规范化的非空标题")


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """尚未发布的 4.1 结果及发布后清理所需的内部闭包。"""

    status: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]
    result: VideoUnderstandingResult
    evidence: tuple[DocumentEvidenceItem, ...]
    document: RenderedDocument
    warnings: tuple[str, ...]
    stage_metrics: Mapping[str, int]
    model_metrics: Mapping[str, StrictInt]
    stage_cache_hits: tuple[str, ...]
    transcript_source: TranscriptSource
    frame_batch: ChapterFrameSearchBatch
    visual_batch: ChapterVisionBatch

    def __post_init__(self) -> None:
        from video_demo.application.document_rendering import render_markdown

        expected_document = render_markdown(self.result, self.evidence)
        if self.document != expected_document:
            raise ValueError("document 必须由同一 result/evidence 确定性渲染生成")
        _validate_complete_metric_map(
            self.stage_metrics,
            RESULT_STAGE_NAMES,
            "阶段",
        )
        _validate_complete_metric_map(
            self.model_metrics,
            MODEL_METRIC_NAMES,
            "模型",
        )
        object.__setattr__(self, "stage_metrics", MappingProxyType(dict(self.stage_metrics)))
        object.__setattr__(self, "model_metrics", MappingProxyType(dict(self.model_metrics)))


def _validate_complete_metric_map(
    metrics: Mapping[str, int],
    expected_names: frozenset[str],
    display_name: str,
) -> None:
    if set(metrics) != expected_names:
        raise ValueError(f"{display_name}指标必须恰好覆盖白名单")
    if any(
        type(value) is not int or not 0 <= value <= MAX_METRIC_VALUE
        for value in metrics.values()
    ):
        raise ValueError(f"{display_name}指标必须是非负严格整数")


class PipelineRunConfig(FrozenModel):
    language_hints: tuple[LanguageCode, ...] = ()
    hotwords: tuple[str, ...] = ()
    core_context: str | None = None
    document_config: DocumentGenerationConfig = Field(
        default_factory=DocumentGenerationConfig,
    )
    result_schema_version: Literal["4.1.0"] = "4.1.0"

    @model_validator(mode="after")
    def normalize_speech_configuration(self) -> Self:
        if len(self.language_hints) != len(set(self.language_hints)):
            raise ValueError("language_hints 不得重复")
        object.__setattr__(self, "hotwords", normalize_hotwords(self.hotwords))
        object.__setattr__(self, "core_context", normalize_core_context(self.core_context))
        return self


_RETIRED_SPEECH_CONFIG_FIELDS = frozenset(
    {"speech_enrichment_mode", "min_speakers", "max_speakers"},
)


def pipeline_run_config_from_snapshot(
    snapshot: Mapping[str, object],
) -> PipelineRunConfig:
    """只在读取历史数据库快照时丢弃三个已退役语音字段。"""

    if snapshot.get("result_schema_version") != "4.1.0":
        raise VideoDemoError(
            ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
            "运行配置快照不是 4.1",
            {"supported_schema_version": "4.1.0"},
        )

    normalized = {
        key: value
        for key, value in snapshot.items()
        if key not in _RETIRED_SPEECH_CONFIG_FIELDS
    }
    return PipelineRunConfig.model_validate(normalized)


@dataclass(frozen=True, slots=True)
class RegisteredAsset:
    source_path: Path
    source_sha256: str
    object_ref: str
    source_size_bytes: int
    source_mime: SupportedMime
    run_relative_root: Path
    config: PipelineRunConfig


@dataclass(frozen=True, slots=True)
class ProbedAsset:
    asset: RegisteredAsset
    manifest: VideoAssetManifest
    limits: ProbeLimits
    warnings: tuple[str, ...] = ()
    timeline_duration_ms: int | None = None

    @property
    def duration_ms(self) -> int:
        if self.timeline_duration_ms is not None:
            return self.timeline_duration_ms
        return self.manifest.duration_ms


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    source: ProbedAsset
    # 视觉输入可以是原始视频，也可以是兼容性 fallback 生成的 proxy.mp4。
    proxy_path: Path
    proxy_sha256: str
    proxy_size_bytes: int
    audio_path: Path | None
    audio_sha256: str | None
    subtitle: ParsedSubtitle | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeechBoundaryCandidate:
    timestamp_ms: int
    source: Literal["silence", "sentence_end", "language_change"]
    score: float = 1.0


@dataclass(frozen=True, slots=True)
class StageMetric:
    stage: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class SpeechAnalysis:
    transcript_source: TranscriptSource
    evidence: tuple[SpeechSegment | SubtitleCue, ...] = ()
    warnings: tuple[str, ...] = ()
    boundary_candidates: tuple[SpeechBoundaryCandidate, ...] = ()
    stage_metrics: tuple[StageMetric, ...] = ()
    stage_cache_hits: tuple[str, ...] = ()
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...] = field(init=False)
    transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        transcript = tuple(
            item
            for item in self.evidence
            if isinstance(item, (SpeechSegment, SubtitleCue))
        )
        ordered = tuple(
            sorted(
                transcript,
                key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
            ),
        )
        evidence_ids = tuple(item.evidence_id for item in ordered)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise VideoDemoError(
                ErrorCode.DUPLICATE_EVIDENCE_ID,
                "转写证据标识不得重复",
            )
        if not _matches_transcript_source(self.transcript_source, ordered):
            raise VideoDemoError(
                ErrorCode.EVIDENCE_TYPE_MISMATCH,
                "转写证据类型与声明来源不一致",
            )
        object.__setattr__(self, "transcript_evidence", ordered)
        object.__setattr__(
            self,
            "transcript_by_id",
            MappingProxyType({item.evidence_id: item for item in ordered}),
        )


def _matches_transcript_source(
    transcript_source: TranscriptSource,
    transcript: tuple[SpeechSegment | SubtitleCue, ...],
) -> bool:
    if transcript_source == "NONE":
        return not transcript
    expected_type = SpeechSegment if transcript_source == "ASR" else SubtitleCue
    return all(isinstance(item, expected_type) for item in transcript)


class EvidencePreparationLimits(FrozenModel):
    max_transcript_evidence_items: int = Field(ge=1)
    max_transcript_chars: int = Field(ge=1)
    max_scene_boundaries: int = Field(ge=1)
    max_base_segments: int = Field(ge=1)


class SceneIndex(FrozenModel):
    proxy_sha256: Sha256
    duration_ms: int = Field(gt=0, le=7_200_000)
    frame_tolerance_ms: int = Field(ge=0, le=100)
    scenes: tuple[SceneBoundary, ...]
    index_sha256: Sha256

    @model_validator(mode="after")
    def validate_index_digest(self) -> Self:
        scene_ids = tuple(scene.evidence_id for scene in self.scenes)
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("场景索引标识不得重复")
        ordered = tuple(
            sorted(
                self.scenes,
                key=lambda scene: (scene.start_ms, scene.end_ms, scene.evidence_id),
            ),
        )
        if (
            not ordered
            or ordered != self.scenes
            or ordered[0].start_ms != 0
            or ordered[-1].end_ms != self.duration_ms
            or any(left.end_ms != right.start_ms for left, right in pairwise(ordered))
        ):
            raise ValueError("场景索引时间轴必须有序、连续并覆盖完整视频")
        if self.index_sha256 != scene_index_sha256(
            proxy_sha256=self.proxy_sha256,
            duration_ms=self.duration_ms,
            frame_tolerance_ms=self.frame_tolerance_ms,
            scenes=self.scenes,
        ):
            raise ValueError("场景索引摘要与规范内容不一致")
        return self


def scene_index_sha256(
    *,
    proxy_sha256: str,
    duration_ms: int,
    frame_tolerance_ms: int,
    scenes: tuple[SceneBoundary, ...],
) -> str:
    encoded = json.dumps(
        {
            "proxy_sha256": proxy_sha256,
            "duration_ms": duration_ms,
            "frame_tolerance_ms": frame_tolerance_ms,
            "scenes": [scene.model_dump(mode="json") for scene in scenes],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DocumentWritingContext(FrozenModel):
    """章节写作和全局编辑共享的本地事实闭包。"""

    run_id: StableId
    asset_sha256: Sha256
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    transcript_source: TranscriptSource
    document_config: DocumentGenerationConfig


def stable_merge_document_evidence(
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...],
    visual_evidence: tuple[KeyframeEvidence | VisualObservationEvidence, ...],
) -> tuple[DocumentEvidenceItem, ...]:
    combined = (*transcript_evidence, *visual_evidence)
    evidence_ids = [item.evidence_id for item in combined]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise VideoDemoError(
            ErrorCode.DUPLICATE_EVIDENCE_ID,
            "最终文档证据标识不得重复",
        )
    ordered_transcript = sorted(
        transcript_evidence,
        key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
    )
    ordered_visual = sorted(visual_evidence, key=_visual_evidence_sort_key)
    return (*ordered_transcript, *ordered_visual)


def require_result_evidence_budget(
    evidence: tuple[DocumentEvidenceItem, ...],
    max_result_evidence_items: int,
) -> None:
    """在写作前校验完整且 ID 唯一的最终证据闭包预算。"""

    if type(max_result_evidence_items) is not int or max_result_evidence_items < 1:
        raise ValueError("max_result_evidence_items 必须是正整数")
    evidence_ids = tuple(item.evidence_id for item in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise VideoDemoError(
            ErrorCode.DUPLICATE_EVIDENCE_ID,
            "最终文档证据标识不得重复",
        )
    if len(evidence) > max_result_evidence_items:
        raise VideoDemoError(
            ErrorCode.INPUT_BUDGET_EXCEEDED,
            "最终文档证据数量超过部署预算",
        )


def _visual_evidence_sort_key(
    item: KeyframeEvidence | VisualObservationEvidence,
) -> tuple[int, int, int, str]:
    if isinstance(item, KeyframeEvidence):
        return (0, item.timestamp_ms, item.timestamp_ms, item.evidence_id)
    return (1, item.start_ms, item.end_ms, item.evidence_id)


def merge_model_metrics(*metrics: Mapping[str, StrictInt]) -> dict[str, StrictInt]:
    merged: dict[str, StrictInt] = dict.fromkeys(MODEL_METRIC_NAMES, 0)
    for stage_metrics in metrics:
        unknown_names = set(stage_metrics) - MODEL_METRIC_NAMES
        if unknown_names:
            raise ValueError("模型指标包含未知白名单键")
        for name, value in stage_metrics.items():
            if type(value) is not int or value < 0:
                raise ValueError("模型指标必须是非负严格整数")
            total = merged.get(name, 0) + value
            if total > MAX_METRIC_VALUE:
                raise ValueError("模型指标累计值溢出 2^63-1")
            merged[name] = total
    return merged


def merge_run_statuses(
    *statuses: Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"],
) -> Literal["SUCCEEDED", "PARTIAL_SUCCEEDED"]:
    if any(status not in {"SUCCEEDED", "PARTIAL_SUCCEEDED"} for status in statuses):
        raise ValueError("运行状态只能是成功或部分成功")
    if "PARTIAL_SUCCEEDED" in statuses:
        return "PARTIAL_SUCCEEDED"
    return "SUCCEEDED"


def stable_merge_warnings(*warnings: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for stage_warnings in warnings:
        for warning in stage_warnings:
            if not warning.strip():
                raise ValueError("warning 码不能为空")
            if warning not in seen:
                seen.add(warning)
                merged.append(warning)
    return tuple(merged)
