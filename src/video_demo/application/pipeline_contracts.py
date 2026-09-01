from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field, StrictInt, model_validator

from video_demo.domain.base import FrozenModel, LanguageCode, Sha256, StableId
from video_demo.domain.document import DocumentGenerationConfig, TranscriptSource
from video_demo.domain.document_artifact import (
    MAX_METRIC_VALUE,
    MODEL_METRIC_NAMES,
    RESULT_STAGE_NAMES,
)
from video_demo.domain.document_plan import BaseSegment
from video_demo.domain.evidence import (
    DocumentEvidenceItem,
    KeyframeEvidence,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.domain.manifest import VideoAssetManifest
from video_demo.domain.speech_config import normalize_core_context, normalize_hotwords
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.audio_format import AUDIO_FORMAT_VERSION
from video_demo.media.probe import ProbeLimits, SupportedMime

if TYPE_CHECKING:
    from video_demo.application.chapter_frames import ChapterFrameSearchBatch
    from video_demo.application.chapter_vision import ChapterVisionBatch
    from video_demo.application.document_rendering import RenderedDocument
    from video_demo.domain.document import VideoUnderstandingResult
    from video_demo.media.subtitles import ParsedSubtitle
    from video_demo.persistence.scope import Scope
    from video_demo.storage.document_cache import DocumentModelCache


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
    result_schema_version: Literal["4.2.0"] = "4.2.0"

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

    if snapshot.get("result_schema_version") != "4.2.0":
        raise VideoDemoError(
            ErrorCode.RESULT_SCHEMA_UNSUPPORTED,
            "运行配置快照不是 4.1",
            {"supported_schema_version": "4.2.0"},
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
    # 兼容下游命名；视觉输入始终是当前 Run 内的原始视频。
    proxy_path: Path
    proxy_sha256: str
    proxy_size_bytes: int
    audio_path: Path | None
    audio_sha256: str | None
    subtitle: ParsedSubtitle | None = None
    warnings: tuple[str, ...] = ()
    audio_format_version: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionCheckpoint:
    """转写阶段完成后交给 LLM 阶段的已验证事实快照。"""

    registered: RegisteredAsset
    prepared: PreparedMedia
    speech: SpeechAnalysis
    base_segments: tuple[BaseSegment, ...]
    stage_metrics: Mapping[str, int] = field(default_factory=dict)
    stage_cache_hits: tuple[str, ...] = ()
    model_cache: DocumentModelCache | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.registered.source_sha256 != self.prepared.source.asset.source_sha256:
            raise ValueError("转写快照的原始视频摘要不一致")
        if (
            self.prepared.audio_path is not None
            and self.prepared.audio_format_version != AUDIO_FORMAT_VERSION
        ):
            raise ValueError("转写快照音频格式版本不受支持")


def transcription_checkpoint_to_payload(checkpoint: TranscriptionCheckpoint) -> dict[str, object]:
    """将转写快照编码为可持久化 JSON，不包含模型缓存实例。"""

    prepared = checkpoint.prepared
    speech = checkpoint.speech
    evidence = [
        {
            "kind": "SUBTITLE_CUE" if isinstance(item, SubtitleCue) else "ASR_SEGMENT",
            "payload": item.model_dump(mode="json"),
        }
        for item in speech.transcript_evidence
    ]
    return {
        "schema_version": "3.0.0",
        "registered": {
            "source_path": str(checkpoint.registered.source_path),
            "source_sha256": checkpoint.registered.source_sha256,
            "object_ref": checkpoint.registered.object_ref,
            "source_size_bytes": checkpoint.registered.source_size_bytes,
            "source_mime": checkpoint.registered.source_mime,
            "run_relative_root": checkpoint.registered.run_relative_root.as_posix(),
            "config": checkpoint.registered.config.model_dump(mode="json"),
        },
        "probed": {
            "manifest": prepared.source.manifest.model_dump(mode="json"),
            "limits": asdict(prepared.source.limits),
            "warnings": list(prepared.source.warnings),
            "timeline_duration_ms": prepared.source.timeline_duration_ms,
        },
        "prepared": {
            "proxy_path": str(prepared.proxy_path),
            "proxy_sha256": prepared.proxy_sha256,
            "proxy_size_bytes": prepared.proxy_size_bytes,
            "audio_path": str(prepared.audio_path) if prepared.audio_path else None,
            "audio_sha256": prepared.audio_sha256,
            "audio_format_version": prepared.audio_format_version,
            "subtitle": prepared.subtitle.model_dump(mode="json") if prepared.subtitle else None,
            "warnings": list(prepared.warnings),
        },
        "speech": {
            "transcript_source": speech.transcript_source,
            "evidence": evidence,
            "warnings": list(speech.warnings),
            "boundary_candidates": [
                {
                    "timestamp_ms": item.timestamp_ms,
                    "source": item.source,
                    "score": item.score,
                }
                for item in speech.boundary_candidates
            ],
            "stage_metrics": [
                {"stage": item.stage, "duration_ms": item.duration_ms}
                for item in speech.stage_metrics
            ],
            "stage_cache_hits": list(speech.stage_cache_hits),
        },
        "base_segments": [item.model_dump(mode="json") for item in checkpoint.base_segments],
        "stage_metrics": dict(checkpoint.stage_metrics),
        "stage_cache_hits": list(checkpoint.stage_cache_hits),
    }


def transcription_checkpoint_from_payload(
    payload: Mapping[str, object],
) -> TranscriptionCheckpoint:
    """从 JSON 快照恢复转写事实，并重新执行领域校验。"""

    if payload.get("schema_version") != "3.0.0":
        raise ValueError("转写快照版本不受支持")
    registered_payload = _mapping(payload, "registered")
    config = PipelineRunConfig.model_validate(_mapping(registered_payload, "config"))
    registered = RegisteredAsset(
        source_path=Path(str(registered_payload["source_path"])),
        source_sha256=str(registered_payload["source_sha256"]),
        object_ref=str(registered_payload["object_ref"]),
        source_size_bytes=int(registered_payload["source_size_bytes"]),
        source_mime=cast(SupportedMime, str(registered_payload["source_mime"])),
        run_relative_root=Path(str(registered_payload["run_relative_root"])),
        config=config,
    )
    probed_payload = _mapping(payload, "probed")
    manifest = VideoAssetManifest.model_validate(_mapping(probed_payload, "manifest"))
    probed = ProbedAsset(
        asset=registered,
        manifest=manifest,
        limits=ProbeLimits(**dict(_mapping(probed_payload, "limits"))),
        warnings=tuple(str(item) for item in _sequence(probed_payload, "warnings")),
        timeline_duration_ms=(
            int(probed_payload["timeline_duration_ms"])
            if probed_payload.get("timeline_duration_ms") is not None
            else None
        ),
    )
    prepared_payload = _mapping(payload, "prepared")
    subtitle_payload = prepared_payload.get("subtitle")
    subtitle_type = importlib.import_module("video_demo.media.subtitles").ParsedSubtitle
    subtitle = (
        subtitle_type.model_validate(subtitle_payload)
        if isinstance(subtitle_payload, dict)
        else None
    )
    prepared = PreparedMedia(
        source=probed,
        proxy_path=Path(str(prepared_payload["proxy_path"])),
        proxy_sha256=str(prepared_payload["proxy_sha256"]),
        proxy_size_bytes=int(prepared_payload["proxy_size_bytes"]),
        audio_path=(
            Path(str(prepared_payload["audio_path"]))
            if prepared_payload.get("audio_path")
            else None
        ),
        audio_sha256=(
            str(prepared_payload["audio_sha256"])
            if prepared_payload.get("audio_sha256")
            else None
        ),
        audio_format_version=(
            str(prepared_payload["audio_format_version"])
            if prepared_payload.get("audio_format_version") is not None
            else None
        ),
        subtitle=subtitle,
        warnings=tuple(str(item) for item in _sequence(prepared_payload, "warnings")),
    )
    if prepared.audio_path is not None and prepared.audio_format_version != AUDIO_FORMAT_VERSION:
        raise ValueError("转写快照音频格式版本不受支持")
    speech_payload = _mapping(payload, "speech")
    evidence: list[SpeechSegment | SubtitleCue] = []
    for item in _sequence(speech_payload, "evidence"):
        item_payload = _mapping(item, "evidence item")
        model_payload = _mapping(item_payload, "payload")
        if item_payload.get("kind") == "SUBTITLE_CUE":
            evidence.append(SubtitleCue.model_validate(model_payload))
        elif item_payload.get("kind") == "ASR_SEGMENT":
            evidence.append(SpeechSegment.model_validate(model_payload))
        else:
            raise ValueError("转写快照包含未知证据类型")
    speech = SpeechAnalysis(
        transcript_source=cast(TranscriptSource, str(speech_payload["transcript_source"])),
        evidence=tuple(evidence),
        warnings=tuple(str(item) for item in speech_payload.get("warnings", [])),
        boundary_candidates=tuple(
            SpeechBoundaryCandidate(**_mapping(item, "boundary candidate"))
            for item in _sequence(speech_payload, "boundary_candidates")
        ),
        stage_metrics=tuple(
            StageMetric(**_mapping(item, "stage metric"))
            for item in _sequence(speech_payload, "stage_metrics")
        ),
        stage_cache_hits=tuple(str(item) for item in speech_payload.get("stage_cache_hits", [])),
    )
    return TranscriptionCheckpoint(
        registered=registered,
        prepared=prepared,
        speech=speech,
        base_segments=tuple(
            BaseSegment.model_validate(item)
            for item in _sequence(payload, "base_segments")
        ),
        stage_metrics={
            str(key): int(value)
            for key, value in _mapping(payload, "stage_metrics").items()
        },
        stage_cache_hits=tuple(str(item) for item in _sequence(payload, "stage_cache_hits")),
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"转写快照字段 {name} 必须是对象")
    return value


def _sequence(payload: Mapping[str, object], name: str) -> tuple[Any, ...]:
    value = payload.get(name, ())
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"转写快照字段 {name} 必须是数组")
    return tuple(value)


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
    max_base_segments: int = Field(ge=1)


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
