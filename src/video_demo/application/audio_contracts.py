"""音频流水线专用的中性运行契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from pydantic import Field, ValidationError

from video_demo.domain.audio_plan import (
    AudioBaseSegment,
    AudioTranscriptEvidence,
    AudioTranscriptSource,
)
from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.evidence import SpeechSegment, SubtitleCue


class AudioEvidencePreparationLimits(FrozenModel):
    max_transcript_evidence_items: int = Field(ge=1)
    max_transcript_chars: int = Field(ge=1)
    max_base_segments: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class AudioSpeechBoundaryCandidate:
    timestamp_ms: int
    source: Literal["sentence_end", "language_change"]
    score: float = 1.0


@dataclass(frozen=True, slots=True)
class AudioStageMetric:
    stage: str
    duration_ms: int


@dataclass(frozen=True, slots=True)
class AudioSpeechAnalysis:
    transcript_source: AudioTranscriptSource
    evidence: tuple[SpeechSegment | SubtitleCue, ...] = ()
    warnings: tuple[str, ...] = ()
    boundary_candidates: tuple[AudioSpeechBoundaryCandidate, ...] = ()
    stage_metrics: tuple[AudioStageMetric, ...] = ()
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...] = field(init=False)
    transcript_by_id: Mapping[str, SpeechSegment | SubtitleCue] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                self.evidence,
                key=lambda item: (item.start_ms, item.end_ms, item.evidence_id),
            ),
        )
        ids = tuple(item.evidence_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise ValueError("音频转写证据标识不得重复")
        expected_type = {
            "ASR": SpeechSegment,
            "SUBTITLE": SubtitleCue,
        }.get(self.transcript_source)
        if expected_type is None and ordered:
            raise ValueError("无转写来源的音频分析不得包含证据")
        if expected_type is not None and any(
            not isinstance(item, expected_type) for item in ordered
        ):
            raise ValueError("音频转写来源与证据类型不一致")
        object.__setattr__(self, "transcript_evidence", ordered)
        object.__setattr__(
            self,
            "transcript_by_id",
            MappingProxyType({item.evidence_id: item for item in ordered}),
        )


class AudioTranscriptionCheckpoint(FrozenModel):
    """音频转写阶段唯一可恢复契约，不包含任何其他媒体业务字段。"""

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str
    asset_sha256: Sha256
    duration_ms: int = Field(gt=0, le=7_200_000)
    title_hint: str = Field(min_length=1, max_length=200)
    transcript_source: AudioTranscriptSource
    transcript_evidence: tuple[SpeechSegment | SubtitleCue, ...]
    base_segments: tuple[AudioBaseSegment, ...]
    warnings: tuple[str, ...] = ()
    stage_metrics: tuple[AudioStageMetric, ...] = ()

    def validate_consistency(self) -> AudioTranscriptionCheckpoint:
        if not self.transcript_evidence or self.transcript_source == "NONE":
            raise ValueError("音频转写快照必须包含转写证据")
        if not self.base_segments:
            raise ValueError("音频转写快照必须包含基础片段")
        if self.base_segments[0].start_ms != 0:
            raise ValueError("音频转写快照必须从 0 开始")
        if self.base_segments[-1].end_ms != self.duration_ms:
            raise ValueError("音频转写快照必须覆盖完整时长")
        if any(
            left.end_ms != right.start_ms
            for left, right in zip(self.base_segments, self.base_segments[1:], strict=False)
        ):
            raise ValueError("音频转写快照片段必须连续")
        evidence_ids = {item.evidence_id for item in self.transcript_evidence}
        if any(
            not set(segment.evidence_refs).issubset(evidence_ids)
            for segment in self.base_segments
        ):
            raise ValueError("音频转写快照片段引用未知证据")
        return self


def audio_transcription_checkpoint_to_payload(
    checkpoint: AudioTranscriptionCheckpoint,
) -> dict[str, object]:
    """将音频转写事实编码为独立 JSON，不复用视频快照结构。"""

    evidence = [
        {
            "kind": "SUBTITLE_CUE" if isinstance(item, SubtitleCue) else "ASR_SEGMENT",
            "payload": item.model_dump(mode="json", exclude_computed_fields=True),
        }
        for item in checkpoint.transcript_evidence
    ]
    return {
        "schema_version": checkpoint.schema_version,
        "run_id": checkpoint.run_id,
        "asset_sha256": checkpoint.asset_sha256,
        "duration_ms": checkpoint.duration_ms,
        "title_hint": checkpoint.title_hint,
        "transcript_source": checkpoint.transcript_source,
        "transcript_evidence": evidence,
        "base_segments": [
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in checkpoint.base_segments
        ],
        "warnings": list(checkpoint.warnings),
        "stage_metrics": [
            {"stage": item.stage, "duration_ms": item.duration_ms}
            for item in checkpoint.stage_metrics
        ],
    }


def audio_transcription_checkpoint_from_payload(
    payload: Mapping[str, object],
) -> AudioTranscriptionCheckpoint:
    """从音频 JSON 快照恢复事实并重新执行一致性校验。"""

    if payload.get("schema_version") != "1.0.0":
        raise ValueError("音频转写快照版本不受支持")
    evidence = tuple(
        _audio_evidence_from_payload(item)
        for item in _sequence(payload, "transcript_evidence")
    )
    checkpoint = AudioTranscriptionCheckpoint(
        run_id=str(payload.get("run_id", "")),
        asset_sha256=str(payload.get("asset_sha256", "")),
        duration_ms=_integer(payload.get("duration_ms"), "duration_ms"),
        title_hint=str(payload.get("title_hint", "")),
        transcript_source=cast(AudioTranscriptSource, str(payload.get("transcript_source", ""))),
        transcript_evidence=evidence,
        base_segments=tuple(
            AudioBaseSegment.model_validate(item)
            for item in _sequence(payload, "base_segments")
        ),
        warnings=tuple(str(item) for item in _sequence(payload, "warnings")),
        stage_metrics=tuple(
            AudioStageMetric(
                stage=str(_mapping(item, "stage metric").get("stage", "")),
                duration_ms=_integer(
                    _mapping(item, "stage metric").get("duration_ms"),
                    "stage metric.duration_ms",
                ),
            )
            for item in _sequence(payload, "stage_metrics")
        ),
    )
    return checkpoint.validate_consistency()


def _audio_evidence_from_payload(value: object) -> AudioTranscriptEvidence:
    item = _mapping(value, "音频证据")
    kind = item.get("kind")
    model_payload = _mapping(item.get("payload"), "音频证据 payload")
    try:
        if kind == "ASR_SEGMENT":
            return SpeechSegment.model_validate(model_payload)
        if kind == "SUBTITLE_CUE":
            return SubtitleCue.model_validate(model_payload)
    except ValidationError as error:
        raise ValueError("音频证据 payload 非法") from error
    raise ValueError("未知音频证据类型")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是对象")
    return value


def _sequence(payload: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = payload.get(name, ())
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是数组")
    return tuple(value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    return value
