"""音频流水线专用的中性运行契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from pydantic import Field

from video_demo.domain.audio_plan import AudioTranscriptSource
from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import SpeechSegment, SubtitleCue


class AudioEvidencePreparationLimits(FrozenModel):
    max_transcript_evidence_items: int = Field(ge=1)
    max_transcript_chars: int = Field(ge=1)
    max_base_segments: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class AudioSpeechBoundaryCandidate:
    timestamp_ms: int
    source: Literal["silence", "sentence_end", "language_change"]
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

