from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from video_demo.application.pipeline import SpeechAnalysis, SpeechBoundaryCandidate
from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    EvidenceItem,
    SpeakerTurn,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result_artifact import TranscriptSource
from video_demo.domain.run import ModelIdentity
from video_demo.speech.language import LanguageSpan

_ASR_COMPONENTS = frozenset({"silero_vad", "faster_whisper"})
_DOWNSTREAM_COMPONENTS = frozenset({"whisperx", "pyannote", "yamnet"})


class AsrSnapshotPayload(FrozenModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    language_spans: tuple[LanguageSpan, ...]
    segments: tuple[SpeechSegment, ...]
    vad_warnings: tuple[str, ...]
    silence_boundaries_ms: tuple[int, ...]
    language_change_boundaries_ms: tuple[int, ...]


class SpeechBoundaryCandidateSnapshot(FrozenModel):
    timestamp_ms: int = Field(gt=0)
    source: Literal["silence", "sentence_end", "speaker_change", "language_change"]
    score: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_candidate(
        cls,
        candidate: SpeechBoundaryCandidate,
    ) -> SpeechBoundaryCandidateSnapshot:
        return cls(
            timestamp_ms=candidate.timestamp_ms,
            source=candidate.source,
            score=candidate.score,
        )

    def to_candidate(self) -> SpeechBoundaryCandidate:
        return SpeechBoundaryCandidate(self.timestamp_ms, self.source, self.score)


class SpeechAnalysisSnapshotPayload(FrozenModel):
    schema_version: Literal["1.0.0", "2.0.0"] = "2.0.0"
    enrichment_mode: Literal["text", "full"]
    evidence: tuple[EvidenceItem, ...]
    warnings: tuple[str, ...]
    boundary_candidates: tuple[SpeechBoundaryCandidateSnapshot, ...]
    transcript_source: TranscriptSource

    @model_validator(mode="before")
    @classmethod
    def normalize_enrichment_mode_by_version(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        version = value.get("schema_version", "2.0.0")
        if version == "1.0.0":
            if "enrichment_mode" not in value:
                return {**value, "enrichment_mode": "full"}
            if value["enrichment_mode"] != "full":
                raise ValueError("历史语音快照只支持 full 模式")
            return value
        if version == "2.0.0" and "enrichment_mode" not in value:
            raise ValueError("2.0.0 语音快照必须显式携带 enrichment_mode")
        return value

    @model_validator(mode="after")
    def validate_transcript_source_evidence(self) -> SpeechAnalysisSnapshotPayload:
        subtitle_cues = tuple(item for item in self.evidence if isinstance(item, SubtitleCue))
        asr_evidence = tuple(
            item
            for item in self.evidence
            if isinstance(item, (SpeechSegment, AlignedWord, SpeakerTurn, AudioEvent))
        )
        if self.enrichment_mode == "text" and any(
            isinstance(item, (AlignedWord, SpeakerTurn, AudioEvent)) for item in self.evidence
        ):
            raise ValueError("text 模式快照不得包含词级、说话人或音频事件证据")
        if self.transcript_source == "SUBTITLE":
            if not subtitle_cues or asr_evidence:
                raise ValueError("字幕快照必须包含字幕且不得包含 ASR 语音证据")
        elif self.transcript_source == "ASR":
            if subtitle_cues:
                raise ValueError("ASR 快照不得包含字幕证据")
        elif subtitle_cues or asr_evidence:
            raise ValueError("无文本来源快照不得包含字幕或 ASR 语音证据")
        return self

    @classmethod
    def from_analysis(cls, analysis: SpeechAnalysis) -> SpeechAnalysisSnapshotPayload:
        return cls(
            evidence=analysis.evidence,
            enrichment_mode=analysis.enrichment_mode,
            warnings=analysis.warnings,
            boundary_candidates=tuple(
                SpeechBoundaryCandidateSnapshot.from_candidate(candidate)
                for candidate in analysis.boundary_candidates
            ),
            transcript_source=analysis.transcript_source,
        )

    def to_analysis(self) -> SpeechAnalysis:
        return SpeechAnalysis(
            transcript_source=self.transcript_source,
            enrichment_mode=self.enrichment_mode,
            evidence=self.evidence,
            warnings=self.warnings,
            boundary_candidates=tuple(
                candidate.to_candidate() for candidate in self.boundary_candidates
            ),
        )


class SpeechFingerprintInputs(FrozenModel):
    model_identities: tuple[ModelIdentity, ...]
    vad_threshold: float = 0.5
    vad_merge_gap_ms: int = 200
    lid_threshold: float = 0.6
    asr_beam_size: int = 5
    asr_compute_type: str = Field(min_length=1, max_length=32)
    yamnet_class_map_sha256: Sha256
    yamnet_thresholds_sha256: Sha256


def asr_fingerprint(
    *,
    audio_sha256: str,
    duration_ms: int,
    language_hints: tuple[str, ...],
    hotwords: tuple[str, ...],
    core_context: str | None,
    inputs: SpeechFingerprintInputs,
) -> str:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    return _canonical_sha256(
        {
            "schema_version": AsrSnapshotPayload.model_fields["schema_version"].default,
            "audio_sha256": audio_sha256,
            "duration_ms": duration_ms,
            "language_hints": language_hints,
            "hotwords": hotwords,
            "core_context": core_context,
            "model_identities": _model_payload(inputs, _ASR_COMPONENTS),
            "vad_threshold": inputs.vad_threshold,
            "vad_merge_gap_ms": inputs.vad_merge_gap_ms,
            "lid_threshold": inputs.lid_threshold,
            "asr_beam_size": inputs.asr_beam_size,
            "asr_compute_type": inputs.asr_compute_type,
        }
    )


def speech_fingerprint(
    *,
    processing_mode: Literal["SUBTITLE", "ASR"],
    transcript_payload_sha256: str,
    media_warnings: tuple[str, ...],
    min_speakers: int | None,
    max_speakers: int | None,
    allow_speaker_fallback: bool,
    inputs: SpeechFingerprintInputs,
    enrichment_mode: Literal["text", "full"] = "full",
) -> str:
    payload: dict[str, object] = {
        "schema_version": SpeechAnalysisSnapshotPayload.model_fields[
            "schema_version"
        ].default,
        "processing_mode": processing_mode,
        "enrichment_mode": enrichment_mode if processing_mode == "ASR" else "text",
        "transcript_payload_sha256": transcript_payload_sha256,
        "media_warnings": sorted(set(media_warnings)),
    }
    if processing_mode == "SUBTITLE":
        payload["subtitle_passthrough_contract"] = "1.0.0"
    elif enrichment_mode == "full":
        payload.update(
            {
                "model_identities": _model_payload(inputs, _DOWNSTREAM_COMPONENTS),
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "allow_speaker_fallback": allow_speaker_fallback,
                "yamnet_class_map_sha256": inputs.yamnet_class_map_sha256,
                "yamnet_thresholds_sha256": inputs.yamnet_thresholds_sha256,
            }
        )
    return _canonical_sha256(payload)


def _speech_fingerprint_v1(
    *,
    processing_mode: Literal["SUBTITLE", "ASR"],
    transcript_payload_sha256: str,
    media_warnings: tuple[str, ...],
    min_speakers: int | None,
    max_speakers: int | None,
    allow_speaker_fallback: bool,
    inputs: SpeechFingerprintInputs,
) -> str:
    """重建历史 1.0.0 speech 指纹，仅用于仍指向旧快照的兼容读取。"""

    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "processing_mode": processing_mode,
        "transcript_payload_sha256": transcript_payload_sha256,
        "media_warnings": sorted(set(media_warnings)),
    }
    if processing_mode == "SUBTITLE":
        payload["subtitle_passthrough_contract"] = "1.0.0"
    else:
        payload.update(
            {
                "model_identities": _model_payload(inputs, _DOWNSTREAM_COMPONENTS),
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "allow_speaker_fallback": allow_speaker_fallback,
                "yamnet_class_map_sha256": inputs.yamnet_class_map_sha256,
                "yamnet_thresholds_sha256": inputs.yamnet_thresholds_sha256,
            },
        )
    return _canonical_sha256(payload)


def subtitle_transcript_payload_sha256(analysis: SpeechAnalysis, artifact_sha256: str) -> str:
    return _canonical_sha256(
        {
            "artifact_sha256": artifact_sha256,
            "cues": [
                item.model_dump(mode="json", exclude_computed_fields=True)
                for item in analysis.evidence
            ],
        }
    )


def _model_payload(
    inputs: SpeechFingerprintInputs,
    components: frozenset[str],
) -> list[dict[str, object]]:
    return [
        identity.model_dump(mode="json", exclude_none=True)
        for identity in sorted(
            (
                item
                for item in inputs.model_identities
                if item.component in components
            ),
            key=lambda item: (item.component, item.model_id, item.revision or ""),
        )
    ]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
