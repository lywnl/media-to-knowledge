from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from video_demo.domain.base import LanguageCode, Probability, StableId
from video_demo.domain.evidence import BoundingBox, SpeakerId
from video_demo.domain.run import TimeRange
from video_demo.domain.speech_config import normalize_core_context, normalize_hotwords


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoObjectResponse(ApiModel):
    object_ref: str
    original_filename: str
    declared_mime: str
    detected_mime: str
    size_bytes: int
    sha256: str
    status: Literal["READY"] = "READY"


class CreateRunRequest(ApiModel):
    object_ref: str = Field(pattern=r"^obj_[0-9a-f]{32}$")
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    language_hints: tuple[Literal["zh", "en", "ja", "ko", "es"], ...] = ()
    hotwords: tuple[str, ...] = Field(default=(), max_length=50)
    core_context: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def normalize_speech_configuration(self) -> Self:
        if len(self.language_hints) != len(set(self.language_hints)):
            raise ValueError("language_hints 不得重复")
        self.hotwords = normalize_hotwords(self.hotwords)
        self.core_context = normalize_core_context(self.core_context)
        return self


class RunResponse(ApiModel):
    run_id: str
    job_id: str
    status: str
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


class RunHistoryItem(ApiModel):
    run_id: str
    object_ref: str
    original_filename: str
    detected_mime: str
    size_bytes: int
    status: str
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None
    created_at: datetime
    updated_at: datetime


class RunHistoryResponse(ApiModel):
    items: tuple[RunHistoryItem, ...]


class JobResponse(ApiModel):
    job_id: str
    resource_id: str
    status: str
    attempt_count: int
    max_attempts: int
    error_code: str | None


class ErrorBody(ApiModel):
    code: str
    message: str
    details: dict[str, object]


class ErrorResponse(ApiModel):
    error: ErrorBody


class PublicTimedEvidence(TimeRange):
    evidence_id: StableId


class PublicSpeechSegment(PublicTimedEvidence):
    evidence_type: Literal["ASR_SEGMENT"] = "ASR_SEGMENT"
    text: str
    language: LanguageCode
    confidence: Probability
    is_fully_evaluated_language: bool


class PublicSubtitleCue(PublicTimedEvidence):
    evidence_type: Literal["SUBTITLE_CUE"] = "SUBTITLE_CUE"
    text: str
    language: LanguageCode
    stream_index: int = Field(ge=0)


class PublicAlignedWord(PublicTimedEvidence):
    evidence_type: Literal["ALIGNED_WORD"] = "ALIGNED_WORD"
    text: str
    language: LanguageCode
    probability: Probability
    speaker: SpeakerId
    overlap_speakers: tuple[SpeakerId, ...]


class PublicSpeakerTurn(PublicTimedEvidence):
    evidence_type: Literal["SPEAKER_TURN"] = "SPEAKER_TURN"
    speaker: SpeakerId
    confidence: Probability | None
    overlap_speakers: tuple[SpeakerId, ...]


class PublicAudioEvent(PublicTimedEvidence):
    evidence_type: Literal["AUDIO_EVENT"] = "AUDIO_EVENT"
    audioset_class: str
    normalized_event: str
    confidence: Probability
    threshold_version: str


class PublicSceneBoundary(PublicTimedEvidence):
    evidence_type: Literal["SCENE"] = "SCENE"
    transition: Literal["hard_cut", "gradual", "candidate"]
    score: Probability


class PublicKeyframeEvidence(PublicTimedEvidence):
    evidence_type: Literal["KEYFRAME"] = "KEYFRAME"
    keyframe_id: StableId
    timestamp_ms: int
    mime_type: Literal["image/jpeg", "image/png"]
    sha256: str
    perceptual_hash: str


class PublicOcrLine(ApiModel):
    text: str
    bounding_box: BoundingBox | None = None
    confidence: Probability


class PublicOcrEvidence(PublicTimedEvidence):
    evidence_type: Literal["OCR"] = "OCR"
    keyframe_id: StableId
    timestamp_ms: int
    language: LanguageCode
    lines: tuple[PublicOcrLine, ...]
    provider_request_id: str


PublicEvidence = Annotated[
    PublicSpeechSegment
    | PublicSubtitleCue
    | PublicAlignedWord
    | PublicSpeakerTurn
    | PublicAudioEvent
    | PublicSceneBoundary
    | PublicKeyframeEvidence
    | PublicOcrEvidence,
    Field(discriminator="evidence_type"),
]


class EvidencePageResponse(ApiModel):
    items: tuple[PublicEvidence, ...]
    next_cursor: str | None
