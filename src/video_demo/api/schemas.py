from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from video_demo.domain.base import LanguageCode, Probability, StableId
from video_demo.domain.document import DocumentGenerationConfig
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
    document_config: DocumentGenerationConfig = Field(
        default_factory=DocumentGenerationConfig,
    )
    result_schema_version: Literal["4.1.0"] = "4.1.0"

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


class MediaObjectResponse(ApiModel):
    object_ref: str
    original_filename: str
    declared_mime: str
    detected_mime: str
    size_bytes: int
    sha256: str
    status: Literal["READY"] = "READY"


class CreateMediaRunRequest(ApiModel):
    object_ref: str = Field(pattern=r"^obj_[0-9a-f]{32}$")
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    language_hints: tuple[Literal["zh", "en", "ja", "ko", "es"], ...] = ()
    hotwords: tuple[str, ...] = Field(default=(), max_length=50)
    core_context: str | None = Field(default=None, max_length=1000)
    document_config: DocumentGenerationConfig = Field(default_factory=DocumentGenerationConfig)


class MediaRunResponse(ApiModel):
    run_id: str
    job_id: str
    status: str
    current_stage: str
    warning_codes: tuple[str, ...]
    error_code: str | None


class MediaRunHistoryItem(MediaRunResponse):
    object_ref: str
    original_filename: str
    detected_mime: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime


class MediaRunHistoryResponse(ApiModel):
    items: tuple[MediaRunHistoryItem, ...]


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


class PublicKeyframeEvidence(PublicTimedEvidence):
    evidence_type: Literal["KEYFRAME"] = "KEYFRAME"
    keyframe_id: StableId
    timestamp_ms: int
    mime_type: Literal["image/jpeg"]
    sha256: str
    perceptual_hash: str
    size_bytes: int = Field(ge=1)


class PublicVisualObservationEvidence(PublicTimedEvidence):
    evidence_type: Literal["VISUAL_OBSERVATION"] = "VISUAL_OBSERVATION"
    chapter_id: StableId
    target_ids: tuple[StableId, ...]
    keyframe_refs: tuple[StableId, ...]
    transcript_evidence_refs: tuple[StableId, ...]
    visual_type: Literal[
        "TEXT",
        "CODE",
        "TABLE",
        "FORMULA",
        "DIAGRAM",
        "UI_CONTROL",
        "TERMINAL",
        "GENERAL",
    ]
    caption: str
    content_blocks: tuple[dict[str, object], ...]
    visual_facts: tuple[dict[str, object], ...]
    frame_relations: tuple[dict[str, object], ...]
    relation_to_transcript: Literal[
        "SUPPORTING",
        "COMPLEMENTARY",
        "DUPLICATE",
        "CONFLICTING",
        "INDEPENDENT",
    ]
    certainty: Probability


PublicEvidence = Annotated[
    PublicSpeechSegment
    | PublicSubtitleCue
    | PublicKeyframeEvidence
    | PublicVisualObservationEvidence,
    Field(discriminator="evidence_type"),
]


class EvidencePageResponse(ApiModel):
    items: tuple[PublicEvidence, ...]
    next_cursor: str | None
