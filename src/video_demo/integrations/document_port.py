from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from video_demo.application.pipeline_contracts import DocumentWritingContext
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.document import (
    ChapterBodyBlock,
    DocumentGenerationConfig,
    GroundedClaim,
    SectionDraft,
    SummaryPoint,
)
from video_demo.domain.document_plan import (
    BaseSegment,
    ChapterDraft,
    ChapterPlan,
    FrameCandidateArtifact,
    VisualSearchTarget,
)
from video_demo.domain.evidence import (
    ChapterVisualObservation,
    SpeechSegment,
    SubtitleCue,
    VisualObservationEvidence,
)
from video_demo.errors import ErrorCode, VideoDemoError

TranscriptEvidence = SpeechSegment | SubtitleCue
_SUSPICIOUS_VALUE = re.compile(
    r"(?i)(?:https?://|data:|bearer\s+|(?:[A-Za-z0-9+/]{80,}={0,2}))",
)
_SENSITIVE_KEY = re.compile(r"(?i)(?:authorization|api[_-]?key|token|password|secret)")


class ChapterPlanningRequest(FrozenModel):
    title_hint: str = Field(min_length=1, max_length=200)
    duration_ms: int = Field(gt=0, le=7_200_000)
    segments: tuple[BaseSegment, ...] = Field(min_length=1, max_length=20_000)
    transcript_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=20_000)
    document_config: DocumentGenerationConfig
    prompt_version: Literal["chapter-planner-v1"]


class ChapterPlanningResponse(FrozenModel):
    chapter_drafts: tuple[ChapterDraft, ...] = Field(min_length=1, max_length=240)


class InvalidModelResponse(FrozenModel):
    """不含原始响应的有界结构修复上下文。"""

    content_sha256: Sha256
    validation_errors: tuple[str, ...] = Field(min_length=1, max_length=32)
    safe_json_excerpt: str | None = Field(default=None, max_length=8_000)

    @field_validator("validation_errors")
    @classmethod
    def validate_error_summaries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("validation_errors 不得重复")
        if any(not item or len(item) > 500 for item in value):
            raise ValueError("validation_errors 每条必须为 1~500 个字符")
        return value


class ModelResponseValidationError(VideoDemoError):
    """在适配器与应用服务之间传递已脱敏的结构修复上下文。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        invalid_response: InvalidModelResponse,
    ) -> None:
        if code not in {
            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
            ErrorCode.QWEN_RESPONSE_INVALID,
        }:
            raise ValueError("模型响应校验异常只能使用文本或视觉响应非法错误码")
        super().__init__(code, message)
        self._invalid_response = invalid_response

    @property
    def invalid_response(self) -> InvalidModelResponse:
        return self._invalid_response


class ChapterPlanRepairRequest(FrozenModel):
    request: ChapterPlanningRequest
    invalid_response: InvalidModelResponse
    allowed_segment_ids: tuple[StableId, ...] = Field(min_length=1, max_length=20_000)
    allowed_transcript_ids: tuple[StableId, ...] = Field(max_length=20_000)
    prompt_version: Literal["chapter-planner-repair-v1"]

    @model_validator(mode="after")
    def validate_allowed_ids(self) -> Self:
        _require_exact_ids(
            self.allowed_segment_ids,
            tuple(item.segment_id for item in self.request.segments),
            "allowed_segment_ids",
        )
        _require_exact_ids(
            self.allowed_transcript_ids,
            tuple(item.evidence_id for item in self.request.transcript_evidence),
            "allowed_transcript_ids",
        )
        return self


class ChapterVisionRequest(FrozenModel):
    chapter_id: StableId
    targets: tuple[VisualSearchTarget, ...] = Field(min_length=1, max_length=6)
    frames: tuple[FrameCandidateArtifact, ...] = Field(min_length=1, max_length=6)
    transcript_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=20_000)
    document_config: DocumentGenerationConfig
    prompt_version: Literal["chapter-vlm-v1"]


class ChapterVisionResponse(FrozenModel):
    observations: tuple[ChapterVisualObservation, ...] = Field(max_length=16)


class ChapterVisionRepairRequest(FrozenModel):
    request: ChapterVisionRequest
    invalid_response: InvalidModelResponse
    allowed_frame_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)
    allowed_target_ids: tuple[StableId, ...] = Field(min_length=1, max_length=6)
    allowed_transcript_evidence_ids: tuple[StableId, ...] = Field(max_length=20_000)
    prompt_version: Literal["chapter-vlm-repair-v1"]

    @model_validator(mode="after")
    def validate_allowed_ids(self) -> Self:
        ordered_frames = sorted(
            self.request.frames,
            key=lambda item: (item.timestamp_ms, item.frame_id),
        )
        _require_exact_ids(
            self.allowed_frame_ids,
            tuple(item.frame_id for item in ordered_frames),
            "allowed_frame_ids",
        )
        _require_exact_ids(
            self.allowed_target_ids,
            tuple(item.target_id for item in self.request.targets),
            "allowed_target_ids",
        )
        _require_exact_ids(
            self.allowed_transcript_evidence_ids,
            tuple(item.evidence_id for item in self.request.transcript_evidence),
            "allowed_transcript_evidence_ids",
        )
        return self


class ChapterWritingRequest(FrozenModel):
    context: DocumentWritingContext
    chapter: ChapterPlan
    transcript_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=20_000)
    visual_observations: tuple[VisualObservationEvidence, ...] = Field(max_length=16)
    prompt_version: Literal["chapter-writer-v1"]


class ChapterWritingResponse(FrozenModel):
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=500)
    body_blocks: tuple[ChapterBodyBlock, ...] = Field(max_length=128)
    claims: tuple[GroundedClaim, ...] = Field(max_length=128)


class ChapterWritingRepairRequest(FrozenModel):
    request: ChapterWritingRequest
    invalid_response: InvalidModelResponse
    allowed_evidence_ids: tuple[StableId, ...] = Field(min_length=1, max_length=20_016)
    prompt_version: Literal["chapter-writer-repair-v1"]

    @model_validator(mode="after")
    def validate_allowed_ids(self) -> Self:
        _require_exact_ids(
            self.allowed_evidence_ids,
            allowed_writing_evidence_ids(self.request),
            "allowed_evidence_ids",
        )
        return self


class GlobalChapterGroup(FrozenModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0, le=7_200_000)
    chapter_refs: tuple[StableId, ...] = Field(min_length=1, max_length=20)
    grounded_chapter_refs: tuple[StableId, ...] = Field(max_length=20)
    digest_zh: str = Field(max_length=4_000)

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("全局章节组的 end_ms 必须大于 start_ms")
        _reject_duplicate_ids(self.chapter_refs, "chapter_refs")
        _reject_duplicate_ids(self.grounded_chapter_refs, "grounded_chapter_refs")
        if not set(self.grounded_chapter_refs).issubset(self.chapter_refs):
            raise ValueError("grounded_chapter_refs 必须属于 chapter_refs")
        if bool(self.grounded_chapter_refs) != bool(self.digest_zh):
            raise ValueError("只有包含事实章节的分组才能提供 digest_zh")
        return self


class GlobalWritingRequest(FrozenModel):
    context: DocumentWritingContext
    groups: tuple[GlobalChapterGroup, ...] = Field(min_length=1, max_length=240)
    prompt_version: Literal["global-editor-v1"]

    @model_validator(mode="after")
    def validate_group_timeline(self) -> Self:
        if self.groups[0].start_ms != 0:
            raise ValueError("全局章节组必须从 0 开始")
        if self.groups[-1].end_ms != self.context.duration_ms:
            raise ValueError("全局章节组必须覆盖完整视频时长")
        for previous, current in zip(self.groups[:-1], self.groups[1:], strict=True):
            if previous.end_ms != current.start_ms:
                raise ValueError("全局章节组必须连续且无重叠")
        chapter_refs = tuple(ref for group in self.groups for ref in group.chapter_refs)
        _reject_duplicate_ids(chapter_refs, "groups.chapter_refs")
        return self


class GlobalWritingResponse(FrozenModel):
    overview_zh: str = Field(max_length=8_000)
    key_points: tuple[SummaryPoint, ...] = Field(max_length=64)
    sections: tuple[SectionDraft, ...] = Field(min_length=1, max_length=240)


class GlobalWritingRepairRequest(FrozenModel):
    request: GlobalWritingRequest
    invalid_response: InvalidModelResponse
    allowed_chapter_ids: tuple[StableId, ...] = Field(min_length=1, max_length=240)
    prompt_version: Literal["global-editor-repair-v1"]

    @model_validator(mode="after")
    def validate_allowed_ids(self) -> Self:
        _require_exact_ids(
            self.allowed_chapter_ids,
            allowed_global_chapter_ids(self.request),
            "allowed_chapter_ids",
        )
        return self


class DocumentTextPort(Protocol):
    def plan_chapters(
        self,
        request: ChapterPlanningRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse: ...

    def repair_chapter_plan(
        self,
        request: ChapterPlanRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse: ...

    def write_chapter(
        self,
        request: ChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse: ...

    def repair_chapter_writing(
        self,
        request: ChapterWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse: ...

    def organize_document(
        self,
        request: GlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse: ...

    def repair_global_writing(
        self,
        request: GlobalWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse: ...


class ChapterVisionPort(Protocol):
    def analyze_chapter(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse: ...

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse: ...


def allowed_writing_evidence_ids(request: ChapterWritingRequest) -> tuple[str, ...]:
    """按请求证据首次出现顺序生成章节写作白名单。"""

    ordered: list[str] = [item.evidence_id for item in request.transcript_evidence]
    for observation in request.visual_observations:
        ordered.append(observation.evidence_id)
    return tuple(dict.fromkeys(ordered))


def allowed_global_chapter_ids(request: GlobalWritingRequest) -> tuple[str, ...]:
    return tuple(ref for group in request.groups for ref in group.chapter_refs)


def invalid_model_response(
    raw_content: bytes,
    validation_errors: tuple[str, ...],
    *,
    parsed_json: object | None,
) -> InvalidModelResponse:
    """从有界响应内容生成不可泄密的结构修复描述。"""

    summaries = tuple(
        dict.fromkeys(_normalize_validation_error(item) for item in validation_errors)
    )[:32]
    return InvalidModelResponse(
        content_sha256=hashlib.sha256(raw_content).hexdigest(),
        validation_errors=summaries or ("response:invalid",),
        safe_json_excerpt=_safe_json_excerpt(parsed_json),
    )


def _safe_json_excerpt(value: object | None) -> str | None:
    if value is None or not _is_safe_json_value(value):
        return None
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return serialized[:8_000]


def _is_safe_json_value(value: object, *, key: str | None = None) -> bool:
    if key is not None and _SENSITIVE_KEY.search(key):
        return False
    if isinstance(value, str):
        return _SUSPICIOUS_VALUE.search(value) is None
    if isinstance(value, dict):
        return all(
            isinstance(item_key, str)
            and _is_safe_json_value(item_value, key=item_key)
            for item_key, item_value in value.items()
        )
    if isinstance(value, list):
        return all(_is_safe_json_value(item) for item in value)
    return value is None or isinstance(value, (bool, int, float))


def _normalize_validation_error(value: str) -> str:
    normalized = " ".join(value.split())
    return (normalized or "response:invalid")[:500]


def _require_exact_ids(actual: tuple[str, ...], expected: tuple[str, ...], field: str) -> None:
    _reject_duplicate_ids(actual, field)
    if actual != expected:
        raise ValueError(f"{field} 必须严格等于原请求的有序引用白名单")


def _reject_duplicate_ids(values: tuple[str, ...], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} 不得重复")
