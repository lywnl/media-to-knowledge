from __future__ import annotations

import hashlib
import json
import re
import unicodedata
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
    r"(?i)(?:"
    r"https?://|data:|bearer\s+|"
    r"(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|xox[baprs]-|AKIA|AIza|hf_|glpat-|npm_|pypi-)"
    r"[A-Za-z0-9_-]{16,}|"
    r"(?:[A-Za-z0-9+/]{20,}={1,2})|(?:[A-Za-z0-9+/]{80,})"
    r")",
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


class ChapterBoundaryInput(FrozenModel):
    """跨批次边界协调所需的最小事实投影。"""

    boundary_index: int = Field(ge=0)
    left_title_hint: str = Field(min_length=1, max_length=200)
    right_title_hint: str = Field(min_length=1, max_length=200)
    left_duration_ms: int = Field(gt=0, le=300_000)
    right_duration_ms: int = Field(gt=0, le=300_000)
    left_tail_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=2)
    right_head_evidence: tuple[TranscriptEvidence, ...] = Field(max_length=2)


class ChapterBoundaryCoordinationRequest(FrozenModel):
    boundaries: tuple[ChapterBoundaryInput, ...] = Field(min_length=1, max_length=63)
    prompt_version: Literal["chapter-boundary-coordinator-v1"]

    @model_validator(mode="after")
    def validate_request_size(self) -> Self:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("章节边界协调请求超过 64 KiB")
        return self


class ChapterBoundaryDecision(FrozenModel):
    boundary_index: int = Field(ge=0)
    decision: Literal["KEEP", "MERGE"]
    merged_title_hint: str | None = Field(default=None, max_length=200)


class ChapterBoundaryCoordinationResponse(FrozenModel):
    decisions: tuple[ChapterBoundaryDecision, ...] = Field(max_length=63)

    @model_validator(mode="after")
    def reject_duplicate_boundaries(self) -> Self:
        _reject_duplicate_ids(
            tuple(str(item.boundary_index) for item in self.decisions),
            "decisions.boundary_index",
        )
        return self


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
        if any(_contains_control_character(item) for item in value):
            raise ValueError("validation_errors 不得包含控制字符")
        if any(_SUSPICIOUS_VALUE.search(item) for item in value):
            raise ValueError("validation_errors 不得包含疑似敏感信息")
        return value

    @field_validator("safe_json_excerpt")
    @classmethod
    def validate_safe_json_excerpt(cls, value: str | None) -> str | None:
        if value is not None:
            if _contains_control_character(value):
                raise ValueError("safe_json_excerpt 不得包含控制字符")
            if _SUSPICIOUS_VALUE.search(value):
                raise ValueError("safe_json_excerpt 不得包含疑似敏感信息")
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
    max_selected_frames: int = Field(default=2, ge=0, le=3)
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
    title_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    summary_zh: str = Field(max_length=500)
    summary_evidence_refs: tuple[StableId, ...] = Field(min_length=1, max_length=32)
    body_blocks: tuple[ChapterBodyBlock, ...] = Field(max_length=128)
    claims: tuple[GroundedClaim, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def reject_duplicate_header_refs(self) -> Self:
        _reject_duplicate_ids(self.title_evidence_refs, "title_evidence_refs")
        _reject_duplicate_ids(self.summary_evidence_refs, "summary_evidence_refs")
        return self


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


class GlobalChapterInput(FrozenModel):
    """全局编辑器所需的章节事实投影，不携带正文证据闭包。"""

    chapter_id: StableId
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0, le=7_200_000)
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(max_length=4_000)
    content_status: Literal["GROUNDED", "NO_SEMANTIC_EVIDENCE"]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("全局章节时间范围非法")
        return self


class GlobalWritingRequest(FrozenModel):
    context: DocumentWritingContext
    chapters: tuple[GlobalChapterInput, ...] = Field(min_length=1, max_length=240)
    prompt_version: Literal["global-editor-v1"]

    @model_validator(mode="after")
    def validate_chapter_timeline(self) -> Self:
        if self.chapters[0].start_ms != 0:
            raise ValueError("全局章节必须从 0 开始")
        if self.chapters[-1].end_ms != self.context.duration_ms:
            raise ValueError("全局章节必须覆盖完整视频时长")
        for previous, current in zip(self.chapters[:-1], self.chapters[1:], strict=True):
            if previous.end_ms != current.start_ms:
                raise ValueError("全局章节必须连续且无重叠")
        _reject_duplicate_ids(
            tuple(chapter.chapter_id for chapter in self.chapters),
            "chapters.chapter_id",
        )
        return self


class GlobalWritingResponse(FrozenModel):
    overview_zh: str = Field(max_length=8_000)


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

    def coordinate_chapter_boundaries(
        self,
        request: ChapterBoundaryCoordinationRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterBoundaryCoordinationResponse: ...

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
    return tuple(chapter.chapter_id for chapter in request.chapters)


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
        return (
            _SUSPICIOUS_VALUE.search(value) is None
            and not _contains_control_character(value)
        )
    if isinstance(value, dict):
        return all(
            isinstance(item_key, str)
            and _SUSPICIOUS_VALUE.search(item_key) is None
            and not _contains_control_character(item_key)
            and _is_safe_json_value(item_value, key=item_key)
            for item_key, item_value in value.items()
        )
    if isinstance(value, list):
        return all(_is_safe_json_value(item) for item in value)
    return value is None or isinstance(value, (bool, int, float))


def _normalize_validation_error(value: str) -> str:
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    normalized = " ".join(without_controls.split())
    return (normalized or "response:invalid")[:500]


def _contains_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _require_exact_ids(actual: tuple[str, ...], expected: tuple[str, ...], field: str) -> None:
    _reject_duplicate_ids(actual, field)
    if actual != expected:
        raise ValueError(f"{field} 必须严格等于原请求的有序引用白名单")


def _reject_duplicate_ids(values: tuple[str, ...], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} 不得重复")
