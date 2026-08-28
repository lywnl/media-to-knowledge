from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from video_demo.domain.document import VisualBlock
from video_demo.domain.document_plan import ChapterDraft, VisualTargetDraft
from video_demo.domain.evidence import VisualObservationEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterPlanningRequest,
    ChapterPlanningResponse,
    ChapterPlanRepairRequest,
    ChapterWritingRepairRequest,
    ChapterWritingRequest,
    ChapterWritingResponse,
    DocumentTextPort,
    GlobalWritingRepairRequest,
    GlobalWritingRequest,
    GlobalWritingResponse,
    ModelResponseValidationError,
    allowed_global_chapter_ids,
    allowed_writing_evidence_ids,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import (
    prompt_for_compact_plan_repair,
    prompt_for_compact_planning,
    prompt_for_global_editing,
    prompt_for_global_repair,
    prompt_for_plan_repair,
    prompt_for_planning,
    prompt_for_writing,
    prompt_for_writing_repair,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
Prompt = tuple[str, str, str]
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
# 紧凑索引协议关闭 thinking 以减少延迟；保留 8192 上限，避免长批次的
# 结构化 JSON 因输出预算过小被截断并触发额外修复调用。
_COMPACT_PLANNING_MAX_OUTPUT_TOKENS = 8_192


class _CompactVisualTargetDraft(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    query_zh: str = Field(min_length=1, max_length=500)
    anchor_transcript_indexes: tuple[int, ...] = Field(min_length=1, max_length=3)


class _CompactChapterDraft(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    start_segment_index: int = Field(ge=0)
    end_segment_index: int = Field(gt=0)
    title_hint: str = Field(min_length=1, max_length=200)
    visual_mode: Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]
    semantic_targets: tuple[_CompactVisualTargetDraft, ...] = Field(max_length=4)


class _CompactChapterPlanningResponse(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    chapter_drafts: tuple[_CompactChapterDraft, ...] = Field(min_length=1, max_length=240)


class OpenAIDocumentClient(DocumentTextPort):
    """OpenAI 兼容文本模型客户端；只负责协议调用和严格响应解析。"""

    def __init__(
        self,
        http_client: httpx.Client,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        max_input_chars: int = 60_000,
        max_input_bytes: int = 1 * 1024 * 1024,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        compact_planning: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http_client = http_client
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._max_input_chars = max_input_chars
        self._max_input_bytes = max_input_bytes
        self._max_response_bytes = max_response_bytes
        if max_output_tokens < 1:
            raise ValueError("文本模型输出 token 上限必须大于 0")
        self._max_output_tokens = max_output_tokens
        self._compact_planning = compact_planning
        self._sleeper = sleeper

    def plan_chapters(
        self,
        request: ChapterPlanningRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse:
        if self._compact_planning:
            compact = self._call(
                prompt_for_compact_planning(request),
                response_type=_CompactChapterPlanningResponse,
                schema_name="chapter_planning_compact_v1",
                max_output_tokens=_COMPACT_PLANNING_MAX_OUTPUT_TOKENS,
                extra_payload={"thinking": {"type": "disabled"}},
                validate_response=lambda response: _validate_compact_planning_response(
                    response,
                    request,
                ),
                on_provider_attempt=on_provider_attempt,
            )
            return _expand_compact_planning_response(compact, request)
        return self._call(
            prompt_for_planning(request),
            response_type=ChapterPlanningResponse,
            schema_name="chapter_planning_v1",
            validate_response=lambda response: _validate_planning_response(
                response,
                allowed_segment_ids={segment.segment_id for segment in request.segments},
                allowed_transcript_ids={
                    item.evidence_id for item in request.transcript_evidence
                },
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def repair_chapter_plan(
        self,
        request: ChapterPlanRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterPlanningResponse:
        if self._compact_planning:
            compact = self._call(
                prompt_for_compact_plan_repair(request),
                response_type=_CompactChapterPlanningResponse,
                schema_name="chapter_planning_compact_repair_v1",
                max_output_tokens=_COMPACT_PLANNING_MAX_OUTPUT_TOKENS,
                extra_payload={"thinking": {"type": "disabled"}},
                validate_response=lambda response: _validate_compact_planning_response(
                    response,
                    request.request,
                ),
                on_provider_attempt=on_provider_attempt,
            )
            return _expand_compact_planning_response(compact, request.request)
        return self._call(
            prompt_for_plan_repair(request),
            response_type=ChapterPlanningResponse,
            schema_name="chapter_planning_repair_v1",
            validate_response=lambda response: _validate_planning_response(
                response,
                allowed_segment_ids=set(request.allowed_segment_ids),
                allowed_transcript_ids=set(request.allowed_transcript_ids),
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def write_chapter(
        self,
        request: ChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        return self._call(
            prompt_for_writing(request),
            response_type=ChapterWritingResponse,
            schema_name="chapter_writing_v2",
            validate_response=lambda response: _validate_writing_response(
                response,
                allowed_evidence_ids=set(allowed_writing_evidence_ids(request)),
                visual_observations=request.visual_observations,
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def repair_chapter_writing(
        self,
        request: ChapterWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterWritingResponse:
        return self._call(
            prompt_for_writing_repair(request),
            response_type=ChapterWritingResponse,
            schema_name="chapter_writing_repair_v2",
            validate_response=lambda response: _validate_writing_response(
                response,
                allowed_evidence_ids=set(request.allowed_evidence_ids),
                visual_observations=request.request.visual_observations,
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def organize_document(
        self,
        request: GlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        return self._call(
            prompt_for_global_editing(request),
            response_type=GlobalWritingResponse,
            schema_name="global_writing_v1",
            validate_response=lambda response: _validate_global_response(
                response,
                allowed_chapter_ids=set(allowed_global_chapter_ids(request)),
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def repair_global_writing(
        self,
        request: GlobalWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> GlobalWritingResponse:
        return self._call(
            prompt_for_global_repair(request),
            response_type=GlobalWritingResponse,
            schema_name="global_writing_repair_v1",
            validate_response=lambda response: _validate_global_response(
                response,
                allowed_chapter_ids=set(request.allowed_chapter_ids),
            ),
            on_provider_attempt=on_provider_attempt,
        )

    def _call(
        self,
        prompt: Prompt,
        *,
        response_type: type[ResponseModel],
        schema_name: str,
        max_output_tokens: int | None = None,
        extra_payload: dict[str, object] | None = None,
        validate_response: Callable[[ResponseModel], None],
        on_provider_attempt: Callable[[], None] | None,
    ) -> ResponseModel:
        version, instruction, data = prompt
        payload = _request_payload(
            model_id=self._model_id,
            version=version,
            instruction=instruction,
            data=data,
            response_type=response_type,
            schema_name=schema_name,
            max_output_tokens=(
                self._max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
        )
        if extra_payload:
            payload.update(extra_payload)
        raw = self._post_with_retry(payload, on_provider_attempt=on_provider_attempt)
        return _parse_and_validate_response(
            raw,
            response_type,
            validate_response=validate_response,
        )

    def _post_with_retry(
        self,
        payload: dict[str, object],
        *,
        on_provider_attempt: Callable[[], None] | None,
    ) -> bytes:
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                if on_provider_attempt is not None:
                    on_provider_attempt()
                with self._http_client.stream(
                    "POST",
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                ) as response:
                    _raise_response_status(response)
                    content = _bounded_response(response, self._max_response_bytes)
                return content
            except (httpx.RequestError, TimeoutError) as error:
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "文本模型暂时不可用",
                )
                if attempt == self._max_attempts:
                    raise last_error from error
            except VideoDemoError as error:
                if (
                    error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
                    or attempt == self._max_attempts
                ):
                    raise
                last_error = error
            if attempt < self._max_attempts:
                self._sleeper(min(2 ** (attempt - 1), 4))
        raise last_error or RuntimeError("文本模型调用状态非法")


def _request_payload(
    *,
    model_id: str,
    version: str,
    instruction: str,
    data: str,
    response_type: type[BaseModel],
    schema_name: str,
    max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, object]:
    return {
        "model": model_id,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_type.model_json_schema(),
            },
        },
        "messages": [
            {
                "role": "system",
                "content": f"PROMPT_VERSION={version}\n{instruction}",
            },
            {
                "role": "user",
                "content": "UNTRUSTED_DOCUMENT_DATA_JSON\n" + data,
            },
        ],
    }


def _bounded_response(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "文本模型响应超过大小上限")
        chunks.append(chunk)
    return b"".join(chunks)


def _raise_response_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code in {401, 403}:
        raise VideoDemoError(ErrorCode.TEXT_LLM_AUTHENTICATION_FAILED, "文本模型鉴权失败")
    if response.status_code in {408, 429} or response.status_code >= 500:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "文本模型暂时不可用")
    if response.status_code in {404, 415, 422}:
        raise VideoDemoError(
            ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE,
            "文本模型或结构化输出能力不可用",
        )
    raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "文本模型请求被拒绝")


def _parse_and_validate_response(
    content: bytes,
    response_type: type[ResponseModel],
    *,
    validate_response: Callable[[ResponseModel], None],
) -> ResponseModel:
    raw_message, parsed = _extract_model_message(content)
    try:
        response = response_type.model_validate(parsed)
        validate_response(response)
        return response
    except ValidationError as error:
        summaries = tuple(_pydantic_error_summary(item) for item in error.errors())
    except _ReferenceValidationError as error:
        summaries = (error.summary,)
    raise ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "文本模型响应结构非法",
        invalid_model_response(
            raw_message,
            summaries,
            parsed_json=parsed,
        ),
    ) from None


def _extract_model_message(content: bytes) -> tuple[bytes, object | None]:
    envelope: object | None = None
    raw: bytes | None = None
    try:
        envelope = json.loads(
            content,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        message = envelope["choices"][0]["message"]["content"]  # type: ignore[index]
        if not isinstance(message, str):
            raise ValueError
        raw = message.encode("utf-8")
        parsed = json.loads(
            message,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        return raw, parsed
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError):
        safe_envelope = envelope if raw is None else None
        raise ModelResponseValidationError(
            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
            "文本模型响应结构非法",
            invalid_model_response(
                raw if raw is not None else content,
                ("response_envelope:invalid",),
                parsed_json=safe_envelope,
            ),
        ) from None


def _pydantic_error_summary(error: object) -> str:
    assert isinstance(error, dict)
    location_value = error.get("loc", ())
    location = ".".join(str(item) for item in location_value) or "response"
    error_type = str(error.get("type", "invalid"))
    return f"{location}:{error_type}"[:500]


def _validate_input_budget(data: str, max_chars: int, max_bytes: int) -> None:
    if len(data) > max_chars or len(data.encode("utf-8")) > max_bytes:
        raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "文本模型输入超过大小上限")


def _validate_planning_response(
    response: ChapterPlanningResponse,
    *,
    allowed_segment_ids: set[str],
    allowed_transcript_ids: set[str],
) -> None:
    for draft in response.chapter_drafts:
        _require_known_ids(
            draft.segment_refs,
            allowed_segment_ids,
            "chapter_drafts.segment_refs",
        )
        for target in draft.semantic_targets:
            _require_known_ids(
                target.anchor_evidence_refs,
                allowed_transcript_ids,
                "chapter_drafts.semantic_targets.anchor_evidence_refs",
            )


def _validate_compact_planning_response(
    response: _CompactChapterPlanningResponse,
    request: ChapterPlanningRequest,
) -> None:
    segment_count = len(request.segments)
    transcript_count = len(request.transcript_evidence)
    segment_transcript_indexes = tuple(
        tuple(
            index
            for index, evidence in enumerate(request.transcript_evidence)
            if evidence.evidence_id in segment.evidence_refs
        )
        for segment in request.segments
    )
    expected_start = 0
    for draft in response.chapter_drafts:
        if draft.start_segment_index != expected_start:
            raise _ReferenceValidationError("chapter_drafts.segment_indexes:not_contiguous")
        if draft.end_segment_index > segment_count:
            raise _ReferenceValidationError("chapter_drafts.end_segment_index:out_of_range")
        if draft.end_segment_index <= draft.start_segment_index:
            raise _ReferenceValidationError("chapter_drafts.segment_indexes:empty")
        expected_start = draft.end_segment_index
        for target in draft.semantic_targets:
            if any(
                index < 0 or index >= transcript_count
                for index in target.anchor_transcript_indexes
            ):
                raise _ReferenceValidationError("semantic_targets.anchor_indexes:out_of_range")
            if len(set(target.anchor_transcript_indexes)) != len(
                target.anchor_transcript_indexes
            ):
                raise _ReferenceValidationError("semantic_targets.anchor_indexes:duplicate")
            chapter_transcript_indexes = {
                index
                for indexes in segment_transcript_indexes[
                    draft.start_segment_index : draft.end_segment_index
                ]
                for index in indexes
            }
            if not set(target.anchor_transcript_indexes) <= chapter_transcript_indexes:
                raise _ReferenceValidationError(
                    "semantic_targets.anchor_indexes:not_in_chapter",
                )
    if expected_start != segment_count:
        raise _ReferenceValidationError("chapter_drafts.segment_indexes:not_complete")


def _expand_compact_planning_response(
    response: _CompactChapterPlanningResponse,
    request: ChapterPlanningRequest,
) -> ChapterPlanningResponse:
    segment_ids = tuple(item.segment_id for item in request.segments)
    transcript_ids = tuple(item.evidence_id for item in request.transcript_evidence)
    drafts = tuple(
        ChapterDraft(
            segment_refs=segment_ids[
                draft.start_segment_index : draft.end_segment_index
            ],
            title_hint=draft.title_hint,
            visual_mode=draft.visual_mode,
            semantic_targets=tuple(
                VisualTargetDraft(
                    query_zh=target.query_zh,
                    anchor_evidence_refs=tuple(
                        transcript_ids[index]
                        for index in target.anchor_transcript_indexes
                    ),
                )
                for target in draft.semantic_targets
            ),
        )
        for draft in response.chapter_drafts
    )
    return ChapterPlanningResponse(chapter_drafts=drafts)


def _validate_writing_response(
    response: ChapterWritingResponse,
    *,
    allowed_evidence_ids: set[str],
    visual_observations: tuple[VisualObservationEvidence, ...],
) -> None:
    allowed_visual_observation_ids = {item.evidence_id for item in visual_observations}
    observation_by_id = {item.evidence_id: item for item in visual_observations}
    _require_known_ids(
        response.title_evidence_refs,
        allowed_evidence_ids,
        "title_evidence_refs",
    )
    _require_known_ids(
        response.summary_evidence_refs,
        allowed_evidence_ids,
        "summary_evidence_refs",
    )
    for block in response.body_blocks:
        _require_known_ids(block.evidence_refs, allowed_evidence_ids, "body_blocks.evidence_refs")
        if isinstance(block, VisualBlock):
            _require_known_ids(
                (block.visual_observation_ref,),
                allowed_visual_observation_ids,
                "body_blocks.visual_observation_ref",
            )
            observation = observation_by_id[block.visual_observation_ref]
            allowed_content_ids = {
                *(item.visual_content_id for item in observation.content_blocks),
                *(item.visual_fact_id for item in observation.visual_facts),
            }
            if (
                bool(allowed_content_ids) != bool(block.visual_content_refs)
                or not set(block.visual_content_refs).issubset(allowed_content_ids)
            ):
                raise _ReferenceValidationError(
                    "body_blocks.visual_content_refs:unknown_reference",
                )
            _require_known_ids(
                (block.visual_observation_ref,),
                set(block.evidence_refs),
                "body_blocks.visual_observation_ref",
            )
    for claim in response.claims:
        _require_known_ids(claim.evidence_refs, allowed_evidence_ids, "claims.evidence_refs")


def _validate_global_response(
    response: GlobalWritingResponse,
    *,
    allowed_chapter_ids: set[str],
) -> None:
    for point in response.key_points:
        _require_known_ids(point.chapter_refs, allowed_chapter_ids, "key_points.chapter_refs")
    for section in response.sections:
        _require_known_ids(section.chapter_refs, allowed_chapter_ids, "sections.chapter_refs")


class _ReferenceValidationError(ValueError):
    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def _require_known_ids(values: tuple[str, ...], allowed: set[str], field: str) -> None:
    if any(value not in allowed for value in values):
        raise _ReferenceValidationError(f"{field}:unknown_reference")
