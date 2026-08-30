from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from video_demo.domain.document import VisualBlock
from video_demo.domain.document_plan import ChapterDraft, VisualTargetDraft
from video_demo.domain.evidence import SpeechSegment, SubtitleCue, VisualObservationEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterBoundaryCoordinationRequest,
    ChapterBoundaryCoordinationResponse,
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
    prompt_for_boundary_coordination,
    prompt_for_compact_plan_repair,
    prompt_for_compact_planning,
    prompt_for_global_editing,
    prompt_for_global_repair,
    prompt_for_plan_repair,
    prompt_for_planning,
    prompt_for_writing,
    prompt_for_writing_repair,
)
from video_demo.integrations.document_writing_normalization import (
    normalize_optional_visual_blocks,
)
from video_demo.integrations.model_response import (
    extract_model_message_content,
    parse_json_content,
    strip_removed_document_fields,
    unwrap_single_response_envelope,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
Prompt = tuple[str, str, str]
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
# 紧凑索引协议关闭 thinking 以减少延迟；保留 8192 上限，避免长批次的
# 结构化 JSON 因输出预算过小被截断并触发额外修复调用。
_COMPACT_PLANNING_MAX_OUTPUT_TOKENS = 8_192
_COMPACT_PLANNING_MIN_OUTPUT_TOKENS = 1_024
_COMPACT_MAX_TARGET_ANCHOR_SPAN_MS = 30_000
_LOGGER = logging.getLogger(__name__)


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
                max_output_tokens=_compact_planning_output_tokens(len(request.segments)),
                extra_payload={"thinking": {"type": "disabled"}},
                validate_response=lambda response: _validate_compact_planning_response(
                    response,
                    request,
                ),
                on_provider_attempt=on_provider_attempt,
            )
            compact = _normalize_compact_planning_response(compact, len(request.segments))
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
                max_output_tokens=_compact_planning_output_tokens(
                    len(request.request.segments),
                ),
                extra_payload={"thinking": {"type": "disabled"}},
                validate_response=lambda response: _validate_compact_planning_response(
                    response,
                    request.request,
                ),
                on_provider_attempt=on_provider_attempt,
            )
            compact = _normalize_compact_planning_response(
                compact,
                len(request.request.segments),
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

    def coordinate_chapter_boundaries(
        self,
        request: ChapterBoundaryCoordinationRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterBoundaryCoordinationResponse:
        result = self._call(
            prompt_for_boundary_coordination(request),
            response_type=ChapterBoundaryCoordinationResponse,
            schema_name="chapter_boundary_coordination_v1",
            extra_payload={"thinking": {"type": "disabled"}},
            validate_response=lambda _response: None,
            on_provider_attempt=on_provider_attempt,
        )
        return result

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
            extra_payload={"thinking": {"type": "disabled"}},
            validate_response=lambda response: _validate_writing_response(
                response,
                allowed_evidence_ids=set(allowed_writing_evidence_ids(request)),
                visual_observations=request.visual_observations,
            ),
            normalize_response=lambda response: normalize_optional_visual_blocks(
                response,
                request.visual_observations,
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
            extra_payload={"thinking": {"type": "disabled"}},
            validate_response=lambda response: _validate_writing_response(
                response,
                allowed_evidence_ids=set(request.allowed_evidence_ids),
                visual_observations=request.request.visual_observations,
            ),
            normalize_response=lambda response: normalize_optional_visual_blocks(
                response,
                request.request.visual_observations,
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
            extra_payload={"thinking": {"type": "disabled"}},
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
            extra_payload={"thinking": {"type": "disabled"}},
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
        normalize_response: Callable[[ResponseModel], ResponseModel] | None = None,
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
            normalize_response=normalize_response,
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


def _compact_planning_output_tokens(segment_count: int) -> int:
    """按批次规模给紧凑规划分配输出预算，避免小批次预留 8192 token。"""

    if segment_count < 1:
        raise ValueError("章节规划批次至少包含一个基础片段")
    return min(
        _COMPACT_PLANNING_MAX_OUTPUT_TOKENS,
        max(_COMPACT_PLANNING_MIN_OUTPUT_TOKENS, segment_count * 192),
    )


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
    normalize_response: Callable[[ResponseModel], ResponseModel] | None = None,
    validate_response: Callable[[ResponseModel], None],
) -> ResponseModel:
    raw_message, parsed, finish_reason = _extract_model_message(content)
    parsed = _normalize_response_payload(
        unwrap_single_response_envelope(strip_removed_document_fields(parsed)),
        response_type,
    )
    try:
        response = response_type.model_validate(parsed)
        if normalize_response is not None:
            response = normalize_response(response)
        validate_response(response)
        return response
    except ValidationError as error:
        summaries = tuple(_pydantic_error_summary(item) for item in error.errors())
    except _ReferenceValidationError as error:
        summaries = (error.summary,)
    _LOGGER.warning(
        "文本模型响应校验失败 finish_reason=%s response_bytes=%d validation_errors=%s",
        finish_reason,
        len(raw_message),
        ",".join(summaries[:8]),
    )
    raise ModelResponseValidationError(
        ErrorCode.TEXT_LLM_RESPONSE_INVALID,
        "文本模型响应结构非法",
        invalid_model_response(
            raw_message,
            summaries,
            parsed_json=parsed,
        ),
    ) from None


def _normalize_response_payload(value: object, response_type: type[BaseModel]) -> object:
    """在适配器边界收敛供应商常见的轻微格式偏差。"""

    if isinstance(value, list):
        if response_type in {_CompactChapterPlanningResponse, ChapterPlanningResponse}:
            return {"chapter_drafts": value}
        if response_type is ChapterBoundaryCoordinationResponse:
            return {"decisions": value}
        return value
    if not isinstance(value, dict):
        return value
    if response_type is _CompactChapterPlanningResponse:
        return _normalize_compact_payload(value)
    if response_type is ChapterPlanningResponse:
        return _normalize_planning_payload(value)
    if response_type is ChapterBoundaryCoordinationResponse:
        return _normalize_boundary_payload(value)
    if response_type is ChapterWritingResponse:
        return _normalize_writing_payload(value)
    return value


def _normalize_planning_payload(value: dict[str, object]) -> dict[str, object]:
    raw_drafts = value.get("chapter_drafts")
    if not isinstance(raw_drafts, list):
        return value
    drafts: list[object] = []
    for raw_draft in raw_drafts:
        if not isinstance(raw_draft, dict):
            drafts.append(raw_draft)
            continue
        draft = {
            key: raw_draft[key]
            for key in ("segment_refs", "title_hint", "visual_mode", "semantic_targets")
            if key in raw_draft
        }
        if "title_hint" not in draft and isinstance(raw_draft.get("title"), str):
            draft["title_hint"] = raw_draft["title"]
        draft.setdefault("visual_mode", "NONE")
        draft.setdefault("semantic_targets", [])
        targets = draft.get("semantic_targets")
        if isinstance(targets, list):
            draft["semantic_targets"] = [
                _normalize_planning_target(target) for target in targets
            ]
        drafts.append(draft)
    return {"chapter_drafts": drafts}


def _normalize_planning_target(value: object) -> object:
    if not isinstance(value, dict):
        return value
    target = {
        key: value[key]
        for key in ("query_zh", "anchor_evidence_refs")
        if key in value
    }
    if "query_zh" not in target and isinstance(value.get("query"), str):
        target["query_zh"] = value["query"]
    if "anchor_evidence_refs" not in target:
        refs = value.get("anchor_refs")
        if isinstance(refs, list):
            target["anchor_evidence_refs"] = refs
    return target


def _normalize_compact_payload(value: dict[str, object]) -> dict[str, object]:
    raw_drafts = value.get("chapter_drafts")
    if not isinstance(raw_drafts, list):
        return value
    drafts: list[object] = []
    for raw_draft in raw_drafts:
        if not isinstance(raw_draft, dict):
            drafts.append(raw_draft)
            continue
        draft = {
            key: raw_draft[key]
            for key in (
                "start_segment_index",
                "end_segment_index",
                "title_hint",
                "visual_mode",
                "semantic_targets",
            )
            if key in raw_draft
        }
        if "title_hint" not in draft and isinstance(raw_draft.get("title"), str):
            draft["title_hint"] = raw_draft["title"]
        draft.setdefault("visual_mode", "NONE")
        draft.setdefault("semantic_targets", [])
        for key in ("start_segment_index", "end_segment_index"):
            if isinstance(draft.get(key), str) and draft[key].strip().lstrip("-").isdigit():
                draft[key] = int(draft[key])
        targets = draft.get("semantic_targets")
        if isinstance(targets, list):
            draft["semantic_targets"] = [
                _normalize_compact_target(target) for target in targets
            ]
        drafts.append(draft)
    return {"chapter_drafts": drafts}


def _normalize_compact_target(value: object) -> object:
    if not isinstance(value, dict):
        return value
    target = {
        key: value[key]
        for key in ("query_zh", "anchor_transcript_indexes")
        if key in value
    }
    if "query_zh" not in target and isinstance(value.get("query"), str):
        target["query_zh"] = value["query"]
    if "anchor_transcript_indexes" not in target:
        indexes = value.get("anchor_indexes")
        if isinstance(indexes, list):
            target["anchor_transcript_indexes"] = indexes
    indexes = target.get("anchor_transcript_indexes")
    if isinstance(indexes, list):
        target["anchor_transcript_indexes"] = [
            int(item) if isinstance(item, str) and item.strip().lstrip("-").isdigit() else item
            for item in indexes
        ]
    return target


def _normalize_writing_payload(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    normalized.setdefault("body_blocks", [])
    normalized.setdefault("claims", [])
    return normalized


def _normalize_boundary_payload(value: dict[str, object]) -> dict[str, object]:
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        return value
    normalized: list[object] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            normalized.append(decision)
            continue
        item = {
            key: decision[key]
            for key in ("boundary_index", "decision", "merged_title_hint")
            if key in decision
        }
        if (
            isinstance(item.get("boundary_index"), str)
            and item["boundary_index"].strip().isdigit()
        ):
            item["boundary_index"] = int(item["boundary_index"])
        normalized.append(item)
    return {"decisions": normalized}


def _extract_model_message(content: bytes) -> tuple[bytes, object | None, str]:
    envelope: object | None = None
    raw: bytes | None = None
    finish_reason = "unknown"
    try:
        envelope = json.loads(
            content,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(envelope, dict):
            raise ValueError
        choices = envelope.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = _safe_finish_reason(choices[0].get("finish_reason"))
        message = extract_model_message_content(envelope)
        if not message:
            raise ValueError
        raw = message.encode("utf-8")
        parsed = parse_json_content(message)
        return raw, parsed, finish_reason
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError):
        safe_envelope = envelope if raw is None else None
        _LOGGER.warning(
            "文本模型响应解析失败 finish_reason=%s response_bytes=%d "
            "validation_errors=response_envelope:invalid",
            finish_reason,
            len(raw) if raw is not None else len(content),
        )
        raise ModelResponseValidationError(
            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
            "文本模型响应结构非法",
            invalid_model_response(
                raw if raw is not None else content,
                ("response_envelope:invalid",),
                parsed_json=safe_envelope,
            ),
        ) from None


def _safe_finish_reason(value: object) -> str:
    if isinstance(value, str):
        normalized = " ".join(value.split())
        if normalized and len(normalized) <= 64 and all(
            character.isprintable() for character in normalized
        ):
            return normalized
    return "unknown"


def _pydantic_error_summary(error: object) -> str:
    assert isinstance(error, dict)
    location_value = error.get("loc", ())
    location = ".".join(str(item) for item in location_value) or "response"
    error_type = str(error.get("type", "invalid"))
    summary = f"{location}:{error_type}"
    if error_type == "value_error":
        context = error.get("ctx")
        reason = context.get("error") if isinstance(context, dict) else None
        reason_text = str(reason).strip() if reason is not None else ""
        if _is_safe_validation_reason(reason_text):
            summary += f":{reason_text}"
    return summary[:500]


def _is_safe_validation_reason(value: str) -> bool:
    """只把固定的业务校验原因带入修复上下文，避免回显模型内容。"""

    if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
        return False
    lowered = value.lower()
    return not any(
        marker in lowered
        for marker in ("http://", "https://", "data:", "bearer ", "api_key", "token")
    )


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
    expected_start = 0
    drafts = _normalize_compact_planning_response(response, segment_count).chapter_drafts
    for draft in drafts:
        if draft.start_segment_index != expected_start:
            raise _ReferenceValidationError("chapter_drafts.segment_indexes:not_contiguous")
        if draft.end_segment_index <= draft.start_segment_index:
            raise _ReferenceValidationError("chapter_drafts.segment_indexes:empty")
        expected_start = draft.end_segment_index
        for target in draft.semantic_targets:
            if any(
                index < 0 or index >= transcript_count
                for index in target.anchor_transcript_indexes
            ):
                continue
            if len(set(target.anchor_transcript_indexes)) != len(
                target.anchor_transcript_indexes
            ):
                raise _ReferenceValidationError("semantic_targets.anchor_indexes:duplicate")
            # 章节范围是模型最容易出错的索引映射。该类错误不会影响章节的
            # 时间分区，展开时会丢弃越界目标并把视觉模式降为纯文本，避免
            # 为可确定修复的单个目标再发起一次付费结构修复调用。
    if expected_start != segment_count:
        raise _ReferenceValidationError("chapter_drafts.segment_indexes:not_complete")


def _normalize_compact_planning_response(
    response: _CompactChapterPlanningResponse,
    segment_count: int,
) -> _CompactChapterPlanningResponse:
    """只把最后一个章节的一位越界结束下标归一化为批次末端。"""

    drafts = list(_trim_trailing_empty_drafts(response).chapter_drafts)
    for index, draft in enumerate(drafts):
        if draft.end_segment_index <= segment_count:
            continue
        if (
            index == len(drafts) - 1
            and draft.end_segment_index == segment_count + 1
        ):
            drafts[index] = draft.model_copy(update={"end_segment_index": segment_count})
            continue
        raise _ReferenceValidationError("chapter_drafts.end_segment_index:out_of_range")
    normalized = tuple(drafts)
    if normalized == response.chapter_drafts:
        return response
    return _CompactChapterPlanningResponse(chapter_drafts=normalized)


def _trim_trailing_empty_drafts(
    response: _CompactChapterPlanningResponse,
) -> _CompactChapterPlanningResponse:
    drafts = tuple(response.chapter_drafts)
    while len(drafts) > 1 and drafts[-1].start_segment_index == drafts[-1].end_segment_index:
        drafts = drafts[:-1]
    if drafts == response.chapter_drafts:
        return response
    return _CompactChapterPlanningResponse(chapter_drafts=drafts)


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
            visual_mode=_expanded_visual_mode(draft, request),
            semantic_targets=_expanded_semantic_targets(draft, request, transcript_ids),
        )
        for draft in response.chapter_drafts
    )
    return ChapterPlanningResponse(chapter_drafts=drafts)


def _expanded_semantic_targets(
    draft: _CompactChapterDraft,
    request: ChapterPlanningRequest,
    transcript_ids: tuple[str, ...],
) -> tuple[VisualTargetDraft, ...]:
    segment_transcript_indexes = {
        index
        for segment in request.segments[draft.start_segment_index : draft.end_segment_index]
        for index, evidence in enumerate(request.transcript_evidence)
        if evidence.evidence_id in segment.evidence_refs
    }
    evidence = request.transcript_evidence
    expanded: list[VisualTargetDraft] = []
    for target in draft.semantic_targets:
        indexes = target.anchor_transcript_indexes
        if not _compact_target_is_valid(indexes, segment_transcript_indexes, evidence):
            continue
        expanded.append(
            VisualTargetDraft(
                query_zh=target.query_zh,
                anchor_evidence_refs=tuple(transcript_ids[index] for index in indexes),
            ),
        )
    return tuple(expanded)


def _compact_target_is_valid(
    indexes: tuple[int, ...],
    allowed_indexes: set[int],
    evidence: tuple[SpeechSegment | SubtitleCue, ...],
) -> bool:
    """丢弃不影响章节边界的视觉锚点错误，避免为整批章节付费修复。"""

    if (
        not indexes
        or len(indexes) != len(set(indexes))
        or any(index < 0 or index >= len(evidence) for index in indexes)
        or not set(indexes) <= allowed_indexes
    ):
        return False
    anchors = tuple(evidence[index] for index in indexes)
    if anchors != tuple(
        sorted(anchors, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id)),
    ):
        return False
    return anchors[-1].end_ms - anchors[0].start_ms <= _COMPACT_MAX_TARGET_ANCHOR_SPAN_MS


def _expanded_visual_mode(
    draft: _CompactChapterDraft,
    request: ChapterPlanningRequest,
) -> Literal["NONE", "SINGLE", "COMPARISON", "MULTI_STEP"]:
    valid_target_count = len(
        _expanded_semantic_targets(
            draft,
            request,
            tuple(item.evidence_id for item in request.transcript_evidence),
        )
    )
    if valid_target_count == 0:
        return "NONE"
    if draft.visual_mode in {"COMPARISON", "MULTI_STEP"} and valid_target_count < 2:
        return "SINGLE"
    return draft.visual_mode


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
    if not allowed_chapter_ids:
        raise _ReferenceValidationError("chapters:empty")


class _ReferenceValidationError(ValueError):
    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def _require_known_ids(values: tuple[str, ...], allowed: set[str], field: str) -> None:
    if any(value not in allowed for value in values):
        raise _ReferenceValidationError(f"{field}:unknown_reference")
