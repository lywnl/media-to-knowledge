from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from video_demo.domain.audio_plan import AudioChapterDraft
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.audio_document_port import (
    AudioChapterBoundaryCoordinationRequest,
    AudioChapterBoundaryCoordinationResponse,
    AudioChapterPlanningRequest,
    AudioChapterPlanningResponse,
    AudioChapterPlanRepairRequest,
    AudioChapterWritingRepairRequest,
    AudioChapterWritingRequest,
    AudioChapterWritingResponse,
    AudioDocumentTextPort,
    AudioGlobalWritingRepairRequest,
    AudioGlobalWritingRequest,
    AudioGlobalWritingResponse,
    AudioModelResponseValidationError,
    audio_invalid_model_response,
)
from video_demo.integrations.audio_document_prompts import (
    prompt_for_audio_boundary_coordination,
    prompt_for_audio_global,
    prompt_for_audio_global_repair,
    prompt_for_audio_plan_repair,
    prompt_for_audio_planning,
    prompt_for_audio_writing,
    prompt_for_audio_writing_repair,
)
from video_demo.integrations.model_response import (
    extract_model_message_content,
    parse_json_content,
    strip_removed_document_fields,
    unwrap_single_response_envelope,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_LOGGER = logging.getLogger(__name__)
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192
_COMPACT_PLANNING_MIN_OUTPUT_TOKENS = 1_024


class _CompactDraft(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    start_segment_index: int = Field(ge=0)
    end_segment_index: int = Field(gt=0)
    title_hint: str = Field(min_length=1, max_length=200)


class _CompactResponse(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}
    chapter_drafts: tuple[_CompactDraft, ...] = Field(min_length=1, max_length=240)


class AudioDocumentClient(AudioDocumentTextPort):
    """音频专用 OpenAI 兼容客户端。"""

    def __init__(
        self,
        http_client: httpx.Client,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        max_response_bytes: int = 2 * 1024 * 1024,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http_client
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model_id = model_id
        self._timeout = timeout_seconds
        self._attempts = max_attempts
        self._max_response_bytes = max_response_bytes
        self._sleeper = sleeper

    def plan_chapters(
        self,
        request: AudioChapterPlanningRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterPlanningResponse:
        compact = self._call(
            prompt_for_audio_planning(request),
            _CompactResponse,
            "audio_chapter_planning_v1",
            on_provider_attempt,
            max_output_tokens=_compact_planning_output_tokens(len(request.segments)),
            extra_payload={"thinking": {"type": "disabled"}},
        )
        try:
            compact = _normalize_audio_compact_response(compact, len(request.segments))
            return _expand_planning_response(compact, request)
        except VideoDemoError as error:
            raise AudioModelResponseValidationError(
                str(error),
                audio_invalid_model_response(
                    _model_dump_bytes(compact),
                    (str(error),),
                    parsed_json=compact.model_dump(mode="json"),
                ),
            ) from None

    def repair_chapter_plan(
        self,
        request: AudioChapterPlanRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterPlanningResponse:
        compact = self._call(
            prompt_for_audio_plan_repair(request),
            _CompactResponse,
            "audio_chapter_planning_repair_v1",
            on_provider_attempt,
            max_output_tokens=_compact_planning_output_tokens(len(request.request.segments)),
            extra_payload={"thinking": {"type": "disabled"}},
        )
        try:
            compact = _normalize_audio_compact_response(
                compact,
                len(request.request.segments),
            )
            return _expand_planning_response(compact, request.request)
        except VideoDemoError as error:
            raise AudioModelResponseValidationError(
                str(error),
                audio_invalid_model_response(
                    _model_dump_bytes(compact),
                    (str(error),),
                    parsed_json=compact.model_dump(mode="json"),
                ),
            ) from None

    def coordinate_chapter_boundaries(
        self,
        request: AudioChapterBoundaryCoordinationRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterBoundaryCoordinationResponse:
        return self._call(
            prompt_for_audio_boundary_coordination(request),
            AudioChapterBoundaryCoordinationResponse,
            "audio_chapter_boundary_coordination_v1",
            on_provider_attempt,
            extra_payload={"thinking": {"type": "disabled"}},
        )

    def write_chapter(
        self,
        request: AudioChapterWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterWritingResponse:
        return self._call(
            prompt_for_audio_writing(request),
            AudioChapterWritingResponse,
            "audio_chapter_writing_v1",
            on_provider_attempt,
            extra_payload={"thinking": {"type": "disabled"}},
        )

    def repair_chapter_writing(
        self,
        request: AudioChapterWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioChapterWritingResponse:
        return self._call(
            prompt_for_audio_writing_repair(request),
            AudioChapterWritingResponse,
            "audio_chapter_writing_repair_v1",
            on_provider_attempt,
            extra_payload={"thinking": {"type": "disabled"}},
        )

    def organize_document(
        self,
        request: AudioGlobalWritingRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioGlobalWritingResponse:
        return self._call(
            prompt_for_audio_global(request),
            AudioGlobalWritingResponse,
            "audio_global_writing_v1",
            on_provider_attempt,
            extra_payload={"thinking": {"type": "disabled"}},
        )

    def repair_global_writing(
        self,
        request: AudioGlobalWritingRepairRequest,
        *,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> AudioGlobalWritingResponse:
        return self._call(
            prompt_for_audio_global_repair(request),
            AudioGlobalWritingResponse,
            "audio_global_writing_repair_v1",
            on_provider_attempt,
            extra_payload={"thinking": {"type": "disabled"}},
        )

    def _call(
        self,
        prompt: tuple[str, str, str],
        response_type: type[ResponseModel],
        schema_name: str,
        callback: Callable[[], None] | None,
        *,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        extra_payload: dict[str, object] | None = None,
    ) -> ResponseModel:
        version, instruction, data = prompt
        payload = {
            "model": self._model_id,
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
                {"role": "system", "content": f"PROMPT_VERSION={version}\n{instruction}"},
                {"role": "user", "content": "UNTRUSTED_AUDIO_CONTEXT_JSON\n" + data},
            ],
        }
        if extra_payload:
            payload.update(extra_payload)
        raw = self._post(payload, callback)
        return _parse_audio_response(raw, response_type)

    def _post(self, payload: dict[str, object], callback: Callable[[], None] | None) -> bytes:
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._attempts + 1):
            started_at = time.monotonic()
            status_code: int | None = None
            try:
                if callback:
                    callback()
                with self._http.stream(
                    "POST",
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout,
                ) as response:
                    status_code = response.status_code
                    _raise_response_status(response)
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self._max_response_bytes:
                            raise VideoDemoError(
                                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                                "音频文本模型响应超过大小上限",
                            )
                        chunks.append(chunk)
                    content = b"".join(chunks)
                _LOGGER.info(
                    "音频文本模型请求成功: schema=%s attempt=%d status=%s "
                    "response_bytes=%d elapsed=%.3fs",
                    _schema_name(payload),
                    attempt,
                    status_code,
                    len(content),
                    time.monotonic() - started_at,
                )
                return content
            except (httpx.RequestError, TimeoutError) as error:
                _LOGGER.warning(
                    "音频文本模型请求失败: attempt=%d status=%s category=network elapsed=%.3fs",
                    attempt,
                    status_code,
                    time.monotonic() - started_at,
                )
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "文本模型暂时不可用"
                )
                if attempt == self._attempts:
                    raise last_error from error
            except VideoDemoError as error:
                _LOGGER.warning(
                    "音频文本模型请求失败: attempt=%d status=%s code=%s elapsed=%.3fs",
                    attempt,
                    status_code,
                    error.code.value,
                    time.monotonic() - started_at,
                )
                if (
                    error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
                    or attempt == self._attempts
                ):
                    raise
                last_error = error
            self._sleeper(min(2 ** (attempt - 1), 4))
        raise last_error or RuntimeError("音频模型调用状态非法")


def _schema_name(payload: dict[str, object]) -> str:
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return "unknown"
    schema = response_format.get("json_schema")
    if not isinstance(schema, dict):
        return "json_object"
    name = schema.get("name")
    return name if isinstance(name, str) and name else "unknown"


def _parse_audio_response(raw: bytes, response_type: type[ResponseModel]) -> ResponseModel:
    raw_message, parsed, finish_reason = _extract_audio_model_message(raw)
    try:
        normalized = _normalize_audio_response_payload(
            unwrap_single_response_envelope(strip_removed_document_fields(parsed)),
            response_type,
        )
        return response_type.model_validate(normalized)
    except ValidationError as error:
        summaries = tuple(_pydantic_error_summary(item) for item in error.errors())
        _LOGGER.warning(
            "音频文本模型响应校验失败 finish_reason=%s response_bytes=%d "
            "validation_errors=%s",
            finish_reason,
            len(raw_message),
            ",".join(summaries[:8]),
        )
        raise AudioModelResponseValidationError(
            "音频文本模型响应结构非法",
            audio_invalid_model_response(
                raw_message,
                summaries,
                parsed_json=parsed,
            ),
        ) from None


def _extract_audio_model_message(
    content: bytes,
) -> tuple[bytes, object, str]:
    envelope: object | None = None
    raw_message: bytes | None = None
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
            value = choices[0].get("finish_reason")
            if isinstance(value, str):
                normalized = " ".join(value.split())
                if normalized and len(normalized) <= 64 and all(
                    character.isprintable() for character in normalized
                ):
                    finish_reason = normalized
        message = extract_model_message_content(envelope)
        if not message:
            raise ValueError
        raw_message = message.encode("utf-8")
        parsed = parse_json_content(message)
        return raw_message, parsed, finish_reason
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError) as error:
        _LOGGER.warning(
            "音频文本模型响应解析失败 finish_reason=%s response_bytes=%d "
            "validation_errors=response_envelope:invalid",
            finish_reason,
            len(raw_message) if raw_message is not None else len(content),
        )
        raise AudioModelResponseValidationError(
            "音频文本模型响应结构非法",
            audio_invalid_model_response(
                raw_message if raw_message is not None else content,
                ("response_envelope:invalid",),
                parsed_json=envelope if raw_message is None else None,
            ),
        ) from error


def _normalize_audio_compact_response(
    response: _CompactResponse,
    segment_count: int,
) -> _CompactResponse:
    """裁掉尾部空草稿，归一化边界并验证范围完整覆盖当前批次。"""

    drafts = list(response.chapter_drafts)
    while len(drafts) > 1 and drafts[-1].start_segment_index == drafts[-1].end_segment_index:
        drafts.pop()
    expected_start = 0
    for index, draft in enumerate(drafts):
        if draft.end_segment_index > segment_count:
            if index != len(drafts) - 1 or draft.end_segment_index != segment_count + 1:
                raise VideoDemoError(
                    ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                    "音频章节片段范围越界",
                )
            drafts[index] = draft.model_copy(update={"end_segment_index": segment_count})
            draft = drafts[index]
        if draft.start_segment_index >= draft.end_segment_index:
            raise VideoDemoError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                "音频章节片段范围为空",
            )
        if draft.start_segment_index != expected_start:
            raise VideoDemoError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID,
                "音频章节片段范围不连续",
            )
        expected_start = draft.end_segment_index
    if expected_start != segment_count:
        raise VideoDemoError(
            ErrorCode.TEXT_LLM_RESPONSE_INVALID,
            "音频章节未完整覆盖片段",
        )
    normalized = tuple(drafts)
    if normalized == response.chapter_drafts:
        return response
    return _CompactResponse(chapter_drafts=normalized)


def _normalize_audio_response_payload(
    value: object,
    response_type: type[ResponseModel],
) -> object:
    """在音频适配器边界兼容常见包装、数组和字段别名。"""

    if isinstance(value, list):
        if response_type is _CompactResponse:
            return _normalize_audio_compact_payload({"chapter_drafts": value})
        if response_type is AudioChapterPlanningResponse:
            return {"chapter_drafts": value}
        if response_type is AudioChapterBoundaryCoordinationResponse:
            return {"decisions": value}
        return value
    if not isinstance(value, dict):
        return value
    if response_type is _CompactResponse:
        return _normalize_audio_compact_payload(value)
    if response_type is AudioChapterBoundaryCoordinationResponse:
        return _normalize_audio_boundary_payload(value)
    if response_type is AudioChapterWritingResponse:
        normalized = dict(value)
        normalized.setdefault("body_blocks", [])
        normalized.setdefault("claims", [])
        return normalized
    return value


def _normalize_audio_compact_payload(value: dict[str, object]) -> dict[str, object]:
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
            for key in ("start_segment_index", "end_segment_index", "title_hint")
            if key in raw_draft
        }
        if "title_hint" not in draft and isinstance(raw_draft.get("title"), str):
            draft["title_hint"] = raw_draft["title"]
        for key in ("start_segment_index", "end_segment_index"):
            if isinstance(draft.get(key), str) and draft[key].strip().lstrip("-").isdigit():
                draft[key] = int(draft[key])
        drafts.append(draft)
    return {"chapter_drafts": drafts}


def _normalize_audio_boundary_payload(value: dict[str, object]) -> dict[str, object]:
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
            and item["boundary_index"].strip().lstrip("-").isdigit()
        ):
            item["boundary_index"] = int(item["boundary_index"])
        normalized.append(item)
    return {"decisions": normalized}


def _pydantic_error_summary(error: object) -> str:
    if not isinstance(error, dict):
        return "response:invalid"
    location = ".".join(str(item) for item in error.get("loc", ())) or "response"
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
    """只把固定的业务校验原因带入音频修复上下文，避免回显模型内容。"""

    if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
        return False
    lowered = value.lower()
    return not any(
        marker in lowered
        for marker in ("http://", "https://", "data:", "bearer ", "api_key", "token")
    )


def _model_dump_bytes(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _compact_planning_output_tokens(segment_count: int) -> int:
    if segment_count < 1:
        raise ValueError("音频章节规划批次至少包含一个片段")
    return min(
        _DEFAULT_MAX_OUTPUT_TOKENS,
        max(_COMPACT_PLANNING_MIN_OUTPUT_TOKENS, segment_count * 192),
    )


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


def _expand_planning_response(
    response: _CompactResponse,
    request: AudioChapterPlanningRequest,
) -> AudioChapterPlanningResponse:
    segment_ids = tuple(item.segment_id for item in request.segments)
    return AudioChapterPlanningResponse(
        chapter_drafts=tuple(
            AudioChapterDraft(
                segment_refs=segment_ids[
                    draft.start_segment_index : draft.end_segment_index
                ],
                title_hint=draft.title_hint,
            )
            for draft in response.chapter_drafts
        ),
    )
