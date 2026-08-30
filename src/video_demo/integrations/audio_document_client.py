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
)
from video_demo.integrations.audio_document_prompts import (
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
        return _expand_planning_response(compact, request)

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
        return _expand_planning_response(compact, request.request)

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
        try:
            raw = self._post(payload, callback)
        except VideoDemoError as error:
            if error.code != ErrorCode.TEXT_LLM_CAPABILITY_UNAVAILABLE:
                raise
            # 部分 OpenAI 兼容代理不接受 json_schema；退到 json_object 后仍由
            # 本地宽松 JSON 解析和音频业务校验收口，不把一次能力差异变成章节兜底。
            _LOGGER.warning(
                "音频文本模型不支持 json_schema，改用 json_object: schema=%s",
                schema_name,
            )
            payload["response_format"] = {"type": "json_object"}
            raw = self._post(payload, callback)
        try:
            envelope = json.loads(raw)
            message = extract_model_message_content(envelope)
            return response_type.model_validate(
                strip_removed_document_fields(parse_json_content(message)),
            )
        except (ValueError, TypeError, KeyError, ValidationError, json.JSONDecodeError) as error:
            raise VideoDemoError(
                ErrorCode.TEXT_LLM_RESPONSE_INVALID, "音频文本模型响应结构非法"
            ) from error

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
    drafts = []
    expected_start = 0
    for draft in response.chapter_drafts:
        if draft.end_segment_index > len(segment_ids):
            if (
                draft is response.chapter_drafts[-1]
                and draft.end_segment_index == len(segment_ids) + 1
            ):
                end = len(segment_ids)
            else:
                raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "音频章节片段范围越界")
        else:
            end = draft.end_segment_index
        if draft.start_segment_index >= end:
            raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "音频章节片段范围为空")
        if draft.start_segment_index != expected_start:
            raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "音频章节片段范围不连续")
        drafts.append(
            AudioChapterDraft(
                segment_refs=segment_ids[draft.start_segment_index : end],
                title_hint=draft.title_hint,
            ),
        )
        expected_start = end
    if sum(len(item.segment_refs) for item in drafts) != len(segment_ids):
        raise VideoDemoError(ErrorCode.TEXT_LLM_RESPONSE_INVALID, "音频章节未完整覆盖片段")
    return AudioChapterPlanningResponse(chapter_drafts=tuple(drafts))
