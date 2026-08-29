from __future__ import annotations

import json
from collections.abc import Callable

import httpx
from pydantic import ValidationError

from video_demo.domain.image_document import ImageDocument
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.model_response import (
    extract_model_message_content,
    parse_json_content,
    strip_removed_document_fields,
)


class ImageVlmClient:
    """面向单张图片的 Qwen VLM 客户端；协议失败不会影响视频链路。"""

    def __init__(
        self,
        http_client: httpx.Client,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
        max_response_bytes: int = 2 * 1024 * 1024,
        sleeper: Callable[[float], None] = lambda _seconds: None,
    ) -> None:
        self._http = http_client
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._max_response_bytes = max_response_bytes
        self._sleeper = sleeper

    def analyze(self, *, image_data_url: str, title_hint: str) -> ImageDocument:
        payload = {
            "model": self._model_id,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "image_understanding_v1",
                    "strict": True,
                    "schema": ImageDocument.model_json_schema(),
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "只根据输入图片生成中文图片文档。不得猜测图片不可见信息；"
                        "只返回 JSON，title 应简洁概括图片主题，overview_zh 为核心概览，"
                        "content_blocks 为图片内容，claims 为关键结论。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"图片标题提示：{title_hint}"},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        }
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._http.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                if response.status_code in {401, 403}:
                    raise VideoDemoError(ErrorCode.QWEN_AUTHENTICATION_FAILED, "图片模型鉴权失败")
                if response.status_code == 429 or response.status_code >= 500:
                    raise VideoDemoError(
                        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                        "图片模型暂时不可用",
                    )
                if response.status_code >= 400:
                    raise VideoDemoError(ErrorCode.IMAGE_VLM_UNAVAILABLE, "图片模型请求被拒绝")
                content = response.content
                if len(content) > self._max_response_bytes:
                    raise VideoDemoError(
                        ErrorCode.QWEN_RESPONSE_INVALID,
                        "图片模型响应超过大小上限",
                    )
                return _parse_image_document(content)
            except httpx.RequestError as error:
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "图片模型网络请求失败",
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
            self._sleeper(min(2 ** (attempt - 1), 4))
        raise last_error or VideoDemoError(ErrorCode.IMAGE_VLM_UNAVAILABLE, "图片模型调用失败")


def _parse_image_document(content: bytes) -> ImageDocument:
    try:
        envelope = json.loads(content)
        message = extract_model_message_content(envelope)
        parsed = strip_removed_document_fields(parse_json_content(message))
        return ImageDocument.model_validate(parsed)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
        raise VideoDemoError(ErrorCode.IMAGE_VLM_UNAVAILABLE, "图片模型响应结构非法") from error
