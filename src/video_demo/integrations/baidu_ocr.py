from __future__ import annotations

import base64
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr, ValidationError

from video_demo.domain.evidence import BoundingBox, OcrLine
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.visual.ocr import OcrDeadlineExceeded, OcrProviderResponse

_LANGUAGE_MAP = {
    "zh": "CHN_ENG",
    "en": "ENG",
    "ja": "JAP",
    "ko": "KOR",
    "es": "SPA",
}


@dataclass(frozen=True, slots=True)
class BaiduOcrCredentials:
    api_key: SecretStr
    secret_key: SecretStr

    def __init__(self, *, api_key: str, secret_key: str) -> None:
        object.__setattr__(self, "api_key", SecretStr(api_key))
        object.__setattr__(self, "secret_key", SecretStr(secret_key))


class BaiduOcrClient:
    def __init__(
        self,
        client: httpx.Client,
        credentials: BaiduOcrCredentials,
        *,
        endpoint: str,
        token_endpoint: str = "https://aip.baidubce.com/oauth/2.0/token",
        clock: Callable[[], float] = time.monotonic,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds 不得小于 0")
        self._client = client
        self._credentials = credentials
        self._endpoint = endpoint
        self._token_endpoint = token_endpoint
        self._clock = clock
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleeper = sleeper
        self._access_token: SecretStr | None = None
        self._access_token_expires_at = 0.0

    def recognize(
        self,
        image: bytes,
        language: str,
        *,
        deadline: float | None = None,
    ) -> OcrProviderResponse:
        provider_language = _LANGUAGE_MAP.get(language)
        if provider_language is None:
            raise VideoDemoError(
                ErrorCode.OCR_LANGUAGE_UNSUPPORTED,
                "百度 OCR 不支持该语言",
                {"language": language},
            )
        token = self._token(deadline=deadline)
        return self._post_with_retry(
            image,
            provider_language,
            token,
            deadline=deadline,
        )

    def _post_with_retry(
        self,
        image: bytes,
        provider_language: str,
        token: SecretStr,
        *,
        deadline: float | None,
    ) -> OcrProviderResponse:
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            timeout = self._request_timeout(deadline)
            try:
                response = self._client.post(
                    self._endpoint,
                    params={"access_token": token.get_secret_value()},
                    data={
                        "image": base64.b64encode(image).decode("ascii"),
                        "language_type": provider_language,
                        "detect_direction": "true",
                        # accurate_basic 的段落结构由该参数启用；行级结果仍作为
                        # 兼容主结果解析，避免改变现有证据和 RAG 文本格式。
                        "paragraph": "true",
                        "probability": "true",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=timeout,
                )
                self._raise_if_deadline_reached(
                    deadline,
                    provider_attempt_count=attempt,
                )
                self._raise_for_status(response)
                payload = _json_object(response)
                self._raise_if_deadline_reached(
                    deadline,
                    provider_attempt_count=attempt,
                )
                _raise_for_provider_error(payload)
                response_payload = _parse_ocr_response(
                    payload,
                    http_status=response.status_code,
                )
                self._raise_if_deadline_reached(
                    deadline,
                    provider_attempt_count=attempt,
                )
                return response_payload.model_copy(
                    update={"provider_attempt_count": attempt},
                )
            except httpx.RequestError:
                self._raise_if_deadline_reached(
                    deadline,
                    provider_attempt_count=attempt,
                )
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "百度 OCR 网络请求失败",
                )
            except VideoDemoError as error:
                if error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE:
                    raise
                last_error = error
            if attempt < self._max_attempts:
                delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
                self._sleep_before_retry(
                    delay,
                    deadline=deadline,
                    provider_attempt_count=attempt,
                )
        if last_error is None:
            raise RuntimeError("百度 OCR 重试状态非法")
        self._raise_if_deadline_reached(
            deadline,
            provider_attempt_count=self._max_attempts,
        )
        raise last_error from None

    def _token(self, *, deadline: float | None) -> SecretStr:
        now = self._clock()
        if self._access_token is not None and now < self._access_token_expires_at:
            return self._access_token
        try:
            response = self._client.get(
                self._token_endpoint,
                params={
                    "grant_type": "client_credentials",
                    "client_id": self._credentials.api_key.get_secret_value(),
                    "client_secret": self._credentials.secret_key.get_secret_value(),
                },
                timeout=self._request_timeout(deadline),
            )
        except httpx.RequestError:
            self._raise_if_deadline_reached(deadline)
            raise VideoDemoError(
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                "百度 OCR Token 网络请求失败",
            ) from None
        self._raise_if_deadline_reached(deadline)
        self._raise_for_status(response)
        payload = _json_object(response)
        self._raise_if_deadline_reached(deadline)
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token:
            raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR Token 响应非法")
        if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)):
            raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR Token 有效期非法")
        try:
            expires_in_seconds = float(expires_in)
        except (OverflowError, ValueError, TypeError):
            raise VideoDemoError(
                ErrorCode.OCR_RESPONSE_INVALID,
                "百度 OCR Token 有效期非法",
            ) from None
        if not math.isfinite(expires_in_seconds) or expires_in_seconds <= 0:
            raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR Token 有效期非法")
        self._raise_if_deadline_reached(deadline)
        self._access_token = SecretStr(access_token)
        self._access_token_expires_at = now + max(0.0, expires_in_seconds - 10.0)
        return self._access_token

    def _request_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self._timeout_seconds
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise OcrDeadlineExceeded("OCR 全局截止时间已到")
        return min(self._timeout_seconds, remaining)

    def _sleep_before_retry(
        self,
        delay: float,
        *,
        deadline: float | None,
        provider_attempt_count: int,
    ) -> None:
        if deadline is not None and self._clock() + delay >= deadline:
            raise OcrDeadlineExceeded(
                "OCR 全局截止时间已到",
                provider_attempt_count=provider_attempt_count,
            )
        self._sleeper(delay)

    def _raise_if_deadline_reached(
        self,
        deadline: float | None,
        *,
        provider_attempt_count: int = 0,
    ) -> None:
        if deadline is not None and self._clock() >= deadline:
            raise OcrDeadlineExceeded(
                "OCR 全局截止时间已到",
                provider_attempt_count=provider_attempt_count,
            )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code in (401, 403):
            raise VideoDemoError(
                ErrorCode.OCR_AUTHENTICATION_FAILED,
                "百度 OCR 鉴权失败",
                {"status_code": response.status_code},
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise VideoDemoError(
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                "百度 OCR 服务暂时不可用",
                {"status_code": response.status_code},
            )
        raise VideoDemoError(
            ErrorCode.OCR_RESPONSE_INVALID,
            "百度 OCR 请求被拒绝",
            {"status_code": response.status_code},
        )


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload: object = response.json()
    except ValueError:
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 返回非法 JSON") from None
    if not isinstance(payload, dict):
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 响应必须是对象")
    return payload


def _parse_ocr_response(
    payload: dict[str, Any],
    *,
    http_status: int,
) -> OcrProviderResponse:
    request_id = payload.get("log_id")
    results = payload.get("words_result")
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 缺少 request ID")
    request_id_text = str(request_id)
    if not request_id_text or len(request_id_text) > 256:
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR request ID 非法")
    if not isinstance(results, list):
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 缺少文字结果")
    lines: list[OcrLine] = []
    try:
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("文字结果必须是对象")
            probability = result["probability"]
            if not isinstance(probability, dict):
                raise ValueError("probability 类型非法")
            words = result["words"]
            if not isinstance(words, str) or not words or len(words) > 10_000:
                raise ValueError("文字必须是受限非空字符串")
            location = result.get("location")
            bounding_box = (
                _parse_bounding_box(location)
                if location is not None
                else None
            )
            lines.append(
                OcrLine(
                    text=words,
                    bounding_box=bounding_box,
                    confidence=_number(probability["average"]),
                ),
            )
        return OcrProviderResponse(
            request_id=request_id_text,
            http_status=http_status,
            lines=tuple(lines),
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 文字结果非法") from None


def _parse_bounding_box(value: object) -> BoundingBox:
    if not isinstance(value, dict):
        raise ValueError("bbox 类型非法")
    return BoundingBox(
        x=_integer(value["left"]),
        y=_integer(value["top"]),
        width=_integer(value["width"]),
        height=_integer(value["height"]),
    )


def _raise_for_provider_error(payload: dict[str, Any]) -> None:
    value = payload.get("error_code")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 错误码非法")
    try:
        error_code = int(value)
    except ValueError:
        raise VideoDemoError(ErrorCode.OCR_RESPONSE_INVALID, "百度 OCR 错误码非法") from None
    if error_code in {100, 101, 110, 111}:
        raise VideoDemoError(ErrorCode.OCR_AUTHENTICATION_FAILED, "百度 OCR 鉴权失败")
    if error_code in {4, 17, 18, 19}:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "百度 OCR 服务暂时不可用")
    raise VideoDemoError(
        ErrorCode.OCR_RESPONSE_INVALID,
        "百度 OCR 返回供应商错误",
        {"provider_error_code": error_code},
    )


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("必须是整数")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("必须是有限数值")
    return number
