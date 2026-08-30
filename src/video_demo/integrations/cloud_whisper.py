from __future__ import annotations

import datetime
import json
import logging
import math
import re
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from video_demo.config import CloudAsrConfiguration
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.asr_contracts import RawAsrSegment, WindowTranscriptionResult
from video_demo.storage.workspace import reject_symlink_components

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_DEFAULT_MAX_UPLOAD_BYTES = _MAX_UPLOAD_BYTES
_PROVIDER_ERROR_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")
_LOGGER = logging.getLogger(__name__)

# OpenAI Whisper 的官方语言名称映射。项目响应只保留稳定语言代码。
_LANGUAGES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bashkir": "ba",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "breton": "br",
    "bulgarian": "bg",
    "burmese": "my",
    "castilian": "es",
    "catalan": "ca",
    "cantonese": "yue",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "faroese": "fo",
    "finnish": "fi",
    "flemish": "nl",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian": "ht",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jw",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "letzeburgesch": "lb",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "mandarin": "zh",
    "maori": "mi",
    "marathi": "mr",
    "moldavian": "ro",
    "moldovan": "ro",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "occitan": "oc",
    "panjabi": "pa",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "pushto": "ps",
    "romanian": "ro",
    "russian": "ru",
    "sanskrit": "sa",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "sinhalese": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tibetan": "bo",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "valencian": "ca",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
}


class _TemporaryModelBranchExhaustion(VideoDemoError):
    pass


class CloudWhisperClient:
    """同步 OpenAI 兼容 Whisper 客户端；不拥有传入的 HTTP Client。"""

    def __init__(
        self,
        http_client: httpx.Client,
        configuration: CloudAsrConfiguration,
        *,
        allowed_audio_root: Path,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http_client = http_client
        self._configuration = configuration
        self._allowed_audio_root = allowed_audio_root.resolve(strict=False)
        self._sleeper = sleeper

    def transcribe_window(
        self,
        audio_slice: Path,
        *,
        language_hint: str | None,
        prompt: str | None,
    ) -> WindowTranscriptionResult:
        path = _validated_audio_path(
            self._allowed_audio_root,
            audio_slice,
            max_upload_bytes=self._configuration.max_upload_bytes,
        )
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._configuration.max_attempts + 1):
            started_at = time.monotonic()
            retry_after_seconds: float | None = None
            status_code: int | None = None
            try:
                with path.open("rb") as stream, self._http_client.stream(
                    "POST",
                    f"{self._configuration.base_url}/audio/transcriptions",
                    headers={
                        "Authorization": (
                            "Bearer " + self._configuration.api_key.get_secret_value()
                        ),
                    },
                    data=_multipart_fields(
                        self._configuration.model,
                        language_hint=language_hint,
                        prompt=prompt,
                    ),
                    files={"file": (path.name, stream, "audio/wav")},
                    timeout=self._configuration.timeout_seconds,
                ) as response:
                    status_code = response.status_code
                    retry_after_seconds = _parse_retry_after(
                        response.headers.get("Retry-After")
                    )
                    content = _bounded_response_content(response)
                    _raise_for_status(
                        response,
                        content,
                        model=self._configuration.model,
                        retry_after_seconds=retry_after_seconds,
                    )
                parsed = _parse_response(content)
                _LOGGER.info(
                    "云端 ASR 请求成功: file=%s attempt=%d status=%s elapsed=%.3fs",
                    path.name,
                    attempt,
                    status_code if status_code is not None else "无响应",
                    time.monotonic() - started_at,
                )
                return parsed
            except httpx.RequestError:
                _log_request_failure(
                    path,
                    attempt=attempt,
                    status_code=status_code,
                    elapsed_seconds=time.monotonic() - started_at,
                    error_category="network",
                )
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "云端语音识别网络请求失败",
                )
            except VideoDemoError as error:
                if error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE:
                    _log_request_failure(
                        path,
                        attempt=attempt,
                        status_code=status_code,
                        elapsed_seconds=time.monotonic() - started_at,
                        error_category="permanent_failure",
                    )
                    raise
                _log_request_failure(
                    path,
                    attempt=attempt,
                    status_code=status_code,
                    elapsed_seconds=time.monotonic() - started_at,
                    error_category="temporary_dependency",
                )
                last_error = error
            if attempt < self._configuration.max_attempts:
                delay = retry_after_seconds
                if delay is None:
                    delay = float(2 ** (attempt - 1))
                _LOGGER.info(
                    "云端 ASR 请求重试: file=%s attempt=%d status=%s wait=%.3fs",
                    path.name,
                    attempt,
                    status_code if status_code is not None else "无响应",
                    delay,
                )
                self._sleeper(delay)
        if last_error is None:
            raise RuntimeError("云端语音识别重试状态非法")
        _LOGGER.error(
            "云端 ASR 请求最终失败: file=%s attempts=%d category=temporary_dependency",
            path.name,
            self._configuration.max_attempts,
        )
        raise last_error from None


def _multipart_fields(
    model: str,
    *,
    language_hint: str | None,
    prompt: str | None,
) -> dict[str, str]:
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if (
        isinstance(language_hint, str)
        and language_hint != "und"
        and _LANGUAGE_CODE.fullmatch(language_hint)
    ):
        fields["language"] = language_hint
    if prompt:
        fields["prompt"] = prompt
    return fields


def _validated_audio_path(
    root: Path,
    candidate: Path,
    *,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
) -> Path:
    path = reject_symlink_components(
        root,
        candidate,
        message="云端语音切片必须位于允许目录且不能包含符号链接",
    )
    if not path.is_file():
        raise VideoDemoError(
            ErrorCode.SPEECH_AUDIO_INVALID,
            "云端语音切片不是普通文件",
        )
    size_bytes = path.stat().st_size
    if size_bytes < 1 or size_bytes > max_upload_bytes:
        raise VideoDemoError(
            ErrorCode.SPEECH_AUDIO_INVALID,
            "云端语音切片大小非法",
        )
    return path


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip()
    if re.fullmatch(r"[0-9]+", normalized):
        return float(normalized)
    try:
        retry_at = parsedate_to_datetime(normalized)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=datetime.UTC)
    delay = retry_at.timestamp() - time.time()
    return max(0.0, delay)


def _log_request_failure(
    path: Path,
    *,
    attempt: int,
    status_code: int | None,
    elapsed_seconds: float,
    error_category: str,
) -> None:
    _LOGGER.warning(
        "云端 ASR 请求失败: file=%s attempt=%d status=%s elapsed=%.3fs category=%s",
        path.name,
        attempt,
        status_code if status_code is not None else "无响应",
        elapsed_seconds,
        error_category,
    )


def _raise_for_status(
    response: httpx.Response,
    content: bytes,
    *,
    model: str,
    retry_after_seconds: float | None = None,
) -> None:
    if response.status_code < 400:
        return
    details: dict[str, object] = {"status_code": response.status_code}
    provider_error_code = _provider_error_code(content)
    if provider_error_code is not None:
        details["provider_error_code"] = provider_error_code
    if retry_after_seconds is not None:
        details["retry_after_seconds"] = retry_after_seconds
    if (
        response.status_code in {408, 429}
        or response.status_code >= 500
        or _is_temporary_model_branch_exhaustion(response.status_code, content, model)
    ):
        code = ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
        message = "云端语音识别服务暂时不可用"
    elif response.status_code in {401, 403}:
        code = ErrorCode.SPEECH_AUTHENTICATION_FAILED
        message = "云端语音识别鉴权失败"
    elif response.status_code == 413:
        code = ErrorCode.SPEECH_AUDIO_INVALID
        message = "云端语音切片超过服务限制"
    else:
        code = ErrorCode.SPEECH_MODEL_UNAVAILABLE
        message = "云端语音识别请求被拒绝"
    if _is_temporary_model_branch_exhaustion(response.status_code, content, model):
        raise _TemporaryModelBranchExhaustion(code, message, details)
    raise VideoDemoError(code, message, details)


def _is_temporary_model_branch_exhaustion(
    status_code: int,
    content: bytes,
    model: str,
) -> bool:
    if status_code != 400:
        return False
    try:
        payload: object = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("message") == (
        f"Model is invalid: no available branch for policy group: {model}"
    )


def _provider_error_code(content: bytes) -> str | None:
    try:
        payload: object = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    value = error.get("code")
    if isinstance(value, str) and _PROVIDER_ERROR_CODE.fullmatch(value):
        return value
    return None


def _parse_response(content: bytes) -> WindowTranscriptionResult:
    try:
        payload: object = json.loads(
            content,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(payload, dict):
            raise ValueError
        text = payload["text"]
        language = payload["language"]
        segments = payload["segments"]
        if not isinstance(text, str) or not isinstance(language, str):
            raise ValueError
        if not isinstance(segments, list):
            raise ValueError
        if text.strip() and not segments:
            raise ValueError
        parsed_segments = tuple(
            segment
            for item in segments
            if (segment := _parse_segment(item)) is not None
        )
        if bool(text.strip()) != bool(parsed_segments):
            raise ValueError
        normalized_language, warning = _normalize_language(language)
        return WindowTranscriptionResult(
            language=normalized_language,
            segments=parsed_segments,
            warnings=((warning,) if warning is not None else ()),
        )
    except (KeyError, TypeError, UnicodeDecodeError, ValueError, OverflowError):
        raise VideoDemoError(
            ErrorCode.SPEECH_MODEL_UNAVAILABLE,
            "云端语音识别响应结构非法",
        ) from None


def _bounded_response_content(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size_bytes = 0
    for chunk in response.iter_bytes():
        size_bytes += len(chunk)
        if size_bytes > _MAX_RESPONSE_BYTES:
            raise VideoDemoError(
                ErrorCode.SPEECH_MODEL_UNAVAILABLE,
                "云端语音识别响应超过大小限制",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_segment(value: object) -> RawAsrSegment | None:
    if not isinstance(value, dict):
        raise ValueError
    text = value.get("text")
    if not isinstance(text, str):
        raise ValueError
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError
    start = _finite_number(value.get("start"))
    end = _finite_number(value.get("end"))
    if start < 0 or end <= start:
        raise ValueError
    avg_logprob = _finite_number(value.get("avg_logprob"))
    no_speech_prob = _finite_number(value.get("no_speech_prob"))
    if not 0 <= no_speech_prob <= 1:
        raise ValueError
    start_ms = round(start * 1000)
    end_ms = round(end * 1000)
    if end_ms <= start_ms:
        raise ValueError
    return RawAsrSegment(
        start_ms=start_ms,
        end_ms=end_ms,
        text=normalized_text,
        confidence=_derived_confidence(avg_logprob, no_speech_prob),
    )


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _derived_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    probability = math.exp(min(0.0, avg_logprob)) * (1 - no_speech_prob)
    return min(1.0, max(0.0, probability))


def _normalize_language(value: str) -> tuple[str, str | None]:
    normalized = value.strip().casefold()
    code = _LANGUAGES.get(normalized)
    if code is not None:
        return code, None
    if _LANGUAGE_CODE.fullmatch(normalized):
        return normalized, None
    return "und", "CLOUD_ASR_LANGUAGE_UNRECOGNIZED"
