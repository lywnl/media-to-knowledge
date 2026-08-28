from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, Field, StrictInt, ValidationError

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.document_plan import FrameCandidateArtifact
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.document_port import (
    ChapterVisionPort,
    ChapterVisionRepairRequest,
    ChapterVisionRequest,
    ChapterVisionResponse,
    ModelResponseValidationError,
    invalid_model_response,
)
from video_demo.integrations.document_prompts import (
    build_vision_payload,
    prompt_for_vision,
    prompt_for_vision_repair,
    vision_payload_json_bytes,
    vision_payload_size_upper_bound,
)
from video_demo.integrations.document_validation import (
    validate_chapter_vision_response,
)
from video_demo.integrations.model_response import (
    extract_model_message_content,
    parse_json_content,
    strip_removed_document_fields,
)
from video_demo.storage.workspace import reject_symlink_components, validate_path_component
from video_demo.visual.candidate_artifacts import read_verified_candidate_jpeg

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_MAX_FRAMES = 6
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_RAW_REQUEST_BYTES = 24 * 1024 * 1024
_MAX_ENCODED_REQUEST_BYTES = 36 * 1024 * 1024


class QwenVisionProviderReceipt(FrozenModel):
    provider_attempt_count: StrictInt = Field(ge=1, le=5)
    final_http_status: StrictInt = Field(ge=200, lt=300)
    response_id_sha256: Sha256 | None = None
    provider_response_sha256: Sha256
    request_json_bytes: StrictInt = Field(ge=0)
    encoded_request_bytes: StrictInt = Field(ge=0)
    elapsed_ms: StrictInt = Field(ge=0)


class QwenVisionProviderFailureReceipt(FrozenModel):
    provider_attempt_count: StrictInt = Field(ge=0, le=5)
    final_http_status: StrictInt | None = Field(default=None, ge=100, le=599)
    provider_response_sha256: Sha256 | None = None


class QwenVisionCallFailure(VideoDemoError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        provider: QwenVisionProviderFailureReceipt,
    ):
        super().__init__(code, message)
        self.provider = provider


@dataclass(frozen=True, slots=True)
class _ProviderCallResult:
    content: bytes | None
    attempts: int
    final_http_status: int | None
    provider_response_sha256: str | None
    response_id_sha256: str | None
    request_json_bytes: int = 0
    encoded_request_bytes: int = 0
    elapsed_ms: int = 0

    def success_receipt(self) -> QwenVisionProviderReceipt:
        if (
            self.content is None
            or self.final_http_status is None
            or self.provider_response_sha256 is None
        ):
            raise ValueError("成功回执缺少供应商响应")
        return QwenVisionProviderReceipt(
            provider_attempt_count=self.attempts,
            final_http_status=self.final_http_status,
            response_id_sha256=self.response_id_sha256,
            provider_response_sha256=self.provider_response_sha256,
            request_json_bytes=self.request_json_bytes,
            encoded_request_bytes=self.encoded_request_bytes,
            elapsed_ms=self.elapsed_ms,
        )

    def failure_receipt(self) -> QwenVisionProviderFailureReceipt:
        return QwenVisionProviderFailureReceipt(
            provider_attempt_count=self.attempts,
            final_http_status=self.final_http_status,
            provider_response_sha256=self.provider_response_sha256,
        )


class _ProviderCallError(VideoDemoError):
    def __init__(self, error: VideoDemoError, result: _ProviderCallResult):
        super().__init__(error.code, error.message, error.details)
        self.result = result


class _ProviderBoundValidationError(ModelResponseValidationError):
    def __init__(
        self,
        error: ModelResponseValidationError,
        result: _ProviderCallResult,
    ) -> None:
        super().__init__(error.code, error.message, error.invalid_response)
        self.result = result


class QwenVisionClient(ChapterVisionPort):
    """Qwen3-VL 图片取证客户端；只接受当次 Run 下的本地 JPEG。"""

    def __init__(
        self,
        http_client: httpx.Client,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        runtime_root: Path,
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
        max_image_bytes: int = _MAX_IMAGE_BYTES,
        max_request_image_bytes: int = _MAX_RAW_REQUEST_BYTES,
        max_encoded_request_bytes: int = _MAX_ENCODED_REQUEST_BYTES,
        max_response_bytes: int = 2 * 1024 * 1024,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http_client = http_client
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model_id = model_id
        self._runtime_root = _verified_directory(runtime_root, "视觉运行根目录非法")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._max_image_bytes = max_image_bytes
        self._max_request_image_bytes = max_request_image_bytes
        self._max_encoded_request_bytes = max_encoded_request_bytes
        self._max_response_bytes = max_response_bytes
        self._sleeper = sleeper

    def analyze_chapter(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        try:
            result, raw_message, parsed, _provider = self._call_with_images(
                request,
                allowed_run_root=allowed_run_root,
                prompt=prompt_for_vision(request),
                schema_name="chapter_vlm_v2",
                on_provider_attempt=on_provider_attempt,
            )
        except _ProviderCallError as error:
            raise VideoDemoError(error.code, error.message, error.details) from error
        return _validate_or_raise(
            result,
            request,
            raw_message=raw_message,
            parsed=parsed,
        )

    def analyze_chapter_with_receipt(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> tuple[ChapterVisionResponse, QwenVisionProviderReceipt]:
        """执行与生产调用相同的请求路径，并返回脱敏供应商回执。"""

        try:
            result, raw_message, parsed, provider_result = self._call_with_images(
                request,
                allowed_run_root=allowed_run_root,
                prompt=prompt_for_vision(request),
                schema_name="chapter_vlm_v2",
                on_provider_attempt=on_provider_attempt,
            )
            response = _validate_or_raise(
                result,
                request,
                raw_message=raw_message,
                parsed=parsed,
            )
        except _ProviderCallError as error:
            raise QwenVisionCallFailure(
                error.code,
                error.message,
                error.result.failure_receipt(),
            ) from error
        except _ProviderBoundValidationError as error:
            raise QwenVisionCallFailure(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 返回内容不符合视觉契约",
                error.result.failure_receipt(),
            ) from error
        except ModelResponseValidationError as error:
            raise QwenVisionCallFailure(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 返回内容不符合视觉契约",
                provider_result.failure_receipt(),
            ) from error
        return response, provider_result.success_receipt()

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        try:
            result, raw_message, parsed, _provider = self._call_with_images(
                request.request,
                allowed_run_root=allowed_run_root,
                prompt=prompt_for_vision_repair(request),
                schema_name="chapter_vlm_repair_v2",
                on_provider_attempt=on_provider_attempt,
            )
        except _ProviderCallError as error:
            raise VideoDemoError(error.code, error.message, error.details) from error
        return _validate_or_raise(
            result,
            request.request,
            raw_message=raw_message,
            parsed=parsed,
            allowed_frames=request.allowed_frame_ids,
            allowed_targets=request.allowed_target_ids,
            allowed_transcripts=request.allowed_transcript_evidence_ids,
        )

    def _call_with_images(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        prompt: tuple[str, str, str],
        schema_name: str,
        on_provider_attempt: Callable[[], None] | None,
    ) -> tuple[ChapterVisionResponse, bytes, object, _ProviderCallResult]:
        run_root = self._validate_run_root(allowed_run_root)
        response_schema = ChapterVisionResponse.model_json_schema()
        provider_result = self._post_with_retry_result(
            request,
            run_root=run_root,
            prompt=prompt,
            schema_name=schema_name,
            response_schema=response_schema,
            on_provider_attempt=on_provider_attempt,
        )
        assert provider_result.content is not None
        try:
            response, raw_message, parsed = _parse_response(provider_result.content)
        except ModelResponseValidationError as error:
            raise _ProviderBoundValidationError(error, provider_result) from error
        return response, raw_message, parsed, provider_result

    def _validate_run_root(self, allowed_run_root: Path) -> Path:
        lexical = allowed_run_root.expanduser()
        if not lexical.is_absolute():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根必须是绝对路径")
        root = reject_symlink_components(
            self._runtime_root,
            lexical,
            message="当前 Run 根必须位于视觉运行目录且不能包含符号链接",
        )
        try:
            relative = root.relative_to(self._runtime_root)
        except ValueError:
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法") from None
        if len(relative.parts) != 3 or relative.parts[0] != "runs" or not root.is_dir():
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "当前 Run 根非法")
        validate_path_component(relative.parts[1], "scope_key")
        validate_path_component(relative.parts[2], "run_id")
        return root

    def _verified_frames(
        self,
        request: ChapterVisionRequest,
        run_root: Path,
    ) -> list[tuple[FrameCandidateArtifact, str]]:
        ordered = sorted(request.frames, key=lambda frame: (frame.timestamp_ms, frame.frame_id))
        verified: list[tuple[FrameCandidateArtifact, str]] = []
        total_bytes = 0
        for frame in ordered:
            content = read_verified_candidate_jpeg(
                run_root,
                frame,
                max_bytes=self._max_image_bytes,
            )
            total_bytes += len(content)
            if total_bytes > self._max_request_image_bytes:
                raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "章节图片原始字节超过上限")
            verified.append((frame, base64.b64encode(content).decode("ascii")))
        return verified

    def _post_with_retry_result(
        self,
        request: ChapterVisionRequest,
        *,
        run_root: Path,
        prompt: tuple[str, str, str],
        schema_name: str,
        response_schema: dict[str, object],
        on_provider_attempt: Callable[[], None] | None,
    ) -> _ProviderCallResult:
        last_error: VideoDemoError | None = None
        attempt_count = 0
        final_http_status: int | None = None
        provider_response_sha256: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            frames: list[tuple[FrameCandidateArtifact, str]] = []
            payload: dict[str, object] | None = None
            request_json_bytes = 0
            encoded_request_bytes = 0
            elapsed_ms = 0
            final_http_status = None
            provider_response_sha256 = None
            try:
                frames = self._verified_frames(request, run_root)
                payload = self._build_checked_payload(
                    frames,
                    prompt=prompt,
                    schema_name=schema_name,
                    response_schema=response_schema,
                )
                # `request_json_bytes` 是同一请求在图片尚未编码时的 JSON 基线；
                # `encoded_request_bytes` 是 httpx 最终发送的、含 Base64 图片的
                # 完整紧凑 JSON 请求体。两者都来自实际 payload 结构，不能用图片
                # Base64 字符串之和近似后者。
                request_json_bytes = len(
                    vision_payload_json_bytes(
                        build_vision_payload(
                            prompt,
                            model_id=self._model_id,
                            schema_name=schema_name,
                            response_schema=response_schema,
                            ordered_encoded_frames=tuple(
                                (frame.frame_id, "") for frame, _encoded in frames
                            ),
                        )
                    )
                )
                encoded_request_bytes = len(vision_payload_json_bytes(payload))
                if on_provider_attempt is not None:
                    on_provider_attempt()
                attempt_count += 1
                started_at = time.monotonic()
                with self._http_client.stream(
                    "POST",
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                ) as response:
                    final_http_status = response.status_code
                    if not 200 <= response.status_code < 300:
                        _raise_response_status(response)
                    content = _bounded_response(response, self._max_response_bytes)
                elapsed_ms = max(0, round((time.monotonic() - started_at) * 1000))
                provider_digest = hashlib.sha256(content).hexdigest()
                response_id_digest = _response_id_sha256(content)
                return _ProviderCallResult(
                    content=content,
                    attempts=attempt_count,
                    final_http_status=final_http_status,
                    provider_response_sha256=provider_digest,
                    response_id_sha256=response_id_digest,
                    request_json_bytes=request_json_bytes,
                    encoded_request_bytes=encoded_request_bytes,
                    elapsed_ms=elapsed_ms,
                )
            except (httpx.RequestError, TimeoutError) as error:
                final_http_status = None
                provider_response_sha256 = None
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "Qwen 视觉模型暂时不可用",
                )
                if attempt == self._max_attempts:
                    raise _ProviderCallError(
                        last_error,
                        _ProviderCallResult(
                            content=None,
                            attempts=attempt_count,
                            final_http_status=None,
                            provider_response_sha256=None,
                            response_id_sha256=None,
                            request_json_bytes=request_json_bytes,
                            encoded_request_bytes=encoded_request_bytes,
                            elapsed_ms=elapsed_ms,
                        ),
                    ) from error
            except _ProviderCallError:
                raise
            except VideoDemoError as error:
                if (
                    error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
                    or attempt == self._max_attempts
                ):
                    raise _ProviderCallError(
                        error,
                        _ProviderCallResult(
                            content=None,
                            attempts=attempt_count,
                            final_http_status=final_http_status,
                            provider_response_sha256=provider_response_sha256,
                            response_id_sha256=None,
                            request_json_bytes=request_json_bytes,
                            encoded_request_bytes=encoded_request_bytes,
                            elapsed_ms=elapsed_ms,
                        ),
                    ) from error
                last_error = error
            finally:
                frames.clear()
                if payload is not None:
                    payload["messages"] = []
            if attempt < self._max_attempts:
                self._sleeper(min(2 ** (attempt - 1), 4))
        terminal_error = last_error or VideoDemoError(
            ErrorCode.SYSTEM_FAILURE,
            "Qwen 重试状态非法",
        )
        raise _ProviderCallError(
            terminal_error,
            _ProviderCallResult(
                content=None,
                attempts=attempt_count,
                final_http_status=final_http_status,
                provider_response_sha256=provider_response_sha256,
                response_id_sha256=None,
                request_json_bytes=request_json_bytes,
                encoded_request_bytes=encoded_request_bytes,
                elapsed_ms=elapsed_ms,
            ),
        )

    def _build_checked_payload(
        self,
        frames: list[tuple[FrameCandidateArtifact, str]],
        *,
        prompt: tuple[str, str, str],
        schema_name: str,
        response_schema: dict[str, object],
    ) -> dict[str, object]:
        payload = build_vision_payload(
            prompt,
            model_id=self._model_id,
            schema_name=schema_name,
            response_schema=response_schema,
            ordered_encoded_frames=tuple((frame.frame_id, encoded) for frame, encoded in frames),
        )
        upper_bound = vision_payload_size_upper_bound(
            prompt,
            model_id=self._model_id,
            schema_name=schema_name,
            response_schema=response_schema,
            ordered_frames=tuple((frame.frame_id, frame.size_bytes) for frame, _encoded in frames),
        )
        actual_size = len(vision_payload_json_bytes(payload))
        if actual_size > upper_bound or actual_size > self._max_encoded_request_bytes:
            payload["messages"] = []
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "视觉模型请求体超过大小上限")
        return payload


def _verified_directory(path: Path, message: str) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute() or lexical.is_symlink() or not lexical.is_dir():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message)
    return lexical.resolve(strict=True)


def _bounded_response(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 响应超过大小上限")
        chunks.append(chunk)
    return b"".join(chunks)


def _response_id_sha256(content: bytes) -> str | None:
    try:
        envelope = json.loads(content, parse_constant=_reject_json_constant)
        if not isinstance(envelope, dict):
            return None
        response_id = envelope.get("id")
        if response_id is None:
            return None
        if not isinstance(response_id, str):
            raise ValueError("供应商响应 ID 必须是字符串")
        return hashlib.sha256(response_id.encode("utf-8", errors="strict")).hexdigest()
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _raise_response_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    if response.status_code in {401, 403}:
        raise VideoDemoError(ErrorCode.QWEN_AUTHENTICATION_FAILED, "Qwen 视觉模型鉴权失败")
    if response.status_code in {408, 429} or response.status_code >= 500:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "Qwen 视觉模型暂时不可用")
    if response.status_code in {404, 415, 422}:
        raise VideoDemoError(ErrorCode.QWEN_CAPABILITY_UNAVAILABLE, "Qwen 端点不支持图片取证能力")
    raise VideoDemoError(ErrorCode.QWEN_REQUEST_REJECTED, "Qwen 视觉模型请求被拒绝")


def _parse_response(content: bytes) -> tuple[ChapterVisionResponse, bytes, object]:
    envelope: object | None = None
    raw_message: bytes | None = None
    try:
        envelope = json.loads(content, parse_constant=_reject_json_constant)
        if not isinstance(envelope, dict):
            raise ValueError
        provider_id = envelope.get("id")
        if provider_id is not None and not isinstance(provider_id, str):
            raise ValueError("供应商响应 ID 必须是字符串")
        message = extract_model_message_content(envelope)
        if not message:
            raise ValueError
        raw_message = message.encode("utf-8")
        parsed = strip_removed_document_fields(parse_json_content(message))
        return ChapterVisionResponse.model_validate(parsed), raw_message, parsed
    except ValidationError as error:
        summaries = tuple(_pydantic_error_summary(item) for item in error.errors())
        raw = locals().get("raw_message", content)
        parsed_value = locals().get("parsed", envelope)
    except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError):
        summaries = ("response_envelope:invalid",)
        raw = raw_message if raw_message is not None else content
        parsed_value = envelope if raw_message is None else None
    raise ModelResponseValidationError(
        ErrorCode.QWEN_RESPONSE_INVALID,
        "Qwen 返回内容不符合视觉契约",
        invalid_model_response(raw, summaries, parsed_json=parsed_value),
    ) from None


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _validate_or_raise(
    response: ChapterVisionResponse,
    request: ChapterVisionRequest,
    *,
    raw_message: bytes,
    parsed: object,
    allowed_frames: tuple[str, ...] | None = None,
    allowed_targets: tuple[str, ...] | None = None,
    allowed_transcripts: tuple[str, ...] | None = None,
) -> ChapterVisionResponse:
    try:
        validate_chapter_vision_response(
            response,
            request,
            max_selected_frames=3,
            allowed_frames=allowed_frames,
            allowed_targets=allowed_targets,
            allowed_transcripts=allowed_transcripts,
        )
        return response
    except ValueError as error:
        summary = str(error)
        raise ModelResponseValidationError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "Qwen 返回内容不符合视觉契约",
            invalid_model_response(
                raw_message,
                (summary,),
                parsed_json=parsed,
            ),
        ) from None


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
