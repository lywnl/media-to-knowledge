from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

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
from video_demo.storage.workspace import reject_symlink_components, validate_path_component
from video_demo.visual.candidate_artifacts import read_verified_candidate_jpeg

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_MAX_FRAMES = 6
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_RAW_REQUEST_BYTES = 24 * 1024 * 1024
_MAX_ENCODED_REQUEST_BYTES = 36 * 1024 * 1024


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
        result, raw_message, parsed = self._call_with_images(
            request,
            allowed_run_root=allowed_run_root,
            prompt=prompt_for_vision(request),
            schema_name="chapter_vlm_v2",
            on_provider_attempt=on_provider_attempt,
        )
        return _validate_or_raise(
            result,
            request,
            raw_message=raw_message,
            parsed=parsed,
        )

    def repair_chapter(
        self,
        request: ChapterVisionRepairRequest,
        *,
        allowed_run_root: Path,
        on_provider_attempt: Callable[[], None] | None = None,
    ) -> ChapterVisionResponse:
        result, raw_message, parsed = self._call_with_images(
            request.request,
            allowed_run_root=allowed_run_root,
            prompt=prompt_for_vision_repair(request),
            schema_name="chapter_vlm_repair_v2",
            on_provider_attempt=on_provider_attempt,
        )
        return _validate_or_raise(
            result,
            request.request,
            raw_message=raw_message,
            parsed=parsed,
            allowed_frames=set(request.allowed_frame_ids),
            allowed_targets=set(request.allowed_target_ids),
            allowed_transcripts=set(request.allowed_transcript_evidence_ids),
        )

    def _call_with_images(
        self,
        request: ChapterVisionRequest,
        *,
        allowed_run_root: Path,
        prompt: tuple[str, str, str],
        schema_name: str,
        on_provider_attempt: Callable[[], None] | None,
    ) -> tuple[ChapterVisionResponse, bytes, object]:
        run_root = self._validate_run_root(allowed_run_root)
        response_schema = ChapterVisionResponse.model_json_schema()
        raw = self._post_with_retry(
            request,
            run_root=run_root,
            prompt=prompt,
            schema_name=schema_name,
            response_schema=response_schema,
            on_provider_attempt=on_provider_attempt,
        )
        return _parse_response(raw)

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

    def _post_with_retry(
        self,
        request: ChapterVisionRequest,
        *,
        run_root: Path,
        prompt: tuple[str, str, str],
        schema_name: str,
        response_schema: dict[str, object],
        on_provider_attempt: Callable[[], None] | None,
    ) -> bytes:
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            frames: list[tuple[FrameCandidateArtifact, str]] = []
            payload: dict[str, object] | None = None
            try:
                frames = self._verified_frames(request, run_root)
                payload = self._build_checked_payload(
                    frames,
                    prompt=prompt,
                    schema_name=schema_name,
                    response_schema=response_schema,
                )
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
                    "Qwen 视觉模型暂时不可用",
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
            finally:
                frames.clear()
                if payload is not None:
                    payload["messages"] = []
            if attempt < self._max_attempts:
                self._sleeper(min(2 ** (attempt - 1), 4))
        raise last_error or RuntimeError("Qwen 重试状态非法")

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
            ordered_encoded_frames=tuple(
                (frame.frame_id, encoded) for frame, encoded in frames
            ),
        )
        upper_bound = vision_payload_size_upper_bound(
            prompt,
            model_id=self._model_id,
            schema_name=schema_name,
            response_schema=response_schema,
            ordered_frames=tuple(
                (frame.frame_id, frame.size_bytes) for frame, _encoded in frames
            ),
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
        message = envelope["choices"][0]["message"]["content"]  # type: ignore[index]
        if not isinstance(message, str):
            raise ValueError
        raw_message = message.encode("utf-8")
        parsed = json.loads(message, parse_constant=_reject_json_constant)
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
    allowed_frames: set[str] | None = None,
    allowed_targets: set[str] | None = None,
    allowed_transcripts: set[str] | None = None,
) -> ChapterVisionResponse:
    try:
        _validate_observation_references(
            response,
            request,
            allowed_frames=allowed_frames,
            allowed_targets=allowed_targets,
            allowed_transcripts=allowed_transcripts,
        )
        return response
    except _ReferenceValidationError as error:
        raise ModelResponseValidationError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "Qwen 返回内容不符合视觉契约",
            invalid_model_response(
                raw_message,
                (error.summary,),
                parsed_json=parsed,
            ),
        ) from None


def _validate_observation_references(
    response: ChapterVisionResponse,
    request: ChapterVisionRequest,
    *,
    allowed_frames: set[str] | None = None,
    allowed_targets: set[str] | None = None,
    allowed_transcripts: set[str] | None = None,
) -> None:
    frame_ids = allowed_frames or {frame.frame_id for frame in request.frames}
    target_ids = allowed_targets or {target.target_id for target in request.targets}
    transcript_ids = allowed_transcripts or {
        item.evidence_id for item in request.transcript_evidence
    }
    frame_targets = {frame.frame_id: set(frame.target_ids) for frame in request.frames}
    for observation in response.observations:
        _require_known_ids(
            observation.selected_frame_ids,
            frame_ids,
            "observations.selected_frame_ids",
        )
        _require_known_ids(observation.target_ids, target_ids, "observations.target_ids")
        _require_known_ids(
            observation.transcript_evidence_refs,
            transcript_ids,
            "observations.transcript_evidence_refs",
        )
        selected_targets = set().union(
            *(frame_targets[frame_id] for frame_id in observation.selected_frame_ids),
        )
        if (
            any(
                not frame_targets[frame_id].intersection(observation.target_ids)
                for frame_id in observation.selected_frame_ids
            )
            or not set(observation.target_ids).issubset(selected_targets)
        ):
            raise _ReferenceValidationError("observations.target_ids:frame_binding_mismatch")


class _ReferenceValidationError(ValueError):
    def __init__(self, summary: str) -> None:
        super().__init__(summary)
        self.summary = summary


def _require_known_ids(values: tuple[str, ...], allowed: set[str], field: str) -> None:
    if any(value not in allowed for value in values):
        raise _ReferenceValidationError(f"{field}:unknown_reference")


def _pydantic_error_summary(error: object) -> str:
    assert isinstance(error, dict)
    location_value = error.get("loc", ())
    location = ".".join(str(item) for item in location_value) or "response"
    return f"{location}:{error.get('type', 'invalid')}"[:500]
