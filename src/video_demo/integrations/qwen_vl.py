from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
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
    prompt_for_vision,
    prompt_for_vision_repair,
)
from video_demo.storage.workspace import reject_symlink_components, validate_path_component

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_MAX_FRAMES = 6
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_RAW_REQUEST_BYTES = 24 * 1024 * 1024
_MAX_ENCODED_REQUEST_BYTES = 36 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


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
    ) -> ChapterVisionResponse:
        result, raw_message, parsed = self._call_with_images(
            request,
            allowed_run_root=allowed_run_root,
            prompt=prompt_for_vision(request),
            schema_name="chapter_vlm_v1",
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
    ) -> ChapterVisionResponse:
        result, raw_message, parsed = self._call_with_images(
            request.request,
            allowed_run_root=allowed_run_root,
            prompt=prompt_for_vision_repair(request),
            schema_name="chapter_vlm_repair_v1",
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
    ) -> tuple[ChapterVisionResponse, bytes, object]:
        run_root = self._validate_run_root(allowed_run_root)
        frames = self._verified_frames(request, run_root)
        version, instruction, data = prompt
        content = _vision_content(data, frames)
        payload = self._payload(
            version,
            instruction,
            content,
            schema_name=schema_name,
        )
        try:
            raw = self._post_with_retry(payload)
        finally:
            frames.clear()
            content.clear()
            payload["messages"] = []
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
            content = _verified_jpeg(
                run_root,
                frame,
                max_image_bytes=self._max_image_bytes,
            )
            total_bytes += len(content)
            if total_bytes > self._max_request_image_bytes:
                raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "章节图片原始字节超过上限")
            verified.append((frame, base64.b64encode(content).decode("ascii")))
        return verified

    def _payload(
        self,
        version: str,
        instruction: str,
        content: list[dict[str, object]],
        *,
        schema_name: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model_id,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": ChapterVisionResponse.model_json_schema(),
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": f"PROMPT_VERSION={version}\n{instruction}",
                },
                {"role": "user", "content": content},
            ],
        }
        encoded_payload_size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
        if encoded_payload_size > self._max_encoded_request_bytes:
            raise VideoDemoError(ErrorCode.INPUT_BUDGET_EXCEEDED, "视觉模型请求体超过大小上限")
        return payload

    def _post_with_retry(self, payload: dict[str, object]) -> bytes:
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
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
            if attempt < self._max_attempts:
                self._sleeper(min(2 ** (attempt - 1), 4))
        raise last_error or RuntimeError("Qwen 重试状态非法")


def _verified_directory(path: Path, message: str) -> Path:
    lexical = path.expanduser()
    if not lexical.is_absolute() or lexical.is_symlink() or not lexical.is_dir():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, message)
    return lexical.resolve(strict=True)


def _verified_jpeg(
    run_root: Path,
    frame: FrameCandidateArtifact,
    *,
    max_image_bytes: int,
) -> bytes:
    if frame.mime_type != "image/jpeg":
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片只允许 JPEG")
    relative = Path(frame.relative_path)
    if relative.is_absolute() or relative.suffix != ".jpg":
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片必须使用 .jpg 相对路径")
    path = reject_symlink_components(
        run_root,
        run_root / relative,
        message="视觉图片必须位于当前 Run 且不能包含符号链接",
    )
    try:
        before = os.lstat(path)
    except OSError as error:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片无法读取") from error
    if stat.S_ISLNK(before.st_mode):
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "视觉图片不能是符号链接")
    if not stat.S_ISREG(before.st_mode):
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片不是普通文件")
    if before.st_size != frame.size_bytes or before.st_size > max_image_bytes:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片大小校验失败")
    descriptor = _open_no_follow(path)
    try:
        opened = os.fstat(descriptor)
        _require_same_file(before, opened)
        content = _read_descriptor(descriptor, max_image_bytes)
        after = os.fstat(descriptor)
        _require_same_file(opened, after)
    finally:
        os.close(descriptor)
    if len(content) != frame.size_bytes:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片大小校验失败")
    if not _has_jpeg_magic(content):
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片媒体类型校验失败")
    if hashlib.sha256(content).hexdigest() != frame.sha256:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片摘要校验失败")
    return content


def _open_no_follow(path: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "当前平台不支持安全图片打开")
    try:
        return os.open(path, os.O_RDONLY | no_follow)
    except OSError as error:
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片安全打开失败") from error


def _read_descriptor(descriptor: int, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1 - total)):
        total += len(chunk)
        if total > max_bytes:
            raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片超过大小上限")
        chunks.append(chunk)
    return b"".join(chunks)


def _require_same_file(before: os.stat_result, after: os.stat_result) -> None:
    if _file_identity(before) != _file_identity(after):
        raise VideoDemoError(ErrorCode.VISUAL_MEDIA_INVALID, "视觉图片读取期间发生变化")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _has_jpeg_magic(content: bytes) -> bool:
    return content.startswith(b"\xff\xd8\xff") and content.endswith(b"\xff\xd9")


def _vision_content(
    data: str,
    frames: list[tuple[FrameCandidateArtifact, str]],
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "text", "text": "UNTRUSTED_VISION_CONTEXT_JSON\n" + data},
    ]
    for frame, encoded in frames:
        content.append({"type": "text", "text": f"FRAME_ID={frame.frame_id}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
        )
    return content


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
    if response.status_code < 400:
        return
    if response.status_code in {401, 403}:
        raise VideoDemoError(ErrorCode.QWEN_AUTHENTICATION_FAILED, "Qwen 视觉模型鉴权失败")
    if response.status_code in {408, 429} or response.status_code >= 500:
        raise VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "Qwen 视觉模型暂时不可用")
    if response.status_code in {404, 415, 422}:
        raise VideoDemoError(ErrorCode.QWEN_CAPABILITY_UNAVAILABLE, "Qwen 端点不支持图片取证能力")
    raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 视觉模型请求被拒绝")


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
        if any(
            not frame_targets[frame_id].intersection(observation.target_ids)
            for frame_id in observation.selected_frame_ids
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
