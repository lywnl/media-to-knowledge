from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
from pydantic import SecretStr, ValidationError

from video_demo.application.pipeline_contracts import PreparedMedia
from video_demo.application.production_speech import run_asr_stage
from video_demo.config import CloudAsrConfiguration
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.cloud_whisper import CloudWhisperClient
from video_demo.speech.runtime import build_subprocess_asr_components
from video_demo.speech.snapshots import AsrSnapshotPayload, asr_fingerprint
from video_demo.speech.subprocess_protocol import (
    ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES,
    SpeechSubprocessFailure,
    SpeechSubprocessRequest,
    SpeechSubprocessResponse,
    SpeechSubprocessSuccess,
    speech_subprocess_failure_message,
)
from video_demo.storage.artifacts import AtomicArtifactStore, canonical_artifact_envelope_bytes
from video_demo.storage.snapshots import AsrWindowSnapshotStore, SnapshotStore
from video_demo.storage.workspace import reject_symlink_components, verified_run_file

_MAX_REQUEST_BYTES = 64 * 1024


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--response", required=True)
    try:
        arguments = parser.parse_args(argv)
        workspace_root, runtime_root = _trusted_roots(
            Path(arguments.workspace_root),
            Path(arguments.runtime_root),
        )
        request_relative = _strict_relative_path(arguments.request)
        response_relative = _strict_relative_path(arguments.response)
        request, request_receipt_sha = _load_request(
            runtime_root,
            request_relative,
            arguments.request_sha256,
        )
        _validate_ipc_paths(request, request_relative, response_relative)
        store = AtomicArtifactStore(runtime_root)
        try:
            response: SpeechSubprocessResponse = _execute_request(
                workspace_root,
                runtime_root,
                request,
            )
        except VideoDemoError as error:
            code = (
                error.code
                if error.code in ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES
                else ErrorCode.SPEECH_MODEL_UNAVAILABLE
            )
            response = SpeechSubprocessFailure(
                request_id=request.request_id,
                asr_fingerprint=request.asr_fingerprint,
                error_code=code,
                message=speech_subprocess_failure_message(code),
            )
        store.write_json(
            response_relative,
            response.model_dump(mode="json", exclude_computed_fields=True),
            schema_version=response.schema_version,
            upstream_sha256=request_receipt_sha,
            file_mode=0o600,
            exclusive=True,
        )
        return 0
    except (OSError, ValueError, ValidationError, VideoDemoError):
        return 2


def _execute_request(
    workspace_root: Path,
    runtime_root: Path,
    request: SpeechSubprocessRequest,
) -> SpeechSubprocessSuccess:
    audio = verified_run_file(
        runtime_root,
        Path(request.run_relative_root),
        runtime_root / request.audio_relative_path,
        expected_sha256=request.audio_sha256,
        digest=_sha256_file,
        message="语音音频必须位于当前运行目录内",
    )
    actual_asr_fingerprint = asr_fingerprint(
        audio_sha256=request.audio_sha256,
        duration_ms=request.duration_ms,
        language_hints=request.config.language_hints,
        hotwords=request.config.hotwords,
        core_context=request.config.core_context,
        inputs=request.runtime.fingerprint_inputs(),
    )
    if actual_asr_fingerprint != request.asr_fingerprint:
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
            "语音请求指纹不匹配",
        )
    ffmpeg = reject_symlink_components(
        workspace_root,
        workspace_root / request.runtime.ffmpeg_relative_path,
        message="FFmpeg 路径非法",
    )
    media = _prepared_media(request, audio)
    configuration = CloudAsrConfiguration(
        base_url=request.runtime.base_url,
        api_key=SecretStr(request.credentials.openai_api_key.get_secret_value()),
        model=request.runtime.model,
        timeout_seconds=request.runtime.timeout_seconds,
        max_attempts=request.runtime.max_attempts,
        max_window_ms=request.runtime.max_window_ms,
        overlap_ms=request.runtime.overlap_ms,
        merge_gap_ms=request.runtime.merge_gap_ms,
        max_upload_bytes=request.runtime.max_upload_bytes,
    )
    artifact_store = AtomicArtifactStore(runtime_root)
    with httpx.Client() as http_client:
        recognizer = CloudWhisperClient(
            http_client,
            configuration,
            allowed_audio_root=runtime_root / request.run_relative_root / "speech" / "slices",
        )
        components = build_subprocess_asr_components(
            media,
            workspace_root=workspace_root,
            runtime_root=runtime_root,
            ffmpeg_path=ffmpeg,
            recognizer=recognizer,
            slice_namespace=request.request_id,
            vad_threshold=request.runtime.vad_threshold,
            vad_merge_gap_ms=request.runtime.vad_merge_gap_ms,
        )
        payload = run_asr_stage(
            media,
            components,
            window_store=AsrWindowSnapshotStore(artifact_store),
            asr_fingerprint=request.asr_fingerprint,
            max_window_ms=request.runtime.max_window_ms,
            overlap_ms=request.runtime.overlap_ms,
            merge_gap_ms=request.runtime.merge_gap_ms,
            max_upload_bytes=request.runtime.max_upload_bytes,
        )
    snapshots = SnapshotStore(artifact_store)
    snapshots.publish(
        Path(request.run_relative_root),
        "asr",
        request.asr_fingerprint,
        payload,
    )
    cached = snapshots.load(
        Path(request.run_relative_root),
        "asr",
        request.asr_fingerprint,
        AsrSnapshotPayload,
    )
    if cached is None or cached[0] != payload:
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
            "语音子进程未发布一致的 ASR 快照",
        )
    return SpeechSubprocessSuccess(
        request_id=request.request_id,
        asr_fingerprint=request.asr_fingerprint,
        payload_receipt=cached[1],
    )


def _prepared_media(request: SpeechSubprocessRequest, audio: Path) -> PreparedMedia:
    return cast(
        PreparedMedia,
        SimpleNamespace(
            source=SimpleNamespace(
                asset=SimpleNamespace(
                    run_relative_root=Path(request.run_relative_root),
                    config=request.config,
                ),
                duration_ms=request.duration_ms,
            ),
            audio_path=audio,
            audio_sha256=request.audio_sha256,
            subtitle=None,
            warnings=request.media_warnings,
        ),
    )


def _trusted_roots(workspace_root: Path, runtime_root: Path) -> tuple[Path, Path]:
    if not workspace_root.is_absolute() or not runtime_root.is_absolute():
        raise ValueError("可信根必须是绝对路径")
    _reject_all_symlink_components(workspace_root)
    _reject_all_symlink_components(runtime_root)
    workspace = workspace_root.resolve(strict=True)
    runtime = runtime_root.resolve(strict=True)
    if not workspace.is_dir() or not runtime.is_dir() or not runtime.is_relative_to(workspace):
        raise ValueError("运行根必须属于工作区")
    return workspace, runtime


def _reject_all_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("可信根不能包含符号链接")


def _strict_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("IPC 路径必须是无穿越相对路径")
    return path


def _load_request(
    runtime_root: Path,
    request_relative: Path,
    expected_sha256: str,
) -> tuple[SpeechSubprocessRequest, str]:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("请求摘要非法")
    path = reject_symlink_components(
        runtime_root,
        runtime_root / request_relative,
        message="语音请求路径非法",
    )
    if not path.is_file() or path.stat().st_size > _MAX_REQUEST_BYTES:
        raise ValueError("语音请求文件非法")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("语音请求摘要不匹配")
    envelope: Any = json.loads(encoded)
    if not isinstance(envelope, dict):
        raise ValueError("语音请求 envelope 非法")
    payload = envelope.get("payload")
    upstream = envelope.get("upstream_sha256")
    if envelope.get("schema_version") != "1.0.0" or not isinstance(payload, dict):
        raise ValueError("语音请求 Schema 非法")
    request = SpeechSubprocessRequest.model_validate(payload)
    if upstream != request.asr_fingerprint:
        raise ValueError("语音请求上游摘要不匹配")
    if encoded != canonical_artifact_envelope_bytes(payload, "1.0.0", upstream):
        raise ValueError("语音请求不是规范编码")
    return request, expected_sha256


def _validate_ipc_paths(
    request: SpeechSubprocessRequest,
    request_relative: Path,
    response_relative: Path,
) -> None:
    ipc_root = Path(request.run_relative_root) / "speech" / "ipc"
    if (
        request_relative.parent != ipc_root
        or response_relative.parent != ipc_root
        or request_relative.name != f"request-{request.request_id}.json"
        or response_relative.name != f"response-{request.request_id}.json"
    ):
        raise ValueError("IPC 路径与当前请求不绑定")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
