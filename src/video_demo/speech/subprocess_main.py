from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import ValidationError

from video_demo.application.pipeline import PreparedMedia
from video_demo.application.production_speech import ProductionSpeechAnalyzer
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.speech.runtime import build_subprocess_component_factory
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    speech_fingerprint,
)
from video_demo.speech.subprocess_protocol import (
    ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES,
    SpeechSubprocessFailure,
    SpeechSubprocessRequest,
    SpeechSubprocessResponse,
    SpeechSubprocessSuccess,
    speech_subprocess_failure_message,
)
from video_demo.storage.artifacts import (
    AtomicArtifactStore,
    canonical_artifact_envelope_bytes,
)
from video_demo.storage.snapshots import SnapshotStore
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
    actual_asr_fingerprint = __import__(
        "video_demo.speech.snapshots",
        fromlist=["asr_fingerprint"],
    ).asr_fingerprint(
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
    component_factory = build_subprocess_component_factory(
        workspace_root=workspace_root,
        runtime_root=runtime_root,
        ffmpeg_path=ffmpeg,
        inference_device=request.runtime.inference_device,
        whisper_compute_type=request.runtime.whisper_compute_type,
        huggingface_token=(
            request.credentials.huggingface_token.get_secret_value()
            if request.credentials.huggingface_token is not None
            else None
        ),
    )
    media = cast(
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
    artifact_store = AtomicArtifactStore(runtime_root)
    snapshots = SnapshotStore(artifact_store)
    analyzer = ProductionSpeechAnalyzer(
        component_factory,
        snapshot_store=snapshots,
        fingerprint_inputs=request.runtime.fingerprint_inputs(),
        allow_speaker_fallback=request.allow_speaker_fallback,
    )
    analyzer.analyze(media)
    cached_asr = snapshots.load(
        Path(request.run_relative_root),
        "asr",
        request.asr_fingerprint,
        AsrSnapshotPayload,
    )
    if cached_asr is None:
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
            "语音子进程未发布 ASR 快照",
        )
    speech_key = speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=cached_asr[1].sha256,
        media_warnings=request.media_warnings,
        min_speakers=request.config.min_speakers,
        max_speakers=request.config.max_speakers,
        allow_speaker_fallback=request.allow_speaker_fallback,
        inputs=request.runtime.fingerprint_inputs(),
    )
    cached_speech = snapshots.load(
        Path(request.run_relative_root),
        "speech",
        speech_key,
        SpeechAnalysisSnapshotPayload,
    )
    if cached_speech is None:
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
            "语音子进程未发布完整快照",
        )
    return SpeechSubprocessSuccess(
        request_id=request.request_id,
        speech_fingerprint=speech_key,
        payload_receipt=cached_speech[1],
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
