from __future__ import annotations

import errno
import hashlib
import os
import stat
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from video_demo.application.pipeline_contracts import PreparedMedia, SpeechAnalysis, StageMetric
from video_demo.application.production_speech import (
    analysis_from_asr_snapshot,
    transcript_shortcut,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import SafeProcessRunner
from video_demo.speech.snapshots import AsrSnapshotPayload, asr_fingerprint
from video_demo.speech.subprocess_protocol import (
    ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES,
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
    SpeechSubprocessFailure,
    SpeechSubprocessRequest,
    SpeechSubprocessSuccess,
    ipc_request_payload,
    speech_subprocess_failure_message,
)
from video_demo.storage.artifacts import ArtifactReceipt, AtomicArtifactStore
from video_demo.storage.snapshots import SnapshotStore, inspect_artifact
from video_demo.storage.workspace import reject_symlink_components

_MAX_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class AsrStageResult:
    payload: AsrSnapshotPayload
    receipt: ArtifactReceipt


ProcessRunnerFactory = Callable[[Callable[[], bool]], SafeProcessRunner]


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


def _fd_directory_flags() -> int:
    required_dir_fd = {os.open, os.stat, os.unlink}
    if (
        os.name != "posix"
        or not all(
            hasattr(os, name)
            for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
        )
        or not required_dir_fd.issubset(os.supports_dir_fd)
        or os.stat not in os.supports_follow_symlinks
    ):
        raise OSError(errno.ENOTSUP, "当前平台缺少 fd 安全清理能力")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _open_directory_descriptor(path: Path) -> int:
    absolute = path.absolute()
    descriptors: list[int] = []
    try:
        descriptor = os.open(absolute.anchor, _fd_directory_flags())
        descriptors.append(descriptor)
        for component in absolute.parts[1:]:
            child = os.open(component, _fd_directory_flags(), dir_fd=descriptors[-1])
            descriptors.append(child)
        for parent in descriptors[:-1]:
            os.close(parent)
        return descriptors[-1]
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise


def _open_descendant_directory(root: Path, relative: Path) -> int:
    if relative.is_absolute():
        raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "语音 IPC 路径非法")
    descriptors: list[int] = [_open_directory_descriptor(root)]
    try:
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "语音 IPC 路径非法")
            child = os.open(component, _fd_directory_flags(), dir_fd=descriptors[-1])
            descriptors.append(child)
        for parent in descriptors[:-1]:
            os.close(parent)
        return descriptors[-1]
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise


class IsolatedSpeechAnalyzer:
    """父进程只处理 shortcut/快照，未命中时启动一次 ASR 子进程。"""

    def __init__(
        self,
        *,
        workspace_root: Path,
        runtime_root: Path,
        snapshot_store: SnapshotStore,
        artifact_store: AtomicArtifactStore,
        speech_runtime: SpeechRuntimeConfig,
        credentials: SpeechSubprocessCredentials,
        timeout_seconds: int | None = None,
        asr_timeout_seconds: int = 3600,
        process_runner_factory: ProcessRunnerFactory | None = None,
        python_executable: Path | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve(strict=False)
        self._runtime_root = runtime_root.resolve(strict=False)
        self._snapshot_store = snapshot_store
        self._artifact_store = artifact_store
        self._speech_runtime = speech_runtime
        self._credentials = credentials
        self._asr_timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else asr_timeout_seconds
        )
        self._process_runner_factory = process_runner_factory or (
            lambda cancel: SafeProcessRunner(
                max_output_bytes=64 * 1024,
                is_cancel_requested=cancel,
                workspace_root=self._workspace_root,
            )
        )
        self._python_executable = python_executable or Path(sys.executable)

    def analyze(
        self,
        media: PreparedMedia,
        *,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> SpeechAnalysis:
        shortcut = transcript_shortcut(media)
        if shortcut is not None:
            return shortcut
        if media.audio_path is None or media.audio_sha256 is None:
            raise VideoDemoError(
                ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                "语音子进程输入缺少音频",
            )
        run_root = media.source.asset.run_relative_root
        config = media.source.asset.config
        asr_key = asr_fingerprint(
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            language_hints=config.language_hints,
            hotwords=config.hotwords,
            core_context=config.core_context,
            inputs=self._speech_runtime.fingerprint_inputs(),
        )
        started_at = time.monotonic()
        cached = self._snapshot_store.load(run_root, "asr", asr_key, AsrSnapshotPayload)
        if cached is not None:
            analysis = analysis_from_asr_snapshot(media, cached[0])
            return replace(
                analysis,
                stage_metrics=(StageMetric("SPEECH_ASR", 0),),
                stage_cache_hits=("SPEECH_ASR",),
            )
        result = self._run_subprocess(
            media,
            asr_key=asr_key,
            is_cancel_requested=is_cancel_requested,
        )
        analysis = analysis_from_asr_snapshot(media, result.payload)
        return replace(
            analysis,
            stage_metrics=(StageMetric("SPEECH_ASR", _elapsed_ms(started_at)),),
        )

    def _run_subprocess(
        self,
        media: PreparedMedia,
        *,
        asr_key: str,
        is_cancel_requested: Callable[[], bool],
    ) -> AsrStageResult:
        assert media.audio_path is not None
        assert media.audio_sha256 is not None
        try:
            audio_relative = media.audio_path.relative_to(self._runtime_root)
        except ValueError:
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "语音音频必须位于运行目录内",
            ) from None
        request_id = f"speech_{uuid.uuid4().hex}"
        request = SpeechSubprocessRequest(
            request_id=request_id,
            run_relative_root=media.source.asset.run_relative_root.as_posix(),
            audio_relative_path=audio_relative.as_posix(),
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            config=media.source.asset.config,
            media_warnings=media.warnings,
            runtime=self._speech_runtime,
            credentials=self._credentials,
            asr_fingerprint=asr_key,
        )
        try:
            return self._run_stage(
                request,
                media.source.asset.run_relative_root,
                is_cancel_requested,
            )
        finally:
            with suppress(OSError, VideoDemoError):
                self._discard_request_slices(
                    media.source.asset.run_relative_root,
                    request_id,
                )

    def _run_stage(
        self,
        request: SpeechSubprocessRequest,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
    ) -> AsrStageResult:
        python = self._verified_python()
        request_id = request.request_id
        ipc_root = run_root / "speech" / "ipc"
        request_relative = ipc_root / f"request-{request_id}.json"
        response_relative = ipc_root / f"response-{request_id}.json"
        request_receipt = self._artifact_store.write_json(
            request_relative,
            ipc_request_payload(request),
            schema_version=request.schema_version,
            upstream_sha256=request.asr_fingerprint,
            file_mode=0o600,
            exclusive=True,
        )
        response_receipt = None
        try:
            command = [
                str(python),
                "-m",
                "video_demo.speech.subprocess_main",
                "--workspace-root",
                str(self._workspace_root),
                "--runtime-root",
                str(self._runtime_root),
                "--request",
                request_relative.as_posix(),
                "--request-sha256",
                request_receipt.sha256,
                "--response",
                response_relative.as_posix(),
            ]
            runner = self._process_runner_factory(is_cancel_requested)
            try:
                result = runner.run(
                    command,
                    timeout_seconds=self._asr_timeout_seconds,
                    output_paths=(self._runtime_root / response_relative,),
                    env=self._subprocess_environment(),
                )
            except VideoDemoError as error:
                self._map_process_error(error)
                raise AssertionError("不可达") from error
            if result.returncode != 0:
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_CRASHED,
                    "语音子进程异常退出",
                    details={"stage": "ASR"},
                )
            try:
                response_receipt = self._response_receipt(
                    response_relative,
                    request_receipt.sha256,
                )
                verified_receipt, payload = inspect_artifact(
                    self._artifact_store,
                    response_relative,
                    schema_version="1.0.0",
                    upstream_sha256=request_receipt.sha256,
                    max_bytes=_MAX_RESPONSE_BYTES,
                )
                if verified_receipt != response_receipt:
                    raise ValueError("响应回执在读取期间发生变化")
                if not isinstance(payload, dict):
                    raise ValueError("响应 payload 必须是对象")
                if payload.get("status") == "FAILED":
                    failure = SpeechSubprocessFailure.model_validate(payload)
                    self._validate_response_binding(request, failure)
                    if failure.error_code not in ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES:
                        raise ValueError("响应错误码不在语音子进程白名单")
                    stable_message = speech_subprocess_failure_message(failure.error_code)
                    if failure.message != stable_message:
                        raise ValueError("响应错误消息不是稳定协议消息")
                    raise VideoDemoError(
                        failure.error_code,
                        stable_message,
                        details={"stage": "ASR"},
                    )
                response = SpeechSubprocessSuccess.model_validate(payload)
                self._validate_response_binding(request, response)
                loaded = self._snapshot_store.load(
                    run_root,
                    "asr",
                    request.asr_fingerprint,
                    AsrSnapshotPayload,
                )
                if loaded is None or loaded[1] != response.payload_receipt:
                    raise ValueError("响应回执未绑定当前 ASR 快照")
                return AsrStageResult(*loaded)
            except VideoDemoError as error:
                if error.code in {
                    ErrorCode.ARTIFACT_NOT_FOUND,
                    ErrorCode.ARTIFACT_DIGEST_MISMATCH,
                    ErrorCode.ARTIFACT_SCHEMA_INVALID,
                    ErrorCode.ARTIFACT_UPSTREAM_MISMATCH,
                }:
                    raise VideoDemoError(
                        ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                        "语音子进程响应非法",
                        details={"stage": "ASR"},
                    ) from None
                raise
            except (OSError, ValueError, ValidationError):
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                    "语音子进程响应非法",
                    details={"stage": "ASR"},
                ) from None
        finally:
            with suppress(OSError, VideoDemoError):
                self._discard_ipc_artifact(
                    run_root,
                    request_id,
                    "request",
                    request_receipt.sha256,
                )
            with suppress(OSError, VideoDemoError):
                self._discard_ipc_artifact(
                    run_root,
                    request_id,
                    "response",
                    response_receipt.sha256 if response_receipt is not None else None,
                )

    @staticmethod
    def _validate_response_binding(
        request: SpeechSubprocessRequest,
        response: SpeechSubprocessSuccess | SpeechSubprocessFailure,
    ) -> None:
        if (
            response.request_id != request.request_id
            or response.asr_fingerprint != request.asr_fingerprint
        ):
            raise ValueError("语音响应请求 ID 或 ASR 指纹不匹配")

    def _discard_request_slices(self, run_root: Path, request_id: str) -> None:
        slices_relative = run_root / "speech" / "slices"
        slices_path = self._runtime_root / slices_relative
        if not slices_path.exists():
            return
        descriptor = _open_descendant_directory(self._runtime_root, slices_relative)
        try:
            prefix = f"{request_id}_"
            for filename in os.listdir(descriptor):
                if not filename.startswith(prefix) or not filename.endswith(".wav"):
                    continue
                details = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISREG(details.st_mode):
                    os.unlink(filename, dir_fd=descriptor)
        finally:
            os.close(descriptor)

    def _discard_ipc_artifact(
        self,
        run_root: Path,
        request_id: str,
        artifact_kind: Literal["request", "response"],
        expected_sha256: str | None,
    ) -> None:
        ipc_relative = run_root / "speech" / "ipc"
        filename = f"{artifact_kind}-{request_id}.json"
        ipc_descriptor = _open_descendant_directory(self._runtime_root, ipc_relative)
        file_descriptor: int | None = None
        try:
            reject_symlink_components(
                self._runtime_root,
                self._runtime_root / ipc_relative / filename,
                message="语音子进程响应路径非法",
            )
            details = os.stat(filename, dir_fd=ipc_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(details.st_mode):
                return
            if expected_sha256 is not None:
                file_descriptor = os.open(
                    filename,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=ipc_descriptor,
                )
                before = os.fstat(file_descriptor)
                if not stat.S_ISREG(before.st_mode):
                    return
                digest = hashlib.sha256()
                while chunk := os.read(file_descriptor, 64 * 1024):
                    digest.update(chunk)
                after = os.fstat(file_descriptor)
                current = os.stat(filename, dir_fd=ipc_descriptor, follow_symlinks=False)
                identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                if (
                    digest.hexdigest() != expected_sha256
                    or not stat.S_ISREG(current.st_mode)
                    or identity
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or identity
                    != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                ):
                    return
            os.unlink(filename, dir_fd=ipc_descriptor)
        finally:
            if file_descriptor is not None:
                with suppress(OSError):
                    os.close(file_descriptor)
            os.close(ipc_descriptor)

    def _response_receipt(
        self,
        response_relative: Path,
        request_sha256: str,
    ) -> ArtifactReceipt:
        response = reject_symlink_components(
            self._runtime_root,
            self._runtime_root / response_relative,
            message="语音子进程响应路径非法",
        )
        if not response.is_file():
            raise ValueError("语音子进程响应不存在")
        with response.open("rb") as stream:
            encoded = stream.read(_MAX_RESPONSE_BYTES + 1)
        if len(encoded) > _MAX_RESPONSE_BYTES:
            raise ValueError("语音子进程响应超过大小上限")
        return ArtifactReceipt(
            relative_path=response_relative.as_posix(),
            schema_version="1.0.0",
            sha256=hashlib.sha256(encoded).hexdigest(),
            upstream_sha256=request_sha256,
        )

    def _verified_python(self) -> Path:
        python = self._python_executable
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise VideoDemoError(
                ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
                "当前 Python 解释器不可执行",
            )
        return python

    def _subprocess_environment(self) -> Mapping[str, str]:
        allowed = (
            "PATH",
            "LANG",
            "LC_ALL",
            "VIRTUAL_ENV",
            "DYLD_LIBRARY_PATH",
            "LD_LIBRARY_PATH",
        )
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONUTF8": "1",
                "PYTHONPATH": str(self._workspace_root / "src"),
                "TMPDIR": str(self._private_environment_directory("tmp")),
                "XDG_CACHE_HOME": str(self._private_environment_directory("cache")),
                "HF_HOME": str(self._private_environment_directory("huggingface")),
                "TORCH_HOME": str(self._private_environment_directory("torch")),
            }
        )
        return environment

    def _private_environment_directory(self, name: str) -> Path:
        path = self._runtime_root / "speech-environment" / name
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def _map_process_error(error: VideoDemoError) -> None:
        if error.code == ErrorCode.VIDEO_PROCESS_CANCELLED:
            raise VideoDemoError(
                ErrorCode.JOB_CANCELLED,
                "语音分析已取消",
                details={"stage": "ASR"},
            ) from None
        if error.code == ErrorCode.VIDEO_PROCESS_TIMEOUT:
            raise VideoDemoError(
                ErrorCode.SPEECH_SUBPROCESS_TIMEOUT,
                "语音子进程执行超时",
                details={"stage": "ASR"},
            ) from None
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_CRASHED,
            "语音子进程执行失败",
            details={"stage": "ASR"},
        ) from None
