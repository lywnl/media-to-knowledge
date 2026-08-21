from __future__ import annotations

import hashlib
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from video_demo.application.pipeline import PreparedMedia, SpeechAnalysis
from video_demo.application.production_speech import ProductionSpeechAnalyzer, SpeechComponents
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import SafeProcessRunner
from video_demo.speech.snapshots import (
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    asr_fingerprint,
)
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


class _NeedSpeechSubprocess(Exception):
    pass


ProcessRunnerFactory = Callable[[Callable[[], bool]], SafeProcessRunner]


class IsolatedSpeechAnalyzer:
    """在父 Worker 中只处理轻量快照路径，其余语音计算交给一次性子进程。"""

    def __init__(
        self,
        *,
        workspace_root: Path,
        runtime_root: Path,
        snapshot_store: SnapshotStore,
        artifact_store: AtomicArtifactStore,
        fingerprint_inputs: SpeechFingerprintInputs,
        speech_runtime: SpeechRuntimeConfig,
        credentials: SpeechSubprocessCredentials,
        timeout_seconds: int,
        allow_speaker_fallback: bool = False,
        process_runner_factory: ProcessRunnerFactory | None = None,
        python_executable: Path | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve(strict=False)
        self._runtime_root = runtime_root.resolve(strict=False)
        self._snapshot_store = snapshot_store
        self._artifact_store = artifact_store
        self._fingerprint_inputs = fingerprint_inputs
        self._speech_runtime = speech_runtime
        self._credentials = credentials
        self._timeout_seconds = timeout_seconds
        self._allow_speaker_fallback = allow_speaker_fallback
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
        shortcut = ProductionSpeechAnalyzer(
            self._require_subprocess,
            snapshot_store=self._snapshot_store,
            fingerprint_inputs=self._fingerprint_inputs,
            allow_speaker_fallback=self._allow_speaker_fallback,
        )
        try:
            return shortcut.analyze(media, is_cancel_requested=is_cancel_requested)
        except _NeedSpeechSubprocess:
            pass
        if media.audio_path is None or media.audio_sha256 is None:
            raise VideoDemoError(
                ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                "语音子进程输入缺少音频",
            )
        return self._run_subprocess(media, is_cancel_requested)

    @staticmethod
    def _require_subprocess(
        _media: PreparedMedia,
        _cancel: Callable[[], bool],
    ) -> SpeechComponents:
        raise _NeedSpeechSubprocess

    def _run_subprocess(
        self,
        media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechAnalysis:
        python = self._verified_python()
        run_root = media.source.asset.run_relative_root
        config = media.source.asset.config
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
        asr_key = asr_fingerprint(
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            language_hints=config.language_hints,
            hotwords=config.hotwords,
            core_context=config.core_context,
            inputs=self._fingerprint_inputs,
        )
        request = SpeechSubprocessRequest(
            request_id=request_id,
            run_relative_root=run_root.as_posix(),
            audio_relative_path=audio_relative.as_posix(),
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            config=config,
            media_warnings=media.warnings,
            runtime=self._speech_runtime,
            credentials=self._credentials,
            asr_fingerprint=asr_key,
            allow_speaker_fallback=self._allow_speaker_fallback,
        )
        ipc_root = run_root / "speech" / "ipc"
        request_relative = ipc_root / f"request-{request_id}.json"
        response_relative = ipc_root / f"response-{request_id}.json"
        request_receipt = self._artifact_store.write_json(
            request_relative,
            ipc_request_payload(request),
            schema_version=request.schema_version,
            upstream_sha256=asr_key,
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
                    timeout_seconds=self._timeout_seconds,
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
                    if failure.request_id != request_id:
                        raise ValueError("响应请求 ID 不匹配")
                    if failure.error_code not in ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES:
                        raise ValueError("响应错误码不在语音子进程白名单")
                    stable_message = speech_subprocess_failure_message(failure.error_code)
                    if failure.message != stable_message:
                        raise ValueError("响应错误消息不是稳定协议消息")
                    raise VideoDemoError(failure.error_code, stable_message)
                response = SpeechSubprocessSuccess.model_validate(payload)
                if response.request_id != request_id:
                    raise ValueError("响应请求 ID 不匹配")
                loaded = self._snapshot_store.load(
                    run_root,
                    "speech",
                    response.speech_fingerprint,
                    SpeechAnalysisSnapshotPayload,
                )
                if loaded is None or loaded[1] != response.payload_receipt:
                    raise ValueError("响应回执未绑定当前完整快照")
                return loaded[0].to_analysis()
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
                    ) from None
                raise
            except (OSError, ValueError, ValidationError):
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                    "语音子进程响应非法",
                ) from None
        finally:
            self._artifact_store.discard(request_receipt)
            if response_receipt is not None:
                self._artifact_store.discard(response_receipt)

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
            raise VideoDemoError(ErrorCode.JOB_CANCELLED, "语音分析已取消") from None
        if error.code == ErrorCode.VIDEO_PROCESS_TIMEOUT:
            raise VideoDemoError(
                ErrorCode.SPEECH_SUBPROCESS_TIMEOUT,
                "语音子进程执行超时",
            ) from None
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_CRASHED,
            "语音子进程执行失败",
        ) from None
