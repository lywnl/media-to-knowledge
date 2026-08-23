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

from video_demo.application.pipeline import PreparedMedia, SpeechAnalysis, StageMetric
from video_demo.application.production_speech import ProductionSpeechAnalyzer, SpeechComponents
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import SafeProcessRunner
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    _speech_fingerprint_v1,
    asr_fingerprint,
    speech_fingerprint,
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


@dataclass(frozen=True, slots=True)
class AsrStageResult:
    payload: AsrSnapshotPayload
    receipt: ArtifactReceipt


@dataclass(frozen=True, slots=True)
class EnrichmentStageResult:
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
        timeout_seconds: int | None = None,
        asr_timeout_seconds: int = 1800,
        enrichment_timeout_seconds: int = 600,
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
        self._asr_timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else asr_timeout_seconds
        )
        self._enrichment_timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else enrichment_timeout_seconds
        )
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
        if media.subtitle is not None or media.audio_path is None:
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
        asr_request = SpeechSubprocessRequest(
            request_id=request_id,
            run_relative_root=run_root.as_posix(),
            audio_relative_path=audio_relative.as_posix(),
            audio_sha256=media.audio_sha256,
            duration_ms=media.source.duration_ms,
            config=config,
            media_warnings=media.warnings,
            runtime=self._speech_runtime,
            credentials=SpeechSubprocessCredentials(),
            asr_fingerprint=asr_key,
            allow_speaker_fallback=self._allow_speaker_fallback,
            stage="ASR",
        )
        stage_cache_hits: list[str] = []
        asr_started_at = time.monotonic()
        cached_asr = self._snapshot_store.load(
            run_root, "asr", asr_key, AsrSnapshotPayload
        )
        if cached_asr is None:
            asr_result = self._run_stage(
                asr_request, run_root, is_cancel_requested, self._asr_timeout_seconds
            )
            assert isinstance(asr_result, AsrStageResult)
            asr_duration_ms = _elapsed_ms(asr_started_at)
        else:
            asr_result = AsrStageResult(*cached_asr)
            asr_duration_ms = 0
            stage_cache_hits.append("SPEECH_ASR")
        asr_payload, asr_receipt = asr_result.payload, asr_result.receipt
        if config.speech_enrichment_mode == "text":
            text_key = speech_fingerprint(
                processing_mode="ASR",
                transcript_payload_sha256=asr_receipt.sha256,
                media_warnings=media.warnings,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                allow_speaker_fallback=self._allow_speaker_fallback,
                inputs=self._fingerprint_inputs,
                enrichment_mode="text",
            )
            cached_text = self._snapshot_store.load(
                run_root, "speech", text_key, SpeechAnalysisSnapshotPayload
            )
            if cached_text is not None:
                analysis = cached_text[0].to_analysis()
            else:
                analysis = ProductionSpeechAnalyzer.analysis_from_asr_snapshot(
                    media, asr_payload, enrichment_mode="text"
                )
                self._snapshot_store.publish(
                    run_root,
                    "speech",
                    text_key,
                    SpeechAnalysisSnapshotPayload.from_analysis(analysis),
                )
            return replace(
                analysis,
                stage_metrics=(StageMetric("SPEECH_ASR", asr_duration_ms),),
                stage_cache_hits=tuple(stage_cache_hits),
            )
        speech_key = speech_fingerprint(
            processing_mode="ASR",
            transcript_payload_sha256=asr_receipt.sha256,
            media_warnings=media.warnings,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            allow_speaker_fallback=self._allow_speaker_fallback,
            inputs=self._fingerprint_inputs,
            enrichment_mode="full",
        )
        enrichment_started_at = time.monotonic()
        cached = self._snapshot_store.load(
            run_root, "speech", speech_key, SpeechAnalysisSnapshotPayload
        )
        if cached is None:
            legacy_key = _speech_fingerprint_v1(
                processing_mode="ASR",
                transcript_payload_sha256=asr_receipt.sha256,
                media_warnings=media.warnings,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                allow_speaker_fallback=self._allow_speaker_fallback,
                inputs=self._fingerprint_inputs,
            )
            cached = self._snapshot_store.load(
                run_root, "speech", legacy_key, SpeechAnalysisSnapshotPayload
            )
        if cached is not None:
            stage_cache_hits.append("SPEECH_ENRICHMENT")
            enrichment_duration_ms = 0
            analysis = cached[0].to_analysis()
        else:
            base_payload = asr_request.model_dump(
                mode="python",
                exclude={
                    "request_id",
                    "credentials",
                    "stage",
                    "speech_fingerprint",
                    "asr_payload_receipt",
                },
            )
            enrich_request = SpeechSubprocessRequest.model_validate(
                {
                    **base_payload,
                    "request_id": f"speech_{uuid.uuid4().hex}",
                    "credentials": self._credentials,
                    "stage": "ENRICHMENT",
                    "speech_fingerprint": speech_key,
                    "asr_payload_receipt": asr_receipt,
                }
            )
            enrichment_result = self._run_stage(
                enrich_request, run_root, is_cancel_requested, self._enrichment_timeout_seconds
            )
            assert isinstance(enrichment_result, EnrichmentStageResult)
            enrichment_duration_ms = _elapsed_ms(enrichment_started_at)
            loaded = self._snapshot_store.load(
                run_root, "speech", speech_key, SpeechAnalysisSnapshotPayload
            )
            if loaded is None:
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                    "语音增强子进程未发布完整快照",
                    details={"stage": "ENRICHMENT"},
                )
            analysis = loaded[0].to_analysis()
        return replace(
            analysis,
            stage_metrics=(
                StageMetric("SPEECH_ASR", asr_duration_ms),
                StageMetric("SPEECH_ENRICHMENT", enrichment_duration_ms),
            ),
            stage_cache_hits=tuple(stage_cache_hits),
        )

    def _run_stage(
        self,
        request: SpeechSubprocessRequest,
        run_root: Path,
        is_cancel_requested: Callable[[], bool],
        timeout_seconds: int,
    ) -> AsrStageResult | EnrichmentStageResult:
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
                    timeout_seconds=timeout_seconds,
                    output_paths=(self._runtime_root / response_relative,),
                    env=self._subprocess_environment(),
                )
            except VideoDemoError as error:
                self._map_process_error(error, request.stage)
                raise AssertionError("不可达") from error
            if result.returncode != 0:
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_CRASHED,
                    "语音子进程异常退出",
                    details={"stage": request.stage},
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
                    if failure.stage != request.stage:
                        raise ValueError("失败响应阶段与请求不匹配")
                    if failure.error_code not in ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES:
                        raise ValueError("响应错误码不在语音子进程白名单")
                    stable_message = speech_subprocess_failure_message(failure.error_code)
                    if failure.message != stable_message:
                        raise ValueError("响应错误消息不是稳定协议消息")
                    raise VideoDemoError(
                        failure.error_code,
                        stable_message,
                        details={"stage": request.stage},
                    )
                response = SpeechSubprocessSuccess.model_validate(payload)
                if response.request_id != request_id:
                    raise ValueError("响应请求 ID 不匹配")
                if request.stage == "ASR":
                    if (
                        response.stage != "ASR"
                        or response.speech_fingerprint != request.asr_fingerprint
                    ):
                        raise ValueError("ASR 响应指纹或阶段不匹配")
                    loaded_asr = self._snapshot_store.load(
                        run_root, "asr", request.asr_fingerprint, AsrSnapshotPayload
                    )
                    if loaded_asr is None or loaded_asr[1] != response.payload_receipt:
                        raise ValueError("响应回执未绑定当前 ASR 快照")
                    return AsrStageResult(*loaded_asr)
                if (
                    response.stage != "ENRICHMENT"
                    or response.speech_fingerprint != request.speech_fingerprint
                ):
                    raise ValueError("增强响应指纹或阶段不匹配")
                loaded_speech = self._snapshot_store.load(
                    run_root, "speech", response.speech_fingerprint, SpeechAnalysisSnapshotPayload
                )
                if loaded_speech is None or loaded_speech[1] != response.payload_receipt:
                    raise ValueError("响应回执未绑定当前完整快照")
                return EnrichmentStageResult(response.payload_receipt)
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
                        details={"stage": request.stage},
                    ) from None
                raise
            except (OSError, ValueError, ValidationError):
                raise VideoDemoError(
                    ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
                    "语音子进程响应非法",
                    details={"stage": request.stage},
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
            # 目录 fd 必须先于路径校验取得；校验回调期间即使命名目录被交换，
            # 后续删除仍只作用于已经持有的原 IPC 目录。
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
                if (
                    digest.hexdigest() != expected_sha256
                    or not stat.S_ISREG(current.st_mode)
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
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
    def _map_process_error(error: VideoDemoError, stage: str) -> None:
        if error.code == ErrorCode.VIDEO_PROCESS_CANCELLED:
            raise VideoDemoError(
                ErrorCode.JOB_CANCELLED,
                "语音分析已取消",
                details={"stage": stage},
            ) from None
        if error.code == ErrorCode.VIDEO_PROCESS_TIMEOUT:
            raise VideoDemoError(
                ErrorCode.SPEECH_SUBPROCESS_TIMEOUT,
                "语音子进程执行超时",
                details={"stage": stage},
            ) from None
        raise VideoDemoError(
            ErrorCode.SPEECH_SUBPROCESS_CRASHED,
            "语音子进程执行失败",
            details={"stage": stage},
        ) from None
