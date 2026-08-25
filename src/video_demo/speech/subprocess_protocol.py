from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator

from video_demo.application.pipeline import PipelineRunConfig
from video_demo.domain.base import FrozenModel, Sha256, StableId
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode
from video_demo.speech.snapshots import SpeechFingerprintInputs
from video_demo.storage.artifacts import ArtifactReceipt

ALLOWED_SPEECH_SUBPROCESS_FAILURE_CODES = frozenset(
    {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
        ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        ErrorCode.SPEECH_AUTHENTICATION_FAILED,
        ErrorCode.SPEECH_AUDIO_INVALID,
        ErrorCode.JOB_CANCELLED,
        ErrorCode.WORKSPACE_PATH_ESCAPE,
        ErrorCode.VIDEO_DIGEST_MISMATCH,
        ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
        # 语音阶段会调用 FFmpeg 切片；这些媒体错误必须透传，不能被误报为模型不可用。
        ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT,
        ErrorCode.VIDEO_PROCESS_FAILED,
        ErrorCode.VIDEO_PROCESS_TIMEOUT,
        ErrorCode.VIDEO_OUTPUT_INVALID,
        ErrorCode.VIDEO_OUTPUT_TOO_LARGE,
        ErrorCode.VIDEO_INPUT_INVALID,
    }
)


class SpeechSubprocessCredentials(FrozenModel):
    openai_api_key: SecretStr = Field(repr=False)


class SpeechRuntimeConfig(FrozenModel):
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=256)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_attempts: int = Field(ge=1, le=5)
    max_window_ms: int = Field(gt=0)
    overlap_ms: int = Field(ge=0)
    merge_gap_ms: int = Field(default=2_000, ge=0)
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=44)
    model_identities: tuple[ModelIdentity, ...]
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_merge_gap_ms: int = Field(default=200, ge=0)
    ffmpeg_relative_path: str = ".codex/video-rag-demo/tools/ffmpeg"

    @model_validator(mode="after")
    def validate_ffmpeg_path(self) -> Self:
        path = Path(self.ffmpeg_relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("FFmpeg 路径必须是工作区内相对路径")
        return self

    def fingerprint_inputs(self) -> SpeechFingerprintInputs:
        return SpeechFingerprintInputs(
            model_identities=self.model_identities,
            cloud_asr_base_url=self.base_url,
            max_window_ms=self.max_window_ms,
            overlap_ms=self.overlap_ms,
            merge_gap_ms=self.merge_gap_ms,
            max_upload_bytes=self.max_upload_bytes,
            vad_threshold=self.vad_threshold,
            vad_merge_gap_ms=self.vad_merge_gap_ms,
        )


class SpeechSubprocessRequest(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request_id: StableId
    run_relative_root: str = Field(min_length=1, max_length=1024)
    audio_relative_path: str = Field(min_length=1, max_length=1024)
    audio_sha256: Sha256
    duration_ms: int = Field(gt=0)
    config: PipelineRunConfig = Field(repr=False)
    media_warnings: tuple[str, ...] = ()
    runtime: SpeechRuntimeConfig
    credentials: SpeechSubprocessCredentials
    asr_fingerprint: Sha256

    @model_validator(mode="after")
    def validate_relative_paths(self) -> Self:
        run_root = Path(self.run_relative_root)
        audio = Path(self.audio_relative_path)
        if (
            run_root.is_absolute()
            or audio.is_absolute()
            or ".." in run_root.parts
            or ".." in audio.parts
            or not audio.is_relative_to(run_root)
        ):
            raise ValueError("语音子进程路径必须属于当前运行目录")
        return self


class SpeechSubprocessSuccess(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    request_id: StableId
    asr_fingerprint: Sha256
    payload_receipt: ArtifactReceipt


class SpeechSubprocessFailure(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["FAILED"] = "FAILED"
    request_id: StableId
    asr_fingerprint: Sha256
    error_code: ErrorCode
    message: str = Field(min_length=1, max_length=200)


SpeechSubprocessResponse = SpeechSubprocessSuccess | SpeechSubprocessFailure


def ipc_request_payload(request: SpeechSubprocessRequest) -> dict[str, object]:
    """仅供写入 0600 IPC 文件；此处是唯一允许解封子进程凭据的编码边界。"""

    payload = request.model_dump(
        mode="json",
        exclude={"credentials"},
        exclude_computed_fields=True,
    )
    payload["credentials"] = {
        "openai_api_key": request.credentials.openai_api_key.get_secret_value()
    }
    return payload


def speech_subprocess_failure_message(code: ErrorCode) -> str:
    return {
        ErrorCode.DEPENDENCY_TEMPORARY_FAILURE: "云端语音识别暂时不可用",
        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE: "语音依赖不可用",
        ErrorCode.SPEECH_MODEL_UNAVAILABLE: "语音模型不可用",
        ErrorCode.SPEECH_AUTHENTICATION_FAILED: "语音模型鉴权失败",
        ErrorCode.SPEECH_AUDIO_INVALID: "语音音频非法",
        ErrorCode.JOB_CANCELLED: "语音分析已取消",
        ErrorCode.WORKSPACE_PATH_ESCAPE: "语音运行路径非法",
        ErrorCode.VIDEO_DIGEST_MISMATCH: "语音音频摘要不匹配",
        ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT: "可用磁盘空间不足",
        ErrorCode.VIDEO_PROCESS_FAILED: "FFmpeg 音频切片失败",
        ErrorCode.VIDEO_PROCESS_TIMEOUT: "FFmpeg 音频切片超时",
        ErrorCode.VIDEO_OUTPUT_INVALID: "FFmpeg 输出音频非法",
        ErrorCode.VIDEO_OUTPUT_TOO_LARGE: "FFmpeg 输出超过大小限制",
        ErrorCode.VIDEO_INPUT_INVALID: "FFmpeg 输入音频非法",
        ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID: "语音请求或响应协议非法",
    }.get(code, "语音模型不可用")
