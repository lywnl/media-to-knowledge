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
        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE,
        ErrorCode.SPEECH_MODEL_UNAVAILABLE,
        ErrorCode.SPEECH_AUTHENTICATION_FAILED,
        ErrorCode.SPEECH_AUDIO_INVALID,
        ErrorCode.PYANNOTE_AUTHENTICATION_FAILED,
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE,
        ErrorCode.JOB_CANCELLED,
        ErrorCode.WORKSPACE_PATH_ESCAPE,
        ErrorCode.VIDEO_DIGEST_MISMATCH,
        ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID,
    }
)


class SpeechSubprocessCredentials(FrozenModel):
    huggingface_token: SecretStr | None = Field(default=None, repr=False)


class SpeechRuntimeConfig(FrozenModel):
    inference_device: Literal["cpu", "mps"]
    whisper_compute_type: str = Field(min_length=1, max_length=32)
    model_identities: tuple[ModelIdentity, ...]
    yamnet_class_map_sha256: Sha256
    yamnet_thresholds_sha256: Sha256
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_merge_gap_ms: int = Field(default=200, ge=0)
    lid_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    asr_beam_size: int = Field(default=5, ge=1)
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
            vad_threshold=self.vad_threshold,
            vad_merge_gap_ms=self.vad_merge_gap_ms,
            lid_threshold=self.lid_threshold,
            asr_beam_size=self.asr_beam_size,
            asr_compute_type=self.whisper_compute_type,
            yamnet_class_map_sha256=self.yamnet_class_map_sha256,
            yamnet_thresholds_sha256=self.yamnet_thresholds_sha256,
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
    allow_speaker_fallback: bool = False
    stage: Literal["ASR", "ENRICHMENT"]
    speech_fingerprint: Sha256 | None = None
    asr_payload_receipt: ArtifactReceipt | None = None

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
        if self.stage == "ASR":
            token = self.credentials.huggingface_token
            if token is not None and token.get_secret_value():
                raise ValueError("ASR 请求不得携带 Hugging Face Token")
            if self.speech_fingerprint is not None or self.asr_payload_receipt is not None:
                raise ValueError("ASR 请求不得携带增强目标或上游回执")
        else:
            if self.config.speech_enrichment_mode != "full":
                raise ValueError("ENRICHMENT 请求必须使用 full 模式")
            if self.speech_fingerprint is None or self.asr_payload_receipt is None:
                raise ValueError("ENRICHMENT 请求必须携带目标指纹和 ASR 回执")
            if self.asr_payload_receipt.upstream_sha256 != self.asr_fingerprint:
                raise ValueError("ENRICHMENT 请求上游回执必须绑定 ASR 指纹")
        return self


class SpeechSubprocessSuccess(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    request_id: StableId
    stage: Literal["ASR", "ENRICHMENT"]
    speech_fingerprint: Sha256
    payload_receipt: ArtifactReceipt


class SpeechSubprocessFailure(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["FAILED"] = "FAILED"
    request_id: StableId
    stage: Literal["ASR", "ENRICHMENT"]
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
        "huggingface_token": (
            request.credentials.huggingface_token.get_secret_value()
            if request.credentials.huggingface_token is not None
            else None
        )
    }
    return payload


def speech_subprocess_failure_message(code: ErrorCode) -> str:
    return {
        ErrorCode.SPEECH_DEPENDENCY_UNAVAILABLE: "语音依赖不可用",
        ErrorCode.SPEECH_MODEL_UNAVAILABLE: "语音模型不可用",
        ErrorCode.SPEECH_AUTHENTICATION_FAILED: "语音模型鉴权失败",
        ErrorCode.SPEECH_AUDIO_INVALID: "语音音频非法",
        ErrorCode.PYANNOTE_AUTHENTICATION_FAILED: "说话人模型鉴权失败",
        ErrorCode.PYANNOTE_MODEL_UNAVAILABLE: "说话人模型不可用",
        ErrorCode.JOB_CANCELLED: "语音分析已取消",
        ErrorCode.WORKSPACE_PATH_ESCAPE: "语音运行路径非法",
        ErrorCode.VIDEO_DIGEST_MISMATCH: "语音音频摘要不匹配",
        ErrorCode.SPEECH_SUBPROCESS_RESPONSE_INVALID: "语音请求或响应协议非法",
    }.get(code, "语音模型不可用")
