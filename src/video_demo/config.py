from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from video_demo.errors import ErrorCode, VideoDemoError


def resolve_workspace_path(root: Path, candidate: Path) -> Path:
    """解析工作区内路径, 并拒绝 `..` 或符号链接造成的目录逃逸。"""

    workspace = root.expanduser().resolve(strict=False)
    unresolved = candidate.expanduser()
    resolved = (
        unresolved.resolve(strict=False)
        if unresolved.is_absolute()
        else (workspace / unresolved).resolve(strict=False)
    )
    if not resolved.is_relative_to(workspace):
        raise VideoDemoError(
            ErrorCode.WORKSPACE_PATH_ESCAPE,
            "路径必须位于项目工作区内",
            {"workspace": str(workspace)},
        )
    return resolved


class Settings(BaseSettings):
    """只从显式参数或 `VIDEO_DEMO_` 环境变量加载的运行配置。"""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_DEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    workspace_root: Path = Field(default_factory=Path.cwd)
    runtime_root: Path | None = None
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None

    inference_device: Literal["cpu", "mps"] = "cpu"
    whisper_compute_type: Literal["int8", "float16", "float32"] = "int8"
    worker_concurrency: int = Field(default=1, ge=1, le=4)
    process_timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    speech_subprocess_timeout_seconds: int = Field(default=1_800, ge=1, le=7_200)
    max_video_bytes: int = Field(default=4 * 1024 * 1024 * 1024, ge=1)
    # 仅供本地演示显式开启；默认保持严格外部依赖失败语义。
    demo_degraded_mode: bool = False

    qwen_base_url: str | None = None
    qwen_model_id: str | None = None
    qwen_api_key: SecretStr | None = Field(default=None, exclude=True)
    qwen_max_video_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    qwen_max_video_duration_ms: int = Field(default=30_000, ge=1, le=30_000)
    qwen_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    oss_endpoint: str | None = None
    oss_bucket: str | None = None
    oss_access_key_id: SecretStr | None = Field(default=None, exclude=True)
    oss_access_key_secret: SecretStr | None = Field(default=None, exclude=True)
    oss_prefix: str = "video-demo/qwen-clips"
    oss_signed_url_ttl_seconds: int = Field(default=3_600, ge=60, le=86_400)
    baidu_ocr_endpoint: str = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    baidu_api_key: SecretStr | None = Field(default=None, exclude=True)
    baidu_secret_key: SecretStr | None = Field(default=None, exclude=True)
    huggingface_token: SecretStr | None = Field(default=None, exclude=True)

    @field_validator("oss_prefix")
    @classmethod
    def validate_oss_prefix(cls, value: str) -> str:
        normalized = value.strip("/")
        parts = normalized.split("/")
        if (
            not normalized
            or "\\" in normalized
            or any(not part or part in {".", ".."} for part in parts)
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        ):
            raise ValueError("OSS 对象前缀非法")
        return normalized

    @model_validator(mode="after")
    def normalize_workspace_paths(self) -> Self:
        workspace = self.workspace_root.expanduser().resolve(strict=False)
        runtime_candidate = self.runtime_root or Path(".codex/video-rag-demo")
        runtime = resolve_workspace_path(workspace, runtime_candidate)
        ffmpeg = (
            resolve_workspace_path(workspace, self.ffmpeg_path)
            if self.ffmpeg_path is not None
            else None
        )
        ffprobe = (
            resolve_workspace_path(workspace, self.ffprobe_path)
            if self.ffprobe_path is not None
            else None
        )
        object.__setattr__(self, "workspace_root", workspace)
        object.__setattr__(self, "runtime_root", runtime)
        object.__setattr__(self, "ffmpeg_path", ffmpeg)
        object.__setattr__(self, "ffprobe_path", ffprobe)
        self._validate_oss_configuration()
        return self

    def has_complete_oss_configuration(self) -> bool:
        return all(self._oss_configuration_presence())

    def _validate_oss_configuration(self) -> None:
        presence = self._oss_configuration_presence()
        if any(presence) and not all(presence):
            raise ValueError("OSS 配置必须全部提供或全部留空")

    def _oss_configuration_presence(self) -> tuple[bool, bool, bool, bool]:
        return (
            bool(self.oss_endpoint and self.oss_endpoint.strip()),
            bool(self.oss_bucket and self.oss_bucket.strip()),
            bool(
                self.oss_access_key_id
                and self.oss_access_key_id.get_secret_value().strip()
            ),
            bool(
                self.oss_access_key_secret
                and self.oss_access_key_secret.get_secret_value().strip()
            ),
        )
