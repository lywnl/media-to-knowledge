from __future__ import annotations

from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from video_demo.domain.base import FrozenModel
from video_demo.errors import ErrorCode, VideoDemoError

_CLOUD_ASR_MAX_WINDOW_MS = 600_000
_CLOUD_ASR_OVERLAP_MS = 1_000
_RETIRED_LOCAL_MODEL_DOTENV_KEYS = frozenset(
    {
        "video_demo_inference_device",
        "video_demo_whisper_compute_type",
        "video_demo_whisper_model_id",
        "video_demo_speech_enrichment_timeout_seconds",
        "video_demo_huggingface_token",
    },
)


class _RetiredLocalModelDotenvFilter(PydanticBaseSettingsSource):
    """只在 dotenv 边界丢弃迁移前已退役的本地模型配置。"""

    def __init__(self, source: PydanticBaseSettingsSource) -> None:
        super().__init__(source.settings_cls)
        self._source = source

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return self._source.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        self._source._set_current_state(self.current_state)
        self._source._set_settings_sources_data(self.settings_sources_data)
        values = self._source()
        return {
            key: value
            for key, value in values.items()
            if key.casefold() not in _RETIRED_LOCAL_MODEL_DOTENV_KEYS
        }


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


class CloudAsrConfiguration(FrozenModel):
    """已校验且可直接交给云端 ASR 适配器的运行配置。"""

    base_url: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    model: str
    timeout_seconds: float
    max_attempts: int
    max_window_ms: int
    overlap_ms: int


class Settings(BaseSettings):
    """只从显式参数或 `VIDEO_DEMO_` 环境变量加载的运行配置。"""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_DEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        hide_input_in_errors=True,
        validate_default=True,
    )

    workspace_root: Path = Field(default_factory=Path.cwd)
    runtime_root: Path | None = None
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None

    worker_concurrency: int = Field(default=1, ge=1, le=4)
    process_timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    speech_subprocess_timeout_seconds: int = Field(default=3_600, ge=1, le=7_200)
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
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "openai_base_url"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    openai_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_MODEL", "openai_model"),
    )
    openai_asr_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        allow_inf_nan=False,
        validation_alias=AliasChoices(
            "OPENAI_ASR_TIMEOUT_SECONDS",
            "openai_asr_timeout_seconds",
        ),
    )
    openai_asr_max_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        validation_alias=AliasChoices(
            "OPENAI_ASR_MAX_ATTEMPTS",
            "openai_asr_max_attempts",
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls
        return (
            init_settings,
            env_settings,
            _RetiredLocalModelDotenvFilter(dotenv_settings),
            file_secret_settings,
        )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_openai_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, SecretStr):
            normalized = value.get_secret_value().strip()
            return SecretStr(normalized) if normalized else None
        return value

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

    def require_cloud_asr_configuration(self) -> CloudAsrConfiguration:
        """返回完整云端 ASR 配置；错误信息不得包含凭据或原始 URL。"""

        base_url = _normalize_cloud_asr_base_url(self.openai_base_url)
        api_key = self.openai_api_key
        if api_key is None:
            raise _invalid_cloud_asr_configuration("openai_api_key")
        model = (self.openai_model or "").strip()
        if not model:
            raise _invalid_cloud_asr_configuration("openai_model")
        return CloudAsrConfiguration(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=self.openai_asr_timeout_seconds,
            max_attempts=self.openai_asr_max_attempts,
            max_window_ms=_CLOUD_ASR_MAX_WINDOW_MS,
            overlap_ms=_CLOUD_ASR_OVERLAP_MS,
        )

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


def _normalize_cloud_asr_base_url(value: str | None) -> str:
    normalized = (value or "").strip().rstrip("/")
    if (
        not normalized
        or any(character.isspace() or ord(character) < 32 for character in normalized)
        or "?" in normalized
        or "#" in normalized
    ):
        raise _invalid_cloud_asr_configuration("openai_base_url")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError as error:
        raise _invalid_cloud_asr_configuration("openai_base_url") from error
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/").casefold().endswith("/audio/transcriptions")
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise _invalid_cloud_asr_configuration("openai_base_url")
    return normalized


def _invalid_cloud_asr_configuration(field: str) -> VideoDemoError:
    return VideoDemoError(
        ErrorCode.INVALID_CONFIGURATION,
        "云端语音识别配置无效",
        {"field": field},
    )
