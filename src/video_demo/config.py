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


class TextLlmConfiguration(FrozenModel):
    """已校验且可交给知识文档文本模型客户端的运行配置。"""

    base_url: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    model_id: str
    timeout_seconds: float
    max_attempts: int
    max_input_chars: int
    max_input_bytes: int
    max_response_bytes: int


class VlmConfiguration(FrozenModel):
    """已校验且可交给章节视觉模型客户端的运行配置。"""

    base_url: str
    api_key: SecretStr = Field(exclude=True, repr=False)
    model_id: str
    timeout_seconds: float
    max_attempts: int
    max_image_bytes: int
    max_request_image_bytes: int
    max_encoded_request_bytes: int
    max_inflight_encoded_bytes: int
    concurrency: int


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
    process_timeout_seconds: int = Field(default=600, ge=1, le=14_400)
    speech_subprocess_timeout_seconds: int = Field(default=3_600, ge=1, le=14_400)
    max_video_bytes: int = Field(default=4 * 1024 * 1024 * 1024, ge=1)
    max_video_duration_ms: int = Field(default=1_800_000, ge=1, le=7_200_000)
    # 仅供本地演示显式开启；默认保持严格外部依赖失败语义。
    demo_degraded_mode: bool = False

    allow_insecure_local_model_endpoint: bool = False

    text_llm_base_url: str | None = None
    text_llm_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    text_llm_model_id: str | None = None
    text_llm_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    text_llm_max_attempts: int = Field(default=3, ge=1, le=5)
    text_llm_max_input_chars: int = Field(default=60_000, ge=1)
    text_llm_max_input_bytes: int = Field(default=1 * 1024 * 1024, ge=1)
    model_max_response_bytes: int = Field(default=2 * 1024 * 1024, ge=1)

    vlm_base_url: str | None = None
    vlm_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    vlm_model_id: str = "qwen3-vl-flash"
    vlm_timeout_seconds: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    vlm_max_attempts: int = Field(default=3, ge=1, le=5)
    vlm_max_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    vlm_max_request_image_bytes: int = Field(default=24 * 1024 * 1024, ge=1)
    vlm_max_encoded_request_bytes: int = Field(default=36 * 1024 * 1024, ge=1)
    vlm_max_inflight_encoded_bytes: int = Field(default=72 * 1024 * 1024, ge=1)
    vlm_concurrency: int = Field(default=2, ge=1, le=2)
    chapter_writer_concurrency: int = Field(default=2, ge=1, le=2)
    visual_proxy_max_edge: int = Field(default=1_920, ge=1_280, le=2_560)
    keyframe_jpeg_quality: int = Field(default=90, ge=1, le=100)
    visual_proxy_estimated_bytes_per_second: int = Field(default=2 * 1024 * 1024, ge=1)

    max_transcript_evidence_items: int = Field(default=20_000, ge=1)
    max_transcript_chars: int = Field(default=2_000_000, ge=1)
    max_scene_boundaries: int = Field(default=20_000, ge=1)
    max_base_segments: int = Field(default=20_000, ge=1)
    max_document_chapters: int = Field(default=240, ge=1)
    max_result_evidence_items: int = Field(default=25_000, ge=1)
    max_candidate_frame_bytes_per_run: int = Field(default=512 * 1024 * 1024, ge=1)
    max_published_keyframe_bytes_per_run: int = Field(default=256 * 1024 * 1024, ge=1)
    max_result_bundle_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_document_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    model_cache_max_entry_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    model_cache_max_run_bytes: int = Field(default=256 * 1024 * 1024, ge=1)
    min_free_disk_reserve_bytes: int = Field(default=512 * 1024 * 1024, ge=1)

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
        if (
            self.vlm_concurrency * self.vlm_max_encoded_request_bytes
            > self.vlm_max_inflight_encoded_bytes
        ):
            raise ValueError("视觉模型在途字节预算不足以覆盖最大并发请求")
        self._validate_document_model_configuration_presence()
        self._validate_oss_configuration()
        return self

    def has_complete_oss_configuration(self) -> bool:
        return all(self._oss_configuration_presence())

    def require_text_llm_configuration(self) -> TextLlmConfiguration:
        """返回完整文本 LLM 配置；配置缺失时不在错误中回显密钥或端点。"""

        values = (
            bool(self.text_llm_base_url and self.text_llm_base_url.strip()),
            _has_secret(self.text_llm_api_key),
            bool(self.text_llm_model_id and self.text_llm_model_id.strip()),
        )
        if not any(values):
            raise _invalid_model_configuration("text_llm")
        if not all(values):
            raise _invalid_model_configuration("text_llm")
        assert self.text_llm_api_key is not None
        assert self.text_llm_base_url is not None
        assert self.text_llm_model_id is not None
        return TextLlmConfiguration(
            base_url=_normalize_model_base_url(
                self.text_llm_base_url,
                allow_insecure_local=self.allow_insecure_local_model_endpoint,
            ),
            api_key=self.text_llm_api_key,
            model_id=self.text_llm_model_id.strip(),
            timeout_seconds=self.text_llm_timeout_seconds,
            max_attempts=self.text_llm_max_attempts,
            max_input_chars=self.text_llm_max_input_chars,
            max_input_bytes=self.text_llm_max_input_bytes,
            max_response_bytes=self.model_max_response_bytes,
        )

    def require_vlm_configuration(self) -> VlmConfiguration:
        """返回完整章节视觉模型配置；默认模型名不视为已配置端点。"""

        values = (
            bool(self.vlm_base_url and self.vlm_base_url.strip()),
            _has_secret(self.vlm_api_key),
            bool(self.vlm_model_id and self.vlm_model_id.strip()),
        )
        if not any(values[:2]):
            raise _invalid_model_configuration("vlm")
        if not all(values):
            raise _invalid_model_configuration("vlm")
        assert self.vlm_api_key is not None
        assert self.vlm_base_url is not None
        return VlmConfiguration(
            base_url=_normalize_model_base_url(
                self.vlm_base_url,
                allow_insecure_local=self.allow_insecure_local_model_endpoint,
            ),
            api_key=self.vlm_api_key,
            model_id=self.vlm_model_id.strip(),
            timeout_seconds=self.vlm_timeout_seconds,
            max_attempts=self.vlm_max_attempts,
            max_image_bytes=self.vlm_max_image_bytes,
            max_request_image_bytes=self.vlm_max_request_image_bytes,
            max_encoded_request_bytes=self.vlm_max_encoded_request_bytes,
            max_inflight_encoded_bytes=self.vlm_max_inflight_encoded_bytes,
            concurrency=self.vlm_concurrency,
        )

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

    def _validate_document_model_configuration_presence(self) -> None:
        text_presence = (
            bool(self.text_llm_base_url and self.text_llm_base_url.strip()),
            _has_secret(self.text_llm_api_key),
            bool(self.text_llm_model_id and self.text_llm_model_id.strip()),
        )
        if any(text_presence) and not all(text_presence):
            raise ValueError("文本模型配置必须全部提供或全部留空")
        if all(text_presence):
            try:
                _normalize_model_base_url(
                    self.text_llm_base_url or "",
                    allow_insecure_local=self.allow_insecure_local_model_endpoint,
                )
            except VideoDemoError as error:
                raise ValueError("文本模型端点配置非法") from error
        vlm_presence = (
            bool(self.vlm_base_url and self.vlm_base_url.strip()),
            _has_secret(self.vlm_api_key),
            bool(self.vlm_model_id and self.vlm_model_id.strip()),
        )
        vlm_was_explicitly_configured = any(vlm_presence[:2]) or (
            "vlm_model_id" in self.model_fields_set
        )
        if vlm_was_explicitly_configured and not all(vlm_presence):
            raise ValueError("视觉模型配置必须全部提供或全部留空")
        if all(vlm_presence):
            try:
                _normalize_model_base_url(
                    self.vlm_base_url or "",
                    allow_insecure_local=self.allow_insecure_local_model_endpoint,
                )
            except VideoDemoError as error:
                raise ValueError("视觉模型端点配置非法") from error

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


def _has_secret(value: SecretStr | None) -> bool:
    return bool(value is not None and value.get_secret_value().strip())


def _invalid_model_configuration(component: str) -> VideoDemoError:
    return VideoDemoError(
        ErrorCode.INVALID_CONFIGURATION,
        "知识文档模型配置不完整",
        {"component": component},
    )


def _normalize_model_base_url(value: str, *, allow_insecure_local: bool) -> str:
    normalized = value.strip().rstrip("/")
    if (
        not normalized
        or any(character.isspace() or ord(character) < 32 for character in normalized)
        or "?" in normalized
        or "#" in normalized
    ):
        raise _invalid_model_configuration("endpoint")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError as error:
        raise _invalid_model_configuration("endpoint") from error
    hostname = (parsed.hostname or "").casefold()
    local_hosts = {"127.0.0.1", "::1", "localhost"}
    allowed_http = allow_insecure_local and hostname in local_hosts
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (
            parsed.scheme.casefold() != "https"
            and not (parsed.scheme.casefold() == "http" and allowed_http)
        )
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise _invalid_model_configuration("endpoint")
    return normalized
