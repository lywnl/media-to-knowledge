from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, Protocol, cast

import httpx
from pydantic import SecretStr

from video_demo.application.pipeline import (
    AssetProbe,
    AssetRegistrar,
    MediaTranscoder,
    PipelineJobHandler,
    PreparedMedia,
    SpeechAnalyzer,
    VideoUnderstandingPipeline,
    VisualAnalyzer,
)
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
    TranscodeClient,
    build_ffmpeg_factory,
    build_ffprobe_factory,
)
from video_demo.application.production_speech import AsrComponentFactory
from video_demo.application.production_visual import (
    LazyBaiduOcrClient,
    ProductionVisualAnalyzer,
    VisualComponents,
)
from video_demo.application.queries import ResultQueryService
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.run import ModelIdentity
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.oss import (
    OssTemporaryVideoPublisher,
    PublishedVideoUnderstanding,
    oss_remote_host,
)
from video_demo.integrations.qwen import DemoFallbackVideoUnderstanding, QwenVideoClient
from video_demo.integrations.video_port import WholeVideoUnderstandingPort
from video_demo.persistence.database import Database
from video_demo.speech.isolated import IsolatedSpeechAnalyzer
from video_demo.speech.runtime import ProductionSpeechModels
from video_demo.speech.runtime import (
    build_diagnostic_speech_models as _build_speech_models,
)
from video_demo.speech.runtime import (
    build_speech_component_factory as _build_speech_component_factory,
)
from video_demo.speech.snapshots import SpeechFingerprintInputs
from video_demo.speech.subprocess_protocol import (
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
)
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.object_store import LocalVideoObjectStore
from video_demo.storage.snapshots import SnapshotStore
from video_demo.visual.keyframes import KeyframeSelector, OpenCvFrameExtractor
from video_demo.visual.scenes import PySceneDetectAdapter
from video_demo.worker.runtime import ReliableWorker


class Closable(Protocol):
    def close(self) -> None: ...


_QWEN_MODEL_ID_PATTERN = re.compile(
    r"qwen(?:2(?:\.5)?|3)-vl-(?:plus|max|flash)"
    r"(?:-[0-9]{4}-[0-9]{2}-[0-9]{2})?\Z",
)
class ProductionModelIdentityReport(FrozenModel):
    """由唯一生产组合根确定且不含敏感值的模型身份。"""

    schema_version: Literal["1.0.0", "2.0.0"]
    models: tuple[ModelIdentity, ...]
    settings_fingerprint: Sha256


class ProductionDiagnosticComponents:
    """持有可供生产流水线和诊断入口共同消费的组件与资源。"""

    def __init__(
        self,
        *,
        ffmpeg_factory: Callable[[Callable[[], bool]], TranscodeClient],
        speech_component_factory: AsrComponentFactory,
        visual_component_factory: ProductionVisualComponentFactory,
        speech_models: ProductionSpeechModels,
        qwen_client: QwenVideoClient,
        qwen_http_client: httpx.Client,
        model_identity_report: ProductionModelIdentityReport,
        owned_resources: tuple[Closable, ...],
    ) -> None:
        self.ffmpeg_factory = ffmpeg_factory
        self.speech_component_factory = speech_component_factory
        self.visual_component_factory = visual_component_factory
        self.speech_models = speech_models
        self.qwen_client = qwen_client
        self.model_identity_report = model_identity_report
        self.baidu_http_client = visual_component_factory.http_client
        self.baidu_ocr_client = visual_component_factory.ocr_client
        self.qwen_http_client = qwen_http_client
        self.models = model_identity_report.models
        self._resource_stack = ExitStack()
        for resource in owned_resources:
            self._resource_stack.callback(resource.close)

    def close(self) -> None:
        self._resource_stack.close()


class ProductionPipeline(VideoUnderstandingPipeline):
    """为唯一完整流水线增加可观察生产 Port 与幂等资源所有权。"""

    def __init__(
        self,
        registrar: AssetRegistrar,
        probe: AssetProbe,
        transcoder: MediaTranscoder,
        speech_analyzer: SpeechAnalyzer,
        visual_analyzer: VisualAnalyzer,
        understanding: WholeVideoUnderstandingPort,
        *,
        owned_resources: tuple[Closable, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            registrar,
            probe,
            transcoder,
            speech_analyzer,
            visual_analyzer,
            understanding,
            clock=clock,
        )
        self.registrar = registrar
        self.probe = probe
        self.transcoder = transcoder
        self.speech_analyzer = speech_analyzer
        self.visual_analyzer = visual_analyzer
        self.understanding = understanding
        self._resource_stack = ExitStack()
        for resource in owned_resources:
            self._resource_stack.callback(resource.close)

    def close(self) -> None:
        self._resource_stack.close()


def build_production_model_identity_report(
    settings: Settings,
) -> ProductionModelIdentityReport:
    """只从生产设置和固定组件契约生成稳定、可序列化的模型身份。"""

    cloud_asr = settings.require_cloud_asr_configuration()
    models = [
        _local_model_identity(
            "silero_vad",
            "silero-vad",
            package="silero-vad",
            device="cpu",
        ),
        ModelIdentity(
            component="cloud_whisper",
            provider="openai_compatible",
            model_id=cloud_asr.model,
        ),
        ModelIdentity(
            component="baidu_ocr",
            provider="baidu_ocr",
            model_id="accurate_basic",
        ),
    ]
    qwen_model_id = _normalized_qwen_model_id(
        settings.qwen_model_id,
        allow_unrecognized=settings.demo_degraded_mode,
    )
    if qwen_model_id is not None:
        models.append(
            ModelIdentity(
                component="qwen",
                provider="qwen",
                model_id=qwen_model_id,
            ),
        )
    return ProductionModelIdentityReport(
        schema_version="2.0.0",
        models=tuple(models),
        settings_fingerprint=_settings_fingerprint(settings),
    )


def build_production_diagnostic_components(
    settings: Settings,
    *,
    allowed_remote_video_hosts: frozenset[str] | None = None,
) -> ProductionDiagnosticComponents:
    """构造唯一生产诊断组合根，Secret 仅在 Adapter 首次真实调用时解封。"""

    settings.require_cloud_asr_configuration()
    assert settings.runtime_root is not None
    ffmpeg = settings.ffmpeg_path or settings.runtime_root / "tools" / "ffmpeg"
    ffmpeg_factory = build_ffmpeg_factory(
        settings.workspace_root,
        settings.runtime_root,
        ffmpeg,
    )
    with ExitStack() as pending_resources:
        visual_factory = _build_visual_component_factory(settings, ffmpeg_factory)
        pending_resources.callback(visual_factory.http_client.close)
        speech_http_client = httpx.Client()
        pending_resources.callback(speech_http_client.close)
        qwen_http_client = httpx.Client()
        pending_resources.callback(qwen_http_client.close)

        def qwen_api_key_provider() -> SecretStr | None:
            return settings.qwen_api_key

        qwen = QwenVideoClient(
            qwen_http_client,
            base_url=settings.qwen_base_url,
            api_key=None,
            api_key_provider=qwen_api_key_provider,
            model_id=settings.qwen_model_id,
            allowed_video_root=settings.runtime_root,
            max_video_bytes=settings.qwen_max_video_bytes,
            max_video_duration_ms=settings.qwen_max_video_duration_ms,
            timeout_seconds=settings.qwen_timeout_seconds,
            allowed_remote_video_hosts=allowed_remote_video_hosts,
        )
        speech_models = _build_speech_models(settings, speech_http_client)
        speech_component_factory = _build_speech_component_factory(
            settings,
            ffmpeg_factory,
            models=speech_models,
        )
        components = ProductionDiagnosticComponents(
            ffmpeg_factory=ffmpeg_factory,
            speech_component_factory=speech_component_factory,
            visual_component_factory=visual_factory,
            speech_models=speech_models,
            qwen_client=qwen,
            qwen_http_client=qwen_http_client,
            model_identity_report=build_production_model_identity_report(settings),
            owned_resources=(
                visual_factory.http_client,
                speech_http_client,
                qwen_http_client,
            ),
        )
        # 所有权已经转交给 components，避免 ExitStack 与 owner 双重关闭同一资源。
        pending_resources.pop_all()
    return components


def _local_model_identity(
    component: str,
    model_id: str,
    *,
    package: str,
    device: str,
) -> ModelIdentity:
    return ModelIdentity(
        component=component,
        provider="local",
        model_id=model_id,
        revision=_installed_package_version(package),
        device=device,
    )


def _installed_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _normalized_qwen_model_id(
    value: str | None,
    *,
    allow_unrecognized: bool = False,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if not _QWEN_MODEL_ID_PATTERN.fullmatch(normalized) and not allow_unrecognized:
        raise ValueError("Qwen 模型 ID 必须是受限稳定标识")
    return normalized


def _settings_fingerprint(settings: Settings) -> str:
    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root
    cloud_asr = settings.require_cloud_asr_configuration()
    ffmpeg = settings.ffmpeg_path or runtime_root / "tools" / "ffmpeg"
    ffprobe = settings.ffprobe_path or runtime_root / "tools" / "ffprobe"
    payload = {
        "schema_version": "1.0.0",
        "paths": {
            "runtime_root": _workspace_relative(settings, runtime_root),
            "ffmpeg": _workspace_relative(settings, ffmpeg),
            "ffprobe": _workspace_relative(settings, ffprobe),
        },
        "execution": {
            "max_video_bytes": settings.max_video_bytes,
            "demo_degraded_mode": settings.demo_degraded_mode,
            "speech_subprocess_timeout_seconds": settings.speech_subprocess_timeout_seconds,
        },
        "cloud_asr": {
            "provider": "openai_compatible",
            "base_url": cloud_asr.base_url,
            "model_id": cloud_asr.model,
            "timeout_seconds": cloud_asr.timeout_seconds,
            "max_attempts": cloud_asr.max_attempts,
            "max_window_ms": cloud_asr.max_window_ms,
            "overlap_ms": cloud_asr.overlap_ms,
            "window_strategy_version": "1.0.0",
        },
        "qwen": {
            "base_url": _normalized_endpoint(settings.qwen_base_url),
            "model_id": _normalized_qwen_model_id(
                settings.qwen_model_id,
                allow_unrecognized=settings.demo_degraded_mode,
            ),
            "max_video_bytes": settings.qwen_max_video_bytes,
            "max_video_duration_ms": settings.qwen_max_video_duration_ms,
            "timeout_seconds": settings.qwen_timeout_seconds,
        },
        "oss": {
            "endpoint": _normalized_endpoint(settings.oss_endpoint),
            "bucket": settings.oss_bucket,
            "prefix": settings.oss_prefix,
            "signed_url_ttl_seconds": settings.oss_signed_url_ttl_seconds,
        },
        "baidu_ocr": {
            "endpoint": settings.baidu_ocr_endpoint,
            "model_id": "accurate_basic",
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _workspace_relative(settings: Settings, path: Path) -> str:
    return path.resolve(strict=False).relative_to(settings.workspace_root).as_posix()


def _normalized_endpoint(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    return normalized or None


def build_production_pipeline(
    settings: Settings,
    database: Database,
    object_store: LocalVideoObjectStore,
) -> ProductionPipeline:
    """组装工作区内安全媒体工具与延迟验证配置的 Qwen Adapter。"""

    settings.require_cloud_asr_configuration()
    assert settings.runtime_root is not None
    ffmpeg = settings.ffmpeg_path or settings.runtime_root / "tools" / "ffmpeg"
    ffprobe = settings.ffprobe_path or settings.runtime_root / "tools" / "ffprobe"
    qwen_configured = _has_complete_qwen_configuration(settings)
    if qwen_configured and not settings.has_complete_oss_configuration():
        raise VideoDemoError(
            ErrorCode.OSS_CONFIGURATION_INVALID,
            "生产 Qwen 视频链路要求完整 OSS 配置",
        )
    oss_host = _configured_oss_host(settings) if qwen_configured else None
    with ExitStack() as pending_resources:
        ffmpeg_factory = build_ffmpeg_factory(
            settings.workspace_root,
            settings.runtime_root,
            ffmpeg,
        )
        visual_factory = _build_visual_component_factory(settings, ffmpeg_factory)
        pending_resources.callback(visual_factory.http_client.close)
        qwen_http_client = httpx.Client()
        pending_resources.callback(qwen_http_client.close)

        def qwen_api_key_provider() -> SecretStr | None:
            return settings.qwen_api_key

        qwen = QwenVideoClient(
            qwen_http_client,
            base_url=settings.qwen_base_url,
            api_key=None,
            api_key_provider=qwen_api_key_provider,
            model_id=settings.qwen_model_id,
            allowed_video_root=settings.runtime_root,
            max_video_bytes=settings.qwen_max_video_bytes,
            max_video_duration_ms=settings.qwen_max_video_duration_ms,
            timeout_seconds=settings.qwen_timeout_seconds,
            allowed_remote_video_hosts=(
                frozenset({oss_host}) if oss_host is not None else None
            ),
        )
        understanding: WholeVideoUnderstandingPort = qwen
        if qwen_configured:
            assert settings.oss_endpoint is not None
            assert settings.oss_bucket is not None
            assert settings.oss_access_key_id is not None
            assert settings.oss_access_key_secret is not None
            understanding = PublishedVideoUnderstanding(
                understanding,
                OssTemporaryVideoPublisher(
                    qwen_http_client,
                    endpoint=settings.oss_endpoint,
                    bucket=settings.oss_bucket,
                    access_key_id=settings.oss_access_key_id,
                    access_key_secret=settings.oss_access_key_secret,
                    allowed_video_root=settings.runtime_root,
                    prefix=settings.oss_prefix,
                    signed_url_ttl_seconds=settings.oss_signed_url_ttl_seconds,
                ),
            )
        if settings.demo_degraded_mode:
            understanding = DemoFallbackVideoUnderstanding(understanding)
        pipeline = ProductionPipeline(
            ProductionAssetRegistrar(database, object_store),
            ProductionAssetProbe(build_ffprobe_factory(settings.workspace_root, ffprobe)),
            ProductionMediaTranscoder(
                settings.runtime_root,
                ffmpeg_factory,
                max_proxy_bytes=settings.max_video_bytes,
            ),
            IsolatedSpeechAnalyzer(
                workspace_root=settings.workspace_root,
                runtime_root=settings.runtime_root,
                snapshot_store=SnapshotStore(AtomicArtifactStore(settings.runtime_root)),
                artifact_store=AtomicArtifactStore(settings.runtime_root),
                speech_runtime=_speech_runtime_config(settings, ffmpeg),
                credentials=SpeechSubprocessCredentials(
                    openai_api_key=settings.require_cloud_asr_configuration().api_key,
                ),
                asr_timeout_seconds=settings.speech_subprocess_timeout_seconds,
            ),
            ProductionVisualAnalyzer(
                settings.runtime_root,
                visual_factory,
                max_video_bytes=settings.max_video_bytes,
            ),
            understanding,
            owned_resources=(visual_factory.http_client, qwen_http_client),
        )
        pending_resources.pop_all()
    return pipeline


def _has_complete_qwen_configuration(settings: Settings) -> bool:
    return bool(
        settings.qwen_base_url
        and settings.qwen_base_url.strip()
        and settings.qwen_model_id
        and settings.qwen_model_id.strip()
        and settings.qwen_api_key
        and settings.qwen_api_key.get_secret_value().strip()
    )


def _configured_oss_host(settings: Settings) -> str:
    if settings.oss_endpoint is None or settings.oss_bucket is None:
        raise VideoDemoError(ErrorCode.OSS_CONFIGURATION_INVALID, "OSS 配置非法")
    return oss_remote_host(settings.oss_endpoint, settings.oss_bucket)


def _speech_fingerprint_inputs(settings: Settings) -> SpeechFingerprintInputs:
    configuration = settings.require_cloud_asr_configuration()
    return SpeechFingerprintInputs(
        model_identities=(
            _local_model_identity(
                "silero_vad",
                "silero-vad",
                package="silero-vad",
                device="cpu",
            ),
            ModelIdentity(
                component="cloud_whisper",
                provider="openai_compatible",
                model_id=configuration.model,
            ),
        ),
        cloud_asr_base_url=configuration.base_url,
        max_window_ms=configuration.max_window_ms,
        overlap_ms=configuration.overlap_ms,
    )


def _speech_runtime_config(settings: Settings, ffmpeg: Path) -> SpeechRuntimeConfig:
    configuration = settings.require_cloud_asr_configuration()
    inputs = _speech_fingerprint_inputs(settings)
    return SpeechRuntimeConfig(
        base_url=configuration.base_url,
        model=configuration.model,
        timeout_seconds=configuration.timeout_seconds,
        max_attempts=configuration.max_attempts,
        max_window_ms=configuration.max_window_ms,
        overlap_ms=configuration.overlap_ms,
        model_identities=inputs.model_identities,
        vad_threshold=inputs.vad_threshold,
        vad_merge_gap_ms=inputs.vad_merge_gap_ms,
        ffmpeg_relative_path=ffmpeg.relative_to(settings.workspace_root).as_posix(),
    )


def _optional_file_sha256(path: Path) -> str:
    if path.is_file() and not path.is_symlink():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(f"MISSING:{path.name}".encode()).hexdigest()


def _build_visual_component_factory(
    settings: Settings,
    ffmpeg_factory: object,
) -> ProductionVisualComponentFactory:
    """复用无任务状态的视觉模型与 OCR 会话，任务级状态按调用新建。"""

    assert settings.runtime_root is not None
    def credentials() -> tuple[str | None, str | None]:
        api_key = (
            settings.baidu_api_key.get_secret_value()
            if settings.baidu_api_key is not None
            else None
        )
        secret_key = (
            settings.baidu_secret_key.get_secret_value()
            if settings.baidu_secret_key is not None
            else None
        )
        return api_key, secret_key

    return ProductionVisualComponentFactory(
        settings.runtime_root,
        ffmpeg_factory,
        credentials,
        endpoint=settings.baidu_ocr_endpoint,
    )


class ProductionVisualComponentFactory:
    """拥有生命周期级视觉组件；任务级调用只创建 clip client。"""

    def __init__(
        self,
        runtime_root: Path,
        ffmpeg_factory: object,
        credentials_provider: Callable[[], tuple[str | None, str | None]],
        *,
        endpoint: str,
    ) -> None:
        with ExitStack() as pending_resources:
            http_client = httpx.Client()
            pending_resources.callback(http_client.close)
            scene_detector = PySceneDetectAdapter()
            frame_extractor = OpenCvFrameExtractor(runtime_root)
            selector = KeyframeSelector()
            ocr_client = LazyBaiduOcrClient(
                http_client,
                credentials_provider,
                endpoint=endpoint,
            )
            self.http_client = http_client
            self._ffmpeg_factory = ffmpeg_factory
            self._scene_detector = scene_detector
            self._frame_extractor = frame_extractor
            self._selector = selector
            self._ocr_client = ocr_client
            pending_resources.pop_all()

    @property
    def ocr_client(self) -> LazyBaiduOcrClient:
        return self._ocr_client

    def __call__(
        self,
        _media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> VisualComponents:
        from video_demo.application.production_media import TranscodeClient

        factory = cast(
            Callable[[Callable[[], bool]], TranscodeClient],
            self._ffmpeg_factory,
        )
        return VisualComponents(
            scene_detector=self._scene_detector,
            frame_extractor=self._frame_extractor,
            keyframe_selector=self._selector,
            ocr_client=self._ocr_client,
            clip_client=factory(is_cancel_requested),  # type: ignore[arg-type]
        )


def build_worker(settings: Settings, *, worker_id: str) -> ReliableWorker:
    """用与 API 相同的数据库和运行目录组装可执行 Worker。"""

    settings.require_cloud_asr_configuration()
    assert settings.runtime_root is not None
    settings.runtime_root.mkdir(parents=True, exist_ok=True)
    database = Database(f"sqlite+pysqlite:///{settings.runtime_root / 'video-demo.db'}")
    database.create_schema()
    object_store = LocalVideoObjectStore(
        settings.runtime_root,
        max_video_bytes=settings.max_video_bytes,
    )
    with ExitStack() as pending_resources:
        pipeline = build_production_pipeline(settings, database, object_store)
        pending_resources.callback(pipeline.close)
        handler = PipelineJobHandler(
            database,
            pipeline,
            ResultQueryService(database, AtomicArtifactStore(settings.runtime_root)),
        )
        worker = ReliableWorker(database, worker_id, handler, owned_resources=(pipeline,))
        pending_resources.pop_all()
    return worker


def production_tool_path(settings: Settings, name: str) -> Path:
    """保留统一工具路径入口，便于能力报告复用。"""

    assert settings.runtime_root is not None
    configured = settings.ffprobe_path if name == "ffprobe" else settings.ffmpeg_path
    return configured or settings.runtime_root / "tools" / name
