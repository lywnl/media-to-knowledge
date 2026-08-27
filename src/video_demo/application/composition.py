from __future__ import annotations

import hashlib
import importlib.metadata
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, Protocol

import httpx

from video_demo.application.chapter_frames import ChapterFrameSearcher
from video_demo.application.chapter_planning import ChapterPlanner
from video_demo.application.chapter_vision import ChapterVisionService
from video_demo.application.document_pipeline import VideoUnderstandingPipeline
from video_demo.application.document_writing import DocumentWriter
from video_demo.application.pipeline import PipelineJobHandler
from video_demo.application.pipeline_contracts import EvidencePreparationLimits
from video_demo.application.production_media import (
    ProductionAssetProbe,
    ProductionAssetRegistrar,
    ProductionMediaTranscoder,
    build_ffmpeg_factory,
    build_ffprobe_factory,
)
from video_demo.application.production_scene import ProductionSceneIndexProvider
from video_demo.application.queries import ResultQueryService
from video_demo.application.visual_cleanup import PublishedVisualCleaner
from video_demo.application.visual_cleanup_recovery import PublishedVisualCleanupRecovery
from video_demo.config import Settings
from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.document import PromptVersions
from video_demo.domain.run import ModelIdentity
from video_demo.integrations.openai_document import OpenAIDocumentClient
from video_demo.integrations.qwen_vl import QwenVisionClient
from video_demo.media.probe import ProbeLimits
from video_demo.persistence.database import Database
from video_demo.persistence.migrations import upgrade_runtime_database
from video_demo.speech.isolated import IsolatedSpeechAnalyzer
from video_demo.speech.snapshots import SpeechFingerprintInputs
from video_demo.speech.subprocess_protocol import (
    SpeechRuntimeConfig,
    SpeechSubprocessCredentials,
)
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.document_cache import DocumentModelCache, ModelInvocationIdentity
from video_demo.storage.object_store import LocalVideoObjectStore
from video_demo.storage.snapshots import SnapshotStore
from video_demo.visual.keyframes import OpenCvFrameExtractor
from video_demo.visual.scenes import PySceneDetectAdapter
from video_demo.worker.runtime import ReliableWorker


class Closable(Protocol):
    def close(self) -> None: ...


class ProductionModelIdentityReport(FrozenModel):
    """3.0 生产组件身份和不含密钥的全局配置指纹。"""

    schema_version: Literal["3.0.0"] = "3.0.0"
    models: tuple[ModelIdentity, ...]
    settings_fingerprint: Sha256


class ProductionPipeline(VideoUnderstandingPipeline):
    """唯一 3.0 生产流水线，并统一拥有模型 HTTP 资源。"""

    def __init__(
        self,
        *args: object,
        owned_resources: tuple[Closable, ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._owned_resources = owned_resources
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in reversed(self._owned_resources):
            resource.close()


def build_production_model_identity_report(settings: Settings) -> ProductionModelIdentityReport:
    cloud_asr = settings.require_cloud_asr_configuration()
    text = settings.require_text_llm_configuration()
    vision = settings.require_vlm_configuration()
    return ProductionModelIdentityReport(
        models=(
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
                component="document_text_llm",
                provider="openai_compatible",
                model_id=text.model_id,
            ),
            ModelIdentity(
                component="chapter_vlm",
                provider="qwen",
                model_id=vision.model_id,
            ),
            ModelIdentity(
                component="scene_detect",
                provider="local",
                model_id="pyscenedetect-content-detector",
            ),
        ),
        settings_fingerprint=_settings_fingerprint(settings),
    )


def build_production_pipeline(
    settings: Settings,
    database: Database,
    object_store: LocalVideoObjectStore,
) -> ProductionPipeline:
    """组装章节级 ASR、文本 LLM、Qwen3-VL 与确定性文档流水线。"""

    assert settings.runtime_root is not None
    cloud_asr = settings.require_cloud_asr_configuration()
    text = settings.require_text_llm_configuration()
    vision = settings.require_vlm_configuration()
    runtime_root = settings.runtime_root
    ffmpeg = production_tool_path(settings, "ffmpeg")
    ffprobe = production_tool_path(settings, "ffprobe")
    ffmpeg_factory = build_ffmpeg_factory(
        settings.workspace_root,
        runtime_root,
        ffmpeg,
        max_output_bytes=settings.max_video_bytes,
        required_free_bytes=settings.min_free_disk_reserve_bytes,
        timeout_seconds=settings.process_timeout_seconds,
        visual_proxy_max_edge=settings.visual_proxy_max_edge,
    )
    prompt_versions = _prompt_versions()
    text_fingerprint = _component_fingerprint(
        {
            "base_url": text.base_url,
            "model_id": text.model_id,
            "timeout_seconds": text.timeout_seconds,
            "max_attempts": text.max_attempts,
            "max_input_chars": text.max_input_chars,
            "max_input_bytes": text.max_input_bytes,
            "max_response_bytes": text.max_response_bytes,
        }
    )
    vision_fingerprint = _component_fingerprint(
        {
            "base_url": vision.base_url,
            "model_id": vision.model_id,
            "timeout_seconds": vision.timeout_seconds,
            "max_attempts": vision.max_attempts,
            "max_image_bytes": vision.max_image_bytes,
            "max_request_image_bytes": vision.max_request_image_bytes,
            "max_encoded_request_bytes": vision.max_encoded_request_bytes,
        }
    )
    with ExitStack() as resources:
        text_http = httpx.Client()
        resources.callback(text_http.close)
        vision_http = httpx.Client()
        resources.callback(vision_http.close)
        text_client = OpenAIDocumentClient(
            text_http,
            base_url=text.base_url,
            api_key=text.api_key.get_secret_value(),
            model_id=text.model_id,
            timeout_seconds=text.timeout_seconds,
            max_attempts=text.max_attempts,
            max_input_chars=text.max_input_chars,
            max_input_bytes=text.max_input_bytes,
            max_response_bytes=text.max_response_bytes,
            compact_planning=True,
        )
        vision_client = QwenVisionClient(
            vision_http,
            base_url=vision.base_url,
            api_key=vision.api_key.get_secret_value(),
            model_id=vision.model_id,
            runtime_root=runtime_root,
            timeout_seconds=vision.timeout_seconds,
            max_attempts=vision.max_attempts,
            max_image_bytes=vision.max_image_bytes,
            max_request_image_bytes=vision.max_request_image_bytes,
            max_encoded_request_bytes=vision.max_encoded_request_bytes,
            max_response_bytes=settings.model_max_response_bytes,
        )
        pipeline = ProductionPipeline(
            ProductionAssetRegistrar(database, object_store),
            ProductionAssetProbe(
                build_ffprobe_factory(settings.workspace_root, ffprobe),
                limits=ProbeLimits(max_duration_ms=settings.max_video_duration_ms),
            ),
            ProductionMediaTranscoder(
                runtime_root,
                ffmpeg_factory,
                max_proxy_bytes=settings.max_video_bytes,
            ),
            IsolatedSpeechAnalyzer(
                workspace_root=settings.workspace_root,
                runtime_root=runtime_root,
                snapshot_store=SnapshotStore(AtomicArtifactStore(runtime_root)),
                artifact_store=AtomicArtifactStore(runtime_root),
                speech_runtime=_speech_runtime_config(settings, ffmpeg),
                credentials=SpeechSubprocessCredentials(openai_api_key=cloud_asr.api_key),
                asr_timeout_seconds=settings.speech_subprocess_timeout_seconds,
            ),
            ProductionSceneIndexProvider(
                runtime_root,
                PySceneDetectAdapter(),
                max_video_bytes=settings.max_video_bytes,
            ),
            ChapterPlanner(
                text_client,
                _identity(
                    "chapter_planning",
                    text_fingerprint,
                    text.model_id,
                    "chapter_planning_compact_v1",
                    prompt_versions.chapter_planner,
                    "chapter_planning_compact_repair_v1",
                    prompt_versions.chapter_planner_repair,
                    generation_config=(("temperature", "0"), ("thinking", "enabled")),
                ),
                max_input_chars=text.max_input_chars,
                max_input_bytes=text.max_input_bytes,
                max_chapters=settings.max_document_chapters,
                invocation_wait_timeout_seconds=text.timeout_seconds,
                concurrency=settings.chapter_planning_concurrency,
                compact_planning=True,
            ),
            ChapterFrameSearcher(
                runtime_root,
                OpenCvFrameExtractor(
                    runtime_root,
                    max_frame_bytes=vision.max_image_bytes,
                    jpeg_quality=settings.keyframe_jpeg_quality,
                ),
                max_candidate_bytes=settings.max_candidate_frame_bytes_per_run,
                max_candidate_files=settings.max_candidate_frame_files_per_run,
                max_candidate_file_bytes=vision.max_image_bytes,
                candidate_lock_timeout_seconds=(
                    settings.candidate_directory_lock_timeout_seconds
                ),
            ),
            ChapterVisionService(
                vision_client,
                _identity(
                    "chapter_vision",
                    vision_fingerprint,
                    vision.model_id,
                    "chapter_vlm_v2",
                    prompt_versions.chapter_vlm,
                    "chapter_vlm_repair_v2",
                    prompt_versions.chapter_vlm_repair,
                ),
                runtime_root=runtime_root,
                concurrency=vision.concurrency,
                max_image_bytes=vision.max_image_bytes,
                max_request_image_bytes=vision.max_request_image_bytes,
                max_encoded_request_bytes=vision.max_encoded_request_bytes,
                max_published_keyframe_bytes=(
                    settings.max_published_keyframe_bytes_per_run
                ),
                max_published_keyframe_files=(
                    settings.max_published_keyframe_files_per_run
                ),
                invocation_wait_timeout_seconds=vision.timeout_seconds,
                candidate_lock_timeout_seconds=(
                    settings.candidate_directory_lock_timeout_seconds
                ),
            ),
            DocumentWriter(
                text_client,
                text_model_id=text.model_id,
                vlm_model_id=vision.model_id,
                prompt_versions=prompt_versions,
                chapter_identity=_identity(
                    "chapter_writing",
                    text_fingerprint,
                    text.model_id,
                    "chapter_writing_v2",
                    prompt_versions.chapter_writer,
                    "chapter_writing_repair_v2",
                    prompt_versions.chapter_writer_repair,
                ),
                global_identity=_identity(
                    "global_editing",
                    text_fingerprint,
                    text.model_id,
                    "global_writing_v1",
                    prompt_versions.global_editor,
                    "global_writing_repair_v1",
                    prompt_versions.global_editor_repair,
                ),
                chapter_writer_concurrency=settings.chapter_writer_concurrency,
                max_input_chars=text.max_input_chars,
                max_input_bytes=text.max_input_bytes,
                invocation_wait_timeout_seconds=text.timeout_seconds,
            ),
            lambda run_root: DocumentModelCache(
                run_root,
                max_entry_bytes=settings.model_cache_max_entry_bytes,
                max_run_bytes=settings.model_cache_max_run_bytes,
            ),
            runtime_root=runtime_root,
            evidence_preparation_limits=_evidence_limits(settings),
            max_result_evidence_items=settings.max_result_evidence_items,
            owned_resources=(text_http, vision_http),
        )
        resources.pop_all()
        return pipeline


def build_worker(settings: Settings, *, worker_id: str) -> ReliableWorker:
    """先迁移和恢复，再验证三套模型配置并构造 Worker。"""

    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite+pysqlite:///{runtime_root / 'video-demo.db'}"
    upgrade_runtime_database(settings.workspace_root, runtime_root, database_url)
    database = Database(database_url)
    artifact_store = AtomicArtifactStore(runtime_root)
    cleaner = PublishedVisualCleaner(
        runtime_root,
        max_candidate_files=settings.max_candidate_frame_files_per_run,
        max_candidate_bytes=settings.max_candidate_frame_bytes_per_run,
        max_published_keyframe_files=settings.max_published_keyframe_files_per_run,
        max_published_keyframe_bytes=settings.max_published_keyframe_bytes_per_run,
        max_keyframe_bytes=settings.vlm_max_image_bytes,
    )
    queries = ResultQueryService(
        database,
        artifact_store,
        max_evidence_items=settings.max_result_evidence_items,
        max_keyframe_bytes=settings.vlm_max_image_bytes,
        max_document_bytes=settings.max_document_bytes,
        max_bundle_bytes=settings.max_result_bundle_bytes,
        visual_cleaner=cleaner,
    )
    PublishedVisualCleanupRecovery(database, queries, cleaner).recover()
    settings.require_cloud_asr_configuration()
    settings.require_text_llm_configuration()
    settings.require_vlm_configuration()
    object_store = LocalVideoObjectStore(
        runtime_root,
        max_video_bytes=settings.max_video_bytes,
    )
    with ExitStack() as resources:
        pipeline = build_production_pipeline(settings, database, object_store)
        resources.callback(pipeline.close)
        handler = PipelineJobHandler(database, pipeline, queries)
        worker = ReliableWorker(
            database,
            worker_id,
            handler,
            owned_resources=(pipeline,),
        )
        resources.pop_all()
        return worker


def production_tool_path(settings: Settings, name: str) -> Path:
    assert settings.runtime_root is not None
    configured = settings.ffprobe_path if name == "ffprobe" else settings.ffmpeg_path
    return configured or settings.runtime_root / "tools" / name


def _evidence_limits(settings: Settings) -> EvidencePreparationLimits:
    return EvidencePreparationLimits(
        max_transcript_evidence_items=settings.max_transcript_evidence_items,
        max_transcript_chars=settings.max_transcript_chars,
        max_scene_boundaries=settings.max_scene_boundaries,
        max_base_segments=settings.max_base_segments,
    )


def _prompt_versions() -> PromptVersions:
    return PromptVersions(
        chapter_planner="chapter-planner-v1",
        chapter_planner_repair="chapter-planner-repair-v1",
        chapter_vlm="chapter-vlm-v1",
        chapter_vlm_repair="chapter-vlm-repair-v1",
        chapter_writer="chapter-writer-v1",
        chapter_writer_repair="chapter-writer-repair-v1",
        global_editor="global-editor-v1",
        global_editor_repair="global-editor-repair-v1",
    )


def _identity(
    operation: str,
    fingerprint: str,
    model_id: str,
    main_schema: str,
    main_prompt: str,
    repair_schema: str,
    repair_prompt: str,
    generation_config: tuple[tuple[str, str], ...] = (("temperature", "0"),),
) -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation=operation,
        provider_config_fingerprint=fingerprint,
        model_id=model_id,
        generation_config=generation_config,
        main_response_schema_name=main_schema,
        main_prompt_version=main_prompt,
        repair_response_schema_name=repair_schema,
        repair_prompt_version=repair_prompt,
    )


def _component_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _settings_fingerprint(settings: Settings) -> str:
    return _component_fingerprint(_settings_payload(settings))


def _settings_payload(settings: Settings) -> dict[str, object]:
    cloud = settings.require_cloud_asr_configuration()
    text = settings.require_text_llm_configuration()
    vision = settings.require_vlm_configuration()
    return {
            "schema_version": "3.0.0",
            "cloud_asr": {
                "base_url": cloud.base_url,
                "model_id": cloud.model,
                "timeout_seconds": cloud.timeout_seconds,
                "max_attempts": cloud.max_attempts,
                "max_window_ms": cloud.max_window_ms,
                "overlap_ms": cloud.overlap_ms,
            },
            "text": {
                "base_url": text.base_url,
                "model_id": text.model_id,
                "timeout_seconds": text.timeout_seconds,
                "max_attempts": text.max_attempts,
            },
            "vision": {
                "base_url": vision.base_url,
                "model_id": vision.model_id,
                "timeout_seconds": vision.timeout_seconds,
                "max_attempts": vision.max_attempts,
                "proxy_max_edge": settings.visual_proxy_max_edge,
                "jpeg_quality": settings.keyframe_jpeg_quality,
            },
            "budgets": {
                "result_evidence": settings.max_result_evidence_items,
                "candidate_bytes": settings.max_candidate_frame_bytes_per_run,
                "published_keyframe_bytes": (
                    settings.max_published_keyframe_bytes_per_run
                ),
                "bundle_bytes": settings.max_result_bundle_bytes,
                "document_bytes": settings.max_document_bytes,
            },
            "prompts": _prompt_versions().model_dump(mode="json"),
        }


def resolution_comparison_settings_fingerprint(settings: Settings) -> str:
    """生成 1280/1920 对照共用的设置指纹，仅忽略代理长边。"""

    payload = _settings_payload(settings)
    vision = payload["vision"]
    assert isinstance(vision, dict)
    vision.pop("proxy_max_edge")
    return _component_fingerprint(payload)


def _local_model_identity(
    component: str,
    model_id: str,
    *,
    package: str,
    device: str,
) -> ModelIdentity:
    try:
        revision = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        revision = "NOT_INSTALLED"
    return ModelIdentity(
        component=component,
        provider="local",
        model_id=model_id,
        revision=revision,
        device=device,
    )


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


__all__ = [
    "ProductionModelIdentityReport",
    "ProductionPipeline",
    "build_production_model_identity_report",
    "build_production_pipeline",
    "build_worker",
    "production_tool_path",
    "resolution_comparison_settings_fingerprint",
]
