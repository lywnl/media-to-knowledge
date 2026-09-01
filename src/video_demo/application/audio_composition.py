"""音频阶段调度器的独立生产装配。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import cast

import httpx

from video_demo.application.audio_chapter_planning import AudioChapterPlanner
from video_demo.application.audio_contracts import AudioEvidencePreparationLimits
from video_demo.application.audio_document_writing import AudioDocumentWriter
from video_demo.application.audio_pipeline import AudioPipeline, AudioSpeechAnalyzer
from video_demo.application.audio_pipeline_executor import AudioStagePipelineExecutor
from video_demo.application.audio_publication import AudioPublicationService
from video_demo.application.audio_scheduler import AudioTaskScheduler
from video_demo.application.audio_speech import AudioSliceClient, VerifiedAudioSlicer
from video_demo.application.audio_transcode import AudioTranscodeClient, build_audio_ffmpeg_factory
from video_demo.config import Settings
from video_demo.integrations.audio_document_client import AudioDocumentClient
from video_demo.integrations.cloud_whisper import CloudWhisperClient
from video_demo.media.audio_probe import FFprobeAudioClient
from video_demo.persistence.database import Database
from video_demo.speech.audio_snapshots import AudioAsrFingerprintInputs
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.audio_object_store import AudioObjectStore
from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore
from video_demo.storage.document_cache import ModelInvocationIdentity


@dataclass(frozen=True, slots=True)
class _AudioProductionRuntime:
    probe: _LazyAudioProbe
    pipeline_factory: Callable[[int, Callable[[], bool]], AudioPipeline]
    transcoder_factory: Callable[[Callable[[], bool]], AudioTranscodeClient]
    object_store: AudioObjectStore
    publication: AudioPublicationService
    http: httpx.Client
    runtime_root: Path


class _LazyAudioProbe:
    """延迟初始化 ffprobe，避免 API 组装阶段因本地工具缺失而无法启动。"""

    def __init__(self, executable: Path, *, workspace_root: Path, timeout_seconds: int) -> None:
        self._executable = executable
        self._workspace_root = workspace_root
        self._timeout_seconds = timeout_seconds
        self._client: FFprobeAudioClient | None = None
        self._lock = Lock()

    def probe(self, source: Path, *, max_duration_ms: int) -> object:
        client = self._client
        if client is None:
            with self._lock:
                client = self._client
                if client is None:
                    client = FFprobeAudioClient.from_path(
                        self._executable,
                        workspace_root=self._workspace_root,
                        timeout_seconds=self._timeout_seconds,
                    )
                    self._client = client
        return client.probe(source, max_duration_ms=max_duration_ms)


def _build_audio_runtime(settings: Settings, database: Database) -> _AudioProductionRuntime:
    """构造音频专用外部依赖；不创建视频或视觉组件。"""

    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    cloud = settings.require_cloud_asr_configuration()
    text = settings.require_text_llm_configuration()
    http = httpx.Client()
    ffmpeg = _production_tool_path(settings, "ffmpeg")
    ffmpeg_factory = build_audio_ffmpeg_factory(
        settings.workspace_root,
        runtime_root,
        ffmpeg,
        max_output_bytes=settings.max_audio_bytes,
        timeout_seconds=settings.process_timeout_seconds,
    )
    recognizer = CloudWhisperClient(http, cloud, allowed_audio_root=runtime_root)
    client = AudioDocumentClient(
        http,
        base_url=text.base_url,
        api_key=text.api_key.get_secret_value(),
        model_id=text.model_id,
        timeout_seconds=text.timeout_seconds,
        max_attempts=text.max_attempts,
        max_response_bytes=text.max_response_bytes,
    )
    fingerprint = _fingerprint({"model": text.model_id, "operation": "audio"})
    planner = AudioChapterPlanner(
        client,
        _identity(
            "audio_chapter_planning",
            fingerprint,
            text.model_id,
            "audio_chapter_planning_v1",
            "audio-chapter-planner-v1",
            "audio_chapter_planning_repair_v1",
            "audio-chapter-planner-repair-v1",
        ),
        boundary_identity=_identity(
            "audio_chapter_boundary_coordination",
            fingerprint,
            text.model_id,
            "audio_chapter_boundary_coordination_v1",
            "audio-chapter-boundary-coordinator-v1",
            "audio_chapter_boundary_coordination_v1",
            "audio-chapter-boundary-coordinator-v1",
        ),
        max_input_chars=text.max_input_chars,
        max_input_bytes=text.max_input_bytes,
        max_chapters=settings.max_document_chapters,
        invocation_wait_timeout_seconds=text.timeout_seconds,
        concurrency=settings.chapter_planning_concurrency,
    )
    writer = AudioDocumentWriter(
        client,
        chapter_identity=_identity(
            "audio_chapter_writing",
            fingerprint,
            text.model_id,
            "audio_chapter_writing_v1",
            "audio-chapter-writer-v1",
            "audio_chapter_writing_repair_v1",
            "audio-chapter-writer-repair-v1",
        ),
        global_identity=_identity(
            "audio_global_editing",
            fingerprint,
            text.model_id,
            "audio_global_writing_v1",
            "audio-global-editor-v1",
            "audio_global_writing_repair_v1",
            "audio-global-editor-repair-v1",
        ),
        concurrency=settings.chapter_writer_concurrency,
        max_input_chars=text.max_input_chars,
        max_input_bytes=text.max_input_bytes,
        invocation_wait_timeout_seconds=text.timeout_seconds,
    )

    def pipeline_factory(
        duration_ms: int,
        is_cancel_requested: Callable[[], bool],
    ) -> AudioPipeline:
        ffmpeg_client = ffmpeg_factory(is_cancel_requested)
        slicer = VerifiedAudioSlicer(
            runtime_root,
            cast(AudioSliceClient, ffmpeg_client),
            duration_ms,
        )
        return AudioPipeline(
            AudioSpeechAnalyzer(
                recognizer,
                slicer,
                max_upload_bytes=cloud.max_upload_bytes,
                window_store=AudioAsrWindowSnapshotStore(
                    AtomicArtifactStore(runtime_root),
                ),
                fingerprint_inputs=AudioAsrFingerprintInputs(
                    model_id=cloud.model,
                    base_url=cloud.base_url,
                    timeout_seconds=cloud.timeout_seconds,
                    max_attempts=cloud.max_attempts,
                    max_upload_bytes=cloud.max_upload_bytes,
                ),
            ),
            planner,
            writer,
            evidence_limits=_evidence_limits(settings),
        )

    publication = AudioPublicationService(
        database,
        AtomicArtifactStore(runtime_root),
        max_document_bytes=settings.max_document_bytes,
        max_bundle_bytes=settings.max_result_bundle_bytes,
    )

    return _AudioProductionRuntime(
        probe=_LazyAudioProbe(
            _production_tool_path(settings, "ffprobe"),
            workspace_root=settings.workspace_root,
            timeout_seconds=settings.process_timeout_seconds,
        ),
        pipeline_factory=pipeline_factory,
        transcoder_factory=lambda callback: ffmpeg_factory(callback),
        object_store=AudioObjectStore(runtime_root, max_bytes=settings.max_audio_bytes),
        publication=publication,
        http=http,
        runtime_root=runtime_root,
    )


def build_audio_scheduler(
    settings: Settings,
    database: Database,
) -> AudioTaskScheduler:
    """构造 FastAPI 进程内音频双阶段调度器。"""

    runtime = _build_audio_runtime(settings, database)
    executor = AudioStagePipelineExecutor(
        database,
        runtime.pipeline_factory,
        runtime.probe,
        runtime.transcoder_factory,
        runtime.object_store,
        runtime.runtime_root,
        runtime.publication,
        max_duration_ms=settings.max_audio_duration_ms,
        max_cache_entry_bytes=settings.model_cache_max_entry_bytes,
        max_cache_run_bytes=settings.model_cache_max_run_bytes,
        owned_resources=(runtime.http,),
    )
    return AudioTaskScheduler(
        executor,
        transcription_concurrency=2,
        llm_concurrency=2,
    )


def _identity(
    operation: str,
    fingerprint: str,
    model_id: str,
    main_schema: str,
    main_prompt: str,
    repair_schema: str,
    repair_prompt: str,
) -> ModelInvocationIdentity:
    return ModelInvocationIdentity(
        logical_operation=operation,
        provider_config_fingerprint=fingerprint,
        model_id=model_id,
        generation_config=(("temperature", "0"), ("thinking", "disabled")),
        main_response_schema_name=main_schema,
        main_prompt_version=main_prompt,
        repair_response_schema_name=repair_schema,
        repair_prompt_version=repair_prompt,
    )


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_limits(settings: Settings) -> AudioEvidencePreparationLimits:
    return AudioEvidencePreparationLimits(
        max_transcript_evidence_items=settings.max_transcript_evidence_items,
        max_transcript_chars=settings.max_transcript_chars,
        max_base_segments=settings.max_base_segments,
    )


def _production_tool_path(settings: Settings, name: str) -> Path:
    assert settings.runtime_root is not None
    configured = settings.ffprobe_path if name == "ffprobe" else settings.ffmpeg_path
    return configured or settings.runtime_root / "tools" / name
