"""音频 Worker 的独立生产装配。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import httpx

from video_demo.application.audio_chapter_planning import AudioChapterPlanner
from video_demo.application.audio_contracts import AudioEvidencePreparationLimits
from video_demo.application.audio_document_writing import AudioDocumentWriter
from video_demo.application.audio_pipeline import AudioPipeline, AudioSpeechAnalyzer
from video_demo.application.audio_publication import AudioPublicationService
from video_demo.application.audio_speech import AudioSliceClient, VerifiedAudioSlicer
from video_demo.application.audio_transcode import build_audio_ffmpeg_factory
from video_demo.application.audio_workers import AudioJobHandler
from video_demo.config import Settings
from video_demo.integrations.audio_document_client import AudioDocumentClient
from video_demo.integrations.cloud_whisper import CloudWhisperClient
from video_demo.media.audio_probe import FFprobeAudioClient
from video_demo.persistence.database import Database
from video_demo.persistence.migrations import upgrade_runtime_database
from video_demo.speech.vad import NativeSileroBackend, SileroVadAdapter
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.audio_object_store import AudioObjectStore
from video_demo.storage.audio_snapshots import AudioAsrWindowSnapshotStore
from video_demo.storage.document_cache import ModelInvocationIdentity
from video_demo.worker.runtime import ReliableWorker


def build_audio_worker(settings: Settings, *, worker_id: str) -> ReliableWorker:
    """构造只领取 AUDIO_UNDERSTANDING 的独立 Worker。"""

    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite+pysqlite:///{runtime_root / 'video-demo.db'}"
    upgrade_runtime_database(settings.workspace_root, runtime_root, database_url)
    database = Database(database_url)
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

    def pipeline_factory(duration_ms: int) -> AudioPipeline:
        ffmpeg_client = ffmpeg_factory(lambda: False)
        slicer = VerifiedAudioSlicer(
            runtime_root,
            cast(AudioSliceClient, ffmpeg_client),
            duration_ms,
        )
        return AudioPipeline(
            AudioSpeechAnalyzer(
                SileroVadAdapter(NativeSileroBackend()),
                recognizer,
                slicer,
                max_window_ms=cloud.max_window_ms,
                overlap_ms=cloud.overlap_ms,
                max_upload_bytes=cloud.max_upload_bytes,
                window_store=AudioAsrWindowSnapshotStore(
                    AtomicArtifactStore(runtime_root),
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
    handler = AudioJobHandler(
        database,
        FFprobeAudioClient.from_path(
            _production_tool_path(settings, "ffprobe"),
            workspace_root=settings.workspace_root,
        ),
        pipeline_factory,
        publication,
        AudioObjectStore(runtime_root, max_bytes=settings.max_audio_bytes),
        lambda callback: ffmpeg_factory(callback),
        runtime_root=runtime_root,
        max_duration_ms=settings.max_audio_duration_ms,
        max_cache_entry_bytes=settings.model_cache_max_entry_bytes,
        max_cache_run_bytes=settings.model_cache_max_run_bytes,
    )
    return ReliableWorker(
        database,
        worker_id,
        handler,
        job_type="AUDIO_UNDERSTANDING",
        owned_resources=(http,),
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
