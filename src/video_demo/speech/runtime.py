from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx

from video_demo.application.pipeline import PreparedMedia
from video_demo.application.production_media import TranscodeClient, build_ffmpeg_factory
from video_demo.application.production_speech import (
    AsrComponents,
    AudioSliceClient,
    VerifiedAudioSlicer,
)
from video_demo.config import Settings
from video_demo.integrations.cloud_whisper import CloudWhisperClient
from video_demo.speech.asr import WindowRecognizerPort
from video_demo.speech.vad import NativeSileroBackend, SileroVadAdapter


@dataclass(frozen=True, slots=True)
class ProductionSpeechModels:
    """生产诊断入口复用的 Silero 与云端识别端口。"""

    vad: SileroVadAdapter
    recognizer: WindowRecognizerPort


def build_diagnostic_speech_models(
    settings: Settings,
    http_client: httpx.Client,
) -> ProductionSpeechModels:
    configuration = settings.require_cloud_asr_configuration()
    return ProductionSpeechModels(
        vad=SileroVadAdapter(NativeSileroBackend()),
        recognizer=CloudWhisperClient(
            http_client,
            configuration,
            allowed_audio_root=settings.runtime_root or settings.workspace_root,
        ),
    )


def build_speech_component_factory(
    settings: Settings,
    ffmpeg_factory: object,
    *,
    models: ProductionSpeechModels,
) -> Callable[[PreparedMedia, Callable[[], bool]], AsrComponents]:
    assert settings.runtime_root is not None
    runtime_root = settings.runtime_root

    def build(
        media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> AsrComponents:
        factory = ffmpeg_factory
        assert callable(factory)
        ffmpeg_client: TranscodeClient = factory(is_cancel_requested)
        return AsrComponents(
            vad=models.vad,
            recognizer=models.recognizer,
            slicer=VerifiedAudioSlicer(
                runtime_root,
                cast(AudioSliceClient, ffmpeg_client),
                media.source.duration_ms,
            ),
            slice_namespace="speech_diagnostic",
        )

    return build


def build_subprocess_asr_components(
    media: PreparedMedia,
    *,
    workspace_root: Path,
    runtime_root: Path,
    ffmpeg_path: Path,
    recognizer: WindowRecognizerPort,
    slice_namespace: str,
    vad_threshold: float,
    vad_merge_gap_ms: int,
    is_cancel_requested: Callable[[], bool] = lambda: False,
) -> AsrComponents:
    """构造单次子进程使用的 VAD、切片器和云端识别端口。"""

    ffmpeg_factory = build_ffmpeg_factory(workspace_root, runtime_root, ffmpeg_path)
    ffmpeg_client: TranscodeClient = ffmpeg_factory(is_cancel_requested)
    return AsrComponents(
        vad=SileroVadAdapter(
            NativeSileroBackend(),
            threshold=vad_threshold,
            merge_gap_ms=vad_merge_gap_ms,
        ),
        recognizer=recognizer,
        slicer=VerifiedAudioSlicer(
            runtime_root,
            cast(AudioSliceClient, ffmpeg_client),
            media.source.duration_ms,
        ),
        slice_namespace=slice_namespace,
    )
