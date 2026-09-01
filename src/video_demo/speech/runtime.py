from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import httpx

from video_demo.application.pipeline_contracts import PreparedMedia
from video_demo.application.production_media import TranscodeClient
from video_demo.application.production_speech import (
    AsrComponents,
    AudioSliceClient,
    VerifiedAudioSlicer,
)
from video_demo.config import Settings
from video_demo.integrations.cloud_whisper import CloudWhisperClient
from video_demo.speech.asr import WindowRecognizerPort


@dataclass(frozen=True, slots=True)
class ProductionSpeechModels:
    """生产诊断入口复用的 Silero 与云端识别端口。"""

    vad: object
    recognizer: WindowRecognizerPort


def build_diagnostic_speech_models(
    settings: Settings,
    http_client: httpx.Client,
) -> ProductionSpeechModels:
    configuration = settings.require_cloud_asr_configuration()
    from video_demo.speech.vad import NativeSileroBackend, SileroVadAdapter

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
    recognizer: WindowRecognizerPort,
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
            recognizer=recognizer,
            slicer=VerifiedAudioSlicer(
                runtime_root,
                cast(AudioSliceClient, ffmpeg_client),
                media.source.duration_ms,
            ),
            slice_namespace="speech_diagnostic",
            is_cancel_requested=is_cancel_requested,
        )

    return build
