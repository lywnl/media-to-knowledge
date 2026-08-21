from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from video_demo.application.pipeline import PreparedMedia
from video_demo.application.production_media import TranscodeClient, build_ffmpeg_factory
from video_demo.application.production_speech import (
    AudioSliceClient,
    ComponentFactory,
    SpeechComponents,
    VerifiedAudioSlicer,
)
from video_demo.audio.yamnet import NativeYamnetBackend, NativeYamnetDetector
from video_demo.config import Settings
from video_demo.speech.alignment import NativeWhisperXBackend, WhisperXAligner
from video_demo.speech.asr import FasterWhisperAdapter, NativeFasterWhisperBackend
from video_demo.speech.diarization import NativePyannoteBackend, PyannoteDiarizer
from video_demo.speech.language import FasterWhisperLanguageDetector, SegmentLanguageIdentifier
from video_demo.speech.vad import NativeSileroBackend, SileroVadAdapter


@dataclass(frozen=True, slots=True)
class ProductionSpeechModels:
    """语音适配器只持有懒加载边界，真实权重在首次调用时加载。"""

    vad: SileroVadAdapter
    language_identifier: SegmentLanguageIdentifier
    recognizer: FasterWhisperAdapter
    aligner: WhisperXAligner
    diarizer: PyannoteDiarizer
    audio_events: NativeYamnetDetector


def build_speech_models(settings: Settings) -> ProductionSpeechModels:
    assert settings.runtime_root is not None

    def token_provider() -> str | None:
        return (
            settings.huggingface_token.get_secret_value()
            if settings.huggingface_token is not None
            else None
        )

    return build_speech_models_from_runtime(
        settings.runtime_root,
        settings.workspace_root / "src/video_demo/audio/thresholds.json",
        inference_device=settings.inference_device,
        whisper_compute_type=settings.whisper_compute_type,
        huggingface_token=token_provider,
    )


def build_speech_models_from_runtime(
    runtime_root: Path,
    thresholds_path: Path,
    *,
    inference_device: str,
    whisper_compute_type: str,
    huggingface_token: str | Callable[[], str | None] | None,
) -> ProductionSpeechModels:
    model_root = runtime_root / "models"
    faster_backend = NativeFasterWhisperBackend(
        model_root,
        device=inference_device,
        compute_type=whisper_compute_type,
    )

    def token_provider() -> str | None:
        return huggingface_token() if callable(huggingface_token) else huggingface_token

    yamnet_backend = NativeYamnetBackend(model_root / "yamnet/saved_model")
    return ProductionSpeechModels(
        vad=SileroVadAdapter(NativeSileroBackend()),
        language_identifier=SegmentLanguageIdentifier(
            FasterWhisperLanguageDetector(faster_backend),
        ),
        recognizer=FasterWhisperAdapter(faster_backend),
        aligner=WhisperXAligner(NativeWhisperXBackend(), model_root),
        diarizer=PyannoteDiarizer(
            NativePyannoteBackend(token_provider, model_root=model_root),
        ),
        audio_events=NativeYamnetDetector(
            backend_factory=lambda: yamnet_backend,
            class_map_path=model_root / "yamnet/yamnet_class_map.csv",
            thresholds_path=thresholds_path,
        ),
    )


def build_speech_component_factory(
    settings: Settings,
    ffmpeg_factory: object,
    *,
    models: ProductionSpeechModels | None = None,
) -> ComponentFactory:
    assert settings.runtime_root is not None
    active_models = models or build_speech_models(settings)
    return component_factory_from_models(
        settings.runtime_root,
        ffmpeg_factory,
        active_models,
    )


def component_factory_from_models(
    runtime_root: Path,
    ffmpeg_factory: object,
    models: ProductionSpeechModels,
) -> ComponentFactory:
    def build(
        media: PreparedMedia,
        is_cancel_requested: Callable[[], bool],
    ) -> SpeechComponents:
        factory = ffmpeg_factory
        assert callable(factory)
        ffmpeg_client: TranscodeClient = factory(is_cancel_requested)
        return SpeechComponents(
            vad=models.vad,
            language_identifier=models.language_identifier,
            recognizer=models.recognizer,
            aligner=models.aligner,
            diarizer=models.diarizer,
            audio_events=models.audio_events,
            slicer=VerifiedAudioSlicer(
                runtime_root,
                cast(AudioSliceClient, ffmpeg_client),
                media.source.duration_ms,
            ),
        )

    return build


def build_subprocess_component_factory(
    *,
    workspace_root: Path,
    runtime_root: Path,
    ffmpeg_path: Path,
    inference_device: str,
    whisper_compute_type: str,
    huggingface_token: str | None,
) -> ComponentFactory:
    models = build_speech_models_from_runtime(
        runtime_root,
        workspace_root / "src/video_demo/audio/thresholds.json",
        inference_device=inference_device,
        whisper_compute_type=whisper_compute_type,
        huggingface_token=huggingface_token,
    )
    ffmpeg_factory = build_ffmpeg_factory(
        workspace_root,
        runtime_root,
        ffmpeg_path,
    )
    return component_factory_from_models(runtime_root, ffmpeg_factory, models)
