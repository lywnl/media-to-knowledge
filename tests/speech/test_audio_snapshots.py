from typing import Literal, get_args, get_origin

from video_demo.domain.run import TimeRange
from video_demo.speech.audio_fixed_asr import AudioFixedAsrWindow
from video_demo.speech.audio_snapshots import (
    AudioAsrFingerprintInputs,
    AudioAsrWindowSnapshotPayload,
    audio_asr_fingerprint,
    audio_asr_window_fingerprint,
)


def _window() -> AudioFixedAsrWindow:
    time_range = TimeRange(start_ms=0, end_ms=1_000)
    return AudioFixedAsrWindow(
        chunk_index=0,
        upload_range=time_range,
        owned_range=time_range,
    )


def _inputs() -> AudioAsrFingerprintInputs:
    return AudioAsrFingerprintInputs(
        model_id="openai/whisper-1",
        base_url="https://ai-proxy.example/v1",
        timeout_seconds=300.0,
        max_attempts=3,
    )


def test_audio_window_snapshot_schema_is_literal_v2() -> None:
    annotation = AudioAsrWindowSnapshotPayload.model_fields["schema_version"].annotation

    assert get_origin(annotation) is Literal
    assert get_args(annotation) == ("2.0.0",)


def test_audio_asr_parent_fingerprint_binds_media_and_provider_inputs() -> None:
    inputs = _inputs()
    arguments = {
        "asset_sha256": "a" * 64,
        "duration_ms": 60_000,
        "language_hints": (),
        "hotwords": ("WebRTC",),
        "core_context": "音频课程",
        "inputs": inputs,
    }
    base = audio_asr_fingerprint(**arguments)

    assert base != audio_asr_fingerprint(**{**arguments, "asset_sha256": "b" * 64})
    assert base != audio_asr_fingerprint(**{**arguments, "duration_ms": 120_000})
    assert base != audio_asr_fingerprint(
        **{**arguments, "inputs": inputs.model_copy(update={"model_id": "second-model"})}
    )
    assert base != audio_asr_fingerprint(
        **{**arguments, "inputs": inputs.model_copy(update={"base_url": "https://second.example/v1"})}
    )


def test_audio_window_fingerprint_binds_parent_and_window_only() -> None:
    base = audio_asr_window_fingerprint(asr_fingerprint="a" * 64, window=_window())
    changed_parent = audio_asr_window_fingerprint(asr_fingerprint="b" * 64, window=_window())
    changed_window = audio_asr_window_fingerprint(
        asr_fingerprint="a" * 64,
        window=AudioFixedAsrWindow(
            chunk_index=1,
            upload_range=TimeRange(start_ms=1_000, end_ms=2_000),
            owned_range=TimeRange(start_ms=1_000, end_ms=2_000),
        ),
    )

    assert base != changed_parent
    assert base != changed_window
