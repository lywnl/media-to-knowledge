"""音频流水线使用的转码接口和生产装配。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from video_demo.domain.run import TimeRange
from video_demo.media.audio_transcode import (
    AudioArtifact,
    AudioSliceArtifact,
    AudioTranscodeLimits,
    AudioTranscoder,
    NoAudioArtifact,
)

__all__ = [
    "AudioArtifact",
    "AudioSliceArtifact",
    "AudioTranscodeClient",
    "NoAudioArtifact",
    "build_audio_ffmpeg_factory",
]


class AudioTranscodeClient(Protocol):
    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool,
        duration_ms: int,
    ) -> AudioArtifact | NoAudioArtifact: ...

    def create_audio_slice(
        self,
        source: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
        *,
        source_duration_ms: int,
    ) -> AudioSliceArtifact: ...


def build_audio_ffmpeg_factory(
    workspace_root: Path,
    runtime_root: Path,
    executable: Path,
    *,
    max_output_bytes: int,
    timeout_seconds: int,
) -> Callable[[Callable[[], bool]], AudioTranscodeClient]:
    return lambda is_cancel_requested: AudioTranscoder.from_path(
        executable,
        runtime_root,
        workspace_root=workspace_root,
        is_cancel_requested=is_cancel_requested,
        limits=AudioTranscodeLimits(
            max_output_bytes=max_output_bytes,
            timeout_seconds=timeout_seconds,
        ),
    )
