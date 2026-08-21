from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from video_demo.domain.base import FrozenModel, LanguageCode, Sha256, StableId


class Rational(FrozenModel):
    numerator: int
    denominator: int

    @model_validator(mode="after")
    def reject_zero_denominator(self) -> Self:
        if self.denominator == 0:
            raise ValueError("有理数分母不能为 0")
        return self

    @property
    def value(self) -> float:
        return self.numerator / self.denominator


class VideoStream(FrozenModel):
    index: int = Field(ge=0)
    codec_name: str = Field(min_length=1, max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    average_frame_rate: Rational
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    pixel_format: str | None = Field(default=None, max_length=64)
    is_variable_frame_rate: bool = False


class AudioStream(FrozenModel):
    index: int = Field(ge=0)
    codec_name: str = Field(min_length=1, max_length=64)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0, le=64)


class SubtitleStream(FrozenModel):
    index: int = Field(ge=0)
    codec_name: str = Field(min_length=1, max_length=64)
    language: LanguageCode = "und"
    is_default: bool = False
    is_forced: bool = False


class VideoAssetManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    object_ref: StableId
    source_sha256: Sha256
    source_size_bytes: int = Field(gt=0)
    source_mime: Literal["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"]
    duration_ms: int = Field(gt=0, le=1_800_000)
    video_stream: VideoStream
    audio_streams: tuple[AudioStream, ...] = ()
    subtitle_streams: tuple[SubtitleStream, ...] = ()
    format_name: str = Field(min_length=1, max_length=128)
    ffprobe_version: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        stream = self.video_stream
        display_width = stream.height if stream.rotation_degrees in (90, 270) else stream.width
        display_height = stream.width if stream.rotation_degrees in (90, 270) else stream.height
        if display_width > 1920 or display_height > 1080:
            raise ValueError("视频分辨率超过 1920×1080")
        if stream.average_frame_rate.value > 60:
            raise ValueError("视频帧率超过 60 FPS")
        return self
