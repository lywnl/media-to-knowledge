from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from video_demo.capabilities import resolve_workspace_binary
from video_demo.domain.manifest import (
    AudioStream,
    Rational,
    SubtitleStream,
    VideoAssetManifest,
    VideoStream,
)
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult, SafeProcessRunner

SupportedMime = Literal["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"]


class ProcessRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
    ) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class ProbeLimits:
    max_duration_ms: int = 1_800_000
    max_width: int = 1920
    max_height: int = 1080
    max_frame_rate: float = 60.0
    max_video_streams: int = 1
    max_audio_streams: int = 16
    max_subtitle_streams: int = 32


@dataclass(frozen=True, slots=True)
class ProbeResult:
    manifest: VideoAssetManifest
    warnings: tuple[str, ...]


class FFprobeClient:
    def __init__(
        self,
        executable: Path,
        runner: ProcessRunner,
        version: str,
        *,
        timeout_seconds: int = 60,
    ) -> None:
        self._executable = executable
        self._runner = runner
        self._version = version
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_path(cls, executable: Path, *, workspace_root: Path) -> FFprobeClient:
        executable = resolve_workspace_binary(
            executable,
            workspace_root=workspace_root,
            unavailable_code=ErrorCode.VIDEO_FFPROBE_UNAVAILABLE,
        )
        runner = SafeProcessRunner(max_output_bytes=32 * 1024 * 1024)
        version_result = runner.run([str(executable), "-version"], timeout_seconds=10)
        if version_result.returncode != 0:
            raise VideoDemoError(ErrorCode.VIDEO_BINARY_PROBE_FAILED, "ffprobe 版本探测失败")
        lines = version_result.stdout.decode("utf-8", errors="replace").splitlines()
        version = lines[0] if lines else "unknown"
        return cls(executable, runner, version)

    def probe(
        self,
        source: Path,
        *,
        object_ref: str,
        source_sha256: str,
        source_size_bytes: int,
        source_mime: SupportedMime,
        limits: ProbeLimits,
        input_fd: int | None = None,
    ) -> ProbeResult:
        input_path = str(source)
        pass_fds: tuple[int, ...] = ()
        if input_fd is not None:
            _require_regular_descriptor(input_fd)
            input_path = f"/dev/fd/{input_fd}"
            pass_fds = (input_fd,)
        arguments = [
            str(self._executable),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            input_path,
        ]
        process_result = (
            self._runner.run(
                arguments,
                timeout_seconds=self._timeout_seconds,
                pass_fds=pass_fds,
            )
            if pass_fds
            else self._runner.run(arguments, timeout_seconds=self._timeout_seconds)
        )
        if process_result.returncode != 0:
            raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "ffprobe 无法解码视频")
        try:
            payload: object = json.loads(process_result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VideoDemoError(
                ErrorCode.VIDEO_PROBE_INVALID,
                "ffprobe 返回了非法 JSON",
            ) from error
        return parse_ffprobe_payload(
            payload,
            object_ref=object_ref,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            source_mime=source_mime,
            ffprobe_version=self._version,
            limits=limits,
        )


def _require_regular_descriptor(descriptor: int) -> None:
    import os
    import stat

    if os.name != "posix" or type(descriptor) is not int or descriptor < 0:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "ffprobe 输入 fd 非法")
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "ffprobe 输入 fd 非法") from None
    if not stat.S_ISREG(details.st_mode):
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "ffprobe 输入 fd 不是普通文件")


def parse_ffprobe_payload(
    payload: object,
    *,
    object_ref: str,
    source_sha256: str,
    source_size_bytes: int,
    source_mime: SupportedMime,
    ffprobe_version: str,
    limits: ProbeLimits,
) -> ProbeResult:
    root = _object(payload, "ffprobe")
    format_payload = _object(root.get("format"), "format")
    streams_payload = _list(root.get("streams"), "streams")
    video_payloads = [stream for stream in streams_payload if _type(stream) == "video"]
    audio_payloads = [stream for stream in streams_payload if _type(stream) == "audio"]
    subtitle_payloads = [stream for stream in streams_payload if _type(stream) == "subtitle"]
    if not video_payloads:
        raise VideoDemoError(ErrorCode.VIDEO_STREAM_MISSING, "视频中没有可解码视频流")
    if len(video_payloads) > limits.max_video_streams:
        raise VideoDemoError(ErrorCode.VIDEO_STREAM_COUNT_EXCEEDED, "视频流数量超过限制")
    if len(audio_payloads) > limits.max_audio_streams:
        raise VideoDemoError(ErrorCode.VIDEO_STREAM_COUNT_EXCEEDED, "音频流数量超过限制")
    if len(subtitle_payloads) > limits.max_subtitle_streams:
        raise VideoDemoError(ErrorCode.VIDEO_STREAM_COUNT_EXCEEDED, "字幕流数量超过限制")

    duration_ms = _duration_ms(format_payload.get("duration"))
    if duration_ms > limits.max_duration_ms:
        raise VideoDemoError(ErrorCode.VIDEO_DURATION_LIMIT_EXCEEDED, "视频时长超过限制")
    video_stream = _video_stream(_object(video_payloads[0], "video_stream"), limits)
    audio_streams = tuple(
        _audio_stream(_object(stream, "audio_stream")) for stream in audio_payloads
    )
    subtitle_streams = tuple(
        _subtitle_stream(_object(stream, "subtitle_stream")) for stream in subtitle_payloads
    )
    manifest = VideoAssetManifest(
        object_ref=object_ref,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        source_mime=source_mime,
        duration_ms=duration_ms,
        video_stream=video_stream,
        audio_streams=audio_streams,
        subtitle_streams=subtitle_streams,
        format_name=_string(format_payload.get("format_name"), "format_name"),
        ffprobe_version=ffprobe_version,
    )
    warnings = () if audio_streams else ("NO_AUDIO_TRACK",)
    return ProbeResult(manifest=manifest, warnings=warnings)


def _video_stream(payload: dict[str, Any], limits: ProbeLimits) -> VideoStream:
    width = _positive_int(payload.get("width"), "width")
    height = _positive_int(payload.get("height"), "height")
    rotation = _rotation(payload)
    display_width = height if rotation in (90, 270) else width
    display_height = width if rotation in (90, 270) else height
    if display_width > limits.max_width or display_height > limits.max_height:
        raise VideoDemoError(ErrorCode.VIDEO_RESOLUTION_LIMIT_EXCEEDED, "视频分辨率超过限制")
    average = _rational(payload.get("avg_frame_rate"), "avg_frame_rate")
    if average.value > limits.max_frame_rate:
        raise VideoDemoError(ErrorCode.VIDEO_FRAME_RATE_LIMIT_EXCEEDED, "视频帧率超过限制")
    nominal = _rational(payload.get("r_frame_rate"), "r_frame_rate")
    is_vfr = not math.isclose(average.value, nominal.value, rel_tol=0.001, abs_tol=0.001)
    return VideoStream(
        index=_non_negative_int(payload.get("index"), "index"),
        codec_name=_string(payload.get("codec_name"), "codec_name"),
        width=width,
        height=height,
        average_frame_rate=average,
        rotation_degrees=cast(Literal[0, 90, 180, 270], rotation),
        pixel_format=_optional_string(payload.get("pix_fmt")),
        is_variable_frame_rate=is_vfr,
    )


def _audio_stream(payload: dict[str, Any]) -> AudioStream:
    return AudioStream(
        index=_non_negative_int(payload.get("index"), "index"),
        codec_name=_string(payload.get("codec_name"), "codec_name"),
        sample_rate_hz=_positive_int(payload.get("sample_rate"), "sample_rate"),
        channels=_positive_int(payload.get("channels"), "channels"),
    )


_SUBTITLE_LANGUAGE_CODES = {
    "zho": "zh",
    "chi": "zh",
    "eng": "en",
    "jpn": "ja",
    "kor": "ko",
    "spa": "es",
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "es": "es",
}


def _subtitle_stream(payload: dict[str, Any]) -> SubtitleStream:
    tags = _object(payload.get("tags", {}), "subtitle_tags")
    disposition = _object(payload.get("disposition", {}), "subtitle_disposition")
    language_value = tags.get("language")
    language = (
        _SUBTITLE_LANGUAGE_CODES.get(language_value.strip().lower(), "und")
        if isinstance(language_value, str)
        else "und"
    )
    return SubtitleStream(
        index=_non_negative_int(payload.get("index"), "index"),
        codec_name=_string(payload.get("codec_name"), "codec_name"),
        language=language,
        is_default=_disposition_flag(disposition.get("default", 0), "default"),
        is_forced=_disposition_flag(disposition.get("forced", 0), "forced"),
    )


def _disposition_flag(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    parsed = _integer(value, field)
    if parsed not in (0, 1):
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是 0 或 1")
    return bool(parsed)


def _rotation(payload: dict[str, Any]) -> int:
    raw: object = _object(payload.get("tags", {}), "tags").get("rotate", 0)
    side_data = payload.get("side_data_list", [])
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                raw = item["rotation"]
                break
    normalized = _integer(raw, "rotation") % 360
    if normalized not in (0, 90, 180, 270):
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, "视频旋转角度不受支持")
    return normalized


def _duration_ms(value: object) -> int:
    try:
        duration = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, "视频时长非法") from error
    if not duration.is_finite() or duration <= 0:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, "视频时长必须为正有限数")
    return int((duration * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _rational(value: object, field: str) -> Rational:
    try:
        numerator_raw, denominator_raw = str(value).split("/", maxsplit=1)
        rational = Rational(numerator=int(numerator_raw), denominator=int(denominator_raw))
    except (TypeError, ValueError) as error:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 不是合法有理数") from error
    if rational.value <= 0:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须为正数")
    return rational


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是对象")
    return value


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是数组")
    return value


def _type(value: object) -> object:
    return value.get("codec_type") if isinstance(value, dict) else None


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是非空字符串")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _non_negative_int(value: object, field: str) -> int:
    parsed = _integer(value, field)
    if parsed < 0:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 不能为负数")
    return parsed


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是整数")
    try:
        return int(value)
    except ValueError as error:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须是整数") from error


def _positive_int(value: object, field: str) -> int:
    parsed = _non_negative_int(value, field)
    if parsed <= 0:
        raise VideoDemoError(ErrorCode.VIDEO_PROBE_INVALID, f"{field} 必须大于 0")
    return parsed
