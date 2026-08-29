from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult, SafeProcessRunner
from video_demo.storage.workspace import reject_symlink_components


@dataclass(frozen=True, slots=True)
class AudioProbeResult:
    duration_ms: int
    sample_rate_hz: int
    channels: int
    codec_name: str
    format_name: str


class AudioProbeClient(Protocol):
    def probe(self, source: Path, *, max_duration_ms: int) -> AudioProbeResult: ...


class FFprobeAudioClient:
    def __init__(
        self,
        executable: Path,
        *,
        workspace_root: Path,
        timeout_seconds: int = 60,
    ) -> None:
        self._executable = reject_symlink_components(
            workspace_root,
            executable,
            message="音频 ffprobe 路径非法",
        )
        self._workspace_root = workspace_root.resolve(strict=False)
        self._timeout_seconds = timeout_seconds

    def probe(self, source: Path, *, max_duration_ms: int) -> AudioProbeResult:
        source = reject_symlink_components(
            self._workspace_root,
            source,
            message="音频输入必须位于工作区内",
        )
        if not source.is_file():
            raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, "音频输入不是普通文件")
        runner = SafeProcessRunner(
            max_output_bytes=4 * 1024 * 1024,
            workspace_root=self._workspace_root,
        )
        result = runner.run(
            [
                str(self._executable),
                "-v", "error", "-select_streams", "a:0",
                "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
                "-of", "json", str(source),
            ],
            timeout_seconds=self._timeout_seconds,
        )
        if result.returncode != 0:
            raise VideoDemoError(ErrorCode.AUDIO_PROBE_INVALID, "音频无法解码")
        return parse_audio_probe_payload(result, max_duration_ms=max_duration_ms)


def parse_audio_probe_payload(result: ProcessResult, *, max_duration_ms: int) -> AudioProbeResult:
    try:
        payload: object = json.loads(result.stdout)
        root = _object(payload)
        streams = root["streams"]
        if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
            raise ValueError
        stream = streams[0]
        format_payload = root.get("format")
        duration_value = (
            format_payload.get("duration") if isinstance(format_payload, dict) else None
        )
        duration = _positive_float(duration_value)
        duration_ms = round(duration * 1_000)
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        codec = stream["codec_name"]
        if (
            duration_ms > max_duration_ms
            or sample_rate < 1
            or channels < 1
            or not isinstance(codec, str)
        ):
            raise ValueError
        return AudioProbeResult(duration_ms, sample_rate, channels, codec, _format_name(root))
    except (KeyError, TypeError, ValueError, OverflowError):
        code = (
            ErrorCode.AUDIO_DURATION_LIMIT_EXCEEDED
            if _payload_duration_exceeds(result, max_duration_ms)
            else ErrorCode.AUDIO_PROBE_INVALID
        )
        raise VideoDemoError(code, "音频预检结果非法或超过时长限制") from None


def _payload_duration_exceeds(result: ProcessResult, max_duration_ms: int) -> bool:
    try:
        payload = json.loads(result.stdout)
        duration = payload["format"]["duration"]
        return float(duration) * 1_000 > max_duration_ms
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError
    return value


def _positive_float(value: object) -> float:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ValueError
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError
    return number


def _format_name(payload: dict[str, Any]) -> str:
    value = payload.get("format", {})
    return str(value.get("format_name", "audio")) if isinstance(value, dict) else "audio"
