"""音频专用 FFmpeg 转码内核。"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from video_demo.capabilities import resolve_workspace_binary
from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.audio_format import (
    AUDIO_BITRATE,
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_ENCODER,
    AUDIO_OUTPUT_EXTENSION,
    AUDIO_SAMPLE_RATE_HZ,
)
from video_demo.media.process import ProcessErrorCodes, ProcessResult, SafeProcessRunner
from video_demo.storage.workspace import atomic_replace, safe_runtime_path, validate_path_component

MAX_DURATION_AWARE_TIMEOUT_SECONDS = 14_400


def duration_aware_timeout_seconds(base_timeout_seconds: int, duration_ms: int) -> int:
    if (
        type(base_timeout_seconds) is not int
        or not 1 <= base_timeout_seconds <= MAX_DURATION_AWARE_TIMEOUT_SECONDS
    ):
        raise ValueError("基础超时必须位于 1~14400 秒")
    if type(duration_ms) is not int or duration_ms < 1:
        raise ValueError("媒体时长必须是正整数毫秒")
    duration_budget = (duration_ms * 3 + 1_999) // 2_000 + 300
    return min(MAX_DURATION_AWARE_TIMEOUT_SECONDS, max(base_timeout_seconds, duration_budget))


class ProcessRunner(Protocol):
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
        output_paths: tuple[Path, ...] = (),
    ) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class AudioTranscodeLimits:
    max_output_bytes: int = 4 * 1024 * 1024 * 1024
    required_free_bytes: int = 512 * 1024 * 1024
    timeout_seconds: int = 1_800

    def __post_init__(self) -> None:
        if self.max_output_bytes < 1 or self.required_free_bytes < 0:
            raise ValueError("音频转码字节预算非法")
        if not 1 <= self.timeout_seconds <= MAX_DURATION_AWARE_TIMEOUT_SECONDS:
            raise ValueError("音频转码基础超时必须位于 1~14400 秒")


class AudioArtifact(FrozenModel):
    relative_path: str
    sha256: Sha256
    size_bytes: int
    sample_rate_hz: int
    channels: int
    codec: str


class NoAudioArtifact(FrozenModel):
    warning_code: str


class AudioSliceArtifact(TimeRange):
    slice_id: str
    relative_path: str
    sha256: Sha256
    size_bytes: int
    sample_rate_hz: int = AUDIO_SAMPLE_RATE_HZ
    channels: int = AUDIO_CHANNELS
    codec: str = AUDIO_CODEC


class AudioTranscoder:
    """只生成 MP3 音频和 ASR 音频切片，不承担其他媒体处理。"""

    def __init__(
        self,
        *,
        executable: Path,
        runner: ProcessRunner,
        runtime_root: Path,
        limits: AudioTranscodeLimits | None = None,
        available_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self._executable = executable
        self._runner = runner
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._limits = limits or AudioTranscodeLimits()
        self._available_bytes = available_bytes or self._disk_free_bytes

    @classmethod
    def from_path(
        cls,
        executable: Path,
        runtime_root: Path,
        *,
        workspace_root: Path,
        is_cancel_requested: Callable[[], bool] = lambda: False,
        limits: AudioTranscodeLimits | None = None,
    ) -> AudioTranscoder:
        executable = resolve_workspace_binary(
            executable,
            workspace_root=workspace_root,
            unavailable_code=ErrorCode.AUDIO_FFMPEG_UNAVAILABLE,
        )
        runner = SafeProcessRunner(
            max_output_bytes=16 * 1024 * 1024,
            is_cancel_requested=is_cancel_requested,
            workspace_root=workspace_root,
            error_codes=ProcessErrorCodes(
                invalid=ErrorCode.AUDIO_PROCESS_FAILED,
                cancelled=ErrorCode.AUDIO_PROCESS_CANCELLED,
                timeout=ErrorCode.AUDIO_PROCESS_TIMEOUT,
                output_too_large=ErrorCode.AUDIO_PROCESS_OUTPUT_TOO_LARGE,
            ),
        )
        version = runner.run([str(executable), "-version"], timeout_seconds=10)
        if version.returncode != 0:
            raise VideoDemoError(ErrorCode.AUDIO_BINARY_PROBE_FAILED, "音频工具版本探测失败")
        return cls(
            executable=executable,
            runner=runner,
            runtime_root=runtime_root,
            limits=limits,
        )

    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool = True,
        duration_ms: int,
        input_fd: int | None = None,
        output_fd: int | None = None,
    ) -> AudioArtifact | NoAudioArtifact:
        if not has_audio:
            return NoAudioArtifact(warning_code="NO_AUDIO_TRACK")
        source = self._trusted_input(source, input_fd)
        relative_path = run_relative_root / "media" / f"audio{AUDIO_OUTPUT_EXTENSION}"
        args = [
            *self._base_args(source),
            "-t", _seconds(duration_ms),
            "-map", "0:a:0",
            "-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE_HZ),
            "-c:a", AUDIO_ENCODER, "-b:a", AUDIO_BITRATE, "-af", "asetpts=PTS-STARTPTS",
        ]
        if input_fd is not None or output_fd is not None:
            input_descriptor, output_descriptor = _require_descriptor_pair(input_fd, output_fd)
            args = [
                *self._base_args(Path(f"/dev/fd/{input_descriptor}")),
                "-t", _seconds(duration_ms),
                "-map", "0:a:0",
                "-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE_HZ),
                "-c:a", AUDIO_ENCODER, "-b:a", AUDIO_BITRATE, "-af", "asetpts=PTS-STARTPTS",
            ]
            size_bytes, sha256 = self._produce_to_fd(
                args,
                output_descriptor,
                (input_descriptor, output_descriptor),
                timeout_seconds=self._timeout_for(duration_ms),
            )
        else:
            final_path = self._destination_path(relative_path)
            size_bytes, sha256 = self._produce(
                args,
                final_path,
                timeout_seconds=self._timeout_for(duration_ms),
            )
        return AudioArtifact(
            relative_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=size_bytes,
            sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
            channels=AUDIO_CHANNELS,
            codec=AUDIO_CODEC,
        )

    def create_audio_slice(
        self,
        source: Path,
        run_relative_root: Path,
        slice_id: str,
        time_range: TimeRange,
        *,
        source_duration_ms: int,
    ) -> AudioSliceArtifact:
        source = self._validate_source(source)
        validate_path_component(slice_id, "slice_id")
        run_root = safe_runtime_path(self._runtime_root, run_relative_root)
        if not source.is_relative_to(run_root):
            raise VideoDemoError(
                ErrorCode.AUDIO_INPUT_INVALID,
                "音频切片输入必须位于当前运行目录内",
            )
        if time_range.end_ms > source_duration_ms:
            raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, "音频切片时间范围越界")
        relative_path = (
            run_relative_root
            / "speech"
            / "slices"
            / f"{slice_id}{AUDIO_OUTPUT_EXTENSION}"
        )
        final_path = self._destination_path(relative_path)
        args = [
            *self._base_args(source),
            "-ss", _seconds(time_range.start_ms),
            "-t", _seconds(time_range.duration_ms),
            "-map", "0:a:0", "-vn", "-ac", str(AUDIO_CHANNELS),
            "-ar", str(AUDIO_SAMPLE_RATE_HZ), "-c:a", AUDIO_ENCODER,
            "-b:a", AUDIO_BITRATE, "-af", "asetpts=PTS-STARTPTS",
        ]
        size_bytes, sha256 = self._produce(
            args,
            final_path,
            timeout_seconds=self._timeout_for(time_range.duration_ms),
        )
        return AudioSliceArtifact(
            slice_id=slice_id,
            start_ms=time_range.start_ms,
            end_ms=time_range.end_ms,
            relative_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def _base_args(self, source: Path) -> list[str]:
        return [
            str(self._executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
        ]

    def _produce(
        self,
        args_without_output: list[str],
        final_path: Path,
        *,
        timeout_seconds: int,
    ) -> tuple[int, str]:
        self._reject_destination_symlinks(final_path)
        self._require_disk_space(final_path.parent)
        temporary = final_path.with_name(
            f".{final_path.stem}.{uuid.uuid4().hex}.part{final_path.suffix}",
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self._reject_destination_symlinks(final_path)
        try:
            result = self._runner.run(
                [
                    *args_without_output,
                    "-y",
                    "-fs",
                    str(self._limits.max_output_bytes),
                    str(temporary),
                ],
                timeout_seconds=timeout_seconds,
                output_paths=(temporary,),
            )
            if result.returncode != 0:
                raise VideoDemoError(ErrorCode.AUDIO_PROCESS_FAILED, "音频工具处理失败")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_INVALID, "音频工具未生成有效输出")
            size_bytes = temporary.stat().st_size
            if size_bytes > self._limits.max_output_bytes:
                raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_TOO_LARGE, "音频输出超过限制")
            sha256 = _sha256_file(temporary)
            self._reject_destination_symlinks(final_path)
            atomic_replace(temporary, final_path)
            return size_bytes, sha256
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _produce_to_fd(
        self,
        args_without_output: list[str],
        output_fd: int,
        pass_fds: tuple[int, ...],
        *,
        timeout_seconds: int,
    ) -> tuple[int, str]:
        result = self._runner.run(
            [
                *args_without_output,
                "-y",
                "-fs",
                str(self._limits.max_output_bytes),
                "-f",
                "mp3",
                f"/dev/fd/{output_fd}",
            ],
            timeout_seconds=timeout_seconds,
            pass_fds=pass_fds,
        )
        if result.returncode != 0:
            raise VideoDemoError(ErrorCode.AUDIO_PROCESS_FAILED, "音频工具处理失败")
        return _snapshot_descriptor(output_fd, self._limits.max_output_bytes)

    def _destination_path(self, relative_path: Path) -> Path:
        safe_runtime_path(self._runtime_root, relative_path)
        candidate = self._runtime_root / relative_path
        self._reject_destination_symlinks(candidate)
        return candidate

    def _reject_destination_symlinks(self, destination: Path) -> None:
        try:
            relative_path = destination.relative_to(self._runtime_root)
        except ValueError as error:
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频输出必须位于运行目录内",
            ) from error
        current = self._runtime_root
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "音频输出路径不能包含符号链接",
                )

    def _validate_source(self, source: Path) -> Path:
        candidate = source.expanduser()
        if not candidate.is_absolute():
            candidate = self._runtime_root / candidate
        try:
            relative_path = candidate.relative_to(self._runtime_root)
        except ValueError as error:
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频输入必须位于运行目录内",
            ) from error
        current = self._runtime_root
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "音频输入路径不能包含符号链接",
                )
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self._runtime_root):
            raise VideoDemoError(ErrorCode.WORKSPACE_PATH_ESCAPE, "音频输入必须位于运行目录内")
        if not resolved.is_file():
            raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, "音频输入不是普通文件")
        return resolved

    def _trusted_input(self, source: Path, input_fd: int | None) -> Path:
        if input_fd is None:
            return self._validate_source(source)
        _require_regular_descriptor(input_fd, "音频输入 fd 非法")
        return source

    def _require_disk_space(self, destination_parent: Path) -> None:
        destination_parent.mkdir(parents=True, exist_ok=True)
        if self._available_bytes(destination_parent) < self._limits.required_free_bytes:
            raise VideoDemoError(
                ErrorCode.AUDIO_DISK_SPACE_INSUFFICIENT,
                "音频转码可用磁盘空间不足",
            )

    def _timeout_for(self, duration_ms: int) -> int:
        return duration_aware_timeout_seconds(self._limits.timeout_seconds, duration_ms)

    @staticmethod
    def _disk_free_bytes(path: Path) -> int:
        return shutil.disk_usage(path).free


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_descriptor_pair(input_fd: int | None, output_fd: int | None) -> tuple[int, int]:
    if input_fd is None or output_fd is None or input_fd == output_fd:
        raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, "音频 fd 能力不完整")
    _require_regular_descriptor(input_fd, "音频输入 fd 非法")
    _require_regular_descriptor(output_fd, "音频输出 fd 非法")
    return input_fd, output_fd


def _require_regular_descriptor(descriptor: int, message: str) -> None:
    if os.name != "posix" or type(descriptor) is not int or descriptor < 0:
        raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, message)
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, message) from None
    if not stat.S_ISREG(details.st_mode):
        raise VideoDemoError(ErrorCode.AUDIO_INPUT_INVALID, message)


def _snapshot_descriptor(descriptor: int, max_bytes: int) -> tuple[int, str]:
    details = os.fstat(descriptor)
    if details.st_size < 1:
        raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_INVALID, "音频工具未生成有效输出")
    if details.st_size > max_bytes:
        raise VideoDemoError(ErrorCode.AUDIO_OUTPUT_TOO_LARGE, "音频输出超过限制")
    digest = hashlib.sha256()
    offset = 0
    while offset < details.st_size:
        chunk = os.pread(descriptor, min(1024 * 1024, details.st_size - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if (
        offset != details.st_size
        or (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise VideoDemoError(
            ErrorCode.AUDIO_OUTPUT_INVALID,
            "音频输出读取期间发生变化",
        )
    return details.st_size, digest.hexdigest()


__all__ = [
    "AudioArtifact",
    "AudioSliceArtifact",
    "AudioTranscodeLimits",
    "AudioTranscoder",
    "NoAudioArtifact",
    "duration_aware_timeout_seconds",
]
