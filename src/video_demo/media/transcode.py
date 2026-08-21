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
from video_demo.media.process import ProcessResult, SafeProcessRunner
from video_demo.storage.workspace import atomic_replace, safe_runtime_path, validate_path_component


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
class TranscodeLimits:
    max_output_bytes: int = 4 * 1024 * 1024 * 1024
    required_free_bytes: int = 512 * 1024 * 1024
    timeout_seconds: int = 1_800


class AudioArtifact(FrozenModel):
    relative_path: str
    sha256: Sha256
    size_bytes: int
    sample_rate_hz: int
    channels: int
    codec: str


class NoAudioArtifact(FrozenModel):
    warning_code: str


class ProxyVideoArtifact(FrozenModel):
    relative_path: str
    sha256: Sha256
    size_bytes: int
    max_edge: int
    normalized_start_ms: int


class ClipArtifact(TimeRange):
    clip_id: str
    relative_path: str
    sha256: Sha256
    size_bytes: int


class AudioSliceArtifact(TimeRange):
    slice_id: str
    relative_path: str
    sha256: Sha256
    size_bytes: int
    sample_rate_hz: int = 16_000
    channels: int = 1
    codec: str = "pcm_s16le"


class FFmpegTranscoder:
    def __init__(
        self,
        *,
        executable: Path,
        runner: ProcessRunner,
        runtime_root: Path,
        limits: TranscodeLimits | None = None,
        available_bytes: Callable[[Path], int] | None = None,
    ) -> None:
        self._executable = executable
        self._runner = runner
        self._runtime_root = runtime_root.expanduser().resolve(strict=False)
        self._limits = limits or TranscodeLimits()
        self._available_bytes = available_bytes or self._disk_free_bytes

    @classmethod
    def from_path(
        cls,
        executable: Path,
        runtime_root: Path,
        *,
        workspace_root: Path,
        is_cancel_requested: Callable[[], bool] = lambda: False,
    ) -> FFmpegTranscoder:
        executable = resolve_workspace_binary(
            executable,
            workspace_root=workspace_root,
            unavailable_code=ErrorCode.VIDEO_FFMPEG_UNAVAILABLE,
        )
        runner = SafeProcessRunner(
            max_output_bytes=16 * 1024 * 1024,
            is_cancel_requested=is_cancel_requested,
            workspace_root=workspace_root,
        )
        version = runner.run([str(executable), "-version"], timeout_seconds=10)
        if version.returncode != 0:
            raise VideoDemoError(ErrorCode.VIDEO_BINARY_PROBE_FAILED, "ffmpeg 版本探测失败")
        return cls(executable=executable, runner=runner, runtime_root=runtime_root)

    def extract_audio(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        has_audio: bool,
        input_fd: int | None = None,
        output_fd: int | None = None,
    ) -> AudioArtifact | NoAudioArtifact:
        if not has_audio:
            return NoAudioArtifact(warning_code="NO_AUDIO_TRACK")
        source = self._trusted_input(source, input_fd)
        relative_path = run_relative_root / "media" / "audio.wav"
        if input_fd is not None or output_fd is not None:
            input_descriptor, output_descriptor = _require_descriptor_pair(
                input_fd,
                output_fd,
            )
            args = [
                *self._base_args(Path(f"/dev/fd/{input_descriptor}")),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-af",
                "asetpts=PTS-STARTPTS",
            ]
            size_bytes, sha256 = self._produce_to_fd(
                args,
                output_descriptor,
                "wav",
                (input_descriptor, output_descriptor),
            )
            return AudioArtifact(
                relative_path=relative_path.as_posix(),
                sha256=sha256,
                size_bytes=size_bytes,
                sample_rate_hz=16_000,
                channels=1,
                codec="pcm_s16le",
            )
        final_path = self._destination_path(relative_path)
        args = [
            *self._base_args(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-af",
            "asetpts=PTS-STARTPTS",
        ]
        size_bytes, sha256 = self._produce(args, final_path)
        return AudioArtifact(
            relative_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=size_bytes,
            sample_rate_hz=16_000,
            channels=1,
            codec="pcm_s16le",
        )

    def create_proxy(
        self,
        source: Path,
        run_relative_root: Path,
        *,
        input_fd: int | None = None,
        output_fd: int | None = None,
    ) -> ProxyVideoArtifact:
        source = self._trusted_input(source, input_fd)
        relative_path = run_relative_root / "media" / "proxy.mp4"
        # MP4 muxer 会回写索引；macOS 的 /dev/fd 共享偏移会因此静默破坏码流。
        # fd 输出改用无需回写头部的 fragmented MP4，普通路径仍保留 faststart。
        movflags = (
            "+frag_keyframe+empty_moov+delay_moov"
            if output_fd is not None
            else "+faststart"
        )
        video_filter = (
            "scale=w='if(gte(iw,ih),min(1280,iw),-2)':"
            "h='if(gte(iw,ih),-2,min(1280,ih))':flags=lanczos,"
            "setpts=PTS-STARTPTS"
        )
        args = [
            *self._base_args(
                Path(f"/dev/fd/{input_fd}") if input_fd is not None else source
            ),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            video_filter,
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            movflags,
        ]
        if input_fd is not None or output_fd is not None:
            input_descriptor, output_descriptor = _require_descriptor_pair(
                input_fd,
                output_fd,
            )
            size_bytes, sha256 = self._produce_to_fd(
                args,
                output_descriptor,
                "mp4",
                (input_descriptor, output_descriptor),
            )
        else:
            final_path = self._destination_path(relative_path)
            size_bytes, sha256 = self._produce(args, final_path)
        return ProxyVideoArtifact(
            relative_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=size_bytes,
            max_edge=1280,
            normalized_start_ms=0,
        )

    def create_clip(
        self,
        source: Path,
        run_relative_root: Path,
        clip_id: str,
        time_range: TimeRange,
    ) -> ClipArtifact:
        source = self._validate_source(source)
        validate_path_component(clip_id, "clip_id")
        relative_path = run_relative_root / "visual" / "clips" / f"{clip_id}.mp4"
        final_path = self._destination_path(relative_path)
        args = [
            str(self._executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _seconds(time_range.start_ms),
            "-i",
            str(source),
            "-t",
            _seconds(time_range.duration_ms),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "setpts=PTS-STARTPTS",
            "-af",
            "asetpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ]
        size_bytes, sha256 = self._produce(args, final_path)
        return ClipArtifact(
            clip_id=clip_id,
            start_ms=time_range.start_ms,
            end_ms=time_range.end_ms,
            relative_path=relative_path.as_posix(),
            sha256=sha256,
            size_bytes=size_bytes,
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
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "音频切片输入必须位于当前运行目录内",
            )
        if time_range.end_ms > source_duration_ms:
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "音频切片时间范围越界")
        relative_path = run_relative_root / "speech" / "slices" / f"{slice_id}.wav"
        final_path = self._destination_path(relative_path)
        args = [
            str(self._executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _seconds(time_range.start_ms),
            "-i",
            str(source),
            "-t",
            _seconds(time_range.duration_ms),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-af",
            "asetpts=PTS-STARTPTS",
        ]
        size_bytes, sha256 = self._produce(args, final_path)
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

    def _produce(self, args_without_output: list[str], final_path: Path) -> tuple[int, str]:
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
                timeout_seconds=self._limits.timeout_seconds,
                output_paths=(temporary,),
            )
            if result.returncode != 0:
                raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "ffmpeg 处理失败")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "ffmpeg 未生成有效输出")
            size_bytes = temporary.stat().st_size
            if size_bytes > self._limits.max_output_bytes:
                raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "ffmpeg 输出超过限制")
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
        format_name: str,
        pass_fds: tuple[int, ...],
    ) -> tuple[int, str]:
        result = self._runner.run(
            [
                *args_without_output,
                "-y",
                "-fs",
                str(self._limits.max_output_bytes),
                "-f",
                format_name,
                f"/dev/fd/{output_fd}",
            ],
            timeout_seconds=self._limits.timeout_seconds,
            pass_fds=pass_fds,
        )
        if result.returncode != 0:
            raise VideoDemoError(ErrorCode.VIDEO_PROCESS_FAILED, "ffmpeg 处理失败")
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
                "FFmpeg 输出必须位于运行目录内",
            ) from error
        current = self._runtime_root
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "FFmpeg 输出路径不能包含符号链接",
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
                "FFmpeg 输入必须位于运行目录内",
            ) from error

        current = self._runtime_root
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                raise VideoDemoError(
                    ErrorCode.WORKSPACE_PATH_ESCAPE,
                    "FFmpeg 输入路径不能包含符号链接",
                )

        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self._runtime_root):
            raise VideoDemoError(
                ErrorCode.WORKSPACE_PATH_ESCAPE,
                "FFmpeg 输入必须位于运行目录内",
            )
        if not resolved.is_file():
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "FFmpeg 输入不是普通文件")
        return resolved

    def _trusted_input(self, source: Path, input_fd: int | None) -> Path:
        if input_fd is None:
            return self._validate_source(source)
        _require_regular_descriptor(input_fd, "FFmpeg 输入 fd 非法")
        return source

    def _require_disk_space(self, destination_parent: Path) -> None:
        destination_parent.mkdir(parents=True, exist_ok=True)
        if self._available_bytes(destination_parent) < self._limits.required_free_bytes:
            raise VideoDemoError(ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT, "可用磁盘空间不足")

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


def _require_descriptor_pair(
    input_fd: int | None,
    output_fd: int | None,
) -> tuple[int, int]:
    if input_fd is None or output_fd is None or input_fd == output_fd:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "FFmpeg fd capability 不完整")
    _require_regular_descriptor(input_fd, "FFmpeg 输入 fd 非法")
    _require_regular_descriptor(output_fd, "FFmpeg 输出 fd 非法")
    return input_fd, output_fd


def _require_regular_descriptor(descriptor: int, message: str) -> None:
    if os.name != "posix" or type(descriptor) is not int or descriptor < 0:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, message)
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, message) from None
    if not stat.S_ISREG(details.st_mode):
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, message)


def _snapshot_descriptor(descriptor: int, max_bytes: int) -> tuple[int, str]:
    details = os.fstat(descriptor)
    if details.st_size < 1:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "ffmpeg 未生成有效输出")
    if details.st_size > max_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_TOO_LARGE, "ffmpeg 输出超过限制")
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
        or (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise VideoDemoError(ErrorCode.VIDEO_OUTPUT_INVALID, "ffmpeg 输出读取期间发生变化")
    return details.st_size, digest.hexdigest()
