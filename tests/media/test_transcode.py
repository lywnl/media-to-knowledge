from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from video_demo.domain.manifest import SubtitleStream
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult
from video_demo.media.transcode import (
    FFmpegTranscoder,
    NoAudioArtifact,
    SubtitleLimits,
    TranscodeLimits,
    duration_aware_timeout_seconds,
)


class WritingRunner:
    def __init__(self, *, returncode: int = 0, output: bytes = b"media") -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[int] = []
        self.returncode = returncode
        self.output = output

    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: int,
        pass_fds: tuple[int, ...] = (),
        output_paths: tuple[Path, ...] = (),
    ) -> ProcessResult:
        command = list(args)
        self.calls.append(command)
        self.timeouts.append(timeout_seconds)
        if self.returncode == 0 and (output_paths or pass_fds):
            if pass_fds:
                output_descriptor = pass_fds[-1]
                os.write(output_descriptor, self.output)
            else:
                output_paths[0].write_bytes(self.output)
        return ProcessResult(self.returncode, b"", b"safe failure")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "runtime" / "runs" / "run_001" / "input" / "source.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source")
    return path


def _transcoder(
    tmp_path: Path,
    runner: WritingRunner,
    *,
    available_bytes: int = 1024 * 1024,
    max_output_bytes: int = 1024,
    subtitle_max_output_bytes: int = 256,
    timeout_seconds: int = 1_800,
) -> FFmpegTranscoder:
    runtime = tmp_path / "runtime"
    return FFmpegTranscoder(
        executable=Path("/tools/ffmpeg"),
        runner=runner,
        runtime_root=runtime,
        limits=TranscodeLimits(
            max_output_bytes=max_output_bytes,
            required_free_bytes=128,
            timeout_seconds=timeout_seconds,
        ),
        subtitle_limits=SubtitleLimits(max_output_bytes=subtitle_max_output_bytes),
        available_bytes=lambda _path: available_bytes,
    )


def test_extract_subtitle_maps_absolute_stream_and_forces_webvtt(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"WEBVTT\n")
    stream = SubtitleStream(
        index=7,
        codec_name="mov_text",
        language="zh",
        is_default=True,
    )

    artifact = _transcoder(tmp_path, runner).extract_subtitle(
        source,
        Path("runs/run_001"),
        stream,
    )

    command = runner.calls[0]
    assert command[command.index("-map") : command.index("-map") + 2] == ["-map", "0:7"]
    assert command[command.index("-c:s") : command.index("-c:s") + 2] == [
        "-c:s",
        "webvtt",
    ]
    assert command[command.index("-f") : command.index("-f") + 2] == ["-f", "webvtt"]
    assert command[-3:-1] == ["-fs", "256"]
    assert artifact.relative_path == "runs/run_001/media/subtitles/7.vtt"
    assert artifact.stream_index == 7
    assert artifact.language == "zh"
    assert artifact.codec_name == "mov_text"


@pytest.mark.parametrize("linked_component", ["subtitles", "media"])
def test_extract_subtitle_rejects_destination_parent_symlink(
    tmp_path: Path,
    source: Path,
    linked_component: str,
) -> None:
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    media = runtime / "runs/run_001/media"
    if linked_component == "media":
        media.parent.mkdir(parents=True, exist_ok=True)
        media.symlink_to(outside, target_is_directory=True)
    else:
        media.mkdir(parents=True, exist_ok=True)
        (media / "subtitles").symlink_to(outside, target_is_directory=True)
    runner = WritingRunner(output=b"must-not-write")

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).extract_subtitle(
            source,
            Path("runs/run_001"),
            SubtitleStream(index=2, codec_name="ass", language="en"),
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []
    assert not (outside / "2.vtt").exists()


def test_extract_audio_builds_16khz_mono_mp3_command(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"mp3")
    transcoder = _transcoder(tmp_path, runner)

    artifact = transcoder.extract_audio(
        source,
        Path("runs/run_001"),
        has_audio=True,
        duration_ms=921_400,
    )

    command = runner.calls[0]
    assert command[:4] == ["/tools/ffmpeg", "-nostdin", "-hide_banner", "-loglevel"]
    assert command[4] == "error"
    assert command[command.index("-map") : command.index("-map") + 2] == ["-map", "0:a:0"]
    assert command[command.index("-ac") : command.index("-ac") + 2] == ["-ac", "1"]
    assert command[command.index("-ar") : command.index("-ar") + 2] == ["-ar", "16000"]
    assert command[command.index("-c:a") : command.index("-c:a") + 2] == [
        "-c:a",
        "libmp3lame",
    ]
    assert command[command.index("-b:a") : command.index("-b:a") + 2] == [
        "-b:a",
        "192k",
    ]
    assert command[command.index("-t") : command.index("-t") + 2] == [
        "-t",
        "921.400",
    ]
    assert "asetpts=PTS-STARTPTS" in command
    assert artifact.relative_path == "runs/run_001/media/audio.mp3"
    assert artifact.sample_rate_hz == 16_000
    assert artifact.channels == 1
    assert (tmp_path / "runtime" / artifact.relative_path).read_bytes() == b"mp3"


def test_extract_audio_without_track_returns_explicit_no_audio(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    artifact = _transcoder(tmp_path, runner).extract_audio(
        source,
        Path("runs/run_001"),
        has_audio=False,
        duration_ms=1_000,
    )

    assert artifact == NoAudioArtifact(warning_code="NO_AUDIO_TRACK")
    assert runner.calls == []


@pytest.mark.skipif(os.name != "posix", reason="fd 输出契约仅适用于 POSIX")
def test_extract_audio_applies_timeline_to_preopened_output(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"mp3")
    source_descriptor = os.open(source, os.O_RDONLY)
    output_path = tmp_path / "audio-output.mp3"
    output_descriptor = os.open(
        output_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _transcoder(tmp_path, runner).extract_audio(
            source,
            Path("runs/run_001"),
            has_audio=True,
            duration_ms=302_101,
            input_fd=source_descriptor,
            output_fd=output_descriptor,
        )
    finally:
        os.close(source_descriptor)
        os.close(output_descriptor)

    command = runner.calls[0]
    assert command[command.index("-t") : command.index("-t") + 2] == [
        "-t",
        "302.101",
    ]
    assert output_path.read_bytes() == b"mp3"


def test_duration_aware_timeout_uses_base_duration_and_hard_cap() -> None:
    assert duration_aware_timeout_seconds(3_600, 10_000) == 3_600
    assert duration_aware_timeout_seconds(1, 1_001) == 302
    assert duration_aware_timeout_seconds(600, 7_200_000) == 11_100
    assert duration_aware_timeout_seconds(600, 86_400_000) == 14_400
    with pytest.raises(ValueError, match="基础超时"):
        duration_aware_timeout_seconds(14_401, 1_000)


def test_full_length_ffmpeg_operations_receive_duration_aware_timeout(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"media")
    transcoder = _transcoder(tmp_path, runner, timeout_seconds=600)

    transcoder.extract_audio(
        source,
        Path("runs/run_001"),
        has_audio=True,
        duration_ms=7_200_000,
    )

    assert runner.timeouts == [11_100]



def test_create_clip_uses_exact_millisecond_half_open_range(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"clip")

    artifact = _transcoder(tmp_path, runner).create_clip(
        source,
        Path("runs/run_001"),
        "clip_001",
        TimeRange(start_ms=1_250, end_ms=3_750),
    )

    command = runner.calls[0]
    assert command[command.index("-ss") + 1] == "1.250"
    assert command[command.index("-t") + 1] == "2.500"
    assert artifact.start_ms == 1_250
    assert artifact.end_ms == 3_750


def test_create_audio_slice_is_scoped_mp3_and_uses_exact_millisecond_range(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"mp3-slice")

    artifact = _transcoder(tmp_path, runner).create_audio_slice(
        source,
        Path("runs/run_001"),
        "lid_vad_001",
        TimeRange(start_ms=1_250, end_ms=3_750),
        source_duration_ms=4_000,
    )

    command = runner.calls[0]
    assert command[command.index("-ss") + 1] == "1.250"
    assert command[command.index("-t") + 1] == "2.500"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "libmp3lame"
    assert command[command.index("-b:a") + 1] == "192k"
    assert command[-3:-1] == ["-fs", "1024"]
    assert artifact.relative_path == "runs/run_001/speech/slices/lid_vad_001.mp3"
    assert artifact.start_ms == 1_250
    assert artifact.end_ms == 3_750


@pytest.mark.parametrize("slice_id", ["../escape", "x", "bad/id", "空 格"])
def test_create_audio_slice_rejects_invalid_id_before_starting(
    tmp_path: Path,
    source: Path,
    slice_id: str,
) -> None:
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_audio_slice(
            source,
            Path("runs/run_001"),
            slice_id,
            TimeRange(start_ms=0, end_ms=1_000),
            source_duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.INVALID_PATH_COMPONENT
    assert runner.calls == []


def test_create_audio_slice_rejects_range_beyond_declared_audio_duration(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_audio_slice(
            source,
            Path("runs/run_001"),
            "lid_vad_001",
            TimeRange(start_ms=3_000, end_ms=4_001),
            source_duration_ms=4_000,
        )

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert runner.calls == []


def test_create_audio_slice_requires_declared_audio_duration(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    with pytest.raises(TypeError):
        _transcoder(tmp_path, runner).create_audio_slice(  # type: ignore[call-arg]
            source,
            Path("runs/run_001"),
            "lid_vad_001",
            TimeRange(start_ms=0, end_ms=1_000),
        )

    assert runner.calls == []


def test_create_audio_slice_rejects_symlink_source_before_starting(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    target = runtime / "runs/run_001/media/audio.mp3"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"mp3")
    source = target.with_name("audio-link.mp3")
    source.symlink_to(target)
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_audio_slice(
            source,
            Path("runs/run_001"),
            "lid_vad_001",
            TimeRange(start_ms=0, end_ms=1_000),
            source_duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []


def test_create_audio_slice_rejects_source_from_another_run(tmp_path: Path) -> None:
    source = tmp_path / "runtime/runs/run_002/media/audio.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mp3")
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_audio_slice(
            source,
            Path("runs/run_001"),
            "lid_vad_001",
            TimeRange(start_ms=0, end_ms=1_000),
            source_duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []


@pytest.mark.parametrize("linked_component", ["speech", "slices"])
def test_create_audio_slice_rejects_destination_parent_symlink_before_runner(
    tmp_path: Path,
    linked_component: str,
) -> None:
    runtime = tmp_path / "runtime"
    source = runtime / "runs/run_001/media/audio.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mp3")
    other_speech = runtime / "runs/run_002/speech"
    other_slices = other_speech / "slices"
    other_slices.mkdir(parents=True)
    current_speech = runtime / "runs/run_001/speech"
    if linked_component == "speech":
        current_speech.parent.mkdir(parents=True, exist_ok=True)
        current_speech.symlink_to(other_speech, target_is_directory=True)
    else:
        current_speech.mkdir(parents=True)
        (current_speech / "slices").symlink_to(other_slices, target_is_directory=True)
    runner = WritingRunner(output=b"must-not-write")

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_audio_slice(
            source,
            Path("runs/run_001"),
            "lid_vad_001",
            TimeRange(start_ms=0, end_ms=1_000),
            source_duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []
    assert not (other_slices / "lid_vad_001.mp3").exists()


def test_transcode_applies_process_output_limit_to_every_ffmpeg_output(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()
    transcoder = _transcoder(tmp_path, runner, max_output_bytes=1024)

    transcoder.extract_audio(
        source,
        Path("runs/run_001"),
        has_audio=True,
        duration_ms=1_000,
    )
    transcoder.create_clip(
        source,
        Path("runs/run_001"),
        "clip_001",
        TimeRange(start_ms=0, end_ms=1_000),
    )

    assert len(runner.calls) == 2
    for command in runner.calls:
        assert command[-3:-1] == ["-fs", "1024"]


def test_transcode_rejects_insufficient_disk_before_starting(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner, available_bytes=64).extract_audio(
            source,
            Path("runs/run_001"),
            has_audio=True,
            duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT
    assert runner.calls == []


def test_transcode_rejects_nonzero_empty_and_oversized_output(
    tmp_path: Path,
    source: Path,
) -> None:
    with pytest.raises(VideoDemoError) as nonzero:
        _transcoder(tmp_path, WritingRunner(returncode=1)).extract_audio(
            source,
            Path("runs/run_001"),
            has_audio=True,
            duration_ms=1_000,
        )
    assert nonzero.value.code == ErrorCode.VIDEO_PROCESS_FAILED

    with pytest.raises(VideoDemoError) as empty:
        _transcoder(tmp_path, WritingRunner(output=b"")).extract_audio(
            source,
            Path("runs/run_001"),
            has_audio=True,
            duration_ms=1_000,
        )
    assert empty.value.code == ErrorCode.VIDEO_OUTPUT_INVALID

    with pytest.raises(VideoDemoError) as oversized:
        _transcoder(tmp_path, WritingRunner(output=b"12345"), max_output_bytes=4).extract_audio(
            source,
            Path("runs/run_001"),
            has_audio=True,
            duration_ms=1_000,
        )
    assert oversized.value.code == ErrorCode.VIDEO_OUTPUT_TOO_LARGE


def test_missing_ffmpeg_is_reported_explicitly(tmp_path: Path) -> None:
    with pytest.raises(VideoDemoError) as raised:
        FFmpegTranscoder.from_path(
            tmp_path / "missing",
            tmp_path / "runtime",
            workspace_root=tmp_path,
        )

    assert raised.value.code == ErrorCode.VIDEO_FFMPEG_UNAVAILABLE


@pytest.mark.parametrize("binary_kind", ["outside", "parent_symlink", "not_executable"])
def test_ffmpeg_constructor_rejects_unsafe_binary_before_execution(
    tmp_path: Path,
    binary_kind: str,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = workspace / ".codex" / "video-rag-demo"
    runtime.mkdir(parents=True)
    if binary_kind == "outside":
        executable = tmp_path / "external" / "ffmpeg"
    elif binary_kind == "parent_symlink":
        external_tools = tmp_path / "external-tools"
        external_tools.mkdir()
        (runtime / "tools").symlink_to(external_tools, target_is_directory=True)
        executable = runtime / "tools" / "ffmpeg"
    else:
        executable = runtime / "tools" / "ffmpeg"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    executable.chmod(0o644 if binary_kind == "not_executable" else 0o755)

    with pytest.raises(VideoDemoError) as raised:
        FFmpegTranscoder.from_path(
            executable,
            runtime,
            workspace_root=workspace,
        )

    assert raised.value.code == ErrorCode.VIDEO_FFMPEG_UNAVAILABLE
