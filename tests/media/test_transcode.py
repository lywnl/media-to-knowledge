from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.process import ProcessResult
from video_demo.media.transcode import (
    FFmpegTranscoder,
    NoAudioArtifact,
    TranscodeLimits,
)


class WritingRunner:
    def __init__(self, *, returncode: int = 0, output: bytes = b"media") -> None:
        self.calls: list[list[str]] = []
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
        del timeout_seconds, pass_fds
        command = list(args)
        self.calls.append(command)
        if self.returncode == 0:
            target = output_paths[0] if output_paths else Path(command[-1])
            target.write_bytes(self.output)
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
) -> FFmpegTranscoder:
    runtime = tmp_path / "runtime"
    return FFmpegTranscoder(
        executable=Path("/tools/ffmpeg"),
        runner=runner,
        runtime_root=runtime,
        limits=TranscodeLimits(
            max_output_bytes=max_output_bytes,
            required_free_bytes=128,
        ),
        available_bytes=lambda _path: available_bytes,
    )


def test_extract_audio_builds_16khz_mono_pcm_command(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"wav")
    transcoder = _transcoder(tmp_path, runner)

    artifact = transcoder.extract_audio(source, Path("runs/run_001"), has_audio=True)

    command = runner.calls[0]
    assert command[:4] == ["/tools/ffmpeg", "-nostdin", "-hide_banner", "-loglevel"]
    assert command[4] == "error"
    assert command[command.index("-map") : command.index("-map") + 2] == ["-map", "0:a:0"]
    assert command[command.index("-ac") : command.index("-ac") + 2] == ["-ac", "1"]
    assert command[command.index("-ar") : command.index("-ar") + 2] == ["-ar", "16000"]
    assert command[
        command.index("-c:a") : command.index("-c:a") + 2
    ] == ["-c:a", "pcm_s16le"]
    assert "asetpts=PTS-STARTPTS" in command
    assert artifact.relative_path == "runs/run_001/media/audio.wav"
    assert artifact.sample_rate_hz == 16_000
    assert artifact.channels == 1
    assert (tmp_path / "runtime" / artifact.relative_path).read_bytes() == b"wav"


def test_extract_audio_without_track_returns_explicit_no_audio(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    artifact = _transcoder(tmp_path, runner).extract_audio(
        source,
        Path("runs/run_001"),
        has_audio=False,
    )

    assert artifact == NoAudioArtifact(warning_code="NO_AUDIO_TRACK")
    assert runner.calls == []


def test_create_proxy_limits_long_edge_and_normalizes_pts(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"proxy")

    artifact = _transcoder(tmp_path, runner).create_proxy(source, Path("runs/run_001"))

    command = runner.calls[0]
    video_filter = command[command.index("-vf") + 1]
    assert "scale" in video_filter
    assert "1280" in video_filter
    assert "setpts=PTS-STARTPTS" in video_filter
    assert "-an" in command
    assert "-vsync" in command
    assert command[command.index("-vsync") + 1] == "vfr"
    assert artifact.relative_path == "runs/run_001/media/proxy.mp4"
    assert artifact.max_edge == 1280
    assert artifact.normalized_start_ms == 0


@pytest.mark.skipif(os.name != "posix", reason="fd 输出契约仅适用于 POSIX")
def test_create_proxy_uses_fragmented_mp4_for_preopened_output(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"proxy")
    source_descriptor = os.open(source, os.O_RDONLY)
    output_path = tmp_path / "proxy-output.mp4"
    output_descriptor = os.open(
        output_path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _transcoder(tmp_path, runner).create_proxy(
            source,
            Path("runs/run_001"),
            input_fd=source_descriptor,
            output_fd=output_descriptor,
        )
    finally:
        os.close(source_descriptor)
        os.close(output_descriptor)

    command = runner.calls[0]
    assert (
        command[command.index("-movflags") + 1]
        == "+frag_keyframe+empty_moov+delay_moov"
    )
    assert output_path.read_bytes() == b"proxy"


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


def test_create_audio_slice_is_scoped_pcm_and_uses_exact_millisecond_range(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner(output=b"wav-slice")

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
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[-3:-1] == ["-fs", "1024"]
    assert artifact.relative_path == "runs/run_001/speech/slices/lid_vad_001.wav"
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
    target = runtime / "runs/run_001/media/audio.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"wav")
    source = target.with_name("audio-link.wav")
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
    source = tmp_path / "runtime/runs/run_002/media/audio.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wav")
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
    source = runtime / "runs/run_001/media/audio.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wav")
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
    assert not (other_slices / "lid_vad_001.wav").exists()


def test_transcode_rejects_source_outside_runtime_before_starting(tmp_path: Path) -> None:
    source = tmp_path / "outside" / "source.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source")
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_proxy(source, Path("runs/run_001"))

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []


def test_transcode_rejects_symlink_source_before_starting(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    target = runtime / "runs" / "run_001" / "input" / "target.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"source")
    source = target.with_name("source.mp4")
    source.symlink_to(target)
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_proxy(source, Path("runs/run_001"))

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
    assert runner.calls == []


@pytest.mark.parametrize("source_kind", ["missing", "directory"])
def test_transcode_rejects_source_that_is_not_a_regular_file(
    tmp_path: Path,
    source_kind: str,
) -> None:
    source = tmp_path / "runtime" / "runs" / "run_001" / "input" / "source.mp4"
    source.parent.mkdir(parents=True)
    if source_kind == "directory":
        source.mkdir()
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner).create_proxy(source, Path("runs/run_001"))

    assert raised.value.code == ErrorCode.VIDEO_INPUT_INVALID
    assert runner.calls == []


def test_transcode_applies_process_output_limit_to_every_ffmpeg_output(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()
    transcoder = _transcoder(tmp_path, runner, max_output_bytes=1024)

    transcoder.extract_audio(source, Path("runs/run_001"), has_audio=True)
    transcoder.create_proxy(source, Path("runs/run_001"))
    transcoder.create_clip(
        source,
        Path("runs/run_001"),
        "clip_001",
        TimeRange(start_ms=0, end_ms=1_000),
    )

    assert len(runner.calls) == 3
    for command in runner.calls:
        assert command[-3:-1] == ["-fs", "1024"]


def test_transcode_rejects_insufficient_disk_before_starting(
    tmp_path: Path,
    source: Path,
) -> None:
    runner = WritingRunner()

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, runner, available_bytes=64).create_proxy(
            source,
            Path("runs/run_001"),
        )

    assert raised.value.code == ErrorCode.VIDEO_DISK_SPACE_INSUFFICIENT
    assert runner.calls == []


def test_transcode_rejects_nonzero_empty_and_oversized_output(
    tmp_path: Path,
    source: Path,
) -> None:
    with pytest.raises(VideoDemoError) as nonzero:
        _transcoder(tmp_path, WritingRunner(returncode=1)).create_proxy(
            source,
            Path("runs/run_001"),
        )
    assert nonzero.value.code == ErrorCode.VIDEO_PROCESS_FAILED

    with pytest.raises(VideoDemoError) as empty:
        _transcoder(tmp_path, WritingRunner(output=b"")).create_proxy(
            source,
            Path("runs/run_001"),
        )
    assert empty.value.code == ErrorCode.VIDEO_OUTPUT_INVALID

    with pytest.raises(VideoDemoError) as oversized:
        _transcoder(tmp_path, WritingRunner(output=b"12345"), max_output_bytes=4).create_proxy(
            source,
            Path("runs/run_001"),
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
