from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.audio_transcode import AudioTranscodeLimits, AudioTranscoder
from video_demo.media.process import ProcessResult


class Runner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[list[str]] = []

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
            assert output_paths
            output_paths[0].write_bytes(b"audio")
        return ProcessResult(self.returncode, b"", b"")


def _transcoder(tmp_path: Path, runner: Runner) -> AudioTranscoder:
    runtime = tmp_path / "runtime"
    return AudioTranscoder(
        executable=Path("/tools/ffmpeg"),
        runner=runner,
        runtime_root=runtime,
        limits=AudioTranscodeLimits(
            max_output_bytes=1024,
            required_free_bytes=0,
            timeout_seconds=60,
        ),
        available_bytes=lambda _path: 1024,
    )


def test_audio_transcoder_reports_audio_error_for_failed_process(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime" / "runs" / "run_audio" / "input" / "source.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    with pytest.raises(VideoDemoError) as raised:
        _transcoder(tmp_path, Runner(returncode=1)).extract_audio(
            source,
            Path("runs/run_audio"),
            duration_ms=1_000,
        )

    assert raised.value.code == ErrorCode.AUDIO_PROCESS_FAILED


def test_audio_transcoder_writes_audio_artifact_without_video_kernel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime" / "runs" / "run_audio" / "input" / "source.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    runner = Runner()

    artifact = _transcoder(tmp_path, runner).extract_audio(
        source,
        Path("runs/run_audio"),
        duration_ms=1_000,
    )

    assert artifact.relative_path == "runs/run_audio/media/audio.mp3"
    assert artifact.codec == "mp3"
    assert runner.calls
