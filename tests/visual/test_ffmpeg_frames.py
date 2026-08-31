from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.media.process import ProcessResult
from video_demo.visual.candidate_artifacts import CandidateArtifactSession
from video_demo.visual.ffmpeg_frames import FFmpegFrameExtractor, FrameSample


class _Runner:
    def __init__(self, *, fail_timestamps: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_timestamps = fail_timestamps or set()

    def run(
        self,
        args: list[str],
        *,
        timeout_seconds: int,
        output_paths: tuple[Path, ...] = (),
    ) -> ProcessResult:
        del timeout_seconds
        self.calls.append(args)
        timestamp = args[args.index("-ss") + 1]
        if timestamp in self.fail_timestamps:
            return ProcessResult(1, b"", b"decode failed")
        payload = b"\xff\xd8\xff" + timestamp.encode() + b"\xff\xd9"
        output_paths[0].write_bytes(payload)
        return ProcessResult(0, b"", b"")


def test_ffmpeg_extracts_one_frame_per_timestamp_and_publishes_sha_addressed_jpeg(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    run_root = Path("runs/scope/run")
    source = runtime / run_root / "input/source.mp4"
    source.parent.mkdir(parents=True)
    (runtime / run_root / "visual").mkdir()
    source.write_bytes(b"video")
    runner = _Runner()
    extractor = FFmpegFrameExtractor(
        Path("ffmpeg"), runner, runtime, max_frame_bytes=1024
    )
    session = CandidateArtifactSession(
        runtime_root=runtime,
        max_unique_bytes=4096,
        max_files=10,
        max_file_bytes=1024,
    )
    try:
        results = extractor.extract_samples(
            source,
            run_root,
            (FrameSample("a", 100), FrameSample("b", 200)),
            is_cancel_requested=lambda: False,
            artifact_session=session,
        )
    finally:
        session.cleanup_unretained(frozenset())
        session.close()
    assert [call[call.index("-ss") + 1] for call in runner.calls] == ["0.100", "0.200"]
    assert all(
        "-frames:v" in call and call[call.index("-frames:v") + 1] == "1"
        for call in runner.calls
    )
    assert all(item.status == "SUCCEEDED" for item in results)
    for result in results:
        assert result.candidate is not None
        timestamp = f"{result.requested_timestamp_ms / 1000:.3f}"
        digest = hashlib.sha256(b"\xff\xd8\xff" + timestamp.encode() + b"\xff\xd9").hexdigest()
        assert result.candidate.relative_path == Path("visual/candidates") / f"{digest}.jpg"


def test_ffmpeg_single_timestamp_failure_does_not_block_following_samples(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    run_root = Path("runs/scope/run")
    source = runtime / run_root / "input/source.mp4"
    source.parent.mkdir(parents=True)
    (runtime / run_root / "visual").mkdir()
    source.write_bytes(b"video")
    runner = _Runner(fail_timestamps={"0.100"})
    extractor = FFmpegFrameExtractor(Path("ffmpeg"), runner, runtime, max_frame_bytes=1024)
    session = CandidateArtifactSession(
        runtime_root=runtime,
        max_unique_bytes=4096,
        max_files=10,
        max_file_bytes=1024,
    )
    try:
        results = extractor.extract_samples(
            source,
            run_root,
            (FrameSample("a", 100), FrameSample("b", 200)),
            is_cancel_requested=lambda: False,
            artifact_session=session,
        )
    finally:
        session.cleanup_unretained(frozenset())
        session.close()
    assert [item.status for item in results] == ["DECODE_FAILED", "SUCCEEDED"]
