from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import FFprobeClient, ProbeLimits, parse_ffprobe_payload
from video_demo.media.process import ProcessResult, SafeProcessRunner

FIXTURES = Path(__file__).parent / "fixtures" / "ffprobe"


def _payload(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parse_valid_manifest_and_integer_millisecond_duration() -> None:
    result = parse_ffprobe_payload(
        _payload("valid_mp4.json"),
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert result.manifest.duration_ms == 12_345
    assert result.manifest.video_stream.average_frame_rate.numerator == 30_000
    assert result.manifest.video_stream.average_frame_rate.denominator == 1001
    assert len(result.manifest.audio_streams) == 1
    assert result.warnings == ()


def test_parse_preserves_container_duration_and_uses_video_stream_timeline() -> None:
    payload = _payload("valid_mp4.json")
    payload["format"]["duration"] = "302.366"
    payload["streams"][0]["duration"] = "302.101313"

    result = parse_ffprobe_payload(
        payload,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert result.manifest.duration_ms == 302_366
    assert result.timeline_duration_ms == 302_101


@pytest.mark.parametrize("video_duration", [None, "N/A"])
def test_parse_falls_back_to_container_duration_when_video_duration_is_unavailable(
    video_duration: str | None,
) -> None:
    payload = _payload("valid_mp4.json")
    if video_duration is not None:
        payload["streams"][0]["duration"] = video_duration

    result = parse_ffprobe_payload(
        payload,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert result.timeline_duration_ms == result.manifest.duration_ms == 12_345


@pytest.mark.parametrize(
    "video_duration",
    [None, "0", "-0.001", "NaN", "Infinity", "invalid"],
)
def test_parse_rejects_explicit_invalid_video_stream_duration(
    video_duration: object,
) -> None:
    payload = _payload("valid_mp4.json")
    payload["streams"][0]["duration"] = video_duration

    with pytest.raises(VideoDemoError) as raised:
        parse_ffprobe_payload(
            payload,
            object_ref="obj_001",
            source_sha256="a" * 64,
            source_size_bytes=1024,
            source_mime="video/mp4",
            ffprobe_version="ffprobe version 7.1",
            limits=ProbeLimits(),
        )

    assert raised.value.code == ErrorCode.VIDEO_PROBE_INVALID


@pytest.mark.parametrize(
    ("container_duration", "video_duration"),
    [("7200.001", "7199.000"), ("7200.000", "7200.001")],
)
def test_parse_rejects_when_container_or_video_stream_exceeds_duration_limit(
    container_duration: str,
    video_duration: str,
) -> None:
    payload = _payload("valid_mp4.json")
    payload["format"]["duration"] = container_duration
    payload["streams"][0]["duration"] = video_duration

    with pytest.raises(VideoDemoError) as raised:
        parse_ffprobe_payload(
            payload,
            object_ref="obj_001",
            source_sha256="a" * 64,
            source_size_bytes=1024,
            source_mime="video/mp4",
            ffprobe_version="ffprobe version 7.1",
            limits=ProbeLimits(),
        )

    assert raised.value.code == ErrorCode.VIDEO_DURATION_LIMIT_EXCEEDED


def test_parse_accepts_two_hour_manifest_only_with_explicit_limit() -> None:
    payload = _payload("valid_mp4.json")
    payload["format"]["duration"] = "7200.000"
    payload["streams"][0]["duration"] = "7200.000"

    result = parse_ffprobe_payload(
        payload,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(max_duration_ms=7_200_000),
    )

    assert result.manifest.duration_ms == 7_200_000


def test_probe_limit_rejects_duration_above_domain_hard_limit() -> None:
    with pytest.raises(ValueError, match="视频时长上限"):
        ProbeLimits(max_duration_ms=7_200_001)


def test_parse_preserves_text_and_bitmap_subtitle_metadata() -> None:
    result = parse_ffprobe_payload(
        _payload("embedded_text_subtitles.json"),
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert [stream.model_dump(mode="json") for stream in result.manifest.subtitle_streams] == [
        {
            "index": 2,
            "codec_name": "mov_text",
            "language": "zh",
            "is_default": True,
            "is_forced": False,
        },
        {
            "index": 3,
            "codec_name": "ass",
            "language": "en",
            "is_default": False,
            "is_forced": False,
        },
        {
            "index": 4,
            "codec_name": "hdmv_pgs_subtitle",
            "language": "ja",
            "is_default": False,
            "is_forced": True,
        },
        {
            "index": 5,
            "codec_name": "webvtt",
            "language": "und",
            "is_default": False,
            "is_forced": False,
        },
    ]


@pytest.mark.parametrize(
    ("language_tag", "expected"),
    [
        ("zho", "zh"),
        ("chi", "zh"),
        ("eng", "en"),
        ("jpn", "ja"),
        ("kor", "ko"),
        ("spa", "es"),
        ("fra", "und"),
        (None, "und"),
    ],
)
def test_parse_normalizes_subtitle_language_tags(
    language_tag: str | None,
    expected: str,
) -> None:
    payload = _payload("embedded_text_subtitles.json")
    subtitle = payload["streams"][2]
    if language_tag is None:
        subtitle.pop("tags", None)
    else:
        subtitle["tags"] = {"language": language_tag}

    result = parse_ffprobe_payload(
        payload,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert result.manifest.subtitle_streams[0].language == expected


def test_parse_rejects_too_many_subtitle_streams() -> None:
    payload = _payload("valid_mp4.json")
    payload["streams"].extend(
        {
            "index": index + 2,
            "codec_type": "subtitle",
            "codec_name": "webvtt",
        }
        for index in range(33)
    )

    with pytest.raises(VideoDemoError) as raised:
        parse_ffprobe_payload(
            payload,
            object_ref="obj_001",
            source_sha256="a" * 64,
            source_size_bytes=1024,
            source_mime="video/mp4",
            ffprobe_version="ffprobe version 7.1",
            limits=ProbeLimits(max_subtitle_streams=32),
        )

    assert raised.value.code == ErrorCode.VIDEO_STREAM_COUNT_EXCEEDED


def test_parse_rotation_vfr_and_no_audio_warning() -> None:
    result = parse_ffprobe_payload(
        _payload("rotated_vfr_no_audio.json"),
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=1024,
        source_mime="video/mp4",
        ffprobe_version="ffprobe version 7.1",
        limits=ProbeLimits(),
    )

    assert result.manifest.video_stream.rotation_degrees == 90
    assert result.manifest.video_stream.is_variable_frame_rate is True
    assert result.manifest.video_stream.width == 1080
    assert result.manifest.video_stream.height == 1920
    assert result.warnings == ("NO_AUDIO_TRACK",)


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda payload: payload["streams"].clear(), ErrorCode.VIDEO_STREAM_MISSING),
        (
            lambda payload: payload["format"].update({"duration": "7200.001"}),
            ErrorCode.VIDEO_DURATION_LIMIT_EXCEEDED,
        ),
        (
            lambda payload: payload["streams"][0].update({"width": 3840}),
            ErrorCode.VIDEO_RESOLUTION_LIMIT_EXCEEDED,
        ),
        (
            lambda payload: payload["streams"][0].update({"avg_frame_rate": "61/1"}),
            ErrorCode.VIDEO_FRAME_RATE_LIMIT_EXCEEDED,
        ),
    ],
)
def test_parse_rejects_invalid_or_over_limit_streams(
    mutation: object,
    error_code: ErrorCode,
) -> None:
    payload = _payload("valid_mp4.json")
    mutation(payload)

    with pytest.raises(VideoDemoError) as raised:
        parse_ffprobe_payload(
            payload,
            object_ref="obj_001",
            source_sha256="a" * 64,
            source_size_bytes=1024,
            source_mime="video/mp4",
            ffprobe_version="ffprobe version 7.1",
            limits=ProbeLimits(),
        )

    assert raised.value.code == error_code


def test_parse_rejects_multiple_video_streams() -> None:
    payload = _payload("valid_mp4.json")
    payload["streams"].append(dict(payload["streams"][0]))

    with pytest.raises(VideoDemoError) as raised:
        parse_ffprobe_payload(
            payload,
            object_ref="obj_001",
            source_sha256="a" * 64,
            source_size_bytes=1024,
            source_mime="video/mp4",
            ffprobe_version="ffprobe version 7.1",
            limits=ProbeLimits(max_video_streams=1),
        )

    assert raised.value.code == ErrorCode.VIDEO_STREAM_COUNT_EXCEEDED


def test_safe_process_runner_uses_argument_array_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_popen = subprocess.Popen

    def tracked_popen(args: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        captured["args"] = args
        captured.update(kwargs)
        return original_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", tracked_popen)

    result = SafeProcessRunner(max_output_bytes=1024).run(
        [sys.executable, "-c", "import sys; print('{}'); print('warn', file=sys.stderr)"],
        timeout_seconds=5,
    )

    assert result.returncode == 0
    assert captured["args"][0] == sys.executable
    assert captured["shell"] is False
    assert captured["start_new_session"] is True
    assert result.stdout == b"{}\n"
    assert result.stderr == b"warn\n"


def test_safe_process_runner_maps_timeout() -> None:
    with pytest.raises(VideoDemoError) as raised:
        SafeProcessRunner(max_output_bytes=4).run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout_seconds=1,
        )
    assert raised.value.code == ErrorCode.VIDEO_PROCESS_TIMEOUT


def test_safe_process_runner_cancels_while_process_is_running() -> None:
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    started_at = time.monotonic()
    timer.start()
    try:
        with pytest.raises(VideoDemoError) as raised:
            SafeProcessRunner(
                max_output_bytes=1024,
                is_cancel_requested=cancelled.is_set,
            ).run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout_seconds=10,
            )
    finally:
        timer.cancel()

    assert raised.value.code == ErrorCode.VIDEO_PROCESS_CANCELLED
    assert time.monotonic() - started_at < 2


def test_safe_process_runner_stops_process_when_output_limit_is_crossed() -> None:
    started_at = time.monotonic()

    with pytest.raises(VideoDemoError) as too_large:
        SafeProcessRunner(max_output_bytes=64).run(
            [
                sys.executable,
                "-c",
                "import os,time; os.write(1,b'x'*8192); time.sleep(60)",
            ],
            timeout_seconds=10,
        )

    assert too_large.value.code == ErrorCode.VIDEO_PROCESS_OUTPUT_TOO_LARGE
    assert time.monotonic() - started_at < 2


@pytest.mark.skipif(os.name != "posix", reason="进程组终止契约仅适用于 POSIX")
def test_safe_process_runner_cancellation_stops_descendant_processes(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant-survived"
    descendant = (
        "import pathlib,time; "
        "time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('alive', encoding='utf-8')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        "time.sleep(60)"
    )
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    try:
        with pytest.raises(VideoDemoError) as raised:
            SafeProcessRunner(
                max_output_bytes=1024,
                is_cancel_requested=cancelled.is_set,
            ).run([sys.executable, "-c", parent], timeout_seconds=10)
    finally:
        timer.cancel()

    time.sleep(0.7)
    assert raised.value.code == ErrorCode.VIDEO_PROCESS_CANCELLED
    assert not marker.exists()


def test_safe_process_runner_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        SafeProcessRunner(max_output_bytes=0)
    with pytest.raises(VideoDemoError) as invalid_timeout:
        SafeProcessRunner().run(
            [sys.executable, "-c", "pass"],
            timeout_seconds=0,
        )
    assert invalid_timeout.value.code == ErrorCode.VIDEO_PROCESS_FAILED


def test_safe_process_runner_maps_start_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(VideoDemoError) as failed:
        SafeProcessRunner().run(
            [str(tmp_path / "missing-binary")],
            timeout_seconds=1,
        )

    assert failed.value.code == ErrorCode.VIDEO_PROCESS_FAILED


def test_ffprobe_client_builds_safe_command_and_parses_json(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class FakeRunner:
        def run(self, args: list[str], *, timeout_seconds: int) -> ProcessResult:
            calls.append(args)
            return ProcessResult(
                returncode=0,
                stdout=json.dumps(_payload("valid_mp4.json")).encode(),
                stderr=b"",
            )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    client = FFprobeClient(Path("/tools/ffprobe"), FakeRunner(), "ffprobe version 7.1")
    result = client.probe(
        source,
        object_ref="obj_001",
        source_sha256="a" * 64,
        source_size_bytes=5,
        source_mime="video/mp4",
        limits=ProbeLimits(),
    )

    assert result.manifest.duration_ms == 12_345
    assert calls == [
        [
            "/tools/ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ],
    ]


def test_ffprobe_client_rejects_corrupted_json_output(tmp_path: Path) -> None:
    class CorruptedRunner:
        def run(self, args: list[str], *, timeout_seconds: int) -> ProcessResult:
            return ProcessResult(returncode=0, stdout=b"{broken", stderr=b"")

    source = tmp_path / "corrupted.mp4"
    source.write_bytes(b"not-a-real-video")
    client = FFprobeClient(Path("/tools/ffprobe"), CorruptedRunner(), "ffprobe version 7.1")

    with pytest.raises(VideoDemoError) as raised:
        client.probe(
            source,
            object_ref="obj_corrupted",
            source_sha256="a" * 64,
            source_size_bytes=source.stat().st_size,
            source_mime="video/mp4",
            limits=ProbeLimits(),
        )

    assert raised.value.code == ErrorCode.VIDEO_PROBE_INVALID


def test_ffprobe_client_rejects_corrupted_media_decode_failure(tmp_path: Path) -> None:
    class DecodeFailureRunner:
        def run(self, args: list[str], *, timeout_seconds: int) -> ProcessResult:
            return ProcessResult(returncode=1, stdout=b"", stderr=b"invalid data")

    source = tmp_path / "corrupted.mp4"
    source.write_bytes(b"not-a-real-video")
    client = FFprobeClient(Path("/tools/ffprobe"), DecodeFailureRunner(), "ffprobe version 7.1")

    with pytest.raises(VideoDemoError) as raised:
        client.probe(
            source,
            object_ref="obj_corrupted",
            source_sha256="a" * 64,
            source_size_bytes=source.stat().st_size,
            source_mime="video/mp4",
            limits=ProbeLimits(),
        )

    assert raised.value.code == ErrorCode.VIDEO_PROCESS_FAILED


def test_missing_real_ffprobe_is_reported_explicitly(tmp_path: Path) -> None:
    with pytest.raises(VideoDemoError) as raised:
        FFprobeClient.from_path(
            tmp_path / "missing-ffprobe",
            workspace_root=tmp_path,
        )

    assert raised.value.code == ErrorCode.VIDEO_FFPROBE_UNAVAILABLE


def test_ffprobe_constructor_rejects_parent_symlink_before_execution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = workspace / ".codex" / "video-rag-demo"
    runtime.mkdir(parents=True)
    external_tools = tmp_path / "external-tools"
    external_tools.mkdir()
    executable = external_tools / "ffprobe"
    executable.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    executable.chmod(0o755)
    (runtime / "tools").symlink_to(external_tools, target_is_directory=True)

    with pytest.raises(VideoDemoError) as raised:
        FFprobeClient.from_path(
            runtime / "tools" / "ffprobe",
            workspace_root=workspace,
        )

    assert raised.value.code == ErrorCode.VIDEO_FFPROBE_UNAVAILABLE
