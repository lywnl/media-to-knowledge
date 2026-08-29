from __future__ import annotations

import json

import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.audio_probe import parse_audio_probe_payload
from video_demo.media.process import ProcessResult


def _result(duration: str = "12.5") -> ProcessResult:
    payload = {
        "format": {"duration": duration, "format_name": "wav"},
        "streams": [{"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}],
    }
    return ProcessResult(returncode=0, stdout=json.dumps(payload).encode(), stderr=b"")


def test_parse_audio_probe_payload_returns_duration_and_stream_properties() -> None:
    result = parse_audio_probe_payload(_result(), max_duration_ms=7_200_000)
    assert result.duration_ms == 12_500
    assert result.sample_rate_hz == 16_000
    assert result.channels == 1
    assert result.codec_name == "pcm_s16le"


def test_parse_audio_probe_payload_rejects_duration_over_limit() -> None:
    with pytest.raises(VideoDemoError) as raised:
        parse_audio_probe_payload(_result("7200.001"), max_duration_ms=7_200_000)
    assert raised.value.code == ErrorCode.AUDIO_DURATION_LIMIT_EXCEEDED


def test_parse_audio_probe_payload_rejects_missing_audio_stream() -> None:
    payload = json.loads(_result().stdout)
    payload["streams"] = []
    with pytest.raises(VideoDemoError) as raised:
        parse_audio_probe_payload(
            ProcessResult(returncode=0, stdout=json.dumps(payload).encode(), stderr=b""),
            max_duration_ms=7_200_000,
        )
    assert raised.value.code == ErrorCode.AUDIO_PROBE_INVALID
