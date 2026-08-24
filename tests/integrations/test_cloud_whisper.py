from __future__ import annotations

import math
import traceback
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from video_demo.config import CloudAsrConfiguration
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.cloud_whisper import CloudWhisperClient

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _configuration(
    *,
    max_attempts: int = 3,
    timeout_seconds: float = 300.0,
) -> CloudAsrConfiguration:
    return CloudAsrConfiguration(
        base_url="https://ai-proxy.example.test/v1",
        api_key="test-openai-key",
        model="openai/whisper",
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        max_window_ms=600_000,
        overlap_ms=1_000,
    )


def _audio(tmp_path: Path, content: bytes = b"RIFF-test-wav") -> Path:
    path = tmp_path / "runs/scope/run_001/speech/slices/window.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _client(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = lambda _delay: None,
) -> tuple[CloudWhisperClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return (
        CloudWhisperClient(
            http_client,
            _configuration(max_attempts=max_attempts),
            allowed_audio_root=tmp_path,
            sleeper=sleeper,
        ),
        http_client,
    )


def _assert_text_part(body: bytes, name: str, value: str) -> None:
    marker = f'name="{name}"'.encode("ascii")
    assert marker in body
    assert f"\r\n\r\n{value}\r\n".encode() in body


def _valid_payload(
    *,
    language: str = "english",
    text: str = " Hello world ",
) -> dict[str, object]:
    return {
        "task": "transcribe",
        "language": language,
        "duration": 1.235,
        "text": text,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.235,
                "text": text,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.1,
            }
        ],
    }


def test_client_posts_verbose_json_multipart_and_converts_segments(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = request.read()
        assert request.method == "POST"
        assert request.url == "https://ai-proxy.example.test/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer test-openai-key"
        assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
        _assert_text_part(body, "model", "openai/whisper")
        _assert_text_part(body, "response_format", "verbose_json")
        _assert_text_part(body, "temperature", "0")
        _assert_text_part(body, "language", "en")
        _assert_text_part(body, "prompt", "产品术语\n核心背景")
        assert b'name="file"; filename="window.wav"' in body
        assert b"Content-Type: audio/wav" in body
        assert audio.read_bytes() in body
        return httpx.Response(200, json=_valid_payload(), request=request)

    client, http_client = _client(tmp_path, handler)

    result = client.transcribe_window(
        audio,
        language_hint="en",
        prompt="产品术语\n核心背景",
    )

    assert result.language == "en"
    assert result.warnings == ()
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert (segment.start_ms, segment.end_ms, segment.text) == (0, 1_235, "Hello world")
    assert segment.confidence == pytest.approx(math.exp(-0.2) * 0.9)
    assert len(requests) == 1
    assert http_client.is_closed is False


@pytest.mark.parametrize("language_hint", (None, "und", "English", "en-US", ""))
def test_client_omits_invalid_or_non_specific_language_hint(
    tmp_path: Path,
    language_hint: str | None,
) -> None:
    audio = _audio(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'name="language"' not in body
        assert b'name="prompt"' not in body
        return httpx.Response(200, json=_valid_payload(), request=request)

    client, _http_client = _client(tmp_path, handler)

    client.transcribe_window(audio, language_hint=language_hint, prompt=None)


@pytest.mark.parametrize(
    ("provider_language", "expected"),
    (
        ("chinese", "zh"),
        ("english", "en"),
        ("japanese", "ja"),
        ("korean", "ko"),
        ("spanish", "es"),
        ("french", "fr"),
        ("german", "de"),
        ("arabic", "ar"),
        ("hindi", "hi"),
        ("mandarin", "zh"),
        ("letzeburgesch", "lb"),
        ("pt", "pt"),
        ("yue", "yue"),
    ),
)
def test_client_normalizes_provider_language(
    tmp_path: Path,
    provider_language: str,
    expected: str,
) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json=_valid_payload(language=provider_language),
            request=request,
        ),
    )

    result = client.transcribe_window(audio, language_hint=None, prompt=None)

    assert result.language == expected
    assert result.warnings == ()


def test_client_marks_unknown_provider_language_without_guessing(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json=_valid_payload(language="klingon"),
            request=request,
        ),
    )

    result = client.transcribe_window(audio, language_hint=None, prompt=None)

    assert result.language == "und"
    assert result.warnings == ("CLOUD_ASR_LANGUAGE_UNRECOGNIZED",)


def test_client_accepts_a_truly_empty_window_without_fabricating_segments(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json={"language": "english", "text": "", "segments": []},
            request=request,
        ),
    )

    result = client.transcribe_window(audio, language_hint=None, prompt=None)

    assert result.language == "en"
    assert result.segments == ()


@pytest.mark.parametrize(
    "payload",
    (
        {"language": "english", "text": "non-empty"},
        {"language": "english", "text": "non-empty", "segments": []},
        {"language": "english", "text": 123, "segments": []},
        {"language": 123, "text": "", "segments": []},
        {"language": "english", "text": "x", "segments": [{}]},
        {
            "language": "english",
            "text": "x",
            "segments": [{"start": True, "end": 1.0, "text": "x"}],
        },
        {
            "language": "english",
            "text": "x",
            "segments": [{"start": 2.0, "end": 1.0, "text": "x"}],
        },
        {
            "language": "english",
            "text": "x",
            "segments": [{"start": 0.0, "end": 1.0, "text": 123}],
        },
        {
            "language": "english",
            "text": "x",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "x",
                    "avg_logprob": True,
                    "no_speech_prob": 0.1,
                }
            ],
        },
        {
            "language": "english",
            "text": "",
            "segments": [{"text": " "}],
        },
        {
            "language": "english",
            "text": "x",
            "segments": [
                {
                    "start": 0.0001,
                    "end": 0.0004,
                    "text": "x",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.1,
                }
            ],
        },
        {
            "language": "english",
            "text": "x",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "x",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 1.1,
                }
            ],
        },
    ),
)
def test_client_rejects_invalid_verbose_json_structure(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(200, json=payload, request=request),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE


@pytest.mark.parametrize(
    "content",
    (
        b'{"language":"english","text":"x","segments":[{"start":NaN,"end":1,"text":"x","avg_logprob":-0.1,"no_speech_prob":0.1}]}',
        b'{"language":"english","text":"x","segments":[{"start":0,"end":Infinity,"text":"x","avg_logprob":-0.1,"no_speech_prob":0.1}]}',
        b'{not-json',
        b'\xff\xfe\xfd',
    ),
)
def test_client_rejects_non_finite_or_invalid_json(
    tmp_path: Path,
    content: bytes,
) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            content=content,
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE


def test_client_rejects_response_larger_than_sixteen_mib(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    oversized = b"{" + b" " * _MAX_RESPONSE_BYTES + b"}"
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(200, content=oversized, request=request),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE


@pytest.mark.parametrize(
    ("status_code", "expected", "expected_attempts"),
    (
        (408, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 3),
        (429, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 3),
        (500, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 3),
        (503, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, 3),
        (401, ErrorCode.SPEECH_AUTHENTICATION_FAILED, 1),
        (403, ErrorCode.SPEECH_AUTHENTICATION_FAILED, 1),
        (413, ErrorCode.SPEECH_AUDIO_INVALID, 1),
        (400, ErrorCode.SPEECH_MODEL_UNAVAILABLE, 1),
        (404, ErrorCode.SPEECH_MODEL_UNAVAILABLE, 1),
    ),
)
def test_client_classifies_http_errors_and_retries_only_temporary_failures(
    tmp_path: Path,
    status_code: int,
    expected: ErrorCode,
    expected_attempts: int,
) -> None:
    audio = _audio(tmp_path)
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code,
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "provider-secret-detail",
                }
            },
            request=request,
        )

    client, _http_client = _client(tmp_path, handler, sleeper=sleeps.append)
    prompt = "private prompt"

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=prompt)

    assert raised.value.code == expected
    assert len(requests) == expected_attempts
    assert sleeps == ([1, 2] if expected_attempts == 3 else [])
    assert raised.value.details["status_code"] == status_code
    assert raised.value.details["provider_error_code"] == "rate_limit_exceeded"
    rendered = "".join(traceback.format_exception(raised.value))
    assert "test-openai-key" not in rendered
    assert "provider-secret-detail" not in rendered
    assert "private prompt" not in rendered


def test_client_retries_proxy_no_available_model_branch_without_leaking_message(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    responses = (
        httpx.Response(
            400,
            json={
                "message": (
                    "Model is invalid: no available branch for policy group: "
                    "openai/whisper"
                ),
            },
        ),
        httpx.Response(200, json=_valid_payload()),
    )
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        response = responses[attempts]
        attempts += 1
        response.request = request
        return response

    client, _http_client = _client(tmp_path, handler, sleeper=sleeps.append)

    result = client.transcribe_window(audio, language_hint=None, prompt=None)

    assert result.segments[0].text == "Hello world"
    assert attempts == 2
    assert sleeps == [60]


def test_client_does_not_retry_other_top_level_400_messages(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            400,
            json={
                "message": (
                    "Model is invalid: no available branch for policy group: "
                    "another/model"
                ),
            },
            request=request,
        )

    client, _http_client = _client(tmp_path, handler)

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_MODEL_UNAVAILABLE
    assert attempts == 1
    assert raised.value.details == {"status_code": 400}


def test_client_hides_proxy_message_after_model_branch_retries_are_exhausted(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    provider_message = (
        "Model is invalid: no available branch for policy group: openai/whisper"
    )
    sleeps: list[float] = []
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            400,
            json={"message": provider_message},
            request=request,
        ),
        sleeper=sleeps.append,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert raised.value.details == {"status_code": 400}
    assert sleeps == [60, 60]
    assert provider_message not in "".join(traceback.format_exception(raised.value))


def test_client_retries_transport_errors_and_reopens_file_from_start(
    tmp_path: Path,
) -> None:
    wav = b"RIFF-reopened-on-every-attempt"
    audio = _audio(tmp_path, wav)
    bodies: list[bytes] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        if len(bodies) == 1:
            raise httpx.ReadTimeout("provider-secret", request=request)
        if len(bodies) == 2:
            return httpx.Response(500, request=request)
        return httpx.Response(200, json=_valid_payload(), request=request)

    client, _http_client = _client(tmp_path, handler, sleeper=sleeps.append)

    result = client.transcribe_window(audio, language_hint=None, prompt=None)

    assert result.segments[0].text == "Hello world"
    assert len(bodies) == 3
    assert all(wav in body for body in bodies)
    assert sleeps == [1, 2]


def test_client_does_not_sleep_after_last_failed_attempt(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client, _http_client = _client(
        tmp_path,
        handler,
        max_attempts=1,
        sleeper=sleeps.append,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert sleeps == []


def test_client_only_exposes_whitelisted_provider_error_code(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(
            400,
            json={"error": {"code": "unsafe code/private", "message": "secret"}},
            request=request,
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.details == {"status_code": 400}


def test_client_rejects_oversized_upload_before_network(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    with audio.open("r+b") as stream:
        stream.truncate(_MAX_UPLOAD_BYTES + 1)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_valid_payload(), request=request)

    client, _http_client = _client(tmp_path, handler)

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_AUDIO_INVALID
    assert called is False


@pytest.mark.parametrize("content", (b"",))
def test_client_rejects_empty_upload_before_network(tmp_path: Path, content: bytes) -> None:
    audio = _audio(tmp_path, content)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_valid_payload(), request=request)

    client, _http_client = _client(tmp_path, handler)

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(audio, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.SPEECH_AUDIO_INVALID
    assert called is False


def test_client_rejects_audio_outside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF-outside")
    client = CloudWhisperClient(
        httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200))),
        _configuration(),
        allowed_audio_root=allowed,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(outside, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE


def test_client_rejects_symlink_audio(tmp_path: Path) -> None:
    target = _audio(tmp_path)
    link = target.parent / "linked.wav"
    link.symlink_to(target)
    client, _http_client = _client(
        tmp_path,
        lambda request: httpx.Response(200, json=_valid_payload(), request=request),
    )

    with pytest.raises(VideoDemoError) as raised:
        client.transcribe_window(link, language_hint=None, prompt=None)

    assert raised.value.code == ErrorCode.WORKSPACE_PATH_ESCAPE
