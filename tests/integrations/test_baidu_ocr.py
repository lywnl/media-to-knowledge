from __future__ import annotations

import base64
import json
import math
import traceback
from pathlib import Path

import httpx
import pytest

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.baidu_ocr import BaiduOcrClient, BaiduOcrCredentials
from video_demo.visual.ocr import OcrDeadlineExceeded


@pytest.mark.parametrize(
    ("language", "provider_value"),
    [("zh", "CHN_ENG"), ("en", "ENG"), ("ja", "JAP"), ("ko", "KOR"), ("es", "SPA")],
)
def test_baidu_ocr_maps_validation_languages(language: str, provider_value: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(
            200,
            json={
                "log_id": 12345,
                "words_result": [
                    {
                        "words": "你好",
                        "location": {"left": 1, "top": 2, "width": 3, "height": 4},
                        "probability": {"average": 0.98},
                    },
                ],
            },
        )

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    result = client.recognize(b"image-bytes", language)

    assert result.request_id == "12345"
    assert result.http_status == 200
    assert result.lines[0].text == "你好"
    assert result.lines[0].confidence == 0.98
    ocr_request = requests[1]
    body = ocr_request.content.decode()
    assert f"language_type={provider_value}" in body
    assert base64.b64encode(b"image-bytes").decode().replace("=", "%3D") in body


def test_oauth_token_is_cached_until_expiry() -> None:
    now = [100.0]
    token_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests
        if request.url.path.endswith("/oauth/2.0/token"):
            token_requests += 1
            return httpx.Response(
                200,
                json={"access_token": f"token-{token_requests}", "expires_in": 100},
            )
        return httpx.Response(200, json={"log_id": token_requests, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    client.recognize(b"one", "zh")
    client.recognize(b"two", "zh")
    now[0] = 191.0
    client.recognize(b"three", "zh")

    assert token_requests == 2


@pytest.mark.parametrize("expires_in", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_oauth_token_expiry_must_be_finite_and_positive_without_secret_leak(
    expires_in: float,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/oauth/2.0/token")
        literal = (
            "NaN"
            if math.isnan(expires_in)
            else "Infinity"
            if expires_in == float("inf")
            else "-Infinity"
            if expires_in == float("-inf")
            else str(expires_in)
        )
        return httpx.Response(
            200,
            content=(
                '{"access_token":"provider-secret","expires_in":' + literal + "}"
            ).encode(),
            headers={"content-type": "application/json"},
        )

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.OCR_RESPONSE_INVALID
    assert raised.value.__cause__ is None
    rendered = "".join(traceback.format_exception(raised.value))
    assert "provider-secret" not in rendered
    assert "api-secret" not in rendered
    assert "secret-value" not in rendered


def test_oauth_token_huge_integer_expiry_is_a_sanitized_provider_error() -> None:
    ocr_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ocr_calls
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(
                200,
                content=(
                    '{"access_token":"provider-secret","expires_in":'
                    + "1"
                    + "0" * 1_000
                    + "}"
                ).encode(),
                headers={"content-type": "application/json"},
            )
        ocr_calls += 1
        return httpx.Response(200, json={"log_id": 1, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.OCR_RESPONSE_INVALID
    assert raised.value.__cause__ is None
    assert ocr_calls == 0
    rendered = "".join(traceback.format_exception(raised.value))
    assert "provider-secret" not in rendered
    assert "api-secret" not in rendered
    assert "secret-value" not in rendered


def test_unsupported_language_fails_before_network_call() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "fr")

    assert raised.value.code == ErrorCode.OCR_LANGUAGE_UNSUPPORTED
    assert called is False


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (401, ErrorCode.OCR_AUTHENTICATION_FAILED),
        (429, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        (500, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
    ],
)
def test_http_errors_are_classified_without_secret_leak(
    status_code: int,
    error_code: ErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(status_code, text="api-secret secret-value token-safe")

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == error_code
    serialized = f"{raised.value.message} {raised.value.details}"
    assert "api-secret" not in serialized
    assert "secret-value" not in serialized
    assert "token-safe" not in serialized


def test_malformed_provider_json_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(200, json={"log_id": 1, "words_result": [{"words": "missing bbox"}]})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.OCR_RESPONSE_INVALID


@pytest.mark.parametrize(
    "payload",
    [
        {"log_id": "", "words_result": []},
        {"log_id": "x" * 257, "words_result": []},
        {
            "log_id": "request-safe",
            "words_result": [
                {
                    "words": {"secret": "provider-secret"},
                    "location": {"left": 1, "top": 2, "width": 3, "height": 4},
                    "probability": {"average": 0.9},
                },
            ],
        },
        {
            "log_id": "request-safe",
            "words_result": [
                {
                    "words": "provider-secret",
                    "location": {"left": 1.5, "top": 2, "width": 3, "height": 4},
                    "probability": {"average": 0.9},
                },
            ],
        },
        {
            "log_id": "request-safe",
            "words_result": [
                {
                    "words": "provider-secret",
                    "location": {"left": 1, "top": 2, "width": 3, "height": 4},
                    "probability": {"average": math.nan},
                },
            ],
        },
        {
            "log_id": "request-safe",
            "words_result": [
                {
                    "words": "provider-secret",
                    "location": {"left": 1, "top": 2, "width": 3, "height": 4},
                    "probability": {"average": math.inf},
                },
            ],
        },
    ],
)
def test_ocr_payload_fields_are_strict_and_traceback_is_redacted(
    payload: dict[str, object],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        results = payload.get("words_result")
        if isinstance(results, list) and results and isinstance(results[0], dict):
            probability = results[0].get("probability")
            if isinstance(probability, dict):
                average = probability.get("average")
                if isinstance(average, float) and not math.isfinite(average):
                    literal = b"NaN" if math.isnan(average) else b"Infinity"
                    return httpx.Response(
                        200,
                        content=(
                            b'{"log_id":"request-safe","words_result":[{"words":'
                            b'"provider-secret","location":{"left":1,"top":2,"width":3,'
                            b'"height":4},"probability":{"average":'
                            + literal
                            + b"}}]}"
                        ),
                        headers={"Content-Type": "application/json"},
                    )
        return httpx.Response(200, json=payload)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.code == ErrorCode.OCR_RESPONSE_INVALID
    assert raised.value.__cause__ is None
    assert "provider-secret" not in rendered


@pytest.mark.parametrize(
    ("provider_error_code", "expected"),
    [
        (110, ErrorCode.OCR_AUTHENTICATION_FAILED),
        (18, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        (17, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        (216201, ErrorCode.OCR_RESPONSE_INVALID),
    ],
)
def test_http_200_provider_errors_are_classified(
    provider_error_code: int,
    expected: ErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(
            200,
            json={"error_code": provider_error_code, "error_msg": "secret provider detail"},
        )

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        max_attempts=1,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == expected
    assert "secret provider detail" not in f"{raised.value.message} {raised.value.details}"


def test_transient_ocr_failures_are_retried_with_bounded_backoff() -> None:
    ocr_attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ocr_attempts
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        ocr_attempts += 1
        if ocr_attempts == 1:
            return httpx.Response(429)
        if ocr_attempts == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"log_id": 3, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        max_attempts=3,
        retry_backoff_seconds=0.25,
        sleeper=delays.append,
    )

    response = client.recognize(b"image", "zh")

    assert response.request_id == "3"
    assert ocr_attempts == 3
    assert delays == [0.25, 0.5]


def test_ocr_timeout_stops_after_finite_attempts_and_is_classified() -> None:
    ocr_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ocr_attempts
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        ocr_attempts += 1
        raise httpx.ReadTimeout("timeout", request=request)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        max_attempts=2,
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert ocr_attempts == 2
    assert raised.value.__cause__ is None
    assert "httpx.ReadTimeout" not in "".join(traceback.format_exception(raised.value))


def test_token_and_ocr_requests_share_remaining_global_deadline() -> None:
    now = [100.0]
    request_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions["timeout"]["read"]
        request_timeouts.append(timeout)
        if request.url.path.endswith("/oauth/2.0/token"):
            now[0] = 103.0
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(200, json={"log_id": 1, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
        timeout_seconds=30.0,
    )

    client.recognize(b"image", "zh", deadline=110.0)

    assert request_timeouts == [10.0, 7.0]


def test_retry_backoff_cannot_cross_global_deadline() -> None:
    now = [10.0]
    sleeps: list[float] = []
    ocr_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ocr_attempts
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        ocr_attempts += 1
        return httpx.Response(503)

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
        max_attempts=3,
        retry_backoff_seconds=2.0,
        sleeper=sleep,
    )

    with pytest.raises(OcrDeadlineExceeded) as raised:
        client.recognize(b"image", "zh", deadline=11.0)

    assert ocr_attempts == 1
    assert sleeps == []
    assert raised.value.provider_attempt_count == 1


@pytest.mark.parametrize("during_token_request", [True, False])
def test_network_timeout_that_consumes_remaining_budget_becomes_deadline_stop(
    during_token_request: bool,
) -> None:
    now = [20.0]

    def handler(request: httpx.Request) -> httpx.Response:
        if not during_token_request and request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        now[0] = 21.0
        raise httpx.ReadTimeout("timeout", request=request)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
        max_attempts=1,
    )

    with pytest.raises(OcrDeadlineExceeded):
        client.recognize(b"image", "zh", deadline=21.0)


def test_deadline_stop_reports_ocr_attempts_already_sent() -> None:
    now = [30.0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        now[0] = 31.0
        raise httpx.ReadTimeout("timeout", request=request)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
        max_attempts=3,
    )

    with pytest.raises(OcrDeadlineExceeded) as raised:
        client.recognize(b"image", "zh", deadline=31.0)

    assert raised.value.provider_attempt_count == 1


def test_successful_provider_response_returned_after_deadline_is_discarded() -> None:
    now = [40.0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        now[0] = 41.0
        return httpx.Response(200, json={"log_id": 1, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    with pytest.raises(OcrDeadlineExceeded) as raised:
        client.recognize(b"image", "zh", deadline=41.0)

    assert raised.value.provider_attempt_count == 1


def test_token_response_returned_after_deadline_is_discarded_without_caching() -> None:
    now = [50.0]
    token_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests
        if request.url.path.endswith("/oauth/2.0/token"):
            token_requests += 1
            if token_requests == 1:
                now[0] = 51.0
            return httpx.Response(
                200,
                json={"access_token": "token-safe", "expires_in": 3600},
            )
        return httpx.Response(200, json={"log_id": 1, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    with pytest.raises(OcrDeadlineExceeded):
        client.recognize(b"image", "zh", deadline=51.0)

    now[0] = 52.0
    client.recognize(b"image", "zh")

    assert token_requests == 2


def test_provider_error_response_returned_after_deadline_is_discarded() -> None:
    now = [60.0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(
                200,
                json={"access_token": "token-safe", "expires_in": 3600},
            )
        now[0] = 61.0
        return httpx.Response(401)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    with pytest.raises(OcrDeadlineExceeded) as raised:
        client.recognize(b"image", "zh", deadline=61.0)

    assert raised.value.provider_attempt_count == 1


def test_invalid_ocr_payload_parsed_after_deadline_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.integrations.baidu_ocr as baidu_ocr_module

    now = [70.0]
    original_json_object = baidu_ocr_module._json_object

    def parse_then_reach_deadline(response: httpx.Response) -> dict[str, object]:
        payload = original_json_object(response)
        if response.request.method == "POST":
            now[0] = 71.0
        return payload

    monkeypatch.setattr(baidu_ocr_module, "_json_object", parse_then_reach_deadline)
    client = BaiduOcrClient(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    httpx.Response(
                        200,
                        json={"access_token": "token-safe", "expires_in": 3600},
                    )
                    if request.url.path.endswith("/oauth/2.0/token")
                    else httpx.Response(200, json={"words_result": []})
                ),
            ),
        ),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    with pytest.raises(OcrDeadlineExceeded) as raised:
        client.recognize(b"image", "zh", deadline=71.0)

    assert raised.value.provider_attempt_count == 1


def test_token_parsed_after_deadline_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import video_demo.integrations.baidu_ocr as baidu_ocr_module

    now = [80.0]
    token_requests = 0
    original_json_object = baidu_ocr_module._json_object

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_requests
        if request.url.path.endswith("/oauth/2.0/token"):
            token_requests += 1
            return httpx.Response(
                200,
                json={"access_token": "token-safe", "expires_in": 3600},
            )
        return httpx.Response(200, json={"log_id": 1, "words_result": []})

    def parse_then_reach_deadline(response: httpx.Response) -> dict[str, object]:
        payload = original_json_object(response)
        if response.request.method == "GET" and token_requests == 1:
            now[0] = 81.0
        return payload

    monkeypatch.setattr(baidu_ocr_module, "_json_object", parse_then_reach_deadline)
    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        clock=lambda: now[0],
    )

    with pytest.raises(OcrDeadlineExceeded):
        client.recognize(b"image", "zh", deadline=81.0)

    now[0] = 82.0
    client.recognize(b"image", "zh")

    assert token_requests == 2


def test_provider_attempt_count_includes_retries_but_not_token_request() -> None:
    ocr_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal ocr_attempts
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        ocr_attempts += 1
        if ocr_attempts < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"log_id": 3, "words_result": []})

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )

    response = client.recognize(b"image", "zh")

    assert response.provider_attempt_count == 3


def test_token_timeout_does_not_retain_third_party_cause_or_secret() -> None:
    secret = "token-secret-in-transport"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("token_request", [True, False])
def test_network_error_is_temporary_and_does_not_retain_third_party_cause(
    token_request: bool,
) -> None:
    secret = "network-secret-in-transport"

    def handler(request: httpx.Request) -> httpx.Response:
        if not token_request and request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        raise httpx.ConnectError(secret, request=request)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
        max_attempts=1,
    )

    with pytest.raises(VideoDemoError) as raised:
        client.recognize(b"image", "zh")

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert raised.value.__cause__ is None
    assert secret not in "".join(traceback.format_exception(raised.value))


def test_redacted_recorded_contract_fixture_is_parsed() -> None:
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "baidu_ocr"
        / "accurate_success.redacted.json"
    )
    fixture_text = fixture_path.read_text(encoding="utf-8")
    payload = json.loads(fixture_text)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/2.0/token"):
            return httpx.Response(200, json={"access_token": "token-safe", "expires_in": 3600})
        return httpx.Response(200, json=payload)

    client = BaiduOcrClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        BaiduOcrCredentials(api_key="api-secret", secret_key="secret-value"),
        endpoint="https://aip.baidubce.com/rest/2.0/ocr/v1/accurate",
    )

    response = client.recognize(b"image", "es")

    assert response.request_id == "987654321"
    assert response.lines[0].text == "[REDACTED]"
    assert response.lines[0].bounding_box.width == 160
    assert response.lines[0].confidence == 0.97
    assert "api-secret" not in fixture_text
    assert "secret-value" not in fixture_text
    assert "token-safe" not in fixture_text
