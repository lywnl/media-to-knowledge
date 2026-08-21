from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from pydantic import SecretStr

from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.oss import OssTemporaryVideoPublisher
from video_demo.integrations.video_port import VideoClipInput

_ACCESS_KEY_ID = "test-access-key-id"
_ACCESS_KEY_SECRET = "test-access-key-secret"
_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"media"


def _clip(tmp_path: Path) -> VideoClipInput:
    path = tmp_path / "runs" / "scope" / "run_001" / "visual" / "clips" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(_MP4)
    return VideoClipInput(
        clip_id="clip_001",
        start_ms=1_000,
        end_ms=6_000,
        path=path,
        mime_type="video/mp4",
        sha256=hashlib.sha256(_MP4).hexdigest(),
    )


def _publisher(
    tmp_path: Path,
    handler: httpx.MockTransport,
) -> OssTemporaryVideoPublisher:
    return OssTemporaryVideoPublisher(
        httpx.Client(transport=handler),
        endpoint="https://oss-cn-hangzhou.aliyuncs.com",
        bucket="private-video-bucket",
        access_key_id=SecretStr(_ACCESS_KEY_ID),
        access_key_secret=SecretStr(_ACCESS_KEY_SECRET),
        allowed_video_root=tmp_path,
        prefix="video-demo/qwen-clips",
        signed_url_ttl_seconds=3_600,
        clock=lambda: 1_800_000_000.0,
    )


def test_publisher_uploads_private_object_verifies_it_and_returns_signed_url(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(404, request=request)
        if len(requests) == 2:
            assert request.method == "PUT"
            assert request.read() == _MP4
            return httpx.Response(200, request=request)
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(_MP4)),
                "x-oss-meta-sha256": hashlib.sha256(_MP4).hexdigest(),
            },
            request=request,
        )

    publisher = _publisher(tmp_path, httpx.MockTransport(handler))

    published = publisher.publish(_clip(tmp_path))

    assert [request.method for request in requests] == ["HEAD", "PUT", "HEAD"]
    assert all(request.url.host == publisher.remote_host for request in requests)
    assert requests[1].headers["content-type"] == "video/mp4"
    assert requests[1].headers["x-oss-meta-sha256"] == hashlib.sha256(_MP4).hexdigest()
    assert requests[1].headers["content-md5"]
    assert all(_ACCESS_KEY_SECRET not in str(request.headers) for request in requests)
    assert published.path is None
    assert published.sha256 == hashlib.sha256(_MP4).hexdigest()
    assert (published.start_ms, published.end_ms) == (1_000, 6_000)
    parsed = urlsplit(published.source_url or "")
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.hostname == publisher.remote_host
    assert query["OSSAccessKeyId"] == [_ACCESS_KEY_ID]
    assert query["Expires"] == ["1800003600"]
    assert len(query["Signature"]) == 1
    assert _ACCESS_KEY_SECRET not in published.source_url


def test_publisher_reuses_matching_private_object_without_upload(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(_MP4)),
                "x-oss-meta-sha256": hashlib.sha256(_MP4).hexdigest(),
            },
            request=request,
        )

    published = _publisher(tmp_path, httpx.MockTransport(handler)).publish(_clip(tmp_path))

    assert published.source_url is not None
    assert [request.method for request in requests] == ["HEAD"]


@pytest.mark.parametrize(
    ("endpoint", "bucket", "prefix"),
    [
        ("http://oss-cn-hangzhou.aliyuncs.com", "private-video-bucket", "clips"),
        ("https://example.com", "private-video-bucket", "clips"),
        ("https://oss-cn-hangzhou.aliyuncs.com?token=x", "private-video-bucket", "clips"),
        ("https://oss-cn-hangzhou.aliyuncs.com", "Invalid_Bucket", "clips"),
        ("https://oss-cn-hangzhou.aliyuncs.com", "private-video-bucket", "../clips"),
    ],
)
def test_publisher_rejects_unsafe_configuration(
    tmp_path: Path,
    endpoint: str,
    bucket: str,
    prefix: str,
) -> None:
    with pytest.raises(VideoDemoError) as raised:
        OssTemporaryVideoPublisher(
            httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200))),
            endpoint=endpoint,
            bucket=bucket,
            access_key_id=SecretStr(_ACCESS_KEY_ID),
            access_key_secret=SecretStr(_ACCESS_KEY_SECRET),
            allowed_video_root=tmp_path,
            prefix=prefix,
            signed_url_ttl_seconds=3_600,
        )

    assert raised.value.code == ErrorCode.OSS_CONFIGURATION_INVALID


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, ErrorCode.OSS_AUTHENTICATION_FAILED),
        (403, ErrorCode.OSS_AUTHENTICATION_FAILED),
        (429, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        (500, ErrorCode.DEPENDENCY_TEMPORARY_FAILURE),
        (400, ErrorCode.OSS_OBJECT_INVALID),
    ],
)
def test_publisher_classifies_oss_http_failures(
    tmp_path: Path,
    status_code: int,
    expected: ErrorCode,
) -> None:
    publisher = _publisher(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(status_code, request=request)),
    )

    with pytest.raises(VideoDemoError) as raised:
        publisher.publish(_clip(tmp_path))

    assert raised.value.code == expected
    assert _ACCESS_KEY_ID not in str(raised.value)
    assert _ACCESS_KEY_SECRET not in str(raised.value)


def test_publisher_rejects_remote_metadata_mismatch(tmp_path: Path) -> None:
    publisher = _publisher(
        tmp_path,
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(_MP4) + 1),
                    "x-oss-meta-sha256": hashlib.sha256(_MP4).hexdigest(),
                },
                request=request,
            ),
        ),
    )

    with pytest.raises(VideoDemoError) as raised:
        publisher.publish(_clip(tmp_path))

    assert raised.value.code == ErrorCode.OSS_OBJECT_INVALID


def test_publisher_maps_transport_error_to_temporary_dependency_failure(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout", request=request)

    publisher = _publisher(tmp_path, httpx.MockTransport(handler))

    with pytest.raises(VideoDemoError) as raised:
        publisher.publish(_clip(tmp_path))

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
