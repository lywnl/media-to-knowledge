from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from collections.abc import Callable
from email.utils import formatdate
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlencode, urlsplit

import httpx
from pydantic import SecretStr

from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.video_port import (
    SegmentUnderstandingRequest,
    SummaryUnderstandingRequest,
    VideoClipInput,
    VideoUnderstandingPort,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingPort,
    WholeVideoUnderstandingRequest,
)
from video_demo.storage.workspace import reject_symlink_components

_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_ENDPOINT_PATTERN = re.compile(r"^oss-[a-z0-9-]+\.aliyuncs\.com$")


class TemporaryVideoPublisher(Protocol):
    @property
    def remote_host(self) -> str: ...

    def publish(self, clip: VideoClipInput) -> VideoClipInput: ...


def oss_remote_host(endpoint: str, bucket: str) -> str:
    """返回经严格校验的 OSS 虚拟主机域名，不读取任何凭据。"""

    return f"{_validate_bucket(bucket)}.{_validate_endpoint(endpoint)}"


class PublishedVideoUnderstanding:
    """生产全片只发布一次；保留片段方法供独立诊断使用。"""

    def __init__(
        self,
        delegate: VideoUnderstandingPort | WholeVideoUnderstandingPort,
        publisher: TemporaryVideoPublisher,
    ) -> None:
        self._delegate = delegate
        self._publisher = publisher

    @property
    def degraded_warnings(self) -> tuple[str, ...]:
        warnings = getattr(self._delegate, "degraded_warnings", ())
        return tuple(warnings) if isinstance(warnings, tuple) else ()

    def understand_segment(
        self,
        request: SegmentUnderstandingRequest,
    ) -> SegmentUnderstanding:
        published = self._publisher.publish(request.clip)
        delegate = self._delegate
        if not isinstance(delegate, VideoUnderstandingPort):
            raise TypeError("底层端口不支持片段理解")
        return delegate.understand_segment(
            request.model_copy(update={"clip": published}),
        )

    def summarize_video(
        self,
        request: SummaryUnderstandingRequest,
    ) -> SummaryUnderstanding:
        delegate = self._delegate
        if not isinstance(delegate, VideoUnderstandingPort):
            raise TypeError("底层端口不支持片段摘要")
        return delegate.summarize_video(request)

    def understand_video(
        self,
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding:
        published = self._publisher.publish(request.video)
        delegate = self._delegate
        if not isinstance(delegate, WholeVideoUnderstandingPort):
            raise TypeError("底层端口不支持全片理解")
        return delegate.understand_video(
            request.model_copy(update={"video": published}),
        )


class OssTemporaryVideoPublisher:
    """把已验证的本地视频发布为私有 OSS 短时签名 URL。"""

    def __init__(
        self,
        client: httpx.Client,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: SecretStr,
        access_key_secret: SecretStr,
        allowed_video_root: Path,
        prefix: str,
        signed_url_ttl_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._endpoint_host = _validate_endpoint(endpoint)
        self._bucket = _validate_bucket(bucket)
        self._access_key_id = _secret_value(access_key_id)
        self._access_key_secret = _secret_value(access_key_secret)
        self._allowed_video_root = allowed_video_root.expanduser().resolve(strict=True)
        self._prefix = _validate_prefix(prefix)
        if not 60 <= signed_url_ttl_seconds <= 86_400:
            raise _configuration_error()
        self._signed_url_ttl_seconds = signed_url_ttl_seconds
        self._clock = clock

    @property
    def remote_host(self) -> str:
        return f"{self._bucket}.{self._endpoint_host}"

    def publish(self, clip: VideoClipInput) -> VideoClipInput:
        path, size_bytes, sha256 = self._verify_local_clip(clip)
        object_key = self._object_key(clip, path, sha256)
        existing = self._head(object_key, missing_allowed=True)
        if existing is None:
            self._put(object_key, path, size_bytes, sha256)
            existing = self._head(object_key, missing_allowed=False)
        if existing is None:
            raise _object_error()
        self._validate_remote_object(existing, size_bytes=size_bytes, sha256=sha256)
        return VideoClipInput(
            clip_id=clip.clip_id,
            start_ms=clip.start_ms,
            end_ms=clip.end_ms,
            source_url=self._signed_get_url(object_key),
            mime_type=clip.mime_type,
            sha256=sha256,
        )

    def _verify_local_clip(self, clip: VideoClipInput) -> tuple[Path, int, str]:
        if clip.path is None or clip.sha256 is None or clip.mime_type != "video/mp4":
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "OSS 发布要求本地 MP4 短片")
        path = reject_symlink_components(
            self._allowed_video_root,
            clip.path,
            message="OSS 发布短片必须位于允许的视频目录内",
        )
        if not path.is_file():
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "OSS 发布短片必须是普通文件")
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        sha256 = digest.hexdigest()
        if size_bytes < 1 or sha256 != clip.sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "OSS 发布短片摘要不匹配")
        return path, size_bytes, sha256

    def _object_key(self, clip: VideoClipInput, path: Path, sha256: str) -> str:
        relative_parent = path.parent.relative_to(self._allowed_video_root).as_posix()
        owner_hash = hashlib.sha256(relative_parent.encode("utf-8")).hexdigest()[:24]
        return f"{self._prefix}/{owner_hash}/{clip.clip_id}-{sha256[:24]}.mp4"

    def _head(self, object_key: str, *, missing_allowed: bool) -> httpx.Headers | None:
        response = self._request("HEAD", object_key)
        if response.status_code == 404 and missing_allowed:
            return None
        self._require_success(response, missing_is_invalid=True)
        return response.headers

    def _put(self, object_key: str, path: Path, size_bytes: int, sha256: str) -> None:
        content_md5 = _file_md5(path)
        headers = self._authorization_headers(
            "PUT",
            object_key,
            content_md5=content_md5,
            content_type="video/mp4",
            oss_headers={"x-oss-meta-sha256": sha256},
        )
        headers["Content-Length"] = str(size_bytes)
        try:
            with path.open("rb") as stream:
                response = self._client.put(
                    self._object_url(object_key),
                    headers=headers,
                    content=stream,
                )
        except httpx.TransportError:
            raise _temporary_error() from None
        self._require_success(response, missing_is_invalid=True)

    def _request(self, method: str, object_key: str) -> httpx.Response:
        try:
            return self._client.request(
                method,
                self._object_url(object_key),
                headers=self._authorization_headers(method, object_key),
            )
        except httpx.TransportError:
            raise _temporary_error() from None

    def _authorization_headers(
        self,
        method: str,
        object_key: str,
        *,
        content_md5: str = "",
        content_type: str = "",
        oss_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        date = formatdate(self._clock(), usegmt=True)
        normalized_oss_headers = {
            key.lower(): value.strip() for key, value in (oss_headers or {}).items()
        }
        canonical_headers = "".join(
            f"{key}:{normalized_oss_headers[key]}\n" for key in sorted(normalized_oss_headers)
        )
        string_to_sign = (
            f"{method}\n{content_md5}\n{content_type}\n{date}\n"
            f"{canonical_headers}/{self._bucket}/{object_key}"
        )
        signature = self._signature(string_to_sign)
        headers = {
            "Date": date,
            "Authorization": f"OSS {self._access_key_id}:{signature}",
        }
        if content_md5:
            headers["Content-MD5"] = content_md5
        if content_type:
            headers["Content-Type"] = content_type
        headers.update(normalized_oss_headers)
        return headers

    def _signed_get_url(self, object_key: str) -> str:
        expires = int(self._clock()) + self._signed_url_ttl_seconds
        string_to_sign = f"GET\n\n\n{expires}\n/{self._bucket}/{object_key}"
        query = urlencode(
            {
                "OSSAccessKeyId": self._access_key_id,
                "Expires": str(expires),
                "Signature": self._signature(string_to_sign),
            },
        )
        return f"{self._object_url(object_key)}?{query}"

    def _signature(self, value: str) -> str:
        digest = hmac.new(
            self._access_key_secret.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def _object_url(self, object_key: str) -> str:
        return f"https://{self.remote_host}/{quote(object_key, safe='/')}"

    @staticmethod
    def _validate_remote_object(
        headers: httpx.Headers,
        *,
        size_bytes: int,
        sha256: str,
    ) -> None:
        try:
            remote_size = int(headers["Content-Length"])
        except (KeyError, TypeError, ValueError):
            raise _object_error() from None
        if remote_size != size_bytes or headers.get("x-oss-meta-sha256") != sha256:
            raise _object_error()

    @staticmethod
    def _require_success(response: httpx.Response, *, missing_is_invalid: bool) -> None:
        if 200 <= response.status_code < 300:
            return
        if response.status_code in (401, 403):
            raise VideoDemoError(
                ErrorCode.OSS_AUTHENTICATION_FAILED,
                "OSS 鉴权失败",
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise _temporary_error()
        if response.status_code == 404 and missing_is_invalid:
            raise _object_error()
        raise _object_error()


def _validate_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        raise _configuration_error() from None
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or host is None
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not _ENDPOINT_PATTERN.fullmatch(host.lower())
    ):
        raise _configuration_error()
    return host.lower()


def _validate_bucket(value: str) -> str:
    normalized = value.strip()
    if not _BUCKET_PATTERN.fullmatch(normalized):
        raise _configuration_error()
    return normalized


def _validate_prefix(value: str) -> str:
    normalized = value.strip("/")
    parts = normalized.split("/")
    if (
        not normalized
        or "\\" in normalized
        or any(not part or part in {".", ".."} for part in parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise _configuration_error()
    return normalized


def _secret_value(value: SecretStr) -> str:
    normalized = value.get_secret_value().strip()
    if not normalized or any(character in "\r\n" for character in normalized):
        raise _configuration_error()
    return normalized


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _configuration_error() -> VideoDemoError:
    return VideoDemoError(ErrorCode.OSS_CONFIGURATION_INVALID, "OSS 配置非法")


def _object_error() -> VideoDemoError:
    return VideoDemoError(ErrorCode.OSS_OBJECT_INVALID, "OSS 临时视频对象校验失败")


def _temporary_error() -> VideoDemoError:
    return VideoDemoError(ErrorCode.DEPENDENCY_TEMPORARY_FAILURE, "OSS 服务暂时不可用")
