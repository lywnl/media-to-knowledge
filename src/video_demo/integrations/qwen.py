from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, TypeVar, overload
from urllib.parse import SplitResult, urlsplit

import httpx
from pydantic import (
    Field,
    SecretStr,
    StrictInt,
    ValidationError,
    field_validator,
)

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import (
    AlignedWord,
    AudioEvent,
    OcrEvidence,
    SceneBoundary,
    SpeakerTurn,
    SpeechSegment,
    SubtitleCue,
)
from video_demo.domain.result import SegmentUnderstanding, SummaryUnderstanding
from video_demo.domain.run import TimeRange
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.integrations.prompts import (
    CAPABILITY_PROBE_INSTRUCTION,
    SEGMENT_SYSTEM_INSTRUCTION,
    SUMMARY_SYSTEM_INSTRUCTION,
    WHOLE_VIDEO_JSON_CONTRACT,
    WHOLE_VIDEO_SYSTEM_INSTRUCTION,
    render_segment_evidence,
    render_summary_segments,
    render_whole_video_evidence,
    select_spread_items,
    whole_video_group_window_indexes,
)
from video_demo.integrations.video_port import (
    SegmentSummaryInput,
    SegmentUnderstandingRequest,
    SummaryUnderstandingRequest,
    VideoClipInput,
    VideoUnderstandingPort,
    WholeVideoUnderstanding,
    WholeVideoUnderstandingPort,
    WholeVideoUnderstandingRequest,
    WholeVideoWindowUnderstanding,
)
from video_demo.storage.workspace import reject_symlink_components

ResponseModel = TypeVar("ResponseModel", bound=FrozenModel)
ApiKeyProvider = Callable[[], SecretStr | None]

_DEFAULT_FULL_VIDEO_VISUAL_INSTRUCTION = """请对这段完整视频执行视觉理解，而不只是泛化总结。
按以下五段输出，使用对应中文标题，每项保持精简且五段必须全部返回：
1. 整体摘要：一段中文摘要；
2. 关键事件：恰好 10 条，标注近似 MM:SS，并区分 visual(画面可见内容)与
speech(语音提到内容)；
3. 画面文字：恰好 12 条，标注近似 MM:SS 与 high/medium/low 置信度。
不要逐条抄录贯穿全片的口播字幕，优先提取账号名、数字、章节标题、软件和操作界面文字；
4. 视觉事实：恰好 6 条，只写画面观察到的人物、场景、物体、账号页面、软件或操作界面；
5. 来源区分：恰好 5 条，说明哪些结论来自画面、哪些来自语音、哪些由两者共同支持。
不得把语音内容伪装成画面文字；无法确认的文字必须标 low；时间为模型观察的近似位置。"""


class CapabilityProbeResponse(FrozenModel):
    supported: Literal[True]


class QwenCapabilities(FrozenModel):
    model_id: str
    protocol: Literal["chat_completions"] = "chat_completions"
    video_input: Literal["data_url", "remote_url"] = "data_url"
    json_schema: Literal[True] = True
    max_video_bytes: int
    max_video_duration_ms: int
    timeout_seconds: float


class QwenProviderReceipt(FrozenModel):
    """仅暴露可哈希的供应商回执，不携带响应正文。"""

    response_id: str = Field(min_length=1, max_length=256)
    http_status: StrictInt = Field(ge=200, le=299)

    @field_validator("response_id")
    @classmethod
    def require_visible_ascii(cls, value: str) -> str:
        if any(not "!" <= character <= "~" for character in value):
            raise ValueError("Qwen response ID 必须是可见 ASCII")
        return value


class _CompactWholeVideoUnderstanding(FrozenModel):
    group_summaries: tuple[
        Annotated[str, Field(min_length=1, max_length=100)],
        ...,
    ] = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=200)
    summary_zh: str = Field(min_length=1, max_length=4000)
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


class QwenVideoClient:
    _DEFAULT_REMOTE_VIDEO_HOSTS = frozenset(
        {"test-demo-video-1374604134.cos.ap-shanghai.myqcloud.com"},
    )

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str | None,
        api_key: str | None = None,
        api_key_provider: ApiKeyProvider | None = None,
        model_id: str | None,
        allowed_video_root: Path,
        max_video_bytes: int = 64 * 1024 * 1024,
        max_video_duration_ms: int = 30_000,
        timeout_seconds: float = 300.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
        allowed_remote_video_hosts: frozenset[str] | None = None,
    ) -> None:
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts 必须大于等于 1")
        if (
            isinstance(retry_backoff_seconds, bool)
            or not isinstance(retry_backoff_seconds, (int, float))
            or not math.isfinite(retry_backoff_seconds)
            or retry_backoff_seconds < 0
        ):
            raise ValueError("retry_backoff_seconds 不得小于 0")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or max_video_bytes < 1
            or max_video_duration_ms < 1
        ):
            raise ValueError("Qwen 视频和超时限制必须大于 0")
        self._client = client
        normalized_base_url = base_url.strip() if base_url is not None else ""
        normalized_model_id = model_id.strip() if model_id is not None else ""
        self._endpoint = (
            f"{normalized_base_url.rstrip('/')}/chat/completions"
            if normalized_base_url
            else None
        )
        self._api_key = SecretStr(api_key) if type(api_key) is str else None
        self._api_key_provider = api_key_provider
        self._provider_api_key: SecretStr | None = None
        self._model_id = normalized_model_id or None
        self._allowed_video_root = allowed_video_root.resolve(strict=True)
        self._max_video_bytes = max_video_bytes
        self._max_video_duration_ms = max_video_duration_ms
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleeper = sleeper
        self._allowed_remote_video_hosts = frozenset(
            host.strip().lower()
            for host in (
                allowed_remote_video_hosts or self._DEFAULT_REMOTE_VIDEO_HOSTS
            )
            if host.strip()
        )
        self._capabilities: QwenCapabilities | None = None
        self._capability_receipt: QwenProviderReceipt | None = None
        self._capability_lock = threading.Lock()
        self._credential_lock = threading.Lock()

    def probe_capabilities(self, clip: VideoClipInput) -> QwenCapabilities:
        capabilities, _receipt = self.probe_capabilities_with_receipt(clip)
        return capabilities

    def probe_capabilities_with_receipt(
        self,
        clip: VideoClipInput,
    ) -> tuple[QwenCapabilities, QwenProviderReceipt]:
        self._require_configuration()
        with self._capability_lock:
            if self._capabilities is None:
                video_content = self._verified_video_content(clip)
                return self._probe_with_verified_video(clip, video_content)
            capabilities = self._capabilities
            receipt = self._capability_receipt
            if receipt is None:
                raise RuntimeError("Qwen 能力回执状态非法")
        self._verified_video_content(clip)
        return capabilities, receipt

    def _probe_with_verified_video(
        self,
        clip: VideoClipInput,
        video_content: dict[str, object],
    ) -> tuple[QwenCapabilities, QwenProviderReceipt]:
        _endpoint, _api_key, model_id = self._require_configuration()
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": CAPABILITY_PROBE_INSTRUCTION},
                ],
            },
        ]
        try:
            _result, receipt = self._call_and_validate(
                messages,
                CapabilityProbeResponse,
                allowed_refs=None,
            )
        except VideoDemoError as error:
            if error.code in (
                ErrorCode.QWEN_AUTHENTICATION_FAILED,
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            ):
                raise
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 端点不支持所需的视频和 JSON Schema 能力",
            ) from None
        capabilities = QwenCapabilities(
            model_id=model_id,
            video_input=("remote_url" if clip.source_url is not None else "data_url"),
            max_video_bytes=self._max_video_bytes,
            max_video_duration_ms=self._max_video_duration_ms,
            timeout_seconds=self._timeout_seconds,
        )
        self._capabilities = capabilities
        self._capability_receipt = receipt
        return capabilities, receipt

    def understand_segment(
        self,
        request: SegmentUnderstandingRequest,
    ) -> SegmentUnderstanding:
        result, _receipt = self.understand_segment_with_receipt(request)
        return result

    def understand_segment_with_receipt(
        self,
        request: SegmentUnderstandingRequest,
    ) -> tuple[SegmentUnderstanding, QwenProviderReceipt]:
        if self._capabilities is None:
            self.probe_capabilities(request.clip)
        video_content = self._verified_video_content(request.clip)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SEGMENT_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": render_segment_evidence(request)},
                ],
            },
        ]
        return self._call_and_validate(
            messages,
            SegmentUnderstanding,
            allowed_refs={item.evidence_id for item in request.evidence},
        )

    def understand_video(
        self,
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding:
        """用一个逻辑请求读取完整视频，并严格绑定全部本地窗口。"""

        video_content = self._verified_whole_video_content(request.video)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": WHOLE_VIDEO_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": render_whole_video_evidence(request)},
                ],
            },
        ]
        if self._model_id == "qwen3-vl-flash":
            result = self._call_whole_video_plain_json(messages, request)
        else:
            result, _receipt = self._call_and_validate(
                messages,
                WholeVideoUnderstanding,
                allowed_refs=None,
            )
        self._validate_whole_video_result(request, result)
        return result

    def _verified_whole_video_content(self, clip: VideoClipInput) -> dict[str, object]:
        if clip.duration_ms > 1_800_000:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 完整视频理解不得超过 30 分钟",
            )
        if clip.source_url is not None:
            self._validate_remote_video_url(clip.source_url)
            return _remote_video_content(clip.source_url)
        return _video_content(
            self._read_verified_clip(
                clip,
                max_video_bytes=128 * 1024 * 1024,
                max_video_duration_ms=1_800_000,
            ),
            clip.mime_type,
        )

    def _call_whole_video_plain_json(
        self,
        messages: list[dict[str, object]],
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding:
        payload = {
            "model": self._require_configuration()[2],
            "messages": [
                *messages,
                {
                    "role": "user",
                    "content": WHOLE_VIDEO_JSON_CONTRACT,
                },
            ],
            "temperature": 0,
            "max_tokens": 8_192,
            "response_format": {"type": "json_object"},
        }
        response = self._post_once(payload)
        try:
            content, _receipt = _response_content(
                response,
                self._require_configuration()[2],
            )
            compact = _CompactWholeVideoUnderstanding.model_validate(content)
            if len(compact.group_summaries) > len(request.windows):
                raise ValueError("Qwen 返回语义组数量非法")
            group_indexes = whole_video_group_window_indexes(
                request,
                len(compact.group_summaries),
            )
            group_by_window = {
                window_index: group_summary
                for group_summary, window_indexes in zip(
                    compact.group_summaries,
                    group_indexes,
                    strict=True,
                )
                for window_index in window_indexes
            }
            return WholeVideoUnderstanding(
                windows=tuple(
                    WholeVideoWindowUnderstanding(
                        window_id=window.window_id,
                        understanding=_map_group_summary_to_window(
                            group_by_window[window_index],
                            request,
                            window_index,
                        ),
                    )
                    for window_index, window in enumerate(request.windows)
                ),
                summary=SummaryUnderstanding(
                    title=compact.title,
                    summary_zh=compact.summary_zh,
                    topics=compact.topics,
                    keywords=compact.keywords,
                ),
            )
        except (VideoDemoError, ValidationError, ValueError, json.JSONDecodeError):
            raise VideoDemoError(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 全片返回内容不符合结构化契约",
            ) from None

    @staticmethod
    def _validate_whole_video_result(
        request: WholeVideoUnderstandingRequest,
        result: WholeVideoUnderstanding,
    ) -> None:
        requested = {item.window_id: item for item in request.windows}
        returned = {item.window_id: item for item in result.windows}
        if requested.keys() != returned.keys() or len(returned) != len(result.windows):
            raise VideoDemoError(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 返回窗口集合与请求不一致",
            )
        for window_id, item in returned.items():
            allowed_refs = {evidence.evidence_id for evidence in requested[window_id].evidence}
            if not item.understanding.evidence_refs or (
                set(item.understanding.evidence_refs) - allowed_refs
            ):
                raise VideoDemoError(
                    ErrorCode.QWEN_RESPONSE_INVALID,
                    "Qwen 返回了未知或越界的证据引用",
                )

    def _read_verified_clip(
        self,
        clip: VideoClipInput,
        *,
        max_video_bytes: int | None = None,
        max_video_duration_ms: int | None = None,
    ) -> bytes:
        if clip.path is None or clip.sha256 is None:
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "Qwen 本地读取要求提供 path 和 sha256",
            )
        video_path = reject_symlink_components(
            self._allowed_video_root,
            clip.path,
            message="Qwen 输入视频路径非法",
        )
        if not video_path.is_file():
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "Qwen 输入视频必须是普通文件",
            )
        size_limit = (
            self._max_video_bytes if max_video_bytes is None else max_video_bytes
        )
        duration_limit = (
            self._max_video_duration_ms
            if max_video_duration_ms is None
            else max_video_duration_ms
        )
        if clip.end_ms - clip.start_ms > duration_limit:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 输入视频超过已探测的时长限制",
            )
        if video_path.stat().st_size > size_limit:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 输入视频超过已探测的大小限制",
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size_bytes = 0
        with video_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > size_limit:
                    raise VideoDemoError(
                        ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                        "Qwen 输入视频超过已探测的大小限制",
                    )
                digest.update(chunk)
                chunks.append(chunk)
        if digest.hexdigest() != clip.sha256:
            raise VideoDemoError(ErrorCode.VIDEO_DIGEST_MISMATCH, "Qwen 输入视频摘要不匹配")
        return b"".join(chunks)

    def _verified_video_content(self, clip: VideoClipInput) -> dict[str, object]:
        if clip.source_url is not None:
            self._validate_remote_video_url(clip.source_url)
            self._validate_clip_duration(clip)
            return _remote_video_content(clip.source_url)
        return _video_content(self._read_verified_clip(clip), clip.mime_type)

    def _validate_clip_duration(self, clip: VideoClipInput) -> None:
        if clip.duration_ms > self._max_video_duration_ms:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 输入视频超过已探测的时长限制",
            )

    def _validate_remote_video_url(self, value: str) -> None:
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            host = None
            port = None
        if not _is_allowed_remote_video_url(
            parsed if host is not None else None,
            host,
            port,
            self._allowed_remote_video_hosts,
        ):
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "Qwen 公网视频 URL 不在允许范围内",
            )
        if host is None:
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "Qwen 公网视频 URL 主机名非法",
            )
        if not _host_resolves_to_public_addresses(host, port or 443):
            raise VideoDemoError(
                ErrorCode.VIDEO_INPUT_INVALID,
                "Qwen 公网视频域名解析到不允许的地址",
            )

    @overload
    def summarize_video(
        self,
        request: SummaryUnderstandingRequest,
    ) -> SummaryUnderstanding: ...

    @overload
    def summarize_video(
        self,
        request: VideoClipInput,
        *,
        instruction: str = _DEFAULT_FULL_VIDEO_VISUAL_INSTRUCTION,
        max_tokens: int = 3_000,
    ) -> str: ...

    def summarize_video(
        self,
        request: SummaryUnderstandingRequest | VideoClipInput,
        *,
        instruction: str = _DEFAULT_FULL_VIDEO_VISUAL_INSTRUCTION,
        max_tokens: int = 3_000,
    ) -> SummaryUnderstanding | str:
        if isinstance(request, VideoClipInput):
            return self._summarize_clip_video(
                request,
                instruction=instruction,
                max_tokens=max_tokens,
            )
        self._require_capabilities()
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SUMMARY_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": render_summary_segments(request)},
                ],
            },
        ]
        result, _receipt = self._call_and_validate(
            messages,
            SummaryUnderstanding,
            allowed_refs=None,
        )
        return result

    def _summarize_clip_video(
        self,
        clip: VideoClipInput,
        *,
        instruction: str,
        max_tokens: int,
    ) -> str:
        """以普通文本协议理解完整视频，公网视频始终直接透传 URL。"""

        if clip.source_url is not None:
            self._validate_remote_video_url(clip.source_url)
            if clip.duration_ms > 1_800_000:
                raise VideoDemoError(
                    ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                    "Qwen 完整视频理解不得超过 30 分钟",
                )
            video_content = _remote_video_content(clip.source_url)
        else:
            video = self._read_verified_clip(
                clip,
                max_video_bytes=128 * 1024 * 1024,
                max_video_duration_ms=1_800_000,
            )
            video_content = _video_content(video, clip.mime_type)
        return self._summarize_video_content(
            video_content,
            instruction=instruction,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _validate_summary_parameters(instruction: str, max_tokens: int) -> None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
            or max_tokens > 8_192
            or not instruction.strip()
            or len(instruction) > 8_000
        ):
            raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "Qwen 总结参数非法")

    def _summarize_video_content(
        self,
        video_content: dict[str, object],
        *,
        instruction: str,
        max_tokens: int,
    ) -> str:
        self._validate_summary_parameters(instruction, max_tokens)
        _endpoint, _api_key, model_id = self._require_configuration()
        payload: dict[str, object] = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [video_content, {"type": "text", "text": instruction.strip()}],
                },
            ],
            "max_tokens": max_tokens,
        }
        response = self._post_with_retry(payload)
        content, _receipt = _plain_text_response(response, model_id)
        return content

    def _require_capabilities(self) -> QwenCapabilities:
        if self._capabilities is None:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 能力探测尚未成功",
            )
        return self._capabilities

    def _require_configuration(self) -> tuple[str, SecretStr, str]:
        if self._endpoint is None or self._model_id is None:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 能力配置不可用",
            )
        api_key = self._resolve_api_key()
        if api_key is None:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 能力配置不可用",
            )
        return self._endpoint, api_key, self._model_id

    def _resolve_api_key(self) -> SecretStr | None:
        if self._api_key is not None or self._api_key_provider is None:
            return self._api_key
        with self._credential_lock:
            if self._api_key is not None:
                return self._api_key
            if self._provider_api_key is not None:
                return self._provider_api_key
            try:
                provided = self._api_key_provider()
            except Exception:
                raise VideoDemoError(
                    ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                    "Qwen 能力配置不可用",
                ) from None
            if provided is not None and not isinstance(provided, SecretStr):
                raise VideoDemoError(
                    ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                    "Qwen 能力配置不可用",
                ) from None
            self._provider_api_key = provided
            return self._provider_api_key

    def _discard_provider_api_key(self, candidate: SecretStr) -> None:
        if self._api_key_provider is None:
            return
        with self._credential_lock:
            if self._provider_api_key is candidate:
                self._provider_api_key = None

    def _call_and_validate(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
        *,
        allowed_refs: set[str] | None,
    ) -> tuple[ResponseModel, QwenProviderReceipt]:
        _endpoint, _api_key, model_id = self._require_configuration()
        current_messages = messages
        for schema_attempt in range(2):
            payload = self._request_payload(current_messages, response_model)
            response = self._post_with_retry(payload)
            try:
                content, receipt = _response_content(response, model_id)
                result = response_model.model_validate(content)
                _validate_evidence_refs(result, allowed_refs)
                return result, receipt
            except VideoDemoError as error:
                if error.code == ErrorCode.QWEN_CAPABILITY_UNAVAILABLE:
                    raise
                if schema_attempt == 1:
                    break
                current_messages = _repair_messages(messages, error)
            except (ValidationError, ValueError) as error:
                if schema_attempt == 1:
                    break
                current_messages = _repair_messages(messages, error)
        if (
            response_model is SegmentUnderstanding
            and self._plain_json_visual_fallback_enabled()
        ):
            return self._call_plain_json_and_validate(
                messages,
                response_model,
                allowed_refs=allowed_refs,
            )
        raise VideoDemoError(
            ErrorCode.QWEN_RESPONSE_INVALID,
            "Qwen 返回内容不符合结构化契约",
        ) from None

    def _plain_json_visual_fallback_enabled(self) -> bool:
        """兼容当前 Qwen-VL 网关不执行 response_format 的实际行为。"""

        return self._model_id == "qwen3-vl-flash"

    def _call_plain_json_and_validate(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
        *,
        allowed_refs: set[str] | None,
    ) -> tuple[ResponseModel, QwenProviderReceipt]:
        schema_json = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fallback_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "STRICT_JSON_UNSUPPORTED_COMPATIBILITY_MODE\n"
                    "只输出一个合法 JSON 对象，不要 Markdown、解释或代码围栏。"
                    "字段必须严格符合给定 Schema，不得添加时间字段或其他字段。"
                    "evidence_refs 只能引用输入中真实存在的 evidence_id。\n"
                    "JSON Schema:\n"
                    f"{schema_json}"
                ),
            },
        ]
        payload = {
            "model": self._require_configuration()[2],
            "messages": fallback_messages,
            "temperature": 0,
        }
        response = self._post_with_retry(payload)
        try:
            content, receipt = _response_content(
                response,
                self._require_configuration()[2],
            )
            result = response_model.model_validate(content)
            _validate_evidence_refs(result, allowed_refs)
            return result, receipt
        except (VideoDemoError, ValidationError, ValueError, json.JSONDecodeError):
            raise VideoDemoError(
                ErrorCode.QWEN_RESPONSE_INVALID,
                "Qwen 返回内容不符合结构化契约",
            ) from None

    def _request_payload(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
    ) -> dict[str, object]:
        _endpoint, _api_key, model_id = self._require_configuration()
        return {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(response_model),
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }

    def _post_with_retry(self, payload: dict[str, object]) -> httpx.Response:
        endpoint, api_key, _model_id = self._require_configuration()
        try:
            authorization = _authorization_header(api_key)
        except VideoDemoError:
            self._discard_provider_api_key(api_key)
            raise
        last_error: VideoDemoError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(
                    endpoint,
                    headers={
                        "Authorization": authorization,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                _raise_for_status(response)
                return response
            except httpx.TransportError:
                last_error = VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "Qwen 请求暂时失败",
                )
            except VideoDemoError as error:
                if error.code != ErrorCode.DEPENDENCY_TEMPORARY_FAILURE:
                    raise
                last_error = error
            if attempt < self._max_attempts:
                self._sleeper(self._retry_backoff_seconds * (2 ** (attempt - 1)))
        if last_error is None:
            raise RuntimeError("Qwen 重试状态非法")
        raise last_error from None

    def _post_once(self, payload: dict[str, object]) -> httpx.Response:
        endpoint, api_key, _model_id = self._require_configuration()
        try:
            authorization = _authorization_header(api_key)
        except VideoDemoError:
            self._discard_provider_api_key(api_key)
            raise
        try:
            response = self._client.post(
                endpoint,
                headers={
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout_seconds,
            )
            _raise_for_status(response)
            return response
        except httpx.TransportError:
            raise VideoDemoError(
                ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                "Qwen 请求暂时失败",
            ) from None


class DemoFallbackVideoUnderstanding:
    """显式 Demo 模式的确定性语义兜底，不改变严格 Qwen 客户端。"""

    _FALLBACK_CODES = frozenset(
        {
            ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
            ErrorCode.QWEN_AUTHENTICATION_FAILED,
            ErrorCode.QWEN_RESPONSE_INVALID,
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            ErrorCode.OSS_CONFIGURATION_INVALID,
            ErrorCode.OSS_AUTHENTICATION_FAILED,
            ErrorCode.OSS_OBJECT_INVALID,
        }
    )

    def __init__(
        self,
        delegate: VideoUnderstandingPort | WholeVideoUnderstandingPort,
    ) -> None:
        self._delegate = delegate
        self._degraded = False

    @property
    def degraded_warnings(self) -> tuple[str, ...]:
        return ("DEMO_DEGRADED_QWEN",) if self._degraded else ()

    def understand_video(
        self,
        request: WholeVideoUnderstandingRequest,
    ) -> WholeVideoUnderstanding:
        delegate = self._delegate
        if not isinstance(delegate, WholeVideoUnderstandingPort):
            raise TypeError("底层端口不支持全片理解")
        try:
            return delegate.understand_video(request)
        except VideoDemoError as error:
            if error.code not in self._FALLBACK_CODES:
                raise
            self._degraded = True
            return _fallback_whole_video_understanding(request)

    def understand_segment(
        self,
        request: SegmentUnderstandingRequest,
    ) -> SegmentUnderstanding:
        delegate = self._delegate
        if not isinstance(delegate, VideoUnderstandingPort):
            raise TypeError("底层端口不支持片段理解")
        try:
            return delegate.understand_segment(request)
        except VideoDemoError as error:
            if error.code not in self._FALLBACK_CODES:
                raise
            self._degraded = True
            return _fallback_segment_understanding(request)

    def summarize_video(
        self,
        request: SummaryUnderstandingRequest,
    ) -> SummaryUnderstanding:
        delegate = self._delegate
        if not isinstance(delegate, VideoUnderstandingPort):
            raise TypeError("底层端口不支持片段摘要")
        try:
            return delegate.summarize_video(request)
        except VideoDemoError as error:
            if error.code not in self._FALLBACK_CODES:
                raise
            self._degraded = True
            return _fallback_summary_understanding(request)


def _fallback_segment_understanding(
    request: SegmentUnderstandingRequest,
) -> SegmentUnderstanding:
    subtitle_items = tuple(
        item
        for item in request.evidence
        if isinstance(item, SubtitleCue) and item.text.strip()
    )
    speech_items = tuple(
        item
        for item in request.evidence
        if isinstance(item, SpeechSegment) and item.text.strip()
    )
    aligned_items = tuple(
        item
        for item in request.evidence
        if isinstance(item, AlignedWord) and item.text.strip()
    )
    spoken_items: tuple[SubtitleCue | SpeechSegment | AlignedWord, ...] = (
        subtitle_items
        if subtitle_items
        else (speech_items if speech_items else aligned_items)
    )
    spoken_values = tuple(
        _truncate_local_semantic_text(item.text, 160)
        for item in select_spread_items(spoken_items, limit=3)
    )
    ocr_items = tuple(
        item
        for item in request.evidence
        if isinstance(item, OcrEvidence)
    )
    all_ocr_values = tuple(
        line.text
        for item in ocr_items
        for line in item.lines
        if line.text.strip()
    )
    ocr_values = tuple(
        _truncate_local_semantic_text(value, 80)
        for value in select_spread_items(all_ocr_values, limit=3)
    )
    text_values = (*spoken_values, *ocr_values)
    title = text_values[0] if text_values else "视频片段"
    speakers = tuple(
        dict.fromkeys(
            item.speaker
            for item in request.evidence
            if isinstance(item, (AlignedWord, SpeakerTurn))
        )
    )
    languages = tuple(
        dict.fromkeys(
            item.language
            for item in request.evidence
            if isinstance(item, (SubtitleCue, SpeechSegment, AlignedWord, OcrEvidence))
        )
    )
    keywords = tuple(dict.fromkeys(text_values))
    event_values = tuple(
        dict.fromkeys(
            item.normalized_event
            for item in request.evidence
            if isinstance(item, AudioEvent)
        )
    )
    scene_values = tuple(
        "画面场景"
        for item in request.evidence
        if isinstance(item, SceneBoundary)
    )
    return SegmentUnderstanding(
        title=title[:200],
        summary_zh=title[:4000],
        speakers=speakers,
        languages=languages,
        topics=tuple(dict.fromkeys((*event_values, *scene_values))),
        keywords=keywords,
        original_keywords=keywords,
        evidence_refs=tuple(item.evidence_id for item in request.evidence),
    )


def _truncate_local_semantic_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _map_group_summary_to_window(
    group_summary: str,
    request: WholeVideoUnderstandingRequest,
    window_index: int,
) -> SegmentUnderstanding:
    window = request.windows[window_index]
    if not window.evidence:
        raise ValueError("全片窗口缺少本地证据")
    local = _fallback_segment_understanding(
        SegmentUnderstandingRequest(
            clip=request.video,
            window=TimeRange(
                start_ms=window.start_ms,
                end_ms=window.end_ms,
            ),
            timeline=window.timeline,
            evidence=window.evidence,
        ),
    )
    normalized_group = " ".join(group_summary.split())
    summary_zh = f"{normalized_group}；本地证据：{local.summary_zh}"
    return SegmentUnderstanding(
        title=local.title,
        summary_zh=summary_zh[:4000],
        speakers=local.speakers,
        languages=local.languages,
        topics=tuple(dict.fromkeys((normalized_group, *local.topics))),
        entities=local.entities,
        actions=local.actions,
        keywords=local.keywords,
        original_keywords=local.original_keywords,
        evidence_refs=tuple(item.evidence_id for item in window.evidence),
    )


def _fallback_summary_understanding(
    request: SummaryUnderstandingRequest,
) -> SummaryUnderstanding:
    first = request.segments[0].understanding
    return SummaryUnderstanding(
        title=first.title,
        summary_zh="；".join(item.understanding.summary_zh for item in request.segments),
        speakers=_ordered_values(request, "speakers"),
        languages=_ordered_values(request, "languages"),
        topics=_ordered_values(request, "topics"),
        entities=_ordered_values(request, "entities"),
        actions=_ordered_values(request, "actions"),
        keywords=_ordered_values(request, "keywords"),
        original_keywords=_ordered_values(request, "original_keywords"),
    )


def _fallback_whole_video_understanding(
    request: WholeVideoUnderstandingRequest,
) -> WholeVideoUnderstanding:
    window_results: list[WholeVideoWindowUnderstanding] = []
    summary_inputs: list[SegmentSummaryInput] = []
    for window in request.windows:
        understanding = _fallback_segment_understanding(
            SegmentUnderstandingRequest(
                clip=request.video,
                window=TimeRange(start_ms=window.start_ms, end_ms=window.end_ms),
                timeline=window.timeline,
                evidence=window.evidence,
            ),
        )
        window_results.append(
            WholeVideoWindowUnderstanding(
                window_id=window.window_id,
                understanding=understanding,
            ),
        )
        summary_inputs.append(
            SegmentSummaryInput(
                segment_ref=window.window_id,
                understanding=understanding,
            ),
        )
    return WholeVideoUnderstanding(
        windows=tuple(window_results),
        summary=_fallback_summary_understanding(
            SummaryUnderstandingRequest(segments=tuple(summary_inputs)),
        ),
    )


def _ordered_values(request: SummaryUnderstandingRequest, field: str) -> tuple[str, ...]:
    values: list[str] = []
    for item in request.segments:
        values.extend(getattr(item.understanding, field))
    return tuple(dict.fromkeys(values))


def _authorization_header(api_key: SecretStr) -> str:
    try:
        raw = api_key.get_secret_value()
        if type(raw) is not str:
            raise ValueError("Qwen 凭据类型非法")
        normalized = raw.strip(" ")
        if not normalized or any(not "!" <= character <= "~" for character in normalized):
            raise ValueError("Qwen 凭据内容非法")
        return f"Bearer {normalized}"
    except Exception:
        raise VideoDemoError(
            ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
            "Qwen 能力配置不可用",
        ) from None


def _video_content(video: bytes, mime_type: str) -> dict[str, object]:
    encoded = base64.b64encode(video).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _remote_video_content(source_url: str) -> dict[str, object]:
    """让供应商直接读取公网视频，避免 Demo 下载完整媒体到本地。"""

    return {
        "type": "video_url",
        "video_url": {"url": source_url},
    }


def _is_allowed_remote_video_url(
    parsed: SplitResult | None,
    host: str | None,
    port: int | None,
    allowed_hosts: frozenset[str],
) -> bool:
    if parsed is None or host is None:
        return False
    if parsed.scheme.lower() != "https" or parsed.username or parsed.password:
        return False
    if port not in (None, 443) or parsed.fragment or not parsed.path:
        return False
    if host.lower() not in allowed_hosts:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    )


def _host_resolves_to_public_addresses(host: str, port: int) -> bool:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not addresses:
        return False
    for _family, _socktype, _protocol, _canonname, sockaddr in addresses:
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
        ):
            return False
    return True


def _response_content(
    response: httpx.Response,
    expected_model: str,
) -> tuple[object, QwenProviderReceipt]:
    try:
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("响应不是对象")
        if payload.get("model") != expected_model:
            raise VideoDemoError(
                ErrorCode.QWEN_CAPABILITY_UNAVAILABLE,
                "Qwen 实际模型与配置不一致",
                {"expected_model": expected_model},
            )
        receipt = QwenProviderReceipt(
            response_id=payload["id"],
            http_status=response.status_code,
        )
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("choices 非法")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("choice 非法")
        message = choice["message"]
        if not isinstance(message, dict):
            raise ValueError("message 非法")
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("content 非法")
        return json.loads(content), receipt
    except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
        raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 响应格式非法") from None


def _plain_text_response(
    response: httpx.Response,
    expected_model: str,
) -> tuple[str, QwenProviderReceipt]:
    try:
        payload: object = response.json()
        if not isinstance(payload, dict) or payload.get("model") != expected_model:
            raise ValueError("响应模型非法")
        receipt = QwenProviderReceipt(
            response_id=payload["id"],
            http_status=response.status_code,
        )
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("choices 非法")
        message = choices[0]["message"]
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 非法")
        return content.strip(), receipt
    except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
        raise VideoDemoError(ErrorCode.QWEN_RESPONSE_INVALID, "Qwen 文本响应格式非法") from None


def _validate_evidence_refs(result: FrozenModel, allowed_refs: set[str] | None) -> None:
    if allowed_refs is None or not isinstance(result, SegmentUnderstanding):
        return
    unknown = sorted(set(result.evidence_refs) - allowed_refs)
    if unknown:
        raise VideoDemoError(
            ErrorCode.UNKNOWN_EVIDENCE_REFERENCE,
            "Qwen 引用了不存在的证据",
        )


def _validation_error_code(error: Exception) -> str:
    if isinstance(error, VideoDemoError):
        return error.code.value
    return ErrorCode.QWEN_RESPONSE_INVALID.value


def _repair_messages(
    messages: list[dict[str, object]],
    error: Exception,
) -> list[dict[str, object]]:
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "SCHEMA_REPAIR_REQUIRED\n"
                f"{_validation_error_code(error)}\n"
                "仅修复 JSON 结构和引用，不得添加新事实。"
            ),
        },
    ]


def _schema_name(response_model: type[FrozenModel]) -> str:
    if response_model is CapabilityProbeResponse:
        return "qwen_capability_probe"
    characters: list[str] = []
    for index, character in enumerate(response_model.__name__):
        if character.isupper() and index > 0:
            characters.append("_")
        characters.append(character.lower())
    return "".join(characters)


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code in (401, 403):
        raise VideoDemoError(
            ErrorCode.QWEN_AUTHENTICATION_FAILED,
            "Qwen 鉴权失败",
            {"status_code": response.status_code},
        )
    if response.status_code == 429 or response.status_code >= 500:
        raise VideoDemoError(
            ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
            "Qwen 服务暂时不可用",
            {"status_code": response.status_code},
        )
    raise VideoDemoError(
        ErrorCode.QWEN_RESPONSE_INVALID,
        "Qwen 请求被拒绝",
        {"status_code": response.status_code},
    )
