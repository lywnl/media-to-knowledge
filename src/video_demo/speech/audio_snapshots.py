"""音频 ASR 窗口快照契约和稳定指纹。"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from video_demo.domain.base import FrozenModel, Sha256
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.audio_fixed_asr import AudioFixedAsrWindow
from video_demo.speech.language import LanguageSpan

_AUDIO_ASR_WINDOW_STRATEGY_VERSION: Literal["fixed-10m-v1"] = "fixed-10m-v1"


class AudioAsrWindowSnapshotPayload(FrozenModel):
    schema_version: Literal["2.0.0"] = "2.0.0"
    chunk_index: int
    upload_range: TimeRange
    owned_range: TimeRange
    language_span: LanguageSpan
    segments: tuple[SpeechSegment, ...]
    warnings: tuple[str, ...] = ()


class AudioAsrFingerprintInputs(FrozenModel):
    """绑定音频 ASR 快照的非敏感运行配置。"""

    model_id: str = Field(min_length=1, max_length=256)
    base_url: str = Field(min_length=1, max_length=2_048)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    max_attempts: int = Field(ge=1, le=5)
    chunk_duration_ms: int = Field(default=600_000, gt=0)
    chunk_concurrency: Literal[1] = 1
    window_strategy_version: Literal["fixed-10m-v1"] = _AUDIO_ASR_WINDOW_STRATEGY_VERSION
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)


def audio_asr_fingerprint(
    *,
    asset_sha256: Sha256,
    duration_ms: int,
    language_hints: tuple[str, ...],
    hotwords: tuple[str, ...],
    core_context: str | None,
    inputs: AudioAsrFingerprintInputs,
) -> str:
    """生成音频专用父级 ASR 指纹，不依赖视频快照契约。"""

    if duration_ms < 1:
        raise ValueError("音频时长必须大于 0")
    return _canonical_sha256(
        {
            "schema_version": AudioAsrWindowSnapshotPayload.model_fields[
                "schema_version"
            ].default,
            "asset_sha256": asset_sha256,
            "duration_ms": duration_ms,
            "language_hints": language_hints,
            "hotwords": hotwords,
            "core_context": core_context,
            "model_id": inputs.model_id,
            "base_url": inputs.base_url,
            "timeout_seconds": inputs.timeout_seconds,
            "max_attempts": inputs.max_attempts,
            "chunk_duration_ms": inputs.chunk_duration_ms,
            "chunk_concurrency": inputs.chunk_concurrency,
            "window_strategy_version": inputs.window_strategy_version,
            "max_upload_bytes": inputs.max_upload_bytes,
        },
    )


def audio_asr_window_fingerprint(
    *,
    asr_fingerprint: Sha256,
    window: AudioFixedAsrWindow,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": AudioAsrWindowSnapshotPayload.model_fields[
                "schema_version"
            ].default,
            "asr_fingerprint": asr_fingerprint,
            "chunk_index": window.chunk_index,
            "upload_range": window.upload_range.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "owned_range": window.owned_range.model_dump(
                mode="json", exclude_computed_fields=True
            ),
        },
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
