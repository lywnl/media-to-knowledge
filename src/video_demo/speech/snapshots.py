from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import ModelIdentity, TimeRange
from video_demo.speech.asr import CloudAsrWindow
from video_demo.speech.language import LanguageSpan
from video_demo.speech.vad import SpeechInterval

_ASR_COMPONENTS = frozenset({"silero_vad", "cloud_whisper"})
_CLOUD_ASR_WINDOW_STRATEGY_VERSION: Literal["1.0.0"] = "1.0.0"


class AsrSnapshotPayload(FrozenModel):
    schema_version: Literal["1.1.0", "1.2.0", "1.3.0"] = "1.3.0"
    language_spans: tuple[LanguageSpan, ...]
    segments: tuple[SpeechSegment, ...]
    vad_warnings: tuple[str, ...]
    silence_boundaries_ms: tuple[int, ...]
    language_change_boundaries_ms: tuple[int, ...]
    asr_warnings: tuple[str, ...] = ()


class AsrWindowSnapshotPayload(FrozenModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    upload_range: TimeRange
    owned_range: TimeRange
    speech_interval: SpeechInterval
    language_span: LanguageSpan
    segments: tuple[SpeechSegment, ...]
    warnings: tuple[str, ...] = ()


class SpeechFingerprintInputs(FrozenModel):
    model_identities: tuple[ModelIdentity, ...]
    cloud_asr_base_url: str = Field(min_length=1, max_length=2048)
    max_window_ms: int = Field(gt=0)
    overlap_ms: int = Field(ge=0)
    window_strategy_version: Literal["1.0.0"] = _CLOUD_ASR_WINDOW_STRATEGY_VERSION
    vad_threshold: float = 0.5
    vad_merge_gap_ms: int = 200


def asr_fingerprint(
    *,
    audio_sha256: str,
    duration_ms: int,
    language_hints: tuple[str, ...],
    hotwords: tuple[str, ...],
    core_context: str | None,
    inputs: SpeechFingerprintInputs,
) -> str:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    return _canonical_sha256(
        {
            "schema_version": AsrSnapshotPayload.model_fields["schema_version"].default,
            "audio_sha256": audio_sha256,
            "duration_ms": duration_ms,
            "language_hints": language_hints,
            "hotwords": hotwords,
            "core_context": core_context,
            "model_identities": _model_payload(inputs, _ASR_COMPONENTS),
            "cloud_asr_base_url": inputs.cloud_asr_base_url,
            "max_window_ms": inputs.max_window_ms,
            "overlap_ms": inputs.overlap_ms,
            "window_strategy_version": inputs.window_strategy_version,
            "vad_threshold": inputs.vad_threshold,
            "vad_merge_gap_ms": inputs.vad_merge_gap_ms,
        }
    )


def asr_window_fingerprint(
    *,
    asr_fingerprint: str,
    window: CloudAsrWindow,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": AsrWindowSnapshotPayload.model_fields[
                "schema_version"
            ].default,
            "asr_fingerprint": asr_fingerprint,
            "upload_range": window.upload_range.model_dump(
                mode="json",
                exclude_computed_fields=True,
            ),
            "owned_range": window.owned_range.model_dump(
                mode="json",
                exclude_computed_fields=True,
            ),
            "speech_interval": window.speech_interval.model_dump(
                mode="json",
                exclude_computed_fields=True,
            ),
        }
    )


def _model_payload(
    inputs: SpeechFingerprintInputs,
    components: frozenset[str],
) -> list[dict[str, object]]:
    return [
        identity.model_dump(mode="json", exclude_none=True)
        for identity in sorted(
            (
                item
                for item in inputs.model_identities
                if item.component in components
            ),
            key=lambda item: (item.component, item.model_id, item.revision or ""),
        )
    ]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
