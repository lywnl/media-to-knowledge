"""音频 ASR 窗口快照契约和稳定指纹。"""

from __future__ import annotations

import hashlib
import json

from video_demo.domain.base import FrozenModel
from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import TimeRange
from video_demo.speech.asr_contracts import CloudAsrWindow
from video_demo.speech.language import LanguageSpan
from video_demo.speech.vad import SpeechInterval


class AudioAsrWindowSnapshotPayload(FrozenModel):
    schema_version: str = "1.0.0"
    upload_range: TimeRange
    owned_range: TimeRange
    speech_interval: SpeechInterval
    source_intervals: tuple[SpeechInterval, ...] = ()
    language_span: LanguageSpan
    segments: tuple[SpeechSegment, ...]
    warnings: tuple[str, ...] = ()


def audio_asr_window_fingerprint(
    *,
    run_root: str,
    window: CloudAsrWindow,
    language_hint: str | None,
    prompt: str | None,
    max_window_ms: int,
    overlap_ms: int,
    max_upload_bytes: int,
) -> str:
    payload = {
        "schema_version": AudioAsrWindowSnapshotPayload.model_fields[
            "schema_version"
        ].default,
        "run_root": run_root,
        "upload_range": window.upload_range.model_dump(
            mode="json", exclude_computed_fields=True
        ),
        "owned_range": window.owned_range.model_dump(
            mode="json", exclude_computed_fields=True
        ),
        "speech_interval": window.speech_interval.model_dump(
            mode="json", exclude_computed_fields=True
        ),
        "source_intervals": tuple(
            item.model_dump(mode="json", exclude_computed_fields=True)
            for item in window.source_intervals
        ),
        "language_hint": language_hint,
        "prompt": prompt,
        "max_window_ms": max_window_ms,
        "overlap_ms": overlap_ms,
        "max_upload_bytes": max_upload_bytes,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
