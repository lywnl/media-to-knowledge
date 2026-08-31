from __future__ import annotations

import pytest

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import ModelIdentity
from video_demo.speech.language import LanguageSpan
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    AsrWindowSnapshotPayload,
    SpeechFingerprintInputs,
    asr_fingerprint,
    asr_window_fingerprint,
)
from video_demo.speech.video_asr import build_fixed_asr_windows


def test_asr_snapshot_uses_video_schema_and_rejects_retired_vad_fields() -> None:
    payload = AsrSnapshotPayload(
        language_spans=(), segments=(), language_change_boundaries_ms=()
    )

    assert payload.schema_version == "2.0.0"
    with pytest.raises(ValueError):
        AsrSnapshotPayload.model_validate({**payload.model_dump(), "vad_warnings": []})
    with pytest.raises(ValueError):
        AsrSnapshotPayload.model_validate(
            {**payload.model_dump(), "silence_boundaries_ms": []}
        )


def test_asr_snapshot_persists_asr_warnings_only() -> None:
    payload = AsrSnapshotPayload(
        language_spans=(),
        segments=(),
        language_change_boundaries_ms=(),
        asr_warnings=("ASR_TIMESTAMP_CLAMPED",),
    )

    assert payload.asr_warnings == ("ASR_TIMESTAMP_CLAMPED",)


def test_asr_window_snapshot_has_fixed_chunk_contract() -> None:
    window = build_fixed_asr_windows(1_200_000)[1]
    payload = AsrWindowSnapshotPayload(
        chunk_index=window.chunk_index,
        upload_range=window.upload_range,
        owned_range=window.owned_range,
        language_span=LanguageSpan(
            evidence_id="lid_window",
            start_ms=window.owned_range.start_ms,
            end_ms=window.owned_range.end_ms,
            language="zh",
            confidence=None,
            is_fully_evaluated_language=True,
        ),
        segments=(
            SpeechSegment(
                evidence_id="asr_window",
                start_ms=600_000,
                end_ms=601_000,
                text="窗口缓存文本",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            ),
        ),
    )

    assert payload.schema_version == "2.0.0"
    assert payload.chunk_index == 1
    with pytest.raises(ValueError):
        AsrWindowSnapshotPayload.model_validate({**payload.model_dump(), "speech_interval": {}})


def test_old_video_snapshot_versions_are_not_accepted() -> None:
    with pytest.raises(ValueError):
        AsrSnapshotPayload.model_validate(
            {
                "schema_version": "1.3.0",
                "language_spans": [],
                "segments": [],
                "language_change_boundaries_ms": [],
            }
        )
    with pytest.raises(ValueError):
        AsrWindowSnapshotPayload.model_validate(
            {
                "schema_version": "1.1.0",
                "chunk_index": 0,
                "upload_range": {"start_ms": 0, "end_ms": 1_000},
                "owned_range": {"start_ms": 0, "end_ms": 1_000},
                "language_span": {
                    "evidence_id": "lid_old",
                    "start_ms": 0,
                    "end_ms": 1_000,
                    "language": "zh",
                    "confidence": None,
                    "is_fully_evaluated_language": True,
                },
                "segments": [],
            }
        )


def test_asr_window_fingerprint_binds_parent_and_window_only() -> None:
    window = build_fixed_asr_windows(1_200_000)[1]
    base = asr_window_fingerprint(asr_fingerprint="a" * 64, window=window)
    changed_parent = asr_window_fingerprint(asr_fingerprint="b" * 64, window=window)
    changed_window = asr_window_fingerprint(
        asr_fingerprint="a" * 64, window=build_fixed_asr_windows(1_800_000)[2]
    )

    assert base != changed_parent
    assert base != changed_window


def test_asr_fingerprint_binds_fixed_strategy_and_cloud_inputs() -> None:
    inputs = _inputs()
    arguments = {
        "audio_sha256": "a" * 64,
        "duration_ms": 60_000,
        "language_hints": ("zh",),
        "hotwords": ("Milvus",),
        "core_context": "向量数据库课程",
        "inputs": inputs,
    }
    base = asr_fingerprint(**arguments)  # type: ignore[arg-type]

    assert inputs.chunk_duration_ms == 600_000
    assert inputs.chunk_concurrency == 2
    assert inputs.window_strategy_version == "fixed-10m-v1"
    assert base != asr_fingerprint(**{**arguments, "hotwords": ("Qwen",)})  # type: ignore[arg-type]
    assert base != asr_fingerprint(
        **{
            **arguments,
            "inputs": inputs.model_copy(update={"cloud_asr_base_url": "https://second.example/v1"}),
        }
    )  # type: ignore[arg-type]
    assert base != asr_fingerprint(
        **{**arguments, "inputs": inputs.model_copy(update={"chunk_duration_ms": 300_000})}
    )  # type: ignore[arg-type]


def _inputs() -> SpeechFingerprintInputs:
    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(
                component="cloud_whisper",
                provider="openai_compatible",
                model_id="openai/whisper",
            ),
        ),
        cloud_asr_base_url="https://ai-proxy.example/v1",
    )
