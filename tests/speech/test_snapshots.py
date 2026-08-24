from __future__ import annotations

import pytest

from video_demo.domain.evidence import SpeechSegment
from video_demo.domain.run import ModelIdentity, TimeRange
from video_demo.speech.asr import CloudAsrWindow
from video_demo.speech.language import LanguageSpan
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    AsrWindowSnapshotPayload,
    SpeechFingerprintInputs,
    asr_fingerprint,
    asr_window_fingerprint,
)
from video_demo.speech.vad import SpeechInterval


def test_asr_snapshot_writes_cloud_contract_and_reads_legacy_versions() -> None:
    assert AsrSnapshotPayload.model_fields["schema_version"].default == "1.3.0"
    for version in ("1.1.0", "1.2.0"):
        payload = AsrSnapshotPayload.model_validate(
            {
                "schema_version": version,
                "language_spans": [],
                "segments": [],
                "vad_warnings": [],
                "silence_boundaries_ms": [],
                "language_change_boundaries_ms": [],
            }
        )
        assert payload.schema_version == version
        assert payload.asr_warnings == ()


def test_asr_snapshot_persists_boundary_warnings() -> None:
    payload = AsrSnapshotPayload(
        language_spans=(),
        segments=(),
        vad_warnings=(),
        silence_boundaries_ms=(),
        language_change_boundaries_ms=(),
        asr_warnings=("ASR_TIMESTAMP_CLAMPED",),
    )

    assert payload.asr_warnings == ("ASR_TIMESTAMP_CLAMPED",)


def test_asr_window_snapshot_has_strict_minimal_payload() -> None:
    window = _window()
    payload = _window_payload(window)

    assert payload.schema_version == "1.0.0"
    assert payload.upload_range == window.upload_range
    assert payload.owned_range == window.owned_range
    assert payload.speech_interval == window.speech_interval
    assert tuple(item.text for item in payload.segments) == ("窗口缓存文本",)


def test_asr_window_fingerprint_binds_parent_and_window_only() -> None:
    window = _window()
    base = asr_window_fingerprint(asr_fingerprint="a" * 64, window=window)
    changed_parent = asr_window_fingerprint(asr_fingerprint="b" * 64, window=window)
    changed_window = asr_window_fingerprint(
        asr_fingerprint="a" * 64,
        window=CloudAsrWindow(
            upload_range=TimeRange(start_ms=10_000, end_ms=370_501),
            owned_range=window.owned_range,
            speech_interval=window.speech_interval,
        ),
    )

    assert base != changed_parent
    assert base != changed_window


def test_asr_window_snapshot_and_fingerprint_do_not_accept_secrets() -> None:
    payload = _window_payload(_window())
    dumped = payload.model_dump(mode="json")

    assert "api_key" not in repr(dumped).casefold()
    with pytest.raises(ValueError):
        AsrWindowSnapshotPayload.model_validate({**dumped, "api_key": "secret"})


def test_asr_fingerprint_binds_cloud_content_inputs() -> None:
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

    assert base != asr_fingerprint(
        **{**arguments, "hotwords": ("Qwen",)}  # type: ignore[arg-type]
    )
    assert base != asr_fingerprint(
        **{**arguments, "core_context": "关系型数据库课程"}  # type: ignore[arg-type]
    )
    assert base != asr_fingerprint(
        **{
            **arguments,
            "inputs": inputs.model_copy(
                update={"cloud_asr_base_url": "https://second.example/v1"}
            ),
        }  # type: ignore[arg-type]
    )
    assert base != asr_fingerprint(
        **{
            **arguments,
            "inputs": inputs.model_copy(update={"max_window_ms": 300_000}),
        }  # type: ignore[arg-type]
    )
    assert base != asr_fingerprint(
        **{
            **arguments,
            "inputs": inputs.model_copy(update={"overlap_ms": 2_000}),
        }  # type: ignore[arg-type]
    )
    assert base != asr_fingerprint(
        **{
            **arguments,
            "inputs": inputs.model_copy(
                update={
                    "model_identities": (
                        *inputs.model_identities[:-1],
                        inputs.model_identities[-1].model_copy(
                            update={"model_id": "new-cloud-model"}
                        ),
                    )
                }
            ),
        }  # type: ignore[arg-type]
    )


def _inputs() -> SpeechFingerprintInputs:
    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(component="silero_vad", provider="local", model_id="silero"),
            ModelIdentity(
                component="cloud_whisper",
                provider="openai_compatible",
                model_id="openai/whisper",
            ),
        ),
        cloud_asr_base_url="https://ai-proxy.example/v1",
        max_window_ms=600_000,
        overlap_ms=1_000,
    )


def _window() -> CloudAsrWindow:
    speech = SpeechInterval(
        evidence_id="vad_window",
        start_ms=10_000,
        end_ms=730_000,
        confidence=0.9,
    )
    return CloudAsrWindow(
        upload_range=TimeRange(start_ms=10_000, end_ms=370_500),
        owned_range=TimeRange(start_ms=10_000, end_ms=370_000),
        speech_interval=speech,
    )


def _window_payload(window: CloudAsrWindow) -> AsrWindowSnapshotPayload:
    return AsrWindowSnapshotPayload(
        upload_range=window.upload_range,
        owned_range=window.owned_range,
        speech_interval=window.speech_interval,
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
                start_ms=10_000,
                end_ms=11_000,
                text="窗口缓存文本",
                language="zh",
                confidence=0.9,
                is_fully_evaluated_language=True,
            ),
        ),
        warnings=("ASR_OVERLAP_TIMESTAMP_CLAMPED",),
    )
