from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import ModelIdentity, TimeRange
from video_demo.speech.asr import CloudAsrWindow
from video_demo.speech.language import LanguageSpan
from video_demo.speech.snapshots import (
    AsrSnapshotPayload,
    AsrWindowSnapshotPayload,
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    _speech_fingerprint_v1,
    asr_fingerprint,
    asr_window_fingerprint,
    speech_fingerprint,
)
from video_demo.speech.vad import SpeechInterval


def test_asr_snapshot_uses_hint_enabled_behavior_contract() -> None:
    assert AsrSnapshotPayload.model_fields["schema_version"].default == "1.1.0"


def test_asr_window_snapshot_has_strict_minimal_payload() -> None:
    window = _window()
    language_span = LanguageSpan(
        evidence_id="lid_window",
        start_ms=window.owned_range.start_ms,
        end_ms=window.owned_range.end_ms,
        language="zh",
        confidence=None,
        is_fully_evaluated_language=True,
    )
    segment = SpeechSegment(
        evidence_id="asr_window",
        start_ms=10_000,
        end_ms=11_000,
        text="窗口缓存文本",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )

    payload = AsrWindowSnapshotPayload(
        upload_range=window.upload_range,
        owned_range=window.owned_range,
        speech_interval=window.speech_interval,
        language_span=language_span,
        segments=(segment,),
        warnings=("ASR_OVERLAP_TIMESTAMP_CLAMPED",),
    )

    assert payload.schema_version == "1.0.0"
    assert payload.model_dump(mode="json", exclude_computed_fields=True) == {
        "schema_version": "1.0.0",
        "upload_range": {"start_ms": 10_000, "end_ms": 370_500},
        "owned_range": {"start_ms": 10_000, "end_ms": 370_000},
        "speech_interval": {
            "start_ms": 10_000,
            "end_ms": 730_000,
            "evidence_id": "vad_window",
            "confidence": 0.9,
        },
        "language_span": {
            "start_ms": 10_000,
            "end_ms": 370_000,
            "evidence_id": "lid_window",
            "language": "zh",
            "confidence": None,
            "detection_source": "MODEL",
            "is_fully_evaluated_language": True,
        },
        "segments": [
            {
                "start_ms": 10_000,
                "end_ms": 11_000,
                "evidence_id": "asr_window",
                "evidence_type": "ASR_SEGMENT",
                "text": "窗口缓存文本",
                "language": "zh",
                "confidence": 0.9,
                "is_fully_evaluated_language": True,
            }
        ],
        "warnings": ["ASR_OVERLAP_TIMESTAMP_CLAMPED"],
    }


def test_asr_window_fingerprint_binds_parent_and_window_only() -> None:
    window = _window()
    base = asr_window_fingerprint(asr_fingerprint="a" * 64, window=window)
    changed_parent = asr_window_fingerprint(asr_fingerprint="b" * 64, window=window)
    changed_window = asr_window_fingerprint(
        asr_fingerprint="a" * 64,
        window=CloudAsrWindow(
            upload_range=TimeRange(start_ms=10_000, end_ms=370_501),
            owned_range=TimeRange(start_ms=10_000, end_ms=370_001),
            speech_interval=window.speech_interval,
        ),
    )

    assert len(base) == 64
    assert base != changed_parent
    assert base != changed_window


def test_asr_window_snapshot_and_fingerprint_do_not_accept_secrets_or_request_data() -> None:
    window = _window()
    forbidden = {
        "prompt": "SECRET_PROMPT_SENTINEL",
        "api_key": "SECRET_KEY_SENTINEL",
        "authorization": "Bearer SECRET_AUTH_SENTINEL",
        "request_headers": {"X-Secret": "SECRET_HEADER_SENTINEL"},
    }

    with pytest.raises(ValidationError):
        AsrWindowSnapshotPayload.model_validate(
            {
                "upload_range": window.upload_range,
                "owned_range": window.owned_range,
                "speech_interval": window.speech_interval,
                "language_span": {
                    "evidence_id": "lid_window",
                    "start_ms": 10_000,
                    "end_ms": 370_000,
                    "language": "zh",
                    "is_fully_evaluated_language": True,
                },
                "segments": [],
                **forbidden,
            }
        )
    fingerprint = asr_window_fingerprint(asr_fingerprint="a" * 64, window=window)
    serialized = repr(fingerprint)
    assert all(str(value) not in serialized for value in forbidden.values())


def test_legacy_speech_snapshot_defaults_to_full_and_keeps_schema_version() -> None:
    payload = SpeechAnalysisSnapshotPayload.model_validate(
        {
            "schema_version": "1.0.0",
            "evidence": [],
            "warnings": [],
            "boundary_candidates": [],
            "transcript_source": "NONE",
        },
    )

    assert payload.schema_version == "1.0.0"
    assert payload.enrichment_mode == "full"


def test_legacy_speech_snapshot_rejects_explicit_text_enrichment_mode() -> None:
    with pytest.raises(ValidationError, match="历史语音快照只支持 full 模式"):
        SpeechAnalysisSnapshotPayload.model_validate(
            {
                "schema_version": "1.0.0",
                "enrichment_mode": "text",
                "evidence": [],
                "warnings": [],
                "boundary_candidates": [],
                "transcript_source": "NONE",
            },
        )


def test_legacy_speech_snapshot_rejects_explicit_null_enrichment_mode() -> None:
    with pytest.raises(ValidationError, match="历史语音快照只支持 full 模式"):
        SpeechAnalysisSnapshotPayload.model_validate(
            {
                "schema_version": "1.0.0",
                "enrichment_mode": None,
                "evidence": [],
                "warnings": [],
                "boundary_candidates": [],
                "transcript_source": "NONE",
            },
        )


def test_current_speech_snapshot_requires_explicit_enrichment_mode() -> None:
    with pytest.raises(ValidationError, match="必须显式携带 enrichment_mode"):
        SpeechAnalysisSnapshotPayload.model_validate(
            {
                "schema_version": "2.0.0",
                "evidence": [],
                "warnings": [],
                "boundary_candidates": [],
                "transcript_source": "NONE",
            },
        )


def test_current_speech_snapshot_rejects_explicit_null_enrichment_mode() -> None:
    with pytest.raises(ValidationError):
        SpeechAnalysisSnapshotPayload.model_validate(
            {
                "schema_version": "2.0.0",
                "enrichment_mode": None,
                "evidence": [],
                "warnings": [],
                "boundary_candidates": [],
                "transcript_source": "NONE",
            }
        )


def test_legacy_speech_fingerprint_is_distinct_from_current_fingerprint() -> None:
    inputs = _inputs()
    current = speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256="a" * 64,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
        enrichment_mode="full",
    )
    legacy = _speech_fingerprint_v1(
        processing_mode="ASR",
        transcript_payload_sha256="a" * 64,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
    )

    assert legacy != current
    assert legacy == "784d7a4c97d590cb4165b042976fdb8d7bdb9a22347c490acd219d5c5fc129a3"


def test_asr_fingerprint_changes_only_with_asr_inputs() -> None:
    inputs = _inputs()
    base = asr_fingerprint(
        audio_sha256="a" * 64,
        duration_ms=60_000,
        language_hints=("zh",),
        hotwords=("Milvus",),
        core_context="向量数据库课程",
        inputs=inputs,
    )

    assert base != asr_fingerprint(
        audio_sha256="a" * 64,
        duration_ms=60_000,
        language_hints=("zh",),
        hotwords=("WhisperX",),
        core_context="向量数据库课程",
        inputs=inputs,
    )
    assert base != asr_fingerprint(
        audio_sha256="a" * 64,
        duration_ms=60_000,
        language_hints=("zh",),
        hotwords=("Milvus",),
        core_context="关系型数据库课程",
        inputs=inputs,
    )
    changed_downstream = inputs.model_copy(
        update={
            "model_identities": tuple(
                item.model_copy(update={"revision": "new"})
                if item.component == "whisperx"
                else item
                for item in inputs.model_identities
            ),
            "yamnet_thresholds_sha256": "f" * 64,
        },
    )
    assert base == asr_fingerprint(
        audio_sha256="a" * 64,
        duration_ms=60_000,
        language_hints=("zh",),
        hotwords=("Milvus",),
        core_context="向量数据库课程",
        inputs=changed_downstream,
    )


def test_full_fingerprint_is_mode_sensitive_and_layers_downstream_inputs() -> None:
    inputs = _inputs()
    changed = inputs.model_copy(update={"yamnet_thresholds_sha256": "f" * 64})
    transcript_sha = "c" * 64

    assert speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
    ) != speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=changed,
    )

    assert speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
        enrichment_mode="text",
    ) == speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=changed,
        enrichment_mode="text",
    )

    assert speech_fingerprint(
        processing_mode="SUBTITLE",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
        enrichment_mode="text",
    ) == speech_fingerprint(
        processing_mode="SUBTITLE",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=9,
        max_speakers=10,
        allow_speaker_fallback=True,
        inputs=changed,
        enrichment_mode="full",
    )
    assert speech_fingerprint(
        processing_mode="SUBTITLE",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
    ) == speech_fingerprint(
        processing_mode="SUBTITLE",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=9,
        max_speakers=10,
        allow_speaker_fallback=True,
        inputs=changed,
    )

    assert speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
        enrichment_mode="text",
    ) != speech_fingerprint(
        processing_mode="ASR",
        transcript_payload_sha256=transcript_sha,
        media_warnings=(),
        min_speakers=1,
        max_speakers=2,
        allow_speaker_fallback=False,
        inputs=inputs,
        enrichment_mode="full",
    )


def test_full_fingerprint_binds_media_warnings_without_invalidating_asr() -> None:
    inputs = _inputs()
    common = {
        "processing_mode": "ASR",
        "transcript_payload_sha256": "c" * 64,
        "min_speakers": 1,
        "max_speakers": 2,
        "allow_speaker_fallback": False,
        "inputs": inputs,
    }

    assert speech_fingerprint(media_warnings=(), **common) != speech_fingerprint(  # type: ignore[arg-type]
        media_warnings=("SUBTITLE_TRACK_REJECTED:2:INCOMPLETE",),
        **common,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("transcript_source", "evidence"),
    [
        (
            "SUBTITLE",
            (
                SpeechSegment(
                    evidence_id="asr_001",
                    start_ms=0,
                    end_ms=1_000,
                    text="语音文本",
                    language="zh",
                    confidence=0.9,
                    is_fully_evaluated_language=True,
                ),
            ),
        ),
        (
            "ASR",
            (
                SubtitleCue(
                    evidence_id="subtitle_001",
                    start_ms=0,
                    end_ms=1_000,
                    text="字幕文本",
                    language="zh",
                    stream_index=2,
                ),
            ),
        ),
        (
            "NONE",
            (
                SpeechSegment(
                    evidence_id="asr_002",
                    start_ms=0,
                    end_ms=1_000,
                    text="不应存在",
                    language="zh",
                    confidence=0.9,
                    is_fully_evaluated_language=True,
                ),
            ),
        ),
    ],
)
def test_speech_snapshot_rejects_transcript_source_evidence_mismatch(
    transcript_source: str,
    evidence: tuple[object, ...],
) -> None:
    with pytest.raises(ValidationError):
        SpeechAnalysisSnapshotPayload(
            enrichment_mode="full",
            evidence=evidence,  # type: ignore[arg-type]
            warnings=(),
            boundary_candidates=(),
            transcript_source=transcript_source,  # type: ignore[arg-type]
        )


def _inputs() -> SpeechFingerprintInputs:
    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(component="silero_vad", provider="local", model_id="silero"),
            ModelIdentity(component="faster_whisper", provider="local", model_id="large-v3"),
            ModelIdentity(component="whisperx", provider="local", model_id="align"),
            ModelIdentity(component="pyannote", provider="local", model_id="diarize"),
            ModelIdentity(component="yamnet", provider="local", model_id="yamnet"),
        ),
        asr_compute_type="int8",
        yamnet_class_map_sha256="d" * 64,
        yamnet_thresholds_sha256="e" * 64,
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
