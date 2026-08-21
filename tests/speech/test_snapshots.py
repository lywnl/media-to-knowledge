from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.run import ModelIdentity
from video_demo.speech.snapshots import (
    SpeechAnalysisSnapshotPayload,
    SpeechFingerprintInputs,
    asr_fingerprint,
    speech_fingerprint,
)


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
