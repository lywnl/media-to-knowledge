from __future__ import annotations

import pytest

from video_demo.application.audio_contracts import AudioSpeechAnalysis
from video_demo.domain.evidence import SpeechSegment


def _speech() -> SpeechSegment:
    return SpeechSegment(
        evidence_id="asr_001",
        start_ms=0,
        end_ms=1_000,
        text="音频内容",
        language="zh",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def test_audio_speech_contract_rejects_mismatched_source() -> None:
    with pytest.raises(ValueError):
        AudioSpeechAnalysis(transcript_source="SUBTITLE", evidence=(_speech(),))


def test_audio_speech_contract_projects_ordered_evidence() -> None:
    result = AudioSpeechAnalysis(transcript_source="ASR", evidence=(_speech(),))

    assert result.transcript_evidence == (_speech(),)
    assert result.transcript_by_id["asr_001"] == _speech()
