from __future__ import annotations

from video_demo.domain.evidence import SpeechSegment
from video_demo.speech.asr import remove_adjacent_cloud_asr_duplicates


def _segment(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
    confidence: float,
) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="en",
        confidence=confidence,
        is_fully_evaluated_language=True,
    )


def test_adjacent_asr_duplicate_removal_keeps_higher_confidence_exact_text() -> None:
    segments = (
        _segment("asr_001", 0, 1_000, "\uff21\uff29   News", 0.7),
        _segment("asr_002", 1_000, 2_000, "ai news", 0.9),
        _segment("asr_003", 2_000, 3_000, "AI news today", 0.8),
        _segment("asr_004", 3_000, 4_000, "news today", 0.6),
    )

    deduplicated = remove_adjacent_cloud_asr_duplicates(segments)

    assert [(item.text, item.confidence) for item in deduplicated] == [
        ("ai news", 0.9),
        ("AI news today", 0.8),
        ("news today", 0.6),
    ]
