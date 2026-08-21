from __future__ import annotations

import pytest

from video_demo.domain.evidence import AudioEvent, SpeechSegment, SubtitleCue, TimelineEvidence
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.fusion.timeline import build_timeline, validate_timeline


def _speech(
    *,
    evidence_id: str = "asr_001",
    start_ms: int = 100,
    end_ms: int = 400,
    text: str = "Hello",
) -> SpeechSegment:
    return SpeechSegment(
        evidence_id=evidence_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="en",
        confidence=0.9,
        is_fully_evaluated_language=True,
    )


def _audio() -> AudioEvent:
    return AudioEvent(
        evidence_id="audio_001",
        start_ms=100,
        end_ms=400,
        audioset_class="Music",
        normalized_event="音乐",
        confidence=0.8,
        threshold_version="eval-unvalidated-v1",
    )


def test_timeline_is_stable_across_input_order_and_exact_duplicates() -> None:
    speech = _speech()
    audio = _audio()

    first = build_timeline((speech, audio, speech))
    second = build_timeline((audio, speech))

    assert tuple(item.model_dump_json() for item in first) == tuple(
        item.model_dump_json() for item in second
    )
    assert len(first) == 1
    assert first[0].start_ms == 100
    assert first[0].end_ms == 400
    assert first[0].evidence_refs == ("asr_001", "audio_001")
    assert first[0].timeline_id.startswith("timeline_")


def test_timeline_orders_half_open_ranges_deterministically() -> None:
    later = _speech(evidence_id="asr_later", start_ms=400, end_ms=800)
    earlier = _speech(evidence_id="asr_earlier", start_ms=0, end_ms=400)

    timeline = build_timeline((later, earlier))

    assert [(item.start_ms, item.end_ms) for item in timeline] == [(0, 400), (400, 800)]


def test_timeline_accepts_subtitle_cues_without_treating_them_as_asr() -> None:
    subtitle = SubtitleCue(
        evidence_id="subtitle_001",
        start_ms=0,
        end_ms=400,
        text="字幕正文",
        language="zh",
        stream_index=2,
    )

    timeline = build_timeline((subtitle,))

    assert timeline[0].evidence_refs == ("subtitle_001",)


def test_timeline_rejects_same_id_with_different_content() -> None:
    with pytest.raises(VideoDemoError) as raised:
        build_timeline((_speech(text="Hello"), _speech(text="Different")))

    assert raised.value.code == ErrorCode.DUPLICATE_EVIDENCE_ID


def test_timeline_validation_rejects_unknown_reference() -> None:
    timeline = TimelineEvidence(
        timeline_id="timeline_bad",
        start_ms=100,
        end_ms=400,
        evidence_refs=("missing_001",),
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_timeline((timeline,), (_speech(),))

    assert raised.value.code == ErrorCode.UNKNOWN_EVIDENCE_REFERENCE


def test_timeline_validation_requires_referenced_evidence_inside_range() -> None:
    timeline = TimelineEvidence(
        timeline_id="timeline_bad",
        start_ms=100,
        end_ms=399,
        evidence_refs=("asr_001",),
    )

    with pytest.raises(VideoDemoError) as raised:
        validate_timeline((timeline,), (_speech(),))

    assert raised.value.code == ErrorCode.EVIDENCE_OUTSIDE_SEGMENT
