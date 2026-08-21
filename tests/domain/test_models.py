from __future__ import annotations

import hashlib
import math

import pytest
from pydantic import ValidationError

from video_demo.domain.evidence import AlignedWord, SpeechSegment
from video_demo.domain.result import (
    SegmentUnderstanding,
    VideoSegment,
    VideoUnderstandingResult,
)
from video_demo.domain.run import TimeRange


def test_time_range_uses_non_empty_half_open_milliseconds() -> None:
    valid = TimeRange(start_ms=0, end_ms=1)

    assert valid.duration_ms == 1
    with pytest.raises(ValidationError):
        TimeRange(start_ms=-1, end_ms=1)
    with pytest.raises(ValidationError):
        TimeRange(start_ms=1, end_ms=1)
    with pytest.raises(ValidationError):
        TimeRange(start_ms=2, end_ms=1)


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TimeRange(start_ms=0, end_ms=1, seconds=0.001)


@pytest.mark.parametrize("invalid_probability", [math.nan, math.inf, -0.01, 1.01])
def test_aligned_word_rejects_non_finite_or_out_of_range_probability(
    invalid_probability: float,
) -> None:
    with pytest.raises(ValidationError):
        AlignedWord(
            evidence_id="ev_word_001",
            start_ms=0,
            end_ms=100,
            text="你好",
            language="zh",
            probability=invalid_probability,
            speaker="SPEAKER_01",
        )


def test_speech_segment_keeps_original_text_and_language() -> None:
    segment = SpeechSegment(
        evidence_id="ev_asr_001",
        start_ms=0,
        end_ms=500,
        text="Hello world",
        language="en",
        confidence=0.92,
        is_fully_evaluated_language=True,
    )

    assert segment.text == "Hello world"
    assert segment.language == "en"


def test_qwen_segment_understanding_cannot_supply_time_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SegmentUnderstanding(
            title="问候",
            summary_zh="讲者向观众问好。",
            speakers=("SPEAKER_01",),
            languages=("en",),
            topics=("问候",),
            entities=(),
            actions=("问好",),
            keywords=("问候",),
            original_keywords=("Hello",),
            evidence_refs=("ev_asr_001",),
            start_ms=0,
            end_ms=500,
        )


def test_video_segment_rejects_duplicate_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="evidence_refs 不得重复"):
        VideoSegment(
            segment_id="seg_001",
            start_ms=0,
            end_ms=500,
            title="问候",
            summary_zh="讲者向观众问好。",
            speakers=("SPEAKER_01",),
            languages=("en",),
            topics=("问候",),
            entities=(),
            actions=("问好",),
            keywords=("问候",),
            original_keywords=("Hello",),
            evidence_refs=("ev_asr_001", "ev_asr_001"),
            retrieval_text="标题：问候",
            retrieval_hash="a" * 64,
        )


def test_public_json_schema_forbids_additional_properties_recursively() -> None:
    schema = VideoUnderstandingResult.model_json_schema()

    assert schema["additionalProperties"] is False
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )


def test_semantic_speakers_reject_inferred_person_names() -> None:
    with pytest.raises(ValidationError):
        SegmentUnderstanding(
            title="发言",
            summary_zh="某人发言。",
            speakers=("张三",),
            evidence_refs=("ev_asr_001",),
        )


def test_video_segment_retrieval_hash_must_match_text() -> None:
    text = "标题：问候"
    with pytest.raises(ValidationError, match="retrieval_hash"):
        VideoSegment(
            segment_id="seg_001",
            start_ms=0,
            end_ms=500,
            title="问候",
            summary_zh="讲者问好。",
            evidence_refs=("ev_asr_001",),
            retrieval_text=text,
            retrieval_hash=hashlib.sha256(b"different").hexdigest(),
        )
