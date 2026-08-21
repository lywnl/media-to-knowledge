from __future__ import annotations

from video_demo.domain.evidence import AlignedWord, SpeakerTurn
from video_demo.speech.speaker_assignment import assign_speakers


def _word(start_ms: int, end_ms: int, text: str = "word") -> AlignedWord:
    return AlignedWord(
        evidence_id=f"word_{start_ms}_{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        language="en",
        probability=0.9,
    )


def _turn(start_ms: int, end_ms: int, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(
        evidence_id=f"turn_{speaker}_{start_ms}_{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=speaker,
    )


def test_assigns_primary_speaker_by_largest_time_intersection() -> None:
    assigned = assign_speakers(
        (_word(1_000, 2_000),),
        (
            _turn(900, 1_400, "SPEAKER_01"),
            _turn(1_300, 2_100, "SPEAKER_02"),
        ),
    )

    assert assigned[0].speaker == "SPEAKER_02"
    assert assigned[0].overlap_speakers == ("SPEAKER_01",)


def test_assigns_unknown_when_no_reliable_overlap() -> None:
    assigned = assign_speakers(
        (_word(1_000, 2_000),),
        (_turn(1_900, 2_100, "SPEAKER_01"),),
        minimum_overlap_ratio=0.2,
    )

    assert assigned[0].speaker == "SPEAKER_UNKNOWN"
    assert assigned[0].overlap_speakers == ()


def test_assignment_never_infers_real_names_from_word_text() -> None:
    assigned = assign_speakers(
        (_word(0, 1_000, "我是王小明"),),
        (_turn(0, 1_000, "SPEAKER_01"),),
    )

    assert assigned[0].speaker == "SPEAKER_01"
