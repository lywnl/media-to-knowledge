from __future__ import annotations

from collections.abc import Sequence

from video_demo.domain.evidence import AlignedWord, SpeakerTurn


def assign_speakers(
    words: Sequence[AlignedWord],
    turns: Sequence[SpeakerTurn],
    *,
    minimum_overlap_ratio: float = 0.1,
) -> tuple[AlignedWord, ...]:
    if not 0 <= minimum_overlap_ratio <= 1:
        raise ValueError("minimum_overlap_ratio 必须在 0 到 1 之间")
    assigned: list[AlignedWord] = []
    for word in words:
        intersections = sorted(
            (
                (_intersection_ms(word, turn), turn.speaker)
                for turn in turns
                if word.overlaps(turn)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        reliable = [
            (duration, speaker)
            for duration, speaker in intersections
            if duration / word.duration_ms >= minimum_overlap_ratio
        ]
        if not reliable:
            assigned.append(
                word.model_copy(
                    update={"speaker": "SPEAKER_UNKNOWN", "overlap_speakers": ()},
                ),
            )
            continue
        primary = reliable[0][1]
        overlaps = tuple(
            dict.fromkeys(speaker for _, speaker in reliable[1:] if speaker != primary),
        )
        assigned.append(
            word.model_copy(update={"speaker": primary, "overlap_speakers": overlaps}),
        )
    return tuple(assigned)


def _intersection_ms(word: AlignedWord, turn: SpeakerTurn) -> int:
    return max(0, min(word.end_ms, turn.end_ms) - max(word.start_ms, turn.start_ms))
