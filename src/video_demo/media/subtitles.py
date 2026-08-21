from __future__ import annotations

import hashlib
import html
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import webvtt  # type: ignore[import-untyped]
from pydantic import Field
from webvtt.errors import (  # type: ignore[import-untyped]
    MalformedCaptionError,
    MalformedFileError,
)

from video_demo.domain.base import FrozenModel, LanguageCode, stable_identifier
from video_demo.domain.evidence import SubtitleCue
from video_demo.domain.manifest import SubtitleStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.transcode import SubtitleArtifact
from video_demo.storage.workspace import verified_run_file

_TEXT_SUBTITLE_CODECS = frozenset({"subrip", "ass", "ssa", "webvtt", "mov_text"})
_SUPPORTED_LANGUAGE_HINTS = frozenset({"zh", "en", "ja", "ko", "es"})
SubtitleRejectionReason = Literal[
    "VTT_INVALID",
    "LIMIT_EXCEEDED",
    "TIMELINE_INVALID",
    "INCOMPLETE",
]


@dataclass(frozen=True, slots=True)
class SubtitleParseLimits:
    max_file_bytes: int = 16 * 1024 * 1024
    max_cues: int = 20_000
    max_cue_chars: int = 4_000
    max_total_chars: int = 2_000_000

    def __post_init__(self) -> None:
        if min(
            self.max_file_bytes,
            self.max_cues,
            self.max_cue_chars,
            self.max_total_chars,
        ) < 1:
            raise ValueError("字幕解析上限必须全部大于 0")


class SubtitleTrackRejected(ValueError):
    def __init__(self, reason: SubtitleRejectionReason) -> None:
        self.reason = reason
        super().__init__(reason)


class ParsedSubtitle(FrozenModel):
    artifact: SubtitleArtifact
    cues: tuple[SubtitleCue, ...] = Field(min_length=1, max_length=20_000)
    normalized_char_count: int = Field(gt=0, le=2_000_000)
    timeline_span_ratio: float = Field(ge=0.0, le=1.0)


def rank_text_subtitle_streams(
    streams: tuple[SubtitleStream, ...],
    language_hints: tuple[LanguageCode, ...],
) -> tuple[SubtitleStream, ...]:
    hint_positions = {
        language: position
        for position, language in enumerate(language_hints)
        if language in _SUPPORTED_LANGUAGE_HINTS and language not in language_hints[:position]
    }
    unique: dict[int, SubtitleStream] = {}
    for stream in streams:
        if stream.codec_name.casefold() in _TEXT_SUBTITLE_CODECS:
            unique.setdefault(stream.index, stream)
    return tuple(
        sorted(
            unique.values(),
            key=lambda stream: (
                hint_positions.get(stream.language, 999),
                0 if stream.is_default else 1,
                0 if not stream.is_forced else 1,
                stream.index,
            ),
        )
    )


def parse_webvtt(
    runtime_root: Path,
    run_relative_root: Path,
    artifact: SubtitleArtifact,
    *,
    duration_ms: int,
    limits: SubtitleParseLimits | None = None,
) -> ParsedSubtitle:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    active_limits = limits or SubtitleParseLimits()
    path = verified_run_file(
        runtime_root,
        run_relative_root,
        runtime_root / artifact.relative_path,
        expected_sha256=artifact.sha256,
        digest=_sha256_file,
        message="字幕产物必须位于当前运行目录内",
    )
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise VideoDemoError(ErrorCode.VIDEO_INPUT_INVALID, "字幕实际大小与声明不一致")
    if actual_size > active_limits.max_file_bytes:
        raise SubtitleTrackRejected("LIMIT_EXCEEDED")

    try:
        document = webvtt.read(str(path), encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        ValueError,
        MalformedCaptionError,
        MalformedFileError,
    ) as error:
        raise SubtitleTrackRejected("VTT_INVALID") from error

    cues: list[SubtitleCue] = []
    total_chars = 0
    for caption in document:
        text = _normalize_caption_text(caption.text)
        if not text:
            continue
        if len(text) > active_limits.max_cue_chars:
            raise SubtitleTrackRejected("LIMIT_EXCEEDED")
        start_ms = _timestamp_ms(caption.start_time.to_tuple())
        end_ms = _timestamp_ms(caption.end_time.to_tuple())
        if start_ms < 0 or start_ms >= end_ms or start_ms >= duration_ms:
            raise SubtitleTrackRejected("TIMELINE_INVALID")
        if end_ms > duration_ms + 1_000:
            raise SubtitleTrackRejected("TIMELINE_INVALID")
        end_ms = min(end_ms, duration_ms)
        total_chars += len(text)
        if total_chars > active_limits.max_total_chars:
            raise SubtitleTrackRejected("LIMIT_EXCEEDED")
        cue = SubtitleCue(
            evidence_id=stable_identifier(
                "subtitle",
                {
                    "artifact_sha256": artifact.sha256,
                    "stream_index": artifact.stream_index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                },
            ),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            language=artifact.language,
            stream_index=artifact.stream_index,
        )
        cues.append(cue)
        if len(cues) > active_limits.max_cues:
            raise SubtitleTrackRejected("LIMIT_EXCEEDED")

    if not cues:
        raise SubtitleTrackRejected("VTT_INVALID")
    ordered = tuple(sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.evidence_id)))
    timeline_span_ratio = (ordered[-1].end_ms - ordered[0].start_ms) / duration_ms
    return ParsedSubtitle(
        artifact=artifact,
        cues=ordered,
        normalized_char_count=total_chars,
        timeline_span_ratio=timeline_span_ratio,
    )


def subtitle_minimum_chars(duration_ms: int) -> int:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    return math.ceil(max(20.0, duration_ms * 24.0 / 60_000.0))


def subtitle_minimum_cues(duration_ms: int) -> int:
    if duration_ms < 1:
        raise ValueError("视频时长必须大于 0")
    return max(1, math.ceil(duration_ms / 30_000))


def is_subtitle_eligible(parsed: ParsedSubtitle, *, duration_ms: int) -> bool:
    return (
        parsed.normalized_char_count >= subtitle_minimum_chars(duration_ms)
        and len(parsed.cues) >= subtitle_minimum_cues(duration_ms)
        and parsed.timeline_span_ratio >= 0.80
    )


def _normalize_caption_text(value: str) -> str:
    if _contains_forbidden_control(value):
        raise SubtitleTrackRejected("VTT_INVALID")
    decoded = html.unescape(value)
    if _contains_forbidden_control(decoded):
        raise SubtitleTrackRejected("VTT_INVALID")
    return " ".join(decoded.split())


def _contains_forbidden_control(value: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    )


def _timestamp_ms(value: tuple[int, int, int, int]) -> int:
    hours, minutes, seconds, milliseconds = value
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + milliseconds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
