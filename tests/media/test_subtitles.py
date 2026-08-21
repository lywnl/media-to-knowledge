from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from video_demo.domain.manifest import SubtitleStream
from video_demo.errors import VideoDemoError
from video_demo.media.subtitles import (
    SubtitleParseLimits,
    SubtitleTrackRejected,
    is_subtitle_eligible,
    parse_webvtt,
    rank_text_subtitle_streams,
    subtitle_minimum_chars,
    subtitle_minimum_cues,
)
from video_demo.media.transcode import SubtitleArtifact


def test_rank_text_subtitles_prefers_language_then_default_non_forced_and_index() -> None:
    streams = (
        _stream(7, "webvtt", "en", default=True),
        _stream(5, "mov_text", "zh", forced=True),
        _stream(4, "ass", "zh"),
        _stream(3, "subrip", "und", default=True),
        _stream(2, "hdmv_pgs_subtitle", "zh", default=True),
    )

    ranked = rank_text_subtitle_streams(streams, ("zh", "en"))

    assert [stream.index for stream in ranked] == [4, 5, 7, 3]


def test_rank_text_subtitles_deduplicates_same_absolute_stream_index() -> None:
    duplicate = _stream(4, "ass", "zh")

    assert rank_text_subtitle_streams((duplicate, duplicate), ("zh",)) == (duplicate,)


def test_parse_webvtt_preserves_milliseconds_and_normalizes_text(tmp_path: Path) -> None:
    artifact = _write_vtt(
        tmp_path,
        """WEBVTT

00:00:00.125 --> 00:00:01.875
<v 张三><b>你好</b> &amp;   世界

00:00:01.875 --> 00:00:03.000
  第二行\t文字  
""",
    )

    parsed = parse_webvtt(
        tmp_path,
        Path("runs/scope/run_001"),
        artifact,
        duration_ms=3_000,
    )

    assert [(cue.start_ms, cue.end_ms, cue.text) for cue in parsed.cues] == [
        (125, 1_875, "你好 & 世界"),
        (1_875, 3_000, "第二行 文字"),
    ]
    assert all(cue.stream_index == 2 and cue.language == "zh" for cue in parsed.cues)
    assert parsed.normalized_char_count == len("你好 & 世界") + len("第二行 文字")
    assert parsed.timeline_span_ratio == pytest.approx(2_875 / 3_000)


def test_parse_webvtt_drops_empty_cues_and_sorts_timeline(tmp_path: Path) -> None:
    artifact = _write_vtt(
        tmp_path,
        """WEBVTT

00:00:02.000 --> 00:00:03.000
后

00:00:00.000 --> 00:00:01.000
<b> </b>

00:00:01.000 --> 00:00:02.000
前
""",
    )

    parsed = parse_webvtt(
        tmp_path,
        Path("runs/scope/run_001"),
        artifact,
        duration_ms=3_000,
    )

    assert [cue.text for cue in parsed.cues] == ["前", "后"]


def test_parse_webvtt_clips_only_small_tail_overflow(tmp_path: Path) -> None:
    artifact = _write_vtt(
        tmp_path,
        """WEBVTT

00:00:08.500 --> 00:00:10.750
结尾
""",
    )

    parsed = parse_webvtt(
        tmp_path,
        Path("runs/scope/run_001"),
        artifact,
        duration_ms=10_000,
    )

    assert parsed.cues[0].end_ms == 10_000


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("WEBVTT\n\n00:00:08.500 --> 00:00:11.001\n越界\n", "TIMELINE_INVALID"),
        ("WEBVTT\n\n00:00:02.000 --> 00:00:01.000\n倒序\n", "TIMELINE_INVALID"),
        ("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n坏\u0000文本\n", "VTT_INVALID"),
        ("不是 WebVTT\n", "VTT_INVALID"),
        ("WEBVTT\n\n", "VTT_INVALID"),
    ],
)
def test_parse_webvtt_rejects_invalid_track(
    tmp_path: Path,
    payload: str,
    reason: str,
) -> None:
    artifact = _write_vtt(tmp_path, payload)

    with pytest.raises(SubtitleTrackRejected) as raised:
        parse_webvtt(
            tmp_path,
            Path("runs/scope/run_001"),
            artifact,
            duration_ms=10_000,
        )

    assert raised.value.reason == reason


@pytest.mark.parametrize("mutation", ["wrong_sha", "wrong_size", "outside", "symlink"])
def test_parse_webvtt_rejects_unverified_artifact(tmp_path: Path, mutation: str) -> None:
    artifact = _write_vtt(
        tmp_path,
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n正文\n",
    )
    if mutation == "wrong_sha":
        artifact = artifact.model_copy(update={"sha256": "f" * 64})
    elif mutation == "wrong_size":
        artifact = artifact.model_copy(update={"size_bytes": artifact.size_bytes + 1})
    elif mutation == "outside":
        outside = tmp_path / "outside.vtt"
        outside.write_text("WEBVTT\n", encoding="utf-8")
        artifact = artifact.model_copy(update={"relative_path": "outside.vtt"})
    else:
        path = tmp_path / artifact.relative_path
        target = path.with_name("target.vtt")
        path.rename(target)
        path.symlink_to(target)

    with pytest.raises(VideoDemoError):
        parse_webvtt(
            tmp_path,
            Path("runs/scope/run_001"),
            artifact,
            duration_ms=1_000,
        )


def test_parse_webvtt_enforces_file_cue_and_text_limits(tmp_path: Path) -> None:
    artifact = _write_vtt(
        tmp_path,
        """WEBVTT

00:00:00.000 --> 00:00:01.000
第一条

00:00:01.000 --> 00:00:02.000
第二条
""",
    )

    for limits in (
        SubtitleParseLimits(max_file_bytes=artifact.size_bytes - 1),
        SubtitleParseLimits(max_cues=1),
        SubtitleParseLimits(max_cue_chars=2),
        SubtitleParseLimits(max_total_chars=4),
    ):
        with pytest.raises(SubtitleTrackRejected) as raised:
            parse_webvtt(
                tmp_path,
                Path("runs/scope/run_001"),
                artifact,
                duration_ms=2_000,
                limits=limits,
            )
        assert raised.value.reason == "LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    ("duration_ms", "minimum_chars", "minimum_cues"),
    [
        (30_000, 20, 1),
        (31_000, 20, 2),
        (119_000, 48, 4),
        (120_000, 48, 4),
        (1_800_000, 720, 60),
    ],
)
def test_subtitle_thresholds_are_continuous(
    duration_ms: int,
    minimum_chars: int,
    minimum_cues: int,
) -> None:
    assert subtitle_minimum_chars(duration_ms) == minimum_chars
    assert subtitle_minimum_cues(duration_ms) == minimum_cues


def test_subtitle_eligibility_requires_chars_cues_and_timeline_span(tmp_path: Path) -> None:
    artifact = _write_vtt(
        tmp_path,
        """WEBVTT

00:00:00.000 --> 00:00:01.000
这是开头的一段足够长的完整字幕文本用来满足字符密度门槛

00:01:58.000 --> 00:02:00.000
这是结尾的一段足够长的完整字幕文本用来满足字符密度门槛
""",
    )
    parsed = parse_webvtt(
        tmp_path,
        Path("runs/scope/run_001"),
        artifact,
        duration_ms=120_000,
    )

    assert parsed.normalized_char_count >= 48
    assert parsed.timeline_span_ratio >= 0.8
    assert len(parsed.cues) < 4
    assert is_subtitle_eligible(parsed, duration_ms=120_000) is False


def _stream(
    index: int,
    codec: str,
    language: str,
    *,
    default: bool = False,
    forced: bool = False,
) -> SubtitleStream:
    return SubtitleStream(
        index=index,
        codec_name=codec,
        language=language,
        is_default=default,
        is_forced=forced,
    )


def _write_vtt(tmp_path: Path, text: str) -> SubtitleArtifact:
    relative = Path("runs/scope/run_001/media/subtitles/2.vtt")
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    path.write_bytes(encoded)
    return SubtitleArtifact(
        relative_path=relative.as_posix(),
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        stream_index=2,
        language="zh",
        codec_name="mov_text",
    )
