from __future__ import annotations

from pathlib import Path

import pytest

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.application.production_speech import (
    AsrComponents,
    analysis_from_asr_snapshot,
    cloud_asr_prompt,
    run_asr_stage,
    transcript_shortcut,
)
from video_demo.domain.manifest import AudioStream, Rational, VideoAssetManifest, VideoStream
from video_demo.errors import ErrorCode, VideoDemoError
from video_demo.media.probe import ProbeLimits
from video_demo.media.subtitles import ParsedSubtitle, SubtitleArtifact
from video_demo.speech.asr import RawAsrSegment, WindowTranscriptionResult
from video_demo.speech.snapshots import AsrSnapshotPayload
from video_demo.speech.vad import SpeechInterval, VadResult
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import AsrWindowSnapshotStore


def test_transcript_shortcut_returns_subtitle_without_constructing_components(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path, subtitle=_subtitle())

    result = transcript_shortcut(media)

    assert result is not None
    assert result.transcript_source == "SUBTITLE"
    assert tuple(item.text for item in result.evidence) == ("字幕优先",)
    assert "TRANSCRIPT_SOURCE_SUBTITLE" in result.warnings


def test_transcript_shortcut_returns_none_source_when_audio_is_missing(tmp_path: Path) -> None:
    media = _media(tmp_path, audio=False)

    result = transcript_shortcut(media)

    assert result is not None
    assert result.transcript_source == "NONE"
    assert result.evidence == ()
    assert "NO_AUDIO_TRACK" in result.warnings


def test_analysis_from_asr_snapshot_projects_warnings_and_boundaries(tmp_path: Path) -> None:
    media = _media(tmp_path)
    payload = AsrSnapshotPayload(
        language_spans=(),
        segments=(),
        vad_warnings=("VAD_WARNING",),
        silence_boundaries_ms=(1_000,),
        language_change_boundaries_ms=(),
        asr_warnings=("ASR_OVERLAP_TIMESTAMP_CLAMPED",),
    )

    result = analysis_from_asr_snapshot(media, payload)

    assert result.transcript_source == "ASR"
    assert result.warnings == (
        "VAD_WARNING",
        "ASR_OVERLAP_TIMESTAMP_CLAMPED",
        "NO_SPEECH_DETECTED",
    )
    assert [(item.timestamp_ms, item.source) for item in result.boundary_candidates] == [
        (1_000, "silence")
    ]


def test_cloud_asr_prompt_combines_context_then_hotwords_without_rewriting() -> None:
    assert cloud_asr_prompt(("Milvus", "Qwen"), "向量数据库课程") == (
        "向量数据库课程\nMilvus Qwen"
    )
    assert cloud_asr_prompt((), None) is None
    assert cloud_asr_prompt(("Milvus",), None) == "Milvus"


def test_run_asr_stage_is_serial_and_resumes_completed_windows(tmp_path: Path) -> None:
    media = _media(tmp_path, duration_ms=1_800_000)
    vad = _Vad(
        (
            SpeechInterval(
                evidence_id="vad_long",
                start_ms=0,
                end_ms=1_800_000,
                confidence=0.9,
            ),
        )
    )
    calls: list[str] = []
    slices: list[Path] = []
    recognizer = _Recognizer(calls, fail_at=3)
    components = AsrComponents(
        vad=vad,
        recognizer=recognizer,
        slicer=_Slicer(tmp_path, calls, slices),
        slice_namespace="speech_request_resume",
    )
    window_store = AsrWindowSnapshotStore(AtomicArtifactStore(tmp_path))

    with pytest.raises(VideoDemoError) as raised:
        run_asr_stage(
            media,
            components,
            window_store=window_store,
            asr_fingerprint="a" * 64,
            max_window_ms=600_000,
            overlap_ms=1_000,
        )

    assert raised.value.code == ErrorCode.DEPENDENCY_TEMPORARY_FAILURE
    assert calls == ["slice-1", "recognize-1", "slice-2", "recognize-2", "slice-3", "recognize-3"]
    assert recognizer.max_active_calls == 1
    assert all(not path.exists() for path in slices)

    calls.clear()
    slices.clear()
    recognizer.fail_at = None
    payload = run_asr_stage(
        media,
        components,
        window_store=window_store,
        asr_fingerprint="a" * 64,
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert calls == ["slice-4", "recognize-4", "slice-5", "recognize-5"]
    assert [segment.start_ms for segment in payload.segments] == sorted(
        segment.start_ms for segment in payload.segments
    )
    assert len(payload.language_spans) == 4
    assert all(not path.exists() for path in slices)


def test_run_asr_stage_uses_single_language_hint_and_exact_prompt(tmp_path: Path) -> None:
    media = _media(
        tmp_path,
        config=PipelineRunConfig(
            language_hints=("zh",),
            hotwords=("Milvus", "Qwen"),
            core_context="向量数据库课程",
        ),
    )
    recognizer = _Recognizer([])
    components = AsrComponents(
        vad=_Vad((_speech(0, 5_000),)),
        recognizer=recognizer,
        slicer=_Slicer(tmp_path, [], []),
        slice_namespace="speech_request_prompt",
    )

    run_asr_stage(
        media,
        components,
        window_store=AsrWindowSnapshotStore(AtomicArtifactStore(tmp_path)),
        asr_fingerprint="b" * 64,
        max_window_ms=600_000,
        overlap_ms=1_000,
    )

    assert recognizer.inputs == [("zh", "向量数据库课程\nMilvus Qwen")]


class _Vad:
    def __init__(self, speech: tuple[SpeechInterval, ...]) -> None:
        self._speech = speech

    def detect(self, _audio: Path, *, duration_ms: int) -> VadResult:
        return VadResult(
            speech=self._speech,
            silence=(),
            long_silence_boundaries_ms=(),
        )


class _Slicer:
    def __init__(self, root: Path, calls: list[str], slices: list[Path]) -> None:
        self._root = root
        self._calls = calls
        self._slices = slices
        self._count = 0

    def create(
        self,
        _audio: Path,
        _run_relative_root: Path,
        _slice_id: str,
        _time_range: object,
    ) -> Path:
        self._count += 1
        self._calls.append(f"slice-{self._count}")
        path = self._root / f"slice-{self._count}.wav"
        path.write_bytes(b"wav")
        self._slices.append(path)
        return path


class _Recognizer:
    def __init__(self, calls: list[str], *, fail_at: int | None = None) -> None:
        self._calls = calls
        self.fail_at = fail_at
        self._count = 0
        self._active_calls = 0
        self.max_active_calls = 0
        self.inputs: list[tuple[str | None, str | None]] = []

    def transcribe_window(
        self,
        _audio_slice: Path,
        *,
        language_hint: str | None,
        prompt: str | None,
    ) -> WindowTranscriptionResult:
        self._count += 1
        self._active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            self._calls.append(f"recognize-{self._count}")
            self.inputs.append((language_hint, prompt))
            if self.fail_at == self._count:
                raise VideoDemoError(
                    ErrorCode.DEPENDENCY_TEMPORARY_FAILURE,
                    "模拟临时失败",
                )
            return WindowTranscriptionResult(
                language=language_hint or "zh",
                segments=(RawAsrSegment(0, 1_000, f"窗口 {self._count}", 0.9),),
            )
        finally:
            self._active_calls -= 1


def _speech(start_ms: int, end_ms: int) -> SpeechInterval:
    return SpeechInterval(
        evidence_id=f"vad_{start_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=0.9,
    )


def _subtitle() -> ParsedSubtitle:
    from video_demo.domain.evidence import SubtitleCue

    return ParsedSubtitle(
        artifact=SubtitleArtifact(
            relative_path="runs/scope/run_001/media/subtitle.vtt",
            sha256="f" * 64,
            size_bytes=10,
            stream_index=0,
            language="zh",
            codec_name="webvtt",
        ),
        cues=(
            SubtitleCue(
                evidence_id="subtitle_001",
                start_ms=0,
                end_ms=1_000,
                text="字幕优先",
                language="zh",
                stream_index=0,
            ),
        ),
        normalized_char_count=4,
        timeline_span_ratio=0.1,
    )


def _media(
    tmp_path: Path,
    *,
    duration_ms: int = 10_000,
    config: PipelineRunConfig | None = None,
    subtitle: ParsedSubtitle | None = None,
    audio: bool = True,
) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source = tmp_path / run_root / "input/source.mp4"
    proxy = tmp_path / run_root / "media/proxy.mp4"
    audio_path = tmp_path / run_root / "media/audio.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    proxy.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    proxy.write_bytes(b"proxy")
    if audio:
        audio_path.write_bytes(b"wav")
    asset = RegisteredAsset(
        source_path=source,
        source_sha256="a" * 64,
        object_ref="obj_" + "a" * 32,
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=config or PipelineRunConfig(),
    )
    manifest = VideoAssetManifest(
        object_ref=asset.object_ref,
        source_sha256=asset.source_sha256,
        source_size_bytes=asset.source_size_bytes,
        source_mime="video/mp4",
        duration_ms=duration_ms,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=1280,
            height=720,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        audio_streams=(
            AudioStream(index=1, codec_name="aac", channels=2, sample_rate_hz=48_000),
        )
        if audio
        else (),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    return PreparedMedia(
        source=ProbedAsset(asset=asset, manifest=manifest, limits=ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256="b" * 64,
        proxy_size_bytes=5,
        audio_path=audio_path if audio else None,
        audio_sha256="c" * 64 if audio else None,
        subtitle=subtitle,
    )
