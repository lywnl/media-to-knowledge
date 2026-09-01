from __future__ import annotations

import hashlib
from pathlib import Path

from video_demo.application.pipeline import (
    PipelineRunConfig,
    PreparedMedia,
    ProbedAsset,
    RegisteredAsset,
)
from video_demo.application.production_speech import (
    AsrComponents,
    DirectSpeechAnalyzer,
)
from video_demo.domain.evidence import SpeechSegment, SubtitleCue
from video_demo.domain.manifest import (
    AudioStream,
    Rational,
    VideoAssetManifest,
    VideoStream,
)
from video_demo.media.probe import ProbeLimits
from video_demo.media.subtitles import ParsedSubtitle, SubtitleArtifact
from video_demo.speech.asr import RawAsrSegment, WindowTranscriptionResult
from video_demo.speech.snapshots import SpeechFingerprintInputs
from video_demo.storage.artifacts import AtomicArtifactStore
from video_demo.storage.snapshots import AsrWindowSnapshotStore, SnapshotStore


def test_subtitle_first_does_not_construct_or_call_asr(tmp_path: Path) -> None:
    media = _media(tmp_path, subtitle=_subtitle())
    analyzer = DirectSpeechAnalyzer(
        snapshot_store=SnapshotStore(AtomicArtifactStore(tmp_path)),
        window_store=AsrWindowSnapshotStore(AtomicArtifactStore(tmp_path)),
        component_factory=lambda *_: (_ for _ in ()).throw(AssertionError("字幕不应调用 ASR")),
        fingerprint_inputs=_fingerprint_inputs(),
    )

    result = analyzer.analyze(media)

    assert result.transcript_source == "SUBTITLE"
    assert tuple(item.text for item in result.evidence) == ("字幕优先",)


def test_direct_video_asr_publishes_snapshot_and_reuses_it(tmp_path: Path) -> None:
    recognizer = _Recognizer()
    components = AsrComponents(
        recognizer=recognizer,
        slicer=_Slicer(tmp_path),
        slice_namespace="video_direct",
    )
    store = AtomicArtifactStore(tmp_path)
    analyzer = DirectSpeechAnalyzer(
        snapshot_store=SnapshotStore(store),
        window_store=AsrWindowSnapshotStore(store),
        component_factory=lambda *_: components,
        fingerprint_inputs=_fingerprint_inputs(),
    )
    media = _media(tmp_path)

    first = analyzer.analyze(media)
    second = analyzer.analyze(media)

    assert first.transcript_source == "ASR"
    assert any(isinstance(item, SpeechSegment) for item in first.evidence)
    assert second.stage_cache_hits == ("SPEECH_ASR",)
    assert recognizer.calls == 1


def _media(tmp_path: Path, *, subtitle: ParsedSubtitle | None = None) -> PreparedMedia:
    run_root = Path("runs/scope/run_001")
    source = tmp_path / run_root / "input/source.mp4"
    proxy = tmp_path / run_root / "media/proxy.mp4"
    audio = tmp_path / run_root / "media/audio.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    proxy.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    proxy.write_bytes(b"proxy")
    audio.write_bytes(b"wav")
    asset = RegisteredAsset(
        source_path=source,
        source_sha256=hashlib.sha256(b"source").hexdigest(),
        object_ref="obj_001",
        source_size_bytes=6,
        source_mime="video/mp4",
        run_relative_root=run_root,
        config=PipelineRunConfig(language_hints=("zh",)),
    )
    manifest = VideoAssetManifest(
        object_ref=asset.object_ref,
        source_sha256=asset.source_sha256,
        source_size_bytes=asset.source_size_bytes,
        source_mime="video/mp4",
        duration_ms=30_000,
        video_stream=VideoStream(
            index=0,
            codec_name="h264",
            width=1280,
            height=720,
            average_frame_rate=Rational(numerator=25, denominator=1),
        ),
        audio_streams=(
            AudioStream(index=1, codec_name="aac", channels=2, sample_rate_hz=48_000),
        ),
        format_name="mov,mp4",
        ffprobe_version="test",
    )
    return PreparedMedia(
        source=ProbedAsset(asset=asset, manifest=manifest, limits=ProbeLimits()),
        proxy_path=proxy,
        proxy_sha256="b" * 64,
        proxy_size_bytes=5,
        audio_path=None if subtitle is not None else audio,
        audio_sha256=None if subtitle is not None else hashlib.sha256(b"wav").hexdigest(),
        subtitle=subtitle,
    )


def _subtitle() -> ParsedSubtitle:
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


def _fingerprint_inputs() -> SpeechFingerprintInputs:
    from video_demo.domain.run import ModelIdentity

    return SpeechFingerprintInputs(
        model_identities=(
            ModelIdentity(component="cloud_whisper", provider="test", model_id="test"),
        ),
        cloud_asr_base_url="https://example.test/v1",
    )


class _Slicer:
    def __init__(self, root: Path) -> None:
        self._root = root

    def create(self, *_args: object) -> Path:
        path = self._root / "slice.wav"
        path.write_bytes(b"wav")
        return path


class _Recognizer:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_window(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> WindowTranscriptionResult:
        self.calls += 1
        return WindowTranscriptionResult(
            language="zh",
            segments=(RawAsrSegment(0, 1_000, "ASR 文本", 0.9),),
        )
